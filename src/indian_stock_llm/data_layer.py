from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread

from .connectors import SourceConnector, default_connectors
from .config import AssistantConfig

DATASET_PATH_MAP = {
    "instrument_master": "instrument_master_path",
    "corporate_actions": "corporate_actions_path",
    "filings": "filings_path",
    "regulatory_updates": "regulatory_updates_path",
    "market_events": "market_events_path",
}


@dataclass(frozen=True)
class EnterpriseDataSnapshot:
    instrument_master: list[dict]
    corporate_actions: list[dict]
    filings: list[dict]
    regulatory_updates: list[dict]
    market_events: list[dict]
    source_hierarchy: tuple[str, ...]
    refreshed_at: str
    lineage: dict[str, str]
    stale_feeds: tuple[str, ...]
    partial_feeds: tuple[str, ...]
    connector_status: dict[str, str]


@dataclass(frozen=True)
class DataReadinessReport:
    ready: bool
    blockers: tuple[str, ...]
    completeness: float
    freshness_ok: bool
    lineage_ok: bool
    fallback_mode: bool


class EnterpriseDataLayer:
    def __init__(self, config: AssistantConfig, connectors: tuple[SourceConnector, ...] | None = None):
        self.config = config
        self.source_hierarchy = (
            "nse_primary",
            "bse_secondary",
            "regulatory_primary",
            "news_events_secondary",
        )
        self.connectors: tuple[SourceConnector, ...] = connectors or default_connectors(self.config)
        self._refresh_stop_event: Event | None = None
        self._refresh_worker: Thread | None = None
        self._refresh_errors: Queue[str] = Queue(maxsize=64)
        self._snapshot = self.refresh_daily()

    def _read_json_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        return [item for item in data if isinstance(item, dict)]

    @staticmethod
    def _parse_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        text = value.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    def _dataset_is_stale(self, dataset: str, rows: list[dict]) -> bool:
        if not rows:
            return True
        if dataset == "instrument_master":
            return False
        timestamps: list[datetime] = []
        for row in rows:
            ts = self._parse_timestamp(row.get("timestamp")) or self._parse_timestamp(row.get("effective_date"))
            if ts:
                timestamps.append(ts)
        if not timestamps:
            return True
        latest = max(timestamps)
        return (datetime.now(timezone.utc) - latest) > timedelta(hours=self.config.max_data_staleness_hours)

    def _load_dataset(self, dataset: str, path: Path) -> tuple[list[dict], str]:
        connector_errors: list[str] = []
        for connector in self.connectors:
            if not connector.supports_dataset(dataset):
                continue
            try:
                payload = connector.fetch(
                    dataset=dataset,
                    timeout_seconds=self.config.connector_timeout_seconds,
                    retries=self.config.connector_retries,
                )
                if payload:
                    return payload, f"connector:{connector.provider}"
                connector_errors.append(f"{connector.provider}:empty")
            except Exception as exc:
                connector_errors.append(f"{connector.provider}:{exc}")
        fallback = self._read_json_list(path)
        detail = "json_fallback"
        if connector_errors:
            detail += f"|errors={'; '.join(connector_errors)}"
        return fallback, detail

    def refresh_daily(self) -> EnterpriseDataSnapshot:
        path_map = {
            dataset: getattr(self.config, attr_name) for dataset, attr_name in DATASET_PATH_MAP.items()
        }
        datasets: dict[str, list[dict]] = {}
        connector_status: dict[str, str] = {}
        stale_feeds: list[str] = []
        partial_feeds: list[str] = []
        for dataset, path in path_map.items():
            rows, status = self._load_dataset(dataset, path)
            datasets[dataset] = rows
            connector_status[dataset] = status
            if status.startswith("json_fallback"):
                partial_feeds.append(dataset)
            if self._dataset_is_stale(dataset, rows):
                stale_feeds.append(dataset)
        refreshed_at = datetime.now(timezone.utc).isoformat()
        lineage = {
            "instrument_master": str(self.config.instrument_master_path),
            "corporate_actions": str(self.config.corporate_actions_path),
            "filings": str(self.config.filings_path),
            "regulatory_updates": str(self.config.regulatory_updates_path),
            "market_events": str(self.config.market_events_path),
            "refreshed_at": refreshed_at,
            "connector_status": json.dumps(connector_status, ensure_ascii=False),
        }
        self._snapshot = EnterpriseDataSnapshot(
            instrument_master=datasets["instrument_master"],
            corporate_actions=datasets["corporate_actions"],
            filings=datasets["filings"],
            regulatory_updates=datasets["regulatory_updates"],
            market_events=datasets["market_events"],
            source_hierarchy=self.source_hierarchy,
            refreshed_at=refreshed_at,
            lineage=lineage,
            stale_feeds=tuple(sorted(set(stale_feeds))),
            partial_feeds=tuple(sorted(set(partial_feeds))),
            connector_status=connector_status,
        )
        return self._snapshot

    def validate_snapshot(self) -> bool:
        if not self._snapshot.instrument_master:
            return False
        if self._snapshot.stale_feeds or self._snapshot.partial_feeds:
            return False
        required = ("symbol", "company_name", "isin")
        return all(all(key in item for key in required) for item in self._snapshot.instrument_master)

    @property
    def snapshot(self) -> EnterpriseDataSnapshot:
        return self._snapshot

    def resolve_entity(self, query: str) -> dict | None:
        q = query.lower()
        for item in self._snapshot.instrument_master:
            candidates = (
                str(item.get("symbol", "")).lower(),
                str(item.get("company_name", "")).lower(),
                str(item.get("isin", "")).lower(),
            )
            if any(candidate and candidate in q for candidate in candidates):
                return item
        return None

    def readiness_report(self) -> DataReadinessReport:
        blockers: list[str] = []
        total_datasets = len(DATASET_PATH_MAP)
        available_datasets = 0
        fallback_mode = False
        for dataset in DATASET_PATH_MAP:
            rows = getattr(self._snapshot, dataset)
            if rows:
                available_datasets += 1
            if dataset in self._snapshot.stale_feeds:
                blockers.append(f"{dataset}:stale")
            if dataset in self._snapshot.partial_feeds:
                blockers.append(f"{dataset}:partial")
                fallback_mode = True
        completeness = available_datasets / total_datasets if total_datasets else 0.0
        if completeness < 1.0:
            blockers.append("dataset:incomplete")
        lineage_ok = all(
            dataset in self._snapshot.lineage and self._snapshot.lineage.get(dataset)
            for dataset in DATASET_PATH_MAP
        )
        if not lineage_ok:
            blockers.append("lineage:missing")
        freshness_ok = not self._snapshot.stale_feeds
        ready = not blockers and self.validate_snapshot()
        return DataReadinessReport(
            ready=ready,
            blockers=tuple(dict.fromkeys(blockers)),
            completeness=completeness,
            freshness_ok=freshness_ok,
            lineage_ok=lineage_ok,
            fallback_mode=fallback_mode,
        )

    def start_background_refresh(self, interval_seconds: float = 300.0) -> None:
        if self._refresh_worker and self._refresh_worker.is_alive():
            return
        self._refresh_stop_event = Event()

        def _loop() -> None:
            while self._refresh_stop_event and not self._refresh_stop_event.wait(interval_seconds):
                try:
                    self.refresh_daily()
                except Exception as exc:
                    try:
                        self._refresh_errors.put_nowait(str(exc))
                    except Exception:
                        pass

        self._refresh_worker = Thread(target=_loop, daemon=True)
        self._refresh_worker.start()

    def stop_background_refresh(self, timeout_seconds: float = 1.0) -> None:
        if self._refresh_stop_event is None or self._refresh_worker is None:
            return
        self._refresh_stop_event.set()
        self._refresh_worker.join(timeout=timeout_seconds)
        self._refresh_stop_event = None
        self._refresh_worker = None

    def background_refresh_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        while True:
            try:
                errors.append(self._refresh_errors.get_nowait())
            except Empty:
                break
        return tuple(errors)
