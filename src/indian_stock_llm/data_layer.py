from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import AssistantConfig


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


class EnterpriseDataLayer:
    def __init__(self, config: AssistantConfig):
        self.config = config
        self.source_hierarchy = (
            "nse_primary",
            "bse_secondary",
            "regulatory_primary",
            "news_events_secondary",
        )
        self._snapshot = self.refresh_daily()

    def _read_json_list(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected list JSON in {path}")
        return [item for item in data if isinstance(item, dict)]

    def refresh_daily(self) -> EnterpriseDataSnapshot:
        instrument_master = self._read_json_list(self.config.instrument_master_path)
        corporate_actions = self._read_json_list(self.config.corporate_actions_path)
        filings = self._read_json_list(self.config.filings_path)
        regulatory_updates = self._read_json_list(self.config.regulatory_updates_path)
        market_events = self._read_json_list(self.config.market_events_path)
        refreshed_at = datetime.now(timezone.utc).isoformat()
        lineage = {
            "instrument_master": str(self.config.instrument_master_path),
            "corporate_actions": str(self.config.corporate_actions_path),
            "filings": str(self.config.filings_path),
            "regulatory_updates": str(self.config.regulatory_updates_path),
            "market_events": str(self.config.market_events_path),
            "refreshed_at": refreshed_at,
        }
        self._snapshot = EnterpriseDataSnapshot(
            instrument_master=instrument_master,
            corporate_actions=corporate_actions,
            filings=filings,
            regulatory_updates=regulatory_updates,
            market_events=market_events,
            source_hierarchy=self.source_hierarchy,
            refreshed_at=refreshed_at,
            lineage=lineage,
        )
        return self._snapshot

    def validate_snapshot(self) -> bool:
        if not self._snapshot.instrument_master:
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

