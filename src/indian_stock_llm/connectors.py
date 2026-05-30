from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING
from typing import Protocol
from urllib import error, request

if TYPE_CHECKING:
    from .config import AssistantConfig

DATASET_KEYS = (
    "instrument_master",
    "corporate_actions",
    "filings",
    "regulatory_updates",
    "market_events",
)

REQUIRED_FIELDS = {
    "instrument_master": ("symbol", "company_name", "isin", "exchange"),
    "corporate_actions": ("symbol", "action", "effective_date", "source"),
    "filings": ("symbol", "filing_type", "timestamp", "source"),
    "regulatory_updates": ("authority", "title", "timestamp", "source"),
    "market_events": ("event", "timestamp", "source"),
}


class ConnectorFetchError(RuntimeError):
    pass


class SourceConnector(Protocol):
    provider: str

    def supports_dataset(self, dataset: str) -> bool: ...

    def fetch(self, dataset: str, timeout_seconds: float, retries: int) -> list[dict]: ...


def normalize_record(dataset: str, record: dict, provider: str) -> dict | None:
    required = REQUIRED_FIELDS.get(dataset, ())
    if not required:
        return None
    normalized = {key: record.get(key) for key in required}
    if any(value in (None, "") for value in normalized.values()):
        return None
    normalized["provider"] = provider
    return normalized


@dataclass(frozen=True)
class FileBackedProviderConnector:
    provider: str
    dataset_paths: dict[str, Path]

    def supports_dataset(self, dataset: str) -> bool:
        return dataset in self.dataset_paths

    def fetch(self, dataset: str, timeout_seconds: float, retries: int) -> list[dict]:
        if dataset not in self.dataset_paths:
            raise ConnectorFetchError(f"{self.provider} connector does not support dataset '{dataset}'")
        path = self.dataset_paths[dataset]
        attempts = max(1, retries + 1)
        last_error: Exception | None = None
        for _ in range(attempts):
            started = perf_counter()
            try:
                if not path.exists():
                    raise FileNotFoundError(path)
                raw = json.loads(path.read_text(encoding="utf-8"))
                if not isinstance(raw, list):
                    raise ValueError(f"Expected list JSON for {dataset}")
                elapsed = perf_counter() - started
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"{self.provider}:{dataset} timed out")
                normalized: list[dict] = []
                for row in raw:
                    if not isinstance(row, dict):
                        continue
                    item = normalize_record(dataset, row, self.provider)
                    if item:
                        normalized.append(item)
                return normalized
            except Exception as exc:
                last_error = exc
        raise ConnectorFetchError(f"{self.provider}:{dataset} fetch failed: {last_error}") from last_error


@dataclass(frozen=True)
class HttpJsonProviderConnector:
    provider: str
    dataset_urls: dict[str, str]
    api_key: str | None = None

    def supports_dataset(self, dataset: str) -> bool:
        return dataset in self.dataset_urls

    def fetch(self, dataset: str, timeout_seconds: float, retries: int) -> list[dict]:
        if dataset not in self.dataset_urls:
            raise ConnectorFetchError(f"{self.provider} connector does not support dataset '{dataset}'")
        url = self.dataset_urls[dataset]
        attempts = max(1, retries + 1)
        last_error: Exception | None = None
        for _ in range(attempts):
            started = perf_counter()
            try:
                req = request.Request(url, method="GET")
                req.add_header("Accept", "application/json")
                if self.api_key:
                    req.add_header("X-API-Key", self.api_key)
                with request.urlopen(req, timeout=timeout_seconds) as response:
                    raw_text = response.read().decode("utf-8")
                elapsed = perf_counter() - started
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"{self.provider}:{dataset} timed out")
                parsed = json.loads(raw_text)
                if isinstance(parsed, dict):
                    parsed = parsed.get("data", [])
                if not isinstance(parsed, list):
                    raise ValueError(f"Expected list JSON for {dataset}")
                normalized: list[dict] = []
                for row in parsed:
                    if not isinstance(row, dict):
                        continue
                    item = normalize_record(dataset, row, self.provider)
                    if item:
                        normalized.append(item)
                return normalized
            except (error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
                last_error = exc
        raise ConnectorFetchError(f"{self.provider}:{dataset} fetch failed: {last_error}") from last_error


@dataclass(frozen=True)
class OpenSourceStockConnector:
    provider: str = "open_source"
    symbols: tuple[str, ...] = ("RELIANCE.NS", "TCS.NS", "INFY.NS")

    def supports_dataset(self, dataset: str) -> bool:
        return dataset in {"instrument_master", "market_events"}

    def fetch(self, dataset: str, timeout_seconds: float, retries: int) -> list[dict]:
        attempts = max(1, retries + 1)
        last_error: Exception | None = None
        for _ in range(attempts):
            started = perf_counter()
            try:
                if dataset == "instrument_master":
                    rows = self._fetch_instrument_master()
                elif dataset == "market_events":
                    rows = self._fetch_market_events()
                else:
                    raise ConnectorFetchError(f"{self.provider} connector does not support dataset '{dataset}'")
                elapsed = perf_counter() - started
                if elapsed > timeout_seconds:
                    raise TimeoutError(f"{self.provider}:{dataset} timed out")
                return [item for item in rows if item]
            except Exception as exc:
                last_error = exc
        raise ConnectorFetchError(f"{self.provider}:{dataset} fetch failed: {last_error}") from last_error

    def _fetch_instrument_master(self) -> list[dict]:
        try:
            import yfinance as yf  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install yfinance to enable open-source instrument feeds.") from exc
        records: list[dict] = []
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            info = getattr(ticker, "info", {}) or {}
            row = normalize_record(
                "instrument_master",
                {
                    "symbol": symbol,
                    "company_name": info.get("shortName") or symbol,
                    "isin": info.get("isin") or f"YF-{symbol.replace('.', '-')}",
                    "exchange": info.get("exchange") or "NSE",
                },
                self.provider,
            )
            if row:
                records.append(row)
        return records

    def _fetch_market_events(self) -> list[dict]:
        try:
            from nsepython import nse_marketStatus  # type: ignore
        except Exception as exc:
            raise RuntimeError("Install nsepython to enable open-source NSE market events.") from exc
        payload = nse_marketStatus()
        timestamp = datetime.now(timezone.utc).isoformat()
        if isinstance(payload, dict):
            payload = [payload]
        rows: list[dict] = []
        for item in payload if isinstance(payload, list) else []:
            event_text = str(item.get("market", "NSE market status"))
            status_text = str(item.get("marketStatus", item.get("status", "unknown")))
            row = normalize_record(
                "market_events",
                {
                    "event": f"{event_text}: {status_text}",
                    "timestamp": timestamp,
                    "source": "nsepython",
                },
                self.provider,
            )
            if row:
                rows.append(row)
        return rows


def default_connectors(config: "AssistantConfig") -> tuple[SourceConnector, ...]:
    base_connectors: tuple[SourceConnector, ...]
    if config.live_connectors_enabled:
        live_endpoints = {
            "nse": {
                "instrument_master": config.nse_connector_url,
                "corporate_actions": config.nse_connector_url,
            },
            "bse": {
                "filings": config.bse_connector_url or config.filings_connector_url,
            },
            "regulatory": {
                "regulatory_updates": config.regulatory_connector_url,
            },
            "news": {
                "market_events": config.news_connector_url,
            },
        }
        live_connectors: list[SourceConnector] = []
        for provider, mapping in live_endpoints.items():
            dataset_urls = {dataset: url for dataset, url in mapping.items() if url}
            if dataset_urls:
                live_connectors.append(
                    HttpJsonProviderConnector(
                        provider=provider,
                        dataset_urls=dataset_urls,
                        api_key=config.connector_api_key,
                    )
                )
        if live_connectors:
            base_connectors = tuple(live_connectors)
        else:
            base_connectors = ()
    else:
        base_connectors = ()

    if not base_connectors:
        base_connectors = (
            FileBackedProviderConnector(
                provider="nse",
                dataset_paths={
                    "instrument_master": config.instrument_master_path,
                    "corporate_actions": config.corporate_actions_path,
                },
            ),
            FileBackedProviderConnector(
                provider="bse",
                dataset_paths={"filings": config.filings_path},
            ),
            FileBackedProviderConnector(
                provider="regulatory",
                dataset_paths={"regulatory_updates": config.regulatory_updates_path},
            ),
            FileBackedProviderConnector(
                provider="news",
                dataset_paths={"market_events": config.market_events_path},
            ),
        )

    if config.open_source_market_data_enabled:
        return (
            OpenSourceStockConnector(symbols=config.open_source_symbols),
            *base_connectors,
        )

    return base_connectors
