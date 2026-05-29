from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Protocol

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

