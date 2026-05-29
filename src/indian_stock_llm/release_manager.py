from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class ReleaseVersion:
    version: str
    created_at: str
    notes: str


class ReleaseRegistry:
    def __init__(self, registry_path: Path | None):
        self.registry_path = registry_path

    def _load(self) -> list[dict]:
        if self.registry_path is None or not self.registry_path.exists():
            return []
        return json.loads(self.registry_path.read_text(encoding="utf-8"))

    def _save(self, entries: list[dict]) -> None:
        if self.registry_path is None:
            return
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_version(self, version: str, notes: str) -> None:
        entries = self._load()
        entries.append(
            ReleaseVersion(version=version, created_at=datetime.now(timezone.utc).isoformat(), notes=notes).__dict__
        )
        self._save(entries)

    def rollback_target(self) -> str | None:
        entries = self._load()
        if len(entries) < 2:
            return None
        return entries[-2]["version"]

