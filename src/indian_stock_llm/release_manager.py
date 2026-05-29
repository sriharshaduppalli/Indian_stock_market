from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import ReleaseGateReport


@dataclass(frozen=True)
class ReleaseVersion:
    version: str
    created_at: str
    notes: str


@dataclass(frozen=True)
class RolloutDecision:
    approved: bool
    rollback_target: str | None
    reason: str
    canary_only: bool = False


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

    def assess_rollout(self, gate_report: ReleaseGateReport, rollback_rate: float, max_rollback_rate: float = 0.1) -> RolloutDecision:
        if not gate_report.passed:
            return RolloutDecision(
                approved=False,
                rollback_target=self.rollback_target(),
                reason=f"release gate failed: {', '.join(gate_report.reasons) or 'unknown reason'}",
            )
        if rollback_rate > max_rollback_rate:
            return RolloutDecision(
                approved=False,
                rollback_target=self.rollback_target(),
                reason="rollback-rate threshold exceeded",
            )
        return RolloutDecision(approved=True, rollback_target=None, reason="rollout criteria satisfied")

    def assess_canary(
        self,
        gate_report: ReleaseGateReport,
        canary_error_rate: float,
        max_canary_error_rate: float = 0.05,
    ) -> RolloutDecision:
        if not gate_report.passed:
            return RolloutDecision(
                approved=False,
                rollback_target=self.rollback_target(),
                reason="canary blocked: release gate unmet",
                canary_only=True,
            )
        if canary_error_rate > max_canary_error_rate:
            return RolloutDecision(
                approved=False,
                rollback_target=self.rollback_target(),
                reason="canary blocked: error-rate threshold exceeded",
                canary_only=True,
            )
        return RolloutDecision(
            approved=True,
            rollback_target=None,
            reason="canary criteria satisfied",
            canary_only=True,
        )
