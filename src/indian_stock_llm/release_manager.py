from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .evaluation import ReleaseGateReport
from .evaluation import (
    RegressionMetrics,
    evaluate_release_gate,
    load_automated_gate_inputs,
    load_automated_gate_inputs_from_endpoint,
    passes_regression_gate,
)
from .acceptance import ProductionAcceptanceCriteria


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


@dataclass(frozen=True)
class RolloutAutomationResult:
    canary: RolloutDecision
    rollout: RolloutDecision
    regression_passed: bool
    promoted: bool


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

    def automate_rollout(
        self,
        *,
        version: str,
        notes: str,
        gate_report: ReleaseGateReport,
        regression: RegressionMetrics,
        rollback_rate: float,
        canary_error_rate: float,
        max_canary_error_rate: float = 0.05,
        max_rollback_rate: float = 0.1,
        auto_promote: bool = False,
    ) -> RolloutAutomationResult:
        canary = self.assess_canary(
            gate_report=gate_report,
            canary_error_rate=canary_error_rate,
            max_canary_error_rate=max_canary_error_rate,
        )
        if not canary.approved:
            blocked = RolloutDecision(
                approved=False,
                rollback_target=self.rollback_target(),
                reason=f"rollout blocked: {canary.reason}",
            )
            return RolloutAutomationResult(canary=canary, rollout=blocked, regression_passed=False, promoted=False)

        regression_passed = passes_regression_gate(regression)
        if not regression_passed:
            blocked = RolloutDecision(
                approved=False,
                rollback_target=self.rollback_target(),
                reason="rollout blocked: regression gate failed",
            )
            return RolloutAutomationResult(canary=canary, rollout=blocked, regression_passed=False, promoted=False)

        rollout = self.assess_rollout(
            gate_report=gate_report,
            rollback_rate=rollback_rate,
            max_rollback_rate=max_rollback_rate,
        )
        promoted = rollout.approved and auto_promote
        if promoted:
            self.add_version(version, notes)
        return RolloutAutomationResult(
            canary=canary,
            rollout=rollout,
            regression_passed=regression_passed,
            promoted=promoted,
        )

    def automate_rollout_from_inputs(
        self,
        *,
        version: str,
        notes: str,
        input_path: Path,
        criteria: ProductionAcceptanceCriteria,
        rollback_rate: float,
        canary_error_rate: float,
        max_age_minutes: int = 30,
        max_canary_error_rate: float = 0.05,
        max_rollback_rate: float = 0.1,
        auto_promote: bool = False,
    ) -> RolloutAutomationResult:
        gate_inputs = load_automated_gate_inputs(input_path, max_age_minutes=max_age_minutes)
        gate_report = evaluate_release_gate(
            benchmark=gate_inputs.benchmark,
            online=gate_inputs.online,
            criteria=criteria,
        )
        return self.automate_rollout(
            version=version,
            notes=notes,
            gate_report=gate_report,
            regression=gate_inputs.regression,
            rollback_rate=rollback_rate,
            canary_error_rate=canary_error_rate,
            max_canary_error_rate=max_canary_error_rate,
            max_rollback_rate=max_rollback_rate,
            auto_promote=auto_promote,
        )

    def automate_rollout_from_endpoint(
        self,
        *,
        version: str,
        notes: str,
        endpoint: str,
        criteria: ProductionAcceptanceCriteria,
        rollback_rate: float,
        canary_error_rate: float,
        api_key: str | None = None,
        timeout_seconds: float = 2.0,
        max_age_minutes: int = 30,
        max_canary_error_rate: float = 0.05,
        max_rollback_rate: float = 0.1,
        auto_promote: bool = False,
    ) -> RolloutAutomationResult:
        gate_inputs = load_automated_gate_inputs_from_endpoint(
            endpoint=endpoint,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
            max_age_minutes=max_age_minutes,
        )
        gate_report = evaluate_release_gate(
            benchmark=gate_inputs.benchmark,
            online=gate_inputs.online,
            criteria=criteria,
        )
        return self.automate_rollout(
            version=version,
            notes=notes,
            gate_report=gate_report,
            regression=gate_inputs.regression,
            rollback_rate=rollback_rate,
            canary_error_rate=canary_error_rate,
            max_canary_error_rate=max_canary_error_rate,
            max_rollback_rate=max_rollback_rate,
            auto_promote=auto_promote,
        )
