from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from urllib import request

from .acceptance import ProductionAcceptanceCriteria


@dataclass(frozen=True)
class BenchmarkResult:
    fact_accuracy: float
    calculation_correctness: float
    groundedness: float
    hallucination_rate: float
    safety_score: float
    routing_accuracy: float


@dataclass(frozen=True)
class OnlineFeedbackMetrics:
    uptime: float
    avg_latency_ms: float
    cost_per_query: float
    blocked_ratio: float
    cache_hit_rate: float
    failure_rate: float


@dataclass(frozen=True)
class RegressionMetrics:
    factuality_drop: float = 0.0
    routing_drop: float = 0.0
    safety_drop: float = 0.0


@dataclass(frozen=True)
class ReleaseGateReport:
    benchmark_passed: bool
    online_passed: bool
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.benchmark_passed and self.online_passed


@dataclass(frozen=True)
class AutomatedGateInputs:
    benchmark: BenchmarkResult
    online: OnlineFeedbackMetrics
    regression: RegressionMetrics
    source: str
    ingested_at: str


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_automated_gate_inputs(path: Path, max_age_minutes: int = 30) -> AutomatedGateInputs:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return _build_automated_gate_inputs(payload, max_age_minutes=max_age_minutes)


def _build_automated_gate_inputs(payload: dict, max_age_minutes: int) -> AutomatedGateInputs:
    benchmark = BenchmarkResult(**payload["benchmark"])
    online = OnlineFeedbackMetrics(**payload["online"])
    regression = RegressionMetrics(**payload.get("regression", {}))
    ingested_at = str(payload.get("ingested_at", ""))
    timestamp = _parse_iso_utc(ingested_at) or datetime.now(timezone.utc)
    age = datetime.now(timezone.utc) - timestamp
    if age > timedelta(minutes=max_age_minutes):
        raise ValueError("automated gate input is stale")
    return AutomatedGateInputs(
        benchmark=benchmark,
        online=online,
        regression=regression,
        source=str(payload.get("source", "unknown")),
        ingested_at=timestamp.isoformat(),
    )


def load_automated_gate_inputs_from_endpoint(
    endpoint: str,
    *,
    api_key: str | None = None,
    timeout_seconds: float = 2.0,
    max_age_minutes: int = 30,
) -> AutomatedGateInputs:
    req = request.Request(endpoint, method="GET")
    if api_key:
        req.add_header("X-API-Key", api_key)
    with request.urlopen(req, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("automated gate endpoint returned invalid payload")
    return _build_automated_gate_inputs(payload, max_age_minutes=max_age_minutes)


def passes_release_gate(result: BenchmarkResult, criteria: ProductionAcceptanceCriteria) -> bool:
    return evaluate_release_gate(result, None, criteria).passed


def evaluate_release_gate(
    benchmark: BenchmarkResult,
    online: OnlineFeedbackMetrics | None,
    criteria: ProductionAcceptanceCriteria,
) -> ReleaseGateReport:
    reasons: list[str] = []
    benchmark_passed = (
        benchmark.fact_accuracy >= criteria.accuracy_min
        and benchmark.calculation_correctness >= criteria.accuracy_min
        and benchmark.groundedness >= criteria.groundedness_min
        and benchmark.hallucination_rate <= 1 - criteria.accuracy_min
        and benchmark.safety_score >= criteria.safety_compliance_min
        and benchmark.routing_accuracy >= criteria.accuracy_min
    )
    if not benchmark_passed:
        reasons.append("benchmark thresholds unmet")
    online_passed = True
    if online is not None:
        online_passed = (
            online.uptime >= criteria.min_uptime
            and online.avg_latency_ms <= criteria.max_latency_ms
            and online.cost_per_query <= criteria.max_cost_per_query
            and online.blocked_ratio <= criteria.max_blocked_ratio
            and online.failure_rate <= criteria.max_failure_rate
        )
        if not online_passed:
            reasons.append("online metrics thresholds unmet")
    return ReleaseGateReport(
        benchmark_passed=benchmark_passed,
        online_passed=online_passed,
        reasons=tuple(reasons),
    )


def passes_operational_gate(
    online: OnlineFeedbackMetrics,
    criteria: ProductionAcceptanceCriteria,
) -> bool:
    return (
        online.uptime >= criteria.min_uptime
        and online.avg_latency_ms <= criteria.max_latency_ms
        and online.cost_per_query <= criteria.max_cost_per_query
        and online.failure_rate <= criteria.max_failure_rate
    )


def passes_regression_gate(regression: RegressionMetrics, max_drop: float = 0.03) -> bool:
    return (
        regression.factuality_drop <= max_drop
        and regression.routing_drop <= max_drop
        and regression.safety_drop <= max_drop
    )
