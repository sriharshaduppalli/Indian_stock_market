from __future__ import annotations

from dataclasses import dataclass

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
class ReleaseGateReport:
    benchmark_passed: bool
    online_passed: bool
    reasons: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return self.benchmark_passed and self.online_passed


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
        and benchmark.groundedness >= criteria.accuracy_min
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
            and online.blocked_ratio <= 0.2
            and online.failure_rate <= 0.1
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
        and online.failure_rate <= 0.1
    )
