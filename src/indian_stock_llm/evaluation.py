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


def passes_release_gate(result: BenchmarkResult, criteria: ProductionAcceptanceCriteria) -> bool:
    return (
        result.fact_accuracy >= criteria.accuracy_min
        and result.calculation_correctness >= criteria.accuracy_min
        and result.groundedness >= criteria.accuracy_min
        and result.hallucination_rate <= 1 - criteria.accuracy_min
        and result.safety_score >= criteria.safety_compliance_min
        and result.routing_accuracy >= criteria.accuracy_min
    )

