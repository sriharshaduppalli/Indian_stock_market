from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_QUERY_CATEGORIES = (
    "stocks",
    "nse_bse_sebi",
    "analysis",
    "calculations",
    "prediction_guidance",
)


@dataclass(frozen=True)
class ProductionAcceptanceCriteria:
    accuracy_min: float = 0.85
    max_latency_ms: int = 1_200
    min_uptime: float = 0.995
    max_cost_per_query: float = 0.02
    safety_compliance_min: float = 0.98

