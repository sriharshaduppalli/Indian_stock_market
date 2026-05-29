from __future__ import annotations

from dataclasses import dataclass

SUPPORTED_QUERY_CATEGORIES = (
    "stocks",
    "nse_bse_sebi",
    "analysis",
    "calculations",
    "prediction_guidance",
)

TARGET_USE_CASES = (
    "grounded_qna",
    "risk_aware_guidance",
)

PROHIBITED_USE_CASES = (
    "trade_execution",
    "guaranteed_return_advice",
)

API_CONTRACT_VERSION = "v1"


@dataclass(frozen=True)
class ProductionAcceptanceCriteria:
    accuracy_min: float = 0.85
    groundedness_min: float = 0.85
    max_latency_ms: int = 1_200
    min_uptime: float = 0.995
    max_cost_per_query: float = 0.02
    safety_compliance_min: float = 0.98
    max_blocked_ratio: float = 0.2
    max_failure_rate: float = 0.1
