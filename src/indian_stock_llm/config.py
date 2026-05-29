from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssistantConfig:
    """Configuration for the local assistant scaffold."""

    knowledge_base_path: Path
    instrument_master_path: Path
    corporate_actions_path: Path
    filings_path: Path
    regulatory_updates_path: Path
    market_events_path: Path
    top_k_context: int = 3
    min_retrieval_score: float = 0.2
    feedback_log_path: Path | None = None
    policy_audit_log_path: Path | None = None
    release_registry_path: Path | None = None
    latency_mode: str = "fast"
    connector_timeout_seconds: float = 1.0
    connector_retries: int = 2
    max_data_staleness_hours: int = 1_080
    max_latency_ms: int = 1_200
    min_uptime: float = 0.995
    max_cost_per_query: float = 0.02
    min_confidence_threshold: float = 0.35
    require_ready_data_for_factual: bool = True
    api_contract_version: str = "v1"
    groundedness_min: float = 0.85


def default_config() -> AssistantConfig:
    root = Path(__file__).resolve().parents[2]
    return AssistantConfig(
        knowledge_base_path=root / "data" / "sample_knowledge.json",
        instrument_master_path=root / "data" / "enterprise" / "instrument_master.json",
        corporate_actions_path=root / "data" / "enterprise" / "corporate_actions.json",
        filings_path=root / "data" / "enterprise" / "filings.json",
        regulatory_updates_path=root / "data" / "enterprise" / "regulatory_updates.json",
        market_events_path=root / "data" / "enterprise" / "market_events.json",
        feedback_log_path=root / "data" / "daily_feedback.log",
        policy_audit_log_path=root / "data" / "policy_audit.log",
        release_registry_path=root / "data" / "release_registry.json",
        latency_mode="fast",
    )
