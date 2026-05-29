from __future__ import annotations

from dataclasses import dataclass
import os
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
    runtime_env: str = "local"
    deployment_region: str = "ap-south-1"
    live_connectors_enabled: bool = False
    nse_connector_url: str | None = None
    bse_connector_url: str | None = None
    filings_connector_url: str | None = None
    regulatory_connector_url: str | None = None
    news_connector_url: str | None = None
    connector_api_key: str | None = None
    monitoring_backend: str = "none"
    monitoring_endpoint: str | None = None
    monitoring_api_key: str | None = None
    rollout_auto_promote: bool = False
    rollout_max_canary_error_rate: float = 0.05
    rollout_max_rollback_rate: float = 0.1


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


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def runtime_config_from_env(base: AssistantConfig | None = None) -> AssistantConfig:
    config = base or default_config()
    return AssistantConfig(
        **{
            **config.__dict__,
            "runtime_env": os.getenv("ISM_RUNTIME_ENV", config.runtime_env),
            "deployment_region": os.getenv("ISM_DEPLOYMENT_REGION", config.deployment_region),
            "latency_mode": os.getenv("ISM_LATENCY_MODE", config.latency_mode),
            "live_connectors_enabled": _env_bool("ISM_LIVE_CONNECTORS_ENABLED", config.live_connectors_enabled),
            "nse_connector_url": os.getenv("ISM_NSE_CONNECTOR_URL", config.nse_connector_url),
            "bse_connector_url": os.getenv("ISM_BSE_CONNECTOR_URL", config.bse_connector_url),
            "filings_connector_url": os.getenv("ISM_FILINGS_CONNECTOR_URL", config.filings_connector_url),
            "regulatory_connector_url": os.getenv("ISM_REGULATORY_CONNECTOR_URL", config.regulatory_connector_url),
            "news_connector_url": os.getenv("ISM_NEWS_CONNECTOR_URL", config.news_connector_url),
            "connector_api_key": os.getenv("ISM_CONNECTOR_API_KEY", config.connector_api_key),
            "monitoring_backend": os.getenv("ISM_MONITORING_BACKEND", config.monitoring_backend),
            "monitoring_endpoint": os.getenv("ISM_MONITORING_ENDPOINT", config.monitoring_endpoint),
            "monitoring_api_key": os.getenv("ISM_MONITORING_API_KEY", config.monitoring_api_key),
            "connector_timeout_seconds": _env_float("ISM_CONNECTOR_TIMEOUT_SECONDS", config.connector_timeout_seconds),
            "connector_retries": _env_int("ISM_CONNECTOR_RETRIES", config.connector_retries),
            "max_data_staleness_hours": _env_int("ISM_MAX_DATA_STALENESS_HOURS", config.max_data_staleness_hours),
            "max_latency_ms": _env_int("ISM_MAX_LATENCY_MS", config.max_latency_ms),
            "min_uptime": _env_float("ISM_MIN_UPTIME", config.min_uptime),
            "max_cost_per_query": _env_float("ISM_MAX_COST_PER_QUERY", config.max_cost_per_query),
            "min_confidence_threshold": _env_float("ISM_MIN_CONFIDENCE_THRESHOLD", config.min_confidence_threshold),
            "groundedness_min": _env_float("ISM_GROUNDEDNESS_MIN", config.groundedness_min),
            "rollout_auto_promote": _env_bool("ISM_ROLLOUT_AUTO_PROMOTE", config.rollout_auto_promote),
            "rollout_max_canary_error_rate": _env_float(
                "ISM_ROLLOUT_MAX_CANARY_ERROR_RATE", config.rollout_max_canary_error_rate
            ),
            "rollout_max_rollback_rate": _env_float(
                "ISM_ROLLOUT_MAX_ROLLBACK_RATE", config.rollout_max_rollback_rate
            ),
        }
    )
