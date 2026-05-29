from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssistantConfig:
    """Configuration for the local assistant scaffold."""

    knowledge_base_path: Path
    top_k_context: int = 3
    min_retrieval_score: float = 0.1
    feedback_log_path: Path | None = None
    latency_mode: str = "fast"


def default_config() -> AssistantConfig:
    root = Path(__file__).resolve().parents[2]
    return AssistantConfig(
        knowledge_base_path=root / "data" / "sample_knowledge.json",
        feedback_log_path=root / "data" / "daily_feedback.log",
        latency_mode="fast",
    )
