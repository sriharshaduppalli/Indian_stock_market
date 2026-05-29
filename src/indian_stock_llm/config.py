from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AssistantConfig:
    """Configuration for the local assistant scaffold."""

    knowledge_base_path: Path
    top_k_context: int = 3


def default_config() -> AssistantConfig:
    root = Path(__file__).resolve().parents[2]
    return AssistantConfig(knowledge_base_path=root / "data" / "sample_knowledge.json")
