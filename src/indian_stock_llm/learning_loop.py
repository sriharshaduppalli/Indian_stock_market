from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


class ContinualLearningManager:
    """Simple hook for capturing daily feedback used in later model updates."""

    def __init__(self, feedback_log_path: Path | None):
        self.feedback_log_path = feedback_log_path

    def record_feedback(self, query: str, intent: str) -> None:
        if self.feedback_log_path is None:
            return

        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        with self.feedback_log_path.open("a", encoding="utf-8") as fp:
            fp.write(f"{ts}\t{intent}\t{query.strip()}\n")

    def daily_learning_summary(self) -> str:
        if self.feedback_log_path is None:
            return "Daily learning loop disabled: no feedback log path configured."

        query_count = 0
        if self.feedback_log_path.exists():
            with self.feedback_log_path.open("r", encoding="utf-8") as fp:
                query_count = sum(1 for _ in fp)
        return (
            f"Daily learning loop enabled: {query_count} feedback samples logged; data can be used to refresh retrieval, "
            "recalibrate prediction factors, and improve next-day responses."
        )

    @staticmethod
    def anonymize_query(query: str) -> str:
        return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()

    def record_anonymized_feedback(self, query: str, intent: str) -> None:
        if self.feedback_log_path is None:
            return
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).isoformat()
        payload = {"ts": ts, "intent": intent, "query_hash": self.anonymize_query(query)}
        with self.feedback_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")
