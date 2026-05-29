from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread


class ContinualLearningManager:
    """Simple hook for capturing daily feedback used in later model updates."""

    def __init__(self, feedback_log_path: Path | None, async_logging: bool = True):
        self.feedback_log_path = feedback_log_path
        self._async_logging = async_logging
        self._queue: Queue[str] | None = None
        self._stop_event: Event | None = None
        self._worker: Thread | None = None
        if self.feedback_log_path is not None:
            self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
            self.feedback_log_path.touch(exist_ok=True)
        if self.feedback_log_path is not None and async_logging:
            self._queue = Queue(maxsize=1_000)
            self._stop_event = Event()
            self._worker = Thread(target=self._run_worker, daemon=True)
            self._worker.start()

    def _append_line(self, line: str) -> None:
        if self.feedback_log_path is None:
            return
        self.feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_log_path.open("a", encoding="utf-8") as fp:
            fp.write(line)

    def _run_worker(self) -> None:
        if self._queue is None or self._stop_event is None:
            return
        while not self._stop_event.is_set():
            try:
                line = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                self._append_line(line)
            finally:
                self._queue.task_done()

    def _write_line(self, line: str) -> None:
        if self._queue is not None:
            try:
                self._queue.put_nowait(line)
                return
            except Exception:
                pass
        self._append_line(line)

    def record_feedback(self, query: str, intent: str) -> None:
        if self.feedback_log_path is None:
            return

        ts = datetime.now(timezone.utc).isoformat()
        self._write_line(f"{ts}\t{intent}\t{query.strip()}\n")

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
        ts = datetime.now(timezone.utc).isoformat()
        payload = {"ts": ts, "intent": intent, "query_hash": self.anonymize_query(query)}
        self._write_line(json.dumps(payload, ensure_ascii=False) + "\n")

    def feedback_metrics(self) -> dict[str, float]:
        if self.feedback_log_path is None or not self.feedback_log_path.exists():
            return {
                "feedback_samples": 0.0,
                "anonymized_samples": 0.0,
                "anonymized_ratio": 0.0,
            }
        total = 0
        anonymized = 0
        with self.feedback_log_path.open("r", encoding="utf-8") as fp:
            for line in fp:
                text = line.strip()
                if not text:
                    continue
                total += 1
                if text.startswith("{") and "query_hash" in text:
                    anonymized += 1
        ratio = (anonymized / total) if total else 0.0
        return {
            "feedback_samples": float(total),
            "anonymized_samples": float(anonymized),
            "anonymized_ratio": ratio,
        }

    def close(self, timeout_seconds: float = 1.0) -> None:
        if self._stop_event is None or self._worker is None:
            return
        self._stop_event.set()
        self._worker.join(timeout=timeout_seconds)
