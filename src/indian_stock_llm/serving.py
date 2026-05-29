from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import logging
from time import perf_counter

from .query_engine import StockMarketAssistant

LOGGER = logging.getLogger(__name__)


@dataclass
class ServiceMetrics:
    total_requests: int = 0
    cache_hits: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests else 0.0


class ChatService:
    def __init__(
        self,
        assistant: StockMarketAssistant,
        rate_limit_per_minute: int = 60,
        circuit_breaker_threshold: int = 3,
        circuit_cooldown_seconds: int = 30,
    ):
        self.assistant = assistant
        self.rate_limit_per_minute = rate_limit_per_minute
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self._cache: dict[str, dict] = {}
        self._request_times: list[datetime] = []
        self._circuit_open = False
        self._circuit_opened_at: datetime | None = None
        self._consecutive_failures = 0
        self.metrics = ServiceMetrics()

    def _allow_request(self) -> bool:
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        self._request_times = [ts for ts in self._request_times if ts >= minute_ago]
        if len(self._request_times) >= self.rate_limit_per_minute:
            return False
        self._request_times.append(now)
        return True

    def query(self, user_query: str) -> dict:
        if not self._allow_request():
            return {"status": "rate_limited", "response": self._fallback_response("Rate limit exceeded")}
        if user_query in self._cache:
            self.metrics.cache_hits += 1
            return {"status": "ok", "response": self._cache[user_query], "cached": True}
        if self._circuit_open:
            now = datetime.now(timezone.utc)
            if self._circuit_opened_at and now - self._circuit_opened_at < timedelta(seconds=self.circuit_cooldown_seconds):
                return {"status": "degraded", "response": self._fallback_response("Service temporarily degraded")}
            self._circuit_open = False
            self._circuit_opened_at = None

        start = perf_counter()
        self.metrics.total_requests += 1
        last_error: Exception | None = None
        for _ in range(2):
            try:
                payload = self.assistant.query(user_query)
                self._cache[user_query] = payload
                self._consecutive_failures = 0
                self.metrics.total_latency_ms += (perf_counter() - start) * 1000
                return {"status": "ok", "response": payload, "cached": False}
            except Exception as exc:
                last_error = exc
                LOGGER.warning("ChatService query retry failed: %s", exc)
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.circuit_breaker_threshold:
            self._circuit_open = True
            self._circuit_opened_at = datetime.now(timezone.utc)
        self.metrics.failures += 1
        if last_error:
            LOGGER.error("ChatService request failed after retries: %s", last_error)
        return {"status": "failed", "response": self._fallback_response("Unable to process query safely")}

    @staticmethod
    def _fallback_response(reason: str) -> dict:
        return {
            "intent": "fallback",
            "answer": f"Fallback response: {reason}. Please retry shortly.",
            "confidence": 0.0,
            "citations": [],
            "disclaimer": "Informational only.",
            "safe_for_trading_advice": False,
        }
