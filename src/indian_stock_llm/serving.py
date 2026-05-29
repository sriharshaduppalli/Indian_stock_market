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
    failed_responses: int = 0
    rate_limited: int = 0
    degraded: int = 0
    safety_blocks: int = 0
    auth_failures: int = 0
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
        tenant_api_keys: dict[str, str] | None = None,
    ):
        self.assistant = assistant
        self.rate_limit_per_minute = rate_limit_per_minute
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self._cache: dict[str, dict] = {}
        self._request_times: list[datetime] = []
        self._tenant_request_times: dict[str, list[datetime]] = {}
        self._tenant_cache: dict[str, dict[str, dict]] = {}
        self._tenant_api_keys = tenant_api_keys or {}
        self._circuit_open = False
        self._circuit_opened_at: datetime | None = None
        self._consecutive_failures = 0
        self.metrics = ServiceMetrics()

    def _allow_request(self, tenant_id: str) -> bool:
        now = datetime.now(timezone.utc)
        minute_ago = now - timedelta(minutes=1)
        self._request_times = [ts for ts in self._request_times if ts >= minute_ago]
        tenant_times = self._tenant_request_times.get(tenant_id, [])
        tenant_times = [ts for ts in tenant_times if ts >= minute_ago]
        self._tenant_request_times[tenant_id] = tenant_times
        if len(tenant_times) >= self.rate_limit_per_minute:
            return False
        self._request_times.append(now)
        self._tenant_request_times[tenant_id].append(now)
        return True

    def register_tenant(self, tenant_id: str, api_key: str) -> None:
        self._tenant_api_keys[tenant_id] = api_key

    def _authenticate(self, tenant_id: str, api_key: str | None) -> bool:
        expected = self._tenant_api_keys.get(tenant_id)
        if expected is None:
            return True
        return api_key == expected

    def query(self, user_query: str, *, tenant_id: str = "public", api_key: str | None = None) -> dict:
        if not self._authenticate(tenant_id, api_key):
            self.metrics.auth_failures += 1
            return {"status": "unauthorized", "response": self._fallback_response("Unauthorized tenant access")}
        if not self._allow_request(tenant_id):
            self.metrics.rate_limited += 1
            return {"status": "rate_limited", "response": self._fallback_response("Rate limit exceeded")}
        tenant_cache = self._tenant_cache.setdefault(tenant_id, {})
        if user_query in tenant_cache:
            self.metrics.cache_hits += 1
            return {"status": "ok", "response": tenant_cache[user_query], "cached": True}
        if self._circuit_open:
            now = datetime.now(timezone.utc)
            if self._circuit_opened_at and now - self._circuit_opened_at < timedelta(seconds=self.circuit_cooldown_seconds):
                self.metrics.degraded += 1
                return {"status": "degraded", "response": self._fallback_response("Service temporarily degraded")}
            self._circuit_open = False
            self._circuit_opened_at = None

        start = perf_counter()
        self.metrics.total_requests += 1
        last_error: Exception | None = None
        for _ in range(2):
            try:
                payload = self.assistant.query(user_query)
                payload["tenant_id"] = tenant_id
                tenant_cache[user_query] = payload
                self._consecutive_failures = 0
                if payload.get("policy_reason", "").lower().startswith(("prompt-injection", "prohibited")):
                    self.metrics.safety_blocks += 1
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
        self.metrics.failed_responses += 1
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

    def export_metrics(self) -> dict[str, float | bool]:
        total = self.metrics.total_requests
        failure_rate = (self.metrics.failures / total) if total else 0.0
        cache_hit_rate = (self.metrics.cache_hits / total) if total else 0.0
        return {
            "total_requests": float(total),
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "failure_rate": failure_rate,
            "cache_hit_rate": cache_hit_rate,
            "failed_responses": float(self.metrics.failed_responses),
            "rate_limited": float(self.metrics.rate_limited),
            "degraded": float(self.metrics.degraded),
            "safety_blocks": float(self.metrics.safety_blocks),
            "auth_failures": float(self.metrics.auth_failures),
            "circuit_open": self._circuit_open,
        }
