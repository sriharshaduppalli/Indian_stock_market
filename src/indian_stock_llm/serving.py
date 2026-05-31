from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hmac
import json
import logging
from pathlib import Path
from time import perf_counter
from typing import Protocol

from .monitoring import evaluate_sre_readiness
from .monitoring import MonitoringBackend, NullMonitoringBackend
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
    invalid_requests: int = 0
    total_latency_ms: float = 0.0
    latency_samples_ms: list[float] | None = None

    def __post_init__(self) -> None:
        if self.latency_samples_ms is None:
            self.latency_samples_ms = []

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.total_requests if self.total_requests else 0.0


@dataclass(frozen=True)
class TenantPolicy:
    rate_limit_per_minute: int | None = None
    max_query_length: int | None = None


@dataclass(frozen=True)
class TenantCredential:
    secret: str
    key_id: str = "active"
    expires_at: str | None = None


class StateBackend(Protocol):
    def allow_request(self, tenant_id: str, now: datetime, per_minute_limit: int) -> bool: ...

    def get_cached_response(self, tenant_id: str, query: str, ttl_seconds: int) -> dict | None: ...

    def set_cached_response(self, tenant_id: str, query: str, payload: dict, now: datetime, max_entries: int) -> None: ...


class InMemoryStateBackend:
    def __init__(self):
        self._tenant_request_times: dict[str, list[datetime]] = {}
        self._tenant_cache: dict[str, OrderedDict[str, tuple[dict, datetime]]] = {}

    def allow_request(self, tenant_id: str, now: datetime, per_minute_limit: int) -> bool:
        minute_ago = now - timedelta(minutes=1)
        tenant_times = [ts for ts in self._tenant_request_times.get(tenant_id, []) if ts >= minute_ago]
        self._tenant_request_times[tenant_id] = tenant_times
        if len(tenant_times) >= per_minute_limit:
            return False
        self._tenant_request_times[tenant_id].append(now)
        return True

    def get_cached_response(self, tenant_id: str, query: str, ttl_seconds: int) -> dict | None:
        tenant_cache = self._tenant_cache.setdefault(tenant_id, OrderedDict())
        entry = tenant_cache.get(query)
        if entry is None:
            return None
        payload, created_at = entry
        if datetime.now(timezone.utc) - created_at >= timedelta(seconds=ttl_seconds):
            tenant_cache.pop(query, None)
            return None
        tenant_cache.move_to_end(query)
        return payload

    def set_cached_response(self, tenant_id: str, query: str, payload: dict, now: datetime, max_entries: int) -> None:
        tenant_cache = self._tenant_cache.setdefault(tenant_id, OrderedDict())
        tenant_cache[query] = (payload, now)
        tenant_cache.move_to_end(query)
        while len(tenant_cache) > max_entries:
            tenant_cache.popitem(last=False)


class FileStateBackend(InMemoryStateBackend):
    """Minimal distributed-ready state backend using a shared JSON file."""

    def __init__(self, path: Path):
        super().__init__()
        self.path = path
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        reqs = payload.get("requests", {}) if isinstance(payload, dict) else {}
        cache = payload.get("cache", {}) if isinstance(payload, dict) else {}
        self._tenant_request_times = {
            tenant: [datetime.fromisoformat(ts) for ts in values if isinstance(ts, str)]
            for tenant, values in reqs.items()
            if isinstance(values, list)
        }
        self._tenant_cache = {}
        for tenant, rows in cache.items():
            ordered: OrderedDict[str, tuple[dict, datetime]] = OrderedDict()
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                query = row.get("query")
                payload_row = row.get("payload")
                created_at = row.get("created_at")
                if not isinstance(query, str) or not isinstance(payload_row, dict) or not isinstance(created_at, str):
                    continue
                ordered[query] = (payload_row, datetime.fromisoformat(created_at))
            self._tenant_cache[tenant] = ordered

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "requests": {
                tenant: [ts.isoformat() for ts in values]
                for tenant, values in self._tenant_request_times.items()
            },
            "cache": {
                tenant: [
                    {"query": query, "payload": item[0], "created_at": item[1].isoformat()}
                    for query, item in rows.items()
                ]
                for tenant, rows in self._tenant_cache.items()
            },
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def allow_request(self, tenant_id: str, now: datetime, per_minute_limit: int) -> bool:
        allowed = super().allow_request(tenant_id, now, per_minute_limit)
        self._save()
        return allowed

    def get_cached_response(self, tenant_id: str, query: str, ttl_seconds: int) -> dict | None:
        payload = super().get_cached_response(tenant_id, query, ttl_seconds)
        self._save()
        return payload

    def set_cached_response(self, tenant_id: str, query: str, payload: dict, now: datetime, max_entries: int) -> None:
        super().set_cached_response(tenant_id, query, payload, now, max_entries)
        self._save()


class ChatService:
    def __init__(
        self,
        assistant: StockMarketAssistant,
        rate_limit_per_minute: int = 60,
        circuit_breaker_threshold: int = 3,
        circuit_cooldown_seconds: int = 30,
        tenant_api_keys: dict[str, str | list[str]] | None = None,
        cache_ttl_seconds: int = 300,
        max_query_length: int = 2_000,
        monitoring_backend: MonitoringBackend | None = None,
        tenant_policies: dict[str, TenantPolicy] | None = None,
        state_backend: StateBackend | None = None,
        max_cache_entries_per_tenant: int = 256,
    ):
        self.assistant = assistant
        self.rate_limit_per_minute = rate_limit_per_minute
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_cooldown_seconds = circuit_cooldown_seconds
        self._tenant_credentials: dict[str, tuple[TenantCredential, ...]] = {}
        self._tenant_api_keys = tenant_api_keys or {}
        for tenant_id, keyset in self._tenant_api_keys.items():
            self.register_tenant(tenant_id, keyset)
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_query_length = max_query_length
        self.max_cache_entries_per_tenant = max_cache_entries_per_tenant
        self.tenant_policies = tenant_policies or {}
        self._circuit_open = False
        self._circuit_opened_at: datetime | None = None
        self._consecutive_failures = 0
        self.metrics = ServiceMetrics()
        self.monitoring_backend = monitoring_backend or NullMonitoringBackend()
        self.state_backend = state_backend or InMemoryStateBackend()
        self._active_slo_alerts: set[str] = set()

    def _emit_event(self, event: str, payload: dict[str, str | float | bool]) -> None:
        self.monitoring_backend.emit_event(event, payload)

    def _emit_metrics(self) -> None:
        metrics = self.export_metrics()
        self.monitoring_backend.emit_metrics(metrics)
        readiness = evaluate_sre_readiness(metrics)
        current_alerts = set(readiness.get("alerts", ()))
        for alert in sorted(current_alerts - self._active_slo_alerts):
            self.monitoring_backend.emit_event(
                "slo.alert",
                {
                    "alert": alert,
                    "runbook": str(readiness.get("runbook", "")),
                    "rollback_drill": str(readiness.get("rollback_drill", "")),
                },
            )
        for cleared in sorted(self._active_slo_alerts - current_alerts):
            self.monitoring_backend.emit_event(
                "slo.alert_cleared",
                {
                    "alert": cleared,
                    "runbook": str(readiness.get("runbook", "")),
                },
            )
        self._active_slo_alerts = current_alerts

    def _policy_for(self, tenant_id: str) -> TenantPolicy:
        return self.tenant_policies.get(tenant_id, TenantPolicy())

    @staticmethod
    def _safe_compare(value: str, expected: str) -> bool:
        return hmac.compare_digest(value.encode("utf-8"), expected.encode("utf-8"))

    @staticmethod
    def _not_expired(expires_at: str | None) -> bool:
        if not expires_at:
            return True
        text = expires_at[:-1] + "+00:00" if expires_at.endswith("Z") else expires_at
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed >= datetime.now(timezone.utc)

    def register_tenant(
        self,
        tenant_id: str,
        api_key: str | list[str],
        *,
        key_id: str = "active",
        expires_at: str | None = None,
    ) -> None:
        keys = api_key if isinstance(api_key, list) else [api_key]
        credentials = tuple(TenantCredential(secret=secret, key_id=key_id, expires_at=expires_at) for secret in keys if secret)
        self._tenant_credentials[tenant_id] = credentials

    def _authenticate(self, tenant_id: str, api_key: str | None) -> bool:
        if not self._tenant_credentials:
            return True
        if not api_key:
            return False
        candidates = self._tenant_credentials.get(tenant_id)
        if not candidates:
            return False
        for credential in candidates:
            if not self._not_expired(credential.expires_at):
                continue
            if self._safe_compare(api_key, credential.secret):
                return True
        return False

    def _allow_request(self, tenant_id: str) -> bool:
        now = datetime.now(timezone.utc)
        policy = self._policy_for(tenant_id)
        limit = policy.rate_limit_per_minute or self.rate_limit_per_minute
        return self.state_backend.allow_request(tenant_id=tenant_id, now=now, per_minute_limit=limit)

    def _valid_query(self, tenant_id: str, query: str) -> bool:
        trimmed = query.strip()
        policy = self._policy_for(tenant_id)
        max_len = policy.max_query_length or self.max_query_length
        return bool(trimmed) and len(trimmed) <= max_len

    def _get_cached_response(self, tenant_id: str, query: str) -> dict | None:
        return self.state_backend.get_cached_response(tenant_id, query, self.cache_ttl_seconds)

    def query(self, user_query: str, *, tenant_id: str = "public", api_key: str | None = None) -> dict:
        if not self._valid_query(tenant_id, user_query):
            self.metrics.invalid_requests += 1
            self._emit_event("request.rejected", {"reason": "bad_request", "tenant_id": tenant_id})
            self._emit_metrics()
            return {"status": "bad_request", "response": self._fallback_response("Query must be non-empty and within max length")}
        if not self._authenticate(tenant_id, api_key):
            self.metrics.auth_failures += 1
            self._emit_event("request.rejected", {"reason": "unauthorized", "tenant_id": tenant_id})
            self._emit_metrics()
            return {"status": "unauthorized", "response": self._fallback_response("Unauthorized tenant access")}
        if not self._allow_request(tenant_id):
            self.metrics.rate_limited += 1
            self._emit_event("request.rejected", {"reason": "rate_limited", "tenant_id": tenant_id})
            self._emit_metrics()
            return {"status": "rate_limited", "response": self._fallback_response("Rate limit exceeded")}
        normalized_query = user_query.strip()
        cached_payload = self._get_cached_response(tenant_id, normalized_query)
        if cached_payload is not None:
            self.metrics.cache_hits += 1
            self._emit_event("request.cached", {"tenant_id": tenant_id})
            self._emit_metrics()
            return {"status": "ok", "response": cached_payload, "cached": True}
        if self._circuit_open:
            now = datetime.now(timezone.utc)
            if self._circuit_opened_at and now - self._circuit_opened_at < timedelta(seconds=self.circuit_cooldown_seconds):
                self.metrics.degraded += 1
                self._emit_event("request.degraded", {"tenant_id": tenant_id})
                self._emit_metrics()
                return {"status": "degraded", "response": self._fallback_response("Service temporarily degraded")}
            self._circuit_open = False
            self._circuit_opened_at = None

        start = perf_counter()
        self.metrics.total_requests += 1
        last_error: Exception | None = None
        for _ in range(2):
            try:
                payload = self.assistant.query(normalized_query)
                payload["tenant_id"] = tenant_id
                now = datetime.now(timezone.utc)
                self.state_backend.set_cached_response(
                    tenant_id=tenant_id,
                    query=normalized_query,
                    payload=payload,
                    now=now,
                    max_entries=self.max_cache_entries_per_tenant,
                )
                self._consecutive_failures = 0
                if payload.get("policy_reason", "").lower().startswith(("prompt-injection", "prohibited")):
                    self.metrics.safety_blocks += 1
                latency_ms = (perf_counter() - start) * 1000
                self.metrics.total_latency_ms += latency_ms
                if self.metrics.latency_samples_ms is not None:
                    self.metrics.latency_samples_ms.append(latency_ms)
                self._emit_event("request.ok", {"tenant_id": tenant_id, "cached": False})
                self._emit_metrics()
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
        self._emit_event("request.failed", {"tenant_id": tenant_id})
        self._emit_metrics()
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

    @staticmethod
    def _percentile(values: list[float], percentile: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = int((len(ordered) - 1) * percentile)
        return float(ordered[index])

    def export_metrics(self) -> dict[str, float | bool]:
        total = self.metrics.total_requests
        failure_rate = (self.metrics.failures / total) if total else 0.0
        cache_hit_rate = (self.metrics.cache_hits / total) if total else 0.0
        latencies = self.metrics.latency_samples_ms or []
        error_budget_remaining = max(0.0, 1.0 - failure_rate)
        return {
            "total_requests": float(total),
            "avg_latency_ms": self.metrics.avg_latency_ms,
            "p95_latency_ms": self._percentile(latencies, 0.95),
            "p99_latency_ms": self._percentile(latencies, 0.99),
            "failure_rate": failure_rate,
            "error_budget_remaining": error_budget_remaining,
            "cache_hit_rate": cache_hit_rate,
            "failed_responses": float(self.metrics.failed_responses),
            "rate_limited": float(self.metrics.rate_limited),
            "degraded": float(self.metrics.degraded),
            "safety_blocks": float(self.metrics.safety_blocks),
            "auth_failures": float(self.metrics.auth_failures),
            "invalid_requests": float(self.metrics.invalid_requests),
            "circuit_open": self._circuit_open,
        }

    def refresh(self) -> None:
        """Trigger an immediate knowledge-base index refresh via the assistant."""
        if hasattr(self.assistant, "trigger_index_refresh"):
            self.assistant.trigger_index_refresh()
