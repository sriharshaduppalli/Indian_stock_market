from __future__ import annotations

import json
import logging
from queue import Empty, Queue
from threading import Event, Thread
from dataclasses import dataclass
from typing import Protocol
from urllib import request

LOGGER = logging.getLogger(__name__)


class MonitoringBackend(Protocol):
    def emit_metrics(self, metrics: dict[str, float | bool]) -> None: ...

    def emit_event(self, event: str, payload: dict[str, str | float | bool]) -> None: ...


@dataclass(frozen=True)
class NullMonitoringBackend:
    def emit_metrics(self, metrics: dict[str, float | bool]) -> None:
        _ = metrics

    def emit_event(self, event: str, payload: dict[str, str | float | bool]) -> None:
        _ = (event, payload)


@dataclass(frozen=True)
class LoggingMonitoringBackend:
    def emit_metrics(self, metrics: dict[str, float | bool]) -> None:
        LOGGER.info("service.metrics %s", metrics)

    def emit_event(self, event: str, payload: dict[str, str | float | bool]) -> None:
        LOGGER.info("service.event %s %s", event, payload)


@dataclass(frozen=True)
class HttpMonitoringBackend:
    endpoint: str
    api_key: str | None = None
    timeout_seconds: float = 1.5

    def _post(self, body: dict[str, object]) -> None:
        req = request.Request(
            self.endpoint,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            req.add_header("X-API-Key", self.api_key)
        try:
            with request.urlopen(req, timeout=self.timeout_seconds):
                return
        except Exception as exc:
            LOGGER.warning("monitoring backend post failed: %s", exc)

    def emit_metrics(self, metrics: dict[str, float | bool]) -> None:
        self._post({"type": "metrics", "payload": metrics})

    def emit_event(self, event: str, payload: dict[str, str | float | bool]) -> None:
        self._post({"type": "event", "event": event, "payload": payload})


class AsyncMonitoringBackend:
    def __init__(self, backend: MonitoringBackend, max_queue_size: int = 1_000):
        self._backend = backend
        self._queue: Queue[tuple[str, dict]] = Queue(maxsize=max_queue_size)
        self._stop_event = Event()
        self._worker = Thread(target=self._run, daemon=True)
        self._worker.start()

    def _enqueue(self, kind: str, payload: dict) -> None:
        try:
            self._queue.put_nowait((kind, payload))
        except Exception:
            LOGGER.debug("monitoring queue full, dropping %s payload", kind)

    def emit_metrics(self, metrics: dict[str, float | bool]) -> None:
        self._enqueue("metrics", {"metrics": metrics})

    def emit_event(self, event: str, payload: dict[str, str | float | bool]) -> None:
        self._enqueue("event", {"event": event, "payload": payload})

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                kind, payload = self._queue.get(timeout=0.2)
            except Empty:
                continue
            try:
                if kind == "metrics":
                    self._backend.emit_metrics(payload["metrics"])
                else:
                    self._backend.emit_event(payload["event"], payload["payload"])
            except Exception as exc:
                LOGGER.warning("async monitoring emit failed: %s", exc)
            finally:
                self._queue.task_done()

    def close(self, timeout_seconds: float = 1.0) -> None:
        self._stop_event.set()
        self._worker.join(timeout=timeout_seconds)


@dataclass(frozen=True)
class ServiceLevelObjectives:
    max_p95_latency_ms: float = 1_200.0
    max_p99_latency_ms: float = 2_000.0
    min_error_budget_remaining: float = 0.5
    max_failure_rate: float = 0.1


def evaluate_sre_readiness(
    metrics: dict[str, float | bool],
    slo: ServiceLevelObjectives | None = None,
) -> dict[str, object]:
    target = slo or ServiceLevelObjectives()
    p95 = float(metrics.get("p95_latency_ms", 0.0))
    p99 = float(metrics.get("p99_latency_ms", 0.0))
    failure_rate = float(metrics.get("failure_rate", 0.0))
    error_budget_remaining = float(metrics.get("error_budget_remaining", 1.0))
    alerts: list[str] = []
    if p95 > target.max_p95_latency_ms:
        alerts.append("latency.p95_exceeded")
    if p99 > target.max_p99_latency_ms:
        alerts.append("latency.p99_exceeded")
    if failure_rate > target.max_failure_rate:
        alerts.append("reliability.failure_rate_exceeded")
    if error_budget_remaining < target.min_error_budget_remaining:
        alerts.append("slo.error_budget_burn")
    return {
        "ready": not alerts,
        "alerts": tuple(alerts),
        "runbook": "docs/runbooks/chat_service_incident.md",
        "rollback_drill": "monthly",
    }


def monitoring_backend_from_config(config_backend: str, endpoint: str | None, api_key: str | None) -> MonitoringBackend:
    backend = config_backend.strip().lower()
    if backend == "http" and endpoint:
        return AsyncMonitoringBackend(HttpMonitoringBackend(endpoint=endpoint, api_key=api_key))
    if backend == "logging":
        return AsyncMonitoringBackend(LoggingMonitoringBackend())
    return AsyncMonitoringBackend(NullMonitoringBackend())
