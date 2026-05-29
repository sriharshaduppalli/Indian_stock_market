from __future__ import annotations

import json
import logging
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


def monitoring_backend_from_config(config_backend: str, endpoint: str | None, api_key: str | None) -> MonitoringBackend:
    backend = config_backend.strip().lower()
    if backend == "http" and endpoint:
        return HttpMonitoringBackend(endpoint=endpoint, api_key=api_key)
    if backend == "logging":
        return LoggingMonitoringBackend()
    return NullMonitoringBackend()
