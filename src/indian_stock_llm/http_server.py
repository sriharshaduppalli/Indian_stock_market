from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
from typing import Callable

from .api import ApiRequest, ChatApi, build_chat_api
from .config import runtime_config_from_env

LOGGER = logging.getLogger(__name__)

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8080

STATUS_TO_HTTP = {
    "ok": HTTPStatus.OK,
    "bad_request": HTTPStatus.BAD_REQUEST,
    "unauthorized": HTTPStatus.UNAUTHORIZED,
    "rate_limited": HTTPStatus.TOO_MANY_REQUESTS,
    "degraded": HTTPStatus.SERVICE_UNAVAILABLE,
    "failed": HTTPStatus.INTERNAL_SERVER_ERROR,
}


@dataclass(frozen=True)
class HttpResponse:
    status_code: int
    payload: dict


class AuditLogger:
    def __init__(self, path: Path, retention_days: int = 30):
        self.path = path
        self.retention_days = retention_days
        self._last_prune_at: datetime | None = None

    def log(self, event: dict[str, object]) -> None:
        self._prune_old()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(event, ensure_ascii=False) + "\n")

    def _prune_old(self) -> None:
        now = datetime.now(timezone.utc)
        if not self.path.exists():
            self._last_prune_at = now
            return
        if self._last_prune_at and (now - self._last_prune_at) < timedelta(hours=1):
            return
        cutoff = now - timedelta(days=max(self.retention_days, 1))
        kept: list[str] = []
        for row in self.path.read_text(encoding="utf-8").splitlines():
            try:
                payload = json.loads(row)
            except json.JSONDecodeError:
                continue
            ts = _parse_timestamp(str(payload.get("timestamp", "")))
            if ts is None or ts >= cutoff:
                kept.append(json.dumps(payload, ensure_ascii=False))
        self.path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        self._last_prune_at = now


def _parse_timestamp(value: str) -> datetime | None:
    text = value.strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _decode_json_body(raw_body: bytes) -> dict:
    if not raw_body:
        return {}
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def dispatch_http_request(
    chat_api: ChatApi,
    *,
    method: str,
    path: str,
    headers: dict[str, str] | None = None,
    raw_body: bytes = b"",
    metrics_admin_token: str | None = None,
    audit_logger: AuditLogger | None = None,
) -> HttpResponse:
    request_headers = {key.lower(): value for key, value in (headers or {}).items()}
    now = datetime.now(timezone.utc).isoformat()
    normalized_path = path.rstrip("/") or "/"
    if method == "GET" and normalized_path == "/health":
        payload = chat_api.health()
        if audit_logger:
            audit_logger.log({"timestamp": now, "event": "health", "status": "ok"})
        return HttpResponse(status_code=HTTPStatus.OK, payload=payload)
    if method == "GET" and normalized_path == "/metrics":
        if metrics_admin_token and request_headers.get("x-admin-token") != metrics_admin_token:
            payload = {"status": "unauthorized", "error": "metrics endpoint requires admin token"}
            if audit_logger:
                audit_logger.log({"timestamp": now, "event": "metrics", "status": "unauthorized"})
            return HttpResponse(status_code=HTTPStatus.UNAUTHORIZED, payload=payload)
        payload = {"status": "ok", "metrics": chat_api.metrics()}
        if audit_logger:
            audit_logger.log({"timestamp": now, "event": "metrics", "status": "ok"})
        return HttpResponse(status_code=HTTPStatus.OK, payload=payload)
    if method == "POST" and normalized_path == "/query":
        body = _decode_json_body(raw_body)
        tenant_id = request_headers.get("x-tenant-id", str(body.get("tenant_id", "public")))
        api_key = request_headers.get("x-api-key") or body.get("api_key")
        query = str(body.get("query", ""))
        response = chat_api.query(ApiRequest(tenant_id=tenant_id, api_key=api_key, query=query))
        status = str(response.get("status", "failed"))
        status_code = int(STATUS_TO_HTTP.get(status, HTTPStatus.INTERNAL_SERVER_ERROR))
        if audit_logger:
            audit_logger.log(
                {
                    "timestamp": now,
                    "event": "query",
                    "status": status,
                    "tenant_id": tenant_id,
                }
            )
        return HttpResponse(status_code=status_code, payload=response)
    return HttpResponse(
        status_code=HTTPStatus.NOT_FOUND,
        payload={"status": "not_found", "error": "Route not found"},
    )


def make_handler(
    chat_api: ChatApi,
    *,
    metrics_admin_token: str | None = None,
    audit_logger: AuditLogger | None = None,
) -> type[BaseHTTPRequestHandler]:
    class ChatApiHttpHandler(BaseHTTPRequestHandler):
        api = chat_api
        admin_token = metrics_admin_token
        logger = audit_logger

        def _handle(self, method: str) -> None:
            length = int(self.headers.get("Content-Length", "0") or "0")
            raw_body = self.rfile.read(length) if length > 0 else b""
            response = dispatch_http_request(
                self.api,
                method=method,
                path=self.path,
                headers={k: v for k, v in self.headers.items()},
                raw_body=raw_body,
                metrics_admin_token=self.admin_token,
                audit_logger=self.logger,
            )
            body = json.dumps(response.payload, ensure_ascii=False).encode("utf-8")
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            self._handle("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._handle("POST")

        def log_message(self, format: str, *args: object) -> None:
            LOGGER.info("http_server %s", format % args)

    return ChatApiHttpHandler


def start_http_server(
    chat_api: ChatApi,
    *,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    metrics_admin_token: str | None = None,
    audit_logger: AuditLogger | None = None,
    server_factory: Callable[[tuple[str, int], type[BaseHTTPRequestHandler]], ThreadingHTTPServer] = ThreadingHTTPServer,
) -> ThreadingHTTPServer:
    server = server_factory((host, port), make_handler(chat_api, metrics_admin_token=metrics_admin_token, audit_logger=audit_logger))
    LOGGER.info("starting ChatApi HTTP server on %s:%s", host, port)
    server.serve_forever()
    return server


def audit_logger_from_env() -> AuditLogger | None:
    path_value = os.getenv("ISM_AUDIT_LOG_PATH", "").strip()
    if not path_value:
        return None
    retention_days = int(os.getenv("ISM_AUDIT_RETENTION_DAYS", "30") or "30")
    return AuditLogger(Path(path_value), retention_days=retention_days)


def main() -> None:
    config = runtime_config_from_env()
    api = build_chat_api(config=config)
    host = os.getenv("ISM_HTTP_HOST", DEFAULT_HOST).strip() or DEFAULT_HOST
    port = int(os.getenv("ISM_HTTP_PORT", str(DEFAULT_PORT)))
    metrics_admin_token = os.getenv("ISM_METRICS_ADMIN_TOKEN", "").strip() or None
    audit_logger = audit_logger_from_env()
    start_http_server(
        api,
        host=host,
        port=port,
        metrics_admin_token=metrics_admin_token,
        audit_logger=audit_logger,
    )


if __name__ == "__main__":
    main()
