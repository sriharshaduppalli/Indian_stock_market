from __future__ import annotations

import json
from pathlib import Path

from indian_stock_llm.http_server import AuditLogger, dispatch_http_request


class StubApi:
    def __init__(self) -> None:
        self.requests: list[dict[str, str | None]] = []

    def health(self) -> dict[str, str]:
        return {"status": "ok", "contract_version": "v1"}

    def metrics(self) -> dict[str, float]:
        return {"p95_latency_ms": 120.0}

    def query(self, request) -> dict:
        self.requests.append(
            {"tenant_id": request.tenant_id, "api_key": request.api_key, "query": request.query}
        )
        if not request.query.strip():
            return {"status": "bad_request", "response": {}}
        if request.api_key != "secret":
            return {"status": "unauthorized", "response": {}}
        if request.query == "slow":
            return {"status": "degraded", "response": {}}
        return {
            "status": "ok",
            "response": {"answer": "ok", "contract_version": "v1"},
            "cached": False,
        }


def test_health_endpoint() -> None:
    response = dispatch_http_request(StubApi(), method="GET", path="/health")
    assert response.status_code == 200
    assert response.payload["contract_version"] == "v1"


def test_metrics_requires_admin_token_when_configured() -> None:
    denied = dispatch_http_request(
        StubApi(),
        method="GET",
        path="/metrics",
        metrics_admin_token="admin-token",
    )
    allowed = dispatch_http_request(
        StubApi(),
        method="GET",
        path="/metrics",
        headers={"X-Admin-Token": "admin-token"},
        metrics_admin_token="admin-token",
    )
    assert denied.status_code == 401
    assert allowed.status_code == 200
    assert allowed.payload["status"] == "ok"


def test_query_uses_tenant_headers_and_maps_status_codes() -> None:
    api = StubApi()
    good = dispatch_http_request(
        api,
        method="POST",
        path="/query",
        headers={"X-Tenant-Id": "tenant-a", "X-API-Key": "secret"},
        raw_body=json.dumps({"query": "hello"}).encode("utf-8"),
    )
    bad_request = dispatch_http_request(
        api,
        method="POST",
        path="/query",
        headers={"X-Tenant-Id": "tenant-a", "X-API-Key": "secret"},
        raw_body=json.dumps({"query": "   "}).encode("utf-8"),
    )
    unauthorized = dispatch_http_request(
        api,
        method="POST",
        path="/query",
        headers={"X-Tenant-Id": "tenant-a", "X-API-Key": "wrong"},
        raw_body=json.dumps({"query": "hello"}).encode("utf-8"),
    )
    degraded = dispatch_http_request(
        api,
        method="POST",
        path="/query",
        headers={"X-Tenant-Id": "tenant-a", "X-API-Key": "secret"},
        raw_body=json.dumps({"query": "slow"}).encode("utf-8"),
    )
    assert good.status_code == 200
    assert bad_request.status_code == 400
    assert unauthorized.status_code == 401
    assert degraded.status_code == 503
    assert api.requests[0]["tenant_id"] == "tenant-a"


def test_audit_logger_retention_prunes_old_entries(tmp_path: Path) -> None:
    log_path = tmp_path / "audit.log"
    logger = AuditLogger(log_path, retention_days=1)
    log_path.write_text(
        "\n".join(
            [
                json.dumps({"timestamp": "2000-01-01T00:00:00Z", "event": "old", "status": "ok"}),
                json.dumps({"timestamp": "2099-01-01T00:00:00Z", "event": "new", "status": "ok"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    logger.log({"timestamp": "2099-01-02T00:00:00Z", "event": "latest", "status": "ok"})
    content = log_path.read_text(encoding="utf-8")
    assert "old" not in content
    assert "new" in content
    assert "latest" in content
