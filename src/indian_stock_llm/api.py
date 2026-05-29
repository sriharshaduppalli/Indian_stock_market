from __future__ import annotations

from dataclasses import dataclass

from .acceptance import API_CONTRACT_VERSION
from .serving import ChatService


@dataclass(frozen=True)
class ApiRequest:
    tenant_id: str
    api_key: str | None
    query: str


class ChatApi:
    """Stable API contract wrapper for downstream chat-box integrations."""

    def __init__(self, service: ChatService):
        self.service = service

    def health(self) -> dict[str, str]:
        return {"status": "ok", "contract_version": API_CONTRACT_VERSION}

    def metrics(self) -> dict[str, float | bool]:
        return self.service.export_metrics()

    def query(self, request: ApiRequest) -> dict:
        result = self.service.query(request.query, tenant_id=request.tenant_id, api_key=request.api_key)
        response = result.get("response", {})
        response["contract_version"] = API_CONTRACT_VERSION
        return {"status": result.get("status", "failed"), "response": response, "cached": result.get("cached", False)}
