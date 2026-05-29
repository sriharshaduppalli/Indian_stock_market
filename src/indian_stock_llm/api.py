from __future__ import annotations

from dataclasses import dataclass

from .acceptance import API_CONTRACT_VERSION
from .config import AssistantConfig
from .monitoring import monitoring_backend_from_config
from .query_engine import StockMarketAssistant
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


def build_chat_api(
    config: AssistantConfig,
    *,
    tenant_api_keys: dict[str, str] | None = None,
) -> ChatApi:
    assistant = StockMarketAssistant(config=config)
    monitoring = monitoring_backend_from_config(
        config_backend=config.monitoring_backend,
        endpoint=config.monitoring_endpoint,
        api_key=config.monitoring_api_key,
    )
    service = ChatService(
        assistant=assistant,
        tenant_api_keys=tenant_api_keys,
        monitoring_backend=monitoring,
    )
    return ChatApi(service)
