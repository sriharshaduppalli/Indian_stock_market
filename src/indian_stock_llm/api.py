from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path

from .acceptance import API_CONTRACT_VERSION
from .config import AssistantConfig
from .monitoring import monitoring_backend_from_config
from .query_engine import StockMarketAssistant
from .serving import ChatService, FileStateBackend, TenantPolicy


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

    def refresh(self) -> None:
        """Trigger an immediate knowledge-base index refresh."""
        self.service.refresh()


def build_chat_api(
    config: AssistantConfig,
    *,
    tenant_api_keys: dict[str, str | list[str]] | None = None,
    tenant_policies: dict[str, TenantPolicy] | None = None,
) -> ChatApi:
    assistant = StockMarketAssistant(config=config)
    monitoring = monitoring_backend_from_config(
        config_backend=config.monitoring_backend,
        endpoint=config.monitoring_endpoint,
        api_key=config.monitoring_api_key,
    )
    service = ChatService(
        assistant=assistant,
        tenant_api_keys=tenant_api_keys or load_tenant_api_keys_from_env(),
        tenant_policies=tenant_policies,
        monitoring_backend=monitoring,
        state_backend=state_backend_from_env(),
    )
    return ChatApi(service)


def load_tenant_api_keys_from_env(env_var: str = "ISM_TENANT_API_KEYS_JSON") -> dict[str, str | list[str]]:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(payload, dict):
        return {}
    keys: dict[str, str | list[str]] = {}
    for tenant_id, value in payload.items():
        if isinstance(value, str) and value:
            keys[str(tenant_id)] = value
        elif isinstance(value, list):
            selected = [item for item in value if isinstance(item, str) and item]
            if selected:
                keys[str(tenant_id)] = selected
    return keys


def state_backend_from_env(env_var: str = "ISM_STATE_BACKEND_FILE") -> FileStateBackend | None:
    raw = os.getenv(env_var, "").strip()
    if not raw:
        return None
    return FileStateBackend(Path(raw))
