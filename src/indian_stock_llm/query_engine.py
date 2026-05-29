from __future__ import annotations

from dataclasses import dataclass

from .config import AssistantConfig, default_config
from .knowledge_base import KnowledgeBase


@dataclass(frozen=True)
class AssistantResponse:
    intent: str
    answer: str


class StockMarketAssistant:
    """A minimal domain assistant scaffold for Indian stocks."""

    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or default_config()
        self.knowledge_base = KnowledgeBase.from_json(self.config.knowledge_base_path)

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        if any(word in q for word in ["price", "target", "entry", "exit"]):
            return "price_action"
        if any(word in q for word in ["pe", "valuation", "fundamental", "balance sheet", "profit"]):
            return "fundamentals"
        if any(word in q for word in ["news", "event", "result", "quarter", "guidance"]):
            return "events_news"
        if any(word in q for word in ["portfolio", "allocation", "risk", "diversification"]):
            return "portfolio"
        return "general_query"

    def ask(self, query: str) -> AssistantResponse:
        intent = self.classify_intent(query)
        context_items = self.knowledge_base.search(query, top_k=self.config.top_k_context)

        if context_items:
            context_text = "\n".join(
                f"- {item.title}: {item.content} (source: {item.source})" for item in context_items
            )
            answer = (
                f"Intent detected: {intent}.\n"
                f"Relevant market context:\n{context_text}\n\n"
                "Use this as a starting point and validate with live NSE/BSE data before decisions. "
                "This assistant does not provide guaranteed-return advice."
            )
        else:
            answer = (
                f"Intent detected: {intent}. I could not find enough domain context in the local knowledge base. "
                "Please enrich data sources for better accuracy."
            )

        return AssistantResponse(intent=intent, answer=answer)
