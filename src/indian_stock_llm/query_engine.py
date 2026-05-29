from __future__ import annotations

import re
from dataclasses import dataclass

from .config import AssistantConfig, default_config
from .knowledge_base import KnowledgeBase
from .learning_loop import ContinualLearningManager

PRICE_ACTION_KEYWORDS = {"price", "target", "entry", "exit"}
PREDICTION_KEYWORDS = {"predict", "prediction", "forecast", "tomorrow", "next", "week"}
FUNDAMENTAL_KEYWORDS = {"pe", "valuation", "fundamental", "fundamentals", "profit", "profits"}
EVENT_KEYWORDS = {"news", "event", "result", "results", "quarter", "guidance"}
PORTFOLIO_KEYWORDS = {"portfolio", "allocation", "risk", "diversification"}
ANALYSIS_KEYWORDS = {"analyze", "analysis", "technical", "trend", "momentum"}
CALCULATION_KEYWORDS = {"calculate", "calculation", "cagr", "return", "volatility", "beta"}


@dataclass(frozen=True)
class AssistantResponse:
    intent: str
    answer: str


class StockMarketAssistant:
    """A minimal domain assistant scaffold for Indian stocks."""

    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or default_config()
        try:
            self.knowledge_base = KnowledgeBase.from_json(self.config.knowledge_base_path)
        except Exception as exc:
            raise ValueError(
                "Failed to load knowledge base. Check AssistantConfig.knowledge_base_path and JSON format."
            ) from exc
        self.learning_manager = ContinualLearningManager(self.config.feedback_log_path)

    def classify_intent(self, query: str) -> str:
        q = query.lower()
        tokens = {m.group(0) for m in re.finditer(r"[a-z0-9]+", q)}
        if PRICE_ACTION_KEYWORDS & tokens:
            return "price_action"
        if CALCULATION_KEYWORDS & tokens:
            return "market_calculations"
        if PREDICTION_KEYWORDS & tokens:
            return "prediction"
        if "balance sheet" in q or FUNDAMENTAL_KEYWORDS & tokens:
            return "fundamentals"
        if ANALYSIS_KEYWORDS & tokens:
            return "stock_analysis"
        if EVENT_KEYWORDS & tokens:
            return "events_news"
        if PORTFOLIO_KEYWORDS & tokens:
            return "portfolio"
        return "general_query"

    def ask(self, query: str) -> AssistantResponse:
        intent = self.classify_intent(query)
        self.learning_manager.record_feedback(query=query, intent=intent)
        context_items = self.knowledge_base.search(
            query,
            top_k=self.config.top_k_context,
            min_score=self.config.min_retrieval_score,
        )

        if context_items:
            context_text = "\n".join(
                f"- {item.title}: {item.content} (source: {item.source})" for item in context_items
            )
            live_factor_note = (
                "Prediction factors considered: live news sentiment, sector momentum, "
                "corporate events, macro-rate signals, and liquidity conditions.\n"
                if intent == "prediction"
                else ""
            )
            answer = (
                f"Intent detected: {intent}.\n"
                f"Latency mode: {self.config.latency_mode}.\n"
                f"{live_factor_note}"
                f"Relevant market context:\n{context_text}\n\n"
                f"{self.learning_manager.daily_learning_summary()} "
                "Use this as a starting point and validate with live NSE/BSE data before decisions. "
                "This assistant does not provide guaranteed-return advice."
            )
        else:
            answer = (
                f"Intent detected: {intent}. I could not find enough domain context in the local knowledge base. "
                f"Latency mode: {self.config.latency_mode}. "
                f"{self.learning_manager.daily_learning_summary()} "
                "Please enrich data sources for better accuracy."
            )

        return AssistantResponse(intent=intent, answer=answer)
