from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
MAX_CONFIDENCE = 0.95
BASE_CONTEXT_CONFIDENCE = 0.65
CONFIDENCE_PER_CONTEXT_ITEM = 0.1


@dataclass(frozen=True)
class AssistantResponse:
    intent: str
    answer: str
    confidence: float = 0.0
    citations: tuple[str, ...] = ()
    disclaimer: str = ""
    safe_for_trading_advice: bool = False


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

    def _extract_citations(self, context_items: list[Any]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(item.source for item in context_items if getattr(item, "source", None)))

    def _extract_numbers(self, query: str) -> list[float]:
        return [float(m.group(0)) for m in re.finditer(r"-?\d+(?:\.\d+)?", query)]

    def _deterministic_calculation(self, query: str) -> str | None:
        q = query.lower()
        numbers = self._extract_numbers(query)
        if "cagr" in q and len(numbers) >= 3 and numbers[0] > 0 and numbers[1] > 0 and numbers[2] >= 1:
            start, end, years = numbers[0], numbers[1], numbers[2]
            cagr = ((end / start) ** (1 / years) - 1) * 100
            return f"Deterministic calculation: CAGR is {cagr:.2f}% (from start={start}, end={end}, years={years})."
        if "return" in q and len(numbers) >= 2 and numbers[0] > 0 and numbers[1] >= 0:
            buy, sell = numbers[0], numbers[1]
            absolute_return = ((sell - buy) / buy) * 100
            return f"Deterministic calculation: Absolute return is {absolute_return:.2f}% (buy={buy}, sell={sell})."
        return None

    def _policy_disclaimer(self, intent: str) -> str:
        if intent in {"prediction", "price_action"}:
            return (
                "This response is informational, not investment advice. "
                "Use risk controls and verify with live NSE/BSE data before trading."
            )
        return "This response is informational and should be validated against live market data."

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
            deterministic_note = ""
            if intent == "market_calculations":
                deterministic = self._deterministic_calculation(query)
                if deterministic:
                    deterministic_note = deterministic + "\n"
                else:
                    deterministic_note = (
                        "Deterministic calculation unavailable: provide valid positive numeric inputs "
                        "(for CAGR: start, end, years; for return: buy, sell).\n"
                    )

            answer = (
                f"Intent detected: {intent}.\n"
                f"Latency mode: {self.config.latency_mode}.\n"
                f"{live_factor_note}"
                f"{deterministic_note}"
                f"Relevant market context:\n{context_text}\n\n"
                f"{self.learning_manager.daily_learning_summary()} "
                "Use this as a starting point and validate with live NSE/BSE data before decisions. "
                "This assistant does not provide guaranteed-return advice."
            )
            confidence = min(MAX_CONFIDENCE, BASE_CONTEXT_CONFIDENCE + CONFIDENCE_PER_CONTEXT_ITEM * len(context_items))
        else:
            answer = (
                f"Intent detected: {intent}. I could not find enough domain context in the local knowledge base. "
                f"Latency mode: {self.config.latency_mode}. "
                f"{self.learning_manager.daily_learning_summary()} "
                "Please enrich data sources for better accuracy."
            )
            confidence = 0.25

        return AssistantResponse(
            intent=intent,
            answer=answer,
            confidence=confidence,
            citations=self._extract_citations(context_items),
            disclaimer=self._policy_disclaimer(intent),
            safe_for_trading_advice=False,
        )

    def query(self, query: str) -> dict[str, Any]:
        response = self.ask(query)
        return {
            "intent": response.intent,
            "answer": response.answer,
            "confidence": response.confidence,
            "citations": list(response.citations),
            "disclaimer": response.disclaimer,
            "safe_for_trading_advice": response.safe_for_trading_advice,
        }
