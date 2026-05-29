from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .acceptance import ProductionAcceptanceCriteria, SUPPORTED_QUERY_CATEGORIES
from .calculations import DeterministicCalculator
from .config import AssistantConfig, default_config
from .data_layer import EnterpriseDataLayer
from .knowledge_base import KnowledgeBase
from .learning_loop import ContinualLearningManager
from .safety import SafetyPolicy

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
    policy_reason: str = ""
    category: str = "stocks"


class StockMarketAssistant:
    """A minimal domain assistant scaffold for Indian stocks."""

    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or default_config()
        self.criteria = ProductionAcceptanceCriteria(
            max_latency_ms=self.config.max_latency_ms,
            min_uptime=self.config.min_uptime,
            max_cost_per_query=self.config.max_cost_per_query,
        )
        try:
            self.knowledge_base = KnowledgeBase.from_json(self.config.knowledge_base_path)
        except Exception as exc:
            raise ValueError(
                "Failed to load knowledge base. Check AssistantConfig.knowledge_base_path and JSON format."
            ) from exc
        self.data_layer = EnterpriseDataLayer(self.config)
        self.learning_manager = ContinualLearningManager(self.config.feedback_log_path)
        self.safety_policy = SafetyPolicy(self.config.policy_audit_log_path)

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
        try:
            if "cagr" in q and len(numbers) >= 3:
                start, end, years = numbers[0], numbers[1], numbers[2]
                cagr = DeterministicCalculator.cagr(start=start, end=end, years=years)
                return f"Deterministic calculation: CAGR is {cagr:.2f}% (from start={start}, end={end}, years={years})."
            if "return" in q and len(numbers) >= 2:
                buy, sell = numbers[0], numbers[1]
                absolute_return = DeterministicCalculator.absolute_return(buy=buy, sell=sell)
                return f"Deterministic calculation: Absolute return is {absolute_return:.2f}% (buy={buy}, sell={sell})."
        except ValueError:
            return None
        return None

    def _category_for_intent(self, intent: str) -> str:
        mapping = {
            "general_query": "stocks",
            "events_news": "nse_bse_sebi",
            "fundamentals": "analysis",
            "stock_analysis": "analysis",
            "market_calculations": "calculations",
            "prediction": "prediction_guidance",
            "portfolio": "analysis",
            "price_action": "analysis",
        }
        category = mapping.get(intent, "stocks")
        return category if category in SUPPORTED_QUERY_CATEGORIES else "stocks"

    def _policy_disclaimer(self, intent: str) -> str:
        if intent in {"prediction", "price_action"}:
            return (
                "This response is informational, not investment advice. "
                "Use risk controls and verify with live NSE/BSE data before trading."
            )
        return "This response is informational and should be validated against live market data."

    def ask(self, query: str) -> AssistantResponse:
        intent = self.classify_intent(query)
        category = self._category_for_intent(intent)
        policy_decision = self.safety_policy.evaluate(query)
        if not policy_decision.allowed:
            return AssistantResponse(
                intent=intent,
                category=category,
                answer=(
                    "I can’t help with that request. It may violate safe-use or compliance rules. "
                    "Please ask for a risk-aware market explanation instead."
                ),
                confidence=0.0,
                citations=(),
                disclaimer=self._policy_disclaimer(intent),
                safe_for_trading_advice=False,
                policy_reason=policy_decision.reason,
            )
        self.learning_manager.record_feedback(query=query, intent=intent)
        self.learning_manager.record_anonymized_feedback(query=query, intent=intent)
        resolved_entity = self.data_layer.resolve_entity(query)
        intent_tag_map = {"events_news": "sebi", "fundamentals": "fundamentals"}
        metadata_filters = {"tag": intent_tag_map[intent]} if intent in intent_tag_map else None
        context_items = self.knowledge_base.search(
            query,
            top_k=self.config.top_k_context,
            min_score=self.config.min_retrieval_score,
            metadata_filters=metadata_filters,
            intent=intent,
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
                f"Category: {category}.\n"
                f"Latency mode: {self.config.latency_mode}.\n"
                f"Data refresh timestamp: {self.data_layer.snapshot.refreshed_at}.\n"
                f"Data lineage verified: {self.data_layer.validate_snapshot()}.\n"
                f"Stale feeds: {self.data_layer.snapshot.stale_feeds or ('none',)}.\n"
                f"Partial feeds: {self.data_layer.snapshot.partial_feeds or ('none',)}.\n"
                f"Source hierarchy: {', '.join(self.data_layer.snapshot.source_hierarchy)}.\n"
                f"Resolved entity: {resolved_entity if resolved_entity else 'None'}.\n"
                f"{live_factor_note}"
                f"{deterministic_note}"
                f"Relevant market context:\n{context_text}\n\n"
                f"{self.learning_manager.daily_learning_summary()} "
                "Use this as a starting point and validate with live NSE/BSE data before decisions. "
                "This assistant does not provide guaranteed-return advice. "
                "Prediction guidance is probabilistic and uncertainty-aware."
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

        citations = self._extract_citations(context_items)
        if intent in {"fundamentals", "events_news", "market_calculations", "stock_analysis"} and not citations:
            answer = (
                "Insufficient grounding: I cannot provide factual/calculation output without citations. "
                "Please refresh enterprise sources."
            )
            confidence = min(confidence, 0.2)

        return AssistantResponse(
            intent=intent,
            answer=answer,
            confidence=confidence,
            citations=citations,
            disclaimer=self._policy_disclaimer(intent),
            safe_for_trading_advice=False,
            policy_reason=policy_decision.reason,
            category=category,
        )

    def query(self, query: str) -> dict[str, Any]:
        response = self.ask(query)
        feedback_metrics = self.learning_manager.feedback_metrics()
        safety_metrics = self.safety_policy.audit_summary()
        return {
            "intent": response.intent,
            "answer": response.answer,
            "confidence": response.confidence,
            "citations": list(response.citations),
            "disclaimer": response.disclaimer,
            "safe_for_trading_advice": response.safe_for_trading_advice,
            "policy_reason": response.policy_reason,
            "category": response.category,
            "acceptance": {
                "accuracy_min": self.criteria.accuracy_min,
                "max_latency_ms": self.criteria.max_latency_ms,
                "min_uptime": self.criteria.min_uptime,
                "max_cost_per_query": self.criteria.max_cost_per_query,
                "safety_compliance_min": self.criteria.safety_compliance_min,
            },
            "monitoring": {
                **feedback_metrics,
                **safety_metrics,
            },
            "data_integrity": {
                "stale_feeds": list(self.data_layer.snapshot.stale_feeds),
                "partial_feeds": list(self.data_layer.snapshot.partial_feeds),
                "connector_status": self.data_layer.snapshot.connector_status,
            },
        }
