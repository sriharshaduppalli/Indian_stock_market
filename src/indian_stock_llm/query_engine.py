from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .acceptance import ProductionAcceptanceCriteria, SUPPORTED_QUERY_CATEGORIES
from .calculations import DeterministicCalculator
from .config import AssistantConfig, default_config, runtime_config_from_env
from .data_layer import EnterpriseDataLayer
from .knowledge_base import (
    HeuristicReranker,
    HttpEmbeddingProvider,
    HttpReranker,
    InMemoryVectorIndex,
    KnowledgeBase,
    LocalHashEmbeddingProvider,
)
from .learning_loop import ContinualLearningManager
from .model_serving import HttpModelBackend, ModelOrchestrator, TemplateModelBackend
from .safety import SafetyPolicy

INTENT_KEYWORDS = {
    "price_action": {"price", "target", "entry", "exit"},
    "prediction": {"predict", "prediction", "forecast", "tomorrow", "next", "week"},
    "fundamentals": {"pe", "valuation", "fundamental", "fundamentals", "profit", "profits", "balance", "sheet"},
    "events_news": {"news", "event", "result", "results", "quarter", "guidance", "sebi", "regulation"},
    "portfolio": {"portfolio", "allocation", "risk", "diversification"},
    "stock_analysis": {"analyze", "analysis", "technical", "trend", "momentum"},
    "market_calculations": {"calculate", "calculation", "cagr", "return", "volatility", "beta"},
}
MAX_CONFIDENCE = 0.95
BASE_CONTEXT_CONFIDENCE = 0.65
CONFIDENCE_PER_CONTEXT_ITEM = 0.1
_SPACY_PIPELINE = None


def _regex_tokens(query: str) -> set[str]:
    return {m.group(0) for m in re.finditer(r"[a-z0-9]+", query.lower())}


def _nlp_tokens(query: str, backend: str) -> set[str]:
    selected = (backend or "auto").strip().lower()
    if selected not in {"auto", "spacy", "nltk", "basic"}:
        selected = "auto"
    if selected in {"auto", "spacy"}:
        global _SPACY_PIPELINE
        try:
            if _SPACY_PIPELINE is None:
                import spacy  # type: ignore

                _SPACY_PIPELINE = spacy.blank("en")
            doc = _SPACY_PIPELINE(query.lower())
            return {
                (token.lemma_ or token.text).lower()
                for token in doc
                if not token.is_space and not token.is_punct and token.text.strip()
            }
        except Exception:
            if selected == "spacy":
                return _regex_tokens(query)
    if selected in {"auto", "nltk"}:
        try:
            from nltk.tokenize import TweetTokenizer  # type: ignore

            tokenizer = TweetTokenizer(strip_handles=True, reduce_len=True)
            return {token.lower() for token in tokenizer.tokenize(query) if re.fullmatch(r"[a-z0-9]+", token.lower())}
        except Exception:
            return _regex_tokens(query)
    return _regex_tokens(query)


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
    """Production-hardened assistant scaffold for Indian stocks."""

    def __init__(self, config: AssistantConfig | None = None):
        self.config = config or runtime_config_from_env(default_config())
        self.criteria = ProductionAcceptanceCriteria(
            max_latency_ms=self.config.max_latency_ms,
            min_uptime=self.config.min_uptime,
            max_cost_per_query=self.config.max_cost_per_query,
            groundedness_min=self.config.groundedness_min,
        )
        embedding_provider = (
            HttpEmbeddingProvider(
                endpoint=self.config.embedding_endpoint,
                api_key=self.config.embedding_api_key,
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
            )
            if self.config.embedding_endpoint
            else LocalHashEmbeddingProvider()
        )
        reranker = (
            HttpReranker(
                endpoint=self.config.reranker_endpoint,
                api_key=self.config.reranker_api_key,
                provider=self.config.reranker_provider,
                model=self.config.reranker_model,
            )
            if self.config.reranker_endpoint
            else HeuristicReranker()
        )
        try:
            self.knowledge_base = KnowledgeBase.from_json(
                self.config.knowledge_base_path,
                embedding_provider=embedding_provider,
                vector_index=InMemoryVectorIndex(),
                reranker=reranker,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load knowledge base. Check AssistantConfig.knowledge_base_path and retrieval config."
            ) from exc
        self.data_layer = EnterpriseDataLayer(self.config)
        if self.config.background_refresh_enabled:
            self.data_layer.start_background_refresh(self.config.background_refresh_interval_seconds)
        self.learning_manager = ContinualLearningManager(self.config.feedback_log_path, async_logging=True)
        self.safety_policy = SafetyPolicy(self.config.policy_audit_log_path)
        primary_model = (
            HttpModelBackend(
                endpoint=self.config.model_endpoint,
                api_key=self.config.model_api_key,
                provider=self.config.model_provider,
                model=self.config.model_name,
                model_name=self.config.model_name or "remote-llm",
            )
            if self.config.model_endpoint
            else TemplateModelBackend()
        )
        self.model_orchestrator = ModelOrchestrator(
            primary=primary_model,
            fallback=TemplateModelBackend(),
            timeout_seconds=self.config.model_timeout_seconds,
        )

    def classify_intent(self, query: str) -> str:
        tokens = _regex_tokens(query) | _nlp_tokens(query, self.config.nlp_backend)
        scores: dict[str, int] = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            scores[intent] = len(tokens & keywords)
        best_intent, best_score = max(scores.items(), key=lambda pair: pair[1], default=("general_query", 0))
        if best_score == 0:
            return "general_query"
        top_scores = sorted(scores.values(), reverse=True)
        if len(top_scores) > 1 and top_scores[0] == top_scores[1]:
            if "calculate" in tokens or "cagr" in tokens or "return" in tokens:
                return "market_calculations"
            if "predict" in tokens or "forecast" in tokens:
                return "prediction"
            return "general_query"
        return best_intent

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
                return f"CAGR is {cagr:.2f}% (from start={start}, end={end}, years={years})."
            if "return" in q and len(numbers) >= 2:
                buy, sell = numbers[0], numbers[1]
                absolute_return = DeterministicCalculator.absolute_return(buy=buy, sell=sell)
                return f"Absolute return is {absolute_return:.2f}% (buy={buy}, sell={sell})."
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
        factual_intents = {"fundamentals", "events_news", "market_calculations", "stock_analysis"}
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
        readiness = self.data_layer.readiness_report()
        if self.config.require_ready_data_for_factual and intent in factual_intents and not readiness.ready:
            return AssistantResponse(
                intent=intent,
                category=category,
                answer=(
                    "Data readiness gate blocked this response for factual safety. "
                    f"Blockers: {', '.join(readiness.blockers) or 'unknown'}."
                ),
                confidence=0.0,
                citations=(),
                disclaimer=self._policy_disclaimer(intent),
                safe_for_trading_advice=False,
                policy_reason="Data readiness gate blocked factual response",
            )
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

        citations = self._extract_citations(context_items)
        deterministic_note = ""
        if intent == "market_calculations":
            deterministic = self._deterministic_calculation(query)
            deterministic_note = (
                f"Deterministic calculation: {deterministic}"
                if deterministic
                else (
                    "Deterministic calculation unavailable: provide valid positive numeric inputs "
                    "(for CAGR: start, end, years; for return: buy, sell)."
                )
            )

        if context_items:
            context_text = "\n".join(
                f"- {item.title}: {item.content} (source: {item.source})" for item in context_items
            )
            readiness_note = (
                f"refreshed_at={self.data_layer.snapshot.refreshed_at}; "
                f"stale={self.data_layer.snapshot.stale_feeds or ('none',)}; "
                f"partial={self.data_layer.snapshot.partial_feeds or ('none',)}; "
                f"entity={resolved_entity if resolved_entity else 'None'}"
            )
            prompt = self.model_orchestrator.compose_prompt(
                query=query,
                intent=intent,
                category=category,
                context_text=context_text,
                citations=citations,
                deterministic_note=deterministic_note,
                policy_disclaimer=self._policy_disclaimer(intent),
                readiness_note=readiness_note,
            )
            generated = self.model_orchestrator.generate(
                prompt,
                require_citations=intent in factual_intents,
                citations=citations,
            )
            prediction_note = (
                "Prediction factors considered: live news sentiment, sector momentum, corporate events, "
                "macro-rate signals, and liquidity conditions.\n"
                if intent == "prediction"
                else ""
            )
            answer = (
                f"Intent detected: {intent}.\n"
                f"Category: {category}.\n"
                f"Latency mode: {generated.latency_mode}.\n"
                f"Model backend: {generated.model_name}.\n"
                f"Data refresh timestamp: {self.data_layer.snapshot.refreshed_at}.\n"
                f"Data lineage verified: {self.data_layer.validate_snapshot()}.\n"
                f"Stale feeds: {self.data_layer.snapshot.stale_feeds or ('none',)}.\n"
                f"Partial feeds: {self.data_layer.snapshot.partial_feeds or ('none',)}.\n"
                f"Resolved entity: {resolved_entity if resolved_entity else 'None'}.\n"
                f"{prediction_note}"
                f"{deterministic_note + chr(10) if deterministic_note else ''}"
                f"Relevant market context:\n{generated.answer}\n\n"
                f"{self.learning_manager.daily_learning_summary()} "
                "Use this as a starting point and validate with live NSE/BSE data before decisions."
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

        if intent in factual_intents and not citations:
            answer = (
                "Insufficient grounding: I cannot provide factual/calculation output without citations. "
                "Please refresh enterprise sources."
            )
            confidence = min(confidence, 0.2)
        if confidence < self.config.min_confidence_threshold and (context_items or intent in factual_intents):
            answer = (
                f"Low-confidence response ({confidence:.2f}) withheld for safety. "
                "Please refine the question or refresh trusted data sources."
            )
            citations = ()

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
                "groundedness_min": self.criteria.groundedness_min,
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
                "background_refresh_errors": list(self.data_layer.background_refresh_errors()),
            },
            "contract": {
                "version": self.config.api_contract_version,
                "target_use_cases": ["grounded_qna", "risk_aware_guidance"],
                "prohibited_use_cases": ["trade_execution", "guaranteed_return_advice"],
            },
        }
