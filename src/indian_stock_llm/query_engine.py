from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from .acceptance import ProductionAcceptanceCriteria, SUPPORTED_QUERY_CATEGORIES
from .calculations import DeterministicCalculator, PandasTaIndicatorCalculator
from .config import AssistantConfig, default_config, runtime_config_from_env
from .data_layer import EnterpriseDataLayer
from .knowledge_base import (
    HeuristicReranker,
    HttpEmbeddingProvider,
    HttpReranker,
    InMemoryVectorIndex,
    KnowledgeBase,
    KnowledgeItem,
    LocalHashEmbeddingProvider,
    MLReranker,
    SentenceTransformerEmbeddingProvider,
)
from .builtin_knowledge import builtin_knowledge_items
from .learning_loop import ContinualLearningManager, DailyFeedbackAnalyzer, FeedbackLearningPipeline
from .model_serving import HttpModelBackend, ModelOrchestrator, TemplateModelBackend
from .prediction import PredictionEngine, PredictionSignals
from .safety import SafetyPolicy
import json
from pathlib import Path

INTENT_KEYWORDS = {
    "price_action": {
        "price", "target", "entry", "exit", "support", "resistance",
        "kharid", "bech", "buy", "sell", "swing", "intraday", "should",
    },
    "prediction": {"predict", "prediction", "forecast", "tomorrow", "next", "week"},
    "fundamentals": {
        # Avoid bare "profit" — it steals "take profit" trade-plan asks.
        "pe", "pb", "valuation", "fundamental", "fundamentals", "profitability",
        "earnings", "balance", "sheet", "roe", "roce", "eps", "dividend", "ltcg", "stcg", "stt",
    },
    "events_news": {
        "news", "event", "result", "results", "quarter", "guidance", "sebi", "regulation",
        "demat", "circuit", "settlement",
    },
    "market_literacy": {
        "beginner", "basics", "meaning", "participants", "depository", "nsdl", "cdsl",
        "kyc", "mistakes", "rumour", "rumor", "how", "works", "work", "guide",
        "primary", "secondary", "shareholder", "marketplace", "trader", "investor",
        "scalper", "swing", "holding", "cagr", "absolute", "varsity", "opinions",
        "candlestick", "marubozu", "doji", "hammer", "engulfing", "fibonacci",
        "technical", "volume", "cpr", "dow",
        "opening", "range", "cas", "chop", "fomo", "revenge", "weekend",
        "habit", "sip", "ter", "nav", "drhp", "sgb",
    },
    "portfolio": {
        "portfolio", "allocation", "risk", "diversification", "sip",
        "paper", "practice", "journal",
        "var", "kelly", "drawdown", "position", "sizing", "bias",
    },
    "derivatives": {
        "f&o", "fno", "futures", "options", "option", "call", "put",
        "iv", "greeks", "premium", "expiry", "straddle", "strangle",
        "lot", "margin", "basis", "oi", "pcr", "maxpain",
        "currency", "usdinr", "forex", "commodity", "commodities", "mcx",
        "ncdex", "gold", "silver", "crude", "gsec", "g-sec", "tbill",
    },
    "stock_analysis": {"analyze", "analysis", "technical", "trend", "momentum", "checklist"},
    "compare": {"compare", "vs", "versus", "against", "comparison"},
    "overbought_check": {
        "overbought", "oversold", "over", "bought", "sold", "extended", "stretched",
    },
    "sector_screen": {
        "sector", "sectors", "defence", "defense", "pharma", "banking", "fmcg",
        "realty", "railway", "psu", "metal", "cement",
    },
    # Keep literacy keyword surface for retail mechanics (phrase overrides also set this).
    # "top"/"best" alone must not force sector_screen — require sector phrase override.
    "market_calculations": {
        "calculate", "calculation", "cagr", "return", "volatility", "beta", "indicator",
        "indicators", "rsi", "sma", "ema", "macd", "bollinger", "formula", "equation",
        "atr", "vwap", "sharpe", "peg",
    },
}
MAX_CONFIDENCE = 0.95
BASE_CONTEXT_CONFIDENCE = 0.65
CONFIDENCE_PER_CONTEXT_ITEM = 0.1
_SPACY_PIPELINE = None

# Map common Hinglish / retail tokens to English cues before intent classify.
_HINGLISH_ALIASES = {
    "kharid": "buy",
    "kharido": "buy",
    "kharidna": "buy",
    "kharidne": "buy",
    "bech": "sell",
    "becho": "sell",
    "bechna": "sell",
    "bechne": "sell",
    "demat": "demat",
    "circuit": "circuit",
    "lot": "lot",
    "margin": "margin",
    "stt": "stt",
    "ltcg": "ltcg",
    "stcg": "stcg",
    "paper": "paper",
    "practice": "practice",
    "journal": "journal",
    "sip": "sip",
}


def _normalize_hinglish(query: str) -> str:
    """Lightweight Hinglish → English cue expansion for intent classification."""
    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mapped = _HINGLISH_ALIASES.get(token.lower())
        if not mapped:
            return token
        # Keep original token and add English cue so keyword sets still match both.
        if mapped.lower() == token.lower():
            return token
        return f"{token} {mapped}"

    return re.sub(r"[A-Za-z0-9]+", _repl, query)


def _load_learned_knowledge_items(path: Path | None) -> list[KnowledgeItem]:
    if path is None or not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    if not isinstance(data, list):
        return []
    items: list[KnowledgeItem] = []
    for row in data:
        if not isinstance(row, dict):
            continue
        try:
            items.append(
                KnowledgeItem(
                    id=str(row["id"]),
                    title=str(row.get("title", row["id"])),
                    content=str(row.get("content", "")),
                    tags=list(row.get("tags") or []),
                    source=str(row.get("source", "learned_feedback_v1")),
                )
            )
        except Exception:
            continue
    return items


def _yfinance_last_close(symbol: str) -> tuple[float, float] | None:
    """Return (last, pct_change) from yfinance SYMBOL.NS; failures stay silent."""
    try:
        import yfinance as yf  # type: ignore

        ticker = yf.Ticker(f"{symbol}.NS")
        hist = ticker.history(period="5d")
        if hist is None or getattr(hist, "empty", True):
            return None
        closes = hist["Close"]
        last = float(closes.iloc[-1])
        prev = float(closes.iloc[-2]) if len(closes) > 1 else last
        pct = ((last - prev) / prev * 100.0) if prev else 0.0
        return last, pct
    except Exception:
        return None


def _live_quote_knowledge_item(symbol: str) -> KnowledgeItem | None:
    """Build a live_quote_v1 KnowledgeItem for grounding; failures stay silent."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return None
    last: float | None = None
    pct: float | None = None
    try:
        from app.market_data import fetch_quote

        quote = fetch_quote(sym)
        if isinstance(quote, dict):
            raw_last = quote.get("last")
            raw_pct = quote.get("pctChange")
            if raw_last is not None:
                last = float(raw_last)
            if raw_pct is not None:
                pct = float(raw_pct)
    except Exception:
        pass
    if last is None:
        yf_pair = _yfinance_last_close(sym)
        if yf_pair is None:
            return None
        last, pct = yf_pair
    pct_s = f"{pct:+.2f}%" if pct is not None else "n/a"
    market_note = ""
    try:
        from nsepython import nse_marketStatus  # type: ignore

        status = nse_marketStatus()
        if isinstance(status, dict):
            state = status.get("marketState") or status.get("status") or status
            market_note = f" NSE market status snapshot: {state}."
        elif status:
            market_note = f" NSE market status snapshot: {status}."
    except Exception:
        market_note = ""
    content = (
        f"Live quote snapshot for {sym}: last={last:.2f}, pctChange={pct_s}.{market_note} "
        "Informational grounding only — not a trade recommendation or guaranteed return."
    )
    return KnowledgeItem(
        id=f"live_{sym}",
        title=f"Live quote {sym}",
        content=content,
        tags=["symbols", "stocks", "live_quote", sym.lower()],
        source="live_quote_v1",
    )


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
    prediction_signals: dict | None = None
    diagnostics: dict | None = None


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
        if self.config.embedding_endpoint:
            embedding_provider = HttpEmbeddingProvider(
                endpoint=self.config.embedding_endpoint,
                api_key=self.config.embedding_api_key,
                provider=self.config.embedding_provider,
                model=self.config.embedding_model,
            )
        elif self.config.embedding_local_model:
            embedding_provider = SentenceTransformerEmbeddingProvider(self.config.embedding_local_model)
        else:
            embedding_provider = LocalHashEmbeddingProvider()

        if self.config.reranker_endpoint:
            reranker = HttpReranker(
                endpoint=self.config.reranker_endpoint,
                api_key=self.config.reranker_api_key,
                provider=self.config.reranker_provider,
                model=self.config.reranker_model,
            )
        elif self.config.reranker_local_model_path:
            reranker = MLReranker(self.config.reranker_local_model_path)
        else:
            reranker = HeuristicReranker()

        try:
            extra = list(builtin_knowledge_items())
            extra.extend(_load_learned_knowledge_items(self.config.learned_knowledge_path))
            self.knowledge_base = KnowledgeBase.from_json(
                self.config.knowledge_base_path,
                embedding_provider=embedding_provider,
                vector_index=InMemoryVectorIndex(),
                reranker=reranker,
                extra_items=extra,
                embedding_cache_path=self.config.embedding_cache_path,
            )
        except Exception as exc:
            raise ValueError(
                "Failed to load knowledge base. Check AssistantConfig.knowledge_base_path and retrieval config."
            ) from exc
        self.data_layer = EnterpriseDataLayer(self.config)
        if self.config.background_refresh_enabled:
            self.data_layer.start_background_refresh(self.config.background_refresh_interval_seconds)
        self.learning_manager = ContinualLearningManager(self.config.feedback_log_path, async_logging=True)
        self.feedback_analyzer = DailyFeedbackAnalyzer(self.config.feedback_log_path)
        self.safety_policy = SafetyPolicy(self.config.policy_audit_log_path)
        self.prediction_engine = PredictionEngine()
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
        self._nightly_refresh_stop: threading.Event | None = None
        if self.config.nightly_refresh_enabled:
            self._start_nightly_refresh(self.config.nightly_refresh_hour_utc)

    def _start_nightly_refresh(self, hour_utc: int) -> None:
        """Launch a daemon thread that calls trigger_index_refresh once per day at *hour_utc* UTC."""
        stop_event = threading.Event()
        self._nightly_refresh_stop = stop_event

        def _loop() -> None:
            while not stop_event.is_set():
                now = datetime.now(timezone.utc)
                next_run = now.replace(hour=hour_utc, minute=0, second=0, microsecond=0)
                if next_run <= now:
                    next_run = next_run + timedelta(days=1)
                wait_secs = (next_run - now).total_seconds()
                if stop_event.wait(wait_secs):
                    break
                try:
                    self.trigger_index_refresh()
                except Exception:
                    pass

        t = threading.Thread(target=_loop, daemon=True)
        t.start()

    def trigger_index_refresh(self) -> None:
        """Promote feedback into learned KB, merge items, rebuild embeddings, save cache.

        Improves retrieval/knowledge over time — NOT full neural LLM fine-tuning.
        Also invoked by the nightly refresh daemon when ``nightly_refresh_enabled``.
        """
        if self.config.feedback_learning_enabled:
            try:
                FeedbackLearningPipeline.promote_from_feedback_log(
                    self.config.feedback_log_path,
                    self.config.learned_knowledge_path,
                    min_count=self.config.feedback_promote_min_count,
                )
            except Exception:
                pass

        combined: list[KnowledgeItem] = []
        seen: set[str] = set()
        # Prefer current in-memory items (builtin + prior learned), then re-merge learned file.
        for item in list(self.knowledge_base.items):
            if item.id not in seen and not str(item.id).startswith("live_"):
                combined.append(item)
                seen.add(item.id)
        for item in _load_learned_knowledge_items(self.config.learned_knowledge_path):
            if item.id not in seen:
                combined.append(item)
                seen.add(item.id)
        # Also keep sample JSON + builtin if somehow empty
        if not combined:
            combined = list(builtin_knowledge_items())

        self.knowledge_base.refresh_index(new_items=combined)
        if self.config.embedding_cache_path is not None:
            try:
                self.knowledge_base.save_embedding_cache(self.config.embedding_cache_path)
            except Exception:
                pass

    def classify_intent(self, query: str) -> str:
        try:
            from .query_contract import resolve_query_contract

            contract = resolve_query_contract(query)
            if int(contract.confidence or 0) >= 62 and contract.ism_intent:
                return contract.ism_intent
        except Exception:
            pass
        normalized = _normalize_hinglish(query)
        qlow = normalized.lower()
        # Phrase-level overrides beat sparse keyword ties.
        if re.search(r"\b(overbought|oversold|over.?bought|over.?sold)\b", qlow):
            return "overbought_check"
        # Retail mechanics / tax / IPO / demat / risk frameworks → literacy.
        if re.search(
            r"\b(gtt|brokerage|trading charges|asba|allotment|"
            r"cnc|mis|nrml|pledge|short delivery|auction market|"
            r"bonus issue|stock split|share split|rights issue|"
            r"corporate actions?|how to open (a )?demat|demat account|"
            r"stcg|ltcg|capital gains|tax on (equity|shares|profits?)|"
            r"ipo (process|allotment|apply)|what is (an? )?ipo|"
            r"fii/?\s*dii|investor protection|"
            r"3-5-7|3–5–7|357 rule|three five seven|risk management|"
            r"15-15-15|15–15–15|151515|1 crore sip|crorepati sip|"
            r"3-6-9|3–6–9|369 rule|three six nine|rule of money|emergency fund)\b",
            qlow,
        ):
            return "market_literacy"
        # Stock-tied follow-ups: keep news/quote/sentiment/TA off the full-analysis dump.
        named_for_ta = None
        try:
            from app.stock_enricher import extract_symbol_from_query

            named_for_ta = extract_symbol_from_query(normalized)
        except Exception:
            named_for_ta = None
        if named_for_ta:
            if re.search(
                r"\b(latest news|news on|headlines?|catalysts?|what.?s happening)\b",
                qlow,
            ) and not re.search(r"\b(sentiment|mood)\b", qlow):
                return "events_news"
            if re.search(
                r"\b(sentiment|market mood|bullish or bearish)\b",
                qlow,
            ):
                return "events_news"
            if re.search(r"\b(what are (the )?risks|key risks|risks in)\b", qlow):
                return "events_news"
            if re.search(
                r"\b(what(?:'s| is) the price|price of|current price|live price|share price|ltp)\b",
                qlow,
            ) or re.search(r"^[a-z0-9.&-]{2,15}\s+price\??$", qlow):
                return "price_action"
            if re.search(
                r"\b(technical analysis|chart analysis|price action|"
                r"practice levels|support and resistance)\b",
                qlow,
            ):
                return "stock_analysis"
        if re.search(
            r"\b(how does the (stock|share) market work|how the (stock|share) market works|"
            r"stock market meaning|key participants|depository participant|"
            r"how (are |do )?share prices|price discovery|how to start investing|"
            r"common mistakes|primary market|secondary market|what is (nsdl|cdsl|demat)|"
            r"what moves (the )?stock|why do stock prices|trader vs investor|day trader|"
            r"scalper|swing trader|holding period|absolute return|where do you fit|"
            r"how to calculate returns|after you (own|buy) (a )?stock|"
            r"technical analysis|candlestick|marubozu|doji|hammer|engulfing|"
            r"morning star|fibonacci|dow theory|central pivot|\bcpr\b)\b",
            qlow,
        ):
            return "market_literacy"
        # Compare only when explicit — bare "which" used to steal literacy/MF asks.
        if re.search(r"\bcompare\b|\bvs\b|\bversus\b", qlow) and not re.search(
            r"\b(stcg|ltcg|cnc|mis|delivery vs|intraday vs|futures vs|gtt)\b",
            qlow,
        ):
            return "compare"
        if re.search(
            r"\b(top|best|list)\b.{0,24}\b(defence|defense|pharma|bank|banking|it|auto|fmcg|energy|metal|realty|psu|railway)\b"
            r"|\b(defence|defense|pharma|banking|fmcg)\s+stocks?\b"
            r"|\bbest\s+it\s+stocks?\b|\bit\s+stocks?\s+under\b",
            qlow,
        ):
            return "sector_screen"
        if re.search(
            r"\b(should i buy|should i sell|buy or sell|swing trade|"
            r"entry zone|best entry|ideal entry|stoploss|stop[\s-]?loss|"
            r"take[\s-]?profit|target price|trade plan)\b",
            qlow,
        ):
            # Prefer price_action over generic sector tokens in the same sentence.
            if not re.search(r"\b(top|best)\s+\w+\s+stocks?\b", qlow):
                return "price_action"
        if getattr(self.config, "language_graph_enabled", True):
            try:
                from .language_graph import build_language_graph

                graph = build_language_graph(normalized)
                if graph.intent_hint and graph.confidence >= 78 and not named_for_ta:
                    return graph.intent_hint
            except Exception:
                pass
        tokens = _regex_tokens(normalized) | _nlp_tokens(normalized, self.config.nlp_backend)
        scores: dict[str, int] = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            scores[intent] = len(tokens & keywords)
        best_intent, best_score = max(scores.items(), key=lambda pair: pair[1], default=("general_query", 0))
        if best_score == 0:
            return "general_query"
        top_scores = sorted(scores.values(), reverse=True)
        if len(top_scores) > 1 and top_scores[0] == top_scores[1]:
            if (
                "calculate" in tokens
                or "cagr" in tokens
                or "return" in tokens
                or "rsi" in tokens
                or "sma" in tokens
                or "ema" in tokens
                or "macd" in tokens
                or "bollinger" in tokens
            ):
                return "market_calculations"
            if "predict" in tokens or "forecast" in tokens:
                return "prediction"
            if "compare" in tokens or "vs" in tokens:
                return "compare"
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
            if ("p/e" in q or "pe " in q or q.strip().startswith("pe") or " pe" in q) and "peg" not in q and len(numbers) >= 2:
                pe = DeterministicCalculator.pe(price=numbers[0], eps=numbers[1])
                return f"P/E is {pe:.2f} (price={numbers[0]}, EPS={numbers[1]})."
            if ("p/b" in q or "pb " in q or "price to book" in q) and len(numbers) >= 2:
                pb = DeterministicCalculator.pb(price=numbers[0], book_value=numbers[1])
                return f"P/B is {pb:.2f} (price={numbers[0]}, book={numbers[1]})."
            if "peg" in q and len(numbers) >= 2:
                peg = DeterministicCalculator.peg(pe=numbers[0], growth_pct=numbers[1])
                return f"PEG is {peg:.2f} (P/E={numbers[0]}, growth%={numbers[1]})."
            if "sharpe" in q and len(numbers) >= 2:
                sharpe = DeterministicCalculator.sharpe(
                    excess_return_pct=numbers[0], volatility_pct=numbers[1]
                )
                return f"Sharpe is {sharpe:.2f} (excess return%={numbers[0]}, volatility%={numbers[1]})."
            if "drawdown" in q and len(numbers) >= 2:
                dd = DeterministicCalculator.drawdown(peak=numbers[0], trough=numbers[1])
                return f"Drawdown is {dd:.2f}% (peak={numbers[0]}, trough={numbers[1]})."
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
            "derivatives": "derivatives",
            "market_literacy": "nse_bse_sebi",
            "price_action": "analysis",
            "compare": "analysis",
            "sector_screen": "stocks",
            "overbought_check": "analysis",
        }
        category = mapping.get(intent, "stocks")
        return category if category in SUPPORTED_QUERY_CATEGORIES else "stocks"

    def _policy_disclaimer(self, intent: str) -> str:
        if intent in {
            "prediction",
            "price_action",
            "compare",
            "sector_screen",
            "overbought_check",
            "derivatives",
        }:
            return (
                "This response is informational, not investment advice. "
                "Use risk controls and verify with live NSE/BSE data before trading."
            )
        return "This response is informational and should be validated against live market data."

    def ask(self, query: str, market_context: dict | None = None) -> AssistantResponse:
        learn_query = query
        try:
            from .query_language import learning_query_fields

            _lf = learning_query_fields(query)
            if _lf.get("normalized"):
                learn_query = str(_lf["normalized"])
        except Exception:
            learn_query = query
        intent = self.classify_intent(learn_query or query)
        # Bare/vague stock asks ("KAYNES", "about INFY") classify as general_query and
        # used to dump raw grounding bullets. Remap using the response profile so
        # news/quote/sentiment asks keep their shape instead of a full analysis dump.
        if (
            intent == "general_query"
            and isinstance(market_context, dict)
            and str(market_context.get("symbol") or "").strip()
        ):
            try:
                from .answer_composer import resolve_stock_response_profile

                profile = resolve_stock_response_profile(query, intent)
                intent = {
                    "news": "events_news",
                    "sentiment": "events_news",
                    "quote": "price_action",
                    "technical": "stock_analysis",
                    "risks": "events_news",
                    "trade_plan": "price_action",
                    "prediction": "prediction",
                    "fundamentals": "fundamentals",
                    "calculations": "market_calculations",
                    "stock_analysis": "stock_analysis",
                }.get(profile, "stock_analysis")
            except Exception:
                intent = "stock_analysis"
        category = self._category_for_intent(intent)
        # prediction_signals is only filled for prediction intent below; keep defined.
        prediction_signals = None
        factual_intents = {
            "fundamentals",
            "events_news",
            "market_calculations",
            "stock_analysis",
            "compare",
            "sector_screen",
            "overbought_check",
        }
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

        # Prefer deterministic education pack for definitions / equations.
        # Skip when the user asks for a live indicator on a named symbol.
        education = None
        try:
            from app.market_education import get_education_answer

            routed = learn_query or query
            wants_live_indicator = (
                PandasTaIndicatorCalculator.indicator_requested(routed)
                and bool(
                    PandasTaIndicatorCalculator._symbol_from_query(routed)
                    or (market_context or {}).get("symbol")
                )
            )
            has_of_symbol = bool(
                re.search(r"\b(of|for|on)\s+[a-z0-9][a-z0-9.&-]{1,15}\b", routed.lower())
            ) or bool((market_context or {}).get("symbol"))
            stock_specific_metric = bool(
                (market_context or {}).get("symbol")
                or PandasTaIndicatorCalculator._symbol_from_query(routed)
            ) and bool(
                re.search(
                    r"\b(p/?e|pe ratio|pb|p/b|roe|eps|valuation|rsi|macd|overbought|oversold)\b",
                    routed.lower(),
                )
            ) and (
                has_of_symbol
                or not re.search(
                    r"\b(formula|equation|define|definition|meaning of|what is|what are|explain)\b",
                    routed.lower(),
                )
            )
            wants_numeric_calc = bool(
                re.search(r"\b(calculate|compute|cagr|return from buy)\b", routed.lower())
                and re.search(r"\d", routed)
                and not re.search(
                    r"\b(what is|what are|explain|define|formula|equation|meaning)\b",
                    routed.lower(),
                )
            )
            live_profile = False
            try:
                from .query_contract import resolve_query_contract

                live_profile = resolve_query_contract(routed).profile in {
                    "quote",
                    "news",
                    "sentiment",
                    "technical",
                    "trade_plan",
                    "prediction",
                    "fundamentals",
                    "stock_analysis",
                    "risks",
                    "compare",
                }
            except Exception:
                live_profile = False
            if (
                not live_profile
                and not wants_live_indicator
                and not stock_specific_metric
                and not wants_numeric_calc
            ):
                education = get_education_answer(routed)
        except Exception:
            education = None
        if education:
            return AssistantResponse(
                intent=intent if intent != "general_query" else "market_literacy",
                category=category if intent != "general_query" else "analysis",
                answer=education,
                confidence=0.92,
                citations=("bysel_market_education",),
                disclaimer=self._policy_disclaimer("market_calculations"),
                safe_for_trading_advice=False,
                policy_reason=policy_decision.reason,
            )

        readiness = self.data_layer.readiness_report()
        # Local JSON enterprise feeds are often marked partial/stale — do not block
        # educational/analysis answers when the knowledge pack is available.
        if (
            self.config.require_ready_data_for_factual
            and intent in factual_intents
            and not readiness.ready
            and not self.knowledge_base.items
        ):
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
        resolved_entity = self.data_layer.resolve_entity(learn_query or query)
        # Prefer explicit symbol from live enrich context (BYSEL /ai/ask).
        if market_context and market_context.get("symbol"):
            resolved_entity = {
                **(resolved_entity or {}),
                "symbol": str(market_context.get("symbol")).upper(),
                "company_name": (resolved_entity or {}).get("company_name")
                or market_context.get("company_name")
                or "",
                "exchange": (resolved_entity or {}).get("exchange") or "NSE",
                "isin": (resolved_entity or {}).get("isin") or "n/a",
            }
        # Soft retrieval — do not hard-filter by tag (that emptied results before).
        search_query = learn_query or query
        if getattr(self.config, "language_graph_enabled", True):
            try:
                from .language_graph import graph_retrieval_query

                expanded = graph_retrieval_query(search_query)
                if expanded:
                    search_query = expanded
            except Exception:
                pass
        context_items = self.knowledge_base.search(
            search_query,
            top_k=self.config.top_k_context,
            min_score=self.config.min_retrieval_score,
            metadata_filters=None,
            intent=intent,
        )
        try:
            signals = getattr(self.learning_manager, "signals", None)
            if signals is not None:
                context_items = signals.rerank_items(search_query, context_items)
        except Exception:
            pass
        # Drop polluted learned/promoted notes that leak unrelated prior answers.
        context_items = [
            item
            for item in context_items
            if "grounded answer note" not in str(getattr(item, "title", "")).lower()
            and "educational retrieved note" not in str(getattr(item, "content", "")).lower()
            and "grounded answer note" not in str(getattr(item, "content", "")).lower()
        ]

        # Sector screens: inject curated liquid names (no paid LLM needed).
        if intent == "sector_screen":
            try:
                from app.stock_enricher import screen_stocks  # type: ignore

                sector = None
                for key in (
                    "defence", "defense", "pharma", "banking", "bank", "it", "auto",
                    "fmcg", "energy", "metal", "realty", "psu", "railway", "cement", "infra",
                ):
                    if re.search(r"\b" + re.escape(key) + r"\b", query.lower()):
                        sector = "DEFENCE" if key in ("defence", "defense") else (
                            "BANKING" if key == "bank" else key.upper()
                        )
                        break
                screened = screen_stocks({"sector": sector} if sector else {}) if sector else []
                if screened:
                    lines = []
                    for row in screened[:7]:
                        sym = row.get("symbol") or row.get("ticker") or "?"
                        pe = row.get("pe") or row.get("trailingPE") or "n/a"
                        lines.append(f"{sym}: P/E {pe}")
                    context_items = [
                        KnowledgeItem(
                            id=f"sector_screen_{sector or 'x'}",
                            title=f"Screened {sector or 'sector'} names",
                            content="Curated liquid names — " + "; ".join(lines),
                            tags=["sector", "screen", str(sector or "").lower()],
                            source="sector_screener_v1",
                        ),
                        *context_items,
                    ][: self.config.top_k_context + 2]
            except Exception:
                pass

        # Inject BYSEL live enrich context (technical / fundamental / levels).
        if market_context:
            bits = []
            if market_context.get("current_price") is not None:
                bits.append(f"price={market_context.get('current_price')}")
            tech = market_context.get("technical") or {}
            fund = market_context.get("fundamental") or {}
            levels = market_context.get("trading_levels") or {}
            if tech:
                bits.append(
                    "technical="
                    + ",".join(
                        f"{k}:{tech.get(k)}"
                        for k in (
                            "rsi", "trend", "ma_signal", "macd_hist", "macd_histogram",
                            "rsi_interpretation", "moving_averages", "bollinger_bands",
                        )
                        if tech.get(k) is not None
                    )
                )
            if fund:
                bits.append(
                    "fundamental="
                    + ",".join(
                        f"{k}:{fund.get(k)}"
                        for k in (
                            "pe", "pe_ratio", "trailingPE", "pb", "priceToBook",
                            "roe", "eps", "market_cap", "dividend_yield",
                        )
                        if fund.get(k) is not None
                    )
                )
            if levels:
                bits.append(
                    "levels="
                    + ",".join(
                        f"{k}:{levels.get(k)}"
                        for k in (
                            "support", "support_1", "support_2",
                            "resistance", "resistance_1", "resistance_2",
                            "stop_loss", "stop", "take_profit", "risk_reward",
                        )
                        if levels.get(k) is not None
                    )
                )
            # Always keep a live-context item when a symbol is in play so the
            # structured composer can answer even if KB retrieval is empty.
            content = " | ".join(bits) if bits else f"symbol={market_context.get('symbol')}"
            context_items = [
                KnowledgeItem(
                    id="live_market_context",
                    title=f"Live enrich {(market_context.get('symbol') or '')}".strip(),
                    content=content,
                    tags=["live", "enrich", "technical", "fundamental"],
                    source="bysel_enrich_v1",
                ),
                *context_items,
            ][: self.config.top_k_context + 2]
            p0 = market_context.get("p0_math") or {}
            if isinstance(p0, dict) and p0.get("ok"):
                try:
                    from .analysis_math import format_p0_for_prompt

                    context_items = [
                        KnowledgeItem(
                            id="p0_math_pack",
                            title=f"Quantitative + trade plan {market_context.get('symbol') or ''}".strip(),
                            content=format_p0_for_prompt(p0),
                            tags=["math", "rsi", "atr", "beta", "valuation", "pivots", "trade_plan", "buy_sell"],
                            source="bysel_quant_math_v2",
                        ),
                        *context_items,
                    ][: self.config.top_k_context + 3]
                except Exception:
                    pass

        # Inject resolved instrument facts into grounding context.
        if resolved_entity:
            entity_blurb = (
                f"Matched instrument: {resolved_entity.get('symbol')} — "
                f"{resolved_entity.get('company_name', '')} "
                f"(exchange={resolved_entity.get('exchange', 'NSE')}, "
                f"ISIN={resolved_entity.get('isin', 'n/a')})."
            )
            context_items = [
                KnowledgeItem(
                    id=f"entity_{resolved_entity.get('symbol', 'x')}",
                    title=f"Instrument {resolved_entity.get('symbol')}",
                    content=entity_blurb,
                    tags=["symbols", "stocks"],
                    source="instrument_master",
                ),
                *context_items,
            ][: self.config.top_k_context + 1]
            live_item = _live_quote_knowledge_item(str(resolved_entity.get("symbol") or ""))
            if live_item is not None:
                context_items = [live_item, *context_items][: self.config.top_k_context + 2]

        citations = self._extract_citations(context_items)
        deterministic_note = ""
        if intent == "market_calculations" or PandasTaIndicatorCalculator.indicator_requested(query):
            note_lines: list[str] = []
            if intent == "market_calculations":
                deterministic = self._deterministic_calculation(query)
                if deterministic:
                    note_lines.append(f"Deterministic calculation: {deterministic}")
                elif "cagr" in query.lower() or "return" in query.lower():
                    note_lines.append(
                        "Deterministic calculation unavailable: provide valid positive numeric inputs "
                        "(for CAGR: start, end, years; for return: buy, sell)."
                    )
            if PandasTaIndicatorCalculator.indicator_requested(query):
                indicator_note = PandasTaIndicatorCalculator.indicator_note(
                    query,
                    symbol_hint=str((resolved_entity or {}).get("symbol") or "") or None,
                )
                if indicator_note:
                    note_lines.append(f"Indicator calculation: {indicator_note}")
                else:
                    note_lines.append(
                        "Indicator calculation unavailable: name a symbol (e.g. RSI of RELIANCE) "
                        "or provide an explicit price series."
                    )
            deterministic_note = "\n".join(note_lines)

        if intent == "prediction" and isinstance(market_context, dict):
            try:
                pre_signals = self.prediction_engine.predict(
                    context_items=context_items,
                    deterministic_note=deterministic_note,
                    resolved_entity=resolved_entity,
                    trade_plan=market_context.get("trade_plan"),
                    p0_math=market_context.get("p0_math"),
                )
                market_context["prediction_signals"] = {
                    "intraday": {
                        "direction": pre_signals.intraday.direction,
                        "probability": pre_signals.intraday.probability,
                        "rationale": pre_signals.intraday.rationale,
                    },
                    "swing": {
                        "direction": pre_signals.swing.direction,
                        "probability": pre_signals.swing.probability,
                        "rationale": pre_signals.swing.rationale,
                    },
                    "medium_term": {
                        "direction": pre_signals.medium_term.direction,
                        "probability": pre_signals.medium_term.probability,
                        "rationale": pre_signals.medium_term.rationale,
                    },
                    "key_signals": list(pre_signals.key_signals),
                    "overall_confidence": pre_signals.overall_confidence,
                    "scenarios": dict(getattr(pre_signals, "scenarios", {}) or {}),
                    "event_blackout": getattr(pre_signals, "event_blackout", None),
                    "advice": getattr(pre_signals, "advice", "HOLD"),
                }
            except Exception:
                pass

        if context_items:
            context_text = "\n".join(
                f"- **{getattr(item, 'title', 'Insight')}**: {item.content}" for item in context_items
            )
            readiness_note = (
                f"refreshed_at={self.data_layer.snapshot.refreshed_at}; "
                f"stale={self.data_layer.snapshot.stale_feeds or ('none',)}; "
                f"partial={self.data_layer.snapshot.partial_feeds or ('none',)}; "
                f"entity={resolved_entity if resolved_entity else 'None'}; "
                f"kb_items={len(self.knowledge_base.items)}"
            )
            market_json = "none"
            if market_context:
                try:
                    from .answer_composer import normalize_market_context

                    normalized = normalize_market_context(market_context)
                    market_json = json.dumps(
                        {
                            "symbol": normalized.get("symbol"),
                            "current_price": normalized.get("current_price"),
                            "technical": normalized.get("technical") or {},
                            "fundamental": normalized.get("fundamental") or {},
                            "trading_levels": normalized.get("trading_levels") or {},
                            "company_name": market_context.get("company_name"),
                            "sector": market_context.get("sector"),
                            "peers": market_context.get("peers") or [],
                            "pre_signals": market_context.get("pre_signals") or {},
                            "sentiment": market_context.get("sentiment") or {},
                            "sentiment_pack": market_context.get("sentiment_pack")
                            or normalized.get("sentiment_pack")
                            or {},
                            "news_headlines": market_context.get("news_headlines")
                            or normalized.get("news_headlines")
                            or [],
                            "news_summary": market_context.get("news_summary")
                            or normalized.get("news_summary"),
                            "p0_math": market_context.get("p0_math") or {},
                            "prediction_signals": market_context.get("prediction_signals") or {},
                            "trade_plan": (
                                market_context.get("trade_plan")
                                or (market_context.get("p0_math") or {}).get("trade_plan")
                                or {}
                            ),
                            "user_sentiment": market_context.get("user_sentiment") or {},
                            "query_nlu": market_context.get("query_nlu") or {},
                            "conversation": market_context.get("conversation") or {},
                            "portfolio_context": market_context.get("portfolio_context") or {},
                        },
                        ensure_ascii=False,
                    )
                except Exception:
                    market_json = "none"
            prompt = self.model_orchestrator.compose_prompt(
                query=query,
                intent=intent,
                category=category,
                context_text=context_text,
                citations=citations,
                deterministic_note=deterministic_note,
                policy_disclaimer=self._policy_disclaimer(intent),
                readiness_note=readiness_note,
                market_context_json=market_json,
            )
            generated = self.model_orchestrator.generate(
                prompt,
                require_citations=False,
                citations=citations,
            )
            prediction_note = (
                "Prediction factors considered: live news sentiment, sector momentum, corporate events, "
                "macro-rate signals, and liquidity conditions.\n"
                if intent == "prediction"
                else ""
            )
            prediction_signals: dict | None = None
            if intent == "prediction" and isinstance(market_context, dict) and market_context.get("prediction_signals"):
                prediction_signals = dict(market_context.get("prediction_signals") or {})
            elif intent == "prediction":
                mc = market_context or {}
                signals = self.prediction_engine.predict(
                    context_items=context_items,
                    deterministic_note=deterministic_note,
                    resolved_entity=resolved_entity,
                    trade_plan=mc.get("trade_plan") if isinstance(mc, dict) else None,
                    p0_math=mc.get("p0_math") if isinstance(mc, dict) else None,
                )
                prediction_signals = {
                    "intraday": {
                        "direction": signals.intraday.direction,
                        "probability": signals.intraday.probability,
                        "rationale": signals.intraday.rationale,
                    },
                    "swing": {
                        "direction": signals.swing.direction,
                        "probability": signals.swing.probability,
                        "rationale": signals.swing.rationale,
                    },
                    "medium_term": {
                        "direction": signals.medium_term.direction,
                        "probability": signals.medium_term.probability,
                        "rationale": signals.medium_term.rationale,
                    },
                    "key_signals": list(signals.key_signals),
                    "overall_confidence": signals.overall_confidence,
                    "scenarios": dict(getattr(signals, "scenarios", {}) or {}),
                    "event_blackout": getattr(signals, "event_blackout", None),
                    "advice": getattr(signals, "advice", "HOLD"),
                }
            answer_parts = [generated.answer.strip()]
            if deterministic_note:
                answer_parts.append(deterministic_note)
            if prediction_signals:
                ps = prediction_signals
                answer_parts.append(
                    "**Multi-horizon outlook (estimate, not a guarantee):**\n"
                    f"• Call: {ps.get('advice') or 'HOLD'}\n"
                    f"• Intraday: {ps['intraday']['direction']} (~{ps['intraday']['probability']:.0%})\n"
                    f"• Swing: {ps['swing']['direction']} (~{ps['swing']['probability']:.0%})\n"
                    f"• Medium-term: {ps['medium_term']['direction']} (~{ps['medium_term']['probability']:.0%})\n"
                    f"• Key cues: {'; '.join(ps.get('key_signals') or []) or 'n/a'}"
                )
            elif prediction_note.strip():
                answer_parts.append(prediction_note.strip())
            answer_parts.append(
                "⚠️ Disclaimer: Educational / informational only — validate with live NSE/BSE data before decisions."
            )
            answer = "\n\n".join(part for part in answer_parts if part)

            diagnostics = {
                "intent": intent,
                "category": category,
                "latency_mode": generated.latency_mode,
                "model_backend": generated.model_name,
                "data_refresh_timestamp": str(self.data_layer.snapshot.refreshed_at),
                "data_lineage_verified": self.data_layer.validate_snapshot(),
                "stale_feeds": list(self.data_layer.snapshot.stale_feeds) if self.data_layer.snapshot.stale_feeds else [],
                "partial_feeds": list(self.data_layer.snapshot.partial_feeds) if self.data_layer.snapshot.partial_feeds else [],
                "resolved_entity": resolved_entity if resolved_entity else "None",
                "prediction_note": prediction_note.strip() if prediction_note else None,
                "deterministic_note": deterministic_note if deterministic_note else None,
                "kb_size": len(self.knowledge_base.items),
            }

            confidence = min(MAX_CONFIDENCE, BASE_CONTEXT_CONFIDENCE + CONFIDENCE_PER_CONTEXT_ITEM * len(context_items))
        else:
            answer = (
                "I could not find enough domain context in the local Indian-market knowledge pack. "
                "Try asking about an indicator (RSI, MACD, P/E), a sector (banking, IT, pharma), "
                "or a specific NSE symbol."
            )
            confidence = 0.25
            prediction_signals = None
            diagnostics = {
                "intent": intent,
                "category": category,
                "latency_mode": self.config.latency_mode,
                "kb_size": len(self.knowledge_base.items),
                "model_backend": "fallback_no_context",
                "resolved_entity": resolved_entity if resolved_entity else "None",
                "error_reason": "insufficient_domain_context",
            }

        # Soft grounding — don't zero out confident KB answers just for missing citation tags.
        if intent in factual_intents and not citations and not resolved_entity and confidence < 0.4:
            answer = (
                "I need a clearer stock symbol or topic (for example RSI, P/E, banking sector, or RELIANCE) "
                "to give a grounded Indian-market answer."
            )
            confidence = min(confidence, 0.3)
            if diagnostics:
                diagnostics["error_reason"] = "insufficient_citations_for_factual_intent"

        if confidence < self.config.min_confidence_threshold and not context_items and intent in factual_intents:
            answer = (
                "I need a listed NSE/BSE symbol or a clearer topic "
                "(for example RSI, P/E, banking sector, or RELIANCE) to analyse this properly."
            )
            citations = ()
            if diagnostics:
                diagnostics["error_reason"] = "low_confidence_safety_threshold"

        grounded = bool(citations) and confidence >= 0.55
        try:
            self.learning_manager.record_interaction(
                query=query,
                intent=intent,
                confidence=confidence,
                citation_count=len(citations),
                grounded=grounded,
            )
        except Exception:
            pass
        if (
            self.config.feedback_learning_enabled
            and grounded
            and self.config.learned_knowledge_path is not None
        ):
            try:
                FeedbackLearningPipeline.promote_grounded_answer(
                    self.config.learned_knowledge_path,
                    query=query,
                    intent=intent,
                    answer=answer,
                    citations=citations,
                    confidence=confidence,
                )
            except Exception:
                pass

        return AssistantResponse(
            intent=intent,
            answer=answer,
            confidence=confidence,
            citations=citations,
            disclaimer=self._policy_disclaimer(intent),
            safe_for_trading_advice=False,
            policy_reason=policy_decision.reason,
            category=category,
            prediction_signals=prediction_signals,
            diagnostics=diagnostics,
        )

    def query(self, query: str, market_context: dict | None = None) -> dict[str, Any]:
        response = self.ask(query, market_context=market_context)
        feedback_metrics = self.learning_manager.feedback_metrics()
        safety_metrics = self.safety_policy.audit_summary()
        answer = response.answer
        try:
            from .query_language import localize_assistant_answer

            answer = localize_assistant_answer(query, answer)
        except Exception:
            answer = response.answer
        return {
            "intent": response.intent,
            "answer": answer,
            "confidence": response.confidence,
            "citations": list(response.citations),
            "disclaimer": response.disclaimer,
            "safe_for_trading_advice": response.safe_for_trading_advice,
            "policy_reason": response.policy_reason,
            "category": response.category,
            "prediction_signals": response.prediction_signals,
            "diagnostics": response.diagnostics,
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
