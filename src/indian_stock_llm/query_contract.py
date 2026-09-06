"""Single query contract for the Indian Stock LLM.

Groq, ISM, and /ai/ask used to classify the same sentence three different ways.
This module is the one brain: profile (answer shape), intent aliases, slots,
history rewrite, and clarifiers.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

_HINGLISH = {
    "kharid": "buy",
    "kharido": "buy",
    "kharidna": "buy",
    "kharidne": "buy",
    "kharidun": "buy",
    "kharidunga": "buy",
    "kharidunqa": "buy",
    "bech": "sell",
    "becho": "sell",
    "bechna": "sell",
    "bechne": "sell",
    "bechdu": "sell",
    "bechdun": "sell",
    "kitna": "how much",
    "kaise": "how",
    "kya": "what",
    "hai": "is",
}


def _language_tools():
    try:
        from .query_language import (
            detect_query_language,
            is_telugu_small_talk,
            normalize_user_query,
            telugu_clarifier,
            telugu_format_instruction,
        )

        return {
            "detect": detect_query_language,
            "normalize": normalize_user_query,
            "small_talk": is_telugu_small_talk,
            "clarifier": telugu_clarifier,
            "format": telugu_format_instruction,
        }
    except Exception:
        return None

PROFILE_TO_GROQ = {
    "quote": "QUOTE",
    "news": "NEWS",
    "sentiment": "SENTIMENT",
    "technical": "TECHNICAL",
    "trade_plan": "BUY_SELL",
    "prediction": "PREDICT",
    "fundamentals": "FUNDAMENTAL",
    "calculations": "CALCULATION",
    "literacy": "EDUCATIONAL",
    "compare": "COMPARE",
    "compare_concepts": "COMPARE_CONCEPTS",
    "sector_screen": "SECTOR_SCREEN",
    "portfolio": "PORTFOLIO",
    "derivatives": "DERIVATIVES",
    "small_talk": "SMALL_TALK",
    "risks": "NEWS",
    "stock_analysis": "GENERAL",
    "session": "EDUCATIONAL",
    "corporate_actions": "NEWS",
}

PROFILE_TO_ISM = {
    "quote": "price_action",
    "news": "events_news",
    "sentiment": "events_news",
    "technical": "stock_analysis",
    "trade_plan": "price_action",
    "prediction": "prediction",
    "fundamentals": "fundamentals",
    "calculations": "market_calculations",
    "literacy": "market_literacy",
    "compare": "compare",
    "compare_concepts": "market_literacy",
    "sector_screen": "sector_screen",
    "portfolio": "portfolio",
    "derivatives": "derivatives",
    "small_talk": "general_query",
    "risks": "events_news",
    "stock_analysis": "stock_analysis",
    "session": "market_literacy",
    "corporate_actions": "events_news",
}

GROQ_TO_ISM = {
    "NEWS": "events_news",
    "SENTIMENT": "events_news",
    "QUOTE": "price_action",
    "TECHNICAL": "stock_analysis",
    "BUY_SELL": "price_action",
    "PREDICT": "prediction",
    "FUNDAMENTAL": "fundamentals",
    "COMPARE": "compare",
    "CALCULATION": "market_calculations",
    "EDUCATIONAL": "market_literacy",
    "SECTOR_SCREEN": "sector_screen",
    "PORTFOLIO": "portfolio",
    "DERIVATIVES": "derivatives",
    "SMALL_TALK": "general_query",
    "COMPARE_CONCEPTS": "market_literacy",
}

_PROFILES_NEEDING_SYMBOL = frozenset(
    {
        "quote",
        "news",
        "sentiment",
        "technical",
        "trade_plan",
        "prediction",
        "fundamentals",
        "risks",
        "stock_analysis",
        "corporate_actions",
    }
)

_PROFILES_NO_INHERIT_SYMBOL = frozenset(
    {
        "literacy",
        "small_talk",
        "compare_concepts",
        "sector_screen",
        "calculations",
        "session",
    }
)

_CONCEPT_PAIRS = (
    "growth vs value",
    "safe vs aggressive",
    "dividend vs growth",
    "large-cap vs small-cap",
    "large cap vs small cap",
    "debt vs equity",
    "long term vs short term",
    "active vs passive",
    "mutual fund vs stock",
    "etf vs mutual fund",
    "sip vs lumpsum",
    "sip vs lump sum",
    "sip vs lump-sum",
    "lumpsum vs sip",
    "lump sum vs sip",
    "gold vs stocks",
    "stocks vs gold",
    "equity vs gold",
)

_TICKER_RE = re.compile(r"\b([A-Z][A-Z0-9.&-]{1,11})\b")
_SKIP_TICKERS = frozenset(
    {
        "A", "AN", "BY", "TO", "OF", "ON", "AS", "IS", "IN", "AT", "OR", "AND",
        "THE", "FOR", "RSI", "MACD", "SMA", "EMA", "ATR", "VWAP", "PE", "PB",
        "ROE", "ROCE", "EPS", "CAGR", "SIP", "NSE", "BSE", "SEBI", "FII", "DII",
        "BUY", "SELL", "HOLD", "WAIT", "TRIM", "NIFTY", "SENSEX", "BANKNIFTY",
        "WHAT", "HOW", "WHY", "WHEN", "WITH", "FROM", "THIS", "THAT", "HAVE",
        "WILL", "SHOULD", "PRICE", "NEWS", "STOCK", "SHARE", "MARKET",
        "PREDICT", "PREDICTION", "FORECAST", "TARGET", "ANALYZE", "ANALYSE",
        "ANALYSIS", "TECHNICAL", "FUNDAMENTAL", "SENTIMENT", "LATEST", "SAME",
        "ABOUT", "OVERVALUED", "UNDERVALUED", "COMPARE",
        "MEANING", "OUTLOOK", "TELL", "WEEK", "MUCH", "GOOD", "NOW",
        "DEFENCE", "DEFENSE", "BANKING", "PHARMA", "HELLO",
        "FOMO", "CAS", "TER", "DRHP", "NAV", "SGB", "ETF", "IST",
        "TEACH", "COVER", "PAPER", "HABIT", "HABITS", "LEARN",
        "BEGINNER", "BEGINNERS", "EDUCATIONAL", "EDUCATION",
        "ARE", "MUTUAL", "FUNDS", "FUND", "DO", "INVESTING",
        "INVESTOR", "INVESTORS", "STOCKS", "VS", "VERSUS",
        "OPTIONS", "FUTURES", "OPTION", "FUTURE", "LOT", "SIZE",
        "MARGIN", "EXPIRY", "ONLY", "NOT",
        "VIEW", "SHOW", "CHART", "LAST", "TODAY", "YEAR", "MONTH",
        "DAYS", "WEEKLY", "DAILY", "MORE", "ADD", "MAIN", "KYA", "KITNA",
        "BETTER", "WHICH", "NAMES", "SWING", "BULLISH", "BEARISH", "UPDATE",
        "CHEAP", "EXPENSIVE", "MOOD", "EXIT", "BOOK", "AVERAGE", "SHORT",
        "AUTO", "PSU", "REACH", "DOWNSIDE", "HAPPENING",
        "CMP", "RATE", "JUMP", "RESULTS", "DMA", "BREAKOUT", "VOLUME",
        "SPIKE", "PLEDGE", "BONUS", "GOLD", "LUMPSUM", "CONCENTRATION",
        "ANNOUNCEMENT", "CIRCUIT", "UPPER", "LOWER", "TRADERS",
        "KA", "HAAL", "DIP",
    }
)


@dataclass(frozen=True)
class QuerySlots:
    symbol: str | None = None
    peer_symbols: tuple[str, ...] = ()
    horizon: str | None = None
    style: str = "concise"
    follow_up: bool = False
    prior_symbol: str | None = None
    prior_profile: str | None = None
    ambiguous_names: tuple[str, ...] = ()
    missing: tuple[str, ...] = ()


@dataclass(frozen=True)
class QueryContract:
    query: str
    resolved_query: str
    profile: str
    ism_intent: str
    groq_intent: str
    confidence: int
    slots: QuerySlots = field(default_factory=QuerySlots)
    clarifier: str | None = None
    format_instructions: str = ""
    needs_symbol: bool = False
    language: str = "en"
    retrieval_query: str = ""
    graph_concepts: tuple[str, ...] = ()
    user_sentiment: dict[str, Any] = field(default_factory=dict)
    secondary_profile: str | None = None
    compound: bool = False
    sentence_count: int = 1
    answer_facet: str = ""
    answer_heading: str = ""
    transformer_profile: str = ""
    transformer_score: float = 0.0


def normalize_query(query: str) -> str:
    tools = _language_tools()
    if tools:
        return tools["normalize"](query)
    text = (query or "").strip()
    if text.lower().startswith("user_query:"):
        text = text.split(":", 1)[1].strip()
    if " | context:" in text:
        text = text.split(" | context:", 1)[0].strip()

    def _repl(match: re.Match[str]) -> str:
        token = match.group(0)
        mapped = _HINGLISH.get(token.lower())
        if not mapped:
            return token
        return f"{token} {mapped}"

    return re.sub(r"\b(" + "|".join(re.escape(k) for k in _HINGLISH) + r")\b", _repl, text, flags=re.I)


def _history_turns(history: Sequence[Mapping[str, Any]] | None) -> list[Mapping[str, Any]]:
    if not history:
        return []
    return [turn for turn in list(history) if isinstance(turn, Mapping) and str(turn.get("content") or "").strip()]


def should_inherit_symbol(profile: str, query: str, named_in_query: bool) -> bool:
    """Screen / last-stock context must not steal index, literacy, or greetings."""
    if (profile or "") in _PROFILES_NO_INHERIT_SYMBOL:
        return False
    if named_in_query:
        return True
    try:
        from .symbol_linker import is_index_or_market_ask

        if is_index_or_market_ask(query) and (profile or "") in {
            "prediction",
            "sentiment",
            "news",
            "stock_analysis",
            "fundamentals",
            "calculations",
            "derivatives",
        }:
            return False
    except Exception:
        pass
    if (profile or "") == "derivatives" and re.search(
        r"\b(what is|what are|meaning|explain|define)\b", (query or "").lower()
    ):
        return False
    if (profile or "") == "portfolio" and re.search(
        r"\b(my watchlist|on my (watch)?list|what('?s| is) on my list|"
        r"my (portfolio|holdings))\b",
        (query or "").lower(),
    ):
        return False
    return True


def _symbol_candidates(query: str) -> list[str]:
    raw = (query or "").strip()
    if not raw:
        return []
    try:
        from .symbol_linker import link_symbols

        linked = [s for s in link_symbols(raw) if s not in _SKIP_TICKERS]
        if linked:
            return list(dict.fromkeys(linked))
    except Exception:
        pass
    try:
        from app.stock_enricher import _company_name_matches, extract_symbol_from_query

        names = [hit[0] for hit in _company_name_matches(raw)]
        names = [n for n in names if str(n).upper() not in _SKIP_TICKERS]
        if names:
            return list(dict.fromkeys(names))
        one = extract_symbol_from_query(raw)
        if one and str(one).upper() not in _SKIP_TICKERS:
            return [str(one).upper()]
    except Exception:
        pass
    found: list[str] = []
    catalog_check = None
    try:
        from app.stock_enricher import _is_catalog_symbol

        catalog_check = _is_catalog_symbol
    except Exception:
        catalog_check = None
    for match in _TICKER_RE.finditer(raw.upper()):
        token = match.group(1)
        if token in _SKIP_TICKERS:
            continue
        if catalog_check is not None and not catalog_check(token):
            continue
        found.append(token)
    return list(dict.fromkeys(found))


def _prior_symbol(history: Sequence[Mapping[str, Any]] | None) -> str | None:
    for turn in reversed(_history_turns(history)[-8:]):
        content = str(turn.get("content") or "")
        cands = _symbol_candidates(content)
        if cands:
            return cands[0]
        bold = re.search(r"\*\*([A-Z][A-Z0-9.&-]{1,11})\*\*", content)
        if bold and bold.group(1) not in _SKIP_TICKERS:
            return bold.group(1)
    return None


def _prior_profile(history: Sequence[Mapping[str, Any]] | None) -> str | None:
    for turn in reversed(_history_turns(history)):
        if str(turn.get("role") or "").lower() != "user":
            continue
        profile, _, _ = detect_profile(str(turn.get("content") or ""), intent_hint="")
        if profile not in {"stock_analysis", "small_talk"}:
            return profile
    return None


def detect_horizon(query: str) -> str | None:
    q = (query or "").lower()
    if re.search(r"\b(intraday|day trade|today|scalp|this session)\b", q):
        return "intraday"
    if re.search(r"\b(swing|few days|1-5 days|this week|next week|weekly)\b", q):
        return "swing"
    if re.search(r"\b(1-?3 months?|quarter|medium[- ]term|this month)\b", q):
        return "medium"
    if re.search(r"\b(long[- ]term|years?|sip|invest|multi[- ]year)\b", q):
        return "long"
    return None


def detect_style(query: str, history: Sequence[Mapping[str, Any]] | None = None) -> str:
    q = (query or "").lower()
    if re.search(r"\b(quick|brief|short|tldr|one line|concise)\b", q):
        return "concise"
    if re.search(r"\b(detailed|in detail|deep dive|breakdown|step by step)\b", q):
        return "detailed"
    words = re.findall(r"\w+", q)
    turns = len(_history_turns(history))
    if len(words) <= 8 and turns < 2:
        return "concise"
    return "detailed"


def _is_small_talk(query: str) -> bool:
    tools = _language_tools()
    if tools and tools["small_talk"](query):
        return True
    q = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return False
    if re.search(r"\b(how are you|how r u|kaise ho|kaise how ho|kya haal)\b", q):
        if any(len(str(s)) >= 3 for s in (_symbol_candidates(query) or [])):
            return False
        return True
    if re.search(
        r"\b(thanks|thank you|thx|ty|bye|goodbye|see you|good night)\b", q
    ) and len(q.split()) <= 6:
        return True
    tokens = q.split()
    greet = {"hi", "hii", "hiii", "hello", "hey", "namaste", "namaskar", "yo", "gm"}
    extra = {"there", "team", "bot", "ai", "assistant", "bysel", "good", "morning", "afternoon", "evening"}
    if q in {"good morning", "good afternoon", "good evening", "gm"}:
        return True
    return len(tokens) <= 5 and any(t in greet or t == "good" for t in tokens) and all(t in greet or t in extra for t in tokens)


def detect_profile(query: str, intent_hint: str = "") -> tuple[str, int, str]:
    """Return (profile, confidence 0-100, reason)."""
    q = normalize_query(query).strip().lower()
    intent_l = (intent_hint or "").strip().lower()
    if intent_l in GROQ_TO_ISM:
        intent_l = GROQ_TO_ISM[intent_l]

    if _is_small_talk(query) or _is_small_talk(q):
        return "small_talk", 98, "greeting/chitchat"

    if re.search(
        r"\b(is (the )?market open|is (the )?market closed|"
        r"market (open|closed|hours|timings)|trading hours|"
        r"when does (the )?market (open|close)|session (today|now)|"
        r"nse (open|closed|hours))\b",
        q,
    ):
        return "session", 94, "session clock"

    if re.search(
        r"\b(position size|portfolio concentration|rebalance my|"
        r"my (portfolio|holdings|watchlist)|"
        r"on my (watch)?list|what('?s| is) on my list)\b",
        q,
    ):
        return "portfolio", 88, "portfolio sizing"

    trade_ask_early = bool(
        re.search(
            r"\b(should i buy|should i sell|can i buy|can i sell|buy or sell|"
            r"trade plan|swing trade|entry zone|stoploss|stop[\s-]?loss|"
            r"take[\s-]?profit|kharid|bech)\b",
            q,
        )
    )
    try:
        from .language_graph import build_language_graph, language_graph_enabled

        if language_graph_enabled():
            graph = build_language_graph(query)
            graph_has_listed = bool(graph.symbols)
            if (
                graph.profile_hint == "literacy"
                and graph.confidence >= 78
                and not graph_has_listed
            ):
                return "literacy", int(graph.confidence), graph.reason or "language graph"
            if (
                graph.profile_hint == "technical"
                and graph.confidence >= 78
                and graph_has_listed
                and not trade_ask_early
            ):
                return "technical", int(graph.confidence), graph.reason or "language graph"
    except Exception:
        pass

    if any(pair in q for pair in _CONCEPT_PAIRS):
        return "compare_concepts", 95, "concept comparison"

    trade_ask = bool(
        re.search(
            r"\b(should i buy|should i sell|can i buy|can i sell|buy or sell|"
            r"trade plan|swing trade|entry zone|stoploss|stop[\s-]?loss|"
            r"take[\s-]?profit|good (time|entry) to (buy|sell)|entry and exit|"
            r"best entry|ideal entry|profit potential|risk vs reward|"
            r"risk-to-reward|risk/?reward for buying|worth buying|"
            r"is it a (good )?buy|a good buy|good buy for|buy now|sell now|"
            r"should i wait|wait for a (dip|pullback)|accumulate (on )?dips|"
            r"hold or exit|should i hold|book profit|average down|add more|top up|"
            r"short [a-z0-9.&-]{2,15}|kharidun|kharidunga|bechdu|bechdun|"
            r"kya .{0,40}\b(buy|sell)|buy the dip|time to (buy|sell|exit)|"
            r"exit now|exit [a-z0-9.&-]{2,15}|\btrim\b|"
            r"accumulate .{0,24}(on )?dips|on every dip|add .{0,20}dip|"
            r"kharid|bech)\b",
            q,
        )
    )
    if re.search(
        r"\b(delivery vs|intraday vs|cnc vs|mis vs|delivery versus|"
        r"how does sip|what is sip|sip work|how to start sip|start sip)\b",
        q,
    ):
        return "literacy", 90, "retail mechanics"

    candidates = _symbol_candidates(query)
    has_symbolish = bool(candidates)
    if (
        has_symbolish
        and not trade_ask
        and re.search(r"\b(buy|sell)\b", q)
        and re.search(r"\b(kya|should|can|main|now|kharid|bech)\b", q)
    ):
        trade_ask = True

    if re.search(
        r"\b(full math|all indicators|quant(?:itative)? stack|indicator stack|"
        r"show (all )?math|p0 math|every indicator|calculate|compute)\b",
        q,
    ) or (
        intent_l == "market_calculations"
        and re.search(r"\b(cagr|return|formula)\b", q)
    ) or (
        re.search(r"\b(rsi|macd|sma|ema|atr|vwap|bollinger)\b", q)
        and re.search(r"\b(indicator|for prices?|price series)\b", q)
        and re.search(r"\d", q)
    ):
        if re.search(r"\b(what is|what are|explain|define|formula|equation|meaning)\b", q) and not has_symbolish:
            return "literacy", 90, "formula definition"
        return "calculations", 88, "calculation ask"

    if re.search(
        r"\b(quote|ltp|last traded|last price|current price|trading at|"
        r"spot price|live price|what(?:'s| is) the price|price of|"
        r"share price|stock price)\b",
        q,
    ) or re.search(r"\bprice\b.{0,16}\bhow much\b|\bhow much\b.{0,16}\bprice\b", q) or re.search(
        r"^[a-z0-9.&-]{2,15}\s+price\??$", q
    ) or (
        has_symbolish
        and re.search(r"\b(how much|kitna|ltp|cmp|ka price|ka rate)\b", q)
        and not trade_ask
        and not re.search(r"\b(buy|sell|target|stop|brokerage|stt|pledge)\b", q)
    ):
        if not trade_ask:
            return "quote", 92, "live price"

    if re.search(
        r"\b(what are (the )?risks|key risks|risks in|risks for|"
        r"downside|what can go wrong|what could go wrong)\b",
        q,
    ) and not trade_ask:
        return "risks", 88, "risk ask"

    definitional = bool(
        re.search(r"\b(what is|what are|explain|define|definition|meaning(?: of)?|formula|equation)\b", q)
    )
    market_term = bool(
        re.search(
            r"\b(rsi|macd|pe|p/e|cagr|demat|circuit|ipo|option|future|nifty|sensex|"
            r"stock market|share market|valuation|roe|eps|sma|ema|atr|vwap|"
            r"bollinger|dividend|stt|ltcg|stcg|fii|dii|"
            r"bonus|rights issue|stock split|lumpsum|lump sum|"
            r"straddle|strangle|covered call|brokerage|pledge)\b",
            q,
        )
    )
    if definitional and not has_symbolish and not trade_ask and market_term:
        return "literacy", 90, "definition"

    sectorish = bool(
        re.search(
            r"\b(top|best|list|which|some)\b.{0,24}\b(defence|defense|pharma|banks?|"
            r"banking|it|auto|fmcg|energy|metal|realty|psu|railway)\b",
            q,
        )
    )
    if (
        re.search(
            r"\b(gtt|brokerage|asba|allotment|cnc|mis|nrml|pledge|demat|"
            r"stcg|ltcg|capital gains|what is (an? )?ipo|ipo process|"
            r"how to apply (for )?(an? )?ipo|apply (for )?(an? )?ipo|"
            r"circuit|t\+1|how does the (stock|share) market work|"
            r"candlestick|marubozu|doji|how to start investing|"
            r"fii|dii|bonus issue|rights issue|stock split|"
            r"brokerage|paper trades?|paper trading|pledge)\b",
            q,
        )
        and not sectorish
        and (
            not has_symbolish
            or re.search(r"\bbrokerage\b", q)
            or (
                re.search(r"\bpledge\b", q)
                and re.search(r"\b(margin|cnc|how to|can i)\b", q)
                and not re.search(r"\bpromoter\b", q)
            )
        )
    ):
        return "literacy", 88, "market literacy"

    if has_symbolish and not trade_ask and (
        re.search(r"\b(view chart|show chart|price chart|chart of|the chart)\b", q)
        or (re.search(r"\bchart\b", q) and not definitional)
    ):
        return "technical", 90, "chart ask"

    if (
        has_symbolish
        and not trade_ask
        and re.search(r"\b(why (is|did|has)|what happened)\b", q)
        and re.search(
            r"\b(up|down|falling|fell|fall|rising|risen|crash(?:ed|ing)?|"
            r"drop(?:ped|ping)?|surge|rally|dump|moving|moved|jump(?:ed|ing)?|spike)\b",
            q,
        )
    ):
        return "news", 86, "move explanation"

    if intent_l == "fundamentals" or (
        re.search(
            r"\b(p/?e|pe ratio|valuation|eps|roe|pb|p/b|dividend yield|"
            r"fundamentals?|overvalued|undervalued|fair value|"
            r"book value|cheap|expensive|costly|promoter pledge|"
            r"market cap|promoter holding|debt of|beta of)\b",
            q,
        )
        and not trade_ask
    ):
        return "fundamentals", 88, "valuation"

    if has_symbolish and re.search(
        r"\b(ex[- ]?date|record date|dividend date|bonus date|"
        r"corporate actions?|rights date)\b",
        q,
    ):
        return "corporate_actions", 90, "dated event"

    if intent_l == "prediction" or re.search(
        r"\b(predict|prediction|forecast|target for (next|this)|"
        r"target (next|this|for)|where will .{0,16} (go|be)|"
        r"will (it|this|the price)|will .{0,16} reach|"
        r"(nifty|sensex|banknifty|bank nifty).{0,16}outlook|"
        r"outlook.{0,12}(nifty|sensex))\b",
        q,
    ):
        return "prediction", 90, "forecast"

    if intent_l == "events_news" or re.search(
        r"\b(news|headlines?|catalysts?|latest results?|earnings (news|update)|"
        r"what.?s happening|what is happening|any update|any news|"
        r"any announcement|q[1-4] results|quarterly results|"
        r"quarterly profit|guidance|block deal|insider|"
        r"fii buying|dii buying|rights issue|stock split|split meaning|"
        r"ex[- ]?dividend|dividend date|bonus|affecting|"
        r"upper circuit|lower circuit)\b",
        q,
    ):
        if re.search(r"\b(sentiment|mood|tone)\b", q):
            return "sentiment", 90, "news+mood"
        return "news", 90, "news"

    if re.search(
        r"\b(sentiment|market mood|mood on|bullish or bearish|"
        r"is (the )?market (bullish|bearish)|how (is|are) (investors|traders) feeling|"
        r"(investors|traders) (bullish|bearish)|bullish on|bearish on)\b",
        q,
    ) and not trade_ask:
        return "sentiment", 90, "sentiment"

    if definitional and re.search(
        r"\b(straddle|strangle|covered call|iron condor|put option|call option)\b", q
    ) and not has_symbolish:
        return "literacy", 90, "f&o definition"
    if re.search(
        r"\b(futures?|options?|call option|put option|straddle|strangle|"
        r"greeks|implied volatility|open interest|max pain|iron condor|"
        r"covered call|lot size|f&o ban|fo ban|"
        r"expir(y|ies)|weekly expiry|monthly expiry)\b",
        q,
    ):
        return "derivatives", 86, "f&o"

    if re.search(
        r"\b(technical analysis|chart analysis|price action|practice levels|"
        r"support and resistance|support[\s/-]?resistance|"
        r"key (support|resistance|levels)|"
        r"200 ?dma|50 ?dma|dma|breakout|volume spike|"
        r"gap up|gap down|hammer|death cross|golden cross|"
        r"\bema\b|\bvwap\b|relative strength)\b",
        q,
    ) and not trade_ask:
        if definitional and not has_symbolish:
            return "literacy", 90, "indicator definition"
        return "technical", 88, "technicals"

    if re.search(
        r"\b(top|best|list|which|some)\b.{0,24}\b(defence|defense|pharma|banks?|banking|it|auto|"
        r"fmcg|energy|metal|realty|psu|railway)\b"
        r"|\b(defence|defense|pharma|banking|fmcg|psu|auto)\s+(stocks?|names)\b"
        r"|\b(auto|psu|defence|defense|pharma|it)\s+names\b"
        r"|\bbest\s+it\s+stocks?\b",
        q,
    ):
        return "sector_screen", 88, "sector list"

    if re.search(
        r"\b(my (portfolio|holdings|watchlist)|rebalance|allocation|diversify|sip plan|"
        r"on my (watch)?list|what('?s| is) on my list|"
        r"portfolio concentration|concentration|position size|"
        r"80 percent|too much in)\b",
        q,
    ):
        return "portfolio", 84, "portfolio"

    vs_pair = re.search(
        r"\b([a-z][a-z0-9.&-]{1,11})\s+vs\s+([a-z][a-z0-9.&-]{1,11})\b", q
    )
    vs_two_names = False
    if vs_pair:
        left, right = vs_pair.group(1).upper(), vs_pair.group(2).upper()
        vs_two_names = left not in _SKIP_TICKERS and right not in _SKIP_TICKERS
    or_pair = re.search(
        r"\b(?:better\s+)?([a-z][a-z0-9.&-]{1,11})\s+or\s+([a-z][a-z0-9.&-]{1,11})\b",
        q,
    )
    or_two_names = False
    if or_pair:
        left, right = or_pair.group(1).upper(), or_pair.group(2).upper()
        or_two_names = left not in _SKIP_TICKERS and right not in _SKIP_TICKERS
    if (
        (
            re.search(r"\bcompare\b|\bversus\b", q)
            or vs_two_names
            or or_two_names
            or (re.search(r"\bvs\b", q) and len(candidates) >= 2)
        )
        and not re.search(r"\b(stcg|ltcg|cnc|mis|delivery vs|intraday vs|hold or exit|buy or sell)\b", q)
    ):
        return "compare", 90, "compare names"

    if (
        has_symbolish
        and not trade_ask
        and not definitional
        and re.search(
            r"\b(how(?:'s| is) .{0,28}(doing|looking|performing|trading)|"
            r"tell me about|long[- ]term view|\boutlook\b|"
            r"kya haal|kaise hai)|"
            r"\babout [a-z0-9.&-]{2,15}\b",
            q,
        )
    ):
        return "stock_analysis", 82, "status ask"

    if trade_ask or (
        intent_l == "price_action"
        and re.search(r"\b(buy|sell|entry|exit|target|stop|swing|trade plan|should i)\b", q)
        and not re.search(r"\b(price of|current price|live price|quote|ltp)\b", q)
    ):
        return "trade_plan", 90, "practice plan"

    if re.search(
        r"\b(technical analysis|chart analysis|price action|practice levels|"
        r"support and resistance|key (support|resistance|levels)|"
        r"rsi|macd|supertrend|moving averages?|overbought|oversold|"
        r"200 ?dma|50 ?dma|\bdma\b|breakout|volume spike|"
        r"gap up|gap down|hammer|death cross|golden cross|"
        r"\bema\b|\bvwap\b|relative strength|"
        r"support[\s/-]?resistance)\b",
        q,
    ) and not trade_ask:
        if definitional and not has_symbolish:
            return "literacy", 90, "indicator definition"
        return "technical", 88, "technicals"

    if intent_l == "prediction":
        return "prediction", 80, "intent hint"
    if intent_l == "events_news":
        return "news", 80, "intent hint"
    if intent_l == "fundamentals":
        return "fundamentals", 80, "intent hint"
    if intent_l in {"market_literacy"}:
        return "literacy", 80, "intent hint"
    if intent_l in {"stock_analysis", "overbought_check"}:
        return "technical" if "overbought" in q or "oversold" in q else "stock_analysis", 70, "intent hint"
    if intent_l == "price_action" and trade_ask:
        return "trade_plan", 80, "intent hint"
    if intent_l in {"stock_analysis", "general_query"}:
        return "stock_analysis", 62, "analyze / general"

    return "stock_analysis", 45, "fallback"


def rewrite_followup(
    query: str,
    history: Sequence[Mapping[str, Any]] | None = None,
    screen_context: Mapping[str, Any] | None = None,
) -> tuple[str, bool, str | None, str | None]:
    """Expand short follow-ups using last ticker / last profile / screen context.

    Returns (resolved_query, is_follow_up, prior_symbol, prior_profile).
    """
    original = (query or "").strip()
    prior_symbol = _prior_symbol(history)
    prior_profile = _prior_profile(history)
    screen_symbol = ""
    if isinstance(screen_context, Mapping):
        screen_symbol = str(screen_context.get("symbol") or "").strip().upper()
    q = original.lower().strip()
    followish = bool(
        re.search(
            r"^(what about|how about|and |also |same |yes\b|ok\b|why\?|simplify|"
            r"tell me more|what next)",
            q,
        )
    )
    if not prior_symbol and screen_symbol and followish:
        prior_symbol = screen_symbol
    current = _symbol_candidates(original)
    try:
        from .conversation import expand_conversational_followup

        expanded = expand_conversational_followup(
            original, prior_symbol=prior_symbol, prior_profile=prior_profile
        )
        if expanded:
            return expanded, True, prior_symbol, prior_profile
    except Exception:
        pass
    same_for = re.search(
        r"\b(same|do the same|apply that|repeat that)\s+(for|on|to)\s+([A-Za-z0-9.&-]{2,15})\b",
        q,
    )
    if same_for:
        other = same_for.group(3).upper()
        if other in _SKIP_TICKERS:
            other = current[0] if current else other
        shape = {
            "news": f"Latest news on {other}",
            "sentiment": f"{other} market sentiment",
            "quote": f"What is the price of {other}?",
            "technical": f"Technical analysis of {other}",
            "fundamentals": f"Is {other} overvalued?",
            "prediction": f"Predict {other} price",
            "trade_plan": f"Should I buy {other}?",
            "risks": f"What are risks in {other} now?",
        }.get(prior_profile or "", f"Analyze {other}")
        return shape, True, prior_symbol, prior_profile

    if current:
        return original, False, prior_symbol, prior_profile

    if not prior_symbol:
        return original, False, prior_symbol, prior_profile

    about = re.search(
        r"^(what about|how about|and|also)\s+(the\s+)?(sentiment|news|price|quote|rsi|"
        r"technicals?|fundamentals?|pe|valuation|risks?|forecast|prediction)\b",
        q,
    )
    if about:
        topic = about.group(3)
        mapping = {
            "sentiment": f"{prior_symbol} market sentiment",
            "news": f"Latest news on {prior_symbol}",
            "price": f"What is the price of {prior_symbol}?",
            "quote": f"What is the price of {prior_symbol}?",
            "rsi": f"RSI of {prior_symbol}",
            "technical": f"Technical analysis of {prior_symbol}",
            "technicals": f"Technical analysis of {prior_symbol}",
            "fundamental": f"Is {prior_symbol} overvalued?",
            "fundamentals": f"Is {prior_symbol} overvalued?",
            "pe": f"P/E of {prior_symbol}",
            "valuation": f"Is {prior_symbol} overvalued?",
            "risk": f"What are risks in {prior_symbol} now?",
            "risks": f"What are risks in {prior_symbol} now?",
            "forecast": f"Predict {prior_symbol} price",
            "prediction": f"Predict {prior_symbol} price",
        }
        return mapping.get(topic, f"{topic} of {prior_symbol}"), True, prior_symbol, prior_profile

    if re.search(r"\b(it|that stock|this stock|the same stock|its)\b", q) and len(q.split()) <= 14:
        rewritten = re.sub(
            r"\b(that stock|this stock|the same stock|it|its)\b",
            prior_symbol,
            original,
            flags=re.I,
            count=1,
        )
        return rewritten, True, prior_symbol, prior_profile

    if len(q.split()) <= 6 and re.search(
        r"\b(sentiment|news|price|rsi|technical|fundamental|overvalued|predict|forecast)\b",
        q,
    ):
        return f"{original} {prior_symbol}", True, prior_symbol, prior_profile

    return original, False, prior_symbol, prior_profile


def format_instructions_for(profile: str, style: str = "concise", language: str = "en") -> str:
    blocks = {
        "quote": (
            "--- LIVE QUOTE FOCUS ---\n"
            "Answer with a quote snapshot only: last, % change, OHLC/volume if present, "
            "nearby support/resistance. No RSI essay, news dump, or buy/sell call."
        ),
        "news": (
            "--- NEWS FOCUS ---\n"
            "4–6 headlines from context + one-line price. No trade plan, RSI, or BUY/SELL."
        ),
        "sentiment": (
            "--- SENTIMENT FOCUS ---\n"
            "Mood label + 2–4 drivers from context + one-line price. No dossier or buy/sell."
        ),
        "technical": (
            "--- TECHNICAL FOCUS ---\n"
            "RSI, MACD, MAs, support/resistance from context only. No valuation dump. "
            "End with a one-line BUY/SELL/HOLD only if they asked for a call; otherwise offer that follow-up."
        ),
        "trade_plan": (
            "--- BUY/SELL/HOLD ADVICE ---\n"
            "Give a clear investment call from context numbers: BUY / SELL / HOLD on the first line, "
            "then entry zone, stop, targets, risk/reward, and who it fits. "
            "Do not invent prices. Add a short 'not a SEBI RA; do your own research' line at the end."
        ),
        "prediction": (
            "--- PREDICTION + ADVICE ---\n"
            "Give a directional call (BUY/SELL/HOLD) plus if/then bands from ATR/support/resistance: "
            "support-holds vs support-breaks. Cap confidence. If a results/event window is "
            "mentioned, widen ranges. Never invent a target that is not in context."
        ),
        "fundamentals": (
            "--- FUNDAMENTALS FOCUS ---\n"
            "P/E, P/B, ROE, EPS, yield from context only. Missing fields = 'not available'. "
            "End with accumulate / hold / avoid for a long-term investor if the numbers support it."
        ),
        "risks": (
            "--- RISKS FOCUS ---\n"
            "3–6 concrete risks from context (valuation, leverage, event, tape). "
            "Say whether those risks justify HOLD/avoid vs still BUY with a smaller size."
        ),
        "calculations": (
            "--- CALCULATION FOCUS ---\n"
            "Show formula, plug the user's numbers, final result with units. "
            "If an input is missing, ask for it. Do not invent prices."
        ),
        "literacy": (
            "--- EDUCATION FOCUS ---\n"
            "Plain definition, formula if any, one NSE-style example, common mistake. "
            "No stock recommendation."
        ),
        "session": (
            "--- SESSION CLOCK ---\n"
            "Say whether the cash market is open now in IST, the 9:15 open, and today's close window. "
            "Do not attach a leftover stock. Do not invent a holiday."
        ),
        "corporate_actions": (
            "--- CORPORATE ACTIONS ---\n"
            "List only dated lines from context. If none, say so and tell the user to check NSE Get Quote. "
            "Never invent an ex-date or bonus ratio."
        ),
        "compare": (
            "--- COMPARE FOCUS ---\n"
            "Side-by-side metrics from context. Name a winner per metric and a preferred "
            "pick for swing vs long-term / SIP."
        ),
        "compare_concepts": (
            "--- CONCEPT COMPARE ---\n"
            "Define each idea, table of risk/horizon/effort, who it fits. No stock picks."
        ),
        "sector_screen": (
            "--- SECTOR SCREEN ---\n"
            "Use SCREENED RESULTS when present. List 3–5 names with one fact each and "
            "which style they fit (value / growth / swing). Do not invent unrelated Nifty names."
        ),
        "portfolio": (
            "--- PAPER BOOK FOCUS ---\n"
            "Concentration, diversification gaps, journaling next steps. Educational only."
        ),
        "derivatives": (
            "--- DERIVATIVES FOCUS ---\n"
            "Name the product, drivers (IV/OI/Greeks) from context, leverage/theta risk. "
            "Ask for missing expiry/strike. No guaranteed-premium claims."
        ),
        "small_talk": "--- SMALL TALK ---\nShort greeting. Do not invent a stock analysis.",
        "stock_analysis": (
            "--- FULL ANALYSIS ---\n"
            "Cover tape, levels, and valuation from context. Direct answer first, then "
            "BUY / SELL / HOLD with confidence. End with a research disclaimer."
        ),
    }
    text = blocks.get(profile, blocks["stock_analysis"])
    if style == "concise":
        text += "\nRESPONSE STYLE: concise — answer first, then 3–6 bullets."
    else:
        text += "\nRESPONSE STYLE: detailed — keep the same sections, add reasoning."
    if language in {"te", "te-en"}:
        tools = _language_tools()
        if tools:
            text += "\n" + tools["format"]()
    return text


def build_clarifier(
    profile: str,
    slots: QuerySlots,
    style: str = "concise",
    language: str = "en",
) -> str | None:
    tools = _language_tools() if language in {"te", "te-en"} else None
    if slots.ambiguous_names and len(slots.ambiguous_names) > 1:
        names = ", ".join(slots.ambiguous_names[:4])
        if tools:
            return tools["clarifier"]("ambiguous", names)
        return (
            f"That name matches more than one listed company ({names}). "
            "Which symbol do you want?"
        )
    if "symbol" in slots.missing:
        if tools:
            return tools["clarifier"]("symbol")
        if style == "concise":
            return (
                "Which NSE/BSE symbol? And do you want a quote, news, technicals, "
                "valuation, comparison, buy/sell call, or forecast?"
            )
        return (
            "I need a bit more to answer precisely:\n"
            "1. Stock symbol or company name\n"
            "2. What you want: quote, news, technicals, valuation, comparison, "
            "buy/sell advice, or forecast\n"
            "3. Horizon if relevant (intraday, swing, long-term)"
        )
    return None


def resolve_query_contract(
    query: str,
    *,
    intent_hint: str = "",
    conversation_history: Sequence[Mapping[str, Any]] | None = None,
    screen_context: Mapping[str, Any] | None = None,
    style: str | None = None,
) -> QueryContract:
    tools = _language_tools()
    language = tools["detect"](query) if tools else "en"
    cleaned = normalize_query(query)
    resolved, is_follow_up, prior_symbol, prior_profile = rewrite_followup(
        cleaned, conversation_history, screen_context
    )
    nlu = None
    try:
        from .sentence_analysis import analyze_user_query

        nlu = analyze_user_query(
            resolved, intent_hint=intent_hint, history=conversation_history
        )
    except Exception:
        nlu = None
    if nlu and nlu.compound and nlu.primary and nlu.primary.text:
        profile, confidence, _reason = detect_profile(
            nlu.primary.text, intent_hint=intent_hint
        )
    else:
        profile, confidence, _reason = detect_profile(resolved, intent_hint=intent_hint)
    xf = None
    try:
        from .query_transformer import analyze_query_transformer, apply_transformer_route

        xf = analyze_query_transformer(resolved)
        profile, confidence, _xf_used = apply_transformer_route(profile, confidence, xf)
    except Exception:
        xf = None
    if (
        is_follow_up
        and prior_profile
        and prior_profile not in {"small_talk"}
        and profile == "stock_analysis"
        and confidence < 70
        and not re.search(r"\b(analyze|analysis|about|outlook|how is)\b", resolved.lower())
    ):
        profile = prior_profile
        confidence = max(confidence, 78)
    effective_style = style or detect_style(resolved, conversation_history)
    named = _symbol_candidates(cleaned) or _symbol_candidates(resolved)
    inherit = should_inherit_symbol(profile, cleaned, bool(named))
    candidates = named
    symbol = candidates[0] if candidates else None
    if inherit and not symbol and isinstance(screen_context, Mapping):
        screen_symbol = str(screen_context.get("symbol") or "").strip().upper()
        if screen_symbol and screen_symbol not in _SKIP_TICKERS:
            symbol = screen_symbol
            if not is_follow_up and profile in _PROFILES_NEEDING_SYMBOL and len(cleaned.split()) <= 8:
                resolved = f"{cleaned} {screen_symbol}".strip()
                is_follow_up = True
    if inherit and not symbol and prior_symbol and profile in _PROFILES_NEEDING_SYMBOL:
        symbol = prior_symbol
        if not is_follow_up:
            resolved = f"{cleaned} {prior_symbol}".strip()
            is_follow_up = True
    if not inherit and symbol and symbol not in named:
        symbol = named[0] if named else None

    ambiguous: tuple[str, ...] = ()
    if len(candidates) > 1 and len(cleaned.split()) <= 4:
        ambiguous = tuple(candidates[:5])

    missing: list[str] = []
    needs_symbol = profile in _PROFILES_NEEDING_SYMBOL
    index_entity = bool(
        re.search(
            r"\b(nifty|sensex|bank\s*nifty|india\s*vix)\b|"
            r"[\u0c28][\u0c3f][\u0c2b][\u0c4d][\u0c1f][\u0c40]|"
            r"[\u0c38][\u0c46][\u0c28][\u0c4d][\u0c38][\u0c46][\u0c15][\u0c4d][\u0c38][\u0c4d]",
            resolved or cleaned,
            flags=re.IGNORECASE,
        )
    )
    low = f"{resolved} {cleaned}".lower()
    index_status = index_entity and bool(
        re.search(
            r"\b(how is|how much|ela undi|entha|dhara|status|trading at)\b",
            low,
        )
    ) and not re.search(
        r"\b(what is|what are|explain|define|pe ratio|valuation|outlook)\b",
        low,
    )
    if index_status and profile in {
        "stock_analysis",
        "fundamentals",
        "literacy",
        "quote",
        "sentiment",
        "technical",
        "news",
    }:
        profile = "quote"
        needs_symbol = True
    if (
        needs_symbol
        and not symbol
        and index_entity
        and profile in {"quote", "stock_analysis", "sentiment", "technical", "news"}
    ):
        if re.search(r"\bbank\s*nifty\b", low):
            symbol = "BANKNIFTY"
        elif "sensex" in low or "\u0c38\u0c46\u0c28\u0c4d\u0c38\u0c46\u0c15\u0c4d\u0c38\u0c4d" in (resolved or cleaned):
            symbol = "SENSEX"
        else:
            symbol = "NIFTY50"
    if needs_symbol and not symbol and not ambiguous and not (
        profile in {"fundamentals", "quote", "prediction", "stock_analysis", "sentiment", "technical", "news"}
        and index_entity
    ):
        missing.append("symbol")

    slots = QuerySlots(
        symbol=symbol,
        peer_symbols=tuple(candidates[1:3]),
        horizon=detect_horizon(resolved),
        style=effective_style,
        follow_up=is_follow_up,
        prior_symbol=prior_symbol,
        prior_profile=prior_profile,
        ambiguous_names=ambiguous,
        missing=tuple(missing),
    )
    clarifier = build_clarifier(profile, slots, effective_style, language=language)
    try:
        from .answer_shape import detect_answer_shape

        answer_facet, answer_heading = detect_answer_shape(resolved, profile)
    except Exception:
        answer_facet, answer_heading = "", ""
    retrieval_query = resolved
    graph_concepts: tuple[str, ...] = ()
    try:
        from .language_graph import build_language_graph, language_graph_enabled

        if language_graph_enabled():
            graph = build_language_graph(resolved)
            graph_concepts = graph.concepts
            if graph.retrieval_query:
                retrieval_query = graph.retrieval_query
    except Exception:
        pass
    return QueryContract(
        query=cleaned,
        resolved_query=resolved,
        profile=profile,
        ism_intent=PROFILE_TO_ISM.get(profile, "stock_analysis"),
        groq_intent=PROFILE_TO_GROQ.get(profile, "GENERAL"),
        confidence=int(confidence),
        slots=slots,
        clarifier=clarifier,
        format_instructions=format_instructions_for(profile, effective_style, language=language),
        needs_symbol=needs_symbol,
        language=language,
        retrieval_query=retrieval_query,
        graph_concepts=graph_concepts,
        user_sentiment=dict((nlu.user_sentiment if nlu else {}) or {}),
        secondary_profile=(nlu.secondary.profile if nlu and nlu.secondary else None),
        compound=bool(nlu.compound) if nlu else False,
        sentence_count=len(nlu.sentences) if nlu and nlu.sentences else 1,
        answer_facet=answer_facet,
        answer_heading=answer_heading,
        transformer_profile=str(getattr(xf, "profile", "") or ""),
        transformer_score=float(getattr(xf, "score", 0.0) or 0.0),
    )


def groq_intent_from_profile(profile: str) -> str:
    return PROFILE_TO_GROQ.get(profile, "GENERAL")


def ism_intent_from_profile(profile: str) -> str:
    return PROFILE_TO_ISM.get(profile, "stock_analysis")
