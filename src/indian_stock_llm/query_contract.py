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
    "bech": "sell",
    "becho": "sell",
    "bechna": "sell",
    "bechne": "sell",
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


def _symbol_candidates(query: str) -> list[str]:
    raw = (query or "").strip()
    if not raw:
        return []
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
    for match in _TICKER_RE.finditer(raw.upper()):
        token = match.group(1)
        if token not in _SKIP_TICKERS:
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
    if re.search(r"\b(how are you|how r u|kaise ho|kya haal)\b", q):
        return True
    if re.search(r"\b(thanks|thank you|thx|ty|bye|goodbye|see you)\b", q) and len(q.split()) <= 6:
        return True
    tokens = q.split()
    greet = {"hi", "hii", "hiii", "hello", "hey", "namaste", "namaskar", "yo"}
    extra = {"there", "team", "bot", "ai", "assistant", "bysel", "good", "morning", "afternoon", "evening"}
    return len(tokens) <= 5 and any(t in greet for t in tokens) and all(t in greet or t in extra for t in tokens)


def detect_profile(query: str, intent_hint: str = "") -> tuple[str, int, str]:
    """Return (profile, confidence 0-100, reason)."""
    q = normalize_query(query).strip().lower()
    intent_l = (intent_hint or "").strip().lower()
    if intent_l in GROQ_TO_ISM:
        intent_l = GROQ_TO_ISM[intent_l]

    if _is_small_talk(q):
        return "small_talk", 98, "greeting/chitchat"

    if any(pair in q for pair in _CONCEPT_PAIRS):
        return "compare_concepts", 95, "concept comparison"

    trade_ask = bool(
        re.search(
            r"\b(should i buy|should i sell|buy or sell|trade plan|swing trade|"
            r"entry zone|stoploss|stop[\s-]?loss|take[\s-]?profit|"
            r"good (time|entry) to (buy|sell)|entry and exit|best entry|"
            r"ideal entry|profit potential|risk vs reward|risk-to-reward|"
            r"risk/?reward for buying|kharid|bech)\b",
            q,
        )
    )
    has_symbolish = bool(_symbol_candidates(query))

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
    ):
        if not trade_ask:
            return "quote", 92, "live price"

    if re.search(r"\b(what are (the )?risks|key risks|risks in|risks for)\b", q) and not trade_ask:
        return "risks", 88, "risk ask"

    definitional = bool(
        re.search(r"\b(what is|what are|explain|define|definition|meaning(?: of)?|formula|equation)\b", q)
    )
    market_term = bool(
        re.search(
            r"\b(rsi|macd|pe|p/e|cagr|demat|circuit|ipo|option|future|nifty|sensex|"
            r"stock market|share market|valuation|roe|eps|sma|ema|atr|vwap|"
            r"bollinger|dividend|stt|ltcg|stcg|fii|dii)\b",
            q,
        )
    )
    if definitional and not has_symbolish and not trade_ask and market_term:
        return "literacy", 90, "definition"

    if re.search(
        r"\b(gtt|brokerage|asba|allotment|cnc|mis|nrml|pledge|demat|"
        r"stcg|ltcg|capital gains|what is (an? )?ipo|ipo process|"
        r"circuit|t\+1|how does the (stock|share) market work|"
        r"candlestick|marubozu|doji|how to start investing)\b",
        q,
    ):
        return "literacy", 88, "market literacy"

    if intent_l == "fundamentals" or (
        re.search(
            r"\b(p/?e|pe ratio|valuation|eps|roe|pb|p/b|dividend yield|"
            r"fundamentals?|overvalued|undervalued|fair value)\b",
            q,
        )
        and not trade_ask
    ):
        return "fundamentals", 88, "valuation"

    if intent_l == "prediction" or re.search(
        r"\b(predict|prediction|forecast|target for (next|this)|where will .{0,12} (go|be)|"
        r"will (it|this|the price)|will reach)\b",
        q,
    ):
        return "prediction", 90, "forecast"

    if intent_l == "events_news" or re.search(
        r"\b(news|headlines?|catalysts?|latest results?|earnings (news|update)|what.?s happening)\b",
        q,
    ):
        if re.search(r"\b(sentiment|mood|tone)\b", q):
            return "sentiment", 90, "news+mood"
        return "news", 90, "news"

    if re.search(
        r"\b(sentiment|market mood|bullish or bearish|how (is|are) (investors|traders) feeling)\b",
        q,
    ) and not trade_ask:
        return "sentiment", 90, "sentiment"

    if re.search(r"\b(futures?|options?|call option|put option|straddle|strangle|"
                 r"greeks|implied volatility|open interest|max pain|iron condor)\b", q):
        return "derivatives", 86, "f&o"

    if re.search(
        r"\b(technical analysis|chart analysis|price action|practice levels|"
        r"support and resistance|key (support|resistance|levels))\b",
        q,
    ) and not trade_ask:
        if definitional and not has_symbolish:
            return "literacy", 90, "indicator definition"
        return "technical", 88, "technicals"

    if re.search(r"\b(top|best|list)\b.{0,24}\b(defence|defense|pharma|bank|banking|it|auto|"
                 r"fmcg|energy|metal|realty|psu|railway)\b"
                 r"|\b(defence|defense|pharma|banking|fmcg)\s+stocks?\b"
                 r"|\bbest\s+it\s+stocks?\b", q):
        return "sector_screen", 88, "sector list"

    if re.search(r"\b(my (portfolio|holdings)|rebalance|allocation|diversify|sip plan)\b", q):
        return "portfolio", 84, "portfolio"

    if (re.search(r"\bcompare\b|\bvs\b|\bversus\b", q)
            and not re.search(r"\b(stcg|ltcg|cnc|mis|delivery vs|intraday vs)\b", q)):
        return "compare", 90, "compare names"

    if trade_ask or (
        intent_l == "price_action"
        and re.search(r"\b(buy|sell|entry|exit|target|stop|swing|trade plan|should i)\b", q)
        and not re.search(r"\b(price of|current price|live price|quote|ltp)\b", q)
    ):
        return "trade_plan", 90, "practice plan"

    if re.search(
        r"\b(technical analysis|chart analysis|price action|practice levels|"
        r"support and resistance|key (support|resistance|levels)|"
        r"rsi|macd|supertrend|moving averages?|overbought|oversold)\b",
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
    if not prior_symbol and screen_symbol:
        prior_symbol = screen_symbol

    q = original.lower().strip()
    current = _symbol_candidates(original)
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
    profile, confidence, _reason = detect_profile(resolved, intent_hint=intent_hint)
    effective_style = style or detect_style(resolved, conversation_history)
    candidates = _symbol_candidates(resolved)
    symbol = candidates[0] if candidates else None
    if not symbol and isinstance(screen_context, Mapping):
        screen_symbol = str(screen_context.get("symbol") or "").strip().upper()
        if screen_symbol:
            symbol = screen_symbol
            if not is_follow_up and profile in _PROFILES_NEEDING_SYMBOL and len(cleaned.split()) <= 8:
                resolved = f"{cleaned} {screen_symbol}".strip()
                is_follow_up = True
    if not symbol and prior_symbol and profile in _PROFILES_NEEDING_SYMBOL:
        symbol = prior_symbol
        if not is_follow_up:
            resolved = f"{cleaned} {prior_symbol}".strip()
            is_follow_up = True

    ambiguous: tuple[str, ...] = ()
    if len(candidates) > 1 and len(cleaned.split()) <= 4:
        ambiguous = tuple(candidates[:5])

    missing: list[str] = []
    needs_symbol = profile in _PROFILES_NEEDING_SYMBOL
    if needs_symbol and not symbol and not ambiguous:
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
    )


def groq_intent_from_profile(profile: str) -> str:
    return PROFILE_TO_GROQ.get(profile, "GENERAL")


def ism_intent_from_profile(profile: str) -> str:
    return PROFILE_TO_ISM.get(profile, "stock_analysis")
