"""Conversational turns for the Indian Stock LLM.

Users talk like a desk chat: “why?”, “and TCS?”, “what next?”, “ok thanks”.
ISM already has live cards. This module keeps the thread going so those
short turns stay useful for paper practice.
"""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

_SHAPE = {
    "news": "Latest news on {sym}",
    "sentiment": "{sym} market sentiment",
    "quote": "What is the price of {sym}?",
    "technical": "Technical analysis of {sym}",
    "fundamentals": "Is {sym} overvalued?",
    "prediction": "Predict {sym} price",
    "trade_plan": "Should I buy {sym}?",
    "risks": "What are risks in {sym} now?",
    "stock_analysis": "Analyze {sym}",
}

_SMALL_TALK = {
    "greeting": (
        "Hi — I can check a live price, read the tape, or walk a paper buy/sell plan "
        "on any NSE name. What are you looking at?"
    ),
    "thanks": (
        "Glad that helped. We can stay on the same name for news, levels, "
        "or a paper plan whenever you want."
    ),
    "bye": (
        "See you next session. If you paper-trade, jot the entry, stop, and why you took it."
    ),
    "how_are_you": (
        "Ready when you are. Ask a stock, a sector, or “should I buy …?” for a paper call."
    ),
}

_WHY = re.compile(
    r"^(why|why\?|why is that|why though|how come|because|and why)\??$",
    re.I,
)
_MORE = re.compile(
    r"^(tell me more|more|explain|explain more|go on|continue|keep going|details?)\??$",
    re.I,
)
_SIMPLE = re.compile(
    r"^(simplify|simpler|simple words|in simple words|eli5|explain simply|plain english)\??$",
    re.I,
)
_NEXT = re.compile(
    r"^(what next|what now|ok then|okay then|so what|what should i do|"
    r"next step|and then)\??$",
    re.I,
)
_AFFIRM = re.compile(r"^(yes|yep|yeah|ok|okay|sure|do it|please|go ahead)\??$", re.I)
_LEVEL = re.compile(
    r"^(the )?(entry|stop|target|levels?|plan|trade plan)\??$",
    re.I,
)
_OTHER = re.compile(
    r"^(and|what about|how about|also|aur)\s+([A-Za-z0-9.&-]{2,15})\??$",
    re.I,
)


def detect_small_talk(query: str) -> str | None:
    try:
        from .query_language import telugu_small_talk_kind

        te_kind = telugu_small_talk_kind(query)
        if te_kind:
            return te_kind
    except Exception:
        pass
    q = re.sub(r"[^a-z0-9\s]", " ", (query or "").lower())
    q = re.sub(r"\s+", " ", q).strip()
    if not q:
        return None
    if re.search(r"\b(how are you|how r u|kaise ho|kaise how ho|kya haal)\b", q):
        try:
            from .query_contract import _symbol_candidates

            if any(len(str(s)) >= 3 for s in (_symbol_candidates(query) or [])):
                return None
        except Exception:
            pass
        return "how_are_you"
    if re.search(r"\b(thanks|thank you|thx|ty|appreciate)\b", q) and len(q.split()) <= 6:
        return "thanks"
    if re.search(r"\b(bye|goodbye|see you|cya|good night)\b", q) and len(q.split()) <= 6:
        return "bye"
    greet = {"hi", "hii", "hiii", "hello", "hey", "namaste", "namaskar", "yo", "gm"}
    extra = {
        "there", "team", "bot", "ai", "assistant", "bysel", "good",
        "morning", "afternoon", "evening",
    }
    tokens = q.split()
    if len(tokens) <= 5 and any(t in greet or t in {"good"} for t in tokens):
        if all(t in greet or t in extra for t in tokens):
            return "greeting"
    return None


def small_talk_reply(query: str) -> str | None:
    kind = detect_small_talk(query)
    if not kind:
        return None
    try:
        from .query_language import is_telugu_query, telugu_small_talk_replies

        if is_telugu_query(query):
            return telugu_small_talk_replies().get(kind) or _SMALL_TALK[kind]
    except Exception:
        pass
    return _SMALL_TALK.get(kind)


def _shaped(profile: str | None, symbol: str) -> str:
    return _SHAPE.get(profile or "", "Analyze {sym}").format(sym=symbol)


def expand_conversational_followup(
    query: str,
    *,
    prior_symbol: str | None,
    prior_profile: str | None,
) -> str | None:
    """Turn a short chat turn into a full ISM ask, or None if it is a new topic."""
    raw = (query or "").strip()
    if not raw or not prior_symbol:
        return None
    q = raw.lower().strip()
    if _WHY.match(q):
        if prior_profile == "trade_plan":
            return f"Why this paper stance on {prior_symbol}?"
        if prior_profile in {"news", "sentiment"}:
            return f"Why is {prior_symbol} moving?"
        return f"Explain the {prior_symbol} snapshot"
    if _MORE.match(q):
        # "Tell me more" after a last-price card should open the snapshot,
        # not print the same quote again.
        if prior_profile in {"quote", None, ""}:
            return f"How is {prior_symbol} doing?"
        return _shaped(prior_profile, prior_symbol)
    if _SIMPLE.match(q):
        return f"Explain {prior_symbol} in simple words"
    if _NEXT.match(q):
        if prior_profile == "trade_plan":
            return f"What are risks in {prior_symbol} now?"
        if prior_profile in {"quote", "technical", "stock_analysis"}:
            return f"Should I buy {prior_symbol}?"
        return _shaped("trade_plan", prior_symbol)
    if _LEVEL.match(q):
        return f"Practice levels for {prior_symbol}"
    if _AFFIRM.match(q):
        return _shaped(prior_profile, prior_symbol)
    other = _OTHER.match(q)
    if other:
        token = other.group(2).upper()
        topicish = {
            "IT", "THE", "NEWS", "PRICE", "RSI", "THIS", "THAT", "SENTIMENT",
            "TECHNICAL", "TECHNICALS", "FUNDAMENTAL", "FUNDAMENTALS", "PE",
            "VALUATION", "RISK", "RISKS", "FORECAST", "PREDICTION", "QUOTE",
            "CHART", "LEVELS", "PLAN",
        }
        if token not in topicish:
            return _shaped(prior_profile, token)
    return None


def conversational_lead(
    *,
    follow_up: bool,
    symbol: str | None,
    profile: str | None,
    sentiment: Mapping[str, Any] | None = None,
    query: str = "",
) -> str | None:
    """One spoken line before the card, only on real follow-ups."""
    if not follow_up:
        return None
    try:
        from .query_language import is_telugu_query

        if is_telugu_query(query):
            return None
    except Exception:
        pass
    tone = str((sentiment or {}).get("emotion") or "neutral")
    name = (symbol or "").upper()
    if tone in {"frustrated", "anxious"}:
        if name:
            return f"I hear the caution — staying on {name}, calmly."
        return "I hear the caution — here is a steady read."
    if profile == "risks" and name:
        return f"Next useful step on {name}: the risks to size against."
    if profile == "trade_plan" and name:
        return f"Still on {name} — paper call only, not advice."
    if name:
        return f"Staying with {name}."
    return None


def last_assistant_excerpt(history: Sequence[Mapping[str, Any]] | None) -> str:
    if not history:
        return ""
    for turn in reversed(list(history)):
        if str((turn or {}).get("role") or "").lower() != "assistant":
            continue
        text = str((turn or {}).get("content") or "").strip()
        if text:
            return text[:240]
    return ""
