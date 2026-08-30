"""Sentence + user-sentiment analysis for the Indian Stock LLM.

Two jobs, both owned by ISM (not Groq):

1. Split a user ask into sentences/clauses and tag each one
   (profile, polarity, emotion) so compound questions stay faithful.
2. Detect the speaker's tone (urgency, risk, emotion, investor vs trader)
   so answers can stay calm, staged, or more direct as needed.

Market headline sentence polarity lives in analysis_math; this module
exposes it through analyze_market_sentences().
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Sequence

_SPLIT_RE = re.compile(
    r"(?<=[.!?])\s+"
    r"|(?:;|\u2014|\u2013)\s+"
    r"|\s+(?:and also|also,)\s+"
    r"|\s+\balso\s+(?=should|can|what|how|why|is|give|show|tell|buy|sell)"
    r"|\s+\band\s+(?=should i|can i|what(?:'s| is)|how |why |is it|give |show |tell )",
    flags=re.I,
)

_URGENT = ("urgent", "asap", "right now", "immediately", "hurry", "jaldi", "abhi")
_CONSERVATIVE = (
    "safe", "conservative", "low risk", "dividend", "stable", "blue chip",
    "sip", "long term", "retirement",
)
_AGGRESSIVE = (
    "aggressive", "leverage", "leveraged", "swing", "day trade", "intraday",
    "scalp", "momentum", "breakout",
)
_FRUSTRATED = (
    "killing me", "frustrated", "loss", "losing", "crash", "panic", "worried",
    "stressed", "scared", "afraid", "tension", "pareshan", "down badly",
)
_EXCITED = (
    "excited", "bullish", "boom", "rocket", "moon", "thrilled", "huge gain",
    "doubled", "pumped",
)
_INVESTOR = ("long term", "sip", "holding", "portfolio", "diversify", "retirement")
_TRADER = ("swing", "day trade", "intraday", "scalp", "technical", "momentum", "breakout")


@dataclass(frozen=True)
class SentenceUnit:
    text: str
    profile: str = "stock_analysis"
    polarity: str = "neutral"
    emotion: str = "neutral"
    urgency: str = "low"


@dataclass
class QueryNLU:
    query: str
    sentences: tuple[SentenceUnit, ...] = ()
    primary: SentenceUnit | None = None
    secondary: SentenceUnit | None = None
    compound: bool = False
    user_sentiment: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "compound": self.compound,
            "primary_profile": (self.primary.profile if self.primary else None),
            "secondary_profile": (self.secondary.profile if self.secondary else None),
            "primary_text": (self.primary.text if self.primary else self.query),
            "secondary_text": (self.secondary.text if self.secondary else None),
            "sentences": [asdict(unit) for unit in self.sentences],
            "user_sentiment": dict(self.user_sentiment),
        }


def split_user_sentences(query: str) -> list[str]:
    """Split a retail ask into clauses without breaking 'buy and sell'."""
    text = re.sub(r"\s+", " ", (query or "").strip())
    if not text:
        return []
    if text.lower().startswith("user_query:"):
        text = text.split(":", 1)[1].strip()
    if " | context:" in text:
        text = text.split(" | context:", 1)[0].strip()
    parts = [p.strip(" ,") for p in _SPLIT_RE.split(text) if p and p.strip()]
    merged: list[str] = []
    for part in parts:
        if merged and len(part.split()) <= 2 and not re.search(r"[.!?]$", merged[-1]):
            merged[-1] = f"{merged[-1]} {part}".strip()
        else:
            merged.append(part)
    return merged or [text]


def _clause_polarity(text: str) -> tuple[str, str]:
    """Return (polarity, emotion) for one user clause."""
    t = (text or "").lower()
    frust = sum(1 for kw in _FRUSTRATED if kw in t)
    excit = sum(1 for kw in _EXCITED if kw in t)
    if frust > excit:
        return "negative", "frustrated"
    if excit > frust:
        return "positive", "excited"
    if re.search(r"\b(worried|fear|down|fall|crash|loss)\b", t):
        return "negative", "anxious"
    if re.search(r"\b(good|great|strong|buy the dip|looking good)\b", t):
        return "positive", "hopeful"
    return "neutral", "neutral"


def _clause_urgency(text: str) -> str:
    t = (text or "").lower()
    bangs = (text or "").count("!") + (text or "").count("?")
    caps = sum(1 for c in (text or "") if c.isupper()) / max(len(text or ""), 1)
    if any(k in t for k in _URGENT) or bangs >= 2 or caps > 0.45:
        return "high"
    if bangs == 1 or "now" in t or "today" in t or "soon" in t:
        return "medium"
    return "low"


def detect_user_sentiment(
    query: str,
    history: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """ISM-owned speaker tone. Safe to call from Groq and /ai/ask."""
    q = (query or "").lower()
    sentiment = {
        "urgency": "low",
        "risk_appetite": "moderate",
        "emotion": "neutral",
        "user_profile": "unknown",
        "polarity": "neutral",
    }
    if not q.strip():
        return sentiment

    units = split_user_sentences(query)
    urgencies = [_clause_urgency(u) for u in units] or ["low"]
    if "high" in urgencies:
        sentiment["urgency"] = "high"
    elif "medium" in urgencies:
        sentiment["urgency"] = "medium"

    cons = sum(1 for kw in _CONSERVATIVE if kw in q)
    agg = sum(1 for kw in _AGGRESSIVE if kw in q)
    if agg > cons:
        sentiment["risk_appetite"] = "aggressive"
    elif cons > agg:
        sentiment["risk_appetite"] = "conservative"

    polarities = [_clause_polarity(u) for u in units] or [("neutral", "neutral")]
    emotions = [e for _, e in polarities]
    if emotions.count("frustrated") + emotions.count("anxious") > emotions.count("excited"):
        sentiment["emotion"] = "frustrated" if "frustrated" in emotions else "anxious"
        sentiment["polarity"] = "negative"
    elif emotions.count("excited") + emotions.count("hopeful") > 0 and "frustrated" not in emotions:
        sentiment["emotion"] = "excited" if "excited" in emotions else "hopeful"
        sentiment["polarity"] = "positive"

    inv = sum(1 for kw in _INVESTOR if kw in q)
    trd = sum(1 for kw in _TRADER if kw in q)
    if history:
        hist = " ".join(
            str((turn or {}).get("content") or "")
            for turn in list(history)[-6:]
            if isinstance(turn, Mapping)
        ).lower()
        inv += sum(1 for kw in _INVESTOR if kw in hist)
        trd += sum(1 for kw in _TRADER if kw in hist)
    if trd > inv:
        sentiment["user_profile"] = "trader"
    elif inv > trd:
        sentiment["user_profile"] = "investor"
    return sentiment


def analyze_user_query(
    query: str,
    *,
    intent_hint: str = "",
    history: Sequence[Mapping[str, Any]] | None = None,
) -> QueryNLU:
    """Sentence-level NLU for one user turn."""
    from .query_contract import detect_profile

    clauses = split_user_sentences(query)
    units: list[SentenceUnit] = []
    for clause in clauses[:4]:
        profile, _, _ = detect_profile(clause, intent_hint=intent_hint)
        polarity, emotion = _clause_polarity(clause)
        units.append(
            SentenceUnit(
                text=clause,
                profile=profile,
                polarity=polarity,
                emotion=emotion,
                urgency=_clause_urgency(clause),
            )
        )
    if not units:
        units.append(SentenceUnit(text=(query or "").strip() or ""))

    primary = units[0]
    secondary = None
    for unit in units[1:]:
        if unit.profile and unit.profile != primary.profile:
            secondary = unit
            break
    return QueryNLU(
        query=(query or "").strip(),
        sentences=tuple(units),
        primary=primary,
        secondary=secondary,
        compound=secondary is not None,
        user_sentiment=detect_user_sentiment(query, history),
    )


def analyze_market_sentences(text: str) -> list[dict[str, Any]]:
    """Sentence polarity for news / headlines (shared ISM API)."""
    try:
        from .analysis_math import analyze_headline_sentences

        return analyze_headline_sentences(text)
    except Exception:
        return []
