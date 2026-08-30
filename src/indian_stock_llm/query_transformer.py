"""Query transformer for ISM — analyze the ask, then pick a profile.

This is not a generative HuggingFace LLM and not LoRA fine-tuning.
Cloud Run stays at 2 Gi: we never import torch here.

Heads:
  profile  — which answer card
  facet    — wording-specific shape
  live     — live tape vs literacy

Encoder:
  MiniLM via the existing KB embedding provider when BYSEL already loaded it
  local hash embeddings otherwise (always available)
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

EncodeFn = Callable[[list[str]], list[tuple[float, ...]]]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Indian-retail prototypes the transformer attends over.
_PROFILE_PROTOTYPES: dict[str, tuple[str, ...]] = {
    "quote": (
        "what is the price of reliance",
        "reliance ka price",
        "kitna hai tcs",
        "sbin cmp",
        "live last traded price",
    ),
    "trade_plan": (
        "should i buy reliance",
        "hold or exit sbin",
        "book profit in infy",
        "kya main hdfcbank kharidun",
        "accumulate on dips",
    ),
    "news": (
        "any update on reliance",
        "why is reliance falling",
        "reliance q2 results",
        "fii buying in reliance",
        "upper circuit on yesbank",
    ),
    "sentiment": (
        "mood on hdfcbank",
        "is market bullish today",
        "are traders bullish on tcs",
    ),
    "technical": (
        "show reliance chart",
        "50 ema of reliance",
        "200 dma of infy",
        "support resistance for sbin",
        "volume spike in sbin",
    ),
    "prediction": (
        "infy target next month",
        "will tcs reach 4000",
        "nifty outlook",
    ),
    "fundamentals": (
        "is tcs cheap",
        "pe of infy",
        "market cap of reliance",
        "promoter pledge in adanient",
        "nifty 50 pe ratio",
    ),
    "risks": (
        "downside in reliance",
        "what can go wrong in infy",
        "key risks now",
    ),
    "literacy": (
        "what is rsi",
        "explain delivery vs intraday",
        "how to apply for ipo",
        "what is a straddle",
        "t plus 1 settlement meaning",
    ),
    "compare": (
        "tcs or infy",
        "hdfcbank vs icicibank",
    ),
    "compare_concepts": (
        "sip vs lumpsum",
        "gold vs stocks",
    ),
    "sector_screen": (
        "which psu stocks",
        "best it stocks",
        "auto names for swing",
    ),
    "portfolio": (
        "what is on my watchlist",
        "position size for reliance",
        "my portfolio concentration",
    ),
    "derivatives": (
        "covered call on reliance",
        "lot size of reliance",
        "bank nifty expiry this week",
    ),
    "calculations": (
        "calculate cagr for 100000 to 180000 in 3 years",
        "compute return from buy",
    ),
    "session": (
        "is the market open",
        "nse trading hours",
        "when does the market close",
    ),
    "corporate_actions": (
        "dividend date of infy",
        "ex date of reliance",
        "corporate actions for tcs",
    ),
    "stock_analysis": (
        "how is reliance doing",
        "tcs ka kya haal",
        "tell me about bel",
    ),
    "small_talk": (
        "good morning",
        "hello",
        "kaise ho",
    ),
}

_PROTECTED = frozenset(
    {
        "small_talk",
        "session",
        "corporate_actions",
        "literacy",
        "compare_concepts",
        "calculations",
    }
)

_encode_fn: EncodeFn | None = None
_proto_cache: dict[str, list[tuple[float, ...]]] | None = None


@dataclass(frozen=True)
class TransformerNLU:
    profile: str
    score: float
    margin: float
    scores: dict[str, float] = field(default_factory=dict)
    encoder: str = "hash"
    used: bool = False


def query_transformer_enabled() -> bool:
    value = os.getenv("ISM_QUERY_TRANSFORMER", "true")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def configure_query_encoder(encode: EncodeFn | None) -> None:
    """Reuse MiniLM (or any KB encoder) when BYSEL already loaded it."""
    global _encode_fn, _proto_cache
    _encode_fn = encode
    _proto_cache = None


def _default_encode(texts: list[str]) -> list[tuple[float, ...]]:
    from .knowledge_base import LocalHashEmbeddingProvider

    return LocalHashEmbeddingProvider().encode(texts)


def _encode(texts: list[str]) -> tuple[list[tuple[float, ...]], str]:
    fn = _encode_fn or _default_encode
    name = "minilm" if _encode_fn is not None else "hash"
    return fn(texts), name


def _tokens(text: str) -> set[str]:
    return {tok for tok in _TOKEN_RE.findall((text or "").lower()) if len(tok) > 1}


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b:
        return 0.0
    return float(sum(x * y for x, y in zip(a, b)))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _softmax(values: list[float], temperature: float = 0.12) -> list[float]:
    if not values:
        return []
    peak = max(values)
    exps = [math.exp((v - peak) / max(temperature, 0.02)) for v in values]
    total = sum(exps) or 1.0
    return [e / total for e in exps]


def _prototype_vectors() -> dict[str, list[tuple[float, ...]]]:
    global _proto_cache
    if _proto_cache is not None:
        return _proto_cache
    texts: list[str] = []
    owners: list[str] = []
    for profile, phrases in _PROFILE_PROTOTYPES.items():
        for phrase in phrases:
            texts.append(phrase)
            owners.append(profile)
    vectors, _ = _encode(texts)
    packed: dict[str, list[tuple[float, ...]]] = {key: [] for key in _PROFILE_PROTOTYPES}
    for owner, vec in zip(owners, vectors):
        packed[owner].append(vec)
    _proto_cache = packed
    return packed


def analyze_query_transformer(query: str) -> TransformerNLU:
    """Multi-head attention over Indian-market prototypes."""
    text = (query or "").strip()
    if not text:
        return TransformerNLU(profile="stock_analysis", score=0.0, margin=0.0)
    q_tokens = _tokens(text)
    (q_vec,), encoder = _encode([text.lower()])
    proto = _prototype_vectors()
    raw: dict[str, float] = {}
    for profile, vectors in proto.items():
        phrases = _PROFILE_PROTOTYPES[profile]
        best = 0.0
        for vec, phrase in zip(vectors, phrases):
            lexical = _jaccard(q_tokens, _tokens(phrase))
            fused = (0.55 * _cosine(q_vec, vec)) + (0.45 * lexical)
            if fused > best:
                best = fused
        raw[profile] = best
    ranked = sorted(raw.items(), key=lambda item: item[1], reverse=True)
    top_profile, top_score = ranked[0]
    second = ranked[1][1] if len(ranked) > 1 else 0.0
    weights = _softmax([score for _, score in ranked[:8]])
    scores = {ranked[i][0]: weights[i] for i in range(len(weights))}
    return TransformerNLU(
        profile=top_profile,
        score=float(scores.get(top_profile, top_score)),
        margin=float(top_score - second),
        scores=scores,
        encoder=encoder,
        used=False,
    )


def apply_transformer_route(
    profile: str,
    confidence: int,
    analysis: TransformerNLU,
) -> tuple[str, int, bool]:
    """Blend rule routing with transformer attention.

    High-confidence literacy / session / dated-event cards stay put.
    Low-confidence fallbacks can be rescued when attention is clear.
    """
    if not query_transformer_enabled():
        return profile, confidence, False
    if analysis.profile == profile:
        return profile, min(96, int(confidence) + 6), True
    if profile in _PROTECTED and int(confidence) >= 84:
        return profile, confidence, False
    strong = analysis.margin >= 0.10 and analysis.score >= 0.18
    weak_rule = int(confidence) < 68
    fallback = profile == "stock_analysis" and int(confidence) <= 55
    if (weak_rule or fallback) and strong and analysis.profile != "small_talk":
        return analysis.profile, max(int(confidence), 76), True
    return profile, confidence, False
