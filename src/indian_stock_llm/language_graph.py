"""Domain language graph for Indian Stock LLM query understanding.

ISM used to classify with keyword bags and a blank spaCy tokenizer. That misses
habit / teach / compare-concept asks. This module builds a small concept graph
(nodes + relations + expansions) so routing and RAG see the same meaning in
more depth — without loading a large language model.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

_SPACY_GRAPH = None
_SPACY_GRAPH_TRIED = False


def language_graph_enabled() -> bool:
    value = os.getenv("ISM_LANGUAGE_GRAPH_ENABLED", "true")
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class GraphNode:
    id: str
    kind: str
    label: str
    weight: float = 1.0


@dataclass(frozen=True)
class GraphEdge:
    src: str
    rel: str
    dst: str


@dataclass
class LanguageGraph:
    query: str
    language: str = "en"
    nodes: tuple[GraphNode, ...] = ()
    edges: tuple[GraphEdge, ...] = ()
    concepts: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    retrieval_query: str = ""
    profile_hint: str | None = None
    intent_hint: str | None = None
    confidence: int = 0
    reason: str = ""


# pattern, concept_id, expansions (RAG cues), profile, intent
_CONCEPTS: tuple[tuple[str, str, tuple[str, ...], str, str], ...] = (
    (
        r"opening range|first[- ]hour|first hour",
        "opening_range",
        ("session clock", "first hour", "preopen", "volatility", "paper habit"),
        "literacy",
        "market_literacy",
    ),
    (
        r"midday chop|revenge trad",
        "midday_chop",
        ("session habit", "lunch hour", "stand aside", "paper habit"),
        "literacy",
        "market_literacy",
    ),
    (
        r"closing auction|\bcas\b|square[- ]off",
        "session_close",
        ("closing auction", "session clock", "square off", "paper habit"),
        "literacy",
        "market_literacy",
    ),
    (
        r"weekend(s)? and holiday|review paper trades on weekend|holiday prep",
        "weekend_prep",
        ("weekend", "holiday", "journal", "watchlist", "paper habit"),
        "literacy",
        "market_literacy",
    ),
    (
        r"stop[- ]loss.{0,40}fomo|avoid fomo|position siz",
        "session_risk",
        ("stop", "size", "fomo", "invalidation", "paper habit", "risk"),
        "literacy",
        "market_literacy",
    ),
    (
        r"long[- ]term investing|horizon first|asset allocation",
        "long_term_habit",
        ("horizon", "allocation", "sip", "investor habit"),
        "literacy",
        "market_literacy",
    ),
    (
        r"mutual funds?|nav, sip|\bsip\b.{0,20}\bter\b|\bter\b.{0,20}nav",
        "mutual_funds",
        ("nav", "sip", "ter", "factsheet", "mutual fund"),
        "literacy",
        "market_literacy",
    ),
    (
        r"\bipo\b.{0,40}(drhp|allotment)|read an? ipo|\bdrhp\b",
        "ipo_habit",
        ("ipo", "drhp", "allotment", "prospectus"),
        "literacy",
        "market_literacy",
    ),
    (
        r"sovereign gold bond|\bsgb\b",
        "sgb_habit",
        ("sgb", "gold bond", "gold etf", "issue circular"),
        "literacy",
        "market_literacy",
    ),
    (
        r"futures vs options|difference between.{0,40}futures and options|"
        r"lot size and margin|lose value as expiry",
        "fno_habit",
        ("futures", "options", "lot size", "margin", "expiry", "paper habit"),
        "literacy",
        "market_literacy",
    ),
    (
        r"\b(what is|what are|explain|define|meaning of|teach|how should beginners)\b",
        "definitional",
        ("education", "meaning"),
        "literacy",
        "market_literacy",
    ),
)

_TEACH_CUE = re.compile(
    r"\b(teach|paper habit|paper[- ]practice|educational|beginner|"
    r"not a stock pick|session habit|investor habit)\b",
    re.I,
)
_SYMBOLISH = re.compile(r"\b([A-Z][A-Z0-9.&-]{1,11})\b")
_SKIP_TICKERS = frozenset(
    {
        "NSE", "BSE", "SEBI", "NIFTY", "SENSEX", "FII", "DII", "IPO", "SGB",
        "NAV", "SIP", "TER", "CAS", "FOMO", "DRHP", "ETF", "RSI", "MACD",
        "SMA", "EMA", "ATR", "VWAP", "PE", "PB", "ROE", "EPS", "CAGR",
        "WHAT", "HOW", "WHY", "WHEN", "THE", "AND", "FOR", "NOT", "ONLY",
        "BUY", "SELL", "HOLD", "WAIT", "TRIM", "STOCK", "SHARE", "MARKET",
        "PAPER", "HABIT", "LEARN", "INDIAN", "BEGINNERS", "COVER",
        "RANGE", "HOUR", "FIRST", "VOLATILITY", "EDUCATIONAL",
        "STOP", "LOSS", "SIZE", "AVOID", "CHOP", "WEEKEND", "HOLIDAY",
    }
)


def _detect_language(query: str) -> str:
    try:
        from .query_language import detect_query_language

        return detect_query_language(query)
    except Exception:
        return "en"


def _normalize(query: str) -> str:
    try:
        from .query_language import normalize_user_query

        return normalize_user_query(query)
    except Exception:
        return (query or "").strip()


def _symbols(query: str) -> tuple[str, ...]:
    found: list[str] = []
    for match in _SYMBOLISH.finditer(query or ""):
        token = match.group(1).upper()
        if token in _SKIP_TICKERS or token in found:
            continue
        found.append(token)
    return tuple(found[:4])


def _spacy_relations(query: str) -> list[tuple[str, str, str]]:
    """Optional dependency-graph edges when a real spaCy model is installed."""
    global _SPACY_GRAPH, _SPACY_GRAPH_TRIED
    if _SPACY_GRAPH_TRIED and _SPACY_GRAPH is None:
        return []
    if _SPACY_GRAPH is None:
        _SPACY_GRAPH_TRIED = True
        model = os.getenv("ISM_SPACY_MODEL", "").strip()
        names = [model] if model else ["en_core_web_sm"]
        try:
            import spacy  # type: ignore
        except Exception:
            return []
        for name in names:
            if not name:
                continue
            try:
                nlp = spacy.load(name)
            except Exception:
                continue
            if "parser" not in nlp.pipe_names:
                continue
            _SPACY_GRAPH = nlp
            break
        if _SPACY_GRAPH is None:
            return []
    try:
        doc = _SPACY_GRAPH(query)
    except Exception:
        return []
    rels: list[tuple[str, str, str]] = []
    for token in doc:
        if token.dep_ in {"nsubj", "dobj", "pobj", "amod", "compound"} and token.head != token:
            rels.append((token.head.lemma_.lower(), token.dep_, token.lemma_.lower()))
    return rels[:12]


def build_language_graph(query: str) -> LanguageGraph:
    raw = (query or "").strip()
    if not raw or not language_graph_enabled():
        return LanguageGraph(query=raw, retrieval_query=raw)

    normalized = _normalize(raw)
    language = _detect_language(raw)
    text = f"{raw} {normalized}".strip()
    low = text.lower()

    nodes: list[GraphNode] = [
        GraphNode("q0", "query", raw[:160]),
        GraphNode("lang", "language", language),
    ]
    edges: list[GraphEdge] = [GraphEdge("q0", "IN_LANGUAGE", "lang")]
    concepts: list[str] = []
    expansions: list[str] = []
    profile = None
    intent = None
    confidence = 0
    reason = ""

    for pattern, concept_id, expand, prof, ism_intent in _CONCEPTS:
        if not re.search(pattern, low, flags=re.I):
            continue
        if concept_id in concepts:
            continue
        concepts.append(concept_id)
        nodes.append(GraphNode(concept_id, "concept", concept_id, 1.2))
        edges.append(GraphEdge("q0", "HAS_CONCEPT", concept_id))
        for cue in expand:
            exp_id = f"exp:{cue.replace(' ', '_')}"
            nodes.append(GraphNode(exp_id, "expansion", cue, 0.8))
            edges.append(GraphEdge(concept_id, "EXPANDS_TO", exp_id))
            expansions.append(cue)
        if concept_id != "definitional":
            profile = prof
            intent = ism_intent
            confidence = max(confidence, 88)
            reason = f"language graph:{concept_id}"

    teach = bool(_TEACH_CUE.search(low))
    if teach:
        nodes.append(GraphNode("teach", "relation", "teach"))
        edges.append(GraphEdge("q0", "HAS_RELATION", "teach"))
        if concepts and profile is None:
            profile = "literacy"
            intent = "market_literacy"
            confidence = max(confidence, 84)
            reason = "language graph:teach"

    # Definitional alone is weak; only promote literacy when another concept exists
    # or the user explicitly asked to teach a habit.
    if profile is None and "definitional" in concepts and teach:
        profile = "literacy"
        intent = "market_literacy"
        confidence = 80
        reason = "language graph:definitional+teach"

    symbols = _symbols(raw)
    for sym in symbols:
        node_id = f"sym:{sym}"
        nodes.append(GraphNode(node_id, "entity", sym, 1.1))
        edges.append(GraphEdge("q0", "MENTIONS", node_id))

    if symbols and profile == "literacy" and not teach:
        # "opening range of RELIANCE" is a tape read, not a habit lesson.
        profile = "technical"
        intent = "stock_analysis"
        confidence = 82
        reason = "language graph:concept+symbol"
        for sym in symbols:
            edges.append(GraphEdge(concepts[0] if concepts else "q0", "OF", f"sym:{sym}"))

    if re.search(r"\b(compare|versus|\bvs\b)\b", low) and not any(
        c in concepts for c in ("fno_habit", "mutual_funds", "sgb_habit", "session_close")
    ) and symbols:
        profile = profile or "compare"
        intent = intent or "compare"
        confidence = max(confidence, 86)
        reason = reason or "language graph:compare"

    for head, dep, child in _spacy_relations(normalized or raw):
        src = f"tok:{head}"
        dst = f"tok:{child}"
        nodes.append(GraphNode(src, "token", head, 0.5))
        nodes.append(GraphNode(dst, "token", child, 0.5))
        edges.append(GraphEdge(src, dep.upper(), dst))

    retrieval_parts = [normalized or raw]
    if expansions:
        retrieval_parts.append(" ".join(dict.fromkeys(expansions)))
    retrieval_query = re.sub(r"\s+", " ", " ".join(retrieval_parts)).strip()

    return LanguageGraph(
        query=raw,
        language=language,
        nodes=tuple(nodes),
        edges=tuple(edges),
        concepts=tuple(concepts),
        symbols=symbols,
        retrieval_query=retrieval_query or raw,
        profile_hint=profile,
        intent_hint=intent,
        confidence=confidence,
        reason=reason,
    )


def graph_retrieval_query(query: str) -> str:
    graph = build_language_graph(query)
    return graph.retrieval_query or (query or "").strip()
