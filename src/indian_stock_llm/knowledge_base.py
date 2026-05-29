from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class KnowledgeItem:
    id: str
    title: str
    content: str
    tags: list[str]
    source: str


_WORD_RE = re.compile(r"[a-zA-Z0-9]+(?:[/-][a-zA-Z0-9]+)*")
_NGRAM_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "of",
    "on",
    "for",
    "in",
    "to",
    "is",
    "are",
    "with",
    "how",
    "what",
    "explain",
}
EMBEDDING_DIM = 64
INTENT_TAG_PRIORS = {
    "fundamentals": {"fundamentals", "valuation", "pe"},
    "events_news": {"sebi", "regulation", "guidance", "earnings"},
    "market_calculations": {"calculation", "cagr", "return", "volatility"},
    "prediction": {"prediction", "forecast", "uncertainty", "risk"},
    "stock_analysis": {"analysis", "technical", "fundamental", "earnings"},
    "portfolio": {"portfolio", "risk", "diversification"},
}


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _WORD_RE.finditer(text) if m.group(0).lower() not in _STOPWORDS}


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    chunks: list[str] = []
    for token in _NGRAM_RE.findall(text.lower()):
        if len(token) < n:
            chunks.append(token)
            continue
        for idx in range(len(token) - n + 1):
            chunks.append(token[idx : idx + n])
    return chunks


def _embedding(text: str) -> tuple[float, ...]:
    vector = [0.0] * EMBEDDING_DIM
    ngrams = _char_ngrams(text)
    if not ngrams:
        return tuple(vector)
    for gram in ngrams:
        vector[hash(gram) % EMBEDDING_DIM] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return tuple(vector)
    return tuple(value / norm for value in vector)


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if not a or not b:
        return 0.0
    return sum(x * y for x, y in zip(a, b))


class KnowledgeBase:
    def __init__(self, items: Iterable[KnowledgeItem]):
        self.items = list(items)
        self._item_tokens = [
            _tokenize(f"{item.title} {item.content} {' '.join(item.tags)}") for item in self.items
        ]
        self._item_embeddings = [
            _embedding(f"{item.title} {item.content} {' '.join(item.tags)}") for item in self.items
        ]

    @classmethod
    def from_json(cls, path: Path) -> KnowledgeBase:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [KnowledgeItem(**item) for item in data]
        return cls(items)

    def _semantic_score(self, query_tokens: set[str], item_tokens: set[str]) -> float:
        if not query_tokens or not item_tokens:
            return 0.0
        overlap = len(query_tokens & item_tokens)
        union = len(query_tokens | item_tokens)
        return overlap / union if union else 0.0

    def _intent_boost(self, intent: str | None, item: KnowledgeItem) -> float:
        if not intent:
            return 0.0
        preferred_tags = INTENT_TAG_PRIORS.get(intent, set())
        if not preferred_tags:
            return 0.0
        overlap = len(preferred_tags & set(item.tags))
        return 0.2 * overlap

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.2,
        metadata_filters: dict[str, str] | None = None,
        intent: str | None = None,
    ) -> list[KnowledgeItem]:
        query_tokens = _tokenize(query)
        query_embedding = _embedding(query)
        scored = []
        for item, item_tokens, item_embedding in zip(self.items, self._item_tokens, self._item_embeddings):
            if metadata_filters:
                source_filter = metadata_filters.get("source")
                tag_filter = metadata_filters.get("tag")
                if source_filter and item.source != source_filter:
                    continue
                if tag_filter and tag_filter not in item.tags:
                    continue
            keyword_score = len(query_tokens & item_tokens)
            semantic_score = self._semantic_score(query_tokens, item_tokens)
            embedding_score = _cosine(query_embedding, item_embedding)
            score = keyword_score + semantic_score + embedding_score
            rerank_score = score + self._intent_boost(intent, item)
            if query_tokens and query_tokens <= item_tokens:
                rerank_score += 0.3
            has_signal = keyword_score > 0 or semantic_score >= min_score or embedding_score >= 0.55
            if has_signal and score >= min_score:
                scored.append((rerank_score, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]
