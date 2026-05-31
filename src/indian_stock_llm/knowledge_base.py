from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
from urllib import request


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
    "market_calculations": {"calculation", "cagr", "return", "volatility", "rsi", "sma", "ema", "macd", "bollinger"},
    "prediction": {"prediction", "forecast", "uncertainty", "risk"},
    "stock_analysis": {"analysis", "technical", "fundamental", "earnings"},
    "portfolio": {"portfolio", "risk", "diversification"},
}


class EmbeddingProvider(Protocol):
    def encode(self, texts: list[str]) -> list[tuple[float, ...]]: ...


class VectorIndex(Protocol):
    def top_k(self, query_embedding: tuple[float, ...], item_embeddings: list[tuple[float, ...]], k: int) -> list[tuple[int, float]]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        intent: str | None,
        scored_items: list[tuple[KnowledgeItem, float, int, float]],
    ) -> list[tuple[KnowledgeItem, float, int, float]]: ...


class LocalHashEmbeddingProvider:
    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        return [_embedding(text) for text in texts]


class SentenceTransformerEmbeddingProvider:
    """Semantic embedding provider backed by a local sentence-transformers model.

    Requires: pip install sentence-transformers  (in requirements-optional.txt)
    Falls back gracefully to LocalHashEmbeddingProvider if the library is unavailable.
    """

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name)
        except Exception:
            self._model = None
        return self._model

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        model = self._load_model()
        if model is None:
            return LocalHashEmbeddingProvider().encode(texts)
        try:
            embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            result: list[tuple[float, ...]] = []
            for emb in embeddings:
                vec = [float(v) for v in emb]
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    vec = [v / norm for v in vec]
                result.append(tuple(vec))
            return result
        except Exception:
            return LocalHashEmbeddingProvider().encode(texts)


@dataclass(frozen=True)
class HttpEmbeddingProvider:
    endpoint: str
    api_key: str | None = None
    provider: str = "generic"
    model: str | None = None
    timeout_seconds: float = 2.0

    def encode(self, texts: list[str]) -> list[tuple[float, ...]]:
        provider = self.provider.strip().lower()
        if provider in {"openai", "azure_openai"}:
            payload = {"input": texts}
            if self.model:
                payload["model"] = self.model
        else:
            payload = {"texts": texts}
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            header = "Authorization" if provider in {"openai", "azure_openai"} else "X-API-Key"
            value = self.api_key
            req.add_header(header, value)
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            raise ValueError("embedding provider returned invalid payload")
        if provider in {"openai", "azure_openai"}:
            rows = body.get("data", [])
            embeddings = [row.get("embedding") for row in rows if isinstance(row, dict)]
        else:
            embeddings = body.get("embeddings", [])
        parsed: list[tuple[float, ...]] = []
        for row in embeddings:
            if not isinstance(row, list) or not row:
                continue
            parsed.append(tuple(float(value) for value in row))
        if len(parsed) != len(texts):
            raise ValueError("embedding provider returned incomplete embeddings")
        return parsed


class InMemoryVectorIndex:
    def top_k(self, query_embedding: tuple[float, ...], item_embeddings: list[tuple[float, ...]], k: int) -> list[tuple[int, float]]:
        scored = [(idx, _cosine(query_embedding, emb)) for idx, emb in enumerate(item_embeddings)]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[: max(1, k)]


class HeuristicReranker:
    @staticmethod
    def _intent_boost(intent: str | None, item: KnowledgeItem) -> float:
        if not intent:
            return 0.0
        preferred_tags = INTENT_TAG_PRIORS.get(intent, set())
        if not preferred_tags:
            return 0.0
        overlap = len(preferred_tags & set(item.tags))
        return 0.2 * overlap

    def rerank(
        self,
        query: str,
        intent: str | None,
        scored_items: list[tuple[KnowledgeItem, float, int, float]],
    ) -> list[tuple[KnowledgeItem, float, int, float]]:
        query_tokens = _tokenize(query)
        ranked: list[tuple[KnowledgeItem, float, int, float]] = []
        for item, score, keyword_score, semantic_score in scored_items:
            rerank_score = score + self._intent_boost(intent, item)
            item_tokens = _tokenize(f"{item.title} {item.content} {' '.join(item.tags)}")
            if query_tokens and query_tokens <= item_tokens:
                rerank_score += 0.3
            ranked.append((item, rerank_score, keyword_score, semantic_score))
        ranked.sort(key=lambda row: row[1], reverse=True)
        return ranked


@dataclass(frozen=True)
class HttpReranker:
    endpoint: str
    api_key: str | None = None
    provider: str = "generic"
    model: str | None = None
    timeout_seconds: float = 2.0

    def rerank(
        self,
        query: str,
        intent: str | None,
        scored_items: list[tuple[KnowledgeItem, float, int, float]],
    ) -> list[tuple[KnowledgeItem, float, int, float]]:
        provider = self.provider.strip().lower()
        items = [
            {
                "id": item.id,
                "title": item.title,
                "content": item.content,
                "tags": item.tags,
                "base_score": score,
            }
            for item, score, _, _ in scored_items
        ]
        if provider == "cohere":
            payload = {
                "query": query,
                "documents": [f"{row['title']}\n{row['content']}" for row in items],
                "top_n": len(items),
            }
            if self.model:
                payload["model"] = self.model
        else:
            payload = {
                "query": query,
                "intent": intent,
                "items": items,
            }
        req = request.Request(
            self.endpoint,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        if self.api_key:
            header = "Authorization" if provider == "cohere" else "X-API-Key"
            value = self.api_key
            req.add_header(header, value)
        with request.urlopen(req, timeout=self.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
        if not isinstance(body, dict):
            return scored_items
        if provider == "cohere":
            results = body.get("results", [])
            if not isinstance(results, list):
                return scored_items
            score_by_index = {
                int(row.get("index")): float(row.get("relevance_score", 0.0))
                for row in results
                if isinstance(row, dict) and isinstance(row.get("index"), int)
            }
            indexed = list(enumerate(scored_items))
            indexed.sort(key=lambda row: score_by_index.get(row[0], row[1][1]), reverse=True)
            return [row[1] for row in indexed]
        remote_scores = body.get("scores", {})
        if not isinstance(remote_scores, dict):
            return scored_items
        by_id = {item.id: score for item, score, _, _ in scored_items}
        ranked = sorted(
            scored_items,
            key=lambda row: float(remote_scores.get(row[0].id, by_id.get(row[0].id, row[1]))),
            reverse=True,
        )
        return ranked


class MLReranker:
    """scikit-learn LogisticRegression reranker trained on knowledge-base items.

    Training is self-supervised: positive pairs come from each item's own title+tags
    as a query against itself; negatives use the same query against other items.
    At inference time only pure Python is needed (no sklearn import required).

    Model weights are persisted as JSON (no pickle) for portability and security.

    Requires for training: pip install scikit-learn  (requirements-optional.txt)
    """

    _FEATURE_DIM = 5  # [kw_norm, semantic, embedding, intent_boost, token_subset]

    def __init__(self, model_path: Path | None = None) -> None:
        self._coef: list[float] = []
        self._intercept: float = 0.0
        self._trained: bool = False
        self._fallback = HeuristicReranker()
        if model_path is not None:
            self._load(model_path)

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def _extract_features(
        self,
        query: str,
        intent: str | None,
        item: KnowledgeItem,
        keyword_score: int,
        semantic_score: float,
        embedding_score: float,
    ) -> list[float]:
        intent_boost = HeuristicReranker._intent_boost(intent, item)
        query_tokens = _tokenize(query)
        item_tokens = _tokenize(f"{item.title} {item.content} {' '.join(item.tags)}")
        token_subset = 1.0 if query_tokens and query_tokens <= item_tokens else 0.0
        return [
            min(1.0, keyword_score / 5.0),
            semantic_score,
            min(1.0, embedding_score),
            intent_boost,
            token_subset,
        ]

    def _build_training_data(
        self,
        items: list[KnowledgeItem],
        embedding_provider: EmbeddingProvider,
    ) -> tuple[list[list[float]], list[int]]:
        texts = [f"{item.title} {item.content} {' '.join(item.tags)}" for item in items]
        try:
            item_embeddings = embedding_provider.encode(texts)
        except Exception:
            item_embeddings = LocalHashEmbeddingProvider().encode(texts)

        X: list[list[float]] = []
        y: list[int] = []

        for i, item in enumerate(items):
            query = f"{item.title} {' '.join(item.tags[:3])}"
            query_tokens = _tokenize(query)
            try:
                query_emb = embedding_provider.encode([query])[0]
            except Exception:
                query_emb = LocalHashEmbeddingProvider().encode([query])[0]

            intent: str | None = None
            for tag in item.tags:
                for intent_key, tag_set in INTENT_TAG_PRIORS.items():
                    if tag.lower() in tag_set:
                        intent = intent_key
                        break
                if intent:
                    break

            for j, candidate in enumerate(items):
                c_tokens = _tokenize(
                    f"{candidate.title} {candidate.content} {' '.join(candidate.tags)}"
                )
                kw = len(query_tokens & c_tokens)
                sem = (
                    len(query_tokens & c_tokens) / len(query_tokens | c_tokens)
                    if (query_tokens | c_tokens)
                    else 0.0
                )
                emb = _cosine(query_emb, item_embeddings[j])
                X.append(self._extract_features(query, intent, candidate, kw, sem, emb))
                y.append(1 if i == j else 0)

        return X, y

    # ------------------------------------------------------------------
    # Training, persistence, inference
    # ------------------------------------------------------------------

    def train(
        self,
        knowledge_base_items: list[KnowledgeItem],
        embedding_provider: EmbeddingProvider,
    ) -> None:
        """Train the reranker from the knowledge base (self-supervised)."""
        try:
            from sklearn.linear_model import LogisticRegression  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "scikit-learn is required for MLReranker training. "
                "Install it with: pip install scikit-learn"
            ) from exc

        X, y = self._build_training_data(knowledge_base_items, embedding_provider)
        if len(X) < 4 or len(set(y)) < 2:
            return

        lr = LogisticRegression(max_iter=200, random_state=42)
        lr.fit(X, y)
        self._coef = lr.coef_[0].tolist()
        self._intercept = float(lr.intercept_[0])
        self._trained = True

    def save(self, path: Path) -> None:
        """Persist model weights as JSON (no pickle)."""
        if not self._trained:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"coef": self._coef, "intercept": self._intercept}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load(self, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self._coef = [float(v) for v in data["coef"]]
            self._intercept = float(data["intercept"])
            self._trained = True
        except Exception:
            self._trained = False

    def _sigmoid(self, features: list[float]) -> float:
        import math as _math
        logit = self._intercept + sum(c * f for c, f in zip(self._coef, features))
        return 1.0 / (1.0 + _math.exp(-logit))

    def rerank(
        self,
        query: str,
        intent: str | None,
        scored_items: list[tuple[KnowledgeItem, float, int, float]],
    ) -> list[tuple[KnowledgeItem, float, int, float]]:
        if not self._trained or not self._coef:
            return self._fallback.rerank(query, intent, scored_items)

        item_probs: list[tuple[tuple[KnowledgeItem, float, int, float], float]] = []
        for entry in scored_items:
            item, score, keyword_score, semantic_score = entry
            emb_score = max(0.0, score - keyword_score - semantic_score)
            features = self._extract_features(query, intent, item, keyword_score, semantic_score, emb_score)
            prob = self._sigmoid(features)
            item_probs.append((entry, prob))

        item_probs.sort(key=lambda pair: pair[1], reverse=True)
        return [entry for entry, _ in item_probs]


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
    def __init__(
        self,
        items: Iterable[KnowledgeItem],
        embedding_provider: EmbeddingProvider | None = None,
        vector_index: VectorIndex | None = None,
        reranker: Reranker | None = None,
    ):
        self.items = list(items)
        self._item_tokens = [_tokenize(f"{item.title} {item.content} {' '.join(item.tags)}") for item in self.items]
        self.embedding_provider = embedding_provider or LocalHashEmbeddingProvider()
        self.vector_index = vector_index or InMemoryVectorIndex()
        self.reranker = reranker or HeuristicReranker()
        texts = [f"{item.title} {item.content} {' '.join(item.tags)}" for item in self.items]
        try:
            self._item_embeddings = self.embedding_provider.encode(texts)
        except Exception:
            self.embedding_provider = LocalHashEmbeddingProvider()
            self._item_embeddings = self.embedding_provider.encode(texts)

    @classmethod
    def from_json(
        cls,
        path: Path,
        embedding_provider: EmbeddingProvider | None = None,
        vector_index: VectorIndex | None = None,
        reranker: Reranker | None = None,
    ) -> KnowledgeBase:
        data = json.loads(path.read_text(encoding="utf-8"))
        items = [KnowledgeItem(**item) for item in data]
        return cls(items, embedding_provider=embedding_provider, vector_index=vector_index, reranker=reranker)

    def _semantic_score(self, query_tokens: set[str], item_tokens: set[str]) -> float:
        if not query_tokens or not item_tokens:
            return 0.0
        overlap = len(query_tokens & item_tokens)
        union = len(query_tokens | item_tokens)
        return overlap / union if union else 0.0

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.2,
        metadata_filters: dict[str, str] | None = None,
        intent: str | None = None,
    ) -> list[KnowledgeItem]:
        query_tokens = _tokenize(query)
        query_embedding = self.embedding_provider.encode([query])[0]
        indexed = self.vector_index.top_k(query_embedding, self._item_embeddings, k=max(top_k * 4, top_k, 1))
        scored: list[tuple[KnowledgeItem, float, int, float]] = []
        for idx, embedding_score in indexed:
            item = self.items[idx]
            item_tokens = self._item_tokens[idx]
            if metadata_filters:
                source_filter = metadata_filters.get("source")
                tag_filter = metadata_filters.get("tag")
                if source_filter and item.source != source_filter:
                    continue
                if tag_filter and tag_filter not in item.tags:
                    continue
            keyword_score = len(query_tokens & item_tokens)
            semantic_score = self._semantic_score(query_tokens, item_tokens)
            score = keyword_score + semantic_score + embedding_score
            lexical_signal = keyword_score > 0 or semantic_score >= min_score
            embedding_signal = embedding_score >= 0.65
            if intent == "general_query" and not lexical_signal:
                embedding_signal = embedding_score >= 0.8
            has_signal = lexical_signal or embedding_signal
            if has_signal and score >= min_score:
                scored.append((item, score, keyword_score, semantic_score))

        ranked = self.reranker.rerank(query=query, intent=intent, scored_items=scored)
        return [item for item, _, _, _ in ranked[:top_k]]

    def refresh_index(self, new_items: list[KnowledgeItem] | None = None) -> None:
        """Re-encode items with the current embedding provider.

        If ``new_items`` is provided the knowledge base is replaced in-place
        before re-encoding; this is useful after a nightly data refresh.
        Otherwise only the embedding vectors are rebuilt, which is useful after
        swapping to a better embedding provider.
        """
        if new_items is not None:
            self.items = list(new_items)
            self._item_tokens = [
                _tokenize(f"{item.title} {item.content} {' '.join(item.tags)}")
                for item in self.items
            ]
        texts = [
            f"{item.title} {item.content} {' '.join(item.tags)}" for item in self.items
        ]
        try:
            self._item_embeddings = self.embedding_provider.encode(texts)
        except Exception:
            self._item_embeddings = LocalHashEmbeddingProvider().encode(texts)
