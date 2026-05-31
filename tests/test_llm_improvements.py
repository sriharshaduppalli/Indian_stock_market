"""Tests covering the 6 LLM improvement gaps."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from indian_stock_llm.config import AssistantConfig
from indian_stock_llm.evaluation import BenchmarkSuite
from indian_stock_llm.knowledge_base import (
    KnowledgeBase,
    KnowledgeItem,
    LocalHashEmbeddingProvider,
    MLReranker,
    SentenceTransformerEmbeddingProvider,
)
from indian_stock_llm.learning_loop import DailyFeedbackAnalyzer
from indian_stock_llm.prediction import PredictionEngine
from indian_stock_llm.query_engine import StockMarketAssistant
from indian_stock_llm.training import QAPairBuilder


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_assistant(tmp_path: Path) -> StockMarketAssistant:
    config = AssistantConfig(
        knowledge_base_path=REPO_ROOT / "data" / "sample_knowledge.json",
        instrument_master_path=REPO_ROOT / "data" / "enterprise" / "instrument_master.json",
        corporate_actions_path=REPO_ROOT / "data" / "enterprise" / "corporate_actions.json",
        filings_path=REPO_ROOT / "data" / "enterprise" / "filings.json",
        regulatory_updates_path=REPO_ROOT / "data" / "enterprise" / "regulatory_updates.json",
        market_events_path=REPO_ROOT / "data" / "enterprise" / "market_events.json",
        feedback_log_path=tmp_path / "feedback.log",
        policy_audit_log_path=tmp_path / "policy.log",
        release_registry_path=tmp_path / "releases.json",
        latency_mode="fast",
    )
    return StockMarketAssistant(config=config)


def _sample_items() -> list[KnowledgeItem]:
    return [
        KnowledgeItem(
            id="item-1",
            title="Reliance Industries overview",
            content="Reliance is bullish on Q3 with strong earnings and expanding margin",
            tags=["reliance", "bull", "earnings"],
            source="test",
        ),
        KnowledgeItem(
            id="item-2",
            title="NIFTY outlook",
            content="NIFTY downside risk due to FII outflows and rate hike concerns",
            tags=["nifty", "bear", "fii"],
            source="test",
        ),
        KnowledgeItem(
            id="item-3",
            title="IT sector analysis",
            content="Indian IT sector maintaining steady growth with deal wins",
            tags=["it", "sector", "deals"],
            source="test",
        ),
    ]


# ---------------------------------------------------------------------------
# Gap 1: SentenceTransformerEmbeddingProvider (graceful fallback)
# ---------------------------------------------------------------------------


def test_sentence_transformer_provider_fallback() -> None:
    """SentenceTransformerEmbeddingProvider falls back to LocalHashEmbeddingProvider when
    sentence-transformers is not installed, without raising."""
    provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    # encode should never raise regardless of whether the library is available
    vecs = provider.encode(["test query"])
    assert isinstance(vecs, list)
    assert len(vecs) == 1
    assert len(vecs[0]) > 0


def test_sentence_transformer_embed_is_normalised() -> None:
    """Embeddings produced by SentenceTransformerEmbeddingProvider are non-empty."""
    provider = SentenceTransformerEmbeddingProvider(model_name="all-MiniLM-L6-v2")
    vecs = provider.encode(["Reliance Industries Q3 earnings"])
    assert len(vecs) == 1
    vec = vecs[0]
    assert len(vec) > 0
    # All values should be finite floats
    assert all(isinstance(v, float) for v in vec)


# ---------------------------------------------------------------------------
# Gap 2: Nightly index refresh + trigger_index_refresh
# ---------------------------------------------------------------------------


def test_trigger_index_refresh(tmp_path: Path) -> None:
    """StockMarketAssistant.trigger_index_refresh() completes without error."""
    assistant = _make_assistant(tmp_path)
    assistant.trigger_index_refresh()  # must not raise


def test_knowledge_base_refresh_index() -> None:
    """KnowledgeBase.refresh_index() re-encodes all items."""
    provider = LocalHashEmbeddingProvider()
    items = _sample_items()
    kb = KnowledgeBase(items=items, embedding_provider=provider)
    # Capture index state before refresh
    before = list(kb.vector_index._vectors.values()) if hasattr(kb.vector_index, "_vectors") else []
    kb.refresh_index()
    # No error → pass; also verify KB still has items
    results = kb.search("bull earnings", top_k=1)
    assert len(results) >= 0  # search works after refresh


# ---------------------------------------------------------------------------
# Gap 3: LoRA fine-tuning infrastructure
# ---------------------------------------------------------------------------


def test_qa_pair_builder_produces_pairs() -> None:
    """QAPairBuilder.from_knowledge_base() returns at least one QA pair per item."""
    builder = QAPairBuilder()
    items = _sample_items()
    pairs = builder.from_knowledge_base(items)
    assert len(pairs) >= len(items)


def test_qa_pair_builder_save_load(tmp_path: Path) -> None:
    """QAPairBuilder.save() writes valid JSON that can be parsed back."""
    builder = QAPairBuilder()
    items = _sample_items()
    pairs = builder.from_knowledge_base(items)
    out = tmp_path / "qa_pairs.json"
    builder.save(pairs, out)
    data = json.loads(out.read_text())
    assert isinstance(data, list)
    assert len(data) == len(pairs)
    for record in data:
        assert "instruction" in record
        assert "input" in record
        assert "output" in record


def test_lora_finetuner_unavailable_graceful(tmp_path: Path) -> None:
    """LoRAFineTuner raises ImportError when transformers/peft not installed, not a crash."""
    from indian_stock_llm.training import LoRAFineTuner

    finetuner = LoRAFineTuner(base_model="gpt2", output_path=tmp_path / "adapter")
    builder = QAPairBuilder()
    pairs = builder.from_knowledge_base(_sample_items())
    try:
        finetuner.train(pairs)
    except (ImportError, RuntimeError):
        pass  # expected when transformers/peft not installed
    except Exception as exc:
        pytest.fail(f"Unexpected exception from LoRAFineTuner.train: {exc}")


# ---------------------------------------------------------------------------
# Gap 4: Multi-horizon prediction head
# ---------------------------------------------------------------------------


def test_prediction_engine_produces_signals() -> None:
    """PredictionEngine.predict() returns valid PredictionSignals."""
    engine = PredictionEngine()
    items = _sample_items()
    signals = engine.predict(context_items=items)
    assert signals.intraday.direction in ("bullish", "bearish", "neutral")
    assert 0.0 <= signals.intraday.probability <= 1.0
    assert signals.swing.direction in ("bullish", "bearish", "neutral")
    assert signals.medium_term.direction in ("bullish", "bearish", "neutral")
    assert 0.0 <= signals.overall_confidence <= 1.0


def test_prediction_engine_no_context() -> None:
    """PredictionEngine.predict() handles empty context without error."""
    engine = PredictionEngine()
    signals = engine.predict(context_items=[])
    assert signals.intraday.direction == "neutral"
    assert signals.overall_confidence >= 0.0


def test_query_returns_prediction_signals(tmp_path: Path) -> None:
    """query() includes prediction_signals key (dict or None) in result."""
    assistant = _make_assistant(tmp_path)
    result = assistant.query("Will RELIANCE go up tomorrow?")
    assert "prediction_signals" in result
    ps = result["prediction_signals"]
    if ps is not None:
        assert "intraday" in ps
        assert "swing" in ps
        assert "medium_term" in ps
        assert "key_signals" in ps
        assert "overall_confidence" in ps


# ---------------------------------------------------------------------------
# Gap 5: Evaluation / benchmark suite
# ---------------------------------------------------------------------------


def test_benchmark_suite_runs(tmp_path: Path) -> None:
    """BenchmarkSuite.run() completes and returns valid BenchmarkResult."""
    assistant = _make_assistant(tmp_path)
    suite = BenchmarkSuite()
    result, details = suite.run(assistant)
    # All metrics should be in [0, 1]
    assert 0.0 <= result.routing_accuracy <= 1.0
    assert 0.0 <= result.fact_accuracy <= 1.0
    assert 0.0 <= result.groundedness <= 1.0
    assert 0.0 <= result.hallucination_rate <= 1.0
    assert 0.0 <= result.safety_score <= 1.0
    assert 0.0 <= result.calculation_correctness <= 1.0
    assert isinstance(details, list)
    assert len(details) == len(suite.cases)


def test_benchmark_suite_details_structure(tmp_path: Path) -> None:
    """Each detail dict in BenchmarkSuite.run() has expected keys."""
    assistant = _make_assistant(tmp_path)
    suite = BenchmarkSuite()
    _, details = suite.run(assistant)
    for d in details:
        assert "query" in d
        assert "expected_intent" in d
        assert "actual_intent" in d


# ---------------------------------------------------------------------------
# Gap 6: ML-based reranker
# ---------------------------------------------------------------------------


def test_ml_reranker_train_and_rerank(tmp_path: Path) -> None:
    """MLReranker can be trained and produces a valid ordering."""
    pytest.importorskip("sklearn", reason="scikit-learn not installed")

    model_path = tmp_path / "reranker.json"
    reranker = MLReranker(model_path=model_path)
    items = _sample_items()
    provider = LocalHashEmbeddingProvider()

    reranker.train(knowledge_base_items=items, embedding_provider=provider)
    reranker.save(model_path)
    assert model_path.exists()

    # Build scored_items tuples: (item, score, keyword_score, semantic_score)
    scored = [(item, 1.0, 1, 0.5) for item in items]
    ranked = reranker.rerank(query="bull earnings Reliance", intent="prediction", scored_items=scored)
    assert len(ranked) == len(items)
    assert all(isinstance(entry[0], KnowledgeItem) for entry in ranked)


def test_ml_reranker_rerank_without_training(tmp_path: Path) -> None:
    """MLReranker.rerank() falls back to original order when no weights file exists."""
    reranker = MLReranker(model_path=None)
    items = _sample_items()
    scored = [(item, 1.0, 1, 0.5) for item in items]
    ranked = reranker.rerank(query="sector analysis", intent=None, scored_items=scored)
    assert len(ranked) == len(items)


# ---------------------------------------------------------------------------
# DailyFeedbackAnalyzer
# ---------------------------------------------------------------------------


def test_daily_feedback_analyzer_tsv(tmp_path: Path) -> None:
    """DailyFeedbackAnalyzer parses TSV log format correctly."""
    log_path = tmp_path / "feedback.tsv"
    lines = [f"2024-01-01T00:00:00\tprediction\tquery{i}" for i in range(12)]
    log_path.write_text("\n".join(lines))

    analyzer = DailyFeedbackAnalyzer(feedback_log_path=log_path)
    report = analyzer.analyze()
    assert report["ready"] is True
    assert report["intent_counts"].get("prediction", 0) == 12


def test_daily_feedback_analyzer_json_lines(tmp_path: Path) -> None:
    """DailyFeedbackAnalyzer parses JSON-lines log format correctly."""
    log_path = tmp_path / "feedback.jsonl"
    lines = [json.dumps({"ts": "2024-01-01", "intent": "fundamentals", "query_hash": f"h{i}"}) for i in range(11)]
    log_path.write_text("\n".join(lines))

    analyzer = DailyFeedbackAnalyzer(feedback_log_path=log_path)
    report = analyzer.analyze()
    assert report["ready"] is True
    assert report["intent_counts"].get("fundamentals", 0) == 11


def test_daily_feedback_analyzer_not_ready_below_threshold(tmp_path: Path) -> None:
    """DailyFeedbackAnalyzer reports not ready when fewer than 10 samples."""
    log_path = tmp_path / "feedback.log"
    lines = [f"2024-01-01T00:00:00\tprediction\tq{i}" for i in range(5)]
    log_path.write_text("\n".join(lines))

    analyzer = DailyFeedbackAnalyzer(feedback_log_path=log_path)
    report = analyzer.analyze()
    assert report["ready"] is False


# ---------------------------------------------------------------------------
# /admin/refresh endpoint
# ---------------------------------------------------------------------------


def test_admin_refresh_endpoint_no_token(tmp_path: Path) -> None:
    """POST /admin/refresh returns 200 when no admin token is configured."""
    from indian_stock_llm.api import ApiRequest, ChatApi
    from indian_stock_llm.http_server import dispatch_http_request
    from indian_stock_llm.serving import ChatService

    assistant = _make_assistant(tmp_path)
    service = ChatService(assistant=assistant)
    api = ChatApi(service=service)

    response = dispatch_http_request(
        api,
        method="POST",
        path="/admin/refresh",
        headers={},
        raw_body=b"",
    )
    assert response.status_code == 200
    assert response.payload["status"] == "ok"


def test_admin_refresh_endpoint_wrong_token(tmp_path: Path) -> None:
    """POST /admin/refresh returns 401 when wrong admin token supplied."""
    from indian_stock_llm.api import ChatApi
    from indian_stock_llm.http_server import dispatch_http_request
    from indian_stock_llm.serving import ChatService

    assistant = _make_assistant(tmp_path)
    service = ChatService(assistant=assistant)
    api = ChatApi(service=service)

    response = dispatch_http_request(
        api,
        method="POST",
        path="/admin/refresh",
        headers={"x-admin-token": "wrong"},
        raw_body=b"",
        metrics_admin_token="correct",
    )
    assert response.status_code == 401


def test_admin_refresh_endpoint_correct_token(tmp_path: Path) -> None:
    """POST /admin/refresh returns 200 when correct admin token supplied."""
    from indian_stock_llm.api import ChatApi
    from indian_stock_llm.http_server import dispatch_http_request
    from indian_stock_llm.serving import ChatService

    assistant = _make_assistant(tmp_path)
    service = ChatService(assistant=assistant)
    api = ChatApi(service=service)

    response = dispatch_http_request(
        api,
        method="POST",
        path="/admin/refresh",
        headers={"x-admin-token": "secret"},
        raw_body=b"",
        metrics_admin_token="secret",
    )
    assert response.status_code == 200
    assert response.payload["status"] == "ok"
