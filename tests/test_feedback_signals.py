"""Thumbs must steer ISM retrieval and must not promote downvoted topics."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def test_three_thumbs_up_promote_coaching(tmp_path: Path):
    from indian_stock_llm.learning_loop import FeedbackLearningPipeline

    log = tmp_path / "daily_feedback.log"
    learned = tmp_path / "learned_knowledge.json"
    line = (
        '{"kind":"thumbs_v1","intent":"price_action","helpful":true,'
        '"query_key":"should i buy reliance","query":"Should I buy RELIANCE?"}\n'
    )
    log.write_text(line * 3, encoding="utf-8")
    added = FeedbackLearningPipeline.promote_from_feedback_log(log, learned, min_count=3)
    assert added == 1
    payload = learned.read_text(encoding="utf-8")
    assert "should i buy reliance" in payload.lower()


def test_thumbs_down_vetoes_promotion(tmp_path: Path):
    from indian_stock_llm.learning_loop import FeedbackLearningPipeline

    log = tmp_path / "daily_feedback.log"
    learned = tmp_path / "learned_knowledge.json"
    lines = [
        '{"kind":"thumbs_v1","intent":"events_news","helpful":false,"query_key":"dividend date of reliance"}\n',
        '{"kind":"thumbs_v1","intent":"events_news","helpful":false,"query_key":"dividend date of reliance"}\n',
        '{"kind":"thumbs_v1","intent":"events_news","helpful":false,"query_key":"dividend date of reliance"}\n',
        "2026-08-30T01:00:00+00:00\tevents_news\tdividend date of reliance\n",
        "2026-08-30T01:01:00+00:00\tevents_news\tdividend date of reliance\n",
        "2026-08-30T01:02:00+00:00\tevents_news\tdividend date of reliance\n",
    ]
    log.write_text("".join(lines), encoding="utf-8")
    added = FeedbackLearningPipeline.promote_from_feedback_log(log, learned, min_count=3)
    assert added == 0
    assert not learned.exists() or "dividend date of reliance" not in learned.read_text(encoding="utf-8")


def test_rerank_drops_learned_note_after_downvote():
    from indian_stock_llm.learning_loop import FeedbackSignalStore

    store = FeedbackSignalStore()
    store.record("should i buy reliance", "price_action", helpful=False)
    items = [
        SimpleNamespace(
            title="Frequent topic coaching: should i buy reliance",
            content="Learners often ask about should i buy reliance",
            source="learned_feedback_v1",
        ),
        SimpleNamespace(title="RSI basics", content="Wilder RSI", source="builtin"),
    ]
    out = store.rerank_items("Should I buy RELIANCE?", items)
    assert [item.title for item in out] == ["RSI basics"]


def test_rerank_boosts_matching_note_after_upvote():
    from indian_stock_llm.learning_loop import FeedbackSignalStore

    store = FeedbackSignalStore()
    store.record("should i buy reliance", "price_action", helpful=True)
    items = [
        SimpleNamespace(title="RSI basics", content="Wilder RSI", source="builtin"),
        SimpleNamespace(
            title="Frequent topic coaching: should i buy reliance",
            content="Learners often ask about should i buy reliance",
            source="learned_feedback_v1",
        ),
    ]
    out = store.rerank_items("Should I buy RELIANCE?", items)
    assert out[0].source == "learned_feedback_v1"
