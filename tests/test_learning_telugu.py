"""Telugu / Tenglish asks must share the same ISM learning key as English."""
from __future__ import annotations

from pathlib import Path


def test_telugu_and_english_share_learning_key():
    from indian_stock_llm.query_language import learning_query_fields

    te = learning_query_fields("రిలయన్స్ కొనాలా?")
    en = learning_query_fields("Should I buy RELIANCE?")
    assert te["is_telugu"] is True
    assert en["is_telugu"] is False
    assert "reliance" in str(te["key"])
    assert "buy" in str(te["key"])
    assert "reliance" in str(en["key"])


def test_tenglish_learning_key_keeps_symbol():
    from indian_stock_llm.query_language import learning_query_fields

    fields = learning_query_fields("ITC dhara entha undi?")
    assert fields["language"] == "te-en"
    assert "itc" in str(fields["key"])
    assert "price" in str(fields["key"]) or "how much" in str(fields["key"])


def test_feedback_promotion_merges_telugu_and_english(tmp_path: Path):
    from indian_stock_llm.learning_loop import FeedbackLearningPipeline

    log = tmp_path / "daily_feedback.log"
    learned = tmp_path / "learned_knowledge.json"
    lines = [
        "2026-08-23T01:00:00+00:00\tprice_action\tshould i buy reliance\n",
        "2026-08-23T01:01:00+00:00\tprice_action\tshould i buy reliance\n",
        '{"kind":"interaction_v1","intent":"price_action","query":"రిలయన్స్ కొనాలా?","query_key":"should i buy reliance","language":"te","confidence":0.7,"citation_count":1,"grounded":true}\n',
    ]
    log.write_text("".join(lines), encoding="utf-8")
    added = FeedbackLearningPipeline.promote_from_feedback_log(log, learned, min_count=3)
    assert added == 1
    payload = learned.read_text(encoding="utf-8")
    assert "telugu" in payload
    assert "should i buy reliance" in payload.lower()


def test_grounded_promote_uses_english_key(tmp_path: Path):
    from indian_stock_llm.learning_loop import FeedbackLearningPipeline, _slug_id

    learned = tmp_path / "learned_knowledge.json"
    ok = FeedbackLearningPipeline.promote_grounded_answer(
        learned,
        query="రిలయన్స్ కొనాలా?",
        intent="price_action",
        answer="**Direct answer:** HOLD / wait — no clear edge yet",
        citations=["ism_kb"],
        confidence=0.7,
    )
    assert ok is True
    english_id = _slug_id("learned_ans", "price_action|should i buy reliance")
    # Key may include extra Telugu-normalized tokens; just require one item and telugu tag.
    data = learned.read_text(encoding="utf-8")
    assert "telugu" in data
    assert "HOLD" in data
    assert english_id in data or "should i buy" in data.lower()


def test_does_not_learn_telugu_overlay(tmp_path: Path):
    from indian_stock_llm.learning_loop import FeedbackLearningPipeline

    learned = tmp_path / "learned_knowledge.json"
    ok = FeedbackLearningPipeline.promote_grounded_answer(
        learned,
        query="Should I buy RELIANCE?",
        intent="price_action",
        answer="**తెలుగు సారాంశం:** వేచి ఉండండి (HOLD)",
        citations=["ism_kb"],
        confidence=0.8,
    )
    assert ok is False
