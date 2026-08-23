"""Query contract: one brain for intent, follow-ups, and answer shape."""
from __future__ import annotations

from indian_stock_llm.answer_composer import resolve_stock_response_profile
from indian_stock_llm.prediction import PredictionEngine
from indian_stock_llm.query_contract import resolve_query_contract


def test_chip_queries_keep_distinct_profiles():
    cases = {
        "Should I buy RELIANCE?": ("trade_plan", "BUY_SELL"),
        "Predict RELIANCE price": ("prediction", "PREDICT"),
        "Technical analysis of RELIANCE": ("technical", "TECHNICAL"),
        "Latest news on RELIANCE": ("news", "NEWS"),
        "RELIANCE market sentiment": ("sentiment", "SENTIMENT"),
        "What is the price of RELIANCE?": ("quote", "QUOTE"),
        "Is RELIANCE overvalued?": ("fundamentals", "FUNDAMENTAL"),
        "What is RSI?": ("literacy", "EDUCATIONAL"),
        "calculate CAGR for 100000 to 180000 in 3 years": ("calculations", "CALCULATION"),
    }
    for query, (profile, groq_intent) in cases.items():
        contract = resolve_query_contract(query)
        assert contract.profile == profile, (query, contract.profile)
        assert contract.groq_intent == groq_intent, (query, contract.groq_intent)
        assert resolve_stock_response_profile(query, "general_query") == profile


def test_followup_reuses_last_symbol_and_changes_shape():
    history = [
        {"role": "user", "content": "Technical analysis of RELIANCE"},
        {"role": "assistant", "content": "**RELIANCE** — RSI 55"},
    ]
    contract = resolve_query_contract("what about sentiment?", conversation_history=history)
    assert contract.slots.follow_up is True
    assert "RELIANCE" in contract.resolved_query.upper()
    assert contract.profile == "sentiment"
    assert contract.clarifier is None


def test_same_for_other_symbol_keeps_prior_shape():
    history = [
        {"role": "user", "content": "Should I buy TCS?"},
        {"role": "assistant", "content": "HOLD TCS"},
    ]
    contract = resolve_query_contract("same for INFY", conversation_history=history)
    assert "INFY" in contract.resolved_query.upper()
    assert contract.profile == "trade_plan"


def test_trade_plan_format_gives_buy_sell_advice():
    text = resolve_query_contract("Should I buy RELIANCE?").format_instructions
    assert "BUY / SELL / HOLD" in text
    assert "NEVER say BUY" not in text


def test_quote_format_does_not_force_a_trade_call():
    text = resolve_query_contract("What is the price of RELIANCE?").format_instructions
    assert "quote snapshot" in text.lower()
    assert "BUY / SELL / HOLD" not in text


def test_prediction_engine_emits_advice_and_bands():
    engine = PredictionEngine()
    signals = engine.predict(
        context_items=[],
        p0_math={
            "price": 1000.0,
            "atr_14": 20.0,
            "wilder_rsi_14": 55,
            "supertrend": {"direction": "bullish"},
            "macd": {"histogram": 1.2},
            "vs_nifty": {"rs_20d": 1.04},
            "trade_plan": {"action": "BUY"},
            "levels": {"support": 960.0, "resistance": 1040.0},
        },
    )
    assert signals.advice in {"BUY", "SELL", "HOLD"}
    assert signals.scenarios.get("swing_band")
    assert signals.intraday.direction in {"bullish", "bearish", "neutral"}


def test_event_window_blocks_buy_advice():
    engine = PredictionEngine()
    signals = engine.predict(
        context_items=[],
        p0_math={
            "price": 1000.0,
            "atr_14": 20.0,
            "event_note": "earnings date tomorrow",
            "trade_plan": {"action": "BUY"},
            "supertrend": {"direction": "bullish"},
        },
    )
    assert signals.event_blackout
    assert signals.advice == "HOLD"


def test_telugu_script_routes_like_english():
    buy = resolve_query_contract("రిలయన్స్ కొనాలా?")
    assert buy.language == "te"
    assert buy.profile == "trade_plan"
    assert buy.slots.symbol == "RELIANCE"

    quote = resolve_query_contract("ITC ధర ఎంత?")
    assert quote.profile == "quote"
    assert quote.slots.symbol == "ITC"

    literacy = resolve_query_contract("RSI అంటే ఏమిటి?")
    assert literacy.profile == "literacy"


def test_tenglish_routes_like_english():
    quote = resolve_query_contract("ITC dhara entha undi?")
    assert quote.language == "te-en"
    assert quote.profile == "quote"
    assert quote.slots.symbol == "ITC"

    buy = resolve_query_contract("Reliance konala?")
    assert buy.profile == "trade_plan"
    assert buy.slots.symbol == "RELIANCE"

    predict = resolve_query_contract("HDFCBANK ela untundi?")
    assert predict.profile == "prediction"


def test_telugu_answer_gets_summary():
    from indian_stock_llm.query_language import localize_assistant_answer

    localized = localize_assistant_answer(
        "రిలయన్స్ కొనాలా?",
        "**Direct answer:** HOLD / wait — no clear edge yet\n**Why:** RSI 52 is mid-range",
    )
    assert "తెలుగు సారాంశం" in localized
    assert "HOLD" in localized
    assert "నేరుగా సమాధానం" in localized
    assert "కారణం" in localized
    assert "వేచి" in localized
    assert "మధ్యస్థం" in localized
    assert localize_assistant_answer("Should I buy RELIANCE?", "**Direct answer:** HOLD") == (
        "**Direct answer:** HOLD"
    )


def test_telugu_literacy_uses_telugu_primer():
    from indian_stock_llm.query_language import localize_assistant_answer

    localized = localize_assistant_answer(
        "RSI అంటే ఏమిటి?",
        "**RSI (Relative Strength Index)**\n\nRSI is a momentum oscillator (0–100).",
    )
    assert "అంటే ఏమిటి" in localized
    assert "overbought" in localized
    assert "పెట్టుబడి సలహా కాదు" in localized


def test_tenglish_answer_is_also_telugu():
    from indian_stock_llm.query_language import localize_assistant_answer

    localized = localize_assistant_answer(
        "ITC dhara entha undi?",
        "**ITC** — live quote\n\n• Last: \u20b9412\n• Support / Resistance: 400 / 430",
    )
    assert "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02" in localized
    assert "\u20b9412" in localized
    assert "\u0c1a\u0c3f\u0c35\u0c30\u0c3f" in localized


def test_indic_trans_shields_rupee_and_tickers():
    from indian_stock_llm.indic_trans import shield_protected_tokens, unshield_protected_tokens

    shielded, tokens = shield_protected_tokens(
        "RELIANCE last is ₹412 and RSI 52 is mid-range"
    )
    assert "RELIANCE" in tokens
    assert "₹412" in tokens
    assert "RSI" in tokens
    assert "[[T0]]" in shielded
    assert unshield_protected_tokens(shielded, tokens).startswith("RELIANCE")


def test_indic_trans_leftover_english_uses_worker(monkeypatch):
    from indian_stock_llm import indic_trans
    from indian_stock_llm.query_language import localize_assistant_answer

    monkeypatch.setenv("INDIC_TRANS_URL", "http://127.0.0.1:8101")
    monkeypatch.setenv("INDIC_TRANS_ENABLED", "true")

    def fake_batch(texts):
        return ["ఈ వాక్యం తెలు�ఈ వాక్యం తెలుగులో ఉంది" for _ in texts]

    monkeypatch.setattr(indic_trans, "translate_english_batch", fake_batch)
    localized = localize_assistant_answer(
        "రిలయన్స్ ఎలా ఉంది?",
        "**Direct answer:** HOLD / wait — no clear edge yet\n"
        "The tape looks mixed today and you should wait for confirmation at support.",
    )
    assert "తెలుగు సారాంశం" in localized
    assert "ఈ వాక్యం తెలుగులో ఉంది" in localized
    assert "నేరుగా సమాధానం" in localized


def test_indic_trans_off_keeps_phrase_table(monkeypatch):
    from indian_stock_llm.query_language import localize_assistant_answer

    monkeypatch.delenv("INDIC_TRANS_URL", raising=False)
    monkeypatch.delenv("BYSEL_INDIC_TRANS_URL", raising=False)
    monkeypatch.delenv("INDIC_TRANS_ENABLED", raising=False)
    localized = localize_assistant_answer(
        "రిలయన్స్ కొనాలా?",
        "**Direct answer:** HOLD / wait — no clear edge yet\n**Why:** RSI 52 is mid-range",
    )
    assert "నేరుగా సమాధానం" in localized
    assert "మధ్యస్థం" in localized
