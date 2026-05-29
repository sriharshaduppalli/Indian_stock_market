from pathlib import Path

from indian_stock_llm.config import AssistantConfig
from indian_stock_llm.query_engine import StockMarketAssistant


def _assistant_with_repo_kb(tmp_path: Path) -> StockMarketAssistant:
    repo_root = Path(__file__).resolve().parents[1]
    config = AssistantConfig(
        knowledge_base_path=repo_root / "data" / "sample_knowledge.json",
        feedback_log_path=tmp_path / "feedback.log",
        latency_mode="fast",
    )
    return StockMarketAssistant(config=config)


def test_intent_detection_for_fundamentals(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("How should I think about PE and valuation for Indian IT stocks?")

    assert response.intent == "fundamentals"
    assert "Relevant market context" in response.answer


def test_fallback_when_context_not_found(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("Explain lunar geology impact on equities")

    assert response.intent == "general_query"
    assert "could not find enough domain context" in response.answer


def test_prediction_intent_and_daily_learning_hook(tmp_path: Path) -> None:
    feedback_log = tmp_path / "feedback.log"
    repo_root = Path(__file__).resolve().parents[1]
    config = AssistantConfig(
        knowledge_base_path=repo_root / "data" / "sample_knowledge.json",
        feedback_log_path=feedback_log,
        latency_mode="fast",
    )
    assistant = StockMarketAssistant(config=config)
    response = assistant.ask("Predict next week outlook for Indian IT stocks based on live news")

    assert response.intent == "prediction"
    assert "Prediction factors considered" in response.answer
    assert "Daily learning loop enabled" in response.answer
    assert feedback_log.exists()


def test_market_calculations_intent(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("How do I calculate CAGR and volatility for Indian stocks?")

    assert response.intent == "market_calculations"
    assert "Relevant market context" in response.answer


def test_stock_analysis_intent(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("Provide technical analysis for NSE listed banking stocks")

    assert response.intent == "stock_analysis"
    assert "Relevant market context" in response.answer


def test_calculation_query_has_deterministic_result(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("Calculate CAGR for start 100 end 133.1 over 3 years")

    assert response.intent == "market_calculations"
    assert "Deterministic calculation: CAGR is" in response.answer
    assert response.confidence > 0.0
    assert response.citations


def test_calculation_query_requires_valid_inputs(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("Calculate CAGR from start 100 end 133 over 0 years")

    assert response.intent == "market_calculations"
    assert "Deterministic calculation unavailable" in response.answer


def test_return_calculation_rejects_negative_sell_value(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("Calculate return from buy 100 sell -50")

    assert response.intent == "market_calculations"
    assert "Deterministic calculation unavailable" in response.answer


def test_prediction_contains_policy_disclaimer(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    response = assistant.ask("Predict next week outlook for NSE banking stocks")

    assert response.intent == "prediction"
    assert "not investment advice" in response.disclaimer
    assert response.safe_for_trading_advice is False


def test_query_method_returns_api_schema(tmp_path: Path) -> None:
    assistant = _assistant_with_repo_kb(tmp_path)
    payload = assistant.query("What is SEBI role in Indian stock market?")

    assert payload["intent"] in {"general_query", "events_news", "fundamentals"}
    assert isinstance(payload["answer"], str)
    assert isinstance(payload["confidence"], float)
    assert isinstance(payload["citations"], list)
    assert "disclaimer" in payload
