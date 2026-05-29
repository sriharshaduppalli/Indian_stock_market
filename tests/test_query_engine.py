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
