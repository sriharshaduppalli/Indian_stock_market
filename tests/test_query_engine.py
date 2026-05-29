from indian_stock_llm.query_engine import StockMarketAssistant


def test_intent_detection_for_fundamentals() -> None:
    assistant = StockMarketAssistant()
    response = assistant.ask("How should I think about PE and valuation for Indian IT stocks?")

    assert response.intent == "fundamentals"
    assert "Relevant market context" in response.answer


def test_fallback_when_context_not_found() -> None:
    assistant = StockMarketAssistant()
    response = assistant.ask("Explain lunar geology impact on equities")

    assert response.intent == "general_query"
    assert "could not find enough domain context" in response.answer
