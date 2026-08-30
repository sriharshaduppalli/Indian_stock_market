from indian_stock_llm.language_graph import build_language_graph
from indian_stock_llm.query_contract import resolve_query_contract


def test_habit_teach_query_is_literacy():
    contract = resolve_query_contract(
        "Teach the NSE opening range and first-hour volatility as a paper habit."
    )
    assert contract.profile == "literacy"
    assert contract.ism_intent == "market_literacy"
    assert "session clock" in contract.retrieval_query


def test_named_buy_ask_is_still_trade_plan():
    contract = resolve_query_contract("Should I buy RELIANCE?")
    assert contract.profile == "trade_plan"


def test_graph_expands_retrieval_terms():
    graph = build_language_graph("What is midday chop and revenge trading on NSE?")
    assert "midday_chop" in graph.concepts
    assert "stand aside" in graph.retrieval_query
