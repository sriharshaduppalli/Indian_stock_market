"""Indian-market agent loop on top of ISM (no remote chat model).

Perceive the user ask → route a contract → pick paper-market tools →
compose a grounded answer. Tools never invent P&L and never scrape NSE.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

# Tool ids the composer / enricher already know how to fill.
_PROFILE_TOOLS: dict[str, tuple[str, ...]] = {
    "quote": ("live_quote",),
    "news": ("news_headlines", "live_quote"),
    "sentiment": ("news_sentences", "live_quote"),
    "technical": ("ohlcv_indicators", "levels"),
    "trade_plan": ("live_quote", "ohlcv_indicators", "levels", "news_headlines", "paper_order"),
    "prediction": ("levels", "ohlcv_indicators"),
    "fundamentals": ("fundamentals", "live_quote"),
    "risks": ("ohlcv_indicators", "fundamentals", "news_headlines"),
    "stock_analysis": ("live_quote", "ohlcv_indicators", "fundamentals"),
    "compare": ("live_quote", "fundamentals", "ohlcv_indicators"),
    "sector_screen": ("sector_universe",),
    "literacy": ("education_primer",),
    "compare_concepts": ("education_primer",),
    "derivatives": ("fo_education", "live_quote"),
    "portfolio": ("portfolio_sizing",),
    "calculations": ("deterministic_math",),
    "small_talk": ("conversation",),
    "session": ("session_clock",),
    "corporate_actions": ("corporate_actions",),
}


def plan_agent_tools(profile: str, *, needs_symbol: bool, has_symbol: bool) -> tuple[str, ...]:
    tools = list(_PROFILE_TOOLS.get(profile, ("education_primer",)))
    if needs_symbol and not has_symbol and profile not in {"literacy", "small_talk", "sector_screen"}:
        tools = ["clarifier", *tools]
    return tuple(tools)


def run_ism_agent(
    query: str,
    *,
    market_context: Mapping[str, Any] | None = None,
    conversation_history: Sequence[Mapping[str, Any]] | None = None,
    screen_context: Mapping[str, Any] | None = None,
    intent_hint: str = "",
) -> dict[str, Any]:
    """One agent step: contract + tool plan + composed answer."""
    from .answer_composer import compose_structured_answer
    from .conversation import small_talk_reply
    from .query_contract import resolve_query_contract

    chitchat = small_talk_reply(query)
    contract = resolve_query_contract(
        query,
        intent_hint=intent_hint,
        conversation_history=conversation_history,
        screen_context=screen_context,
    )
    tools = plan_agent_tools(
        contract.profile,
        needs_symbol=contract.needs_symbol,
        has_symbol=bool(contract.slots.symbol),
    )
    if chitchat and contract.profile == "small_talk":
        answer = chitchat
    else:
        ctx = dict(market_context or {})
        if contract.slots.symbol:
            ctx.setdefault("symbol", contract.slots.symbol)
        elif "symbol" in ctx and not contract.slots.symbol:
            ctx.pop("symbol", None)
        if "session_clock" in tools:
            try:
                from .agent_tools import session_snapshot

                ctx.setdefault("session", session_snapshot())
            except Exception:
                pass
        if "corporate_actions" in tools and contract.slots.symbol:
            try:
                from .agent_tools import load_corporate_actions

                ctx.setdefault(
                    "corporate_actions",
                    load_corporate_actions(contract.slots.symbol, hydrate=True),
                )
            except Exception:
                pass
        answer = compose_structured_answer(
            query=contract.resolved_query or query,
            intent=contract.ism_intent,
            market_context=ctx,
            context_lines=[],
            profile=contract.profile,
        ) or (contract.clarifier or "")
    return {
        "answer": answer,
        "profile": contract.profile,
        "symbol": contract.slots.symbol,
        "tools": list(tools),
        "follow_up": contract.slots.follow_up,
        "language": contract.language,
        "agent": "ism-indian-market",
    }
