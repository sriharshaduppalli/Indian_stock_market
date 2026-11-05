"""Structured answer composer for the local Indian Stock LLM tier.

Turns live market context + grounding snippets into actionable educational
answers designed to match/beat paid-LLM usefulness on Indian equities.
"""
from __future__ import annotations

import re
from typing import Any

# Curated liquid universes — used when screener/KB retrieval is noisy.
_SECTOR_UNIVERSE: dict[str, list[str]] = {
    "defence": ["HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "DATAPATTNS"],
    "defense": ["HAL", "BEL", "BDL", "MAZDOCK", "COCHINSHIP", "GRSE", "DATAPATTNS"],
    "it": ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "LTIM", "PERSISTENT"],
    "banking": ["HDFCBANK", "ICICIBANK", "SBIN", "AXISBANK", "KOTAKBANK", "INDUSINDBK"],
    "pharma": ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN", "BIOCON"],
    "fmcg": ["HINDUNILVR", "ITC", "NESTLEIND", "BRITANNIA", "DABUR", "MARICO"],
    "auto": ["MARUTI", "TMPV", "TMCV", "M&M", "BAJAJ-AUTO", "HEROMOTOCO", "EICHERMOT"],
    "energy": ["RELIANCE", "NTPC", "POWERGRID", "ONGC", "IOC", "BPCL"],
    "metal": ["TATASTEEL", "HINDALCO", "JSWSTEEL", "VEDL", "SAIL"],
    "realty": ["DLF", "GODREJPROP", "OBEROIRLTY", "PRESTIGE", "LODHA"],
    "psu": ["SBIN", "NTPC", "ONGC", "BEL", "HAL", "COALINDIA"],
    "railway": ["IRCTC", "IRFC", "RVNL", "IRCON", "RAILTEL"],
    "cement": ["ULTRACEMCO", "SHREECEM", "AMBUJACEM", "ACC"],
    "infra": ["LT", "ADANIPORTS", "ADANIENT", "IRCON", "RVNL"],
}


def _num(value: Any) -> float | None:
    """Parse enrich values like '58.1 (neutral)', '1,319.00', '22.4%'."""
    if value is None or value == "" or value == "N/A" or value == "n/a":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, dict):
        for key in ("value", "rsi", "pe", "last", "price", "close"):
            if key in value:
                return _num(value.get(key))
        return None
    text = str(value).strip()
    if not text:
        return None
    # Strip currency / percent / commas, then take leading number.
    text = text.replace(",", "").replace("₹", "").replace("%", "")
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _fmt(value: Any, digits: int = 2) -> str:
    n = _num(value)
    if n is None:
        return "n/a"
    if abs(n) >= 100:
        return f"{n:.1f}"
    return f"{n:.{digits}f}"


# Paper-practice action states from signal_engine → shown in chat + UI legend.
_ACTION_MEANINGS: dict[str, str] = {
    "BUY": "Fresh long bias — look for entries in the zone (paper)",
    "ACCUMULATE": "Staged adds on dips — avoid chasing strength (paper)",
    "HOLD": "No clear edge — stay flat or keep what you have (paper)",
    "TRIM": "Lighten / reduce on strength — not a full exit (paper)",
    "SELL": "Avoid fresh buys / cut exposure (paper)",
    "WAIT": "Skip for now — setup or data quality not ready (paper)",
    "STRONG BUY": "High-conviction fresh long bias (paper)",
    "STRONG SELL": "High-conviction exit / avoid (paper)",
    "NEUTRAL": "No directional edge (paper)",
}

_ACTION_LEGEND_LINE = (
    "**Action legend (paper practice):** "
    "BUY = fresh long · ACCUMULATE = staged adds on dips · "
    "HOLD = no clear edge · TRIM = lighten/reduce on strength · "
    "SELL = exit/avoid · WAIT = skip until setup improves"
)


def _action_meaning(action: Any) -> str | None:
    key = str(action or "").strip().upper().replace("_", " ")
    if not key:
        return None
    return _ACTION_MEANINGS.get(key)


def _safe_div_yield(value: Any) -> str:
    """Hide absurd yield strings from bad upstream scaling."""
    if value in (None, "", "N/A", "Not declared"):
        return str(value or "n/a")
    n = _num(value)
    if n is None:
        return str(value)
    # Yields above 25% for large-caps are almost always bad data.
    if n > 25:
        return "n/a"
    return str(value)


def resolve_stock_response_profile(query: str, intent: str) -> str:
    """Decide which answer shape to render for a named-symbol ask.

    Profiles: quote | news | sentiment | technical | trade_plan | prediction |
    fundamentals | stock_analysis | calculations | risks
    """
    from .query_contract import resolve_query_contract

    return resolve_query_contract(query or "", intent_hint=intent or "").profile


def _first(*values: Any) -> Any:
    for value in values:
        if value is None or value == "" or value == "N/A":
            continue
        return value
    return None


def normalize_market_context(ctx: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize BYSEL enrich payloads into composer-friendly dicts."""
    raw = dict(ctx or {})
    tech_in = raw.get("technical") if isinstance(raw.get("technical"), dict) else {}
    fund_in = raw.get("fundamental") if isinstance(raw.get("fundamental"), dict) else {}
    levels_in = raw.get("trading_levels") if isinstance(raw.get("trading_levels"), dict) else {}

    tech = {
        "rsi": _first(tech_in.get("rsi"), tech_in.get("RSI")),
        "rsi_interpretation": tech_in.get("rsi_interpretation"),
        "trend": _first(tech_in.get("trend"), tech_in.get("ma_signal"), tech_in.get("moving_averages")),
        "macd_hist": _first(
            tech_in.get("macd_hist"),
            tech_in.get("macd_histogram"),
            tech_in.get("histogram"),
        ),
        "bollinger_bands": tech_in.get("bollinger_bands"),
        "moving_averages": tech_in.get("moving_averages"),
    }
    fund = {
        "pe": _first(fund_in.get("pe"), fund_in.get("pe_ratio"), fund_in.get("trailingPE"), fund_in.get("PE")),
        "pb": _first(fund_in.get("pb"), fund_in.get("priceToBook"), fund_in.get("pb_ratio")),
        "roe": _first(fund_in.get("roe"), fund_in.get("returnOnEquity")),
        "eps": _first(fund_in.get("eps"), fund_in.get("trailingEps")),
        "market_cap": fund_in.get("market_cap"),
        "dividend_yield": fund_in.get("dividend_yield"),
        "week_52": fund_in.get("week_52"),
        "pe_sector_avg": fund_in.get("pe_sector_avg"),
    }
    levels = {
        "support": _first(levels_in.get("support"), levels_in.get("support_1")),
        "support_2": levels_in.get("support_2"),
        "resistance": _first(levels_in.get("resistance"), levels_in.get("resistance_1")),
        "resistance_2": levels_in.get("resistance_2"),
        "stop_loss": _first(levels_in.get("stop_loss"), levels_in.get("stop")),
        "take_profit": levels_in.get("take_profit"),
        "risk_reward": levels_in.get("risk_reward"),
    }
    # Prefer Wilder RSI from P0 pack when present.
    p0 = raw.get("p0_math") if isinstance(raw.get("p0_math"), dict) else {}
    if p0.get("wilder_rsi_14") is not None:
        tech["rsi"] = p0.get("wilder_rsi_14")
    if p0.get("rsi_divergence"):
        tech["rsi_divergence"] = p0.get("rsi_divergence")
    bb = p0.get("bollinger") if isinstance(p0.get("bollinger"), dict) else {}
    if bb.get("pct_b") is not None:
        tech["pct_b"] = bb.get("pct_b")
        tech["bb_bandwidth"] = bb.get("bandwidth")
        tech["bb_regime"] = bb.get("bandwidth_regime")
    if p0.get("atr_14") is not None:
        tech["atr"] = p0.get("atr_14")
    atr_stop = p0.get("atr_stop") if isinstance(p0.get("atr_stop"), dict) else {}
    if atr_stop.get("stop") is not None:
        levels["atr_stop"] = atr_stop.get("stop")
        levels["atr_target_1r"] = atr_stop.get("target_1r")
        if levels.get("stop_loss") is None:
            levels["stop_loss"] = atr_stop.get("stop")
        if atr_stop.get("risk_reward") is not None:
            levels["risk_reward"] = atr_stop.get("risk_reward")
    val = p0.get("valuation_math") if isinstance(p0.get("valuation_math"), dict) else {}
    if fund.get("pe") is None and val.get("pe") is not None:
        fund["pe"] = val.get("pe")
    if fund.get("eps") is None and val.get("eps") is not None:
        fund["eps"] = val.get("eps")

    trade_plan = raw.get("trade_plan") if isinstance(raw.get("trade_plan"), dict) else {}
    if (not trade_plan or not trade_plan.get("action")) and isinstance(p0.get("trade_plan"), dict):
        trade_plan = p0["trade_plan"]
    # Prefer plan levels for stop/target when enrich levels are thin.
    if trade_plan.get("stop") is not None and levels.get("stop_loss") is None:
        levels["stop_loss"] = trade_plan.get("stop")
    if trade_plan.get("target_1") is not None and levels.get("take_profit") is None:
        levels["take_profit"] = trade_plan.get("target_1")
    st = p0.get("supertrend") if isinstance(p0.get("supertrend"), dict) else {}
    if st.get("direction") and not tech.get("supertrend"):
        tech["supertrend"] = st.get("direction")
    macd_p = p0.get("macd") if isinstance(p0.get("macd"), dict) else {}
    if macd_p.get("histogram") is not None and tech.get("macd_hist") is None:
        tech["macd_hist"] = macd_p.get("histogram")

    return {
        **raw,
        "symbol": str(raw.get("symbol") or "").upper().strip(),
        "current_price": raw.get("current_price"),
        "technical": tech,
        "fundamental": fund,
        "trading_levels": levels,
        "peers": raw.get("peers") if isinstance(raw.get("peers"), list) else [],
        "pre_signals": raw.get("pre_signals") if isinstance(raw.get("pre_signals"), dict) else {},
        "sentiment": raw.get("sentiment") if isinstance(raw.get("sentiment"), dict) else {},
        "sentiment_pack": (
            raw.get("sentiment_pack") if isinstance(raw.get("sentiment_pack"), dict) else {}
        ),
        "news_headlines": (
            list(raw.get("news_headlines") or [])
            if isinstance(raw.get("news_headlines"), list)
            else []
        ),
        "news_summary": raw.get("news_summary"),
        "p0_math": p0,
        "trade_plan": trade_plan,
        "conversation_summary": raw.get("conversation_summary"),
        "user_sentiment": (
            raw.get("user_sentiment") if isinstance(raw.get("user_sentiment"), dict) else {}
        ),
        "query_nlu": raw.get("query_nlu") if isinstance(raw.get("query_nlu"), dict) else {},
        "conversation": raw.get("conversation") if isinstance(raw.get("conversation"), dict) else {},
        "portfolio_context": raw.get("portfolio_context"),
    }


def hydrate_from_context_lines(ctx: dict[str, Any], context_lines: list[str]) -> dict[str, Any]:
    """Fill missing RSI/levels/price from 'Live enrich …' knowledge lines."""
    out = dict(ctx)
    tech = dict(out.get("technical") or {})
    fund = dict(out.get("fundamental") or {})
    levels = dict(out.get("trading_levels") or {})
    blob = "\n".join(context_lines)

    if _num(tech.get("rsi")) is None:
        m = re.search(r"rsi\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", blob, flags=re.I)
        if m:
            tech["rsi"] = m.group(1)
    if not tech.get("trend") or str(tech.get("trend")).lower() in {"", "n/a", "none"}:
        m = re.search(r"trend\s*[:=]\s*([a-z_]+)", blob, flags=re.I)
        if m:
            tech["trend"] = m.group(1)
    if _num(out.get("current_price")) is None:
        m = re.search(r"(?:price|last)\s*[:=]\s*₹?\s*([0-9,]+(?:\.[0-9]+)?)", blob, flags=re.I)
        if m:
            out["current_price"] = m.group(1)
    if _num(levels.get("support")) is None:
        m = re.search(r"support(?:_1)?\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)", blob, flags=re.I)
        if m:
            levels["support"] = m.group(1)
    if _num(levels.get("resistance")) is None:
        m = re.search(r"resistance(?:_1)?\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)", blob, flags=re.I)
        if m:
            levels["resistance"] = m.group(1)
    if _num(levels.get("stop_loss")) is None:
        m = re.search(r"stop_loss\s*[:=]\s*([0-9,]+(?:\.[0-9]+)?)", blob, flags=re.I)
        if m:
            levels["stop_loss"] = m.group(1)
    if _num(fund.get("pe")) is None:
        m = re.search(r"(?:pe_ratio|pe)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", blob, flags=re.I)
        if m:
            fund["pe"] = m.group(1)

    out["technical"] = tech
    out["fundamental"] = fund
    out["trading_levels"] = levels
    return out


def _clean_line(text: str) -> str | None:
    low = text.lower()
    if "grounded answer note" in low or "educational retrieved note" in low:
        return None
    if "nse market status" in text:
        text = text.split("NSE market status")[0].strip().rstrip(".")
    text = text.strip()
    return text or None


def _stance_from_tech(tech: dict, fund: dict, *, weekly: bool = False) -> tuple[str, str, int]:
    """Return (stance, rationale, score)."""
    rsi = _num(tech.get("rsi"))
    trend = str(tech.get("trend") or tech.get("moving_averages") or "").lower()
    macd_hist = _num(tech.get("macd_hist"))
    pe = _num(fund.get("pe"))

    score = 0
    reasons: list[str] = []
    if rsi is not None:
        if rsi >= 70:
            score -= 2 if weekly else 1
            reasons.append(f"RSI {_fmt(rsi)} is elevated (overbought zone)")
        elif rsi <= 30:
            score += 2 if weekly else 1
            reasons.append(f"RSI {_fmt(rsi)} is washed out (oversold zone)")
        elif rsi >= 60:
            score += 1
            reasons.append(f"RSI {_fmt(rsi)} shows bullish momentum building")
        elif rsi <= 40:
            score -= 1
            reasons.append(f"RSI {_fmt(rsi)} shows soft momentum")
        else:
            reasons.append(f"RSI {_fmt(rsi)} is mid-range")
    if "strong_bull" in trend or trend == "bullish":
        score += 1
        reasons.append(f"trend cue: {trend}")
    elif "bull" in trend or "up" in trend:
        score += 1
        reasons.append(f"trend cue: {trend}")
    elif "bear" in trend or "down" in trend:
        score -= 1
        reasons.append(f"trend cue: {trend}")
    if macd_hist is not None:
        if macd_hist > 0:
            score += 1
            reasons.append(f"MACD histogram positive ({_fmt(macd_hist)})")
        elif macd_hist < 0:
            score -= 1
            reasons.append(f"MACD histogram negative ({_fmt(macd_hist)})")
    if pe is not None and pe > 0:
        if pe > 45:
            score -= 1
            reasons.append(f"P/E {_fmt(pe)} is rich vs typical large-cap bands")
        elif pe < 15:
            score += 1
            reasons.append(f"P/E {_fmt(pe)} is comparatively modest")
        else:
            reasons.append(f"P/E {_fmt(pe)}")

    if score >= 2:
        stance = "BUY bias (paper / wait for your entry zone)"
    elif score <= -2:
        stance = "SELL / avoid fresh buys; prefer trim or tight trail if already long"
    else:
        stance = "HOLD / wait — risk-reward not clearly skewed"
    rationale = "; ".join(reasons) if reasons else "limited live signals available"
    return stance, rationale, score


def _detect_sector(query: str) -> str | None:
    q = query.lower()
    for key in (
        "defence", "defense", "pharma", "banking", "fmcg", "railway", "realty",
        "cement", "infra", "energy", "metal", "auto", "psu",
    ):
        if re.search(r"\b" + re.escape(key) + r"\b", q):
            return "defence" if key == "defense" else ("banking" if key == "bank" else key)
    if re.search(r"\bit\b|\binformation technology\b|\bnifty\s*it\b", q):
        return "it"
    if re.search(r"\bbank(s|ing)?\b", q):
        return "banking"
    return None


_LEAK_SUBS = (
    (re.compile(r"\bWilder\s+", re.I), ""),
    (re.compile(r"\bEV/?EBITDA\b", re.I), "valuation multiple"),
    (re.compile(r"\bNetDebt/?EBITDA\b", re.I), "leverage"),
    (re.compile(r"\bSortino\(60d\)\b", re.I), "downside-risk score"),
    (re.compile(r"\bP0\b"), ""),
    (re.compile(r"\bquantitative (engine|stack)\b", re.I), "live tape"),
)


def _plain_reason(text: Any) -> str:
    """Keep reasons user-facing: no method names or internal scoring labels."""
    out = str(text or "").strip()
    if not out:
        return ""
    for pat, repl in _LEAK_SUBS:
        out = pat.sub(repl, out)
    return re.sub(r"\s{2,}", " ", out).strip(" ;,")


def _plain_reasons(items: list[Any] | None, limit: int = 3) -> list[str]:
    return [line for line in (_plain_reason(item) for item in (items or [])) if line][:limit]


def _shape_heading(query: str, profile: str) -> str:
    try:
        from .answer_shape import detect_answer_shape

        _facet, heading = detect_answer_shape(query, profile)
        return heading
    except Exception:
        return ""


def _named_title(header_base: str, query: str, profile: str, fallback: str) -> str:
    heading = _shape_heading(query, profile) or fallback
    return f"{header_base} — {heading}"


def _cagr_card(query: str) -> str | None:
    nums = [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", query or "")]
    if len(nums) < 3:
        return (
            "**CAGR working**\n\n"
            "**Your ask:** " + (query or "").strip()[:90] + "\n\n"
            "Need start value, end value, and years.\n"
            "Example: calculate CAGR for 100000 to 180000 in 3 years.\n"
            "`CAGR = (end / start) ^ (1 / years) − 1`\n"
            "_Educational math — not a return promise._"
        )
    start, end, years = nums[0], nums[1], nums[2]
    if start <= 0 or years <= 0:
        return None
    try:
        from indian_stock_llm.calculations import DeterministicCalculator

        pct = DeterministicCalculator.cagr(start=start, end=end, years=years)
    except Exception:
        pct = ((end / start) ** (1.0 / years) - 1.0) * 100.0
    return (
        "**CAGR working**\n\n"
        f"**Your ask:** {(query or '').strip()[:90]}\n\n"
        f"• Start: {start:g} → End: {end:g} over {years:g} years\n"
        f"• Formula: (end / start) ^ (1 / years) − 1\n"
        f"• Result: **{pct:.2f}%** CAGR\n"
        "_Educational math — not a return promise._"
    )


def _ask_line(query: str) -> str | None:
    """Echo a short user ask so the first screen matches their question."""
    try:
        from .query_language import display_user_ask

        text = display_user_ask(query)
    except Exception:
        text = re.sub(r"\s+", " ", (query or "").strip())
    if not text or len(text) > 90:
        return None
    if text.lower().startswith("user_query:"):
        return None
    return f"**Your ask:** {text}"


def _is_short_followup(query: str) -> bool:
    """why / simplify / explain-the-snapshot should not reprint the previous card."""
    q = (query or "").lower()
    return bool(
        re.search(
            r"\b(why this paper stance|explain the\b.{0,40}\bsnapshot|in simple words)\b",
            q,
        )
    )


def _short_followup_card(
    *,
    symbol: str,
    query: str,
    price: Any,
    tech: dict[str, Any],
    support: Any,
    resistance: Any,
) -> str:
    stance, rationale, _score = _stance_from_tech(tech, {})
    rationale = _plain_reason(rationale)
    lines = [
        f"**{symbol} — in short**",
    ]
    ask = _ask_line(query)
    if ask:
        lines.append(ask)
    lines.extend(
        [
            "",
            f"**Direct answer:** {stance}",
            f"**Why:** {rationale}",
            f"• Last: ₹{_fmt(price)}",
        ]
    )
    if _num(support) is not None or _num(resistance) is not None:
        lines.append(f"• Support / Resistance: {_fmt(support)} / {_fmt(resistance)}")
    lines.extend(
        [
            "",
            "Short read for paper practice — not a new buy or sell call.",
            "_Educational only — not investment advice._",
        ]
    )
    return "\n".join(lines)


def _headline_about_symbol(title: str, symbol: str, company: str = "") -> bool:
    """Keep a headline only when it is about this listing, not a lookalike name."""
    text = re.sub(r"\s+", " ", (title or "")).strip()
    if not text:
        return False
    low = text.lower()
    if re.search(
        r"\b(bitcoin|crypto|litecoin|chainlink|ethereum|dogecoin|solana)\b",
        low,
    ):
        return False
    foreign_listing = bool(re.search(r"\((?:nyse|nasdaq|amex):", low))
    india_listing = bool(re.search(r"\b(nse|bse|india|nifty)\b", low))
    if foreign_listing and not india_listing:
        return False
    sym = (symbol or "").strip().lower()
    needles: list[str] = []
    if sym:
        needles.append(sym)
    company_l = re.sub(r"\s+", " ", (company or "").lower()).strip()
    if company_l:
        needles.append(company_l)
        core = re.sub(r"\b(ltd|limited|inc|corporation|corp)\b\.?", "", company_l).strip()
        if len(core) >= 4:
            needles.append(core)
    needles.extend(
        {
            "infy": ("infosys",),
            "tcs": ("tata consultancy",),
            "hdfcbank": ("hdfc bank",),
            "sbin": ("state bank of india", "sbi"),
            "reliance": ("reliance industries",),
            "icicibank": ("icici bank",),
            "hcltech": ("hcl technologies",),
            "itc": ("itc",),
        }.get(sym, ())
    )
    for needle in needles:
        if not needle:
            continue
        if " " in needle:
            if needle in low:
                return True
        elif re.search(rf"\b{re.escape(needle)}\b", low):
            return True
    return False


def compose_structured_answer(
    *,
    query: str,
    intent: str,
    market_context: dict[str, Any] | None,
    context_lines: list[str],
    deterministic: str = "",
    profile: str | None = None,
) -> str | None:
    """Build a structured answer when we have enough live/domain signal."""
    answer = _compose_structured_answer_impl(
        query=query,
        intent=intent,
        market_context=market_context,
        context_lines=context_lines,
        deterministic=deterministic,
        profile=profile,
    )
    threaded = _with_compound_followthrough(answer, query, market_context, profile)
    voiced = _with_conversation_voice(threaded, market_context, profile, query=query)
    return _with_desk_footnote(voiced, market_context)


def _with_desk_footnote(answer: str | None, market_context: dict[str, Any] | None) -> str | None:
    if not answer:
        return answer
    ctx = normalize_market_context(market_context)
    try:
        from app.ism_desk import render_desk_footnote

        note = render_desk_footnote(ctx.get("desk") if isinstance(ctx.get("desk"), dict) else None)
    except Exception:
        note = ""
    if not note or note in answer:
        return answer
    return f"{answer.rstrip()}\n\n{note}"


def _compact_secondary_block(
    secondary_profile: str,
    symbol: str,
    ctx: dict[str, Any],
) -> str | None:
    """Short second card when the user asked two things in one message."""
    if not symbol or not secondary_profile:
        return None
    tech = ctx.get("technical") or {}
    levels = ctx.get("trading_levels") or {}
    price = ctx.get("current_price")
    plan = ctx.get("trade_plan") if isinstance(ctx.get("trade_plan"), dict) else {}
    p0 = ctx.get("p0_math") if isinstance(ctx.get("p0_math"), dict) else {}
    if not plan and isinstance(p0.get("trade_plan"), dict):
        plan = p0["trade_plan"]
    if secondary_profile == "quote":
        return (
            f"**Also — live quote for {symbol}**\n"
            f"• Last: ₹{_fmt(price)}"
        )
    if secondary_profile == "trade_plan":
        action = plan.get("action") or "HOLD"
        meaning = _action_meaning(action) or "Paper stance from the live tape."
        stop = plan.get("stop") or levels.get("stop_loss")
        target = plan.get("target_1") or levels.get("take_profit")
        return (
            f"**Also — paper call for {symbol}**\n"
            f"• **{action}** — {meaning}\n"
            f"• Stop idea {_fmt(stop)} | Target idea {_fmt(target)}"
        )
    if secondary_profile == "sentiment":
        sent = ctx.get("sentiment") or {}
        pack = ctx.get("sentiment_pack") or {}
        label = pack.get("label") or sent.get("overall") or "mixed"
        return f"**Also — sentiment for {symbol}**\n• Tape mood: **{label}**"
    if secondary_profile == "news":
        heads = list(ctx.get("news_headlines") or [])[:2]
        if not heads:
            return None
        lines = [f"**Also — news on {symbol}**"]
        for h in heads:
            lines.append(f"• {str(h).strip()[:120]}")
        return "\n".join(lines)
    if secondary_profile == "technical":
        return (
            f"**Also — tape for {symbol}**\n"
            f"• RSI {_fmt(tech.get('rsi'))} | "
            f"Support {_fmt(levels.get('support'))} / "
            f"Resistance {_fmt(levels.get('resistance'))}"
        )
    return None


def _with_compound_followthrough(
    answer: str | None,
    query: str,
    market_context: dict[str, Any] | None,
    profile: str | None,
) -> str | None:
    if not answer:
        return answer
    ctx = normalize_market_context(market_context)
    nlu = ctx.get("query_nlu") if isinstance(ctx.get("query_nlu"), dict) else {}
    secondary = str(nlu.get("secondary_profile") or "").strip()
    primary = str(nlu.get("primary_profile") or profile or "").strip()
    symbol = str(ctx.get("symbol") or "").upper().strip()
    if not nlu.get("compound") or not secondary or secondary == primary or not symbol:
        return answer
    extra = _compact_secondary_block(secondary, symbol, ctx)
    if not extra:
        return answer
    return f"{answer.rstrip()}\n\n{extra}"


def _with_conversation_voice(
    answer: str | None,
    market_context: dict[str, Any] | None,
    profile: str | None,
    query: str = "",
) -> str | None:
    if not answer:
        return answer
    ctx = normalize_market_context(market_context)
    convo = ctx.get("conversation") if isinstance(ctx.get("conversation"), dict) else {}
    try:
        from .conversation import conversational_lead

        lead = conversational_lead(
            follow_up=bool(convo.get("follow_up")),
            symbol=str(ctx.get("symbol") or convo.get("prior_symbol") or "").upper() or None,
            profile=profile or str(convo.get("prior_profile") or "") or None,
            sentiment=ctx.get("user_sentiment") if isinstance(ctx.get("user_sentiment"), dict) else None,
            query=query or str(convo.get("query") or ""),
        )
    except Exception:
        lead = None
    if not lead:
        return answer
    return f"{lead}\n\n{answer.lstrip()}"


def _compose_structured_answer_impl(
    *,
    query: str,
    intent: str,
    market_context: dict[str, Any] | None,
    context_lines: list[str],
    deterministic: str = "",
    profile: str | None = None,
) -> str | None:
    """Build a structured answer when we have enough live/domain signal."""
    q = (query or "").lower()
    ctx = hydrate_from_context_lines(normalize_market_context(market_context), context_lines)
    symbol = str(ctx.get("symbol") or "").upper().strip()
    try:
        from .query_contract import should_inherit_symbol
        from .symbol_linker import link_symbols

        named = bool(link_symbols(query))
        if symbol and not should_inherit_symbol(profile or "", query, named):
            symbol = ""
            ctx["symbol"] = ""
        if not symbol and (profile or intent) in {"prediction", "sentiment"}:
            from .symbol_linker import is_index_or_market_ask

            if is_index_or_market_ask(query):
                if (profile or intent) == "prediction":
                    return (
                        "**Index outlook (educational)**\n\n"
                        "**Your ask:** " + (query or "").strip()[:90] + "\n\n"
                        "This is a market-level ask, not a single-stock forecast.\n"
                        "• Use Nifty/Sensex structure and the open risk window — not one name on screen.\n"
                        "• Ask a listed stock if you want a paper scenario range for that name.\n"
                        "_Not a price guarantee — educational only._"
                    )
                return (
                    "**Market mood (educational)**\n\n"
                    "**Your ask:** " + (query or "").strip()[:90] + "\n\n"
                    "Broad tape mood is mixed until you name a stock or index contract.\n"
                    "Ask “Mood on RELIANCE” or “Nifty sentiment” with a listed name for a tighter read.\n"
                    "_Sentiment framing for paper practice — not investment advice._"
                )
    except Exception:
        pass

    if re.search(r"\bcagr\b", q) and (profile or "") in {"calculations", "literacy", ""}:
        card = _cagr_card(query)
        if card:
            return card

    if (profile or "") == "fundamentals":
        try:
            from .symbol_linker import is_index_or_market_ask

            if is_index_or_market_ask(query) and not symbol:
                return (
                    "**Index valuation (educational)**\n\n"
                    f"**Your ask:** {(query or '').strip()[:90]}\n\n"
                    "This is an index valuation ask, not a single-stock P/E dump.\n"
                    "• Use the Nifty/Sensex multiple as a market-level cue only.\n"
                    "• A lower index P/E is not a buy call; a richer multiple is not a sell call.\n"
                    "Ask a listed name if you want that stock's P/E snapshot.\n"
                    "_Educational — not a SEBI RA valuation._"
                )
        except Exception:
            pass

    if (profile or intent) == "session" or re.search(
        r"\b(is (the )?market open|market hours|trading hours|when does (the )?market)\b",
        q,
    ):
        try:
            from .agent_tools import format_session_card, session_snapshot

            snap = ctx.get("session") if isinstance(ctx.get("session"), dict) else session_snapshot()
            return format_session_card(query, snap)
        except Exception:
            pass

    if (profile or "") == "corporate_actions" or (
        symbol
        and re.search(r"\b(ex[- ]?date|dividend date|corporate actions?|record date)\b", q)
    ):
        try:
            from .agent_tools import format_corporate_actions_card

            rows = ctx.get("corporate_actions")
            dated = rows if isinstance(rows, list) else None
            return format_corporate_actions_card(query, symbol or "", rows=dated)
        except Exception:
            pass

    try:
        from app.ism_desk import render_desk_answer

        desk_card = render_desk_answer(query, ctx.get("desk") if isinstance(ctx.get("desk"), dict) else None)
        if desk_card:
            return desk_card
    except Exception:
        pass

    tech = ctx.get("technical") or {}
    fund = ctx.get("fundamental") or {}
    levels = ctx.get("trading_levels") or {}
    price = ctx.get("current_price")
    peers = ctx.get("peers") or []
    pre_signals = ctx.get("pre_signals") or {}
    sentiment = ctx.get("sentiment") or {}
    sentiment_pack = (
        ctx.get("sentiment_pack") if isinstance(ctx.get("sentiment_pack"), dict) else {}
    )
    p0 = ctx.get("p0_math") if isinstance(ctx.get("p0_math"), dict) else {}

    # ── Beginner market literacy (no symbol required) ───────────────
    literacy_phrase = bool(
        re.search(
            r"\b(how does the (stock|share) market work|how the (stock|share) market works|"
            r"stock market meaning|key participants|depository participant|"
            r"how (are |do )?share prices|price discovery|how to start investing|"
            r"common mistakes|primary market|secondary market)\b",
            q,
        )
    )
    named_on_query = False
    try:
        from .symbol_linker import link_symbols

        named_on_query = any(
            s not in {"NIFTY", "SENSEX", "NIFTY50", "BANKNIFTY"}
            for s in link_symbols(query)
        )
    except Exception:
        named_on_query = bool(symbol)
    live_profile = (profile or "") in {
        "quote",
        "news",
        "sentiment",
        "technical",
        "trade_plan",
        "prediction",
        "fundamentals",
        "risks",
        "stock_analysis",
        "compare",
    }
    # "RELIANCE ela undi?" / "How is this one?" must not become the IPO primer
    # just because the word "how" scored as market literacy.
    if (
        ((profile or "") == "literacy" or intent == "market_literacy" or literacy_phrase)
        and not ((live_profile or named_on_query or bool(symbol)) and not literacy_phrase)
    ):
        try:
            from app.market_education import get_education_answer

            edu = get_education_answer(query)
            if edu:
                return edu
        except Exception:
            pass
        # Fallback structured primer from local knowledge (no paid LLM).
        if re.search(r"\b(mistake|rumour|rumor|tips?)\b", q):
            return (
                "**Common share-market mistakes to avoid**\n\n"
                "• Investing without basic knowledge\n"
                "• Following tips or rumours blindly\n"
                "• Overtrading / revenge trading\n"
                "• Ignoring diversification\n"
                "• Trying to time every tick perfectly\n"
                "• Letting fear or greed override a written plan\n"
                "• No clear goal or risk budget\n\n"
                "Process fix: thesis → entry → stop → invalidation → journal.\n"
                "_Educational checklist — not personalized advice._"
            )
        if re.search(r"\b(participants?|nsdl|cdsl|depository participant|depositories)\b|\bdp\b", q):
            return (
                "**Key stock-market participants (India)**\n\n"
                "• Retail investors & traders\n"
                "• Institutions (MF / FII / DII)\n"
                "• Exchanges: NSE & BSE\n"
                "• Depositories: NSDL & CDSL\n"
                "• Depository Participants (broker/bank DPs)\n"
                "• Regulator: SEBI\n\n"
                "_Educational overview — not investment advice._"
            )
        if re.search(r"\b(price|demand|supply|discovery)\b", q):
            return (
                "**How share prices are determined**\n\n"
                "Primary driver: demand vs supply on the exchange order book.\n"
                "Also: company earnings, macro (rates/inflation), sector trends, news, "
                "sentiment, institutional flows, and liquidity.\n\n"
                "_Educational framing — not a price prediction._"
            )
        if re.search(r"\b(start|kyc|demat|begin)\b", q):
            return (
                "**How to start investing (India — educational)**\n\n"
                "1. Open trading + demat via a registered broker/DP\n"
                "2. Complete KYC (PAN, Aadhaar, bank — verify current rules)\n"
                "3. Add funds\n"
                "4. Research (not tips)\n"
                "5. Place an order you understand\n"
                "6. Start small; diversify; review vs goals\n\n"
                "Use BYSEL for paper practice of research and process first.\n"
                "_Not onboarding advice for any specific broker._"
            )
        return (
            "**How the Indian stock market works**\n\n"
            "1. IPO (primary) → company raises capital\n"
            "2. Listing on NSE/BSE (secondary trading)\n"
            "3. Place orders via broker app\n"
            "4. Exchange matches buy/sell in real time\n"
            "5. T+1 settlement into demat / funds ledger\n\n"
            "Regulator: SEBI. Depositories: NSDL/CDSL via your DP.\n"
            "_Educational beginner primer — not investment advice._"
        )

    # ── Overbought / oversold direct answer ─────────────────────────
    if intent == "overbought_check" or re.search(
        r"\b(overbought|oversold|over.?bought|over.?sold)\b", q
    ):
        if not symbol:
            return None
        rsi = _num(tech.get("rsi"))
        trend = str(tech.get("trend") or "n/a")
        if rsi is None:
            return (
                f"**{symbol} — overbought check**\n\n"
                "I could not read a live RSI yet. Retry in a moment or ask "
                f"`RSI of {symbol}`.\n\n"
                "_Educational only — not investment advice._"
            )
        if "oversold" in q:
            verdict = "Yes" if rsi <= 30 else "No"
            detail = (
                f"RSI is {_fmt(rsi)} "
                + ("(≤ 30 oversold threshold)." if rsi <= 30 else "(above the classic ≤ 30 oversold threshold).")
            )
        else:
            verdict = "Yes" if rsi >= 70 else "No"
            detail = (
                f"RSI is {_fmt(rsi)} "
                + ("(≥ 70 overbought threshold)." if rsi >= 70 else "(below the classic ≥ 70 overbought threshold).")
            )
        parts = [
            f"**{symbol} — momentum check**",
            "",
            f"**Direct answer:** {verdict}. {detail}",
            f"• Price: {_fmt(price)} | Trend: {trend}",
            f"• Support: {_fmt(levels.get('support'))} | Resistance: {_fmt(levels.get('resistance'))}",
        ]
        if tech.get("rsi_interpretation"):
            parts.append(f"• RSI read: {tech.get('rsi_interpretation')}")
        if tech.get("rsi_divergence") or p0.get("rsi_divergence"):
            parts.append(f"• Divergence: {tech.get('rsi_divergence') or p0.get('rsi_divergence')}")
        if p0.get("ok"):
            bb = p0.get("bollinger") or {}
            vs = p0.get("vs_nifty") or {}
            vol = p0.get("volume") or {}
            parts.append(
                f"• %B={_fmt(bb.get('pct_b'), 3)} | BW={bb.get('bandwidth_regime')} | "
                f"Vol z20={_fmt(vol.get('zscore_20'))} | RS20 vs Nifty={_fmt(vs.get('rs_20d'), 3)}"
            )
        parts.extend(
            [
                "",
                "RSI alone is not enough — confirm with trend and volume before paper trades.",
                "_Grounded by BYSEL._",
            ]
        )
        return "\n".join(parts)

    # ── Sector screen ───────────────────────────────────────────────
    # Only when the user asked for a list, or intent is already sector_screen.
    # A PE / prediction / TA sentence that merely mentions "IT stocks" is not a screen.
    sector = _detect_sector(q)
    list_ask = bool(
        re.search(
            r"\b(top|best|list)\b.{0,24}\bstocks?\b|\bstocks?\s+under\b|\bunder nifty\b",
            q,
        )
    )
    if intent == "sector_screen" or (
        sector
        and list_ask
        and intent in {"general_query", "sector_screen", ""}
    ):
        names = _SECTOR_UNIVERSE.get(sector or "", [])
        parts = [
            f"**{ (sector or 'sector').upper() } screen (educational)**",
            "",
            "Curated liquid names to research further (not a ranked buy list):",
        ]
        if names:
            parts.append("• " + ", ".join(names))
            if sector in {"defence", "defense"}:
                parts.append(
                    "• Drivers: MoD order-book, indigenisation, export clearances, quarterly execution."
                )
            elif sector == "it":
                parts.append(
                    "• Drivers: US/EU deal wins, EBIT margins, attrition, rupee, guidance quality."
                )
            elif sector == "banking":
                parts.append(
                    "• Drivers: NIM, credit growth, asset quality, RBI policy stance."
                )
        # Optional screened rows from KB
        # Keep only screener rows / matching sector primers — drop unrelated literacy.
        extra = 0
        for ln in context_lines:
            text = _clean_line(ln[2:].strip() if ln.startswith("- ") else ln)
            if not text:
                continue
            low = text.lower()
            if "fmcg and defence" in low and sector in {"defence", "defense"}:
                # Prefer the dedicated defence universe line already added.
                continue
            if "common liquid large-cap" in low or "nse and bse basics" in low:
                continue
            if "screened" in low or (sector and sector in low and any(n.lower() in low for n in names[:3])):
                parts.append(f"• {text}")
                extra += 1
            if extra >= 2:
                break
        parts.extend(
            [
                "",
                "Check liquidity, valuation vs peers, and news before any paper trade.",
                "_Grounded by BYSEL._",
            ]
        )
        if names or intent == "sector_screen":
            return "\n".join(parts)

    # ── Compare two+ names ──────────────────────────────────────────
    compare_named = False
    try:
        from app.stock_enricher import extract_all_symbols_from_query as _cmp_syms

        compare_named = len(_cmp_syms(query) or []) >= 2
    except Exception:
        compare_named = False
    if (
        intent == "compare"
        or profile == "compare"
        or re.search(r"\bcompare\b|\bversus\b", q)
        or (re.search(r"\bvs\b", q) and compare_named)
        or (re.search(r"\b(?:better\s+)?[a-z][a-z0-9.&-]{1,11}\s+or\s+[a-z][a-z0-9.&-]{1,11}\b", q) and compare_named)
    ):
        # Only scorecard legs the user named — drop holdings/context leaks.
        named_ordered: list[str] = []
        try:
            from .query_contract import _symbol_candidates

            named_ordered = [str(s).upper() for s in (_symbol_candidates(query) or []) if s]
        except Exception:
            named_ordered = []
        if not named_ordered:
            try:
                from app.stock_enricher import extract_all_symbols_from_query

                named_ordered = [
                    str(s).upper()
                    for s in (extract_all_symbols_from_query(query) or [])
                    if s
                ]
            except Exception:
                named_ordered = []
        mentioned = set(named_ordered)
        if symbol:
            mentioned.add(symbol)
            if symbol not in named_ordered:
                named_ordered.insert(0, symbol)

        by_sym: dict[str, dict[str, Any]] = {}
        if symbol and (not mentioned or symbol in mentioned):
            by_sym[symbol] = {
                "symbol": symbol,
                "price": price,
                "technical": tech,
                "fundamental": fund,
            }
        for peer in peers:
            if not isinstance(peer, dict):
                continue
            psym = str(peer.get("symbol") or "").upper()
            if not psym:
                continue
            if mentioned and psym not in mentioned:
                continue
            by_sym[psym] = {
                "symbol": psym,
                "price": peer.get("current_price"),
                "technical": normalize_market_context(peer).get("technical"),
                "fundamental": normalize_market_context(peer).get("fundamental"),
            }

        legs = named_ordered[:3] or list(by_sym.keys())
        rows = []
        for name in legs:
            rows.append(
                by_sym.get(
                    name,
                    {
                        "symbol": name,
                        "price": None,
                        "technical": {},
                        "fundamental": {},
                    },
                )
            )

        parts = ["**Comparison scorecard (educational)**", ""]
        if len(rows) >= 2:
            parts.append("| Stock | Price | RSI | Trend | P/E | P/B | ROE% | Mcap | Notes |")
            parts.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
            for row in rows[:3]:
                t = row.get("technical") or {}
                f = row.get("fundamental") or {}
                rsi = _num(t.get("rsi"))
                note = (
                    "overbought risk" if rsi is not None and rsi >= 70
                    else "oversold watch" if rsi is not None and rsi <= 30
                    else "momentum OK" if rsi is not None and rsi >= 55
                    else "neutral"
                )
                parts.append(
                    f"| {row.get('symbol')} | {_fmt(row.get('price'))} | {_fmt(t.get('rsi'))} | "
                    f"{t.get('trend') or 'n/a'} | {_fmt(f.get('pe'))} | "
                    f"{_fmt(f.get('pb') or f.get('p/b'))} | {_fmt(f.get('roe'))} | "
                    f"{f.get('market_cap') or 'n/a'} | {note} |"
                )
            if "cheaper" in q or "better" in q:
                priced = [
                    (str(row.get("symbol") or ""), _num(row.get("price")))
                    for row in rows[:3]
                ]
                priced = [(name, px) for name, px in priced if name and px is not None]
                if "cheaper" in q and len(priced) >= 2:
                    winner = min(priced, key=lambda item: item[1])
                    parts.append("")
                    parts.append(
                        f"**On last price:** {winner[0]} is lower "
                        f"(₹{_fmt(winner[1])}). That is not a valuation call."
                    )
                elif "cheaper" in q:
                    parts.append("")
                    parts.append(
                        "Both names are on the card. A last price is missing, "
                        "so this is not a cheaper call."
                    )
            # Per-name fundamental bullets (second-name depth).
            parts.append("")
            parts.append("**Fundamental detail**")
            for row in rows[:3]:
                f = row.get("fundamental") or {}
                parts.append(
                    f"• **{row.get('symbol')}**: P/E={_fmt(f.get('pe'))} | "
                    f"P/B={_fmt(f.get('pb') or f.get('p/b'))} | ROE={_fmt(f.get('roe'))}% | "
                    f"EPS={_fmt(f.get('eps'))} | Div={_safe_div_yield(f.get('dividend_yield'))} | "
                    f"Mcap={f.get('market_cap') or 'n/a'}"
                )

            # Simple winner heuristic
            def _row_score(row: dict) -> float:
                t = row.get("technical") or {}
                f = row.get("fundamental") or {}
                s = 0.0
                rsi = _num(t.get("rsi"))
                if rsi is not None:
                    if 45 <= rsi <= 65:
                        s += 1
                    if rsi >= 70:
                        s -= 1
                trend = str(t.get("trend") or "").lower()
                if "bull" in trend:
                    s += 1
                if "bear" in trend:
                    s -= 1
                pe = _num(f.get("pe"))
                if pe is not None and 0 < pe < 30:
                    s += 0.5
                roe = _num(f.get("roe"))
                if roe is not None and roe >= 15:
                    s += 0.5
                return s

            ranked = sorted(rows, key=_row_score, reverse=True)
            parts.append("")
            parts.append(
                f"**Bias (paper):** {ranked[0].get('symbol')} currently screens cleaner on "
                "available RSI/trend/P/E/ROE cues — still verify guidance and valuation vs history."
            )
        else:
            named = [s for s in mentioned if s]
            if symbol and symbol not in named:
                named = [symbol, *named]
            if len(named) < 2:
                try:
                    from .query_contract import _symbol_candidates

                    for extra in _symbol_candidates(query) or []:
                        if extra and extra not in named:
                            named.append(extra)
                except Exception:
                    pass
            other = [s for s in named if s != (symbol or "")]
            asked = " vs ".join(named[:2]) if len(named) >= 2 else (symbol or "these names")
            parts.append(f"You asked **{asked}**.")
            parts.append(f"Live tape loaded for **{symbol or 'n/a'}** @ {_fmt(price)}")
            parts.append(
                f"• RSI={_fmt(tech.get('rsi'))} | trend={tech.get('trend') or 'n/a'} | "
                f"P/E={_fmt(fund.get('pe'))}"
            )
            if other:
                parts.append(
                    f"• **{other[0]}** tape is not on this card yet — "
                    f"ask `Compare {symbol} and {other[0]}` once both quotes load."
                )
            else:
                parts.append(
                    "Name a second ticker (e.g. `Compare TCS and INFY`) for a full scorecard."
                )
        parts.extend(
            [
                "",
                "Do not pick solely on lower P/E — prefer growth durability + cleaner technicals for your horizon.",
                "_Grounded by BYSEL._",
            ]
        )
        return "\n".join(parts)

    # Trade-plan level asks must not fall into the bare fundamentals card.
    stop_loss_ask = bool(
        symbol
        and re.search(
            r"\b(stop[\s-]?loss|take[\s-]?profit|entry(?:\s+price)?|entry zone|best entry)\b",
            q,
        )
        and not re.search(
            r"\b(what is|define|definition|meaning of|target (next|this|for|price|of))\b",
            q,
        )
    )

    # ── Stock-specific valuation ────────────────────────────────────
    if symbol and not stop_loss_ask and (
        intent == "fundamentals"
        or re.search(r"\b(p/?e|pe ratio|valuation|eps|roe|pb|p/b)\b", q)
    ):
        pe = fund.get("pe")
        pb = fund.get("pb")
        roe = fund.get("roe")
        eps = fund.get("eps")
        parts = [
            _named_title(f"**{symbol}**", query, "fundamentals", "fundamental snapshot"),
        ]
        ask = _ask_line(query)
        if ask:
            parts.append(ask)
        parts.extend([
            "",
            f"• Last price: {_fmt(price)}",
            f"• P/E: {_fmt(pe)}"
            + (f" (sector cue: {fund.get('pe_sector_avg')})" if fund.get("pe_sector_avg") else ""),
            f"• P/B: {_fmt(pb)}",
            f"• ROE: {_fmt(roe)}",
            f"• EPS: {_fmt(eps)}",
            f"• Market cap: {fund.get('market_cap') or 'n/a'}",
            f"• Dividend yield: {_safe_div_yield(fund.get('dividend_yield'))}",
        ])
        if fund.get("week_52"):
            parts.append(f"• 52-week context: {fund.get('week_52')}")
        if re.search(r"\b(cheap|expensive|costly|undervalued|overvalued)\b", q):
            parts.append(
                "• Cheap/expensive: one multiple is not enough — read P/E vs growth and peers."
            )
        if re.search(r"\bbook value\b", q) and pb not in (None, "", "n/a"):
            parts.append(f"• Book-value cue: P/B {_fmt(pb)}.")
        parts.extend(
            [
                "",
                "Interpret P/E vs sector peers and growth — a low P/E alone is not 'cheap'.",
                "_Educational snapshot — not investment advice._",
            ]
        )
        if any(_num(x) is not None for x in (pe, pb, roe, eps, price)) or fund.get("market_cap") or p0.get("ok"):
            return "\n".join(parts)

    # ── Trading systems / momentum ──────────────────────────────────
    if re.search(r"\b(momentum portfolio|momentum|trading system)\b", q) and symbol:
        mom = p0.get("momentum") if isinstance(p0.get("momentum"), dict) else {}
        vs = p0.get("vs_nifty") if isinstance(p0.get("vs_nifty"), dict) else {}
        if mom.get("roc_20d_pct") is not None or vs.get("rs_20d") is not None:
            parts = [
                f"**{symbol} — momentum system cues**",
                "",
                f"• ROC 20d≈{_fmt(mom.get('roc_20d_pct'), 2)}% | ROC 60d≈{_fmt(mom.get('roc_60d_pct'), 2)}%",
                f"• vs Nifty RS20≈{_fmt(vs.get('rs_20d'), 3)} | RS60≈{_fmt(vs.get('rs_60d'), 3)}",
                "",
                f"Note: {mom.get('note') or 'Rank vs universe for portfolio sleeves.'}",
                "_Educational momentum metrics — not a buy list._",
            ]
            return "\n".join(parts)

    # ── Risk management snapshot ────────────────────────────────────
    if re.search(
        r"\b(risk management|position siz|value at risk|\bvar\b|kelly|recovery trauma|"
        r"equity curve|portfolio variance|percent risk)\b",
        q,
    ) and not re.search(r"\b(usdinr|mcx gold|bull call|straddle)\b", q):
        rm = p0.get("risk_management") if isinstance(p0.get("risk_management"), dict) else {}
        if rm.get("ok") or symbol:
            pr = rm.get("percent_risk_size") if isinstance(rm.get("percent_risk_size"), dict) else {}
            pv = rm.get("percent_vol_size") if isinstance(rm.get("percent_vol_size"), dict) else {}
            rec = rm.get("recovery_table_pct") if isinstance(rm.get("recovery_table_pct"), dict) else {}
            parts = [
                f"**{symbol or 'Account'} — risk management paper card**",
                "",
                f"• Assumed equity ₹{_fmt(rm.get('equity_assumed'))} | risk/trade {_fmt(rm.get('risk_pct_assumed'))}%",
                f"• Daily vol (from HV20)≈{_fmt(rm.get('daily_vol_pct_from_hv20'), 3)}% | "
                f"VaR 1d~95%≈₹{_fmt(rm.get('var_1d_95_rs'))} | 10d≈₹{_fmt(rm.get('var_10d_95_rs'))}",
            ]
            if pr.get("qty") is not None:
                parts.append(
                    f"• % risk size: risk₹={_fmt(pr.get('risk_rupees'))} → qty≈{pr.get('qty')} "
                    f"(notional≈{_fmt(pr.get('notional'))})"
                )
            if pv.get("qty") is not None:
                parts.append(
                    f"• % volatility size (ATR×1.5): qty≈{pv.get('qty')}"
                )
            if rm.get("kelly_quarter") is not None:
                parts.append(
                    f"• Kelly@45%WR: full≈{_fmt(rm.get('kelly_full_at_45pct_wr'), 3)} | "
                    f"quarter≈{_fmt(rm.get('kelly_quarter'), 3)} | "
                    f"E[R]≈{_fmt(rm.get('expectancy_r_at_45pct_wr'), 3)}"
                )
            if rec:
                parts.append(
                    f"• Recovery needed after losses: −5%→+{_fmt(rec.get('lose_5_need'))}% | "
                    f"−10%→+{_fmt(rec.get('lose_10_need'))}% | −20%→+{_fmt(rec.get('lose_20_need'))}% | "
                    f"−50%→+{_fmt(rec.get('lose_50_need'))}%"
                )
            if rm.get("diversification_note"):
                parts.append(f"• {rm.get('diversification_note')}")
            parts.extend(
                [
                    "",
                    f"Note: {rm.get('note') or 'Educational risk math — not a guarantee.'}",
                    "_Varsity-style risk management for paper practice — not advice._",
                ]
            )
            return "\n".join(parts)

    # ── Currency / Commodity / G-Sec snapshot ───────────────────────
    if re.search(
        r"\b(usdinr|usd/?inr|currency pair|mcx|ncdex|gold|crude|silver|natural gas|"
        r"g-?sec|treasury bill|t-bills?|commodity trading|interest rate parity)\b",
        q,
    ) and not re.search(r"\b(bull call|straddle|iron condor|option greeks)\b", q):
        try:
            from .analysis_math import fetch_ccg_market_snapshot, parse_gsec_symbol, tbill_discount_yield

            snap = fetch_ccg_market_snapshot()
            parts = ["**Currency / Commodity / G-Sec notes**", ""]
            if snap.get("ok"):
                parts.extend(
                    [
                        f"• USDINR≈{_fmt(snap.get('usdinr'), 4)} | IRP 30d fair≈{_fmt(snap.get('usdinr_30d_irp_fair'), 4)}",
                        f"• Gold≈{_fmt(snap.get('gold_usd_oz'))} $/oz → India≈₹{_fmt(snap.get('india_gold_rs_per_10g_approx'))}/10g (approx)",
                        f"• Crude≈{_fmt(snap.get('crude_usd_bbl'))} $/bbl | Silver≈{_fmt(snap.get('silver_usd_oz'))} $/oz",
                    ]
                )
                gedu = snap.get("gold_mcx_big_edu") or {}
                if gedu.get("pnl_per_tick_rs") is not None:
                    parts.append(
                        f"• Gold contract edu: value≈{_fmt(gedu.get('contract_value'))} | "
                        f"P&L/tick≈{_fmt(gedu.get('pnl_per_tick_rs'))}"
                    )
            if re.search(r"\bt-?bill", q):
                y = tbill_discount_yield(100, 97, 91)
                parts.append(f"• T-bill yield illustration (97→100 in 91d)≈{_fmt(y, 2)}% annualised form")
            gsec_m = re.search(r"\b(\d{3,4}GS\d{4}[A-Z]?)\b", q.upper())
            if gsec_m:
                parsed = parse_gsec_symbol(gsec_m.group(1))
                if parsed.get("ok"):
                    parts.append(
                        f"• Parsed {gsec_m.group(1)}: coupon≈{parsed.get('coupon_pct')}% "
                        f"(semi≈{parsed.get('semi_annual_pct')}%) maturity≈{parsed.get('maturity_year')}"
                    )
            parts.extend(
                [
                    "",
                    "_Educational CCG math/quotes — confirm NSE/MCX/RBI live specs. Not advice._",
                ]
            )
            if snap.get("ok") or gsec_m or re.search(r"\bt-?bill", q):
                return "\n".join(parts)
        except Exception:
            pass

    # ── Option theory / Greeks snapshot ─────────────────────────────
    theory_ask = bool(
        re.search(
            r"\b(option theory|greeks?|delta|gamma|theta|vega|moneyness|intrinsic|"
            r"time value|historical volatility|\bhv\b|black.?scholes|call option|"
            r"put option|option premium)\b",
            q,
        )
    ) and not re.search(
        r"\b(bull call|bull put|bear put|bear call|straddle|strangle|iron condor|"
        r"ratio back|synthetic long|option strateg)\b",
        q,
    )
    if theory_ask:
        fo = p0.get("fo") if isinstance(p0.get("fo"), dict) else {}
        ot = fo.get("option_theory") if isinstance(fo.get("option_theory"), dict) else {}
        if ot.get("ok"):
            call = ot.get("call") if isinstance(ot.get("call"), dict) else {}
            put = ot.get("put") if isinstance(ot.get("put"), dict) else {}
            parts = [
                f"**{symbol or 'Underlying'} — option theory snapshot (edu BS / HV σ)**",
                "",
                f"• Spot≈{_fmt(call.get('spot') or price or p0.get('price'))} | "
                f"ATM strike≈{_fmt(ot.get('atm_strike'))} | T≈{ot.get('days_to_expiry')}d | "
                f"HV≈{_fmt(ot.get('hv_pct'))}%",
                f"• Call: moneyness={call.get('moneyness')} | theo≈{_fmt(call.get('theoretical_premium'))} | "
                f"IV={_fmt(call.get('intrinsic'))} TV={_fmt(call.get('time_value'))} | "
                f"Δ={_fmt(call.get('delta'), 3)} Γ={_fmt(call.get('gamma'), 5)} "
                f"Θ/day={_fmt(call.get('theta_per_day'), 3)} Vega/volpt={_fmt(call.get('vega_per_vol_point'), 3)}",
                f"• Put: moneyness={put.get('moneyness')} | theo≈{_fmt(put.get('theoretical_premium'))} | "
                f"Δ={_fmt(put.get('delta'), 3)} Θ/day={_fmt(put.get('theta_per_day'), 3)}",
                f"• Expiry BE (theo prem): Call≈{_fmt(ot.get('call_breakeven'))} | "
                f"Put≈{_fmt(ot.get('put_breakeven'))}",
            ]
            cbs = ot.get("call_buyer_pnl_at_expiry_sample") or {}
            if cbs:
                parts.append(
                    f"• Sample call-buyer expiry P&L (theo): unchanged={_fmt(cbs.get('unchanged'))} | "
                    f"+2%={_fmt(cbs.get('up_2pct'))} | −2%={_fmt(cbs.get('down_2pct'))}"
                )
            parts.extend(
                [
                    "",
                    f"Note: {(call.get('note') or 'HV as σ — not live market IV.')}",
                    "_Varsity-style option theory math for paper practice — not advice._",
                ]
            )
            return "\n".join(parts)

    # ── Option strategies (payoff cards) ────────────────────────────
    strat_ask = bool(
        re.search(
            r"\b(bull call|bull put|bear put|bear call|straddle|strangle|iron condor|"
            r"ratio back|synthetic long|option strateg|max pain|pcr|put call ratio)\b",
            q,
        )
    )
    if strat_ask:
        fo = p0.get("fo") if isinstance(p0.get("fo"), dict) else {}
        os_ = fo.get("option_strategies") if isinstance(fo.get("option_strategies"), dict) else {}
        # Pure literacy (no symbol / no pack) falls through to education layer upstream.
        if os_.get("ok"):
            def _pay(v: Any) -> str:
                if isinstance(v, str):
                    return v.replace("_", " ")
                return _fmt(v)

            def _card(name: str, key: str) -> str:
                c = os_.get(key) if isinstance(os_.get(key), dict) else {}
                if not c:
                    return ""
                bits = [f"**{name}**"]
                if c.get("view"):
                    bits.append(f"view={c.get('view')}")
                if c.get("max_loss") is not None:
                    bits.append(f"maxLoss={_pay(c.get('max_loss'))}")
                if c.get("max_profit") is not None:
                    bits.append(f"maxProfit={_pay(c.get('max_profit'))}")
                if c.get("breakeven") is not None:
                    bits.append(f"BE={_fmt(c.get('breakeven'))}")
                if c.get("breakeven_up") is not None:
                    bits.append(
                        f"BE↑={_fmt(c.get('breakeven_up'))} BE↓={_fmt(c.get('breakeven_down'))}"
                    )
                if c.get("lower_breakeven") is not None:
                    bits.append(
                        f"BE↓={_fmt(c.get('lower_breakeven'))} BE↑={_fmt(c.get('upper_breakeven'))}"
                    )
                return " · ".join(bits)

            # Pick the strategy mentioned, else show a short menu.
            wanted: list[tuple[str, str]] = []
            mapping = [
                (r"bull call", "Bull call spread", "bull_call_spread"),
                (r"bull put", "Bull put spread", "bull_put_spread"),
                (r"bear put", "Bear put spread", "bear_put_spread"),
                (r"bear call spread|bear call\b", "Bear call spread", "bear_call_spread"),
                (r"short straddle", "Short straddle", "short_straddle"),
                (r"long straddle|\bstraddle\b", "Long straddle", "long_straddle"),
                (r"short strangle", "Short strangle", "short_strangle"),
                (r"long strangle|\bstrangle\b", "Long strangle", "long_strangle"),
                (r"call ratio", "Call ratio back spread", "call_ratio_back_spread"),
                (r"put ratio", "Put ratio back spread", "put_ratio_back_spread"),
                (r"iron condor", "Iron condor", "iron_condor"),
                (r"synthetic", "Synthetic long", "synthetic_long"),
            ]
            for pat, title, key in mapping:
                if re.search(pat, q):
                    wanted.append((title, key))
            if not wanted:
                wanted = [
                    ("Bull call spread", "bull_call_spread"),
                    ("Long straddle", "long_straddle"),
                    ("Iron condor", "iron_condor"),
                ]
            parts = [
                f"**{symbol or 'Underlying'} — option strategy paper cards**",
                "",
                f"• Spot≈{_fmt(os_.get('spot') or price or p0.get('price'))} | "
                f"ATM≈{_fmt(os_.get('atm_strike'))} | width={_fmt(os_.get('width'))} | "
                f"ATM prem proxy≈{_fmt(os_.get('atm_premium_proxy'))}",
                "",
            ]
            for title, key in wanted[:4]:
                line = _card(title, key)
                if line:
                    parts.append(f"• {line}")
            if re.search(r"\b(max pain|pcr|put call)\b", q):
                parts.append(f"• {os_.get('max_pain_note')}")
                parts.append(f"• {os_.get('pcr_note')}")
            parts.extend(
                [
                    "",
                    f"Note: {os_.get('note') or 'Educational premiums — not live chain.'}",
                    "_Varsity-style payoff math for paper practice — not trade advice._",
                ]
            )
            return "\n".join(parts)

    if (profile or "") == "derivatives" and re.search(r"\bexpir", q) and not symbol:
        return (
            "**F&O · expiry (educational)**\n\n"
            f"**Your ask:** {(query or '').strip()[:90]}\n\n"
            "Nifty and Bank Nifty options follow the exchange expiry calendar "
            "(weekly plus the monthly contract).\n"
            "Confirm the live expiry on the NSE F&O chain before any paper size.\n"
            "_Educational — not a live option chain / not SPAN._"
        )

    # ── Derivatives / F&O educational numbers ───────────────────────
    if intent == "derivatives" or (profile or "") == "derivatives" or re.search(
        r"\b(f&o|fno|futures?|options?|lot size|margin|iv|greeks|basis|straddle)\b", q
    ):
        fo = p0.get("fo") if isinstance(p0.get("fo"), dict) else {}
        if symbol or fo:
            parts = [
                f"**{symbol or 'Index'} — {_shape_heading(query, 'derivatives') or 'F&O paper-practice snapshot'}**",
                "",
                f"• Spot: {_fmt(price or p0.get('price'))}",
                f"• Indicative lot size: {fo.get('lot_size', 'n/a')} ({fo.get('lot_source', 'n/a')})",
                f"• Notional / lot: {_fmt(fo.get('notional_per_lot'))}",
                f"• Indicative margin / lot (~{fo.get('indicative_margin_pct', 0.15):.0%}): "
                f"{_fmt(fo.get('indicative_margin_per_lot'))}",
                f"• HV20 (IV proxy): {_fmt(fo.get('hv20_pct'))}% | HV60: {_fmt(fo.get('hv60_pct'))}%",
                f"• Futures fair (30d CoC≈{fo.get('rf_carry', 0.065):.1%}): "
                f"{_fmt(fo.get('futures_fair_30d'))} | "
                f"basis≈{_fmt(fo.get('basis_fair_30d'))} ({fo.get('basis_regime') or 'n/a'})",
                f"• Pricing: {fo.get('pricing_formula') or 'F ≈ S*(1+Rf*T/365) − D'}",
                f"• Leverage≈{_fmt(fo.get('leverage'), 2)}× | "
                f"≈{_fmt(fo.get('margin_wipeout_pct'), 2)}% adverse move can wipe indicative margin",
                f"• Sample MTM / lot (±1%): +{_fmt(fo.get('mtm_pnl_long_lot_plus_1pct'))} / "
                f"{_fmt(fo.get('mtm_pnl_long_lot_minus_1pct'))}",
            ]
            hedge = fo.get("hedge_example_1L") if isinstance(fo.get("hedge_example_1L"), dict) else {}
            if hedge.get("lots_exact") is not None:
                parts.append(
                    f"• Beta-hedge example (₹10L portfolio): hedge value≈{_fmt(hedge.get('hedge_value'))} → "
                    f"~{_fmt(hedge.get('lots_exact'), 2)} Nifty lots "
                    f"(round {hedge.get('lots_floor')}–{hedge.get('lots_ceil')}; short futures)"
                )
            if fo.get("atm_premium_approx_21d") is not None:
                parts.append(
                    f"• ATM premium approx (21d, HV): {_fmt(fo.get('atm_premium_approx_21d'))} | "
                    f"Call BE≈{_fmt(fo.get('call_breakeven_atm'))} | Put BE≈{_fmt(fo.get('put_breakeven_atm'))}"
                )
            parts.extend(
                [
                    "",
                    f"Note: {fo.get('note') or 'Confirm live NSE lot/SPAN before any paper F&O size.'}",
                    "_Educational F&O math — not a live option chain / not SPAN._",
                ]
            )
            return "\n".join(parts)

    # ── Portfolio / SIP / sizing / personal finance ─────────────────
    if intent == "portfolio" or (profile or "") == "portfolio" or re.search(
        r"\b(sip|allocation|diversif|position size|kelly|personal finance|"
        r"retirement|emergency fund|mutual fund|my watchlist|on my (watch)?list)\b",
        q,
    ):
        pc = ctx.get("portfolio_context") if isinstance(ctx.get("portfolio_context"), dict) else {}
        holdings_syms = [str(s).upper() for s in (pc.get("symbols") or []) if s]
        watch = [str(s).upper() for s in (pc.get("watchlist") or []) if s]
        qty_map = pc.get("concentrations") if isinstance(pc.get("concentrations"), dict) else {}
        list_ask = bool(
            re.search(
                r"\b(my watchlist|on my (watch)?list|what('?s| is) on my list|"
                r"my (portfolio|holdings))\b",
                q,
            )
        )
        if list_ask or holdings_syms or watch:
            named: list[str] = []
            for sym in holdings_syms[:16]:
                qty = qty_map.get(sym) or qty_map.get(sym.lower())
                named.append(f"{sym}×{int(qty)}" if qty not in (None, "") else sym)
            list_parts = [
                "**On your list**",
                "",
                f"• Holdings: {', '.join(named) if named else 'none yet'}",
                f"• Watchlist: {', '.join(watch[:16]) if watch else 'none yet'}",
                "",
                "Symbols only — book last is not live P&L and we do not invent a return.",
                "_Educational paper book — not investment advice._",
            ]
            if list_ask:
                return "\n".join(list_parts)
        plan = p0.get("trade_plan") if isinstance(p0.get("trade_plan"), dict) else {}
        costs = p0.get("india_costs") if isinstance(p0.get("india_costs"), dict) else {}
        tax = p0.get("tax_drag_example") if isinstance(p0.get("tax_drag_example"), dict) else {}
        pf_cue = bool(
            re.search(
                r"\b(personal finance|retirement|emergency fund|mutual fund|tvm|"
                r"time value|expense ratio|asset allocation)\b",
                q,
            )
        )
        parts = [
            f"**{'Personal finance / SIP' if pf_cue and not symbol else ('Portfolio / sizing' if not symbol else symbol + ' — sizing cues')}**",
            "",
            f"• SIP example (₹5k/mo @12% for 10y): {_fmt(p0.get('sip_example_5k_12pct_10y'))}",
        ]
        if pf_cue:
            try:
                from indian_stock_llm.analysis_math import build_personal_finance_pack

                pf = build_personal_finance_pack()
                ef = pf.get("emergency_fund_3_to_12_months") or {}
                parts.extend(
                    [
                        f"• Real return (12% nom / 6% infl)≈{_fmt(pf.get('real_return_pct'), 2)}%",
                        f"• Rule of 72 years-to-double≈{_fmt(pf.get('years_to_double_rule72'), 2)}",
                        f"• Retirement corpus (4% rule, ₹50k/mo exp)≈₹{_fmt(pf.get('retirement_corpus_4pct_rule'))}",
                        f"• Emergency fund band≈₹{_fmt(ef.get('low'))}–₹{_fmt(ef.get('high'))}",
                    ]
                )
            except Exception:
                pass
        parts.extend(
            [
                f"• Kelly quarter-cap fraction: {_fmt(plan.get('kelly_fraction_capped') or p0.get('kelly_quarter'), 3)}",
                f"• Qty for ₹{(plan.get('position_qty_for_risk') and (p0.get('atr_stop') or {}).get('risk_rupees')) or 5000} "
                f"risk: {plan.get('position_qty_for_risk') or (p0.get('atr_stop') or {}).get('qty_for_risk', 'n/a')}",
            ]
        )
        if costs.get("roundtrip_cost_pct_note"):
            parts.append(f"• Costs: {costs.get('roundtrip_cost_pct_note')}")
        if tax.get("note"):
            parts.append(f"• Tax: {tax.get('note')}")
        if plan.get("action"):
            parts.append(
                f"• Current paper stance: {plan.get('action')} | stop={_fmt(plan.get('stop'))} | "
                f"T1={_fmt(plan.get('target_1'))}"
            )
            meaning = _action_meaning(plan.get("action"))
            if meaning:
                parts.append(f"• Meaning: {meaning}")
            parts.append(_ACTION_LEGEND_LINE)
        parts.extend(
            [
                "",
                "Size from risk budget / goals first; never from FOMO. Educational only.",
                "_Educational only — not investment advice._",
            ]
        )
        return "\n".join(parts)

    # ── Named-symbol asks — profile decides which sections to render ──
    if symbol and (
        intent in {
            "price_action",
            "stock_analysis",
            "prediction",
            "fundamentals",
            "overbought_check",
            "general_query",
            "events_news",
            "market_calculations",
        }
        or (profile or "") in {
            "quote",
            "news",
            "sentiment",
            "technical",
            "trade_plan",
            "prediction",
            "fundamentals",
            "risks",
            "stock_analysis",
            "compare",
            "corporate_actions",
        }
        or stop_loss_ask
        or re.search(r"\b(should i buy|should i sell|buy or sell|trade plan|swing trade)\b", q)
    ):
        profile = (profile or "").strip() or resolve_stock_response_profile(
            q if stop_loss_ask else query, intent
        )
        if stop_loss_ask:
            profile = "trade_plan"
        ask_line = _ask_line(query)

        company = str(ctx.get("company_name") or "").strip()
        header_base = f"**{symbol}**" + (f" — {company}" if company else "")
        plan = p0.get("trade_plan") if isinstance(p0.get("trade_plan"), dict) else {}
        root_plan = ctx.get("trade_plan") if isinstance(ctx.get("trade_plan"), dict) else {}
        if root_plan and (not plan or not plan.get("action")):
            plan = root_plan
        support = levels.get("support")
        resistance = levels.get("resistance")
        stop = plan.get("stop") or levels.get("stop_loss")
        take = plan.get("target_1") or levels.get("take_profit")
        pct = ctx.get("pct_change") or ctx.get("pctChange") or tech.get("pct_change")
        if _is_short_followup(query):
            return _short_followup_card(
                symbol=symbol,
                query=query,
                price=price,
                tech=tech if isinstance(tech, dict) else {},
                support=support,
                resistance=resistance,
            )

        def _append_news_block(parts: list[str], *, max_heads: int = 4) -> None:
            news_heads = list(ctx.get("news_headlines") or [])
            if not news_heads:
                for ev in sentiment.get("recent_events") or []:
                    if isinstance(ev, str) and ev.strip():
                        news_heads.append(ev.strip())
            if not news_heads:
                for h in (sentiment_pack.get("news") or {}).get("headlines") or []:
                    if isinstance(h, dict) and h.get("title"):
                        news_heads.append(str(h.get("title")))
                    elif isinstance(h, str) and h.strip():
                        news_heads.append(h.strip())
            tagged = list((sentiment_pack.get("news") or {}).get("tagged") or [])
            if not tagged:
                tagged = list(sentiment.get("tagged") or [])
            company_name = str(ctx.get("company_name") or "")
            news_heads = [
                h
                for h in news_heads
                if _headline_about_symbol(str(h), symbol, company_name)
            ]
            tagged = [
                t
                for t in tagged
                if isinstance(t, dict)
                and _headline_about_symbol(str(t.get("title") or ""), symbol, company_name)
            ]
            sector_trend = sentiment.get("sector_trend")
            if not (news_heads or tagged or sector_trend):
                return
            parts.append("")
            parts.append("**News & catalysts:**")
            if sector_trend:
                parts.append(f"• Sector/tape cue: {sector_trend}")
            shown = 0
            for t in tagged[:max_heads]:
                if not isinstance(t, dict) or not t.get("title"):
                    continue
                pol = t.get("polarity") or "neutral"
                parts.append(f"• ({pol}) {str(t.get('title')).strip()[:150]}")
                shown += 1
            if shown == 0:
                for h in news_heads[:max_heads]:
                    parts.append(f"• {str(h).strip()[:150]}")

        def _append_sentiment_block(parts: list[str]) -> None:
            if not (
                sentiment_pack.get("ok")
                or sentiment.get("overall")
                or sentiment.get("composite_score") is not None
            ):
                return
            parts.append("")
            parts.append("**Sentiment:**")
            label = sentiment_pack.get("label") or sentiment.get("overall") or "n/a"
            score = sentiment_pack.get("composite_score")
            if score is None:
                score = sentiment.get("composite_score")
            conf = sentiment_pack.get("confidence") or sentiment.get("confidence")
            score_txt = f"{score:+.2f}" if isinstance(score, (int, float)) else "n/a"
            parts.append(
                f"• Overall: **{label}** ({score_txt}"
                + (f", confidence {conf}" if conf is not None else "")
                + ")"
            )
            summary = sentiment_pack.get("summary") or sentiment.get("summary")
            if summary:
                parts.append(f"• Read: {summary}")
            for fac in (sentiment_pack.get("factors") or sentiment.get("factors") or [])[:4]:
                if isinstance(fac, dict) and fac.get("label"):
                    parts.append(f"• {fac.get('name')}: {fac.get('label')}")

        # ── Focused profiles (query-aware shapes) ───────────────────
        if profile == "quote":
            parts = [_named_title(header_base, query, "quote", "live quote")]
            if ask_line:
                parts.append(ask_line)
            open_px = ctx.get("open") or tech.get("open")
            high_px = ctx.get("high") or tech.get("high") or levels.get("day_high")
            low_px = ctx.get("low") or tech.get("low") or levels.get("day_low")
            prev_px = ctx.get("prev_close") or ctx.get("previousClose") or tech.get("prev_close")
            volume = ctx.get("volume") or tech.get("volume")
            parts.extend([
                "",
                f"• Last: ₹{_fmt(price)}"
                + (f" ({_fmt(pct)}%)" if _num(pct) is not None else ""),
            ])
            if any(_num(v) is not None for v in (open_px, high_px, low_px)):
                parts.append(
                    f"• Open / High / Low: {_fmt(open_px)} / {_fmt(high_px)} / {_fmt(low_px)}"
                )
            if _num(prev_px) is not None:
                parts.append(f"• Prev close: {_fmt(prev_px)}")
            if volume not in (None, "", 0, "0", "n/a"):
                parts.append(f"• Volume: {volume}")
            if _num(support) is not None or _num(resistance) is not None:
                parts.append(
                    f"• Support / Resistance: {_fmt(support)} / {_fmt(resistance)}"
                )
            parts.extend([
                "",
                "Ask for technicals, news, valuation, or a paper trade plan if you need more depth.",
                "_Live quote snapshot — educational only._",
            ])
            return "\n".join(parts)

        if profile == "news":
            parts = [_named_title(header_base, query, "news", "news & catalysts")]
            if ask_line:
                parts.append(ask_line)
            parts.append("")
            _append_news_block(parts, max_heads=6)
            if len(parts) <= (3 if ask_line else 2):
                parts.append("• No fresh headlines in the current feed — try again shortly.")
            parts.extend(
                [
                    "",
                    f"• Price context: ₹{_fmt(price)} | RSI {_fmt(tech.get('rsi'))}",
                    "_News digest only — not a buy/sell call._",
                ]
            )
            return "\n".join(parts)

        if profile == "sentiment":
            parts = [_named_title(header_base, query, "sentiment", "sentiment view")]
            if ask_line:
                parts.append(ask_line)
            parts.append("")
            _append_sentiment_block(parts)
            _append_news_block(parts, max_heads=3)
            if len(parts) <= (3 if ask_line else 2):
                parts.append("• Sentiment pack unavailable right now.")
            parts.extend(
                [
                    "",
                    f"• Tape: ₹{_fmt(price)} | trend {tech.get('trend') or 'n/a'}",
                    "_Sentiment framing for paper practice — not investment advice._",
                ]
            )
            return "\n".join(parts)

        if profile == "risks":
            parts = [_named_title(header_base, query, "risks", "key risks now")]
            if ask_line:
                parts.append(ask_line)
            parts.append("")
            rsi = _num(tech.get("rsi"))
            if rsi is not None:
                if rsi >= 70:
                    parts.append(f"• Momentum risk: RSI {_fmt(rsi)} is stretched — chase risk is elevated.")
                elif rsi <= 30:
                    parts.append(f"• Tape risk: RSI {_fmt(rsi)} is washed out — volatility can stay high.")
                else:
                    parts.append(f"• Momentum: RSI {_fmt(rsi)} is mid-range, not an extreme.")
            pe = fund.get("pe")
            if pe not in (None, "", "n/a"):
                parts.append(f"• Valuation: P/E {_fmt(pe)} — a rich multiple leaves less room for disappointment.")
            parts.append(
                f"• Level risk: support {_fmt(support)} / resistance {_fmt(resistance)} "
                f"(stop idea {_fmt(stop)})"
            )
            _append_news_block(parts, max_heads=3)
            if len(parts) <= 2:
                parts.append("• No extra risk flags in the current pack — still size from a stop, not conviction.")
            parts.extend(
                [
                    "",
                    f"• Spot: ₹{_fmt(price)} | trend {tech.get('trend') or 'n/a'}",
                    "_Risk list for paper practice — not a buy/sell call._",
                ]
            )
            return "\n".join(parts)

        if profile == "technical":
            asked = (query or "").lower()
            single_indicator = bool(
                re.search(r"\b(rsi|macd|pe ratio|p/e|p\/e|vwap|atr|supertrend|bollinger)\b", asked)
            ) and not re.search(r"\b(buy|sell|plan|should i|trade|entry|stop)\b", asked)
            if single_indicator:
                bits = [f"**{symbol or 'This symbol'} · indicator**", ""]
                if re.search(r"\brsi\b", asked):
                    bits.append(
                        f"• RSI: {_fmt(tech.get('rsi'))}"
                        + (f" — {tech.get('rsi_interpretation')}" if tech.get("rsi_interpretation") else "")
                    )
                if re.search(r"\bmacd\b", asked):
                    bits.append(f"• MACD hist: {_fmt(tech.get('macd_hist') or (p0.get('macd') or {}).get('histogram'))}")
                if re.search(r"\b(pe ratio|p/e|p\/e)\b", asked):
                    bits.append(f"• P/E: {_fmt(fund.get('pe'))}")
                if re.search(r"\bvwap\b", asked):
                    bits.append(f"• VWAP: {_fmt(tech.get('vwap'))}")
                if re.search(r"\batr\b", asked):
                    bits.append(f"• ATR: {_fmt(tech.get('atr'))}")
                if re.search(r"\bsupertrend\b", asked):
                    bits.append(f"• Supertrend: {tech.get('supertrend') or (p0.get('supertrend') or {}).get('direction') or 'n/a'}")
                if re.search(r"\bbollinger\b", asked):
                    bits.append(f"• Bollinger: {_fmt(tech.get('bb_lower'))} / {_fmt(tech.get('bb_upper'))}")
                if len(bits) == 2:
                    bits.append("• That indicator is not in the current pack.")
                bits.extend(["", f"Last: ₹{_fmt(price)}. The print is delayed.", "_One indicator — not a buy or sell plan._"])
                return "\n".join(bits)
            parts = [_named_title(header_base, query, "technical", "technical analysis")]
            if ask_line:
                parts.append(ask_line)
            parts.extend([
                "",
                "**Technical readout:**",
                f"• Price: ₹{_fmt(price)} | Trend: {tech.get('trend') or 'n/a'}",
                f"• RSI: {_fmt(tech.get('rsi'))}"
                + (
                    f" — {tech.get('rsi_interpretation')}"
                    if tech.get("rsi_interpretation")
                    else ""
                ),
            ])
            st_dir = tech.get("supertrend") or (p0.get("supertrend") or {}).get("direction")
            macd_h = tech.get("macd_hist") or (p0.get("macd") or {}).get("histogram")
            if st_dir or _num(macd_h) is not None:
                parts.append(
                    f"• Supertrend: {st_dir or 'n/a'} | MACD hist: {_fmt(macd_h)}"
                )
            parts.extend([
                f"• Support: {_fmt(support)} | Resistance: {_fmt(resistance)}"
                if _num(support) is not None or _num(resistance) is not None
                else "• No live support or resistance in the current tape.",
                f"• Stop idea: {_fmt(stop)} | Target idea: {_fmt(take)}",
            ])
            parts.extend(
                [
                    "",
                    "Technical structure only — ask “Should I buy …?” for a paper trade plan.",
                    "_Educational only — not investment advice._",
                ]
            )
            return "\n".join(parts)

        if profile == "prediction":
            ps = ctx.get("prediction_signals") if isinstance(ctx.get("prediction_signals"), dict) else {}
            scenarios = (ps or {}).get("scenarios") if isinstance(ps, dict) else {}
            parts = [_named_title(header_base, query, "prediction", "scenario range")]
            if ask_line:
                parts.append(ask_line)
            parts.extend([
                "",
                "**Not a price guarantee** — if/then bands from live support and resistance.",
                f"• Spot: ₹{_fmt(price)} | Trend: {tech.get('trend') or 'n/a'} | RSI {_fmt(tech.get('rsi'))}",
                f"• Near support / resistance: {_fmt(support)} / {_fmt(resistance)}",
            ])
            if (ps or {}).get("advice"):
                parts.append(f"• Investment call: **{ps.get('advice')}** (from tape + structure; not a guarantee)")
            if (ps or {}).get("event_blackout"):
                parts.append(f"• Event window: {ps.get('event_blackout')}")
            if isinstance(scenarios, dict) and scenarios:
                hold = scenarios.get("support_hold")
                brk = scenarios.get("support_break")
                if hold:
                    parts.append(f"• If support holds: {hold}")
                if brk:
                    parts.append(f"• If support breaks: {brk}")
                band = scenarios.get("swing_band") or scenarios.get("intraday_band")
                if band:
                    parts.append(f"• Volatility band: {band}")
            elif plan.get("target_1") or plan.get("target_2"):
                parts.append(
                    f"• Practice levels from structure: T1 {_fmt(plan.get('target_1'))} | "
                    f"T2 {_fmt(plan.get('target_2'))} | invalidation {_fmt(stop)}"
                )
            if ps.get("intraday"):
                intra = ps.get("intraday") or {}
                swing = ps.get("swing") or {}
                parts.append(
                    f"• Tape bias (estimate): intraday {intra.get('direction', 'n/a')} "
                    f"(~{float(intra.get('probability') or 0):.0%}) · "
                    f"swing {swing.get('direction', 'n/a')} "
                    f"(~{float(swing.get('probability') or 0):.0%})"
                )
            parts.extend(
                [
                    "",
                    "Forecasts fail often near events — size from stop distance, not conviction.",
                    "_Investment view from live tape + structure — not a SEBI RA guarantee._",
                ]
            )
            return "\n".join(parts)

        if profile == "fundamentals":
            pe = fund.get("pe")
            pb = fund.get("pb")
            roe = fund.get("roe")
            eps = fund.get("eps")
            parts = [_named_title(header_base, query, "fundamentals", "fundamental snapshot")]
            if ask_line:
                parts.append(ask_line)
            parts.extend([
                "",
                f"• Last price: {_fmt(price)}",
                f"• P/E: {_fmt(pe)}"
                + (
                    f" (sector cue: {fund.get('pe_sector_avg')})"
                    if fund.get("pe_sector_avg")
                    else ""
                ),
                f"• P/B: {_fmt(pb)}",
                f"• ROE: {_fmt(roe)}",
                f"• EPS: {_fmt(eps)}",
                f"• Market cap: {fund.get('market_cap') or 'n/a'}",
                f"• Dividend yield: {_safe_div_yield(fund.get('dividend_yield'))}",
            ])
            if fund.get("week_52"):
                parts.append(f"• 52-week context: {fund.get('week_52')}")
            parts.extend(
                [
                    "",
                    "Interpret P/E vs peers and growth — a low P/E alone is not cheap.",
                    "_Educational snapshot — not investment advice._",
                ]
            )
            return "\n".join(parts)

        if profile == "stock_analysis":
            how = (
                f"**How it's doing:** Last ₹{_fmt(price)}"
                + (f" ({_fmt(pct)}%)" if _num(pct) is not None else "")
                + f" · trend {tech.get('trend') or 'n/a'} · RSI {_fmt(tech.get('rsi'))}"
            )
            parts = [_named_title(header_base, query, "stock_analysis", "snapshot")]
            if ask_line:
                parts.append(ask_line)
            parts.extend(
                [
                    "",
                    how,
                ]
            )
            if _num(support) is not None or _num(resistance) is not None:
                parts.append(
                    f"• Support / Resistance: {_fmt(support)} / {_fmt(resistance)}"
                )
            if fund.get("pe") not in (None, "", "n/a"):
                parts.append(f"• Valuation cue: P/E {_fmt(fund.get('pe'))}")
            _append_news_block(parts, max_heads=2)
            parts.extend(
                [
                    "",
                    f"Ask “Should I buy {symbol}?” if you want a paper entry, stop, and targets.",
                    "_Educational snapshot — not a buy/sell call._",
                ]
            )
            return "\n".join(parts)

        # Full paper-practice / trade plan / calculations keep the rich template.
        weekly = bool(re.search(r"\b(this week|weekly|swing)\b", q))
        want_full_math = profile == "calculations" or bool(
            re.search(
                r"\b(full math|all indicators|quant(?:itative)? stack|indicator stack|"
                r"show (all )?math|p0 math|every indicator)\b",
                q,
            )
            or intent in {"overbought_check", "market_calculations"}
        )
        include_trade_plan = profile in {"trade_plan", "calculations"}
        include_sentiment = profile in {"stock_analysis", "trade_plan"}
        include_debate = profile == "trade_plan"

        stance, rationale, score = _stance_from_tech(tech, fund, weekly=weekly)
        if plan.get("action"):
            action = str(plan["action"])
            stance = {
                "BUY": "BUY bias (paper) — wait for entry zone",
                "ACCUMULATE": "ACCUMULATE on dips (paper) — staged entries",
                "HOLD": "HOLD / wait — no clear edge yet",
                "TRIM": "TRIM / reduce on strength (paper)",
                "SELL": "SELL / avoid fresh buys (paper)",
                "WAIT": "WAIT — insufficient signal quality",
            }.get(action, stance)
            if plan.get("reasons_for"):
                rationale = "; ".join(_plain_reasons(plan["reasons_for"], 3))
            elif plan.get("reason"):
                rationale = _plain_reason(plan["reason"])
        rationale = _plain_reason(rationale)

        rsi = _num(tech.get("rsi"))
        if (
            weekly
            and rsi is not None
            and rsi >= 70
            and re.search(r"\b(buy|sell)\b", q)
            and not plan.get("action")
        ):
            stance = "SELL bias for short-term / avoid chasing; wait for cool-off toward support"
            rationale = f"{rationale}; short-horizon ask + overbought RSI"

        fallback_heading = {
            "trade_plan": "paper trade plan",
            "calculations": "live checks",
            "stock_analysis": "snapshot",
        }.get(profile, "snapshot")
        parts = [_named_title(header_base, query, profile, fallback_heading)]
        if ask_line:
            parts.append(ask_line)
        parts.extend([
            "",
            f"**Direct answer:** {stance}",
            f"**Why:** {rationale}",
        ])
        if profile == "stock_analysis":
            parts.append(
                f"Ask “Should I buy {symbol}?” if you want a paper entry, stop, and targets."
            )

        if include_trade_plan and plan.get("action"):
            ez = plan.get("entry_zone") or []
            entry_txt = f"{_fmt(ez[0])} – {_fmt(ez[1])}" if len(ez) == 2 else "n/a"
            meaning = _action_meaning(plan.get("action"))
            parts.extend(
                [
                    "",
                    f"**Action:** {plan.get('action')} "
                    f"({plan.get('horizon') or 'swing'})",
                ]
            )
            if meaning:
                parts.append(f"• Meaning: {meaning}")
            parts.extend(
                [
                    f"• Entry zone: {entry_txt}",
                    f"• Stop: {_fmt(stop)} | Target 1: {_fmt(plan.get('target_1'))} | "
                    f"Target 2: {_fmt(plan.get('target_2'))}",
                    f"• Risk/reward: {_fmt(plan.get('risk_reward'))} | "
                    f"suggested qty (risk budget): {plan.get('position_qty_for_risk')}",
                    "• Risk frame (3-5-7): ≤3% capital risk / trade · ≤5% total open risk · "
                    "prefer ~7%+ favorable room on winners (paper checklist)",
                ]
            )
            if plan.get("invalidation"):
                parts.append(f"• Invalidation: {plan.get('invalidation')}")
            parts.extend(["", _ACTION_LEGEND_LINE])

        if profile == "trade_plan" and symbol:
            try:
                from .agent_tools import paper_order_from_plan

                ticket_plan = plan if plan.get("action") else (ctx.get("trade_plan") or {})
                ticket = paper_order_from_plan(symbol, ticket_plan, stop=stop)
                if ticket and ticket.get("line"):
                    parts.extend(["", "**Paper ticket:**", ticket["line"]])
            except Exception:
                pass

        conv_summary = str(ctx.get("conversation_summary") or "").strip()
        user_tone = ctx.get("user_sentiment") if isinstance(ctx.get("user_sentiment"), dict) else {}
        if include_trade_plan and (conv_summary or user_tone):
            parts.append("")
            parts.append("**Context awareness:**")
            if conv_summary:
                parts.append("• Continuing from your recent chat turns on this topic.")
            urgency = str(user_tone.get("urgency") or "").lower()
            risk_ap = str(user_tone.get("risk_appetite") or "").lower()
            emotion = str(user_tone.get("emotion") or "").lower()
            if urgency == "high":
                parts.append("• Tone cue: time-sensitive ask — prefer staged size and hard stops.")
            if risk_ap == "conservative":
                parts.append("• Tone cue: conservative stance — wait for cleaner levels over chase entries.")
            elif risk_ap == "aggressive":
                parts.append("• Tone cue: aggressive stance — still size from stop distance, not conviction.")
            if emotion in {"frustrated", "anxious", "fearful"}:
                parts.append("• Tone cue: elevated stress — avoid revenge trades; re-check thesis first.")

        if include_sentiment:
            _append_sentiment_block(parts)
            _append_news_block(parts, max_heads=3)

        parts.extend(
            [
                "",
                "**Key levels & tape:**",
                f"• Price: {_fmt(price)}",
                f"• RSI: {_fmt(tech.get('rsi'))}"
                + (f" — {tech.get('rsi_interpretation')}" if tech.get("rsi_interpretation") else ""),
                f"• Trend: {tech.get('trend') or 'n/a'}",
                f"• Support: {_fmt(support)} | Resistance: {_fmt(resistance)}",
                f"• Stop idea: {_fmt(stop)} | Target idea: {_fmt(take)}",
            ]
        )
        st_dir = tech.get("supertrend") or (p0.get("supertrend") or {}).get("direction")
        macd_h = tech.get("macd_hist") or (p0.get("macd") or {}).get("histogram")
        if st_dir or macd_h is not None:
            parts.append(
                f"• Supertrend: {st_dir or 'n/a'} | MACD hist: {_fmt(macd_h)}"
            )
        if fund.get("pe") or fund.get("market_cap"):
            parts.append(
                f"• Fundamentals: P/E={_fmt(fund.get('pe'))}, "
                f"mcap={fund.get('market_cap') or 'n/a'}, "
                f"div={_safe_div_yield(fund.get('dividend_yield'))}"
            )

        if pre_signals and include_trade_plan:
            parts.append("")
            parts.append("**Signal highlights:**")
            seen: set[str] = set()
            for key in (
                "sentiment_signal",
                "rsi_signal",
                "trend_signal",
                "macd_signal",
                "ma_signal",
                "levels_signal",
                "level_signal",
                "valuation_signal",
                "pe_signal",
                "week52_signal",
            ):
                val = pre_signals.get(key)
                if val and val not in seen:
                    parts.append(f"• {val}")
                    seen.add(str(val))
                if len(seen) >= 5:
                    break

        if p0.get("ok") and want_full_math:
            parts.append("")
            parts.append("**Requested checks:**")
            bb = p0.get("bollinger") or {}
            atr_s = p0.get("atr_stop") or {}
            vs = p0.get("vs_nifty") or {}
            piv = p0.get("pivots_classic") or {}
            vol = p0.get("volume") or {}
            st = p0.get("supertrend") or {}
            fib = p0.get("fibonacci") or {}
            rs = p0.get("risk_stats") or {}
            parts.append(
                f"• RSI={_fmt(p0.get('wilder_rsi_14'))} | {p0.get('rsi_divergence') or 'n/a'}"
            )
            parts.append(
                f"• Bollinger %B={_fmt(bb.get('pct_b'), 3)} | bandwidth={_fmt(bb.get('bandwidth'), 4)} "
                f"({bb.get('bandwidth_regime') or 'n/a'})"
            )
            parts.append(
                f"• ATR(14)={_fmt(p0.get('atr_14'))} | ATR stop={_fmt(atr_s.get('stop'))} | "
                f"qty for ₹{atr_s.get('risk_rupees') or 5000} risk={atr_s.get('qty_for_risk', 'n/a')}"
            )
            parts.append(
                f"• Supertrend={st.get('direction')} @{_fmt(st.get('line'))} | "
                f"Fib 0.618={_fmt(fib.get('retracement_618'))} | VWAP20={_fmt(p0.get('vwap_20d'))}"
            )
            parts.append(
                f"• Pivots P={_fmt(piv.get('P'))} | S1={_fmt(piv.get('S1'))} | R1={_fmt(piv.get('R1'))}"
            )
            parts.append(
                f"• vs Nifty: RS20={_fmt(vs.get('rs_20d'), 3)} | β60="
                f"{_fmt(vs.get('beta_60d') or vs.get('beta_fallback'), 3)}"
            )
            parts.append(
                f"• Sortino60={_fmt(rs.get('sortino_60d'), 3)} | maxDD60="
                f"{_fmt(rs.get('max_drawdown_60d_pct'))}% | Vol z20={_fmt(vol.get('zscore_20'))}"
            )
            dq = p0.get("data_quality") or {}
            if dq.get("degraded"):
                parts.append("• Data quality: partial feeds — treat confidence as capped")
        elif p0.get("ok") and include_trade_plan and want_full_math:
            atr_s = p0.get("atr_stop") or {}
            parts.append("")
            parts.append("**Practice levels:**")
            parts.append(
                f"• Stop idea {_fmt(atr_s.get('stop') or stop)} | "
                f"RSI {_fmt(p0.get('wilder_rsi_14') or tech.get('rsi'))}"
            )

        if include_debate:
            parts.append("")
            parts.append("**For (setup):**")
            for item in _plain_reasons(plan.get("reasons_for"), 3):
                parts.append(f"• {item}")
            if not plan.get("reasons_for"):
                if rsi is not None and rsi <= 30:
                    parts.append(
                        f"• Oversold RSI {_fmt(rsi)} can favor staged paper entries near support."
                    )
                else:
                    parts.append("• Mix of signals — size small and demand confirmation at levels.")

            parts.append("")
            parts.append("**Against:**")
            for item in _plain_reasons(plan.get("reasons_against"), 3):
                parts.append(f"• {item}")
            if not plan.get("reasons_against"):
                parts.append(
                    "• Single-indicator calls fail in strong trends; confirm with volume/news."
                )
            if fund.get("pe") and _num(fund.get("pe")) and _num(fund.get("pe")) > 40:
                parts.append(
                    "• Rich valuation leaves less margin of safety on disappointment."
                )

        if deterministic and want_full_math:
            parts.extend(["", "**Computed checks**", deterministic])
        parts.extend(
            [
                "",
                "Educational paper view from live quotes — "
                "not investment advice / not a price guarantee.",
                "_Grounded by BYSEL._",
            ]
        )
        return "\n".join(parts)

    return None
