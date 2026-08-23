"""Buy / sell / hold trade-plan engine for BYSEL Indian Stock LLM.

Combines P0+B/C quantitative packs into an educational paper-practice plan:
  action, confidence, entry zone, stop, targets, invalidation, horizon.
Not SEBI advice — deterministic scoring over live math only.
"""
from __future__ import annotations

from typing import Any, Optional


def _f(value: Any) -> Optional[float]:
    try:
        if value is None or value == "" or value == "n/a":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _risk_reward(entry: float, stop: float, target: float) -> Optional[float]:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return reward / risk


def build_trade_plan(pack: dict[str, Any], *, horizon: str = "swing") -> dict[str, Any]:
    """Score signals and produce a structured trade plan."""
    if not pack or not pack.get("ok"):
        return {
            "action": "WAIT",
            "confidence": 0.0,
            "score": 0,
            "reason": "insufficient quantitative data",
            "educational": True,
        }

    price = _f(pack.get("price")) or 0.0
    rsi = _f(pack.get("wilder_rsi_14"))
    bb = pack.get("bollinger") or {}
    pct_b = _f(bb.get("pct_b"))
    bw_regime = str(bb.get("bandwidth_regime") or "")
    atr_s = pack.get("atr_stop") or {}
    stop = _f(atr_s.get("stop"))
    target_1r = _f(atr_s.get("target_1r"))
    structural_t1 = _f(atr_s.get("structural_t1"))
    structural_t2 = _f(atr_s.get("structural_t2"))
    rr = _f(atr_s.get("risk_reward"))
    vol = pack.get("volume") or {}
    vz = _f(vol.get("zscore_20"))
    delivery = _f(vol.get("delivery_pct"))
    vs = pack.get("vs_nifty") or {}
    rs20 = _f(vs.get("rs_20d"))
    beta60 = _f(vs.get("beta_60d")) or _f(vs.get("beta_fallback")) or _f(vs.get("yf_beta"))
    val = pack.get("valuation_math") or {}
    ey = _f(val.get("earnings_yield_pct"))
    pe = _f(val.get("pe"))
    ev = _f(val.get("ev_ebitda"))
    sector_prem = _f(val.get("sector_pe_premium_pct"))
    st = pack.get("supertrend") or {}
    st_dir = str(st.get("direction") or "")
    fib = pack.get("fibonacci") or {}
    sortino = _f((pack.get("risk_stats") or {}).get("sortino_60d"))
    max_dd = _f((pack.get("risk_stats") or {}).get("max_drawdown_60d_pct"))
    macd_hist = _f((pack.get("macd") or {}).get("histogram"))
    peg = _f(val.get("peg"))
    div = str(pack.get("rsi_divergence") or "")
    stoch_k = _f((pack.get("stochastic") or {}).get("k"))
    adx = _f(pack.get("adx_approx"))
    dq = pack.get("data_quality") or {}

    score = 0
    reasons: list[str] = []
    against: list[str] = []

    # RSI
    if rsi is not None:
        if rsi <= 30:
            score += 2
            reasons.append(f"Wilder RSI {_fmt(rsi)} oversold — bounce/watchlist zone")
        elif rsi >= 70:
            score -= 2
            against.append(f"Wilder RSI {_fmt(rsi)} overbought — avoid chasing")
        elif 45 <= rsi <= 60:
            score += 1
            reasons.append(f"Wilder RSI {_fmt(rsi)} mid-healthy")
        elif rsi > 60:
            score += 1
            reasons.append(f"Wilder RSI {_fmt(rsi)} bullish momentum")
        else:
            score -= 1
            against.append(f"Wilder RSI {_fmt(rsi)} soft")

    if "bullish divergence" in div:
        score += 2
        reasons.append(div)
    elif "bearish divergence" in div:
        score -= 2
        against.append(div)

    # Bollinger
    if pct_b is not None:
        if pct_b <= 0.15:
            score += 1
            reasons.append(f"%B {_fmt(pct_b, 3)} near lower band")
        elif pct_b >= 0.85:
            score -= 1
            against.append(f"%B {_fmt(pct_b, 3)} near upper band")
    if bw_regime == "squeeze":
        reasons.append("Bollinger squeeze — wait for expansion direction")
    elif bw_regime == "expansion" and pct_b is not None and pct_b > 0.7:
        score += 1
        reasons.append("Bandwidth expansion with upper-half price")

    # Supertrend
    if st_dir == "bullish":
        score += 2
        reasons.append(f"Supertrend bullish @ {_fmt(st.get('line'))}")
    elif st_dir == "bearish":
        score -= 2
        against.append(f"Supertrend bearish @ {_fmt(st.get('line'))}")

    # MACD
    if macd_hist is not None:
        if macd_hist > 0:
            score += 1
            reasons.append(f"MACD hist +{_fmt(macd_hist)}")
        elif macd_hist < 0:
            score -= 1
            against.append(f"MACD hist {_fmt(macd_hist)}")

    # Stoch / ADX
    if stoch_k is not None:
        if stoch_k <= 20:
            score += 1
            reasons.append(f"Stoch %K {_fmt(stoch_k)} oversold")
        elif stoch_k >= 80:
            score -= 1
            against.append(f"Stoch %K {_fmt(stoch_k)} overbought")
    if adx is not None and adx >= 25 and st_dir == "bullish":
        score += 1
        reasons.append(f"ADX~{_fmt(adx)} supports trend strength")
    elif adx is not None and adx >= 25 and st_dir == "bearish":
        score -= 1
        against.append(f"ADX~{_fmt(adx)} supports downtrend")

    # RS vs Nifty
    if rs20 is not None:
        if rs20 >= 1.03:
            score += 1
            reasons.append(f"Outperforming Nifty (RS20={_fmt(rs20, 3)})")
        elif rs20 <= 0.97:
            score -= 1
            against.append(f"Underperforming Nifty (RS20={_fmt(rs20, 3)})")

    # Volume + delivery conviction
    if vz is not None:
        if vz >= 1.0 and score >= 0:
            if delivery is not None and delivery < 25:
                against.append(
                    f"Volume spike z={_fmt(vz)} but low delivery {delivery:.0f}% — likely churn"
                )
                score -= 1
            else:
                score += 1
                note = f"Volume confirmation z={_fmt(vz)}"
                if delivery is not None:
                    note += f" | delivery {delivery:.0f}%"
                reasons.append(note)
        elif vz <= -1.0:
            against.append(f"Weak volume z={_fmt(vz)}")
    if delivery is not None and delivery >= 50 and score >= 0:
        score += 1
        reasons.append(f"High delivery {delivery:.0f}% supports accumulation narrative")

    # Valuation
    if ey is not None and ey >= 5:
        score += 1
        reasons.append(f"Earnings yield {_fmt(ey)}% supportive")
    if pe is not None and pe > 45:
        score -= 1
        against.append(f"Rich P/E {_fmt(pe)}")
    if peg is not None and 0 < peg < 1.2:
        score += 1
        reasons.append(f"PEG {_fmt(peg, 2)} reasonable vs growth")
    if ev is not None and 0 < ev < 12:
        score += 1
        reasons.append(f"EV/EBITDA {_fmt(ev)} not stretched")
    elif ev is not None and ev > 25:
        score -= 1
        against.append(f"EV/EBITDA {_fmt(ev)} elevated")
    if sector_prem is not None and sector_prem > 40:
        score -= 1
        against.append(f"Sector PE premium +{sector_prem:.0f}% stretched vs peers")
    elif sector_prem is not None and sector_prem < -20:
        score += 1
        reasons.append(f"Trading at {abs(sector_prem):.0f}% discount vs sector PE")

    # Risk stats
    if sortino is not None and sortino > 1:
        score += 1
        reasons.append(f"Sortino(60d) {_fmt(sortino)} solid")
    if max_dd is not None and max_dd <= -20:
        against.append(f"Deep 60d drawdown {_fmt(max_dd)}%")
    if beta60 is not None and beta60 > 1.4:
        against.append(f"High beta {_fmt(beta60, 2)} vs Nifty — wider swings")

    # Horizon tweak
    h = (horizon or "swing").lower()
    if h in {"intraday", "week", "weekly"} and rsi is not None and rsi >= 68:
        score -= 1
        against.append("Short-horizon ask + elevated RSI — prefer cool-off")
    if h in {"long", "long_term", "invest"} and ey is not None and ey >= 4:
        score += 1

    # Map score → action
    if score >= 4:
        action = "BUY"
    elif score >= 2:
        action = "ACCUMULATE"
    elif score <= -4:
        action = "SELL"
    elif score <= -2:
        action = "TRIM"
    else:
        action = "HOLD"

    # Hard veto: do not chase overbought / extended RSI on short horizons.
    if action in {"BUY", "ACCUMULATE"} and rsi is not None:
        if rsi >= 70 and h in {"intraday", "week", "weekly", "swing"}:
            action = "TRIM" if rsi >= 75 else "HOLD"
            against.append("Overbought veto: no fresh BUY while Wilder RSI ≥ 70 on this horizon")
            score = min(score, 1)
        elif rsi >= 68 and h in {"intraday", "week", "weekly"} and action == "BUY":
            action = "ACCUMULATE"
            against.append("Elevated RSI on short horizon — prefer staged entries, not chase")
            score = min(score, 3)

    # Entry / targets from structure
    fib_618 = _f(fib.get("retracement_618"))
    fib_382 = _f(fib.get("retracement_382"))
    piv = pack.get("pivots_classic") or {}
    s1 = _f(piv.get("S1"))
    r1 = _f(piv.get("R1"))

    if action in {"BUY", "ACCUMULATE"}:
        plan_stop = stop or (price * 0.97)
        candidates = [price * 0.985, price * 0.995]
        for lvl in (s1, fib_618, fib_382):
            if lvl is not None and lvl < price and lvl > plan_stop:
                candidates.append(lvl)
        entry_low = min(candidates)
        entry_high = price
        if plan_stop >= entry_low:
            plan_stop = entry_low * 0.985
        risk = max(price - plan_stop, price * 0.01)
        t1 = structural_t1 or r1 or (price + risk * 1.5)
        if t1 <= price:
            t1 = price + risk * 1.5
        t2 = structural_t2 or _f(fib.get("extension_1618"))
        if t2 is None or t2 <= t1:
            t2 = t1 + risk
        rr = _risk_reward(price, plan_stop, t1) or rr
    elif action in {"SELL", "TRIM"}:
        entry_low = price
        candidates = [price * 1.015]
        for lvl in (r1, fib_382, _f(bb.get("upper"))):
            if lvl is not None and lvl > price:
                candidates.append(lvl)
        entry_high = max(candidates)
        t1 = structural_t1 or r1 or target_1r or price * 1.02
        if t1 <= price:
            t1 = price * 1.02
        t2 = structural_t2 or _f(bb.get("upper")) or (t1 * 1.02)
        plan_stop = (price * 1.03) if action == "SELL" else (stop or price * 0.97)
        if action == "SELL":
            rr = _risk_reward(price, plan_stop, t1) or rr
    else:
        entry_low = s1 or price * 0.98
        entry_high = r1 or price * 1.02
        if entry_low > entry_high:
            entry_low, entry_high = entry_high, entry_low
        t1 = structural_t1 or target_1r or r1 or price * 1.02
        t2 = structural_t2 or _f(bb.get("upper")) or (t1 * 1.02 if t1 else None)
        plan_stop = stop or price * 0.97
        if plan_stop and t1:
            rr = _risk_reward(price, plan_stop, t1) or rr

    # Confidence from |score|, data completeness, and feed quality
    completeness = sum(
        1
        for x in (rsi, pct_b, stop, rs20, ey, st_dir, delivery)
        if x not in (None, "", "n/a")
    )
    conf = min(0.92, 0.35 + 0.08 * abs(score) + 0.05 * completeness)
    if dq.get("degraded"):
        conf = min(conf, 0.72)
        against.append("Partial feeds (delivery/beta) — treat confidence as capped")

    # Kelly fractional (educational) using heuristic p from score
    p_win = min(0.65, max(0.35, 0.5 + score * 0.03))
    b = rr if rr and rr > 0 else 1.0
    kelly = p_win - (1 - p_win) / b if b else 0.0
    kelly_frac = max(0.0, min(0.25, kelly / 4.0))  # quarter-Kelly cap 25%

    # Qty from plan stop (not stale ATR 1R pair)
    risk_rupees = _f(atr_s.get("risk_rupees")) or 5000.0
    risk_ps = abs(price - plan_stop) if plan_stop else None
    qty = int(risk_rupees // risk_ps) if risk_ps and risk_ps > 0 else atr_s.get("qty_for_risk")
    charges = pack.get("india_costs") or {}
    fo = pack.get("fo") or {}

    return {
        "action": action,
        "score": score,
        "confidence": round(conf, 2),
        "horizon": h,
        "price": round(price, 2),
        "entry_zone": [round(entry_low, 2), round(entry_high, 2)] if entry_low and entry_high else None,
        "stop": round(plan_stop, 2) if plan_stop else None,
        "target_1": round(t1, 2) if t1 else None,
        "target_2": round(t2, 2) if t2 else None,
        "risk_reward": round(rr, 2) if rr else None,
        "position_qty_for_risk": qty,
        "kelly_fraction_capped": round(kelly_frac, 3),
        "win_rate_prior": round(p_win, 2),
        "reasons_for": reasons[:6],
        "reasons_against": against[:6],
        "invalidation": (
            f"Close below stop {_fmt(plan_stop)}" if action in {"BUY", "ACCUMULATE", "HOLD"}
            else f"Close above {_fmt(plan_stop)} invalidates short/trim thesis"
        ),
        "india_cost_note": charges.get("roundtrip_cost_pct_note"),
        "delivery_pct": delivery,
        "fo_lot_size": fo.get("lot_size"),
        "fo_notional_per_lot": fo.get("notional_per_lot"),
        "fo_margin_per_lot": fo.get("indicative_margin_per_lot"),
        "data_degraded": bool(dq.get("degraded")),
        "educational": True,
        "disclaimer": (
            "Paper-practice plan from deterministic math — not investment advice, "
            "not a SEBI-registered recommendation, not a price guarantee."
        ),
    }


def _fmt(value: Any, digits: int = 2) -> str:
    n = _f(value)
    if n is None:
        return "n/a"
    if abs(n) >= 100:
        return f"{n:.1f}"
    return f"{n:.{digits}f}"


def format_trade_plan(plan: dict[str, Any]) -> str:
    if not plan:
        return ""
    ez = plan.get("entry_zone") or []
    entry = f"{ez[0]} – {ez[1]}" if len(ez) == 2 else "n/a"
    lines = [
        f"TRADE PLAN: {plan.get('action')} | score={plan.get('score')} | conf={plan.get('confidence')} | horizon={plan.get('horizon')}",
        f"Entry zone: {entry} | Stop: {plan.get('stop')} | T1: {plan.get('target_1')} | T2: {plan.get('target_2')}",
        f"R:R={plan.get('risk_reward')} | qty@{plan.get('position_qty_for_risk')} | kelly_frac={plan.get('kelly_fraction_capped')}",
        "FOR: " + "; ".join(plan.get("reasons_for") or ["n/a"]),
        "AGAINST: " + "; ".join(plan.get("reasons_against") or ["n/a"]),
        f"Invalidation: {plan.get('invalidation')}",
    ]
    return " | ".join(lines)
