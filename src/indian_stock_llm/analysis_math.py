"""Quantitative math for BYSEL Indian Stock LLM (P0 + Phase B/C).

Deterministic formulas for analysis and paper-practice buy/sell plans:
  Wilder RSI, Bollinger %B/bandwidth, ATR stops/sizing, R:R, pivots,
  volume z-score, RS/beta vs Nifty, earnings yield, EV/EBITDA,
  Supertrend, Fibonacci, MACD, Sortino/maxDD/Calmar, PEG, VWAP,
  Kelly fraction, India STT/charge estimates, circuit distance, SIP FV,
  personal-finance TVM/retirement/TER packs,
  multi-factor sentiment analysis (news + technicals + volume + RS).
"""
from __future__ import annotations

import math
import re
from typing import Any, Optional


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "" or value == "N/A":
            return None
        if isinstance(value, str):
            text = value.replace(",", "").replace("₹", "").replace("%", "").strip()
            match = None
            import re

            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if not match:
                return None
            return float(match.group(0))
        return float(value)
    except (TypeError, ValueError):
        return None


def wilder_rsi(closes: list[float], period: int = 14) -> Optional[float]:
    """Wilder-smoothed RSI (vendor-comparable)."""
    if len(closes) <= period + 1:
        return None
    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        ch = closes[i] - closes[i - 1]
        gains.append(max(ch, 0.0))
        losses.append(max(-ch, 0.0))
    if len(gains) < period:
        return None
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def rsi_divergence_cue(closes: list[float], period: int = 14, lookback: int = 20) -> str:
    """Lightweight divergence cue over recent swing."""
    if len(closes) < lookback + period + 2:
        return "n/a"
    rsi_series: list[float] = []
    for end in range(period + 1, len(closes) + 1):
        val = wilder_rsi(closes[:end], period=period)
        if val is not None:
            rsi_series.append(val)
    if len(rsi_series) < lookback or len(closes) < lookback:
        return "n/a"
    price_window = closes[-lookback:]
    rsi_window = rsi_series[-lookback:]
    p_first, p_last = price_window[0], price_window[-1]
    r_first, r_last = rsi_window[0], rsi_window[-1]
    if p_last > p_first * 1.01 and r_last < r_first - 3:
        return "bearish divergence cue (price up, RSI down)"
    if p_last < p_first * 0.99 and r_last > r_first + 3:
        return "bullish divergence cue (price down, RSI up)"
    return "no clear divergence"


def bollinger_pct_b_bandwidth(
    closes: list[float], period: int = 20, std_mult: float = 2.0
) -> dict[str, Optional[float]]:
    if len(closes) < period:
        return {"pct_b": None, "bandwidth": None, "upper": None, "middle": None, "lower": None}
    window = closes[-period:]
    middle = sum(window) / period
    variance = sum((x - middle) ** 2 for x in window) / period
    std = math.sqrt(variance)
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    width = upper - lower
    price = closes[-1]
    pct_b = (price - lower) / width if width > 0 else None
    bandwidth = width / middle if middle else None
    return {
        "pct_b": pct_b,
        "bandwidth": bandwidth,
        "upper": upper,
        "middle": middle,
        "lower": lower,
    }


def atr(highs: list[float], lows: list[float], closes: list[float], period: int = 14) -> Optional[float]:
    if len(closes) <= period:
        return None
    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    if len(trs) < period:
        return None
    # Wilder ATR
    atr_val = sum(trs[:period]) / period
    for tr in trs[period:]:
        atr_val = (atr_val * (period - 1) + tr) / period
    return atr_val


def atr_stop_and_size(
    entry: float,
    atr_val: float,
    *,
    side: str = "long",
    atr_mult: float = 1.5,
    risk_rupees: float = 5000.0,
) -> dict[str, Optional[float]]:
    if entry <= 0 or atr_val is None or atr_val <= 0:
        return {"stop": None, "target_1r": None, "qty": None, "risk_per_share": None}
    risk_per_share = atr_mult * atr_val
    if side == "short":
        stop = entry + risk_per_share
        target = entry - risk_per_share
    else:
        stop = entry - risk_per_share
        target = entry + risk_per_share
    qty = math.floor(risk_rupees / risk_per_share) if risk_per_share > 0 else 0
    return {
        "stop": stop,
        "target_1r": target,
        "qty": float(max(qty, 0)),
        "risk_per_share": risk_per_share,
        "atr_mult": atr_mult,
        "risk_rupees": risk_rupees,
    }


def risk_reward(entry: float, stop: float, target: float) -> Optional[float]:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return reward / risk


def expectancy(win_rate: float, reward_risk: float) -> Optional[float]:
    """Expected R per trade: p*R - (1-p)*1."""
    if reward_risk is None or reward_risk < 0:
        return None
    p = min(max(win_rate, 0.0), 1.0)
    return p * reward_risk - (1.0 - p)


def classic_pivots(high: float, low: float, close: float) -> dict[str, float]:
    p = (high + low + close) / 3.0
    r1 = 2 * p - low
    s1 = 2 * p - high
    r2 = p + (high - low)
    s2 = p - (high - low)
    r3 = high + 2 * (p - low)
    s3 = low - 2 * (high - p)
    return {"P": p, "R1": r1, "R2": r2, "R3": r3, "S1": s1, "S2": s2, "S3": s3}


def camarilla_pivots(high: float, low: float, close: float) -> dict[str, float]:
    rng = high - low
    return {
        "R3": close + rng * 1.1 / 4,
        "R2": close + rng * 1.1 / 6,
        "R1": close + rng * 1.1 / 12,
        "S1": close - rng * 1.1 / 12,
        "S2": close - rng * 1.1 / 6,
        "S3": close - rng * 1.1 / 4,
    }


def central_pivot_range(high: float, low: float, close: float) -> dict[str, float]:
    """Central Pivot Range (CPR): Pivot, TC, BC from prior session H/L/C."""
    pivot = (high + low + close) / 3.0
    bc = (high + low) / 2.0
    tc = (pivot - bc) + pivot  # 2P − BC
    width = abs(tc - bc)
    # Narrow CPR often watched as potential breakout day (educational heuristic).
    mid = (tc + bc) / 2.0 if (tc or bc) else pivot
    width_pct = (width / mid * 100.0) if mid else 0.0
    return {
        "P": pivot,
        "TC": tc,
        "BC": bc,
        "width": width,
        "width_pct": width_pct,
        "regime": "narrow" if width_pct < 0.35 else "wide" if width_pct > 0.75 else "normal",
    }


def volume_zscore(volumes: list[float], window: int = 20) -> dict[str, Optional[float]]:
    if len(volumes) < window + 1:
        return {"z": None, "ratio_vs_avg": None, "avg": None}
    hist = volumes[-(window + 1) : -1]
    last = volumes[-1]
    avg = sum(hist) / window
    var = sum((v - avg) ** 2 for v in hist) / window
    std = math.sqrt(var)
    z = (last - avg) / std if std > 1e-9 else 0.0
    ratio = last / avg if avg > 0 else None
    return {"z": z, "ratio_vs_avg": ratio, "avg": avg, "last": last}


def _align_close_series(
    stock_dates: list[str],
    stock_closes: list[float],
    index_dates: list[str],
    index_closes: list[float],
) -> tuple[list[float], list[float]]:
    """Inner-join closes on calendar dates; positional fallback if dates blank/thin."""

    def _positional() -> tuple[list[float], list[float]]:
        n = min(len(stock_closes), len(index_closes))
        if n < 5:
            return [], []
        s = [float(c) for c in stock_closes[-n:] if c and c > 0]
        m = [float(c) for c in index_closes[-n:] if c and c > 0]
        k = min(len(s), len(m))
        return s[-k:], m[-k:]

    if not stock_dates or not index_dates:
        return _positional()

    idx_map = {
        str(d).strip()[:10]: float(c)
        for d, c in zip(index_dates, index_closes)
        if str(d or "").strip() and c and float(c) > 0
    }
    aligned_s: list[float] = []
    aligned_m: list[float] = []
    for d, c in zip(stock_dates, stock_closes):
        key = str(d or "").strip()[:10]
        if not key or not c or float(c) <= 0:
            continue
        m = idx_map.get(key)
        if m is None:
            continue
        aligned_s.append(float(c))
        aligned_m.append(float(m))
    if len(aligned_s) >= 20:
        return aligned_s, aligned_m
    return _positional()


def relative_strength(
    stock_closes: list[float],
    index_closes: list[float],
    days: int = 20,
    *,
    stock_dates: Optional[list[str]] = None,
    index_dates: Optional[list[str]] = None,
) -> Optional[float]:
    """RS = stock growth / index growth over N overlapping sessions."""
    s, m = _align_close_series(
        stock_dates or [],
        stock_closes,
        index_dates or [],
        index_closes,
    )
    need = days + 1
    if len(s) < need or len(m) < need:
        return None
    s0, s1 = s[-need], s[-1]
    i0, i1 = m[-need], m[-1]
    if s0 <= 0 or i0 <= 0 or i1 <= 0:
        return None
    stock_ret = s1 / s0
    index_ret = i1 / i0
    if index_ret == 0:
        return None
    return stock_ret / index_ret


def beta_vs_index(
    stock_closes: list[float],
    index_closes: list[float],
    lookback: int = 60,
    *,
    stock_dates: Optional[list[str]] = None,
    index_dates: Optional[list[str]] = None,
    min_obs: Optional[int] = None,
) -> Optional[float]:
    """β = Cov(rs, rm) / Var(rm) on date-aligned daily returns."""
    s, m = _align_close_series(
        stock_dates or [],
        stock_closes,
        index_dates or [],
        index_closes,
    )
    need = lookback + 1
    if len(s) < max(25, need // 2):
        return None
    s = s[-need:] if len(s) >= need else s
    m = m[-need:] if len(m) >= need else m
    rs = [(s[i] / s[i - 1] - 1.0) for i in range(1, len(s)) if s[i - 1] > 0 and s[i] > 0]
    rm = [(m[i] / m[i - 1] - 1.0) for i in range(1, len(m)) if m[i - 1] > 0 and m[i] > 0]
    k = min(len(rs), len(rm))
    required = min_obs if min_obs is not None else max(20, min(lookback - 5, lookback // 2))
    if k < required:
        return None
    rs = rs[-k:]
    rm = rm[-k:]
    mean_s = sum(rs) / k
    mean_m = sum(rm) / k
    cov = sum((rs[i] - mean_s) * (rm[i] - mean_m) for i in range(k)) / k
    var_m = sum((rm[i] - mean_m) ** 2 for i in range(k)) / k
    if var_m <= 1e-12:
        return None
    beta = cov / var_m
    # Avoid publishing a rounded 0.000 that looks like "no beta".
    if abs(beta) < 0.01:
        return None
    # Cap absurd values from tiny samples / holidays mis-sync.
    if abs(beta) > 5:
        return None
    return beta


def earnings_yield(price: float, eps: Optional[float] = None, pe: Optional[float] = None) -> Optional[float]:
    """EY = EPS/Price, or 1/PE when EPS missing. Returned in percent."""
    if price and price > 0 and eps is not None and eps != 0:
        return (eps / price) * 100.0
    if pe is not None and pe > 0:
        return (1.0 / pe) * 100.0
    return None


def ev_ebitda(
    market_cap: Optional[float],
    total_debt: Optional[float],
    cash: Optional[float],
    ebitda: Optional[float],
) -> dict[str, Optional[float]]:
    if ebitda is None or ebitda == 0:
        return {"ev": None, "ev_ebitda": None, "net_debt_ebitda": None}
    debt = total_debt or 0.0
    cash_v = cash or 0.0
    mcap = market_cap or 0.0
    ev = mcap + debt - cash_v
    net_debt = debt - cash_v
    return {
        "ev": ev,
        "ev_ebitda": ev / ebitda if ebitda else None,
        "net_debt_ebitda": net_debt / ebitda if ebitda else None,
    }


def _clean_ohlcv_lists(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    volumes: list[float],
    dates: Optional[list[str]] = None,
) -> Optional[dict[str, Any]]:
    """Drop bars with non-positive OHLC so beta/RS aren't poisoned by zeros."""
    n = min(len(opens), len(highs), len(lows), len(closes), len(volumes))
    if dates is not None:
        n = min(n, len(dates))
    o2: list[float] = []
    h2: list[float] = []
    l2: list[float] = []
    c2: list[float] = []
    v2: list[float] = []
    d2: list[str] = []
    for i in range(n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        if min(o, h, l, c) <= 0:
            continue
        o2.append(float(o))
        h2.append(float(h))
        l2.append(float(l))
        c2.append(float(c))
        v2.append(float(volumes[i] or 0.0))
        if dates is not None:
            d2.append(str(dates[i])[:10])
    if len(c2) < 30:
        return None
    out: dict[str, Any] = {
        "open": o2,
        "high": h2,
        "low": l2,
        "close": c2,
        "volume": v2,
        "dates": d2,
        "source": "unknown",
    }
    return out


def _resolve_listed_symbol(symbol: str) -> str:
    try:
        from app.market_data import normalize_listed_symbol

        return normalize_listed_symbol(symbol)
    except Exception:
        aliases = {"TATAMOTORS": "TMPV"}
        sym = (symbol or "").strip().upper()
        return aliases.get(sym, sym)


def _prefer_nse_twin(symbol: str) -> str:
    """For dual-listed BSE codes, prefer the NSE scrip_id (more reliable Yahoo OHLCV)."""
    sym = (symbol or "").strip().upper()
    if not sym:
        return sym
    try:
        from app.stock_enricher import get_nse_equity_map, lookup_bse_listing

        rec = lookup_bse_listing(sym)
        if not rec:
            return sym
        sid = str(rec.get("scrip_id") or "").upper()
        if sid and sid in get_nse_equity_map():
            return sid
    except Exception:
        pass
    return sym


def _load_ohlcv(symbol: str, period: str = "6mo") -> Optional[dict[str, Any]]:
    symbol = _prefer_nse_twin(_resolve_listed_symbol(symbol))
    try:
        from app.market_data import fetch_quote_history

        candles = fetch_quote_history(symbol, period=period, interval="1d")
        if candles and len(candles) >= 30:
            from datetime import datetime, timezone

            def _candle_date(c: dict) -> str:
                for key in ("date", "datetime", "t"):
                    if c.get(key):
                        return str(c.get(key))[:10]
                ts = c.get("timestamp")
                if ts is None:
                    return ""
                try:
                    # ms or seconds
                    ts_f = float(ts)
                    if ts_f > 1e12:
                        ts_f /= 1000.0
                    return datetime.fromtimestamp(ts_f, tz=timezone.utc).strftime("%Y-%m-%d")
                except Exception:
                    return ""

            frame = _clean_ohlcv_lists(
                [float(c.get("open") or 0) for c in candles],
                [float(c.get("high") or 0) for c in candles],
                [float(c.get("low") or 0) for c in candles],
                [float(c.get("close") or 0) for c in candles],
                [float(c.get("volume") or 0) for c in candles],
                [_candle_date(c) for c in candles],
            )
            if frame:
                frame["source"] = "market_data"
                return frame
    except Exception:
        pass
    try:
        from .calculations import PandasTaIndicatorCalculator

        import yfinance as yf  # type: ignore

        try:
            from app.market_data import _yf_ticker

            primary = symbol if symbol.startswith("^") else _yf_ticker(symbol)
        except Exception:
            primary = symbol if symbol.startswith("^") else f"{symbol}.NS"
        hist = yf.Ticker(primary).history(period=period, interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            # Cross-exchange fallback for dual-listed names.
            alt = None
            if str(primary).endswith(".NS"):
                alt = f"{symbol}.BO" if not (len(symbol) == 6 and symbol.isdigit()) else f"{symbol}.BO"
                try:
                    from app.market_data import _yf_ticker as _yt

                    # Prefer resolved BSE numeric code when available.
                    forced = _yt(f"BSE:{symbol}")
                    if forced and forced.endswith(".BO"):
                        alt = forced
                except Exception:
                    pass
            elif str(primary).endswith(".BO"):
                base = symbol[:-3] if symbol.endswith(".BO") else symbol
                if not (len(base) == 6 and base.isdigit()):
                    alt = f"{base}.NS"
            if alt and alt != primary:
                hist = yf.Ticker(alt).history(period=period, interval="1d", auto_adjust=True)
        if hist is None or hist.empty:
            raw = PandasTaIndicatorCalculator._load_ohlcv_frame(symbol)
            if not raw:
                return None
            frame = _clean_ohlcv_lists(
                list(raw.get("open") or []),
                list(raw.get("high") or []),
                list(raw.get("low") or []),
                list(raw.get("close") or []),
                list(raw.get("volume") or []),
                list(raw.get("dates") or []) if raw.get("dates") else None,
            )
            if frame:
                frame["source"] = "calculator"
            return frame
        dates = [str(x)[:10] for x in hist.index.astype(str).tolist()]
        frame = _clean_ohlcv_lists(
            [float(x) for x in hist["Open"].tolist()],
            [float(x) for x in hist["High"].tolist()],
            [float(x) for x in hist["Low"].tolist()],
            [float(x) for x in hist["Close"].tolist()],
            [float(x) for x in hist["Volume"].tolist()],
            dates,
        )
        if frame:
            frame["source"] = "yfinance"
        return frame
    except Exception:
        try:
            from .calculations import PandasTaIndicatorCalculator

            raw = PandasTaIndicatorCalculator._load_ohlcv_frame(symbol)
            if not raw:
                return None
            frame = _clean_ohlcv_lists(
                list(raw.get("open") or []),
                list(raw.get("high") or []),
                list(raw.get("low") or []),
                list(raw.get("close") or []),
                list(raw.get("volume") or []),
                list(raw.get("dates") or []) if raw.get("dates") else None,
            )
            if frame:
                frame["source"] = "calculator"
            return frame
        except Exception:
            return None


def _load_nifty_series(period: str = "6mo") -> dict[str, list]:
    """Return {close, dates} for Nifty; empty lists on failure."""
    for sym in ("NIFTY50", "^NSEI"):
        frame = _load_ohlcv(sym, period=period)
        if frame and len(frame.get("close") or []) >= 40:
            return {"close": list(frame["close"]), "dates": list(frame.get("dates") or [])}
    try:
        import yfinance as yf  # type: ignore

        hist = yf.Ticker("^NSEI").history(period=period, interval="1d", auto_adjust=True)
        if hist is not None and not hist.empty:
            return {
                "close": [float(x) for x in hist["Close"].tolist()],
                "dates": [str(x)[:10] for x in hist.index.astype(str).tolist()],
            }
    except Exception:
        pass
    return {"close": [], "dates": []}


def _load_nifty_closes(period: str = "6mo") -> list[float]:
    return list(_load_nifty_series(period).get("close") or [])


def _fundamentals_from_yfinance(symbol: str) -> dict[str, Optional[float]]:
    out: dict[str, Optional[float]] = {
        "market_cap": None,
        "total_debt": None,
        "cash": None,
        "ebitda": None,
        "eps": None,
        "pe": None,
        "beta": None,
        "roe": None,
        "revenue_growth": None,
        "earnings_growth": None,
        "book_value": None,
        "fifty_two_week_high": None,
        "fifty_two_week_low": None,
        "dividend_yield": None,
    }
    try:
        import yfinance as yf  # type: ignore

        try:
            from app.market_data import _yf_ticker

            yf_sym = _yf_ticker(symbol)
        except Exception:
            yf_sym = f"{symbol}.NS"
        info = yf.Ticker(yf_sym).info or {}
        if not info and str(yf_sym).endswith(".BO"):
            info = yf.Ticker(f"{symbol}.NS").info or {}
        elif not info:
            info = yf.Ticker(f"{symbol}.BO").info or {}
        out["market_cap"] = _safe_float(info.get("marketCap"))
        out["total_debt"] = _safe_float(info.get("totalDebt"))
        out["cash"] = _safe_float(info.get("totalCash") or info.get("cash"))
        out["ebitda"] = _safe_float(info.get("ebitda"))
        out["eps"] = _safe_float(info.get("trailingEps") or info.get("epsTrailingTwelveMonths"))
        out["pe"] = _safe_float(info.get("trailingPE") or info.get("forwardPE"))
        out["beta"] = _safe_float(info.get("beta"))
        out["roe"] = _safe_float(info.get("returnOnEquity"))
        if out["roe"] is not None and abs(out["roe"]) <= 1.5:
            out["roe"] = out["roe"] * 100.0
        out["revenue_growth"] = _safe_float(info.get("revenueGrowth"))
        out["earnings_growth"] = _safe_float(info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth"))
        # growth often as fraction
        for gk in ("revenue_growth", "earnings_growth"):
            g = out.get(gk)
            if g is not None and abs(g) <= 2:
                out[gk] = g * 100.0
        out["book_value"] = _safe_float(info.get("bookValue"))
        out["fifty_two_week_high"] = _safe_float(info.get("fiftyTwoWeekHigh"))
        out["fifty_two_week_low"] = _safe_float(info.get("fiftyTwoWeekLow"))
        dy = _safe_float(info.get("dividendYield") or info.get("trailingAnnualDividendYield"))
        if dy is not None:
            # yfinance may return fraction (0.012) or percent-like (>1)
            out["dividend_yield"] = dy * 100.0 if dy <= 1.5 else dy
    except Exception:
        pass
    return out


# ── Phase B/C formulas ──────────────────────────────────────────────


def macd(closes: list[float]) -> dict[str, Optional[float]]:
    if len(closes) < 35:
        return {"macd": None, "signal": None, "histogram": None}
    from .calculations import _macd_last

    try:
        m, s, h = _macd_last(closes)
        return {"macd": m, "signal": s, "histogram": h}
    except Exception:
        return {"macd": None, "signal": None, "histogram": None}


def supertrend(
    highs: list[float],
    lows: list[float],
    closes: list[float],
    period: int = 10,
    multiplier: float = 3.0,
) -> dict[str, Any]:
    """ATR-based Supertrend direction + line."""
    n = len(closes)
    if n < period + 2:
        return {"direction": "n/a", "line": None}
    # True ranges + Wilder ATR series
    trs: list[float] = [highs[0] - lows[0]]
    for i in range(1, n):
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    atr_s = sum(trs[:period]) / period
    atr_series = [None] * (period - 1) + [atr_s]
    for i in range(period, n):
        atr_s = (atr_s * (period - 1) + trs[i]) / period
        atr_series.append(atr_s)

    final_upper: list[Optional[float]] = [None] * n
    final_lower: list[Optional[float]] = [None] * n
    direction = 1  # 1 bullish, -1 bearish
    line = None
    for i in range(period - 1, n):
        hl2 = (highs[i] + lows[i]) / 2.0
        a = atr_series[i] or 0.0
        basic_upper = hl2 + multiplier * a
        basic_lower = hl2 - multiplier * a
        if i == period - 1:
            final_upper[i] = basic_upper
            final_lower[i] = basic_lower
            direction = 1 if closes[i] >= basic_lower else -1
        else:
            prev_fu = final_upper[i - 1] or basic_upper
            prev_fl = final_lower[i - 1] or basic_lower
            final_upper[i] = basic_upper if (basic_upper < prev_fu or closes[i - 1] > prev_fu) else prev_fu
            final_lower[i] = basic_lower if (basic_lower > prev_fl or closes[i - 1] < prev_fl) else prev_fl
            if direction == 1:
                if closes[i] < (final_lower[i] or closes[i]):
                    direction = -1
            else:
                if closes[i] > (final_upper[i] or closes[i]):
                    direction = 1
        line = final_lower[i] if direction == 1 else final_upper[i]
    return {
        "direction": "bullish" if direction == 1 else "bearish",
        "line": round(line, 2) if line is not None else None,
        "multiplier": multiplier,
        "period": period,
    }


def fibonacci_levels(highs: list[float], lows: list[float], closes: list[float], lookback: int = 60) -> dict[str, Optional[float]]:
    if len(closes) < lookback:
        lookback = len(closes)
    if lookback < 10:
        return {}
    window_h = highs[-lookback:]
    window_l = lows[-lookback:]
    swing_high = max(window_h)
    swing_low = min(window_l)
    rng = swing_high - swing_low
    if rng <= 0:
        return {}
    # Assume uptrend retracement from low→high if last close nearer high
    uptrend = closes[-1] >= (swing_high + swing_low) / 2
    if uptrend:
        return {
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "retracement_382": round(swing_high - 0.382 * rng, 2),
            "retracement_500": round(swing_high - 0.5 * rng, 2),
            "retracement_618": round(swing_high - 0.618 * rng, 2),
            "extension_1618": round(swing_high + 0.618 * rng, 2),
            "bias": "uptrend_retracement",
        }
    return {
        "swing_high": round(swing_high, 2),
        "swing_low": round(swing_low, 2),
        "retracement_382": round(swing_low + 0.382 * rng, 2),
        "retracement_500": round(swing_low + 0.5 * rng, 2),
        "retracement_618": round(swing_low + 0.618 * rng, 2),
        "extension_1618": round(swing_low - 0.618 * rng, 2),
        "bias": "downtrend_retracement",
    }


def session_vwap_approx(
    highs: list[float], lows: list[float], closes: list[float], volumes: list[float], window: int = 20
) -> Optional[float]:
    """Anchored-ish VWAP over last N daily bars (proxy when intraday bars absent)."""
    n = min(len(closes), len(volumes), window)
    if n < 5:
        return None
    num = 0.0
    den = 0.0
    for i in range(-n, 0):
        typical = (highs[i] + lows[i] + closes[i]) / 3.0
        v = max(volumes[i], 0.0)
        num += typical * v
        den += v
    if den <= 0:
        return None
    return num / den


def risk_stats(closes: list[float], lookback: int = 60, rf_daily: float = 0.00025) -> dict[str, Optional[float]]:
    """Sortino, max drawdown, Calmar on daily closes."""
    if len(closes) < lookback + 1:
        lookback = max(20, len(closes) - 1)
    if lookback < 20:
        return {"sortino_60d": None, "max_drawdown_60d_pct": None, "calmar_60d": None}
    window = closes[-(lookback + 1) :]
    rets = [(window[i] / window[i - 1] - 1.0) for i in range(1, len(window)) if window[i - 1] > 0]
    if len(rets) < 20:
        return {"sortino_60d": None, "max_drawdown_60d_pct": None, "calmar_60d": None}
    avg = sum(rets) / len(rets)
    downside = [min(0.0, r - rf_daily) for r in rets]
    down_var = sum(d * d for d in downside) / len(downside)
    down_std = math.sqrt(down_var)
    sortino = ((avg - rf_daily) / down_std) * math.sqrt(252) if down_std > 1e-12 else None

    peak = window[0]
    max_dd = 0.0
    for px in window:
        peak = max(peak, px)
        dd = (px - peak) / peak if peak > 0 else 0.0
        max_dd = min(max_dd, dd)
    # Annualized return approx
    total_ret = window[-1] / window[0] - 1.0 if window[0] > 0 else 0.0
    ann = total_ret * (252 / lookback)
    calmar = (ann / abs(max_dd)) if max_dd < -1e-6 else None
    return {
        "sortino_60d": round(sortino, 3) if sortino is not None else None,
        "max_drawdown_60d_pct": round(max_dd * 100.0, 2),
        "calmar_60d": round(calmar, 3) if calmar is not None else None,
    }


def peg_ratio(pe: Optional[float], growth_pct: Optional[float]) -> Optional[float]:
    if pe is None or growth_pct is None or growth_pct == 0:
        return None
    return pe / growth_pct


def justified_pe_rough(growth_pct: Optional[float], k: float = 1.0) -> Optional[float]:
    """Very rough PE≈k·g heuristic for education (not a DCF)."""
    if growth_pct is None or growth_pct <= 0:
        return None
    return k * growth_pct


def kelly_fraction(win_rate: float, reward_risk: float) -> float:
    if reward_risk <= 0:
        return 0.0
    p = min(max(win_rate, 0.0), 1.0)
    f = p - (1.0 - p) / reward_risk
    return max(0.0, min(1.0, f))


def portfolio_expected_return(
    weights: list[float], expected_returns_pct: list[float]
) -> Optional[float]:
    """E(Rp) = Σ wi·Ri (weights as fractions, returns in %)."""
    if not weights or len(weights) != len(expected_returns_pct):
        return None
    return round(sum(w * r for w, r in zip(weights, expected_returns_pct)), 4)


def two_asset_portfolio_vol(
    w1: float,
    w2: float,
    sigma1_pct: float,
    sigma2_pct: float,
    corr: float,
) -> Optional[float]:
    """Portfolio σ% from two-asset variance with correlation."""
    if abs(w1 + w2 - 1.0) > 0.15:
        # still compute; caller may use raw weights
        pass
    s1, s2 = sigma1_pct / 100.0, sigma2_pct / 100.0
    var = (w1 * w1 * s1 * s1) + (w2 * w2 * s2 * s2) + (2 * w1 * w2 * s1 * s2 * corr)
    if var < 0:
        return None
    return round(math.sqrt(var) * 100.0, 4)


def recovery_return_needed(loss_pct: float) -> Optional[float]:
    """Return % needed to recover after a loss_pct drawdown on capital."""
    if loss_pct <= 0 or loss_pct >= 100:
        return None
    f = loss_pct / 100.0
    return round((f / (1.0 - f)) * 100.0, 2)


def value_at_risk_parametric(
    portfolio_value: float,
    daily_vol_pct: float,
    *,
    z: float = 1.65,
    days: float = 1.0,
) -> Optional[float]:
    """Educational parametric VaR ≈ Value × z × σ_daily × √t (e.g. z≈1.65 ~95%)."""
    if portfolio_value <= 0 or daily_vol_pct is None or daily_vol_pct < 0 or days <= 0:
        return None
    return round(portfolio_value * z * (daily_vol_pct / 100.0) * math.sqrt(days), 2)


def percent_risk_position_qty(
    equity: float,
    risk_pct: float,
    entry: float,
    stop: float,
) -> dict[str, Any]:
    """% risk model: risk ₹ = equity×risk%; qty = risk ₹ / |entry−stop|."""
    out: dict[str, Any] = {"ok": False}
    if equity <= 0 or risk_pct <= 0 or entry <= 0:
        return out
    risk_rupees = equity * (risk_pct / 100.0)
    per_unit = abs(entry - stop)
    if per_unit <= 0:
        return out
    qty = int(math.floor(risk_rupees / per_unit))
    out.update(
        {
            "ok": True,
            "model": "percent_risk",
            "equity": round(equity, 2),
            "risk_pct": risk_pct,
            "risk_rupees": round(risk_rupees, 2),
            "risk_per_unit": round(per_unit, 4),
            "qty": qty,
            "notional": round(qty * entry, 2),
        }
    )
    return out


def percent_volatility_position_qty(
    equity: float,
    risk_pct: float,
    atr: float,
    atr_mult: float = 1.0,
    price: float = 0.0,
) -> dict[str, Any]:
    """% volatility: treat ATR×mult as 1R stop distance."""
    stop_dist = atr * atr_mult
    if price <= 0 or stop_dist <= 0:
        return {"ok": False}
    return {
        **percent_risk_position_qty(equity, risk_pct, price, price - stop_dist),
        "model": "percent_volatility",
        "atr": atr,
        "atr_mult": atr_mult,
    }


def _pearson_corr(xs: list[float], ys: list[float]) -> Optional[float]:
    n = min(len(xs), len(ys))
    if n < 10:
        return None
    x = xs[-n:]
    y = ys[-n:]
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denx = math.sqrt(sum((a - mx) ** 2 for a in x))
    deny = math.sqrt(sum((b - my) ** 2 for b in y))
    if denx <= 1e-12 or deny <= 1e-12:
        return None
    return num / (denx * deny)


def _series_zscore(values: list[float]) -> dict[str, Optional[float]]:
    if len(values) < 10:
        return {"z": None, "mean": None, "stdev": None, "last": None}
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    std = math.sqrt(var)
    last = values[-1]
    z = (last - mean) / std if std > 1e-12 else None
    return {
        "z": round(z, 3) if z is not None else None,
        "mean": round(mean, 6),
        "stdev": round(std, 6),
        "last": round(last, 6),
    }


def linear_regression_xy(
    xs: list[float], ys: list[float]
) -> dict[str, Optional[float]]:
    """Simple OLS: y ≈ a + b·x (educational pair hedge ratio)."""
    n = min(len(xs), len(ys))
    out: dict[str, Optional[float]] = {
        "ok": False,
        "slope": None,
        "intercept": None,
        "r_squared": None,
        "n": n,
    }
    if n < 20:
        return out
    x = [float(v) for v in xs[-n:]]
    y = [float(v) for v in ys[-n:]]
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    syy = sum((b - my) ** 2 for b in y)
    if sxx <= 1e-12:
        return out
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_res = sum((b - (intercept + slope * a)) ** 2 for a, b in zip(x, y))
    r2 = 1.0 - (ss_res / syy) if syy > 1e-12 else None
    residuals = [b - (intercept + slope * a) for a, b in zip(x, y)]
    out.update(
        {
            "ok": True,
            "slope": round(slope, 6),
            "intercept": round(intercept, 6),
            "r_squared": round(r2, 4) if r2 is not None else None,
            "residuals": residuals,
            "error_ratio": round(ss_res / syy, 4) if syy > 1e-12 else None,
        }
    )
    return out


def adf_stationarity_proxy(series: list[float]) -> dict[str, Any]:
    """Lightweight DF-style cue (not a full ADF p-value). Educational only."""
    out: dict[str, Any] = {"ok": False, "note": "Proxy — confirm with full ADF in stats software."}
    if len(series) < 30:
        return out
    y = [float(v) for v in series]
    # Δy_t = γ·y_{t-1} + ε  (no lag/diff controls)
    dys = [y[i] - y[i - 1] for i in range(1, len(y))]
    lags = y[:-1]
    n = len(dys)
    ml = sum(lags) / n
    md = sum(dys) / n
    sxx = sum((a - ml) ** 2 for a in lags)
    sxy = sum((a - ml) * (b - md) for a, b in zip(lags, dys))
    if sxx <= 1e-12:
        return out
    gamma = sxy / sxx
    # Rough: more negative gamma → stronger pull to mean
    out.update(
        {
            "ok": True,
            "gamma": round(gamma, 6),
            "mean_reverting_cue": gamma < -0.05,
            "interpretation": (
                "residuals/spread look mean-reverting (proxy)"
                if gamma < -0.05
                else "weak/no mean-reversion cue on this proxy — be cautious"
            ),
        }
    )
    return out


def momentum_roc_pct(closes: list[float], lookback: int = 20) -> Optional[float]:
    """Rate-of-change momentum % over lookback bars."""
    if len(closes) <= lookback or closes[-lookback - 1] <= 0:
        return None
    return round((closes[-1] / closes[-lookback - 1] - 1.0) * 100.0, 3)


def build_pair_trade_pack(symbol_a: str, symbol_b: str) -> dict[str, Any]:
    """Educational pair-trade snapshot (ratio z-score + regression hedge)."""
    sa = (symbol_a or "").upper().strip()
    sb = (symbol_b or "").upper().strip()
    out: dict[str, Any] = {"ok": False, "symbol_a": sa, "symbol_b": sb}
    if not sa or not sb or sa == sb:
        out["error"] = "need_two_distinct_symbols"
        return out
    fa = _load_ohlcv(sa, period="1y")
    fb = _load_ohlcv(sb, period="1y")
    if not fa or not fb:
        out["error"] = "ohlcv_unavailable"
        return out
    ca, cb = _align_close_series(
        list(fa.get("dates") or []),
        list(fa.get("close") or []),
        list(fb.get("dates") or []),
        list(fb.get("close") or []),
    )
    if len(ca) < 40 or len(cb) < 40:
        out["error"] = "insufficient_aligned_bars"
        return out
    # Daily % returns for correlation (Varsity PTM1 often uses return diffs)
    ra = [(ca[i] / ca[i - 1] - 1.0) for i in range(1, len(ca)) if ca[i - 1] > 0]
    rb = [(cb[i] / cb[i - 1] - 1.0) for i in range(1, len(cb)) if cb[i - 1] > 0]
    n = min(len(ra), len(rb))
    corr = _pearson_corr(ra[-n:], rb[-n:])
    ratios = [a / b for a, b in zip(ca, cb) if b > 0]
    rz = _series_zscore(ratios[-60:] if len(ratios) >= 60 else ratios)
    # Regression: A ≈ intercept + slope·B  → hedge ~ slope shares of B per share A
    reg = linear_regression_xy(cb, ca)
    resid_z = {}
    adf = {"ok": False}
    if reg.get("ok") and isinstance(reg.get("residuals"), list):
        resid = list(reg["residuals"])
        resid_z = _series_zscore(resid[-60:] if len(resid) >= 60 else resid)
        adf = adf_stationarity_proxy(resid)
        reg = {k: v for k, v in reg.items() if k != "residuals"}  # drop bulky series
    z = rz.get("z")
    signal = "no_extreme"
    if z is not None:
        if z >= 2.0:
            signal = "ratio_rich_consider_short_pair_mean_reversion"
        elif z <= -2.0:
            signal = "ratio_cheap_consider_long_pair_mean_reversion"
        elif abs(z) >= 1.0:
            signal = "mild_deviation_wait_for_stronger_density_cue"
    out.update(
        {
            "ok": True,
            "bars_aligned": len(ca),
            "corr_returns_approx": round(corr, 3) if corr is not None else None,
            "pair_quality": (
                "tight_corr_candidate"
                if corr is not None and corr >= 0.7
                else "loose_corr_be_careful"
                if corr is not None
                else "n/a"
            ),
            "ratio": rz,
            "ratio_z_signal": signal,
            "regression_a_on_b": reg,
            "residual_z": resid_z,
            "adf_proxy": adf,
            "last_prices": {"a": round(ca[-1], 2), "b": round(cb[-1], 2)},
            "note": (
                "Educational pair snapshot (Varsity-style). Correlation ≠ cointegration. "
                "Confirm ADF/half-life and liquidity before any paper trade."
            ),
        }
    )
    return out


def build_risk_management_pack(
    *,
    price: float,
    hv20_pct: Optional[float],
    atr_val: Optional[float],
    stop: Optional[float],
    reward_risk: Optional[float],
    equity: float = 500_000.0,
    risk_pct: float = 1.0,
    beta60: Optional[float] = None,
) -> dict[str, Any]:
    """Educational risk pack: VaR, recovery, %risk sizing, Kelly, 2-asset demo."""
    out: dict[str, Any] = {"ok": True, "equity_assumed": equity, "risk_pct_assumed": risk_pct}
    daily_vol = (hv20_pct / math.sqrt(252)) if hv20_pct is not None else None
    out["daily_vol_pct_from_hv20"] = round(daily_vol, 4) if daily_vol is not None else None
    out["var_1d_95_rs"] = (
        value_at_risk_parametric(equity, daily_vol, z=1.65, days=1.0)
        if daily_vol is not None
        else None
    )
    out["var_10d_95_rs"] = (
        value_at_risk_parametric(equity, daily_vol, z=1.65, days=10.0)
        if daily_vol is not None
        else None
    )
    out["recovery_table_pct"] = {
        "lose_5_need": recovery_return_needed(5),
        "lose_10_need": recovery_return_needed(10),
        "lose_20_need": recovery_return_needed(20),
        "lose_50_need": recovery_return_needed(50),
    }
    if stop is not None and price > 0:
        out["percent_risk_size"] = percent_risk_position_qty(equity, risk_pct, price, stop)
    if atr_val is not None and price > 0:
        out["percent_vol_size"] = percent_volatility_position_qty(
            equity, risk_pct, atr_val, atr_mult=1.5, price=price
        )
    if reward_risk is not None:
        k = kelly_fraction(0.45, reward_risk)
        out["kelly_full_at_45pct_wr"] = round(k, 4)
        out["kelly_quarter"] = round(max(0.0, min(0.25, k / 4.0)), 4)
        out["expectancy_r_at_45pct_wr"] = expectancy(0.45, reward_risk)
    # Diversification demo: stock vs equal Nifty-like sleeve (σ≈HV, corr≈β heuristic)
    if hv20_pct is not None:
        nifty_vol = 12.0  # educational index vol ballpark
        corr = 0.55 if beta60 is None else max(-0.2, min(0.95, float(beta60) * 0.5))
        out["two_asset_50_50_vol_pct"] = two_asset_portfolio_vol(
            0.5, 0.5, hv20_pct, nifty_vol, corr
        )
        out["portfolio_e_return_demo"] = portfolio_expected_return(
            [0.5, 0.5], [12.0, 10.0]
        )
        out["diversification_note"] = (
            f"50/50 stock+index sleeve vol≈{out['two_asset_50_50_vol_pct']}% "
            f"vs stock HV20={hv20_pct}% (corr≈{round(corr, 2)} edu)."
        )
    out["note"] = (
        "Educational risk math (Varsity-style). VaR assumes normal returns; "
        "not a guarantee. Confirm capital rules before live size."
    )
    return out


def india_roundtrip_cost_pct(side: str = "delivery") -> dict[str, Any]:
    """Approximate educational cost stack (broker-agnostic ballpark)."""
    # Rough retail delivery sell-side heavy STT; intraday lower.
    if side == "intraday":
        # STT ~0.025% on sell + exchange/GST/stamp ~0.05% roundtrip ballpark
        note = "~0.05–0.10% roundtrip excl. brokerage (intraday ballpark)"
        pct = 0.08
    else:
        note = "~0.15–0.25% roundtrip excl. brokerage (delivery ballpark; STT on sell)"
        pct = 0.20
    return {"roundtrip_cost_pct": pct, "roundtrip_cost_pct_note": note, "side": side}


def net_pnl_after_costs(gross_pnl_pct: float, cost_pct: float = 0.20) -> float:
    return gross_pnl_pct - cost_pct


def tax_drag(gain_pct: float, holding: str = "stcg") -> dict[str, float]:
    """Educational equity tax drag (rates may change — verify)."""
    rate = 0.20 if holding == "stcg" else 0.125  # illustrative post-budget ballparks
    post = gain_pct * (1.0 - rate)
    return {"tax_rate": rate, "post_tax_gain_pct": post, "holding": holding}


def sip_future_value(pmt: float, annual_rate_pct: float, years: float, compounds_per_year: int = 12) -> Optional[float]:
    if pmt <= 0 or years <= 0:
        return None
    r = (annual_rate_pct / 100.0) / compounds_per_year
    n = int(years * compounds_per_year)
    if abs(r) < 1e-12:
        return pmt * n
    return pmt * (((1 + r) ** n - 1) / r) * (1 + r)


def lump_sum_future_value(principal: float, annual_rate_pct: float, years: float) -> Optional[float]:
    if principal <= 0 or years < 0:
        return None
    return round(principal * ((1 + annual_rate_pct / 100.0) ** years), 2)


def present_value(future: float, annual_rate_pct: float, years: float) -> Optional[float]:
    if future <= 0 or years < 0:
        return None
    return round(future / ((1 + annual_rate_pct / 100.0) ** years), 2)


def rule_of_72(annual_rate_pct: float) -> Optional[float]:
    """Approx years to double ≈ 72 / rate%."""
    if annual_rate_pct <= 0:
        return None
    return round(72.0 / annual_rate_pct, 2)


def real_return_pct(nominal_pct: float, inflation_pct: float) -> float:
    """(1+n)/(1+i) − 1."""
    return round(((1 + nominal_pct / 100.0) / (1 + inflation_pct / 100.0) - 1.0) * 100.0, 3)


def retirement_corpus_needed(
    annual_expense: float,
    *,
    withdrawal_rate_pct: float = 4.0,
) -> Optional[float]:
    """Educational corpus ≈ annual expense / withdrawal rate (e.g. 4% rule)."""
    if annual_expense <= 0 or withdrawal_rate_pct <= 0:
        return None
    return round(annual_expense / (withdrawal_rate_pct / 100.0), 2)


def sip_required_for_goal(
    goal_amount: float,
    annual_rate_pct: float,
    years: float,
    compounds_per_year: int = 12,
) -> Optional[float]:
    """Monthly SIP needed to reach a future goal (ordinary annuity FV invert)."""
    if goal_amount <= 0 or years <= 0:
        return None
    r = (annual_rate_pct / 100.0) / compounds_per_year
    n = int(years * compounds_per_year)
    if n <= 0:
        return None
    if abs(r) < 1e-12:
        return round(goal_amount / n, 2)
    # FV = PMT * ((1+r)^n - 1) / r * (1+r)  [matches sip_future_value convention]
    factor = (((1 + r) ** n - 1) / r) * (1 + r)
    if factor <= 0:
        return None
    return round(goal_amount / factor, 2)


def mf_units_from_nav(amount: float, nav: float) -> Optional[float]:
    if amount <= 0 or nav <= 0:
        return None
    return round(amount / nav, 4)


def expense_ratio_drag(
    gross_annual_return_pct: float, expense_ratio_pct: float, years: float = 10.0
) -> dict[str, Any]:
    """Illustrate TER drag on ₹1L lump sum over years."""
    out: dict[str, Any] = {"ok": False}
    if years <= 0:
        return out
    gross = lump_sum_future_value(100_000, gross_annual_return_pct, years)
    net_rate = gross_annual_return_pct - expense_ratio_pct
    net = lump_sum_future_value(100_000, net_rate, years)
    if gross is None or net is None:
        return out
    out.update(
        {
            "ok": True,
            "principal": 100_000,
            "years": years,
            "gross_fv": gross,
            "net_fv_after_ter": net,
            "drag_rupees": round(gross - net, 2),
            "expense_ratio_pct": expense_ratio_pct,
            "note": "Illustrative TER subtraction from return — actual TER compounding differs slightly.",
        }
    )
    return out


def build_personal_finance_pack(
    *,
    monthly_sip: float = 5000.0,
    annual_rate_pct: float = 12.0,
    years: float = 20.0,
    inflation_pct: float = 6.0,
    monthly_expense: float = 50_000.0,
) -> dict[str, Any]:
    """Educational personal-finance calculator pack."""
    sip_fv = sip_future_value(monthly_sip, annual_rate_pct, years)
    lump = lump_sum_future_value(100_000, annual_rate_pct, years)
    real = real_return_pct(annual_rate_pct, inflation_pct)
    annual_exp = monthly_expense * 12
    corpus = retirement_corpus_needed(annual_exp, withdrawal_rate_pct=4.0)
    sip_for_corpus = (
        sip_required_for_goal(corpus, annual_rate_pct, years) if corpus else None
    )
    ter = expense_ratio_drag(annual_rate_pct, 1.0, years=years)
    return {
        "ok": True,
        "sip_fv": round(sip_fv, 2) if sip_fv is not None else None,
        "sip_monthly": monthly_sip,
        "assumptions": {
            "rate_pct": annual_rate_pct,
            "years": years,
            "inflation_pct": inflation_pct,
            "monthly_expense": monthly_expense,
        },
        "lump_1L_fv": lump,
        "years_to_double_rule72": rule_of_72(annual_rate_pct),
        "real_return_pct": real,
        "retirement_corpus_4pct_rule": corpus,
        "sip_to_fund_that_corpus": sip_for_corpus,
        "expense_ratio_drag_example": ter,
        "emergency_fund_3_to_12_months": {
            "low": round(monthly_expense * 3, 2),
            "high": round(monthly_expense * 12, 2),
            "note": "Park in liquid instruments — liquidity > return.",
        },
        "note": "Educational PF math (Varsity-style). Not personalized advice.",
    }


def historical_volatility_pct(closes: list[float], lookback: int = 20) -> Optional[float]:
    """Annualised HV% from daily log returns (IV proxy when option chain unavailable)."""
    if len(closes) < lookback + 1:
        return None
    window = closes[-(lookback + 1) :]
    rets = []
    for i in range(1, len(window)):
        if window[i - 1] <= 0 or window[i] <= 0:
            continue
        rets.append(math.log(window[i] / window[i - 1]))
    if len(rets) < max(10, lookback // 2):
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / len(rets)
    daily = math.sqrt(var)
    return daily * math.sqrt(252) * 100.0


def stochastic_kd(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> dict[str, Optional[float]]:
    if len(closes) < period:
        return {"k": None, "d": None}
    window_h = highs[-period:]
    window_l = lows[-period:]
    hh, ll = max(window_h), min(window_l)
    if hh <= ll:
        return {"k": None, "d": None}
    k = (closes[-1] - ll) / (hh - ll) * 100.0
    # %D ≈ 3-period SMA of %K using trailing closes as proxy
    ks: list[float] = []
    for end in range(period, len(closes) + 1):
        wh = highs[end - period : end]
        wl = lows[end - period : end]
        hhi, lli = max(wh), min(wl)
        if hhi <= lli:
            continue
        ks.append((closes[end - 1] - lli) / (hhi - lli) * 100.0)
    d = sum(ks[-3:]) / min(3, len(ks)) if ks else None
    return {"k": k, "d": d}


def adx_approx(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> Optional[float]:
    """Lightweight ADX approximation for trend strength scoring."""
    n = len(closes)
    if n < period + 2:
        return None
    plus_dm: list[float] = []
    minus_dm: list[float] = []
    trs: list[float] = []
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm.append(up if up > down and up > 0 else 0.0)
        minus_dm.append(down if down > up and down > 0 else 0.0)
        trs.append(
            max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
        )
    if len(trs) < period:
        return None
    atr_w = sum(trs[-period:]) / period
    if atr_w <= 0:
        return None
    plus_di = 100.0 * (sum(plus_dm[-period:]) / period) / atr_w
    minus_di = 100.0 * (sum(minus_dm[-period:]) / period) / atr_w
    denom = plus_di + minus_di
    if denom <= 0:
        return None
    dx = 100.0 * abs(plus_di - minus_di) / denom
    return dx


def fo_lot_size(symbol: str, spot: float) -> dict[str, Any]:
    """Indicative F&O lot + notional/margin ballpark (not live SPAN)."""
    hints = {
        "NIFTY": 65,
        "BANKNIFTY": 30,
        "FINNIFTY": 65,
        "RELIANCE": 500,
        "TCS": 175,
        "INFY": 400,
        "SBIN": 750,
        "ITC": 1600,
        "HDFCBANK": 550,
        "ICICIBANK": 700,
        "WIPRO": 3000,
        "LT": 150,
        "BHARTIARTL": 475,
    }
    sym = (symbol or "").upper().strip()
    if sym in hints:
        lot = hints[sym]
        lot_source = "static_hint_table"
    elif spot > 0:
        lot = max(10, int(round(120_000.0 / spot / 5.0) * 5))
        lot_source = "notional_heuristic_120k"
    else:
        lot = 100
        lot_source = "default"
    notional = round(spot * lot, 2) if spot > 0 else None
    margin_pct = 0.15  # educational SPAN+exposure ballpark for stock FUT
    margin = round(notional * margin_pct, 2) if notional else None
    return {
        "lot_size": lot,
        "lot_source": lot_source,
        "notional_per_lot": notional,
        "indicative_margin_pct": margin_pct,
        "indicative_margin_per_lot": margin,
        "note": "Lot/margin are indicative — confirm on NSE/broker before any F&O paper size.",
    }


def _fetch_nse_delivery_pct(symbol: str) -> dict[str, Any]:
    """Live delivery% from NSE quote-equity trade_info when reachable."""
    out: dict[str, Any] = {
        "delivery_pct": None,
        "delivery_qty": None,
        "traded_qty": None,
        "source": None,
        "error": None,
    }
    try:
        import requests

        session = requests.Session()
        session.headers.update(
            {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                ),
                "Accept": "application/json,text/plain,*/*",
                "Referer": "https://www.nseindia.com/",
            }
        )
        session.get("https://www.nseindia.com/", timeout=8)
        resp = session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={symbol.upper()}&section=trade_info",
            timeout=8,
        )
        if resp.status_code != 200:
            out["error"] = f"nse_http_{resp.status_code}"
            return out
        data = resp.json() or {}
        sw = data.get("securityWiseDP") or data.get("marketDeptOrderBook", {}).get("tradeInfo") or {}
        # Prefer deliveryToTradedQuantity (already %).
        del_pct = _safe_float(
            sw.get("deliveryToTradedQuantity")
            or sw.get("deliveryToTradedQuantityPercentage")
            or data.get("deliveryToTradedQuantity")
        )
        del_qty = _safe_float(sw.get("deliveryQuantity") or sw.get("quantityTraded"))
        traded = _safe_float(sw.get("quantityTraded") or sw.get("totalTradedVolume"))
        if del_pct is None and del_qty and traded and traded > 0:
            del_pct = (del_qty / traded) * 100.0
        if del_pct is not None:
            # Some feeds return ratio 0-1
            if 0 < del_pct <= 1.5:
                del_pct *= 100.0
            out["delivery_pct"] = round(del_pct, 2)
            out["delivery_qty"] = del_qty
            out["traded_qty"] = traded
            out["source"] = "nse_trade_info"
        else:
            out["error"] = "delivery_fields_missing"
    except Exception as exc:
        out["error"] = str(exc)[:120]
    return out


def structural_targets(
    price: float,
    stop: Optional[float],
    *,
    pivots: Optional[dict[str, float]] = None,
    fib: Optional[dict[str, Any]] = None,
    bb_upper: Optional[float] = None,
    resistance: Optional[float] = None,
    atr_val: Optional[float] = None,
) -> dict[str, Optional[float]]:
    """Build T1/T2 from structure (pivots/Fib/BB/ATR), not ATR 1R echo."""
    pivots = pivots or {}
    fib = fib or {}
    risk = abs(price - stop) if stop and stop < price else (atr_val or price * 0.015) * 1.5
    # Prefer targets that clear at least ~1.2R when possible.
    min_t1 = price + max(risk * 1.2, price * 0.012)
    candidates_t1 = [
        _safe_float(pivots.get("R1")),
        _safe_float(pivots.get("R2")),
        _safe_float(fib.get("extension_1618")),
        _safe_float(resistance),
        bb_upper,
        price + risk * 1.5,
        price + risk * 2.0,
    ]
    t1_opts = sorted(x for x in candidates_t1 if x is not None and x >= min_t1)
    t1 = t1_opts[0] if t1_opts else min_t1
    candidates_t2 = [
        _safe_float(pivots.get("R2")),
        _safe_float(pivots.get("R3")),
        _safe_float(fib.get("extension_1618")),
        bb_upper,
        t1 + risk,
        price + risk * 2.5,
    ]
    t2_opts = sorted(x for x in candidates_t2 if x is not None and x > t1 * 1.002)
    t2 = t2_opts[0] if t2_opts else (t1 + risk)
    rr = risk_reward(price, stop, t1) if stop and t1 else None
    rr2 = risk_reward(price, stop, t2) if stop and t2 else None
    return {"target_1": t1, "target_2": t2, "risk_reward_t1": rr, "risk_reward_t2": rr2}


def circuit_distance(price: float, week_high: Optional[float], week_low: Optional[float], band_pct: float = 10.0) -> dict[str, Optional[float]]:
    """Distance to illustrative ±band and 52w extremes."""
    if price <= 0:
        return {}
    up_band = price * (1 + band_pct / 100.0)
    dn_band = price * (1 - band_pct / 100.0)
    return {
        "illust_upper_band": round(up_band, 2),
        "illust_lower_band": round(dn_band, 2),
        "pct_to_52w_high": round((week_high - price) / price * 100.0, 2) if week_high else None,
        "pct_to_52w_low": round((price - week_low) / price * 100.0, 2) if week_low else None,
        "band_pct_assumed": band_pct,
    }


def option_breakevens(spot: float, strike: float, premium: float, option_type: str = "call") -> dict[str, float]:
    if option_type == "put":
        return {"breakeven": strike - premium, "max_loss": premium, "spot": spot}
    return {"breakeven": strike + premium, "max_loss": premium, "spot": spot}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def call_intrinsic(spot: float, strike: float) -> float:
    return max(0.0, spot - strike)


def put_intrinsic(spot: float, strike: float) -> float:
    return max(0.0, strike - spot)


def option_time_value(premium: float, intrinsic: float) -> float:
    return round(premium - intrinsic, 4)


def option_moneyness(spot: float, strike: float, option_type: str = "call") -> str:
    if spot <= 0 or strike <= 0:
        return "n/a"
    rel = abs(spot - strike) / spot
    ot = (option_type or "call").lower()
    if rel <= 0.005:
        return "ATM"
    if ot == "put":
        if spot < strike:
            return "ITM"
        return "OTM"
    if spot > strike:
        return "ITM"
    return "OTM"


def call_buyer_pnl_expiry(spot: float, strike: float, premium: float) -> float:
    return round(call_intrinsic(spot, strike) - premium, 2)


def call_seller_pnl_expiry(spot: float, strike: float, premium: float) -> float:
    return round(-call_buyer_pnl_expiry(spot, strike, premium), 2)


def put_buyer_pnl_expiry(spot: float, strike: float, premium: float) -> float:
    return round(put_intrinsic(spot, strike) - premium, 2)


def put_seller_pnl_expiry(spot: float, strike: float, premium: float) -> float:
    return round(-put_buyer_pnl_expiry(spot, strike, premium), 2)


def black_scholes_greeks(
    spot: float,
    strike: float,
    *,
    days_to_expiry: float = 21.0,
    hv_pct: float,
    rf: float = 0.065,
    option_type: str = "call",
) -> dict[str, Any]:
    """Educational BS European Greeks (HV as σ). Not a live IV surface."""
    out: dict[str, Any] = {"ok": False, "model": "black_scholes_hv_proxy"}
    if spot <= 0 or strike <= 0 or hv_pct is None or hv_pct <= 0:
        return out
    t_days = max(float(days_to_expiry), 1.0 / 24.0)  # avoid 0
    t = t_days / 365.0
    sigma = hv_pct / 100.0
    try:
        d1 = (math.log(spot / strike) + (rf + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
        d2 = d1 - sigma * math.sqrt(t)
    except (ValueError, ZeroDivisionError):
        return out
    pdf = _norm_pdf(d1)
    disc = math.exp(-rf * t)
    ot = (option_type or "call").lower()
    if ot == "put":
        delta = _norm_cdf(d1) - 1.0
        price = strike * disc * _norm_cdf(-d2) - spot * _norm_cdf(-d1)
        theta_yr = (
            -(spot * pdf * sigma) / (2.0 * math.sqrt(t))
            + rf * strike * disc * _norm_cdf(-d2)
        )
    else:
        delta = _norm_cdf(d1)
        price = spot * _norm_cdf(d1) - strike * disc * _norm_cdf(d2)
        theta_yr = (
            -(spot * pdf * sigma) / (2.0 * math.sqrt(t))
            - rf * strike * disc * _norm_cdf(d2)
        )
    gamma = pdf / (spot * sigma * math.sqrt(t))
    vega_per_vol_point = spot * pdf * math.sqrt(t) / 100.0  # P&L per 1 vol point
    theta_per_day = theta_yr / 365.0
    intrinsic = call_intrinsic(spot, strike) if ot != "put" else put_intrinsic(spot, strike)
    tv = option_time_value(price, intrinsic)
    out.update(
        {
            "ok": True,
            "option_type": ot,
            "spot": round(spot, 2),
            "strike": round(strike, 2),
            "days_to_expiry": round(t_days, 2),
            "hv_pct_used": round(hv_pct, 2),
            "rf": rf,
            "moneyness": option_moneyness(spot, strike, ot),
            "theoretical_premium": round(price, 2),
            "intrinsic": round(intrinsic, 2),
            "time_value": round(tv, 2),
            "delta": round(delta, 4),
            "gamma": round(gamma, 6),
            "theta_per_day": round(theta_per_day, 4),
            "vega_per_vol_point": round(vega_per_vol_point, 4),
            "note": "HV used as σ — not market IV; European BS assumptions.",
        }
    )
    return out


def commodity_pnl_per_tick(lot_size: float, quote_unit: float, tick_size: float) -> Optional[float]:
    """Varsity-style: P&L/tick = (Lot size / Quotation unit) × Tick size."""
    if lot_size <= 0 or quote_unit <= 0:
        return None
    return round((lot_size / quote_unit) * tick_size, 4)


def commodity_contract_value(price: float, lot_size: float, quote_unit: float) -> Optional[float]:
    """Contract value = (Price × Lot size) / Quotation unit."""
    if price <= 0 or lot_size <= 0 or quote_unit <= 0:
        return None
    return round((price * lot_size) / quote_unit, 2)


def tbill_discount_yield(par: float, price: float, days_to_maturity: float) -> Optional[float]:
    """T-bill yield ≈ (Discount/Price) × (365/days) — Varsity educational form."""
    if par <= 0 or price <= 0 or days_to_maturity <= 0 or price > par:
        return None
    discount = par - price
    return round((discount / price) * (365.0 / days_to_maturity) * 100.0, 4)


def interest_rate_parity_forward(
    spot: float,
    r_quote: float,
    r_base: float,
    days: float = 30.0,
) -> Optional[float]:
    """FX forward ≈ S × (1+r_quote·T)/(1+r_base·T). For USDINR: quote=INR, base=USD."""
    if spot <= 0 or days < 0:
        return None
    t = days / 365.0
    denom = 1.0 + r_base * t
    if denom <= 0:
        return None
    return round(spot * (1.0 + r_quote * t) / denom, 4)


def india_gold_price_per_10g(
    usd_per_oz: float,
    usdinr: float,
    *,
    duty_factor: float = 1.0,
) -> Optional[float]:
    """Rough MCX-style Rs/10g from international $/oz and USDINR (duties via factor)."""
    if usd_per_oz <= 0 or usdinr <= 0:
        return None
    grams_per_oz = 31.1035
    return round((usd_per_oz / grams_per_oz) * 10.0 * usdinr * duty_factor, 2)


def parse_gsec_symbol(symbol: str) -> dict[str, Any]:
    """Parse NSE-style G-Sec tags like 740GS2035A → coupon/maturity hints."""
    out: dict[str, Any] = {"ok": False, "raw": symbol}
    text = (symbol or "").upper().strip()
    m = re.search(r"^(\d{3,4})GS(\d{4})", text)
    if not m:
        m2 = re.search(r"^(\d{1,2}\.\d{2})([A-Z]{2})SDL(\d{4})", text)
        if m2:
            return {
                "ok": True,
                "kind": "sdl",
                "coupon_pct": float(m2.group(1)),
                "state": m2.group(2),
                "maturity_year": int(m2.group(3)),
                "semi_annual_pct": round(float(m2.group(1)) / 2.0, 4),
            }
        return out
    raw_coup = m.group(1)
    # 740 → 7.40%; 662 → 6.62%; 1050 → 10.50%
    coupon = int(raw_coup) / (100.0 if len(raw_coup) >= 3 else 10.0)
    out.update(
        {
            "ok": True,
            "kind": "gsec",
            "coupon_pct": coupon,
            "maturity_year": int(m.group(2)),
            "semi_annual_pct": round(coupon / 2.0, 4),
        }
    )
    return out


def fetch_ccg_market_snapshot() -> dict[str, Any]:
    """Best-effort USDINR / gold / crude quotes via yfinance (educational)."""
    out: dict[str, Any] = {"ok": False, "source": "yfinance"}
    try:
        import yfinance as yf  # type: ignore

        tickers = {
            "usdinr": "INR=X",
            "gold_usd_oz": "GC=F",
            "crude_usd_bbl": "CL=F",
            "silver_usd_oz": "SI=F",
            "natural_gas_usd": "NG=F",
        }
        prices: dict[str, Optional[float]] = {}
        for key, tic in tickers.items():
            try:
                hist = yf.Ticker(tic).history(period="5d")
                if hist is not None and not hist.empty:
                    prices[key] = float(hist["Close"].iloc[-1])
                else:
                    prices[key] = None
            except Exception:
                prices[key] = None
        usdinr = prices.get("usdinr")
        gold = prices.get("gold_usd_oz")
        india_gold = (
            india_gold_price_per_10g(gold, usdinr, duty_factor=1.0)
            if gold and usdinr
            else None
        )
        # Educational IRP: assume INR 6.5%, USD 4.5%, 30d
        irp = (
            interest_rate_parity_forward(usdinr, 0.065, 0.045, 30.0)
            if usdinr
            else None
        )
        gold_big = None
        if india_gold:
            gold_big = {
                "quote_unit_grams": 10,
                "lot_grams": 1000,
                "price_per_10g_approx": india_gold,
                "contract_value": commodity_contract_value(india_gold, 1000, 10),
                "pnl_per_tick_rs": commodity_pnl_per_tick(1000, 10, 1.0),
                "note": "Approx from COMEX$/oz×USDINR; MCX includes duties/taxes — confirm live MCX.",
            }
        out.update(
            {
                "ok": any(v is not None for v in prices.values()),
                "usdinr": round(usdinr, 4) if usdinr else None,
                "gold_usd_oz": round(gold, 2) if gold else None,
                "silver_usd_oz": round(prices["silver_usd_oz"], 2)
                if prices.get("silver_usd_oz")
                else None,
                "crude_usd_bbl": round(prices["crude_usd_bbl"], 2)
                if prices.get("crude_usd_bbl")
                else None,
                "natural_gas_usd": round(prices["natural_gas_usd"], 3)
                if prices.get("natural_gas_usd")
                else None,
                "india_gold_rs_per_10g_approx": india_gold,
                "usdinr_30d_irp_fair": irp,
                "gold_mcx_big_edu": gold_big,
                "tbill_example_91d": {
                    "par": 100,
                    "price": 97,
                    "days": 91,
                    "yield_pct": tbill_discount_yield(100, 97, 91),
                    "note": "Illustrative Varsity-style example — not a live auction yield.",
                },
            }
        )
    except Exception as exc:
        out["error"] = str(exc)[:160]
    return out


def build_option_theory_pack(
    spot: float,
    hv_pct: Optional[float],
    *,
    days_to_expiry: float = 21.0,
) -> dict[str, Any]:
    """ATM call/put theory snapshot: moneyness, IV/TV, expiry P&L samples, Greeks."""
    out: dict[str, Any] = {"ok": False}
    if spot <= 0 or hv_pct is None or hv_pct <= 0:
        return out
    strike = _round_strike(spot)
    call_g = black_scholes_greeks(
        spot, strike, days_to_expiry=days_to_expiry, hv_pct=hv_pct, option_type="call"
    )
    put_g = black_scholes_greeks(
        spot, strike, days_to_expiry=days_to_expiry, hv_pct=hv_pct, option_type="put"
    )
    c_prem = float(call_g.get("theoretical_premium") or 0)
    p_prem = float(put_g.get("theoretical_premium") or 0)
    # Sample expiry P&L at ±2% / unchanged
    spots = {
        "unchanged": spot,
        "up_2pct": spot * 1.02,
        "down_2pct": spot * 0.98,
    }
    call_buyer = {k: call_buyer_pnl_expiry(v, strike, c_prem) for k, v in spots.items()}
    put_buyer = {k: put_buyer_pnl_expiry(v, strike, p_prem) for k, v in spots.items()}
    out.update(
        {
            "ok": True,
            "atm_strike": strike,
            "days_to_expiry": days_to_expiry,
            "hv_pct": round(hv_pct, 2),
            "call": call_g,
            "put": put_g,
            "call_buyer_pnl_at_expiry_sample": call_buyer,
            "put_buyer_pnl_at_expiry_sample": put_buyer,
            "call_breakeven": round(strike + c_prem, 2),
            "put_breakeven": round(strike - p_prem, 2),
            "summary_rules": {
                "call_buyer_max_loss": "premium_paid",
                "call_seller_max_profit": "premium_received",
                "put_buyer_max_loss": "premium_paid",
                "put_seller_max_profit": "premium_received",
                "call_intrinsic": "max(0, spot − strike)",
                "put_intrinsic": "max(0, strike − spot)",
                "premium": "intrinsic + time_value",
            },
        }
    )
    return out


def _strike_step(spot: float) -> float:
    if spot >= 20000:
        return 100.0
    if spot >= 5000:
        return 50.0
    if spot >= 500:
        return 10.0
    if spot >= 100:
        return 5.0
    return 1.0


def _round_strike(spot: float, step: Optional[float] = None) -> float:
    st = step or _strike_step(spot)
    if st <= 0:
        return round(spot, 2)
    return round(round(spot / st) * st, 2)


def bull_call_spread(k_buy: float, k_sell: float, net_debit: float) -> dict[str, Any]:
    width = k_sell - k_buy
    if width <= 0 or net_debit < 0:
        return {"ok": False, "error": "need k_buy < k_sell and debit >= 0"}
    return {
        "ok": True,
        "name": "bull_call_spread",
        "view": "moderately_bullish",
        "cashflow": "debit",
        "structure": "buy_lower_call_sell_higher_call",
        "width": round(width, 2),
        "max_loss": round(net_debit, 2),
        "max_profit": round(width - net_debit, 2),
        "breakeven": round(k_buy + net_debit, 2),
    }


def bull_put_spread(k_sell: float, k_buy: float, net_credit: float) -> dict[str, Any]:
    width = k_sell - k_buy
    if width <= 0 or net_credit < 0:
        return {"ok": False, "error": "need k_sell > k_buy and credit >= 0"}
    return {
        "ok": True,
        "name": "bull_put_spread",
        "view": "moderately_bullish",
        "cashflow": "credit",
        "structure": "sell_higher_put_buy_lower_put",
        "width": round(width, 2),
        "max_profit": round(net_credit, 2),
        "max_loss": round(width - net_credit, 2),
        "breakeven": round(k_sell - net_credit, 2),
    }


def bear_put_spread(k_buy: float, k_sell: float, net_debit: float) -> dict[str, Any]:
    width = k_buy - k_sell
    if width <= 0 or net_debit < 0:
        return {"ok": False, "error": "need k_buy > k_sell and debit >= 0"}
    return {
        "ok": True,
        "name": "bear_put_spread",
        "view": "moderately_bearish",
        "cashflow": "debit",
        "structure": "buy_higher_put_sell_lower_put",
        "width": round(width, 2),
        "max_loss": round(net_debit, 2),
        "max_profit": round(width - net_debit, 2),
        "breakeven": round(k_buy - net_debit, 2),
    }


def bear_call_spread(k_sell: float, k_buy: float, net_credit: float) -> dict[str, Any]:
    width = k_buy - k_sell
    if width <= 0 or net_credit < 0:
        return {"ok": False, "error": "need k_buy > k_sell and credit >= 0"}
    return {
        "ok": True,
        "name": "bear_call_spread",
        "view": "moderately_bearish",
        "cashflow": "credit",
        "structure": "sell_lower_call_buy_higher_call",
        "width": round(width, 2),
        "max_profit": round(net_credit, 2),
        "max_loss": round(width - net_credit, 2),
        "breakeven": round(k_sell + net_credit, 2),
    }


def long_straddle(strike: float, call_prem: float, put_prem: float) -> dict[str, Any]:
    net = call_prem + put_prem
    if strike <= 0 or net < 0:
        return {"ok": False}
    return {
        "ok": True,
        "name": "long_straddle",
        "view": "large_move_either_way",
        "cashflow": "debit",
        "max_loss": round(net, 2),
        "max_profit": "unlimited_theoretical",
        "breakeven_up": round(strike + net, 2),
        "breakeven_down": round(strike - net, 2),
    }


def short_straddle(strike: float, call_prem: float, put_prem: float) -> dict[str, Any]:
    net = call_prem + put_prem
    if strike <= 0 or net < 0:
        return {"ok": False}
    return {
        "ok": True,
        "name": "short_straddle",
        "view": "range_bound_vol_crush",
        "cashflow": "credit",
        "max_profit": round(net, 2),
        "max_loss": "unlimited_theoretical",
        "breakeven_up": round(strike + net, 2),
        "breakeven_down": round(strike - net, 2),
        "risk_note": "Naked short straddle has large margin and tail risk.",
    }


def long_strangle(
    k_put: float, k_call: float, put_prem: float, call_prem: float
) -> dict[str, Any]:
    net = put_prem + call_prem
    if k_call <= k_put or net < 0:
        return {"ok": False}
    return {
        "ok": True,
        "name": "long_strangle",
        "view": "large_move_either_way_cheaper_than_straddle",
        "cashflow": "debit",
        "max_loss": round(net, 2),
        "breakeven_up": round(k_call + net, 2),
        "breakeven_down": round(k_put - net, 2),
    }


def short_strangle(
    k_put: float, k_call: float, put_prem: float, call_prem: float
) -> dict[str, Any]:
    net = put_prem + call_prem
    if k_call <= k_put or net < 0:
        return {"ok": False}
    return {
        "ok": True,
        "name": "short_strangle",
        "view": "range_bound_between_strikes",
        "cashflow": "credit",
        "max_profit": round(net, 2),
        "max_loss": "unlimited_theoretical",
        "breakeven_up": round(k_call + net, 2),
        "breakeven_down": round(k_put - net, 2),
        "risk_note": "Short strangle needs margin; losses grow outside wings.",
    }


def call_ratio_back_spread(k_short: float, k_long: float, net_credit: float) -> dict[str, Any]:
    """Classic 1×2: sell 1 lower call, buy 2 higher calls (often net credit)."""
    width = k_long - k_short
    if width <= 0:
        return {"ok": False}
    max_loss = width - net_credit
    return {
        "ok": True,
        "name": "call_ratio_back_spread",
        "view": "strongly_bullish",
        "structure": "sell_1_lower_call_buy_2_higher_calls",
        "width": round(width, 2),
        "max_loss": round(max_loss, 2),
        "max_loss_at": round(k_long, 2),
        "profit_if_expires_well_below": round(net_credit, 2),
        "lower_breakeven": round(k_short + net_credit, 2),
        "upper_breakeven": round(k_long + max_loss, 2),
        "upside": "large_if_strong_rally_past_upper_be",
    }


def put_ratio_back_spread(k_short: float, k_long: float, net_credit: float) -> dict[str, Any]:
    """Classic 1×2: sell 1 higher put, buy 2 lower puts (strongly bearish)."""
    width = k_short - k_long
    if width <= 0:
        return {"ok": False}
    max_loss = width - net_credit
    return {
        "ok": True,
        "name": "put_ratio_back_spread",
        "view": "strongly_bearish",
        "structure": "sell_1_higher_put_buy_2_lower_puts",
        "width": round(width, 2),
        "max_loss": round(max_loss, 2),
        "max_loss_at": round(k_long, 2),
        "profit_if_expires_well_above": round(net_credit, 2),
        "upper_breakeven": round(k_short - net_credit, 2),
        "lower_breakeven": round(k_long - max_loss, 2),
        "downside": "large_if_strong_selloff_past_lower_be",
    }


def iron_condor(
    k_put_long: float,
    k_put_short: float,
    k_call_short: float,
    k_call_long: float,
    net_credit: float,
) -> dict[str, Any]:
    put_w = k_put_short - k_put_long
    call_w = k_call_long - k_call_short
    if put_w <= 0 or call_w <= 0 or k_call_short <= k_put_short or net_credit < 0:
        return {"ok": False}
    wing = max(put_w, call_w)
    return {
        "ok": True,
        "name": "iron_condor",
        "view": "range_bound",
        "structure": "short_put_spread_+_short_call_spread",
        "max_profit": round(net_credit, 2),
        "max_loss": round(wing - net_credit, 2),
        "breakeven_down": round(k_put_short - net_credit, 2),
        "breakeven_up": round(k_call_short + net_credit, 2),
        "note": "Assumes equal-ish wings; confirm live premiums/margin.",
    }


def synthetic_long(call_prem: float, put_prem: float, strike: float) -> dict[str, Any]:
    """Long call + short put ≈ long futures/synthetic long (parity intuition)."""
    return {
        "ok": True,
        "name": "synthetic_long",
        "structure": "long_call_short_put_same_strike",
        "view": "bullish_like_futures",
        "approx_cost_vs_strike": round(call_prem - put_prem, 2),
        "breakeven": round(strike + (call_prem - put_prem), 2),
        "note": "Parity/arbitrage gaps can exist briefly after costs.",
    }


def build_option_strategy_examples(
    spot: float,
    atm_premium: Optional[float],
    *,
    width_steps: int = 2,
) -> dict[str, Any]:
    """Educational ATM±OTM strategy cards using HV-proxy premiums (not live chain)."""
    out: dict[str, Any] = {"ok": False, "note": "Educational premiums — not live option chain."}
    if spot <= 0 or atm_premium is None or atm_premium <= 0:
        return out
    step = _strike_step(spot)
    atm = _round_strike(spot, step)
    width = step * max(1, width_steps)
    k_otm_call = round(atm + width, 2)
    k_otm_put = round(atm - width, 2)
    # Rough OTM premium ~55% of ATM for ~2 steps (illustrative).
    otm_prem = round(atm_premium * 0.55, 2)
    itm_call_prem = round(atm_premium * 1.45, 2)
    debit_bc = round(max(0.5, atm_premium - otm_prem), 2)
    credit_bp = round(max(0.5, atm_premium - otm_prem), 2)
    debit_bp_bear = debit_bc
    credit_bc_bear = credit_bp
    strangle_debit = round(otm_prem * 2, 2)
    # Call ratio: sell ITM, buy 2 OTM → credit ≈ itm - 2*otm
    cr_credit = round(itm_call_prem - 2 * otm_prem, 2)
    # Iron condor credit ≈ 2*otm_wing_credit rough
    ic_credit = round(max(0.5, otm_prem * 0.7), 2)
    k_put_long = round(k_otm_put - width, 2)
    k_call_long = round(k_otm_call + width, 2)

    out.update(
        {
            "ok": True,
            "spot": round(spot, 2),
            "atm_strike": atm,
            "width": width,
            "atm_premium_proxy": round(atm_premium, 2),
            "otm_premium_proxy": otm_prem,
            "bull_call_spread": bull_call_spread(atm, k_otm_call, debit_bc),
            "bull_put_spread": bull_put_spread(atm, k_otm_put, credit_bp),
            "bear_put_spread": bear_put_spread(atm, k_otm_put, debit_bp_bear),
            "bear_call_spread": bear_call_spread(atm, k_otm_call, credit_bc_bear),
            "long_straddle": long_straddle(atm, atm_premium, atm_premium),
            "short_straddle": short_straddle(atm, atm_premium, atm_premium),
            "long_strangle": long_strangle(k_otm_put, k_otm_call, otm_prem, otm_prem),
            "short_strangle": short_strangle(k_otm_put, k_otm_call, otm_prem, otm_prem),
            "call_ratio_back_spread": call_ratio_back_spread(atm - width, k_otm_call, max(cr_credit, 1.0)),
            "put_ratio_back_spread": put_ratio_back_spread(atm + width, k_otm_put, max(cr_credit, 1.0)),
            "iron_condor": iron_condor(k_put_long, k_otm_put, k_otm_call, k_call_long, ic_credit),
            "synthetic_long": synthetic_long(atm_premium, atm_premium, atm),
            "pcr_note": "PCR = Put OI ÷ Call OI (or volume). >1 often more put heavy — not a standalone signal.",
            "max_pain_note": (
                "Max pain ≈ strike where total option-writer pain (calls+puts) is minimized; "
                "needs full chain OI — not computed without live chain."
            ),
        }
    )
    return out


def futures_basis(futures: float, spot: float) -> Optional[float]:
    if spot <= 0:
        return None
    return futures - spot


def futures_fair_value(
    spot: float,
    *,
    rf: float = 0.065,
    days_to_expiry: float = 30.0,
    dividend_cash: float = 0.0,
) -> Optional[float]:
    """Cost-of-carry fair value: F ≈ S·(1 + Rf·T/365) − D (Varsity-style)."""
    if spot <= 0 or days_to_expiry < 0:
        return None
    t = days_to_expiry / 365.0
    return round(spot * (1.0 + rf * t) - max(0.0, dividend_cash), 2)


def futures_leverage(contract_value: float, margin: float) -> dict[str, Optional[float]]:
    """Leverage = contract value / margin; wipeout ≈ 1/leverage adverse move."""
    if contract_value <= 0 or margin <= 0:
        return {"leverage": None, "wipeout_pct": None, "double_pct": None}
    lev = contract_value / margin
    wipe = 100.0 / lev
    return {
        "leverage": round(lev, 2),
        "wipeout_pct": round(wipe, 2),
        "double_pct": round(wipe, 2),  # same magnitude doubles margin equity if favorable
    }


def futures_mtm_pnl(
    entry: float,
    settle: float,
    lot_size: int,
    *,
    side: str = "long",
) -> Optional[float]:
    """Daily/period MTM on one lot: (settle−entry)*lot for long; reverse for short."""
    if lot_size <= 0:
        return None
    diff = settle - entry
    if (side or "long").lower() in {"short", "sell", "s"}:
        diff = -diff
    return round(diff * lot_size, 2)


def impact_cost_pct(best_ask: float, best_bid: float) -> Optional[float]:
    """Round-trip impact cost % = (ask−bid) / mid  (Varsity-style liquidity cue)."""
    if best_ask <= 0 or best_bid <= 0 or best_ask < best_bid:
        return None
    mid = (best_ask + best_bid) / 2.0
    if mid <= 0:
        return None
    return round((best_ask - best_bid) / mid * 100.0, 4)


def hedge_nifty_lots(
    portfolio_value: float,
    portfolio_beta: float,
    nifty_futures: float,
    nifty_lot: int = 65,
) -> dict[str, Any]:
    """Portfolio hedge: hedge_value = β·PV; lots = hedge_value / (F·lot)."""
    out: dict[str, Any] = {
        "hedge_value": None,
        "contract_value": None,
        "lots_exact": None,
        "lots_floor": None,
        "lots_ceil": None,
        "note": "Educational beta-hedge — not live SPAN; round lots under/over-hedge.",
    }
    if portfolio_value <= 0 or nifty_futures <= 0 or nifty_lot <= 0:
        return out
    beta = portfolio_beta if portfolio_beta is not None else 1.0
    hedge_value = abs(beta) * portfolio_value
    cv = nifty_futures * nifty_lot
    lots = hedge_value / cv if cv > 0 else None
    out.update(
        {
            "portfolio_beta": round(beta, 3),
            "hedge_value": round(hedge_value, 2),
            "contract_value": round(cv, 2),
            "lots_exact": round(lots, 2) if lots is not None else None,
            "lots_floor": int(math.floor(lots)) if lots is not None else None,
            "lots_ceil": int(math.ceil(lots)) if lots is not None else None,
            "side": "short_nifty_futures_to_hedge_long_portfolio",
        }
    )
    return out


def oi_price_interpretation(price_up: bool, oi_up: bool) -> str:
    """Classic price + OI reading (educational; confirm with volume)."""
    if price_up and oi_up:
        return "price_up_oi_up_new_longs_build"
    if price_up and not oi_up:
        return "price_up_oi_down_short_covering"
    if (not price_up) and oi_up:
        return "price_down_oi_up_new_shorts_build"
    return "price_down_oi_down_long_liquidation"


def build_p0_analysis_pack(
    symbol: str,
    *,
    fund_hints: Optional[dict[str, Any]] = None,
    risk_rupees: float = 5000.0,
    horizon: str = "swing",
) -> dict[str, Any]:
    """Full quantitative pack (P0+B/C) + trade plan. Safe to call from ask_llm."""
    sym = (symbol or "").upper().strip()
    pack: dict[str, Any] = {"symbol": sym, "ok": False, "formulas": []}
    if not sym:
        return pack

    frame = _load_ohlcv(sym, period="1y")
    if not frame or len(frame.get("close") or []) < 30:
        frame = _load_ohlcv(sym, period="6mo")
    if not frame or len(frame.get("close") or []) < 30:
        pack["error"] = "insufficient_ohlcv"
        return pack

    opens = frame["open"]
    highs = frame["high"]
    lows = frame["low"]
    closes = frame["close"]
    volumes = frame["volume"]
    price = closes[-1]
    prev_high, prev_low, prev_close = highs[-2], lows[-2], closes[-2]

    rsi = wilder_rsi(closes, 14)
    div = rsi_divergence_cue(closes, 14, 20)
    bb = bollinger_pct_b_bandwidth(closes, 20, 2.0)
    atr_val = atr(highs, lows, closes, 14)
    sizing = atr_stop_and_size(price, atr_val or 0.0, side="long", atr_mult=1.5, risk_rupees=risk_rupees)
    stop = sizing.get("stop")
    target_1r = sizing.get("target_1r")
    pivots = classic_pivots(prev_high, prev_low, prev_close)
    cam = camarilla_pivots(prev_high, prev_low, prev_close)
    cpr = central_pivot_range(prev_high, prev_low, prev_close)
    vol = volume_zscore(volumes, 20)

    stock_dates = list(frame.get("dates") or [])
    nifty_series = _load_nifty_series("1y")
    if len(nifty_series.get("close") or []) < 40:
        nifty_series = _load_nifty_series("6mo")
    nifty = list(nifty_series.get("close") or [])
    nifty_dates = list(nifty_series.get("dates") or [])
    rs20 = (
        relative_strength(closes, nifty, 20, stock_dates=stock_dates, index_dates=nifty_dates)
        if nifty
        else None
    )
    rs60 = (
        relative_strength(closes, nifty, 60, stock_dates=stock_dates, index_dates=nifty_dates)
        if nifty
        else None
    )
    beta60 = (
        beta_vs_index(closes, nifty, 60, stock_dates=stock_dates, index_dates=nifty_dates, min_obs=40)
        if nifty
        else None
    )
    beta120 = (
        beta_vs_index(
            closes, nifty, 120, stock_dates=stock_dates, index_dates=nifty_dates, min_obs=80
        )
        if nifty
        else None
    )

    hints = fund_hints or {}
    yf_fund = _fundamentals_from_yfinance(sym)
    pe = _safe_float(hints.get("pe") or hints.get("pe_ratio") or yf_fund.get("pe"))
    eps = _safe_float(hints.get("eps") or yf_fund.get("eps"))
    pe_sector = _safe_float(
        hints.get("pe_sector")
        or hints.get("pe_sector_avg")
        or hints.get("sector_pe")
    )
    # Ignore invented "~25 (market avg)" style strings already coerced poorly.
    if pe_sector is not None and pe_sector <= 0:
        pe_sector = None
    sector_pe_premium = None
    if pe is not None and pe_sector is not None and pe_sector > 0:
        sector_pe_premium = (pe / pe_sector) - 1.0
    ey = earnings_yield(price, eps=eps, pe=pe)
    ev_pack = ev_ebitda(
        yf_fund.get("market_cap") or _safe_float(hints.get("market_cap")),
        yf_fund.get("total_debt"),
        yf_fund.get("cash"),
        yf_fund.get("ebitda"),
    )
    growth = yf_fund.get("earnings_growth") or yf_fund.get("revenue_growth")
    peg = peg_ratio(pe, growth)
    jpe = justified_pe_rough(growth)

    bw = bb.get("bandwidth")
    if bw is None:
        bw_label = "n/a"
    elif bw < 0.05:
        bw_label = "squeeze"
    elif bw > 0.12:
        bw_label = "expansion"
    else:
        bw_label = "normal"

    macd_pack = macd(closes)
    st_pack = supertrend(highs, lows, closes, period=10, multiplier=3.0)
    fib_pack = fibonacci_levels(highs, lows, closes, lookback=60)
    vwap = session_vwap_approx(highs, lows, closes, volumes, window=20)
    rstats = risk_stats(closes, lookback=60)
    stoch = stochastic_kd(highs, lows, closes, 14)
    adx_val = adx_approx(highs, lows, closes, 14)
    hv20 = historical_volatility_pct(closes, 20)
    hv60 = historical_volatility_pct(closes, 60)
    delivery = _fetch_nse_delivery_pct(sym)
    fo = fo_lot_size(sym, price)
    # Educational ATM option BE using HV-implied premium ≈ 0.4*S*σ*√(T/365)
    atm_prem = None
    if hv20 is not None and price > 0:
        atm_prem = round(0.4 * price * (hv20 / 100.0) * math.sqrt(21 / 365.0), 2)
    opt_call = option_breakevens(price, price, atm_prem or 0.0, "call") if atm_prem else {}
    opt_put = option_breakevens(price, price, atm_prem or 0.0, "put") if atm_prem else {}
    # Cost-of-carry: F ≈ S*(1+Rf*T/365) − D  (D ≈ spot * div_yield * T)
    days_to_expiry = 30.0
    rf_carry = 0.065
    dy_pct = _safe_float(yf_fund.get("dividend_yield"))
    div_cash = (
        round(price * (dy_pct / 100.0) * (days_to_expiry / 365.0), 4)
        if dy_pct is not None and price > 0
        else 0.0
    )
    fut_fair = futures_fair_value(
        price, rf=rf_carry, days_to_expiry=days_to_expiry, dividend_cash=div_cash
    )
    basis_fair = futures_basis(fut_fair, price) if fut_fair is not None else None
    basis_regime = None
    if basis_fair is not None:
        if basis_fair > 0:
            basis_regime = "premium_contango_proxy"
        elif basis_fair < 0:
            basis_regime = "discount_backwardation_proxy"
        else:
            basis_regime = "flat_basis"
    lev_pack = futures_leverage(
        float(fo.get("notional_per_lot") or 0),
        float(fo.get("indicative_margin_per_lot") or 0),
    )
    # Sample MTM: ±1% move on one long lot (educational)
    mtm_up = futures_mtm_pnl(price, price * 1.01, int(fo.get("lot_size") or 0), side="long")
    mtm_dn = futures_mtm_pnl(price, price * 0.99, int(fo.get("lot_size") or 0), side="long")
    nifty_spot = float(nifty[-1]) if nifty else None
    nifty_lot_hint = int(fo_lot_size("NIFTY", nifty_spot or 0).get("lot_size") or 65)
    hedge_ex = (
        hedge_nifty_lots(
            1_000_000.0,
            float(beta60 if beta60 is not None else (yf_fund.get("beta") or 1.0)),
            float(fut_fair if sym in {"NIFTY", "NIFTY50"} and fut_fair else (nifty_spot or 0)),
            nifty_lot=nifty_lot_hint,
        )
        if nifty_spot
        else {}
    )
    opt_strategies = build_option_strategy_examples(price, atm_prem)
    opt_theory = build_option_theory_pack(price, hv20, days_to_expiry=21.0)

    struct = structural_targets(
        price,
        stop,
        pivots=pivots,
        fib=fib_pack,
        bb_upper=bb.get("upper"),
        atr_val=atr_val,
    )
    rr = struct.get("risk_reward_t1")
    exp = expectancy(0.45, rr) if rr is not None else None
    costs = india_roundtrip_cost_pct(
        "delivery"
        if horizon in {"long", "long_term", "invest"}
        else "intraday"
        if horizon == "intraday"
        else "delivery"
    )
    tax = {
        "stcg": tax_drag(5.0, "stcg"),
        "ltcg": tax_drag(5.0, "ltcg"),
        "note": "Example tax drag on +5% gross gain (illustrative slabs).",
    }
    circuit = circuit_distance(
        price,
        yf_fund.get("fifty_two_week_high"),
        yf_fund.get("fifty_two_week_low"),
        band_pct=10.0,
    )
    circuit["note"] = "±10% band is illustrative; actual NSE circuit depends on stock group."
    k_full = kelly_fraction(0.45, rr) if rr else 0.0
    sip_ex = sip_future_value(5000, 12.0, 10.0)
    risk_mgmt = build_risk_management_pack(
        price=price,
        hv20_pct=hv20,
        atr_val=atr_val,
        stop=stop,
        reward_risk=rr,
        equity=500_000.0,
        risk_pct=1.0,
        beta60=beta60,
    )
    mom20 = momentum_roc_pct(closes, 20)
    mom60 = momentum_roc_pct(closes, 60)

    # Recompute qty from ATR stop (risk budget / |entry−stop|).
    risk_ps = abs(price - stop) if stop else None
    qty_struct = int(math.floor(risk_rupees / risk_ps)) if risk_ps and risk_ps > 0 else 0

    data_quality = {
        "ohlcv_source": frame.get("source"),
        "bars": len(closes),
        "nifty_bars": len(nifty),
        "delivery_ok": delivery.get("delivery_pct") is not None,
        "delivery_error": delivery.get("error"),
        "beta60_ok": beta60 is not None,
        "beta120_ok": beta120 is not None,
        "degraded": False,
    }
    if not data_quality["delivery_ok"] or beta120 is None:
        data_quality["degraded"] = True
    if frame.get("source") in {None, "unknown", "calculator"} and len(closes) < 80:
        data_quality["degraded"] = True

    pack.update(
        {
            "ok": True,
            "price": round(price, 2),
            "wilder_rsi_14": round(rsi, 2) if rsi is not None else None,
            "rsi_divergence": div,
            "bollinger": {
                "pct_b": round(bb["pct_b"], 3) if bb.get("pct_b") is not None else None,
                "bandwidth": round(bb["bandwidth"], 4) if bb.get("bandwidth") is not None else None,
                "bandwidth_regime": bw_label,
                "upper": round(bb["upper"], 2) if bb.get("upper") is not None else None,
                "middle": round(bb["middle"], 2) if bb.get("middle") is not None else None,
                "lower": round(bb["lower"], 2) if bb.get("lower") is not None else None,
            },
            "atr_14": round(atr_val, 2) if atr_val is not None else None,
            "atr_stop": {
                "stop": round(stop, 2) if stop is not None else None,
                "target_1r": round(target_1r, 2) if target_1r is not None else None,
                "qty_for_risk": qty_struct or int(sizing.get("qty") or 0),
                "risk_rupees": risk_rupees,
                "atr_mult": 1.5,
                # Structural R:R (not ATR 1R echo). ATR 1R kept only for sizing helper.
                "risk_reward": round(rr, 2) if rr is not None else None,
                "structural_t1": round(struct["target_1"], 2) if struct.get("target_1") else None,
                "structural_t2": round(struct["target_2"], 2) if struct.get("target_2") else None,
                "expectancy_r_at_45pct_wr": round(exp, 3) if exp is not None else None,
            },
            "pivots_classic": {k: round(v, 2) for k, v in pivots.items()},
            "pivots_camarilla": {k: round(v, 2) for k, v in cam.items()},
            "cpr": {
                "P": round(cpr["P"], 2),
                "TC": round(cpr["TC"], 2),
                "BC": round(cpr["BC"], 2),
                "width": round(cpr["width"], 2),
                "width_pct": round(cpr["width_pct"], 3),
                "regime": cpr["regime"],
            },
            "volume": {
                "zscore_20": round(vol["z"], 2) if vol.get("z") is not None else None,
                "ratio_vs_20d_avg": round(vol["ratio_vs_avg"], 2) if vol.get("ratio_vs_avg") is not None else None,
                "delivery_pct": delivery.get("delivery_pct"),
                "delivery_source": delivery.get("source"),
                "note": (
                    f"Delivery {delivery.get('delivery_pct')}% (NSE trade_info)"
                    if delivery.get("delivery_pct") is not None
                    else "Delivery% unavailable (NSE blocked) — using volume z-score only"
                ),
            },
            "vs_nifty": {
                "rs_20d": round(rs20, 3) if rs20 is not None else None,
                "rs_60d": round(rs60, 3) if rs60 is not None else None,
                "beta_60d": round(beta60, 3) if beta60 is not None else None,
                "beta_120d": round(beta120, 3) if beta120 is not None else None,
                "yf_beta": yf_fund.get("beta"),
                "beta_fallback": (
                    round(yf_fund["beta"], 3)
                    if beta60 is None and yf_fund.get("beta") is not None
                    else None
                ),
            },
            "valuation_math": {
                "earnings_yield_pct": round(ey, 2) if ey is not None else None,
                "pe": round(pe, 2) if pe is not None else None,
                "pe_sector": round(pe_sector, 2) if pe_sector is not None else None,
                "sector_pe_premium_pct": round(sector_pe_premium * 100.0, 1)
                if sector_pe_premium is not None
                else None,
                "eps": round(eps, 2) if eps is not None else None,
                "peg": round(peg, 2) if peg is not None else None,
                "growth_pct_used": round(growth, 2) if growth is not None else None,
                "justified_pe_rough": round(jpe, 2) if jpe is not None else None,
                "roe_pct": round(yf_fund["roe"], 2) if yf_fund.get("roe") is not None else None,
                "ev_ebitda": round(ev_pack["ev_ebitda"], 2) if ev_pack.get("ev_ebitda") is not None else None,
                "net_debt_ebitda": round(ev_pack["net_debt_ebitda"], 2)
                if ev_pack.get("net_debt_ebitda") is not None
                else None,
            },
            "macd": {
                "macd": round(macd_pack["macd"], 2) if macd_pack.get("macd") is not None else None,
                "signal": round(macd_pack["signal"], 2) if macd_pack.get("signal") is not None else None,
                "histogram": round(macd_pack["histogram"], 2) if macd_pack.get("histogram") is not None else None,
            },
            "stochastic": {
                "k": round(stoch["k"], 2) if stoch.get("k") is not None else None,
                "d": round(stoch["d"], 2) if stoch.get("d") is not None else None,
            },
            "adx_approx": round(adx_val, 2) if adx_val is not None else None,
            "supertrend": st_pack,
            "fibonacci": fib_pack,
            "vwap_20d": round(vwap, 2) if vwap is not None else None,
            "vwap_note": "20d volume-weighted daily proxy (not session VWAP)",
            "risk_stats": rstats,
            "india_costs": costs,
            "tax_drag_example": tax,
            "circuit": circuit,
            "fo": {
                **fo,
                "hv20_pct": round(hv20, 2) if hv20 is not None else None,
                "hv60_pct": round(hv60, 2) if hv60 is not None else None,
                "iv_proxy_note": "HV used as IV proxy when live option IV unavailable",
                "atm_premium_approx_21d": atm_prem,
                "call_breakeven_atm": opt_call.get("breakeven"),
                "put_breakeven_atm": opt_put.get("breakeven"),
                "rf_carry": rf_carry,
                "days_to_expiry_assumed": days_to_expiry,
                "dividend_cash_assumed_30d": div_cash,
                "futures_fair_30d": fut_fair,
                "basis_fair_30d": round(basis_fair, 2) if basis_fair is not None else None,
                "basis_regime": basis_regime,
                "leverage": lev_pack.get("leverage"),
                "margin_wipeout_pct": lev_pack.get("wipeout_pct"),
                "mtm_pnl_long_lot_plus_1pct": mtm_up,
                "mtm_pnl_long_lot_minus_1pct": mtm_dn,
                "hedge_example_1L": hedge_ex,
                "pricing_formula": "F ≈ S*(1+Rf*T/365) − D",
                "option_strategies": opt_strategies,
                "option_theory": opt_theory,
            },
            "data_quality": data_quality,
            "kelly_full": round(k_full, 3),
            "kelly_quarter": round(max(0.0, min(0.25, k_full / 4.0)), 3),
            "sip_example_5k_12pct_10y": round(sip_ex, 2) if sip_ex is not None else None,
            "risk_management": risk_mgmt,
            "momentum": {
                "roc_20d_pct": mom20,
                "roc_60d_pct": mom60,
                "note": "Rate-of-change momentum (Varsity-style); rank vs peers for portfolios.",
            },
            "formulas": [
                "Wilder RSI(14)+divergence",
                "Bollinger %B & bandwidth",
                "ATR stop/size + structural R:R",
                "Classic+Camarilla pivots + CPR",
                "Volume z-score(20) + NSE delivery%",
                "Date-aligned RS & beta vs Nifty",
                "EY; PEG; EV/EBITDA; sector PE premium",
                "MACD; Supertrend; Stoch; ADX",
                "Fibonacci; VWAP(20d proxy)",
                "Sortino/maxDD/Calmar",
                "Kelly (quarter-cap)",
                "Risk mgmt: VaR / %risk size / recovery / 2-asset vol",
                "Momentum ROC(20/60)",
                "India costs + tax drag example",
                "F&O lot/notional/HV-IV proxy/basis",
                "Futures CoC fair value + leverage/MTM/beta-hedge",
                "Option strategy payoff cards (spreads/straddle/IC)",
                "Option theory: BS Greeks/moneyness/IV-TV (HV σ)",
                "Circuit/52w distance (illustrative)",
            ],
        }
    )
    try:
        from .signal_engine import build_trade_plan

        pack["trade_plan"] = build_trade_plan(pack, horizon=horizon)
    except Exception:
        pack["trade_plan"] = None
    return pack


def build_analysis_pack(
    symbol: str,
    *,
    fund_hints: Optional[dict[str, Any]] = None,
    risk_rupees: float = 5000.0,
    horizon: str = "swing",
) -> dict[str, Any]:
    """Alias for full quantitative pack."""
    return build_p0_analysis_pack(
        symbol, fund_hints=fund_hints, risk_rupees=risk_rupees, horizon=horizon
    )


# ---------------------------------------------------------------------------
# Multi-factor sentiment analysis (news + price + volume + RS)
# ---------------------------------------------------------------------------

_SENT_POS = {
    "surge", "rally", "growth", "profit", "strong", "beat", "record", "gain",
    "rise", "bull", "upgrade", "positive", "wins", "boost", "outperform",
    "overweight", "robust", "better", "recovery", "rebound", "momentum",
    "expansion", "milestone", "all-time", "ath", "breakout", "buy", "bullish",
    "upbeat", "optimistic", "soar", "jumps", "jumped", "climbs", "climbed",
    "order", "win", "award", "partnership", "deal", "approval", "dividend",
    "buyback", "bonus", "split", "raise", "raised", "guidance",
    "exceed", "stronger", "improves", "improved", "highest", "inflow",
    "fii buying", "dii buying", "institutional buying",
    "block deal buy", "stake hike", "promoter buying", "margin expansion",
    "earnings beat", "profit jump", "revenue growth", "order win",
    "capacity expansion", "new high", "hits high", "all time high",
}
_SENT_NEG = {
    "fall", "decline", "loss", "weak", "miss", "cut", "drop", "bear",
    "downgrade", "risk", "concern", "warning", "crash", "slump", "worry",
    "underperform", "underweight", "poor", "disappointing", "slowdown",
    "contraction", "pressure", "debt", "fraud", "probe", "investigation",
    "penalty", "fine", "default", "pledge", "ban", "suspend",
    "layoff", "layoffs", "weakness", "bearish", "selloff", "sell-off",
    "plunge", "plunges", "tumbles", "tumbled", "slides", "slid",
    "outflow", "fii selling", "dii selling", "short", "scam", "litigation",
    "writedown", "impairment", "delay", "reject", "rejected", "fails",
    "earnings miss", "profit fall", "revenue miss", "guidance cut",
    "stake sale", "promoter pledging", "margin pressure", "volume dry",
    "hits low", "52-week low", "circuit limit", "lower circuit",
}


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence/clause split for headline analysis (no NLTK)."""
    raw = (text or "").strip()
    if not raw:
        return []
    # Prefer clause breaks common in news wires.
    parts = re.split(r"(?<=[.!?])\s+|;\s*|\s+\|\s+|\s+ - \s+|\s+—\s+", raw)
    out: list[str] = []
    for p in parts:
        s = p.strip(" -—|;")
        if len(s) >= 12:
            out.append(s)
    return out or [raw]


def _polarity_score(text: str) -> float:
    """Continuous polarity in [-1, 1] for a sentence/headline fragment."""
    t = (text or "").lower()
    if not t.strip():
        return 0.0
    pos = 0.0
    neg = 0.0
    # Multi-word phrases carry more weight.
    for p in _SENT_POS:
        if " " in p and p in t:
            pos += 1.6
    for n in _SENT_NEG:
        if " " in n and n in t:
            neg += 1.6
    words = set(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", t))
    pos += 1.0 * len(words & {w for w in _SENT_POS if " " not in w})
    neg += 1.0 * len(words & {w for w in _SENT_NEG if " " not in w})
    # Soft negation / contrast.
    if re.search(r"\b(not|fails? to|miss(?:es|ed)?|despite|although|however)\b.{0,24}\b(beat|growth|profit|rally|surge)\b", t):
        neg += 2.2
        pos = max(0.0, pos - 1.0)
    if re.search(r"\b(despite|although|however)\b.{0,24}\b(fall|loss|miss|cut|drop)\b", t):
        pos += 1.2
        neg = max(0.0, neg - 0.5)
    # Intensity boosters.
    if re.search(r"\b(sharp|steep|massive|huge|strong|weak)\b", t):
        if pos > neg:
            pos += 0.6
        elif neg > pos:
            neg += 0.6
    total = pos + neg
    if total <= 0:
        return 0.0
    return max(-1.0, min(1.0, (pos - neg) / total))


def _label_from_polarity_score(score: float) -> str:
    if score >= 0.18:
        return "positive"
    if score <= -0.18:
        return "negative"
    return "neutral"


def _headline_polarity(title: str) -> int:
    """Return +1 / 0 / -1 for a single headline (compat wrapper)."""
    score = _polarity_score(title)
    if score >= 0.18:
        return 1
    if score <= -0.18:
        return -1
    return 0


def analyze_headline_sentences(text: str) -> list[dict[str, Any]]:
    """Sentence-level polarity tags for a headline or short blurb."""
    rows: list[dict[str, Any]] = []
    for sent in _split_sentences(text)[:4]:
        score = _polarity_score(sent)
        rows.append(
            {
                "text": sent[:180],
                "score": round(score, 3),
                "polarity": _label_from_polarity_score(score),
            }
        )
    return rows


def score_news_headlines(headlines: list[Any]) -> dict[str, Any]:
    """Score headlines with recency weighting + sentence-level tags."""
    titles: list[str] = []
    for h in headlines or []:
        if isinstance(h, str) and h.strip():
            titles.append(h.strip())
        elif isinstance(h, dict):
            title = str(h.get("title") or h.get("headline") or "").strip()
            if title:
                titles.append(title)
    if not titles:
        return {
            "ok": False,
            "overall": "Unavailable",
            "score": 0.0,
            "breakdown": "no headlines",
            "positive": 0,
            "negative": 0,
            "neutral": 0,
            "headlines": [],
            "tagged": [],
            "sentences": [],
        }

    tagged: list[dict[str, Any]] = []
    sentences: list[dict[str, Any]] = []
    pos = neg = neu = 0
    weighted = 0.0
    weight_sum = 0.0
    # Newest-first lists (RSS) get higher weight on earlier items.
    for idx, title in enumerate(titles[:10]):
        score = _polarity_score(title)
        label = _label_from_polarity_score(score)
        if label == "positive":
            pos += 1
        elif label == "negative":
            neg += 1
        else:
            neu += 1
        recency_w = 1.0 / (1.0 + 0.18 * idx)
        weighted += score * recency_w
        weight_sum += recency_w
        sent_rows = analyze_headline_sentences(title)
        sentences.extend(sent_rows[:2])
        tagged.append(
            {
                "title": title[:180],
                "polarity": label,
                "score": round(score, 3),
                "sentences": sent_rows,
            }
        )

    total = pos + neg + neu or 1
    raw = weighted / weight_sum if weight_sum else 0.0
    # Blend count balance with continuous score.
    count_bias = (pos - neg) / total
    blended = 0.65 * raw + 0.35 * count_bias
    if blended >= 0.22:
        overall = "Positive"
    elif blended <= -0.22:
        overall = "Negative"
    else:
        overall = "Neutral"
    return {
        "ok": True,
        "overall": overall,
        "score": round(max(-1.0, min(1.0, blended)), 3),
        "breakdown": f"Positive {pos * 100 // total}% / Neutral {neu * 100 // total}% / Negative {neg * 100 // total}%",
        "positive": pos,
        "negative": neg,
        "neutral": neu,
        "headlines": titles[:5],
        "tagged": tagged[:6],
        "sentences": sentences[:8],
    }


def _label_from_score(score: float) -> str:
    if score >= 0.45:
        return "Bullish"
    if score >= 0.18:
        return "Mildly bullish"
    if score <= -0.45:
        return "Bearish"
    if score <= -0.18:
        return "Mildly bearish"
    return "Neutral"


def build_sentiment_analysis_pack(
    symbol: str | None = None,
    *,
    enrich: Optional[dict[str, Any]] = None,
    p0: Optional[dict[str, Any]] = None,
    headlines: Optional[list[Any]] = None,
) -> dict[str, Any]:
    """Multi-factor sentiment for a stock or broad market (Nifty proxy).

    Combines: news headlines, RSI/momentum, trend/Supertrend, MACD,
    volume confirmation, RS vs Nifty, and short-term ROC.
    Educational — not a buy/sell recommendation.
    """
    enrich = enrich or {}
    sym = (symbol or enrich.get("symbol") or "").upper().strip()
    if not sym:
        sym = "NIFTY50"
    market_mode = sym in {"NIFTY", "NIFTY50", "^NSEI", "NSEI", "MARKET", "SENSEX"}

    pack_p0 = p0 if isinstance(p0, dict) and p0.get("ok") else None
    if pack_p0 is None:
        try:
            pack_p0 = build_p0_analysis_pack("NIFTY50" if market_mode else sym)
        except Exception:
            pack_p0 = None
    if not pack_p0 or not pack_p0.get("ok"):
        # Still try news-only if we have headlines.
        news_only = score_news_headlines(
            headlines
            or enrich.get("news_headlines")
            or (enrich.get("sentiment") or {}).get("recent_events")
            or []
        )
        return {
            "ok": bool(news_only.get("ok")),
            "symbol": sym,
            "market_mode": market_mode,
            "composite_score": news_only.get("score", 0.0),
            "label": news_only.get("overall", "Unavailable"),
            "confidence": 0.35 if news_only.get("ok") else 0.0,
            "factors": [],
            "news": news_only,
            "summary": "Limited data — news-only or unavailable.",
            "note": "Educational multi-factor sentiment — not investment advice.",
        }

    # News factor
    news = score_news_headlines(
        headlines
        or enrich.get("news_headlines")
        or (enrich.get("sentiment") or {}).get("recent_events")
        or []
    )
    # If enrich already tagged overall, keep as soft prior when headlines empty.
    if not news.get("ok"):
        prior = str((enrich.get("sentiment") or {}).get("overall") or "").lower()
        if prior.startswith("pos"):
            news = {
                "ok": True,
                "overall": "Positive",
                "score": 0.35,
                "breakdown": "enrich prior",
                "positive": 1,
                "negative": 0,
                "neutral": 0,
                "headlines": [],
                "tagged": [],
            }
        elif prior.startswith("neg"):
            news = {
                "ok": True,
                "overall": "Negative",
                "score": -0.35,
                "breakdown": "enrich prior",
                "positive": 0,
                "negative": 1,
                "neutral": 0,
                "headlines": [],
                "tagged": [],
            }

    factors: list[dict[str, Any]] = []
    weights: list[tuple[float, float]] = []  # (score, weight)

    # 1) News
    n_score = float(news.get("score") or 0.0) if news.get("ok") else 0.0
    n_w = 0.28 if news.get("ok") else 0.0
    if n_w:
        factors.append(
            {
                "name": "News headlines",
                "score": round(n_score, 3),
                "label": news.get("overall"),
                "detail": news.get("breakdown"),
            }
        )
        weights.append((n_score, n_w))

    # 2) RSI / momentum
    rsi = _safe_float(pack_p0.get("wilder_rsi_14"))
    if rsi is not None:
        if rsi >= 70:
            r_score = -0.55  # overbought crowding / pullback risk
            r_label = f"RSI {rsi:.0f} overbought (caution)"
        elif rsi <= 30:
            r_score = 0.45  # oversold bounce bias (not a tip)
            r_label = f"RSI {rsi:.0f} oversold (relief bias)"
        elif rsi >= 55:
            r_score = 0.35
            r_label = f"RSI {rsi:.0f} bullish momentum"
        elif rsi <= 45:
            r_score = -0.35
            r_label = f"RSI {rsi:.0f} soft momentum"
        else:
            r_score = 0.0
            r_label = f"RSI {rsi:.0f} neutral"
        div = str(pack_p0.get("rsi_divergence") or "")
        if "bearish" in div:
            r_score -= 0.15
            r_label += "; bearish divergence cue"
        elif "bullish" in div:
            r_score += 0.15
            r_label += "; bullish divergence cue"
        r_score = max(-1.0, min(1.0, r_score))
        factors.append({"name": "Momentum (RSI)", "score": round(r_score, 3), "label": r_label})
        weights.append((r_score, 0.18))

    # 3) Trend / Supertrend
    st = pack_p0.get("supertrend") or {}
    direction = str(st.get("direction") or "").lower()
    if "bull" in direction or direction == "up":
        t_score, t_label = 0.55, "Supertrend bullish"
    elif "bear" in direction or direction == "down":
        t_score, t_label = -0.55, "Supertrend bearish"
    else:
        t_score, t_label = 0.0, "Trend mixed / n/a"
    # Soft blend with enrich MA trend if present
    ma = str((enrich.get("technical") or {}).get("trend") or "").lower()
    if "strong_bull" in ma or "bull" in ma:
        t_score = max(t_score, 0.4) if t_score >= 0 else (t_score + 0.25) / 2
        t_label += "; MA bias up"
    elif "bear" in ma:
        t_score = min(t_score, -0.4) if t_score <= 0 else (t_score - 0.25) / 2
        t_label += "; MA bias down"
    factors.append({"name": "Trend", "score": round(t_score, 3), "label": t_label})
    weights.append((t_score, 0.18))

    # 4) MACD histogram
    macd_p = pack_p0.get("macd") or {}
    hist = _safe_float(macd_p.get("histogram"))
    if hist is not None:
        if hist > 0:
            m_score = min(1.0, 0.25 + abs(hist) / max(abs(pack_p0.get("price") or 1), 1) * 50)
            m_label = f"MACD hist +{hist:.2f} (bullish)"
        elif hist < 0:
            m_score = -min(1.0, 0.25 + abs(hist) / max(abs(pack_p0.get("price") or 1), 1) * 50)
            m_label = f"MACD hist {hist:.2f} (bearish)"
        else:
            m_score, m_label = 0.0, "MACD flat"
        factors.append({"name": "MACD", "score": round(m_score, 3), "label": m_label})
        weights.append((m_score, 0.12))

    # 5) Volume confirmation
    vol = pack_p0.get("volume") or {}
    z = _safe_float(vol.get("zscore_20"))
    ratio = _safe_float(vol.get("ratio_vs_20d_avg"))
    mom = pack_p0.get("momentum") or {}
    roc20 = _safe_float(mom.get("roc_20d_pct"))
    v_score = 0.0
    v_label = "Volume average"
    if z is not None:
        # Only elevated volume confirms; quiet/thin tape ≈ neutral (not a bullish cue).
        sign = 1.0 if (roc20 or 0) >= 0 else -1.0
        if z >= 1.0:
            v_score = max(-1.0, min(1.0, sign * min(1.0, z / 2.5)))
            v_label = f"Vol z={z:.1f} confirms move"
        elif z <= -1.0:
            v_score = 0.0
            v_label = f"Vol z={z:.1f} quiet / thin"
        else:
            v_label = f"Vol z={z:.1f} average"
    elif ratio is not None and ratio >= 1.4:
        sign = 1.0 if (roc20 or 0) >= 0 else -1.0
        v_score = 0.35 * sign
        v_label = f"Vol {ratio:.1f}× 20d avg"
    delivery = _safe_float(vol.get("delivery_pct"))
    if delivery is not None and delivery >= 55:
        v_score = max(-1.0, min(1.0, v_score + 0.1))
        v_label += f"; delivery {delivery:.0f}%"
    factors.append({"name": "Volume", "score": round(v_score, 3), "label": v_label})
    weights.append((v_score, 0.10))

    # 6) Relative strength vs Nifty (ratio: stock_ret / index_ret; 1.0 = in-line)
    vs = pack_p0.get("vs_nifty") or {}
    rs20 = _safe_float(vs.get("rs_20d"))
    if rs20 is not None and not market_mode:
        rs_edge = rs20 - 1.0
        if rs_edge > 0.03:
            rs_score, rs_label = 0.5, f"RS20={rs20:.3f} ({rs_edge:+.1%} vs Nifty) outperforming"
        elif rs_edge < -0.03:
            rs_score, rs_label = -0.5, f"RS20={rs20:.3f} ({rs_edge:+.1%} vs Nifty) underperforming"
        else:
            rs_score, rs_label = 0.0, f"RS20={rs20:.3f} ({rs_edge:+.1%} vs Nifty) in-line"
        factors.append({"name": "vs Nifty", "score": rs_score, "label": rs_label})
        weights.append((rs_score, 0.14))

    # 7) Short-term ROC
    if roc20 is not None:
        if roc20 >= 8:
            roc_score, roc_label = 0.55, f"20d ROC +{roc20:.1f}% strong"
        elif roc20 >= 2:
            roc_score, roc_label = 0.3, f"20d ROC +{roc20:.1f}%"
        elif roc20 <= -8:
            roc_score, roc_label = -0.55, f"20d ROC {roc20:.1f}% weak"
        elif roc20 <= -2:
            roc_score, roc_label = -0.3, f"20d ROC {roc20:.1f}%"
        else:
            roc_score, roc_label = 0.0, f"20d ROC {roc20:.1f}% flat"
        factors.append({"name": "Price ROC", "score": roc_score, "label": roc_label})
        weights.append((roc_score, 0.10))

    # Sector cue from enrich (soft)
    sector_trend = str((enrich.get("sentiment") or {}).get("sector_trend") or "")
    if sector_trend:
        st_l = sector_trend.lower()
        if "bull" in st_l or "outperform" in st_l or "strong" in st_l:
            s_score = 0.25
        elif "bear" in st_l or "weak" in st_l or "under" in st_l:
            s_score = -0.25
        else:
            s_score = 0.0
        factors.append(
            {"name": "Sector cue", "score": s_score, "label": sector_trend[:120]}
        )
        weights.append((s_score, 0.08))

    tw = sum(w for _, w in weights) or 1.0
    composite = sum(s * w for s, w in weights) / tw
    composite = max(-1.0, min(1.0, composite))
    label = _label_from_score(composite)

    # Confidence: more factors + news availability
    conf = min(0.92, 0.4 + 0.08 * len(factors) + (0.12 if news.get("ok") else 0.0))
    agreeing = sum(1 for f in factors if (f.get("score") or 0) * composite > 0)
    if agreeing >= max(3, len(factors) // 2 + 1):
        conf = min(0.94, conf + 0.06)

    bull_bits = [f["label"] for f in factors if (f.get("score") or 0) > 0.15][:3]
    bear_bits = [f["label"] for f in factors if (f.get("score") or 0) < -0.15][:3]
    summary_bits = []
    if bull_bits:
        summary_bits.append("Supports: " + "; ".join(bull_bits))
    if bear_bits:
        summary_bits.append("Drags: " + "; ".join(bear_bits))
    if not summary_bits:
        summary_bits.append("Factors mostly balanced — wait for clearer confirmation.")

    return {
        "ok": True,
        "symbol": sym,
        "market_mode": market_mode,
        "price": pack_p0.get("price"),
        "composite_score": round(composite, 3),
        "label": label,
        "confidence": round(conf, 2),
        "factors": factors,
        "news": news,
        "summary": " | ".join(summary_bits),
        "note": (
            "Multi-factor educational sentiment (news + technicals + volume + RS). "
            "Not a prediction or investment advice."
        ),
    }


def format_sentiment_card(pack: dict[str, Any]) -> str:
    """Render a structured sentiment analysis card for the LLM answer."""
    if not pack or not pack.get("ok"):
        return (
            "**Sentiment analysis**\n\n"
            "Insufficient live data to score sentiment right now.\n"
            "_Educational — not investment advice._"
        )
    sym = pack.get("symbol") or "MARKET"
    title = (
        f"**{sym} — market sentiment snapshot**"
        if pack.get("market_mode")
        else f"**{sym} — sentiment analysis**"
    )
    lines = [
        title,
        "",
        f"**Overall:** {pack.get('label')} "
        f"(score {pack.get('composite_score'):+.2f}, confidence {pack.get('confidence')})",
        f"**Read:** {pack.get('summary')}",
        "",
        "**Factor stack:**",
    ]
    for f in pack.get("factors") or []:
        sc = f.get("score")
        sc_txt = f"{sc:+.2f}" if isinstance(sc, (int, float)) else "n/a"
        lines.append(f"• {f.get('name')}: {f.get('label')} [{sc_txt}]")
    news = pack.get("news") or {}
    if news.get("ok"):
        lines.append("")
        lines.append(
            f"**News tone:** {news.get('overall')} — {news.get('breakdown')}"
        )
        for t in (news.get("tagged") or [])[:4]:
            if isinstance(t, dict) and t.get("title"):
                sc = t.get("score")
                sc_txt = f"{sc:+.2f}" if isinstance(sc, (int, float)) else ""
                pol = t.get("polarity") or "neutral"
                suffix = f" {sc_txt}" if sc_txt else ""
                lines.append(f"  – ({pol}{suffix}) {t.get('title')[:120]}")
                for sent in (t.get("sentences") or [])[:2]:
                    if not isinstance(sent, dict) or not sent.get("text"):
                        continue
                    if sent.get("text") == t.get("title"):
                        continue
                    s_pol = sent.get("polarity") or "neutral"
                    lines.append(f"      · [{s_pol}] {str(sent.get('text'))[:100]}")
        # Standalone sentence cues when tagged is thin.
        if not news.get("tagged"):
            for sent in (news.get("sentences") or [])[:4]:
                if isinstance(sent, dict) and sent.get("text"):
                    lines.append(
                        f"  – ({sent.get('polarity')}) {str(sent.get('text'))[:120]}"
                    )
    lines.extend(
        [
            "",
            "Use sentiment with price structure, levels, and risk — never alone.",
            f"_{pack.get('note') or 'Educational — not investment advice.'}_",
        ]
    )
    return "\n".join(lines)


def format_p0_for_prompt(pack: dict[str, Any]) -> str:
    """Compact bullet block for composer / KB injection."""
    if not pack or not pack.get("ok"):
        return ""
    atr_s = pack.get("atr_stop") or {}
    bb = pack.get("bollinger") or {}
    vol = pack.get("volume") or {}
    vs = pack.get("vs_nifty") or {}
    val = pack.get("valuation_math") or {}
    piv = pack.get("pivots_classic") or {}
    st = pack.get("supertrend") or {}
    macd_p = pack.get("macd") or {}
    fib = pack.get("fibonacci") or {}
    rs = pack.get("risk_stats") or {}
    plan = pack.get("trade_plan") or {}
    fo = pack.get("fo") or {}
    dq = pack.get("data_quality") or {}
    lines = [
        f"Full math for {pack.get('symbol')}:",
        f"Wilder RSI(14)={pack.get('wilder_rsi_14')} | divergence={pack.get('rsi_divergence')}",
        f"Bollinger %B={bb.get('pct_b')} | bandwidth={bb.get('bandwidth')} ({bb.get('bandwidth_regime')})",
        f"ATR(14)={pack.get('atr_14')} | stop={atr_s.get('stop')} | structT1={atr_s.get('structural_t1')} | "
        f"qty={atr_s.get('qty_for_risk')} | structural R:R={atr_s.get('risk_reward')}",
        f"MACD hist={macd_p.get('histogram')} | Supertrend={st.get('direction')} @{st.get('line')} | "
        f"ADX≈{pack.get('adx_approx')} | StochK={(pack.get('stochastic') or {}).get('k')}",
        f"Fib 0.618={fib.get('retracement_618')} | ext1.618={fib.get('extension_1618')} | VWAP20={pack.get('vwap_20d')}",
        f"Pivots P={piv.get('P')} S1={piv.get('S1')} R1={piv.get('R1')}",
        (
            f"CPR P={(pack.get('cpr') or {}).get('P')} TC={(pack.get('cpr') or {}).get('TC')} "
            f"BC={(pack.get('cpr') or {}).get('BC')} ({(pack.get('cpr') or {}).get('regime')})"
        ),
        f"Vol z20={vol.get('zscore_20')} | delivery%={vol.get('delivery_pct')} | "
        f"RS20={vs.get('rs_20d')} β60={vs.get('beta_60d')} β120={vs.get('beta_120d')}",
        f"EY%={val.get('earnings_yield_pct')} PEG={val.get('peg')} EV/EBITDA={val.get('ev_ebitda')} "
        f"sectorPE_prem%={val.get('sector_pe_premium_pct')}",
        f"Sortino={rs.get('sortino_60d')} maxDD%={rs.get('max_drawdown_60d_pct')} Calmar={rs.get('calmar_60d')}",
        f"F&O lot={fo.get('lot_size')} notional={fo.get('notional_per_lot')} margin≈{fo.get('indicative_margin_per_lot')} "
        f"lev≈{fo.get('leverage')}x wipe≈{fo.get('margin_wipeout_pct')}% "
        f"HV20%={fo.get('hv20_pct')} fair30d={fo.get('futures_fair_30d')} "
        f"basis={fo.get('basis_fair_30d')} ({fo.get('basis_regime')}) "
        f"hedge1L_lots≈{(fo.get('hedge_example_1L') or {}).get('lots_exact')} "
        f"ATMΔcall≈{((fo.get('option_theory') or {}).get('call') or {}).get('delta')} "
        f"θ/day≈{((fo.get('option_theory') or {}).get('call') or {}).get('theta_per_day')}",
        f"data_degraded={dq.get('degraded')} ohlcv={dq.get('ohlcv_source')}",
        f"TRADE PLAN action={plan.get('action')} score={plan.get('score')} conf={plan.get('confidence')} "
        f"entry={plan.get('entry_zone')} stop={plan.get('stop')} T1={plan.get('target_1')} T2={plan.get('target_2')} "
        f"R:R={plan.get('risk_reward')}",
        (
            f"RISK VaR1d95≈{(pack.get('risk_management') or {}).get('var_1d_95_rs')} "
            f"qty%risk={(pack.get('risk_management') or {}).get('percent_risk_size', {}).get('qty')} "
            f"KellyQ={(pack.get('risk_management') or {}).get('kelly_quarter')}"
        ),
    ]
    return " | ".join(lines)
