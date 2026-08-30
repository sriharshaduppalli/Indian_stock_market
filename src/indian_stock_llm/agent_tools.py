"""Indian-market agent tools for ISM.

Session clock, dated corporate actions, and a parseable paper-order line.
Never invent an ex-date, P&L, or a live brokerage ticket.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


def session_snapshot(now: datetime | None = None) -> dict[str, Any]:
    """Live NSE/BSE session phase in IST, with holiday awareness when available."""
    now_ist = now.astimezone(IST) if now is not None else datetime.now(IST)
    pack: dict[str, Any] = {
        "nowIst": now_ist.strftime("%Y-%m-%d %H:%M IST"),
        "weekday": now_ist.strftime("%A"),
        "isOpen": False,
        "phase": "closed",
        "message": "Market closed",
        "preOpen": "09:00-09:15 IST",
        "cashOpen": "09:15 IST",
        "uiOpenUntil": "15:40 IST",
    }
    try:
        from app.market_session import (
            closed_after_hours_message,
            is_within_equity_session,
            literacy_blurb,
            open_message,
            session_timeline,
        )

        holidays: set[str] = set()
        try:
            from app.routes.trading import NSE_HOLIDAYS_2026

            holidays = set(NSE_HOLIDAYS_2026)
        except Exception:
            holidays = set()
        pack.update({k: v for k, v in session_timeline(now_ist.date()).items() if v})
        pack["literacy"] = literacy_blurb()
        today = now_ist.strftime("%Y-%m-%d")
        if today in holidays:
            pack["phase"] = "holiday"
            pack["isOpen"] = False
            pack["message"] = "Market closed — NSE holiday"
        elif now_ist.weekday() >= 5:
            pack["phase"] = "weekend"
            pack["isOpen"] = False
            pack["message"] = "Market closed — weekend"
        elif is_within_equity_session(now_ist):
            pack["phase"] = "open"
            pack["isOpen"] = True
            pack["message"] = open_message(now_ist)
        else:
            pack["phase"] = "closed"
            pack["isOpen"] = False
            pack["message"] = closed_after_hours_message(now_ist)
    except Exception:
        t = now_ist.time()
        weekday_open = now_ist.weekday() < 5 and t.hour * 60 + t.minute >= 9 * 60 + 15
        weekday_open = weekday_open and t.hour * 60 + t.minute <= 15 * 60 + 40
        pack["isOpen"] = weekday_open
        pack["phase"] = "open" if weekday_open else ("weekend" if now_ist.weekday() >= 5 else "closed")
        pack["message"] = "Market is OPEN" if weekday_open else "Market closed"
    return pack


def format_session_card(query: str, snap: dict[str, Any] | None = None) -> str:
    snap = snap or session_snapshot()
    ask = (query or "").strip()[:90]
    parts = [
        "**NSE / BSE session (IST)**",
        f"**Your ask:** {ask}" if ask else "",
        "",
        f"• Now: {snap.get('nowIst') or 'n/a'} ({snap.get('weekday') or ''})",
        f"• Status: **{'OPEN' if snap.get('isOpen') else 'CLOSED'}** — {snap.get('message')}",
        f"• Cash open: {snap.get('cashOpen') or '09:15 IST'}",
        f"• Paper UI treats the tape live until: {snap.get('uiOpenUntil') or '15:40 IST'}",
    ]
    if snap.get("nextOpen"):
        parts.append(f"• Next open idea: {snap.get('nextOpen')}")
    if snap.get("closingAuctionEnd"):
        parts.append(f"• Closing auction (F&O cash): till {snap.get('closingAuctionEnd')}")
    parts.extend(
        [
            "",
            "Weekend and holiday sessions stay closed for paper fills at last.",
            "_Session clock — educational. Confirm holidays on the exchange calendar._",
        ]
    )
    return "\n".join(line for line in parts if line is not None)


def _corporate_action_paths() -> list[Path]:
    here = Path(__file__).resolve()
    return [
        here.parents[2] / "data" / "enterprise" / "corporate_actions.json",
        here.parents[4] / "app" / ".." / "llm_data" / "enterprise" / "corporate_actions.json",
        Path("backend/llm_data/enterprise/corporate_actions.json"),
    ]


def load_corporate_actions(
    symbol: str | None = None,
    *,
    hydrate: bool = False,
) -> list[dict[str, Any]]:
    """Dated lines only. Empty means we do not know — never invent.

    ``hydrate=True`` merges Yahoo actions/calendar dates when BYSEL is present.
    Never scrapes nseindia.com. Call hydrate only on the corporate_actions profile.
    """
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in _corporate_action_paths():
        try:
            raw = json.loads(Path(path).resolve().read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(raw, list):
            continue
        for item in raw:
            if not isinstance(item, dict):
                continue
            key = f"{item.get('symbol')}|{item.get('action')}|{item.get('effective_date')}"
            if key in seen:
                continue
            seen.add(key)
            rows.append(item)
    want = (symbol or "").strip().upper()
    if want:
        rows = [r for r in rows if str(r.get("symbol") or "").upper() == want]
    if hydrate and want:
        try:
            from app.corp_actions import merge_yahoo_dated_actions

            rows = merge_yahoo_dated_actions(want, rows)
        except Exception:
            pass
    return rows


def format_corporate_actions_card(
    query: str,
    symbol: str,
    *,
    rows: list[dict[str, Any]] | None = None,
) -> str:
    symbol = (symbol or "").upper().strip()
    dated = rows if rows is not None else load_corporate_actions(symbol)
    ask = (query or "").strip()[:90]
    parts = [
        f"**{symbol or 'Stock'} — corporate actions**",
        f"**Your ask:** {ask}" if ask else "",
        "",
    ]
    if dated:
        parts.append("Dated lines we store (pack + last Yahoo actions — not an NSE scrape):")
        for row in dated[:6]:
            action = str(row.get("action") or "Event")
            when = str(row.get("effective_date") or row.get("ex_date") or "")
            if not when:
                continue
            extra = ""
            if row.get("amount") not in (None, "", 0, 0.0):
                extra = f" (₹{row.get('amount')})"
            elif row.get("ratio") not in (None, "", 0, 0.0):
                extra = f" (ratio {row.get('ratio')})"
            parts.append(f"• {action}: {when}{extra}")
    else:
        parts.extend(
            [
                f"No dated corporate-action line is stored for **{symbol or 'this name'}**.",
                "We do not invent an ex-date, record date, or bonus ratio.",
                "Confirm on NSE Get Quote → corporate actions before any paper entitlement check.",
            ]
        )
    parts.extend(
        [
            "",
            "Holders generally need the stock before the ex-date for entitlement.",
            "_Educational calendar — not a corporate announcement._",
        ]
    )
    return "\n".join(line for line in parts if line is not None)


def paper_order_from_plan(
    symbol: str,
    plan: dict[str, Any] | None,
    *,
    stop: Any = None,
) -> dict[str, Any] | None:
    """Educational paper ticket the Android parser can turn into Practice buy/sell."""
    symbol = (symbol or "").upper().strip()
    plan = plan if isinstance(plan, dict) else {}
    action = str(plan.get("action") or "").upper().strip()
    if not symbol or action in {"", "HOLD", "WAIT", "NEUTRAL"}:
        stop_n = stop if stop is not None else plan.get("stop")
        if symbol and stop_n not in (None, "", "n/a"):
            return {
                "symbol": symbol,
                "side": None,
                "qty": None,
                "kind": "alert",
                "line": f"Set a price alert on {symbol} below {stop_n} (paper).",
            }
        return None
    qty = plan.get("position_qty_for_risk") or plan.get("qty") or 1
    try:
        qty = max(1, int(float(qty)))
    except (TypeError, ValueError):
        qty = 1
    if action in {"BUY", "ACCUMULATE", "STRONG BUY"}:
        return {
            "symbol": symbol,
            "side": "BUY",
            "qty": qty,
            "kind": "paper_buy",
            "line": f"Consider buying {qty} shares of {symbol} (paper practice only).",
        }
    if action in {"SELL", "TRIM", "STRONG SELL"}:
        return {
            "symbol": symbol,
            "side": "SELL",
            "qty": qty,
            "kind": "paper_sell",
            "line": f"Consider selling {qty} shares of {symbol} (paper practice only).",
        }
    return None
