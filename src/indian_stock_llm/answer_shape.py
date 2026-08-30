"""Query-faithful card headings.

ISM already routes a profile. This layer reads the user's wording again
and picks a distinct heading so two asks in the same profile do not
share one generic dump.
"""
from __future__ import annotations

import re


def detect_answer_shape(query: str, profile: str) -> tuple[str, str]:
    """Return (facet, heading fragment after the name)."""
    q = re.sub(r"\s+", " ", (query or "").strip().lower())
    profile = (profile or "").strip()
    handler = {
        "quote": _quote,
        "news": _news,
        "sentiment": _sentiment,
        "technical": _technical,
        "trade_plan": _trade_plan,
        "prediction": _prediction,
        "fundamentals": _fundamentals,
        "risks": _risks,
        "portfolio": _portfolio,
        "derivatives": _derivatives,
        "calculations": _calculations,
        "corporate_actions": _corporate,
        "session": lambda _q: ("session_clock", "session clock"),
        "compare": lambda _q: ("pair", "comparison scorecard"),
        "compare_concepts": lambda _q: ("concepts", "concept compare"),
        "sector_screen": lambda _q: ("sector", "sector screen"),
        "literacy": _literacy,
        "stock_analysis": _analysis,
        "small_talk": lambda _q: ("greeting", "hello"),
    }.get(profile)
    if handler is None:
        return "generic", "snapshot"
    return handler(q)


def _quote(q: str) -> tuple[str, str]:
    if re.search(r"\bcmp\b", q):
        return "cmp", "live quote · CMP"
    if re.search(r"\b(kitna|ka (price|rate)|how much)\b", q):
        return "how_much", "live quote · how much now"
    return "ltp", "live quote"


def _news(q: str) -> tuple[str, str]:
    if re.search(r"\b(why (is|did|has)|what happened)\b", q):
        return "why_move", "news · why it moved"
    if re.search(r"\b(q[1-4]|quarterly|results?|earnings|guidance)\b", q):
        return "results", "news · results tape"
    if re.search(r"\b(fii|dii)\b", q):
        return "flows", "news · FII / DII flow"
    if re.search(r"\b(upper circuit|lower circuit|circuit)\b", q):
        return "circuit", "news · circuit / tape event"
    return "headlines", "news & catalysts"


def _sentiment(q: str) -> tuple[str, str]:
    if re.search(r"\b(market|nifty|sensex|traders|investors)\b", q) and not re.search(
        r"\b(mood on|bullish on|bearish on)\b", q
    ):
        return "market", "market mood"
    return "stock", "sentiment view"


def _technical(q: str) -> tuple[str, str]:
    m = re.search(r"\b(\d{1,3})\s*ema\b", q)
    if m:
        return "ema", f"technical · {m.group(1)} EMA"
    m = re.search(r"\b(\d{1,3})\s*dma\b", q)
    if m:
        return "dma", f"technical · {m.group(1)} DMA"
    if re.search(r"\brsi\b", q):
        return "rsi", "technical · RSI"
    if re.search(r"\bmacd\b", q):
        return "macd", "technical · MACD"
    if re.search(r"\b(volume spike|volume)\b", q):
        return "volume", "technical · volume / tape"
    if re.search(r"\b(support|resistance|levels?)\b", q):
        return "levels", "technical · support & resistance"
    if re.search(r"\bchart\b", q):
        return "chart", "technical · chart"
    if re.search(r"\bbreakout\b", q):
        return "breakout", "technical · breakout"
    return "tape", "technical analysis"


def _trade_plan(q: str) -> tuple[str, str]:
    if re.search(r"\b(book profit|book profits|trim)\b", q):
        return "book", "paper book-profit plan"
    if re.search(r"\b(hold or exit|should i hold|exit now|time to sell|sell now)\b|\bbech", q):
        return "exit", "paper exit / hold plan"
    if re.search(r"\b(dip|accumulate|add more|average down|top up)\b", q):
        return "dip", "paper dip-add plan"
    if re.search(r"\b(should i sell|can i sell|short )\b", q):
        return "sell", "paper sell plan"
    if re.search(r"\b(should i buy|can i buy|buy now|good buy|worth buying)\b|\bkharid", q):
        return "buy", "paper buy plan"
    return "plan", "paper trade plan"


def _prediction(q: str) -> tuple[str, str]:
    if re.search(r"\b(nifty|sensex|banknifty|bank nifty)\b", q) and not re.search(
        r"\bpe\b|\bp/e\b", q
    ):
        return "index", "index outlook"
    if re.search(r"\bwill .{0,20} reach\b|\breach\b", q):
        return "reach", "will-it-reach band"
    return "target", "scenario range"


def _fundamentals(q: str) -> tuple[str, str]:
    if re.search(r"\b(nifty|sensex)\b", q) and re.search(r"\b(p/?e|pe ratio|valuation)\b", q):
        return "index_pe", "index valuation"
    if re.search(r"\b(cheap|expensive|costly|overvalued|undervalued)\b", q):
        return "cheap_rich", "fundamental · cheap or rich"
    if re.search(r"\b(market cap|mcap)\b", q):
        return "mcap", "fundamental · market cap"
    if re.search(r"\b(dividend yield|yield)\b", q):
        return "yield", "fundamental · dividend yield"
    if re.search(r"\b(promoter pledge|pledge|promoter holding)\b", q):
        return "pledge", "fundamental · pledge / ownership"
    if re.search(r"\b(p/?e|pe ratio)\b", q):
        return "pe", "fundamental · P/E readout"
    if re.search(r"\b(book value|p/b|pb)\b", q):
        return "pb", "fundamental · P/B readout"
    return "snapshot", "fundamental snapshot"


def _risks(q: str) -> tuple[str, str]:
    if re.search(r"\bdownside\b", q):
        return "downside", "key risks · downside"
    return "risks", "key risks now"


def _portfolio(q: str) -> tuple[str, str]:
    if re.search(r"\b(watchlist|on my list|on my watch)\b", q):
        return "watchlist", "on your list"
    if re.search(r"\b(position size|qty|size for)\b", q):
        return "sizing", "position sizing"
    return "book", "paper book"


def _derivatives(q: str) -> tuple[str, str]:
    if re.search(r"\blot size\b", q):
        return "lot", "F&O · lot size"
    if re.search(r"\bexpir(y|ies)\b", q):
        return "expiry", "F&O · expiry"
    if re.search(r"\bcovered call\b", q):
        return "covered_call", "F&O · covered call"
    if re.search(r"\bstraddle\b", q):
        return "straddle", "F&O · straddle"
    return "fo", "F&O paper-practice snapshot"


def _calculations(q: str) -> tuple[str, str]:
    if re.search(r"\bcagr\b", q):
        return "cagr", "CAGR working"
    if re.search(r"\breturn\b", q):
        return "return", "return working"
    return "math", "live checks"


def _corporate(q: str) -> tuple[str, str]:
    if re.search(r"\bdividend\b", q):
        return "dividend", "corporate actions · dividend"
    if re.search(r"\bbonus\b", q):
        return "bonus", "corporate actions · bonus"
    if re.search(r"\bsplit\b", q):
        return "split", "corporate actions · split"
    return "events", "corporate actions"


def _literacy(q: str) -> tuple[str, str]:
    if re.search(r"\bipo\b", q):
        return "ipo", "literacy · IPO"
    if re.search(r"\brsi\b", q):
        return "rsi", "literacy · RSI"
    if re.search(r"\bstraddle\b", q):
        return "straddle", "literacy · straddle"
    if re.search(r"\b(delivery|intraday|cnc|mis)\b", q):
        return "product", "literacy · delivery vs intraday"
    return "primer", "literacy"


def _analysis(q: str) -> tuple[str, str]:
    if re.search(r"\b(kya haal|kaise hai)\b", q):
        return "haal", "snapshot · kya haal"
    return "status", "snapshot"
