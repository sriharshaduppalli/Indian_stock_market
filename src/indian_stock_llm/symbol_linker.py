"""Indian listed-name linker for ISM.

Maps spoken / typed names (Reliance, HDFC Bank, Relaince, SBI) to NSE
symbols without requiring a large NER model. Uses exact aliases first,
then a small fuzzy pass (rapidfuzz if installed, else difflib).
"""
from __future__ import annotations

import re
from difflib import get_close_matches
from functools import lru_cache
from typing import Iterable

_SKIP = frozenset(
    {
        "NIFTY", "SENSEX", "BANKNIFTY", "MARKET", "STOCK", "SHARE", "INDEX",
        "BANK", "BANKS", "THE", "AND", "FOR", "HOW", "WHAT", "WHY",
    }
)

# Spoken / typed forms that users use more than the ticker.
_ALIASES: dict[str, str] = {
    "reliance": "RELIANCE",
    "reliance industries": "RELIANCE",
    "ril": "RELIANCE",
    "tcs": "TCS",
    "tata consultancy": "TCS",
    "tata consultancy services": "TCS",
    "infosys": "INFY",
    "infy": "INFY",
    "hdfc bank": "HDFCBANK",
    "hdfcbank": "HDFCBANK",
    "icici bank": "ICICIBANK",
    "icicibank": "ICICIBANK",
    "icicbank": "ICICIBANK",
    "icici": "ICICIBANK",
    "sbi": "SBIN",
    "state bank": "SBIN",
    "state bank of india": "SBIN",
    "itc": "ITC",
    "hindustan unilever": "HINDUNILVR",
    "hul": "HINDUNILVR",
    "l&t": "LT",
    "larsen": "LT",
    "larsen and toubro": "LT",
    "tata motors": "TMPV",
    "hal": "HAL",
    "bel": "BEL",
    "wipro": "WIPRO",
    "hcl": "HCLTECH",
    "hcl tech": "HCLTECH",
    "axis bank": "AXISBANK",
    "kotak": "KOTAKBANK",
    "kotak bank": "KOTAKBANK",
    "maruti": "MARUTI",
    "bajaj finance": "BAJFINANCE",
    "airtel": "BHARTIARTL",
    "bharti airtel": "BHARTIARTL",
    "sun pharma": "SUNPHARMA",
    "asian paints": "ASIANPAINT",
    "tata steel": "TATASTEEL",
    "ntpc": "NTPC",
    "ongc": "ONGC",
    "coal india": "COALINDIA",
    "power grid": "POWERGRID",
    "adani enterprises": "ADANIENT",
    "tech mahindra": "TECHM",
    "yes bank": "YESBANK",
}

_TOKEN_RE = re.compile(r"[a-z0-9&.]+")

# Market vocabulary that sits one letter away from a listed name. "delivery"
# must stay the settlement word, not DELHIVERY; "future" is not FUTURE ENT.
_VOCAB = frozenset(
    {
        "delivery", "intraday", "option", "options", "future", "futures",
        "margin", "dividend", "bonus", "split", "equity", "index", "trading",
        "trade", "order", "limit", "market", "stop", "loss", "target", "sip",
        "mutual", "fund", "funds", "etf", "gold", "silver", "bond", "bonds",
        "swing", "scalp", "position", "positional", "broker", "demat",
        "settlement", "premium", "strike", "expiry", "lot", "lots", "call",
        "put", "straddle", "strangle", "spread", "hedge", "volume", "price",
        "condor", "iron", "chain", "candle", "candlestick", "grey", "gray",
        "gmp", "depth", "expense", "ratio", "qwerty",
        "explain", "meaning", "difference", "versus", "between",
    }
)


def _fuzzy_one(phrase: str, choices: Iterable[str], cutoff: float = 0.86) -> str | None:
    phrase = (phrase or "").strip().lower()
    if len(phrase) < 4:
        return None
    keys = list(choices)
    try:
        from rapidfuzz import process

        hit = process.extractOne(phrase, keys, score_cutoff=int(cutoff * 100))
        if hit:
            return str(hit[0])
    except Exception:
        pass
    matches = get_close_matches(phrase, keys, n=1, cutoff=cutoff)
    return matches[0] if matches else None


@lru_cache(maxsize=1)
def _alias_table() -> dict[str, str]:
    table = dict(_ALIASES)
    try:
        from app.stock_enricher import _NAME_TO_SYMBOL

        for name, sym in (_NAME_TO_SYMBOL or {}).items():
            key = str(name or "").strip().lower()
            val = str(sym or "").strip().upper()
            if key and val and val not in _SKIP:
                table.setdefault(key, val)
    except Exception:
        pass
    return table


def link_symbols(query: str) -> list[str]:
    """Return listed symbols mentioned in the user text, longest alias first."""
    text = (query or "").strip()
    if not text:
        return []
    q = text.lower()
    table = _alias_table()
    found: list[str] = []
    seen: set[str] = set()

    for name, sym in sorted(table.items(), key=lambda item: -len(item[0])):
        if len(name) < 2 or name not in q:
            continue
        if not re.search(r"(?<![a-z0-9])" + re.escape(name) + r"(?![a-z0-9])", q):
            continue
        if sym in _SKIP or sym in seen:
            continue
        found.append(sym)
        seen.add(sym)

    if found:
        return found

    tokens = _TOKEN_RE.findall(q)
    windows: list[str] = []
    for i, tok in enumerate(tokens):
        windows.append(tok)
        if i + 1 < len(tokens):
            windows.append(f"{tok} {tokens[i + 1]}")
        if i + 2 < len(tokens):
            windows.append(f"{tok} {tokens[i + 1]} {tokens[i + 2]}")
    for phrase in sorted(set(windows), key=len, reverse=True):
        if phrase in table:
            sym = table[phrase]
            if sym not in seen and sym not in _SKIP:
                found.append(sym)
                seen.add(sym)
                continue
        if len(phrase) < 5:
            continue
        if all(word in _VOCAB for word in phrase.split()):
            continue
        near = _fuzzy_one(phrase, table.keys())
        if not near:
            continue
        sym = table[near]
        if sym not in seen and sym not in _SKIP:
            found.append(sym)
            seen.add(sym)
    return found


def is_index_or_market_ask(query: str) -> bool:
    q = (query or "").lower()
    if not re.search(
        r"\b(nifty|sensex|banknifty|bank nifty|the market|is market|"
        r"market (today|bullish|bearish|mood|outlook))\b",
        q,
    ):
        return False
    named = [s for s in link_symbols(query) if s not in {"NIFTY", "SENSEX"}]
    return not named
