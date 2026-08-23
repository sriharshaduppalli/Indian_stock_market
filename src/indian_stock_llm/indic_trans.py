"""IndicTrans2 client for English → Telugu.

The 200M/1B models do not fit Cloud Run `bysel-services` (2Gi, 1 CPU, max 1).
This module only talks HTTP to a separate worker. If INDIC_TRANS_URL is unset
or the worker is down, callers keep the phrase-table Telugu layer.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Iterable

logger = logging.getLogger(__name__)

_SRC_LANG = "eng_Latn"
_TGT_LANG = "tel_Telu"
_TOKEN_RE = re.compile(r"\[\[T(\d+)\]\]")
_SPACED_TOKEN_RE = re.compile(r"\[\[\s*T\s*(\d+)\s*\]\]")

_KEEP_LATIN_RE = re.compile(
    r"\b(?:RSI|MACD|P/?E|EPS|SMA|EMA|CAGR|SIP|NSE|BSE|SEBI|NIFTY|SENSEX|"
    r"BUY|SELL|HOLD|WAIT|TRIM|AVOID|ACCUMULATE|NSDL|CDSL|KYC|FOMO|"
    r"IPO|ETF|F&O|FNO)\b",
    flags=re.I,
)
_RUPEE_RE = re.compile(r"₹[\d,]+(?:\.\d+)?")
_TICKER_RE = re.compile(r"\b[A-Z]{3,15}(?:\.(?:NS|BO))?\b")
_SKIP_TICKER_WORDS = {
    "THE", "AND", "FOR", "NOT", "WHY", "HOW", "WHAT", "WITH", "FROM",
    "THIS", "THAT", "YOUR", "LIVE", "LAST", "OPEN", "HIGH", "LOW",
    "NEWS", "TAPE", "SPOT", "READ", "ZONE", "IDEA", "ONLY",
}

_EN_WORD_RE = re.compile(r"\b[A-Za-z]{3,}\b")
_TELUGU_RE = re.compile(r"[\u0C00-\u0C7F]")


def indic_trans_url() -> str:
    return (
        os.getenv("INDIC_TRANS_URL")
        or os.getenv("BYSEL_INDIC_TRANS_URL")
        or ""
    ).strip().rstrip("/")


def indic_trans_enabled() -> bool:
    flag = (os.getenv("INDIC_TRANS_ENABLED") or "").strip().lower()
    if flag in {"0", "false", "no", "off"}:
        return False
    if flag in {"1", "true", "yes", "on"}:
        return bool(indic_trans_url())
    return bool(indic_trans_url())


def _timeout_s() -> float:
    raw = os.getenv("INDIC_TRANS_TIMEOUT_S") or "8"
    try:
        return max(1.0, float(raw))
    except ValueError:
        return 8.0


def shield_protected_tokens(text: str) -> tuple[str, list[str]]:
    """Replace tickers, ₹ amounts, and literacy tokens so NMT cannot rewrite them."""
    tokens: list[str] = []

    def _repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        if raw.isalpha() and raw.upper() in _SKIP_TICKER_WORDS:
            return raw
        tokens.append(raw)
        return f"[[T{len(tokens) - 1}]]"

    shielded = _RUPEE_RE.sub(_repl, text or "")
    shielded = _KEEP_LATIN_RE.sub(_repl, shielded)
    shielded = _TICKER_RE.sub(_repl, shielded)
    return shielded, tokens


def unshield_protected_tokens(text: str, tokens: list[str]) -> str:
    if not text:
        return text

    def _repl(match: re.Match[str]) -> str:
        idx = int(match.group(1))
        if 0 <= idx < len(tokens):
            return tokens[idx]
        return match.group(0)

    restored = _TOKEN_RE.sub(_repl, text)
    return _SPACED_TOKEN_RE.sub(_repl, restored)


def needs_nmt(line: str) -> bool:
    """True when a line still has enough English prose to send to IndicTrans2."""
    if not (line or "").strip():
        return False
    english = _EN_WORD_RE.findall(line)
    if _TELUGU_RE.search(line):
        return len(english) >= 8
    return len(english) >= 4


def translate_english_batch(texts: list[str]) -> list[str] | None:
    """Translate English strings via the IndicTrans2 worker. None = unavailable."""
    if not texts or not indic_trans_enabled():
        return None
    url = indic_trans_url()
    if not url:
        return None
    try:
        import httpx
    except Exception:
        logger.debug("IndicTrans2 skipped: httpx missing")
        return None
    try:
        with httpx.Client(timeout=_timeout_s()) as client:
            response = client.post(
                f"{url}/translate",
                json={
                    "texts": texts,
                    "src_lang": _SRC_LANG,
                    "tgt_lang": _TGT_LANG,
                },
            )
        response.raise_for_status()
        payload = response.json()
        translations = payload.get("translations")
        if not isinstance(translations, list) or len(translations) != len(texts):
            logger.warning("IndicTrans2 worker returned a bad payload")
            return None
        return [str(item or "") for item in translations]
    except Exception as exc:
        logger.warning("IndicTrans2 worker missed: %s", exc)
        return None


def translate_leftover_english(text: str) -> str:
    """Translate leftover English lines. Unchanged when the worker is off."""
    if not text or not indic_trans_enabled():
        return text
    lines = text.splitlines()
    jobs: list[tuple[int, str, list[str]]] = []
    for index, line in enumerate(lines):
        if not needs_nmt(line):
            continue
        shielded, tokens = shield_protected_tokens(line)
        jobs.append((index, shielded, tokens))
    if not jobs:
        return text
    translated = translate_english_batch([shielded for _, shielded, _ in jobs])
    if not translated:
        return text
    out = list(lines)
    for (index, _, tokens), te_line in zip(jobs, translated):
        cleaned = unshield_protected_tokens((te_line or "").strip(), tokens)
        if cleaned:
            out[index] = cleaned
    return "\n".join(out)


def translate_english_lines(lines: Iterable[str]) -> list[str] | None:
    return translate_english_batch(list(lines))
