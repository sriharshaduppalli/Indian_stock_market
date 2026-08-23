"""Indic / retail-language helpers for the Indian Stock LLM.

Hinglish was already understood. This adds Telugu script and romanized Tenglish
so routing, symbols, and answer shape work the same as English asks.
"""
from __future__ import annotations

import re
import unicodedata

_TELUGU_SCRIPT_RE = re.compile(r"[\u0C00-\u0C7F]")

# Longest phrases first. Values are English cues the query contract already knows.
_TELUGU_PHRASES: tuple[tuple[str, str], ...] = (
    ("\u0c39\u0c46\u0c1a\u0c4d\u200c\u0c21\u0c40\u0c0e\u0c2b\u0c4d\u200c\u0c38\u0c40 \u0c2c\u0c4d\u0c2f\u0c3e\u0c02\u0c15\u0c4d", "HDFCBANK"),
    ("\u0c1f\u0c46\u0c15\u0c4d\u0c28\u0c3f\u0c15\u0c32\u0c4d \u0c05\u0c28\u0c3e\u0c32\u0c3f\u0c38\u0c3f\u0c38\u0c4d", "technical analysis"),
    ("\u0c1f\u0c46\u0c15\u0c4d\u0c28\u0c3f\u0c15\u0c32\u0c4d \u0c35\u0c3f\u0c36\u0c4d\u0c32\u0c47\u0c37\u0c23", "technical analysis"),
    ("\u0c27\u0c30 \u0c0e\u0c02\u0c24\u0c41\u0c02\u0c26\u0c3f", "what is the price"),
    ("\u0c27\u0c30 \u0c0e\u0c02\u0c24 \u0c09\u0c02\u0c26\u0c3f", "what is the price"),
    ("\u0c27\u0c30 \u0c0e\u0c02\u0c24", "what is the price"),
    ("\u0c0e\u0c02\u0c24 \u0c09\u0c02\u0c26\u0c3f", "how much is"),
    ("\u0c0e\u0c32\u0c3e \u0c09\u0c02\u0c1f\u0c41\u0c02\u0c26\u0c3f", "predict outlook"),
    ("\u0c08 \u0c35\u0c3e\u0c30\u0c02", "this week"),
    ("\u0c15\u0c4a\u0c28\u0c3e\u0c32\u0c4d\u0c38\u0c3f\u0c02\u0c26\u0c3e", "should I buy"),
    ("\u0c15\u0c4a\u0c28\u0c35\u0c1a\u0c4d\u0c1a\u0c3e", "can I buy"),
    ("\u0c15\u0c4a\u0c28\u0c4a\u0c1a\u0c4d\u0c1a\u0c3e", "can I buy"),
    ("\u0c05\u0c2e\u0c4d\u0c2e\u0c3e\u0c32\u0c4d\u0c38\u0c3f\u0c02\u0c26\u0c3e", "should I sell"),
    ("\u0c30\u0c3f\u0c32\u0c2f\u0c28\u0c4d\u0c38\u0c4d", "RELIANCE"),
    ("\u0c07\u0c28\u0c4d\u0c2b\u0c4b\u0c38\u0c3f\u0c38\u0c4d", "INFY"),
    ("\u0c1f\u0c40\u0c38\u0c40\u0c0e\u0c38\u0c4d", "TCS"),
    ("\u0c1f\u0c3f\u0c38\u0c3f\u0c0e\u0c38\u0c4d", "TCS"),
    ("\u0c10\u0c1f\u0c40\u0c38\u0c40", "ITC"),
    ("\u0c39\u0c46\u0c1a\u0c4d\u200c\u0c21\u0c40\u0c0e\u0c2b\u0c4d\u200c\u0c38\u0c40", "HDFCBANK"),
    ("\u0c39\u0c46\u0c1a\u0c4d\u0c21\u0c40\u0c0e\u0c2b\u0c4d\u0c38\u0c40", "HDFCBANK"),
    ("\u0c0e\u0c38\u0c4d\u200c\u0c2c\u0c40\u0c10", "SBIN"),
    ("\u0c0e\u0c38\u0c4d\u0c2c\u0c40\u0c10", "SBIN"),
    ("\u0c10\u0c38\u0c40\u0c10\u0c38\u0c40\u0c10", "ICICIBANK"),
    ("\u0c35\u0c3f\u0c2a\u0c4d\u0c30\u0c4b", "WIPRO"),
    ("\u0c39\u0c46\u0c1a\u0c4d\u200c\u0c38\u0c40\u0c0e\u0c32\u0c4d", "HCLTECH"),
    ("\u0c15\u0c4a\u0c28\u0c3e\u0c32\u0c3e", "should I buy"),
    ("\u0c15\u0c4a\u0c28\u0c3e\u0c32\u0c3f", "buy"),
    ("\u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f", "buy"),
    ("\u0c05\u0c2e\u0c4d\u0c2e\u0c3e\u0c32\u0c3e", "should I sell"),
    ("\u0c05\u0c2e\u0c4d\u0c2e\u0c3e\u0c32\u0c3f", "sell"),
    ("\u0c05\u0c2e\u0c4d\u0c2e\u0c02\u0c21\u0c3f", "sell"),
    ("\u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41", "news"),
    ("\u0c38\u0c46\u0c02\u0c1f\u0c3f\u0c2e\u0c46\u0c02\u0c1f\u0c4d", "sentiment"),
    ("\u0c35\u0c3f\u0c36\u0c4d\u0c32\u0c47\u0c37\u0c23", "analysis"),
    ("\u0c2a\u0c4b\u0c32\u0c4d\u0c1a\u0c02\u0c21\u0c3f", "compare"),
    ("\u0c2a\u0c4b\u0c32\u0c3f\u0c15", "compare"),
    ("\u0c05\u0c02\u0c1a\u0c28\u0c3e", "predict"),
    ("\u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41", "now"),
    ("\u0c2c\u0c3e\u0c17\u0c41\u0c02\u0c26\u0c3e", "is it good"),
    ("\u0c2c\u0c3e\u0c17\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c3e", "how are you"),
    ("\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02", "hello"),
    ("\u0c28\u0c2e\u0c38\u0c4d\u0c24\u0c47", "hello"),
    ("\u0c37\u0c47\u0c30\u0c4d\u0c32\u0c41", "stocks"),
    ("\u0c37\u0c47\u0c30\u0c41", "stock"),
    ("\u0c38\u0c4d\u0c1f\u0c3e\u0c15\u0c4d", "stock"),
    ("\u0c2e\u0c3e\u0c30\u0c4d\u0c15\u0c46\u0c1f\u0c4d", "market"),
    ("\u0c21\u0c3f\u0c2b\u0c46\u0c28\u0c4d\u0c38\u0c4d", "defence"),
    ("\u0c2c\u0c4d\u0c2f\u0c3e\u0c02\u0c15\u0c3f\u0c02\u0c17\u0c4d", "banking"),
    ("\u0c2b\u0c3e\u0c30\u0c4d\u0c2e\u0c3e", "pharma"),
    ("\u0c0f\u0c2e\u0c3f\u0c1f\u0c3f", "what is"),
    ("\u0c0f\u0c02\u0c1f\u0c3f", "what is"),
    ("\u0c05\u0c02\u0c1f\u0c47", "meaning"),
    ("\u0c05\u0c30\u0c4d\u0c25\u0c02", "meaning"),
    ("\u0c1a\u0c46\u0c2a\u0c4d\u0c2a\u0c02\u0c21\u0c3f", "tell"),
    ("\u0c1a\u0c46\u0c2a\u0c4d\u0c2a\u0c41", "tell"),
    ("\u0c17\u0c41\u0c30\u0c3f\u0c02\u0c1a\u0c3f", "about"),
    ("\u0c27\u0c30", "price"),
    ("\u0c0e\u0c02\u0c24\u0c41\u0c02\u0c26\u0c3f", "how much"),
    ("\u0c0e\u0c02\u0c24", "how much"),
)

_TENGLISH_PHRASES: tuple[tuple[str, str], ...] = (
    (r"\bela\s+untundh?i\b", "predict outlook"),
    (r"\bentha\s+undh?i\b", "how much is"),
    (r"\bdhara\s+entha\b", "what is the price"),
    (r"\bee\s+varam\b", "this week"),
    (r"\be\s+varam\b", "this week"),
    (r"\bkonalsinda\b", "should I buy"),
    (r"\bkonaalsinda\b", "should I buy"),
    (r"\bkonavaccha\b", "can I buy"),
    (r"\bkonoccha\b", "can I buy"),
    (r"\bkonaala\b", "should I buy"),
    (r"\bkonala\b", "should I buy"),
    (r"\bkonaali\b", "should I buy"),
    (r"\bkonali\b", "should I buy"),
    (r"\bkonandi\b", "buy"),
    (r"\bammalsinda\b", "should I sell"),
    (r"\bammaala\b", "should I sell"),
    (r"\bammala\b", "should I sell"),
    (r"\bammali\b", "should I sell"),
    (r"\bammandi\b", "sell"),
    (r"\bvarthalu\b", "news"),
    (r"\bdhara\b", "price"),
    (r"\benthaundi\b", "how much is"),
    (r"\bentha\b", "how much"),
    (r"\bippudu\b", "now"),
    (r"\bbaagundh?a\b", "is it good"),
    (r"\bbagundh?a\b", "is it good"),
    (r"\bgurinchi\b", "about"),
    (r"\bcheppandi\b", "tell"),
    (r"\bcheppu\b", "tell"),
    (r"\bemiti[o]?\b", "what is"),
    (r"\benti\b", "what is"),
    (r"\bpolchandi\b", "compare"),
    (r"\bpolchu\b", "compare"),
    (r"\bpolika\b", "compare"),
    (r"\banchana\b", "predict"),
    (r"\bnamaskaram\b", "hello"),
)

_HINGLISH = {
    "kharid": "buy",
    "kharido": "buy",
    "kharidna": "buy",
    "kharidne": "buy",
    "bech": "sell",
    "becho": "sell",
    "bechna": "sell",
    "bechne": "sell",
    "kitna": "how much",
    "kaise": "how",
    "kya": "what",
    "hai": "is",
}

_STANCE_TE = {
    "BUY": "\u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f (BUY)",
    "ACCUMULATE": "\u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40\u0c17\u0c3e \u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f (ACCUMULATE)",
    "HOLD": "\u0c39\u0c4b\u0c32\u0c4d\u0c21\u0c4d \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f (HOLD)",
    "WAIT": "\u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f (WAIT)",
    "TRIM": "\u0c24\u0c17\u0c4d\u0c17\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f (TRIM)",
    "SELL": "\u0c05\u0c2e\u0c4d\u0c2e\u0c02\u0c21\u0c3f (SELL)",
    "AVOID": "\u0c26\u0c42\u0c30\u0c02\u0c17\u0c3e \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f (AVOID)",
}

_HEADER_TE = (
    (r"\*\*Direct answer:\*\*", "**\u0c28\u0c47\u0c30\u0c41\u0c17\u0c3e \u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02:**"),
    (r"\*\*Why:\*\*", "**\u0c15\u0c3e\u0c30\u0c23\u0c02:**"),
    (r"\*\*Action:\*\*", "**\u0c1a\u0c30\u0c4d\u0c2f:**"),
    (r"\*\*Sentiment:\*\*", "**\u0c38\u0c46\u0c02\u0c1f\u0c3f\u0c2e\u0c46\u0c02\u0c1f\u0c4d:**"),
    (r"\*\*News & catalysts:\*\*", "**\u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41 / \u0c15\u0c3e\u0c30\u0c15\u0c3e\u0c32\u0c41:**"),
    (r"\*\*Key levels & tape:\*\*", "**\u0c2e\u0c41\u0c16\u0c4d\u0c2f \u0c38\u0c4d\u0c25\u0c3e\u0c2f\u0c3f\u0c32\u0c41:**"),
    (r"\*\*Signal highlights:\*\*", "**\u0c38\u0c02\u0c15\u0c47\u0c24\u0c3e\u0c32\u0c41:**"),
    (r"\*\*Quick math:\*\*", "**\u0c32\u0c46\u0c15\u0c4d\u0c15:**"),
    (r"\*\*For \(setup\):\*\*", "**\u0c05\u0c28\u0c41\u0c15\u0c42\u0c32\u0c02:**"),
    (r"\*\*Against:\*\*", "**\u0c2a\u0c4d\u0c30\u0c24\u0c3f\u0c15\u0c42\u0c32\u0c02:**"),
)

_TE_GREETING_RE = re.compile(
    r"(\u0c28\u0c2e\u0c38\u0c4d\u0c24\u0c47|\u0c28\u0c2e\u0c38\u0c4d\u0c15\u0c3e\u0c30\u0c02|\u0c2c\u0c3e\u0c17\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c3e|\bnamaskaram\b|\bela\s+unnav\b)",
    re.I,
)


def _strip_wrappers(query: str) -> str:
    text = unicodedata.normalize("NFC", (query or "").strip())
    text = text.replace("\u200c", "").replace("\u200d", "")
    if text.lower().startswith("user_query:"):
        text = text.split(":", 1)[1].strip()
    if " | context:" in text:
        text = text.split(" | context:", 1)[0].strip()
    return text


def has_telugu_script(query: str) -> bool:
    return bool(_TELUGU_SCRIPT_RE.search(query or ""))


def detect_query_language(query: str) -> str:
    """Return te | te-en | hi-en | en."""
    raw = _strip_wrappers(query)
    if has_telugu_script(raw):
        return "te"
    low = raw.lower()
    if any(re.search(pat, low) for pat, _ in _TENGLISH_PHRASES):
        return "te-en"
    if re.search(r"\b(kharid|bech|kitna|kaise|kya hai|chahiye)\b", low):
        return "hi-en"
    return "en"


def is_telugu_query(query: str) -> bool:
    return detect_query_language(query) in {"te", "te-en"}


def is_telugu_small_talk(query: str) -> bool:
    return telugu_small_talk_kind(query) is not None


def telugu_small_talk_kind(query: str) -> str | None:
    raw = _strip_wrappers(query)
    if not raw or not _TE_GREETING_RE.search(raw):
        return None
    leftover = _TELUGU_SCRIPT_RE.sub(" ", raw)
    leftover = re.sub(r"[^\w\s]", " ", leftover, flags=re.UNICODE)
    extra = [t for t in leftover.split() if t.lower() not in {"hi", "hello", "hey", "namaste", "namaskaram"}]
    if len(extra) > 2:
        return None
    if re.search(r"(\u0c2c\u0c3e\u0c17\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c30\u0c3e|\bela\s+unnav\b)", raw, flags=re.I):
        return "how_are_you"
    return "greeting"


def telugu_small_talk_replies() -> dict[str, str]:
    return {
        "greeting": (
            "\u0c28\u0c2e\u0c38\u0c4d\u0c24\u0c47! \u0c28\u0c47\u0c28\u0c41 BYSEL AI. "
            "\u0c38\u0c4d\u0c1f\u0c3e\u0c15\u0c4d \u0c27\u0c30, \u0c15\u0c4a\u0c28\u0c3e\u0c32\u0c3e/\u0c05\u0c2e\u0c4d\u0c2e\u0c3e\u0c32\u0c3e, "
            "\u0c2a\u0c4b\u0c32\u0c3f\u0c15 \u0c32\u0c47\u0c26\u0c3e \u0c35\u0c3e\u0c32\u0c4d\u0c2f\u0c41\u0c2f\u0c47\u0c37\u0c28\u0c4d \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f."
        ),
        "thanks": (
            "\u0c2a\u0c30\u0c4d\u0c35\u0c3e\u0c32\u0c47\u0c26\u0c41. "
            "\u0c2e\u0c3e\u0c30\u0c4d\u0c15\u0c46\u0c1f\u0c4d \u0c38\u0c02\u0c26\u0c47\u0c39\u0c02 \u0c09\u0c02\u0c1f\u0c47 \u0c2e\u0c33\u0c4d\u0c32\u0c40 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f."
        ),
        "bye": "\u0c35\u0c46\u0c33\u0c4d\u0c24\u0c3e! \u0c36\u0c41\u0c2d\u0c02. Happy investing.",
        "how_are_you": (
            "\u0c2c\u0c3e\u0c17\u0c41\u0c28\u0c4d\u0c28\u0c3e\u0c28\u0c41. "
            "\u0c0f \u0c38\u0c4d\u0c1f\u0c3e\u0c15\u0c4d \u0c17\u0c41\u0c30\u0c3f\u0c02\u0c1a\u0c3f \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f?"
        ),
    }


def normalize_user_query(query: str) -> str:
    """Expand Telugu / Tenglish / Hinglish into English routing cues."""
    text = _strip_wrappers(query)
    # Phrase table uses NFC + no ZWJ, same as _strip_wrappers.
    for src, dest in _TELUGU_PHRASES:
        src_norm = unicodedata.normalize("NFC", src).replace("\u200c", "").replace("\u200d", "")
        if src_norm and src_norm in text:
            text = text.replace(src_norm, f" {dest} ")
    for pat, dest in _TENGLISH_PHRASES:
        text = re.sub(pat, dest, text, flags=re.I)
    pattern = r"\b(" + "|".join(re.escape(k) for k in _HINGLISH) + r")\b"

    def _hi(match: re.Match[str]) -> str:
        token = match.group(0)
        mapped = _HINGLISH.get(token.lower())
        if not mapped:
            return token
        return f"{token} {mapped}"

    text = re.sub(pattern, _hi, text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def learning_query_fields(query: str) -> dict[str, str | bool]:
    """Normalize a Telugu / Tenglish / English ask onto one learning key.

    Closed-loop RAG merges repeats of the same topic. Without this, Telugu
    script and English rewrites never hit the promote threshold together.
    """
    original = _strip_wrappers(query)
    language = detect_query_language(original)
    normalized = normalize_user_query(original)
    key = re.sub(r"\s+", " ", (normalized or original)).strip().lower()
    return {
        "original": original,
        "normalized": normalized,
        "key": key,
        "language": language,
        "is_telugu": language in {"te", "te-en"},
    }


def _stance_from_answer(answer: str) -> str | None:
    head = (answer or "")[:500]
    match = re.search(
        r"\b(ACCUMULATE|BUY|SELL|HOLD|WAIT|TRIM|AVOID)\b",
        head,
        flags=re.I,
    )
    if not match:
        return None
    return match.group(1).upper()


def localize_assistant_answer(query: str, answer: str) -> str:
    """Reply in Telugu when the user asked in Telugu / Tenglish."""
    if not answer or not is_telugu_query(query):
        return answer
    try:
        from .telugu_response import render_telugu_answer

        return render_telugu_answer(query, answer)
    except Exception:
        localized = answer
        for pat, repl in _HEADER_TE:
            localized = re.sub(pat, repl, localized)
        if localized.startswith("**\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02:**"):
            return localized
        stance = _stance_from_answer(answer)
        stance_te = _STANCE_TE.get(stance or "", "")
        if stance_te:
            lead = (
                f"**\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02:** {stance_te} "
                "\u2014 \u0c35\u0c3f\u0c35\u0c30\u0c3e\u0c32\u0c41 \u0c15\u0c3f\u0c02\u0c26 \u0c09\u0c28\u0c4d\u0c28\u0c3e\u0c2f\u0c3f.\n\n"
            )
        else:
            lead = (
                "**\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02:** "
                "\u0c2e\u0c40 \u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28 \u0c05\u0c30\u0c4d\u0c25\u0c2e\u0c48\u0c02\u0c26\u0c3f. "
                "\u0c38\u0c02\u0c16\u0c4d\u0c2f\u0c32\u0c41 / \u0c38\u0c3f\u0c02\u0c2c\u0c32\u0c4d\u0c38\u0c4d "
                "\u0c07\u0c02\u0c17\u0c4d\u0c32\u0c40\u0c37\u0c4d\u200c\u0c32\u0c4b\u0c28\u0c47 \u0c09\u0c02\u0c1a\u0c3e\u0c28\u0c41.\n\n"
            )
        return lead + localized


def telugu_format_instruction() -> str:
    return (
        "The user wrote in Telugu. Reply in clear Telugu (native script) for headings "
        "and the direct call. Keep ticker symbols, rupee amounts, and indicator names "
        "(RSI, MACD, P/E) in Latin. End with a one-line English disclaimer."
    )


def telugu_clarifier(kind: str, names: str = "") -> str:
    if kind == "ambiguous":
        return (
            f"\u0c06 \u0c2a\u0c47\u0c30\u0c41 \u0c12\u0c15\u0c1f\u0c3f \u0c15\u0c02\u0c1f\u0c47 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35 "
            f"\u0c15\u0c02\u0c2a\u0c46\u0c28\u0c40\u0c32\u0c15\u0c41 \u0c38\u0c30\u0c3f\u0c2a\u0c4b\u0c24\u0c4b\u0c02\u0c26\u0c3f ({names}). "
            "\u0c0f \u0c38\u0c3f\u0c02\u0c2c\u0c32\u0c4d \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f?"
        )
    return (
        "\u0c0f NSE/BSE \u0c38\u0c3f\u0c02\u0c2c\u0c32\u0c4d? \u0c15\u0c4b\u0c1f\u0c4d, \u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41, "
        "\u0c1f\u0c46\u0c15\u0c4d\u0c28\u0c3f\u0c15\u0c32\u0c4d\u0c38\u0c4d, \u0c35\u0c3e\u0c32\u0c4d\u0c2f\u0c41\u0c2f\u0c47\u0c37\u0c28\u0c4d, "
        "\u0c2a\u0c4b\u0c32\u0c3f\u0c15, \u0c15\u0c4a\u0c28\u0c3e\u0c32\u0c3e/\u0c05\u0c2e\u0c4d\u0c2e\u0c3e\u0c32\u0c3e, "
        "\u0c32\u0c47\u0c26\u0c3e \u0c05\u0c02\u0c1a\u0c28\u0c3e \u2014 \u0c0f\u0c26\u0c3f \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c4b \u0c1a\u0c46\u0c2a\u0c4d\u0c2a\u0c02\u0c21\u0c3f."
    )
