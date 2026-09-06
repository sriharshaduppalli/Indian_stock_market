"""Turn ISM English answers into Telugu when the user asked in Telugu / Tenglish.

Composer still writes English so numbers stay exact. This layer translates
headings, stance lines, labels, and common literacy topics. Tickers, rupee
amounts, and indicator names (RSI, MACD, P/E) stay Latin.
"""
from __future__ import annotations

import re

_LOCALIZED_MARK = "\u0c24\u0c46\u0c32\u0c41\u0c17\u0c41 \u0c38\u0c3e\u0c30\u0c3e\u0c02\u0c36\u0c02"

_STANCE_TE = {
    "BUY": "\u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f (BUY)",
    "ACCUMULATE": "\u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40\u0c17\u0c3e \u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f (ACCUMULATE)",
    "HOLD": "\u0c39\u0c4b\u0c32\u0c4d\u0c21\u0c4d \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f (HOLD)",
    "WAIT": "\u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f (WAIT)",
    "TRIM": "\u0c24\u0c17\u0c4d\u0c17\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f (TRIM)",
    "SELL": "\u0c05\u0c2e\u0c4d\u0c2e\u0c02\u0c21\u0c3f (SELL)",
    "AVOID": "\u0c26\u0c42\u0c30\u0c02\u0c17\u0c3e \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f (AVOID)",
}

_HEADER_TE: tuple[tuple[str, str], ...] = (
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
    (r"\*\*Context awareness:\*\*", "**\u0c38\u0c02\u0c26\u0c30\u0c4d\u0c2d\u0c02:**"),
    (r"\*\*Fundamental detail\*\*", "**\u0c2b\u0c02\u0c21\u0c2e\u0c46\u0c02\u0c1f\u0c32\u0c4d \u0c35\u0c3f\u0c35\u0c30\u0c02**"),
    (r"\*\*Quantitative stack \(computed\):\*\*", "**\u0c15\u0c4d\u0c35\u0c3e\u0c02\u0c1f\u0c4d \u0c38\u0c4d\u0c1f\u0c3e\u0c15\u0c4d:**"),
    (r"\*\*Action legend \(paper practice\):\*\*",
     "**\u0c1a\u0c30\u0c4d\u0c2f \u0c32\u0c46\u0c1c\u0c46\u0c02\u0c21\u0c4d (\u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c2a\u0c4d\u0c30\u0c3e\u0c15\u0c4d\u0c1f\u0c40\u0c38\u0c4d):**"),
    (r"\*\*Technical readout:\*\*", "**\u0c1f\u0c46\u0c15\u0c4d\u0c28\u0c3f\u0c15\u0c32\u0c4d \u0c35\u0c3f\u0c35\u0c30\u0c02:**"),
    (r"\*\*Paper ticket:\*\*", "**\u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c1f\u0c3f\u0c15\u0c46\u0c1f\u0c4d:**"),
    (r"\*\*Requested checks:\*\*", "**\u0c05\u0c21\u0c3f\u0c17\u0c3f\u0c28 \u0c1a\u0c46\u0c15\u0c4d\u0c32\u0c41:**"),
    (r"\*\*Practice levels:\*\*", "**\u0c2a\u0c4d\u0c30\u0c3e\u0c15\u0c4d\u0c1f\u0c40\u0c38\u0c4d \u0c38\u0c4d\u0c25\u0c3e\u0c2f\u0c3f\u0c32\u0c41:**"),
    (r"\*\*Bias \(paper\):\*\*", "**\u0c2c\u0c2f\u0c3e\u0c38\u0c4d (\u0c2a\u0c47\u0c2a\u0c30\u0c4d):**"),
    (r"\*\*How it's doing:\*\*", "**\u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c0e\u0c32\u0c3e \u0c09\u0c02\u0c26\u0c3f:**"),
    (r"\*\*Overall:\*\*", "**\u0c2e\u0c4a\u0c24\u0c4d\u0c24\u0c02:**"),
    (r"\*\*Read:\*\*", "**\u0c35\u0c3f\u0c35\u0c30\u0c02:**"),
    (r"\*\*What is driving the mood:\*\*", "**\u0c2e\u0c42\u0c21\u0c4d \u0c0e\u0c02\u0c26\u0c41\u0c15\u0c41:**"),
    (r"\*\*News sentences:\*\*", "**\u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41:**"),
    (r"\*\*Comparison scorecard \(educational\)\*\*", "**\u0c2a\u0c4b\u0c32\u0c3f\u0c15 (\u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e):**"),
    (r"\*\*([A-Z0-9.&-]+) \u2014 sentiment analysis\*\*",
     "**\\1 \u2014 \u0c38\u0c46\u0c02\u0c1f\u0c3f\u0c2e\u0c46\u0c02\u0c1f\u0c4d**"),
    (r"\*\*([A-Z0-9.&-]+) \u2014 indicator snapshot\*\*",
     "**\\1 \u2014 RSI / indicator**"),
    (r"\*\*([A-Z0-9.&-]+) \u2014 market sentiment snapshot\*\*",
     "**\\1 \u2014 \u0c2e\u0c3e\u0c30\u0c4d\u0c15\u0c46\u0c1f\u0c4d \u0c38\u0c46\u0c02\u0c1f\u0c3f\u0c2e\u0c46\u0c02\u0c1f\u0c4d**"),
)

_BODY_TE: tuple[tuple[str, str], ...] = (
    (
        "SELL bias for short-term / avoid chasing; wait for cool-off toward support",
        "\u0c38\u0c4d\u0c35\u0c32\u0c4d\u0c2a\u0c15\u0c3e\u0c32\u0c02\u0c32\u0c4b SELL \u0c2c\u0c2f\u0c3e\u0c38\u0c4d \u2014 "
        "\u0c35\u0c46\u0c02\u0c1f\u0c3e\u0c21\u0c15\u0c02\u0c21\u0c3f; support \u0c26\u0c17\u0c4d\u0c17\u0c30 "
        "cool-off \u0c15\u0c4b\u0c38\u0c02 \u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f",
    ),
    (
        "BUY bias (paper) \u2014 wait for entry zone",
        "\u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f (BUY) \u0c2c\u0c2f\u0c3e\u0c38\u0c4d (\u0c2a\u0c47\u0c2a\u0c30\u0c4d) \u2014 "
        "entry zone \u0c15\u0c4b\u0c38\u0c02 \u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f",
    ),
    (
        "ACCUMULATE on dips (paper) \u2014 staged entries",
        "dips\u0c32\u0c4b ACCUMULATE (\u0c2a\u0c47\u0c2a\u0c30\u0c4d) \u2014 "
        "\u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40 entry",
    ),
    (
        "HOLD / wait \u2014 no clear edge yet",
        "\u0c39\u0c4b\u0c32\u0c4d\u0c21\u0c4d / \u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f \u2014 "
        "\u0c07\u0c02\u0c15\u0c3e \u0c38\u0c4d\u0c2a\u0c37\u0c4d\u0c1f\u0c2e\u0c48\u0c28 edge \u0c32\u0c47\u0c26\u0c41",
    ),
    (
        "TRIM / reduce on strength (paper)",
        "TRIM / \u0c2a\u0c48\u0c15\u0c3f \u0c35\u0c1a\u0c4d\u0c1a\u0c3f\u0c28\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 "
        "\u0c24\u0c17\u0c4d\u0c17\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)",
    ),
    (
        "SELL / avoid fresh buys (paper)",
        "\u0c05\u0c2e\u0c4d\u0c2e\u0c02\u0c21\u0c3f (SELL) / \u0c15\u0c4a\u0c24\u0c4d\u0c24 "
        "\u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c33\u0c4d\u0c32\u0c41 \u0c35\u0c26\u0c4d\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)",
    ),
    (
        "WAIT \u2014 insufficient signal quality",
        "\u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f (WAIT) \u2014 signal \u0c38\u0c30\u0c3f\u0c2a\u0c4b\u0c32\u0c47\u0c26\u0c41",
    ),
    (" \u2014 paper buy plan", " \u2014 \u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c32 \u0c2a\u0c4d\u0c32\u0c3e\u0c28\u0c4d"),
    (" \u2014 paper sell plan", " \u2014 \u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c05\u0c2e\u0c4d\u0c2e\u0c15\u0c02 \u0c2a\u0c4d\u0c32\u0c3e\u0c28\u0c4d"),
    (" \u2014 paper trade plan", " \u2014 \u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c1f\u0c4d\u0c30\u0c47\u0c21\u0c4d \u0c2a\u0c4d\u0c32\u0c3e\u0c28\u0c4d"),
    (" \u2014 news & catalysts", " \u2014 \u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41 / \u0c15\u0c3e\u0c30\u0c15\u0c3e\u0c32\u0c41"),
    (" \u2014 indicator snapshot", " \u2014 RSI / indicator"),
    (" \u2014 snapshot", " \u2014 \u0c38\u0c28\u0c4d\u0c28\u0c3f\u0c35\u0c47\u0c36\u0c02"),
    (" \u2014 live quote \u00b7 how much now", " \u2014 \u0c32\u0c48\u0c35\u0c4d \u0c15\u0c4b\u0c1f\u0c4d \u00b7 \u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c0e\u0c02\u0c24"),
    (" \u2014 live quote", " \u2014 \u0c32\u0c48\u0c35\u0c4d \u0c15\u0c4b\u0c1f\u0c4d"),
    (" \u2014 sentiment view", " \u2014 \u0c38\u0c46\u0c02\u0c1f\u0c3f\u0c2e\u0c46\u0c02\u0c1f\u0c4d"),
    (" \u2014 market mood", " \u2014 \u0c2e\u0c3e\u0c30\u0c4d\u0c15\u0c46\u0c1f\u0c4d \u0c2e\u0c42\u0c21\u0c4d"),
    ("_Live quote snapshot \u2014 educational only._",
     "_\u0c32\u0c48\u0c35\u0c4d \u0c15\u0c4b\u0c1f\u0c4d \u2014 \u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e._"),
    ("_News digest only \u2014 not a buy/sell call._",
     "_\u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41 \u0c2e\u0c3e\u0c24\u0c4d\u0c30\u0c2e\u0c47 \u2014 buy/sell call \u0c15\u0c3e\u0c26\u0c41._"),
    ("\u2022 Sector/tape cue:", "\u2022 Sector/tape:"),
    ("Avoid fresh buys / cut exposure (paper)",
     "\u0c15\u0c4a\u0c24\u0c4d\u0c24 \u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c32\u0c41 \u0c35\u0c26\u0c4d\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)"),
    ("Skip for now \u2014 setup or data quality not ready (paper)",
     "\u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 skip \u2014 setup \u0c38\u0c3f\u0c26\u0c4d\u0c27\u0c02 \u0c15\u0c3e\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)"),
    ("Fresh long bias \u2014 look for entries in the zone (paper)",
     "\u0c15\u0c4a\u0c24\u0c4d\u0c24 long \u0c2c\u0c2f\u0c3e\u0c38\u0c4d \u2014 entry zone \u0c1a\u0c42\u0c21\u0c02\u0c21\u0c3f (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)"),
    ("No directional edge (paper)", "\u0c26\u0c3f\u0c36 \u0c32\u0c47\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)"),
    (" outlook (educational snapshot)",
     " \u2014 \u0c32\u0c48\u0c35\u0c4d \u0c38\u0c4d\u0c28\u0c3e\u0c2a\u0c4d\u0c37\u0c3e\u0c1f\u0c4d"),
    (
        "Read as a short-term bias from enrich signals \u2014 not a futures tip / not SPAN.",
        "\u0c38\u0c4d\u0c35\u0c32\u0c4d\u0c2a\u0c15\u0c3e\u0c32\u0c02 bias \u2014 F&O tip \u0c15\u0c3e\u0c26\u0c41.",
    ),
    ("\u2022 Last\u2248", "\u2022 \u0c1a\u0c3f\u0c35\u0c30\u0c3f\u2248"),
    (" | Trend\u2248", " | trend\u2248"),
    (" | RSI\u2248", " | RSI\u2248"),
    ("\u2022 Support\u2248", "\u2022 \u0c38\u0c2a\u0c4b\u0c30\u0c4d\u0c1f\u0c4d\u2248"),
    (" | Resistance\u2248", " | \u0c30\u0c46\u0c38\u0c3f\u0c38\u0c4d\u0c1f\u0c46\u0c28\u0c4d\u0c38\u0c4d\u2248"),
    (" | Resistance:", " | \u0c30\u0c46\u0c38\u0c3f\u0c38\u0c4d\u0c1f\u0c46\u0c28\u0c4d\u0c38\u0c4d:"),
    ("_Educational \u2014 confirm with live NSE quotes before trading._",
     "_\u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u2014 live NSE/BSE\u0c24\u0c4b \u0c38\u0c30\u0c3f\u0c2a\u0c3e\u0c26\u0c3f\u0c1a\u0c3e\u0c32\u0c3f._"),
    ("\u2022 MACD bias\u2248None", "\u2022 MACD: n/a"),
    ("trend\u2248bearish", "trend\u2248\u0c2c\u0c47\u0c30\u0c3f\u0c37\u0c4d"),
    ("trend\u2248bullish", "trend\u2248\u0c2c\u0c41\u0c32\u0c3f\u0c37\u0c4d"),
    ("trend\u2248neutral", "trend\u2248\u0c24\u0c1f\u0c38\u0c4d\u0c25\u0c02"),
    ("RSI\u2248", "RSI\u2248"),
    (" (neutral)", " (neutral)"),
    (" \u2014 paper-practice view", " \u2014 \u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c2a\u0c4d\u0c30\u0c3e\u0c15\u0c4d\u0c1f\u0c40\u0c38\u0c4d"),
    (" \u2014 quantitative stack", " \u2014 \u0c15\u0c4d\u0c35\u0c3e\u0c02\u0c1f\u0c4d \u0c38\u0c4d\u0c1f\u0c3e\u0c15\u0c4d"),
    (" \u2014 momentum check", " \u2014 momentum check"),
    (" \u2014 overbought check", " \u2014 overbought check"),
    (" \u2014 indicator snapshot", " \u2014 indicator snapshot"),
    (" fundamental snapshot", " fundamental snapshot"),
    (" \u2014 last price", " \u2014 \u0c1a\u0c3f\u0c35\u0c30\u0c3f \u0c27\u0c30"),
    (
        "RSI alone is not enough \u2014 confirm with trend and volume before paper trades.",
        "RSI \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c47 \u0c1a\u0c3e\u0c32\u0c26\u0c41 \u2014 paper trade \u0c15\u0c41 \u0c2e\u0c41\u0c02\u0c26\u0c41 "
        "trend, volume\u0c24\u0c4b confirm \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f.",
    ),
    (
        "Use with price structure / volume \u2014 one indicator is not a trade plan.",
        "\u0c27\u0c30 / volume\u0c24\u0c4b \u0c15\u0c32\u0c3f\u0c2a\u0c3f \u0c1a\u0c42\u0c21\u0c02\u0c21\u0c3f \u2014 "
        "\u0c12\u0c15\u0c4d\u0c15 indicator trade plan \u0c15\u0c3e\u0c26\u0c41.",
    ),
    (
        "I could not read a live RSI yet. Retry in a moment or ask",
        "live RSI \u0c07\u0c02\u0c15\u0c3e \u0c30\u0c3e\u0c32\u0c47\u0c26\u0c41. \u0c15\u0c4a\u0c26\u0c4d\u0c26\u0c3f\u0c38\u0c47\u0c2a\u0c1f\u0c3f "
        "\u0c24\u0c30\u0c4d\u0c35\u0c3e\u0c24 \u0c2e\u0c33\u0c4d\u0c32\u0c40 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f \u0c32\u0c47\u0c26\u0c3e \u0c07\u0c32\u0c3e \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f",
    ),
    (
        "Interpret P/E vs sector peers and growth \u2014 a low P/E alone is not 'cheap'.",
        "P/E\u0c28\u0c3f sector peers, growth\u0c24\u0c4b \u0c2a\u0c4b\u0c32\u0c4d\u0c1a\u0c02\u0c21\u0c3f \u2014 "
        "\u0c24\u0c15\u0c4d\u0c15\u0c41\u0c35 P/E \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c47 \u0c1a\u0c4c\u0c15 \u0c15\u0c3e\u0c26\u0c41.",
    ),
    (
        "Size from risk budget / goals first; never from FOMO. Educational only.",
        "\u0c2e\u0c41\u0c02\u0c26\u0c41 risk budget / \u0c32\u0c15\u0c4d\u0c37\u0c4d\u0c2f\u0c3e\u0c32 \u0c2a\u0c4d\u0c30\u0c15\u0c3e\u0c30\u0c02 size \u2014 "
        "FOMO\u0c24\u0c4b \u0c15\u0c3e\u0c26\u0c41. \u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02.",
    ),
    (
        "No fresh headlines in the current feed \u2014 try again shortly.",
        "\u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c15\u0c4a\u0c24\u0c4d\u0c24 headlines \u0c32\u0c47\u0c35\u0c41 \u2014 "
        "\u0c15\u0c4a\u0c26\u0c4d\u0c26\u0c3f\u0c38\u0c47\u0c2a\u0c1f\u0c3f \u0c24\u0c30\u0c4d\u0c35\u0c3e\u0c24 \u0c1a\u0c42\u0c21\u0c02\u0c21\u0c3f.",
    ),
    ("Sentiment pack unavailable right now.", "sentiment pack \u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c32\u0c47\u0c26\u0c41."),
    (
        "Mix of signals \u2014 size small and demand confirmation at levels.",
        "signals mix \u2014 \u0c1a\u0c3f\u0c28\u0c4d\u0c28 size, levels \u0c35\u0c26\u0c4d\u0c26 confirm \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f.",
    ),
    (
        "Single-indicator calls fail in strong trends; confirm with volume/news.",
        "\u0c2c\u0c32\u0c2e\u0c48\u0c28 trend\u0c32\u0c4b \u0c12\u0c15\u0c4d\u0c15 indicator \u0c1a\u0c3e\u0c32\u0c26\u0c41; volume/news\u0c24\u0c4b confirm \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f.",
    ),
    (
        "No clear edge \u2014 stay flat or keep what you have (paper)",
        "\u0c38\u0c4d\u0c2a\u0c37\u0c4d\u0c1f\u0c2e\u0c48\u0c28 edge \u0c32\u0c47\u0c26\u0c41 \u2014 "
        "\u0c15\u0c4a\u0c24\u0c4d\u0c24 \u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c32\u0c41 \u0c35\u0c26\u0c4d\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)",
    ),
    (
        "Ask for technicals, news, valuation, or a paper trade plan if you need more depth.",
        "\u0c07\u0c02\u0c15\u0c3e technicals, \u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41, valuation \u0c32\u0c47\u0c26\u0c3e "
        "paper trade plan \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c02\u0c1f\u0c47 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f.",
    ),
    (
        "if you want a paper entry, stop, and targets.",
        "paper entry / stop / targets \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c02\u0c1f\u0c47 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f.",
    ),
    ("**How it's doing:**", "**\u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c0e\u0c32\u0c3e \u0c09\u0c02\u0c26\u0c3f:**"),
    ("screen (educational)", "screen (\u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e)"),
    ("Primary symbol:", "\u0c2a\u0c4d\u0c30\u0c27\u0c3e\u0c28 symbol:"),
    ("\u2022 Last price:", "\u2022 \u0c1a\u0c3f\u0c35\u0c30\u0c3f \u0c27\u0c30:"),
    ("\u2022 Price context:", "\u2022 \u0c27\u0c30 \u0c38\u0c02\u0c26\u0c30\u0c4d\u0c2d\u0c02:"),
    ("\u2022 Price:", "\u2022 \u0c27\u0c30:"),
    ("\u2022 Last:", "\u2022 \u0c1a\u0c3f\u0c35\u0c30\u0c3f:"),
    ("\u2022 Open / High / Low:", "\u2022 \u0c13\u0c2a\u0c46\u0c28\u0c4d / \u0c39\u0c48 / \u0c32\u0c4b:"),
    ("\u2022 Prev close:", "\u2022 \u0c2e\u0c41\u0c28\u0c41\u0c2a\u0c1f\u0c3f close:"),
    ("\u2022 Volume:", "\u2022 Volume:"),
    ("\u2022 Support / Resistance:", "\u2022 \u0c38\u0c2a\u0c4b\u0c30\u0c4d\u0c1f\u0c4d / \u0c30\u0c46\u0c38\u0c3f\u0c38\u0c4d\u0c1f\u0c46\u0c28\u0c4d\u0c38\u0c4d:"),
    ("\u2022 Support:", "\u2022 \u0c38\u0c2a\u0c4b\u0c30\u0c4d\u0c1f\u0c4d:"),
    ("\u2022 Resistance:", "\u2022 \u0c30\u0c46\u0c38\u0c3f\u0c38\u0c4d\u0c1f\u0c46\u0c28\u0c4d\u0c38\u0c4d:"),
    ("\u2022 Stop idea:", "\u2022 \u0c38\u0c4d\u0c1f\u0c3e\u0c2a\u0c4d:"),
    ("\u2022 Target idea:", "\u2022 \u0c1f\u0c3e\u0c30\u0c4d\u0c17\u0c46\u0c1f\u0c4d:"),
    ("\u2022 Entry zone:", "\u2022 \u0c0e\u0c02\u0c1f\u0c4d\u0c30\u0c40 \u0c1c\u0c4b\u0c28\u0c4d:"),
    ("\u2022 Target 1:", "\u2022 \u0c1f\u0c3e\u0c30\u0c4d\u0c17\u0c46\u0c1f\u0c4d 1:"),
    ("\u2022 Target 2:", "\u2022 \u0c1f\u0c3e\u0c30\u0c4d\u0c17\u0c46\u0c1f\u0c4d 2:"),
    ("\u2022 Risk/reward:", "\u2022 \u0c30\u0c3f\u0c38\u0c4d\u0c15\u0c4d/\u0c30\u0c3f\u0c35\u0c3e\u0c30\u0c4d\u0c21\u0c4d:"),
    ("\u2022 suggested qty (risk budget):", "\u2022 \u0c38\u0c42\u0c1a\u0c3f\u0c02\u0c1a\u0c3f\u0c28 qty (risk budget):"),
    ("\u2022 Invalidation:", "\u2022 \u0c07\u0c28\u0c4d\u200c\u0c35\u0c3e\u0c32\u0c3f\u0c21\u0c47\u0c37\u0c28\u0c4d:"),
    ("\u2022 Meaning:", "\u2022 \u0c05\u0c30\u0c4d\u0c25\u0c02:"),
    ("\u2022 Trend:", "\u2022 Trend:"),
    ("\u2022 RSI read:", "\u2022 RSI:"),
    ("\u2022 RSI:", "\u2022 RSI:"),
    ("\u2022 Divergence:", "\u2022 Divergence:"),
    ("\u2022 Overall:", "\u2022 \u0c2e\u0c4a\u0c24\u0c4d\u0c24\u0c02:"),
    ("\u2022 Read:", "\u2022 \u0c35\u0c3f\u0c35\u0c30\u0c02:"),
    ("\u2022 Sector/tape cue:", "\u2022 sector/tape:"),
    ("\u2022 Market cap:", "\u2022 Market cap:"),
    ("\u2022 Dividend yield:", "\u2022 Dividend yield:"),
    ("\u2022 Fundamentals:", "\u2022 Fundamentals:"),
    ("\u2022 Tape:", "\u2022 tape:"),
    ("\u2022 Spot:", "\u2022 spot:"),
    ("\u2022 Stop:", "\u2022 \u0c38\u0c4d\u0c1f\u0c3e\u0c2a\u0c4d:"),
    (" | Target 1:", " | \u0c1f\u0c3e\u0c30\u0c4d\u0c17\u0c46\u0c1f\u0c4d 1:"),
    (" | Target 2:", " | \u0c1f\u0c3e\u0c30\u0c4d\u0c17\u0c46\u0c1f\u0c4d 2:"),
    (" | Target idea:", " | \u0c1f\u0c3e\u0c30\u0c4d\u0c17\u0c46\u0c1f\u0c4d:"),
    ("_Educational checklist \u2014 not personalized advice._",
     "_\u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e checklist \u2014 \u0c35\u0c4d\u0c2f\u0c15\u0c4d\u0c24\u0c3f\u0c17\u0c24 \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"),
    ("_Educational overview \u2014 not investment advice._",
     "_\u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"),
    ("_Educational framing \u2014 not a price prediction._",
     "_\u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u2014 \u0c27\u0c30 \u0c05\u0c02\u0c1a\u0c28\u0c3e \u0c15\u0c3e\u0c26\u0c41._"),
    ("_Educational beginner primer \u2014 not investment advice._",
     "_\u0c2c\u0c3f\u0c17\u0c3f\u0c28\u0c30\u0c4d primer \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"),
    ("_Educational only \u2014 not investment advice._",
     "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"),
    (
        "_Educational snapshot from live enrich + P0 math \u2014 not investment advice._",
        "_live data + \u0c32\u0c46\u0c15\u0c4d\u0c15\u0c32 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._",
    ),
    (
        "_Educational calculation from recent market history \u2014 not investment advice._",
        "_\u0c07\u0c1f\u0c40\u0c35\u0c32\u0c3f market history \u0c28\u0c41\u0c02\u0c1a\u0c3f \u0c32\u0c46\u0c15\u0c4d\u0c15 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._",
    ),
    (
        "Educational calculation from recent market history \u2014 not investment advice.",
        "\u0c07\u0c1f\u0c40\u0c35\u0c32\u0c3f market history \u0c28\u0c41\u0c02\u0c1a\u0c3f \u0c32\u0c46\u0c15\u0c4d\u0c15 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41.",
    ),
    ("_Grounded by BYSEL Indian Stock LLM + P0 math._", "_BYSEL._"),
    ("_Grounded by BYSEL Indian Stock LLM \u2014 educational only._", "_BYSEL._"),
    ("_Grounded by BYSEL Indian Stock LLM._", "_BYSEL._"),
    ("_Grounded by BYSEL._", "_BYSEL._"),
    ("not investment advice", "\u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41"),
    ("Educational only", "\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02"),
    ("how much now", "\u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c0e\u0c02\u0c24"),
    ("Staged adds on dips \u2014 avoid chasing strength (paper)",
     "dips\u0c32\u0c4b \u0c26\u0c36\u0c32\u0c35\u0c3e\u0c30\u0c40 \u0c15\u0c4a\u0c28\u0c02\u0c21\u0c3f \u2014 strength chase \u0c35\u0c26\u0c4d\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)"),
    ("Lighten / reduce on strength \u2014 not a full exit (paper)",
     "\u0c2a\u0c48\u0c15\u0c3f \u0c35\u0c1a\u0c4d\u0c1a\u0c3f\u0c28\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 \u0c24\u0c17\u0c4d\u0c17\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f \u2014 full exit \u0c15\u0c3e\u0c26\u0c41 (\u0c2a\u0c47\u0c2a\u0c30\u0c4d)"),
    ("BUY = fresh long \u00b7 ACCUMULATE = staged adds on dips \u00b7 HOLD = no clear edge \u00b7 TRIM = lighten/reduce on strength \u00b7 SELL = exit/avoid \u00b7 WAIT = skip until setup improves",
     "BUY = \u0c15\u0c4a\u0c24\u0c4d\u0c24 long \u00b7 ACCUMULATE = dips\u0c32\u0c4b \u00b7 HOLD = edge \u0c32\u0c47\u0c26\u0c41 \u00b7 TRIM = \u0c24\u0c17\u0c4d\u0c17\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f \u00b7 SELL = \u0c05\u0c2e\u0c4d\u0c2e\u0c02\u0c21\u0c3f \u00b7 WAIT = setup \u0c15\u0c4b\u0c38\u0c02 \u0c35\u0c47\u0c1a\u0c3f"),
    ("\u22643% capital risk / trade \u00b7 \u22645% total open risk \u00b7 prefer ~7%+ favorable room on winners (paper checklist)",
     "\u22643% capital risk / trade \u00b7 \u22645% open risk \u00b7 winners\u0c2a\u0c48\u0c28 ~7%+ room (\u0c2a\u0c47\u0c2a\u0c30\u0c4d checklist)"),
    ("Risk frame (3-5-7):", "Risk frame (3-5-7):"),
    ("soft momentum", "momentum \u0c2c\u0c32\u0c39\u0c40\u0c28\u0c02"),
    ("underperforming", "\u0c35\u0c46\u0c28\u0c41\u0c15 \u0c2a\u0c28\u0c3f\u0c24\u0c40\u0c30\u0c41"),
    ("MA bias down", "MA bias \u0c15\u0c3f\u0c02\u0c26\u0c3f\u0c15\u0c3f"),
    ("MA bias up", "MA bias \u0c2a\u0c48\u0c15\u0c3f"),
    ("Supertrend bullish", "Supertrend \u0c2c\u0c41\u0c32\u0c3f\u0c37\u0c4d"),
    ("Supertrend bearish", "Supertrend \u0c2c\u0c47\u0c30\u0c3f\u0c37\u0c4d"),
    ("invalidates short/trim thesis", "short/trim thesis \u0c30\u0c26\u0c4d\u0c26\u0c41"),
    ("near lower band", "lower band \u0c26\u0c17\u0c4d\u0c17\u0c30"),
    ("bullish momentum", "bullish momentum"),
    ("\u2022 Volume:", "\u2022 Volume:"),
    ("NSE\u2019s flagship index of 50 large liquid companies. Used as the main market benchmark and as an F&O underlier (lot size set by exchange).",
     "NIFTY 50 NSE\u0c32\u0c4b 50 \u0c2a\u0c46\u0c26\u0c4d\u0c26 \u0c32\u0c3f\u0c15\u0c4d\u0c35\u0c3f\u0c21\u0c4d companies \u0c07\u0c02\u0c21\u0c46\u0c15\u0c4d\u0c38\u0c4d. market benchmark + F&O underlier."),
    ("NSE's flagship index of 50 large liquid companies. Used as the main market benchmark and as an F&O underlier (lot size set by exchange).",
     "NIFTY 50 NSE\u0c32\u0c4b 50 \u0c2a\u0c46\u0c26\u0c4d\u0c26 \u0c32\u0c3f\u0c15\u0c4d\u0c35\u0c3f\u0c21\u0c4d companies \u0c07\u0c02\u0c21\u0c46\u0c15\u0c4d\u0c38\u0c4d. market benchmark + F&O underlier."),
    ("In BYSEL, the index symbol is often referenced as NIFTY50.",
     "BYSEL\u0c32\u0c4b symbol: NIFTY50."),
    ("suggested qty (risk budget):", "\u0c38\u0c42\u0c1a\u0c3f\u0c02\u0c1a\u0c3f\u0c28 qty:"),
    ("Close below stop", "stop \u0c15\u0c3f\u0c02\u0c26 close"),
    ("Close above ", "\u0c2a\u0c48\u0c28 close "),
    ("paper practice only", "\u0c2a\u0c47\u0c2a\u0c30\u0c4d \u0c2a\u0c4d\u0c30\u0c3e\u0c15\u0c4d\u0c1f\u0c40\u0c38\u0c4d \u0c2e\u0c3e\u0c24\u0c4d\u0c30\u0c2e\u0c47"),
    ("Use sentiment with price structure, levels, and risk \u2014 never alone.",
     "sentiment\u0c28\u0c3f \u0c27\u0c30 / levels / risk\u0c24\u0c4b \u0c15\u0c32\u0c3f\u0c2a\u0c3f \u0c1a\u0c42\u0c21\u0c02\u0c21\u0c3f \u2014 \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c3f\u0c17\u0c3e \u0c15\u0c3e\u0c26\u0c41."),
    (
        "Multi-factor educational sentiment (news + technicals + volume + RS). Not a prediction or investment advice.",
        "\u0c35\u0c3e\u0c30\u0c4d\u0c24\u0c32\u0c41 + technicals + volume + RS \u2014 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e. \u0c05\u0c02\u0c1a\u0c28\u0c3e / \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41.",
    ),
    (
        "This response is informational, \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41. Use risk controls and verify with live NSE/BSE data before trading.",
        "\u0c07\u0c26\u0c3f \u0c38\u0c2e\u0c3e\u0c1a\u0c3e\u0c30\u0c02 \u0c2e\u0c3e\u0c24\u0c4d\u0c30\u0c2e\u0c47 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41. live NSE/BSE\u0c24\u0c4b \u0c38\u0c30\u0c3f\u0c2a\u0c3e\u0c26\u0c3f\u0c1a\u0c3e\u0c32\u0c3f.",
    ),
    (
        "This response is informational and should be validated against live market data.",
        "\u0c07\u0c26\u0c3f \u0c38\u0c2e\u0c3e\u0c1a\u0c3e\u0c30\u0c02 \u0c2e\u0c3e\u0c24\u0c4d\u0c30\u0c2e\u0c47 \u2014 live data\u0c24\u0c4b \u0c38\u0c30\u0c3f\u0c2a\u0c3e\u0c26\u0c3f\u0c1a\u0c3e\u0c32\u0c3f.",
    ),
    (
        "This response is informational, not investment advice. Use risk controls and verify with live NSE/BSE data before trading.",
        "\u0c07\u0c26\u0c3f \u0c38\u0c2e\u0c3e\u0c1a\u0c3e\u0c30\u0c02 \u0c2e\u0c3e\u0c24\u0c4d\u0c30\u0c2e\u0c47 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41.",
    ),
    (
        "\u26a0\ufe0f Disclaimer: Educational / informational only \u2014 validate with live NSE/BSE data before decisions.",
        "\u26a0\ufe0f \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e / \u0c38\u0c2e\u0c3e\u0c1a\u0c3e\u0c30\u0c02 \u0c2e\u0c3e\u0c24\u0c4d\u0c30\u0c2e\u0c47 \u2014 live NSE/BSE\u0c24\u0c4b \u0c38\u0c30\u0c3f\u0c2a\u0c3e\u0c26\u0c3f\u0c1a\u0c3e\u0c32\u0c3f.",
    ),
    ("Last \u20b9", "\u0c1a\u0c3f\u0c35\u0c30\u0c3f \u20b9"),
    ("trend bearish", "trend \u0c2c\u0c47\u0c30\u0c3f\u0c37\u0c4d"),
    ("trend neutral", "trend \u0c24\u0c1f\u0c38\u0c4d\u0c25\u0c02"),
    ("trend bullish", "trend \u0c2c\u0c41\u0c32\u0c3f\u0c37\u0c4d"),
    ("Do not pick solely on lower P/E \u2014 prefer growth durability + cleaner technicals for your horizon.",
     "\u0c24\u0c15\u0c4d\u0c15\u0c41\u0c35 P/E \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c3f\u0c17\u0c3e \u0c0e\u0c02\u0c1a\u0c41\u0c15\u0c4b\u0c35\u0c26\u0c4d\u0c26\u0c41 \u2014 growth + technicals\u0c24\u0c4b \u0c1a\u0c42\u0c21\u0c02\u0c21\u0c3f."),
    ("TCS currently screens cleaner on available RSI/trend/P/E/ROE cues \u2014 still verify guidance and valuation vs history.",
     "TCS \u0c07\u0c2a\u0c4d\u0c2a\u0c41\u0c21\u0c41 RSI / trend / P/E / ROE\u0c32\u0c4b \u0c15\u0c4d\u0c32\u0c40\u0c28\u0c30\u0c4d\u0c17\u0c3e \u0c15\u0c28\u0c2a\u0c21\u0c41\u0c24\u0c41\u0c02\u0c26\u0c3f \u2014 guidance / valuation \u0c38\u0c30\u0c3f\u0c2a\u0c3e\u0c26\u0c3f\u0c1a\u0c3e\u0c32\u0c3f."),
    ("\u0c32\u0c48\u0c35\u0c4d \u0c15\u0c4b\u0c1f\u0c4d \u0c38\u0c4d\u0c28\u0c3e\u0c2a\u0c4d\u0c37\u0c3e\u0c1f\u0c4d-\u0c0e\u0c21\u0c4d\u0c2f\u0c41\u0c15\u0c47\u0c37\u0c28\u0c32\u0c4d only._",
     "_\u0c32\u0c48\u0c35\u0c4d \u0c15\u0c4b\u0c1f\u0c4d \u2014 \u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e._"),
)

# Composer uses ASCII hyphen-minus in many stance lines; keep both dash forms.
_BODY_TE = _BODY_TE + tuple(
    (src.replace("\u2014", "\u2014"), dest) for src, dest in ()
)

_ASCII_DASH_BODY = tuple(
    (src.replace("\u2014", "-"), dest.replace("\u2014", "-"))
    for src, dest in _BODY_TE
    if "\u2014" in src
)

_REGEX_TE: tuple[tuple[str, str], ...] = (
    (r"RSI ([0-9]+(?:\.[0-9]+)?) is elevated \(overbought zone\)",
     "RSI \\1 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35\u0c17\u0c3e \u0c09\u0c02\u0c26\u0c3f (overbought)"),
    (r"RSI ([0-9]+(?:\.[0-9]+)?) is washed out \(oversold zone\)",
     "RSI \\1 \u0c24\u0c17\u0c4d\u0c17\u0c3f\u0c02\u0c26\u0c3f (oversold)"),
    (r"RSI ([0-9]+(?:\.[0-9]+)?) shows bullish momentum building",
     "RSI \\1 bullish momentum \u0c1a\u0c42\u0c2a\u0c3f\u0c38\u0c4d\u0c24\u0c4b\u0c02\u0c26\u0c3f"),
    (r"RSI ([0-9]+(?:\.[0-9]+)?) shows soft momentum",
     "RSI \\1 momentum \u0c2c\u0c32\u0c39\u0c40\u0c28\u0c02"),
    (r"RSI ([0-9]+(?:\.[0-9]+)?) is mid-range",
     "RSI \\1 \u0c2e\u0c27\u0c4d\u0c2f\u0c38\u0c4d\u0c25\u0c02\u0c32\u0c4b \u0c09\u0c02\u0c26\u0c3f"),
    (r"MACD histogram positive \(([^)]+)\)", "MACD histogram positive (\\1)"),
    (r"MACD histogram negative \(([^)]+)\)", "MACD histogram negative (\\1)"),
    (r"P/E ([0-9]+(?:\.[0-9]+)?) is rich vs typical large-cap bands",
     "P/E \\1 large-cap \u0c15\u0c02\u0c1f\u0c47 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35"),
    (r"P/E ([0-9]+(?:\.[0-9]+)?) is comparatively modest",
     "P/E \\1 \u0c38\u0c3e\u0c2a\u0c47\u0c15\u0c4d\u0c37\u0c02\u0c17\u0c3e \u0c24\u0c15\u0c4d\u0c15\u0c41\u0c35"),
    (r"Wilder RSI is ([0-9]+(?:\.[0-9]+)?)", "Wilder RSI \\1"),
    (r"score ([0-9]+(?:\.[0-9]+)?), confidence ",
     "\u0c38\u0c4d\u0c15\u0c4b\u0c30\u0c41 \\1, confidence "),
    (r"\(score ([+\-]?[0-9]+(?:\.[0-9]+)?)", "(\u0c38\u0c4d\u0c15\u0c4b\u0c30\u0c41 \\1"),
    (r"(?:pandas-)?ta RSI\((\d+)\) is ([0-9.]+) from live market history for ([A-Z0-9.&-]+)\.?",
     "\\3 RSI(\\1) = \\2 (live)"),
    (r"Consider buying (\d+) shares of ([A-Z0-9.&-]+)",
     "\\2 \\1 \u0c37\u0c47\u0c30\u0c4d\u0c32\u0c41 \u0c15\u0c4a\u0c28\u0c21\u0c3e\u0c02 \u0c06\u0c32\u0c4b\u0c1a\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f"),
    (r"Consider selling (\d+) shares of ([A-Z0-9.&-]+)",
     "\\2 \\1 \u0c37\u0c47\u0c30\u0c4d\u0c32\u0c41 \u0c05\u0c2e\u0c4d\u0c2e\u0c21\u0c3e\u0c02 \u0c06\u0c32\u0c4b\u0c1a\u0c3f\u0c02\u0c1a\u0c02\u0c21\u0c3f"),
    (r'Ask \u201cShould I buy ([A-Z0-9.&-]+)\?\u201d paper entry / stop / targets',
     '\\1 paper entry / stop / targets \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c02\u0c1f\u0c47 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f'),
    (r'Ask "Should I buy ([A-Z0-9.&-]+)\?" paper entry / stop / targets',
     '\\1 paper entry / stop / targets \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c02\u0c1f\u0c47 \u0c05\u0c21\u0c17\u0c02\u0c21\u0c3f'),
    (r"Supports: ", "\u0c38\u0c2a\u0c4b\u0c30\u0c4d\u0c1f\u0c4d: "),
    (r"Drags: ", "\u0c32\u0c3e\u0c17\u0c41\u0c26\u0c41: "),
    (r"Earnings yield ([0-9.]+)% supportive",
     "Earnings yield \\1% \u0c38\u0c2a\u0c4b\u0c30\u0c4d\u0c1f\u0c3f\u0c35\u0c4d"),
    (r"MACD hist ([+\-]?[0-9.]+) \(bearish\)", "MACD hist \\1 (bearish)"),
    (r"MACD hist ([+\-]?[0-9.]+) \(bullish\)", "MACD hist \\1 (bullish)"),
    (r"20d ROC ([+\-]?[0-9.]+)%", "20d ROC \\1%"),
    (r"\| Stock \| Price \| RSI \| Trend \| P/E \| P/B \| ROE% \| Mcap \| Notes \|",
     "| Stock | \u0c27\u0c30 | RSI | Trend | P/E | P/B | ROE% | Mcap | Notes |"),
)

_LITERACY_TE: dict[str, str] = {
    "rsi": (
        "**RSI (Relative Strength Index) \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "RSI 0\u2013100 \u0c2e\u0c27\u0c4d\u0c2f \u0c24\u0c3f\u0c30\u0c3f\u0c17\u0c47 momentum oscillator.\n"
        "\u2022 70 \u0c2a\u0c48\u0c28: overbought \u2014 \u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c33\u0c4d\u0c32\u0c41 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35\u0c48\u0c28 zone\n"
        "\u2022 30 \u0c15\u0c3f\u0c02\u0c26: oversold \u2014 \u0c05\u0c2e\u0c4d\u0c2e\u0c15\u0c3e\u0c32\u0c41 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35\u0c48\u0c28 zone\n"
        "\u2022 trend market\u0c32\u0c4b RSI \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35\u0c38\u0c47\u0c2a\u0c41 70 \u0c2a\u0c48\u0c28 \u0c09\u0c02\u0c21\u0c35\u0c1a\u0c4d\u0c1a\u0c41 \u2014 "
        "\u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c47 \u0c1a\u0c42\u0c38\u0c3f short \u0c35\u0c26\u0c4d\u0c26\u0c41.\n\n"
        "`RSI = 100 \u2212 (100 / (1 + RS))`\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "macd": (
        "**MACD \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "MACD \u0c30\u0c46\u0c02\u0c21\u0c41 EMA\u0c32 \u0c24\u0c47\u0c21\u0c3e \u2014 trend + momentum.\n"
        "\u2022 Histogram positive: bullish momentum\n"
        "\u2022 Histogram negative: bearish momentum\n"
        "\u2022 signal line cross\u0c28\u0c41 \u0c27\u0c30 \u0c28\u0c3f\u0c30\u0c4d\u0c2e\u0c3e\u0c23\u0c02\u0c24\u0c4b confirm \u0c1a\u0c47\u0c2f\u0c02\u0c21\u0c3f.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "pe": (
        "**P/E ratio \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "P/E = share \u0c27\u0c30 \u00f7 EPS.\n"
        "\u2022 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35 P/E: market \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35 growth \u0c06\u0c36\u0c3f\u0c38\u0c4d\u0c24\u0c4b\u0c02\u0c26\u0c3f\n"
        "\u2022 \u0c24\u0c15\u0c4d\u0c15\u0c41\u0c35 P/E: \u0c1a\u0c4c\u0c15 \u0c15\u0c3e\u0c35\u0c1a\u0c4d\u0c1a\u0c41 \u0c32\u0c47\u0c26\u0c3e risk \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35\u0c17\u0c3e \u0c09\u0c02\u0c21\u0c35\u0c1a\u0c4d\u0c1a\u0c41\n"
        "sector peers, growth\u0c24\u0c4b \u0c2a\u0c4b\u0c32\u0c4d\u0c1a\u0c02\u0c21\u0c3f \u2014 \u0c24\u0c15\u0c4d\u0c15\u0c41\u0c35 P/E \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c47 \u0c1a\u0c4c\u0c15 \u0c15\u0c3e\u0c26\u0c41.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "sma": (
        "**SMA / EMA \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "moving average \u0c27\u0c30 trend\u0c28\u0c41 smooth \u0c1a\u0c47\u0c38\u0c4d\u0c24\u0c41\u0c02\u0c26\u0c3f.\n"
        "\u2022 SMA: simple average\n"
        "\u2022 EMA: \u0c07\u0c1f\u0c40\u0c35\u0c32\u0c3f \u0c27\u0c30\u0c15\u0c41 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35 weight\n"
        "\u0c27\u0c30 MA \u0c2a\u0c48\u0c28 \u0c09\u0c02\u0c1f\u0c47 bullish, \u0c15\u0c3f\u0c02\u0c26 \u0c09\u0c02\u0c1f\u0c47 bearish \u2014 "
        "\u0c15\u0c3e\u0c28\u0c40 \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c47 trade \u0c15\u0c3e\u0c26\u0c41.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "support": (
        "**Support / Resistance \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "\u2022 Support: \u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c33\u0c4d\u0c32\u0c41 \u0c35\u0c1a\u0c4d\u0c1a\u0c3f \u0c27\u0c30 \u0c06\u0c17\u0c47 \u0c38\u0c4d\u0c25\u0c3e\u0c2f\u0c3f\n"
        "\u2022 Resistance: \u0c05\u0c2e\u0c4d\u0c2e\u0c15\u0c3e\u0c32\u0c41 \u0c35\u0c1a\u0c4d\u0c1a\u0c3f \u0c27\u0c30 \u0c06\u0c17\u0c47 \u0c38\u0c4d\u0c25\u0c3e\u0c2f\u0c3f\n"
        "break \u0c05\u0c2f\u0c3f\u0c24\u0c47 \u0c06 \u0c38\u0c4d\u0c25\u0c3e\u0c2f\u0c3f roll \u0c05\u0c35\u0c41\u0c24\u0c41\u0c02\u0c26\u0c3f. "
        "stop\u0c28\u0c41 \u0c08 levels \u0c26\u0c17\u0c4d\u0c17\u0c30 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c21\u0c02 \u0c2a\u0c47\u0c2a\u0c30\u0c4d practice.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "sip": (
        "**SIP \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "SIP \u0c05\u0c02\u0c1f\u0c47 \u0c2a\u0c4d\u0c30\u0c24\u0c3f \u0c28\u0c46\u0c32\u0c3e \u0c2b\u0c3f\u0c15\u0c4d\u0c38\u0c4d\u0c21\u0c4d \u0c2e\u0c4a\u0c24\u0c4d\u0c24\u0c02 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u2014 rupee cost averaging.\n"
        "market timing \u0c15\u0c02\u0c1f\u0c47 \u0c15\u0c4d\u0c30\u0c2e\u0c02 \u0c2e\u0c41\u0c16\u0c4d\u0c2f\u0c02. return guarantee \u0c15\u0c3e\u0c26\u0c41.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "demat": (
        "**Demat \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "Demat account\u0c32\u0c4b shares electronic\u0c17\u0c3e \u0c09\u0c02\u0c1f\u0c3e\u0c2f\u0c3f (NSDL/CDSL).\n"
        "trading\u0c15\u0c41 broker + demat + bank link \u0c15\u0c3e\u0c35\u0c3e\u0c32\u0c3f. KYC \u0c2a\u0c42\u0c30\u0c4d\u0c24\u0c3f \u0c1a\u0c47\u0c2f\u0c3e\u0c32\u0c3f.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "cagr": (
        "**CAGR \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "CAGR = \u0c38\u0c02\u0c35\u0c24\u0c4d\u0c38\u0c30\u0c3e\u0c28\u0c3f\u0c15\u0c3f \u0c38\u0c2e\u0c3e\u0c28 \u0c30\u0c47\u0c1f\u0c41\u0c24\u0c4b \u0c2a\u0c46\u0c30\u0c3f\u0c17\u0c3f\u0c28\u0c1f\u0c4d\u0c1f\u0c41 return.\n"
        "`CAGR = (end / start) ^ (1 / years) \u2212 1`\n"
        "\u0c17\u0c24 return \u0c2d\u0c35\u0c3f\u0c37\u0c4d\u0c2f\u0c24\u0c4d\u0c24\u0c41 guarantee \u0c15\u0c3e\u0c26\u0c41.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "beta": (
        "**Beta \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "Beta stock Nifty\u0c32\u0c3e\u0c02\u0c1f\u0c3f index\u0c24\u0c4b \u0c0e\u0c02\u0c24 \u0c15\u0c26\u0c41\u0c32\u0c41\u0c24\u0c41\u0c02\u0c26\u0c4b \u0c1a\u0c46\u0c2c\u0c41\u0c24\u0c41\u0c02\u0c26\u0c3f.\n"
        "\u2022 Beta > 1: index \u0c15\u0c02\u0c1f\u0c47 \u0c0e\u0c15\u0c4d\u0c15\u0c41\u0c35 volatility\n"
        "\u2022 Beta < 1: \u0c24\u0c15\u0c4d\u0c15\u0c41\u0c35 volatility\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "sentiment": (
        "**Market sentiment \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "sentiment \u0c05\u0c02\u0c1f\u0c47 news, \u0c27\u0c30 \u0c1a\u0c30\u0c4d\u0c2f, \u0c2a\u0c3e\u0c32\u0c4d\u0c17\u0c4a\u0c28\u0c47\u0c35\u0c3e\u0c30\u0c3f mood.\n"
        "bullish = \u0c15\u0c4a\u0c28\u0c41\u0c17\u0c4b\u0c32\u0c41 \u0c12\u0c24\u0c4d\u0c24\u0c3f\u0c21\u0c3f; bearish = \u0c05\u0c2e\u0c4d\u0c2e\u0c15\u0c02 \u0c12\u0c24\u0c4d\u0c24\u0c3f\u0c21\u0c3f.\n"
        "sentiment \u0c12\u0c15\u0c4d\u0c15\u0c1f\u0c47 trade call \u0c15\u0c3e\u0c26\u0c41 \u2014 \u0c27\u0c30, risk\u0c24\u0c4b \u0c15\u0c32\u0c3f\u0c2a\u0c3f \u0c1a\u0c42\u0c21\u0c02\u0c21\u0c3f.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "market": (
        "**\u0c2d\u0c3e\u0c30\u0c24\u0c4d stock market \u0c0e\u0c32\u0c3e \u0c2a\u0c28\u0c3f\u0c1a\u0c47\u0c38\u0c4d\u0c24\u0c41\u0c02\u0c26\u0c3f?**\n\n"
        "1. IPO (primary) \u2014 company capital \u0c38\u0c47\u0c15\u0c30\u0c23\n"
        "2. NSE/BSE\u0c32\u0c4b listing (secondary trading)\n"
        "3. broker \u0c26\u0c4d\u0c35\u0c3e\u0c30\u0c3e order\n"
        "4. exchange buy/sell match \u0c1a\u0c47\u0c38\u0c4d\u0c24\u0c41\u0c02\u0c26\u0c3f\n"
        "5. T+1 settlement demat / funds\u0c32\u0c4b\u0c15\u0c3f\n\n"
        "regulator: SEBI. depositories: NSDL/CDSL.\n"
        "_\u0c2c\u0c3f\u0c17\u0c3f\u0c28\u0c30\u0c4d primer \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "nifty": (
        "**NIFTY 50 \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "NIFTY 50 NSE\u0c32\u0c4b 50 \u0c2a\u0c46\u0c26\u0c4d\u0c26 \u0c32\u0c3f\u0c15\u0c4d\u0c35\u0c3f\u0c21\u0c4d companies \u0c07\u0c02\u0c21\u0c46\u0c15\u0c4d\u0c38\u0c4d.\n"
        "\u0c2e\u0c3e\u0c30\u0c4d\u0c15\u0c46\u0c1f\u0c4d benchmark + F&O underlier.\n"
        "BYSEL\u0c32\u0c4b symbol: NIFTY50.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
    "sensex": (
        "**SENSEX \u0c05\u0c02\u0c1f\u0c47 \u0c0f\u0c2e\u0c3f\u0c1f\u0c3f?**\n\n"
        "SENSEX BSE\u0c32\u0c4b 30 \u0c2a\u0c46\u0c26\u0c4d\u0c26 companies \u0c07\u0c02\u0c21\u0c46\u0c15\u0c4d\u0c38\u0c4d.\n"
        "BYSEL\u0c32\u0c4b symbol: SENSEX.\n\n"
        "_\u0c15\u0c47\u0c35\u0c32\u0c02 \u0c35\u0c3f\u0c26\u0c4d\u0c2f\u0c3e \u0c2a\u0c4d\u0c30\u0c2f\u0c4b\u0c1c\u0c28\u0c02 \u2014 \u0c2a\u0c46\u0c1f\u0c4d\u0c1f\u0c41\u0c2c\u0c21\u0c3f \u0c38\u0c32\u0c39\u0c3e \u0c15\u0c3e\u0c26\u0c41._"
    ),
}

_LITERACY_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("rsi", ("rsi",)),
    ("macd", ("macd",)),
    ("pe", ("p/e", "pe ratio")),
    ("sma", ("sma", "ema", "moving average")),
    ("support", ("support", "resistance", "\u0c38\u0c2a\u0c4b\u0c30\u0c4d\u0c1f\u0c4d", "\u0c30\u0c46\u0c38\u0c3f\u0c38\u0c4d\u0c1f\u0c46\u0c28\u0c4d\u0c38\u0c4d")),
    ("sip", ("sip",)),
    ("demat", ("demat", "\u0c21\u0c40\u0c2e\u0c4d\u0c2f\u0c3e\u0c1f\u0c4d")),
    ("cagr", ("cagr",)),
    ("beta", ("beta", "\u0c2c\u0c40\u0c1f\u0c3e")),
    ("sentiment", ("sentiment", "\u0c38\u0c46\u0c02\u0c1f\u0c3f\u0c2e\u0c46\u0c02\u0c1f\u0c4d")),
    ("market", ("stock market", "share market", "how does the stock")),
    ("nifty", ("nifty", "\u0c28\u0c3f\u0c2b\u0c4d\u0c1f\u0c40")),
    ("sensex", ("sensex", "\u0c38\u0c46\u0c28\u0c4d\u0c38\u0c46\u0c15\u0c4d\u0c38\u0c4d")),
)

_DEFINITIONAL_RE = re.compile(
    r"(\u0c05\u0c02\u0c1f\u0c47|\u0c0f\u0c2e\u0c3f\u0c1f\u0c3f|\u0c05\u0c30\u0c4d\u0c25\u0c02|"
    r"\bwhat is\b|\bmeaning\b|\bdefine\b|\bdefinition\b|\bemiti\b|\benti\b|\bante\b)",
    re.I,
)


def _already_localized(answer: str) -> bool:
    return _LOCALIZED_MARK in (answer or "")[:180]


_STANCE_LINE_RE = (
    r"\*\*(?:Action|Direct answer|\u0c1a\u0c30\u0c4d\u0c2f|\u0c28\u0c47\u0c30\u0c41\u0c17\u0c3e \u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02):\*\*"
    r".{0,80}?\b(ACCUMULATE|BUY|SELL|HOLD|WAIT|TRIM|AVOID)\b"
)
_FALSE_BUY_RE = re.compile(
    r"(?i)paper buy plan|should i buy|buy/sell|buy plan|buy \?",
)


def _stance_from_answer(answer: str) -> str | None:
    text = answer or ""
    match = re.search(_STANCE_LINE_RE, text, flags=re.I | re.S)
    if match:
        return match.group(1).upper()
    head = _FALSE_BUY_RE.sub(" ", text[:700])
    match = re.search(r"\b(ACCUMULATE|SELL|HOLD|WAIT|TRIM|AVOID|BUY)\b", head, flags=re.I)
    if not match:
        return None
    return match.group(1).upper()


def _is_definitional(query: str) -> bool:
    return bool(_DEFINITIONAL_RE.search(query or ""))


def _literacy_topic(query: str) -> str | None:
    low = (query or "").lower()
    for topic, aliases in _LITERACY_ALIASES:
        if any(alias in low for alias in aliases):
            return topic
    return None


def telugu_literacy_answer(query: str) -> str | None:
    if not _is_definitional(query):
        return None
    topic = _literacy_topic(query)
    if not topic:
        return None
    return _LITERACY_TE.get(topic)


def _extract_numeric_lines(answer: str) -> str:
    kept: list[str] = []
    for line in (answer or "").splitlines():
        if re.search(r"(₹|RSI\s*[:=]|P/?E\s*[:=]|Wilder RSI)", line):
            kept.append(line.strip())
        if len(kept) >= 6:
            break
    return "\n".join(kept)


_DROP_ASK_RE = re.compile(r"(?im)^\*\*(?:Your ask|\u0c2e\u0c40 \u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28):\*\*.*\n?")


def _drop_internal_ask(answer: str) -> str:
    """Do not show the English rewrite ('RELIANCE should I buy ?') to Telugu users."""
    return _DROP_ASK_RE.sub("", answer or "")


def _translate_structured(answer: str) -> str:
    text = _drop_internal_ask(answer or "").replace("\u2013", "\u2014").replace(" - ", " \u2014 ")
    for pat, repl in _HEADER_TE:
        text = re.sub(pat, repl, text)
    for src, dest in _BODY_TE + _ASCII_DASH_BODY:
        if src in text:
            text = text.replace(src, dest)
    # Composer also uses "HOLD / wait" without the em dash phrase in some cards.
    text = text.replace(
        "HOLD / wait",
        "\u0c39\u0c4b\u0c32\u0c4d\u0c21\u0c4d / \u0c35\u0c47\u0c1a\u0c3f \u0c09\u0c02\u0c21\u0c02\u0c21\u0c3f",
    )
    for pat, repl in _REGEX_TE:
        try:
            text = re.sub(pat, repl, text)
        except re.PatternError:
            continue
    return apply_telugu_leftovers(text)


def apply_telugu_leftovers(text: str) -> str:
    """Second pass for leftover English that phrase-table + NMT still miss."""
    if not text:
        return text
    out = text
    for pat, repl in _HEADER_TE + _REGEX_TE:
        try:
            out = re.sub(pat, repl, out)
        except re.PatternError:
            continue
    for src, dest in _BODY_TE + _ASCII_DASH_BODY:
        if src and src in out and src != dest:
            out = out.replace(src, dest)
    out = re.sub(r"only\._", "._", out)
    out = re.sub(r"call\._", "._", out)
    return out


def _with_lead(answer_for_stance: str, body: str) -> str:
    if body.startswith(f"**{_LOCALIZED_MARK}:**"):
        return body
    stance = _stance_from_answer(answer_for_stance)
    stance_te = _STANCE_TE.get(stance or "", "")
    if stance_te:
        lead = (
            f"**{_LOCALIZED_MARK}:** {stance_te} \u2014 "
            "\u0c35\u0c3f\u0c35\u0c30\u0c3e\u0c32\u0c41 \u0c15\u0c3f\u0c02\u0c26 \u0c09\u0c28\u0c4d\u0c28\u0c3e\u0c2f\u0c3f.\n\n"
        )
    else:
        lead = (
            f"**{_LOCALIZED_MARK}:** "
            "\u0c2e\u0c40 \u0c2a\u0c4d\u0c30\u0c36\u0c4d\u0c28\u0c15\u0c41 \u0c38\u0c2e\u0c3e\u0c27\u0c3e\u0c28\u0c02 \u0c15\u0c3f\u0c02\u0c26 \u0c09\u0c02\u0c26\u0c3f.\n\n"
        )
    return lead + body


def render_telugu_answer(query: str, answer: str) -> str:
    """Render an ISM English answer in Telugu for a Telugu / Tenglish ask."""
    if not answer:
        return answer
    if _already_localized(answer):
        return answer
    literacy = telugu_literacy_answer(query)
    if literacy:
        extras = _extract_numeric_lines(answer)
        body = literacy
        if extras:
            body = (
                f"{literacy}\n\n"
                "**live / \u0c32\u0c46\u0c15\u0c4d\u0c15\u0c32\u0c41:**\n"
                f"{extras}"
            )
        return _with_lead(answer, body)
    structured = _translate_structured(answer)
    try:
        from .indic_trans import translate_leftover_english

        structured = translate_leftover_english(structured)
    except Exception:
        pass
    return _with_lead(answer, structured)


def apply_response_language(query: str, result: dict | None) -> dict | None:
    """Localize a chat payload when the original user text is Telugu."""
    if not result:
        return result
    answer = result.get("answer")
    if not answer:
        return result
    try:
        from .query_language import is_telugu_query
    except Exception:
        return result
    if not is_telugu_query(query):
        return result
    localized = render_telugu_answer(query, str(answer))
    if localized == answer:
        return result
    out = dict(result)
    out["answer"] = localized
    return out
