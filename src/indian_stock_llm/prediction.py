"""Multi-horizon prediction head for the Indian Stock Market assistant.

Produces calibrated, risk-aware directional signals (intraday, swing, medium-term)
by scoring grounded knowledge-base context items for bullish/bearish signals.
No guaranteed-return claims are made; every signal includes an uncertainty note.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeItem

# Content signals used for directional scoring
_BULLISH_TAGS = frozenset({"earnings", "guidance", "momentum", "fundamental", "growth"})
_BEARISH_TAGS = frozenset({"risk", "uncertainty", "volatility", "regulation", "compliance"})
_BULLISH_CONTENT = (
    "strong earnings",
    "deal win",
    "beat estimate",
    "guidance upgrade",
    "momentum",
    "sector tailwind",
    "positive",
    "growth",
    "recovery",
)
_BEARISH_CONTENT = (
    "miss estimate",
    "guidance cut",
    "margin pressure",
    "headwind",
    "volatility",
    "uncertainty",
    "slowdown",
    "decline",
    "risk",
)

_BASE_PROB = 0.50
_SIGNAL_STEP = 0.05
_MAX_PROB = 0.75
_MIN_PROB = 0.25


@dataclass(frozen=True)
class HorizonSignal:
    """Directional signal for a single time horizon."""

    direction: str  # "bullish" | "bearish" | "neutral"
    probability: float  # calibrated probability [0, 1]
    rationale: str  # human-readable explanation with uncertainty note


@dataclass(frozen=True)
class PredictionSignals:
    """Multi-horizon prediction output produced by PredictionEngine."""

    intraday: HorizonSignal
    swing: HorizonSignal
    medium_term: HorizonSignal
    key_signals: tuple[str, ...]
    overall_confidence: float


class PredictionEngine:
    """Heuristic prediction engine that scores grounded context for directional signals.

    All probabilities are calibrated estimates, not guaranteed outcomes.
    """

    def _score_items(
        self, context_items: list[KnowledgeItem]
    ) -> tuple[int, int, list[str]]:
        bullish = 0
        bearish = 0
        signals: list[str] = []
        for item in context_items:
            tags = set(item.tags)
            content_lower = item.content.lower()
            b_tag = len(tags & _BULLISH_TAGS)
            r_tag = len(tags & _BEARISH_TAGS)
            b_content = sum(1 for s in _BULLISH_CONTENT if s in content_lower)
            r_content = sum(1 for s in _BEARISH_CONTENT if s in content_lower)
            if b_tag + b_content > r_tag + r_content:
                bullish += 1
                signals.append(f"Bullish: {item.title}")
            elif r_tag + r_content > b_tag + b_content:
                bearish += 1
                signals.append(f"Bearish risk: {item.title}")
        return bullish, bearish, signals

    @staticmethod
    def _direction_and_prob(bullish: int, bearish: int) -> tuple[str, float]:
        net = bullish - bearish
        probability = _BASE_PROB + _SIGNAL_STEP * net
        probability = min(_MAX_PROB, max(_MIN_PROB, probability))
        if net > 0:
            return "bullish", probability
        if net < 0:
            return "bearish", 1.0 - probability
        return "neutral", 0.50

    @staticmethod
    def _build_signal(direction: str, probability: float, horizon: str, calc_note: str) -> HorizonSignal:
        ind_note = f"; indicator context: {calc_note}" if calc_note and calc_note.strip().lower() != "none" else ""
        rationale = (
            f"{horizon} outlook is {direction} (estimated probability {probability:.0%}) "
            f"based on grounded knowledge-base context{ind_note}. "
            "This is an estimate, not a guarantee. Validate with live NSE/BSE data before trading."
        )
        return HorizonSignal(direction=direction, probability=round(probability, 4), rationale=rationale)

    def predict(
        self,
        context_items: list[KnowledgeItem],
        deterministic_note: str = "",
        resolved_entity: dict | None = None,
    ) -> PredictionSignals:
        """Generate multi-horizon prediction signals from grounded context."""
        bullish, bearish, key_signals = self._score_items(context_items)

        if resolved_entity:
            entity_label = (
                f"{resolved_entity.get('symbol', '')} ({resolved_entity.get('company_name', '')})"
            ).strip()
            if entity_label and entity_label != "()":
                key_signals.insert(0, f"Entity context: {entity_label}")

        intraday_dir, intraday_prob = self._direction_and_prob(bullish, bearish)

        # Swing and medium-term are increasingly uncertain; compress the signal toward 0.5
        swing_prob = 0.5 + (intraday_prob - 0.5) * 0.8
        swing_dir = intraday_dir if abs(swing_prob - 0.5) >= 0.02 else "neutral"

        mt_prob = 0.5 + (intraday_prob - 0.5) * 0.6
        mt_dir = intraday_dir if abs(mt_prob - 0.5) >= 0.04 else "neutral"

        calc_context = deterministic_note.split("\n")[0] if deterministic_note else ""
        intraday_signal = self._build_signal(intraday_dir, intraday_prob, "Intraday", calc_context)
        swing_signal = self._build_signal(swing_dir, swing_prob, "Swing (1-5 days)", calc_context)
        mt_signal = self._build_signal(mt_dir, mt_prob, "Medium-term (1-3 months)", calc_context)

        overall_confidence = max(0.10, min(0.60, 0.25 + 0.08 * len(context_items)))

        return PredictionSignals(
            intraday=intraday_signal,
            swing=swing_signal,
            medium_term=mt_signal,
            key_signals=tuple(key_signals[:5]),
            overall_confidence=round(overall_confidence, 4),
        )
