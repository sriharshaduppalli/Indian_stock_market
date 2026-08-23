"""Multi-horizon scenario head for the Indian Stock Market assistant.

Produces calibrated, risk-aware directional *estimates* (intraday, swing,
medium-term) from live indicators, ATR bands, and the quantitative trade-plan
stance. This is not a price oracle: every signal includes an uncertainty note.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from .knowledge_base import KnowledgeItem

_BULLISH_CONTENT = (
    "strong earnings",
    "deal win",
    "beat estimate",
    "guidance upgrade",
    "sector tailwind",
)
_BEARISH_CONTENT = (
    "miss estimate",
    "guidance cut",
    "margin pressure",
    "sector headwind",
    "earnings miss",
)

_BASE_PROB = 0.50
_SIGNAL_STEP = 0.05
_MAX_PROB = 0.72
_MIN_PROB = 0.28
_NONE_PLACEHOLDER = "none"
_ACTION_SCORE = {
    "BUY": 2.0,
    "ACCUMULATE": 1.5,
    "HOLD": 0.0,
    "WAIT": 0.0,
    "TRIM": -1.5,
    "SELL": -2.0,
}
_EVENT_RE = re.compile(
    r"\b(earnings|results? date|result day|board meeting|ex-?date|record date)\b",
    re.I,
)


@dataclass(frozen=True)
class HorizonSignal:
    """Directional estimate for a single time horizon."""

    direction: str  # "bullish" | "bearish" | "neutral"
    probability: float
    rationale: str
    range_low: float | None = None
    range_high: float | None = None


@dataclass(frozen=True)
class PredictionSignals:
    """Multi-horizon scenario output produced by PredictionEngine."""

    intraday: HorizonSignal
    swing: HorizonSignal
    medium_term: HorizonSignal
    key_signals: tuple[str, ...]
    overall_confidence: float
    scenarios: dict[str, str] = field(default_factory=dict)
    event_blackout: str | None = None
    advice: str = "HOLD"


class PredictionEngine:
    """Indicator + ATR scenario engine. KB phrases nudge lightly; actions do not."""

    def _score_items(
        self, context_items: list[KnowledgeItem]
    ) -> tuple[int, int, list[str]]:
        bullish = 0
        bearish = 0
        signals: list[str] = []
        for item in context_items or []:
            content_lower = str(getattr(item, "content", "") or "").lower()
            title = str(getattr(item, "title", "") or "")
            # Ignore generic literacy articles — only score specific event phrases.
            b_content = sum(1 for s in _BULLISH_CONTENT if s in content_lower)
            r_content = sum(1 for s in _BEARISH_CONTENT if s in content_lower)
            if b_content > r_content and b_content:
                bullish += 1
                signals.append(f"Constructive cue: {title or 'context'}")
            elif r_content > b_content and r_content:
                bearish += 1
                signals.append(f"Caution cue: {title or 'context'}")
        return bullish, bearish, signals

    @staticmethod
    def _direction_and_prob(net: float) -> tuple[str, float]:
        probability = _BASE_PROB + _SIGNAL_STEP * net
        probability = min(_MAX_PROB, max(_MIN_PROB, probability))
        if net > 0.5:
            return "bullish", probability
        if net < -0.5:
            return "bearish", min(_MAX_PROB, max(_MIN_PROB, 1.0 - probability + _BASE_PROB - 0.5))
        return "neutral", 0.50

    @staticmethod
    def _build_signal(
        direction: str,
        probability: float,
        horizon: str,
        calc_note: str,
        lo: float | None = None,
        hi: float | None = None,
    ) -> HorizonSignal:
        ind_note = (
            f"; indicator context: {calc_note}"
            if calc_note and calc_note.strip().lower() != _NONE_PLACEHOLDER
            else ""
        )
        band = ""
        if isinstance(lo, (int, float)) and isinstance(hi, (int, float)) and lo > 0 and hi > lo:
            band = f" Scenario band ₹{lo:.1f}–₹{hi:.1f}."
        rationale = (
            f"{horizon} tape bias is {direction} (estimated probability {probability:.0%}) "
            f"from indicators + structure{ind_note}.{band} "
            "Estimate only — not a guarantee. Validate with live NSE/BSE data."
        )
        return HorizonSignal(
            direction=direction,
            probability=round(probability, 4),
            rationale=rationale,
            range_low=round(float(lo), 2) if isinstance(lo, (int, float)) else None,
            range_high=round(float(hi), 2) if isinstance(hi, (int, float)) else None,
        )

    @staticmethod
    def _indicator_net(p0_math: Optional[dict[str, Any]]) -> tuple[float, list[str]]:
        notes: list[str] = []
        net = 0.0
        if not isinstance(p0_math, dict):
            return net, notes
        rsi = p0_math.get("wilder_rsi_14")
        st = (p0_math.get("supertrend") or {}).get("direction")
        macd_h = (p0_math.get("macd") or {}).get("histogram")
        rs20 = (p0_math.get("vs_nifty") or {}).get("rs_20d")
        if st == "bullish":
            net += 1
            notes.append("Supertrend bullish")
        elif st == "bearish":
            net -= 1
            notes.append("Supertrend bearish")
        if isinstance(macd_h, (int, float)):
            net += 0.5 if macd_h > 0 else -0.5
            notes.append("MACD histogram " + ("positive" if macd_h > 0 else "negative"))
        if isinstance(rs20, (int, float)):
            if rs20 >= 1.03:
                net += 0.5
                notes.append(f"RS vs Nifty {rs20:.3f}")
            elif rs20 <= 0.97:
                net -= 0.5
                notes.append(f"RS vs Nifty {rs20:.3f}")
        if isinstance(rsi, (int, float)):
            if rsi >= 70:
                net -= 1.5
                notes.append(f"Wilder RSI {rsi:.1f} stretched")
            elif rsi <= 30:
                net += 1.0
                notes.append(f"Wilder RSI {rsi:.1f} washed out")
        return net, notes

    @staticmethod
    def _price_and_atr(
        p0_math: Optional[dict[str, Any]],
        resolved_entity: dict | None,
    ) -> tuple[float | None, float | None, float | None, float | None]:
        price = None
        atr_val = None
        support = None
        resistance = None
        if isinstance(p0_math, dict):
            price = p0_math.get("price") or p0_math.get("last") or p0_math.get("close")
            atr_val = p0_math.get("atr_14")
            levels = p0_math.get("levels") or p0_math.get("trading_levels") or {}
            if isinstance(levels, dict):
                support = levels.get("support")
                resistance = levels.get("resistance")
            atr_stop = p0_math.get("atr_stop") or {}
            if isinstance(atr_stop, dict) and atr_val is None:
                atr_val = atr_stop.get("atr")
        if price is None and isinstance(resolved_entity, dict):
            price = resolved_entity.get("last") or resolved_entity.get("price")
        try:
            price_f = float(price) if price is not None else None
        except (TypeError, ValueError):
            price_f = None
        try:
            atr_f = float(atr_val) if atr_val is not None else None
        except (TypeError, ValueError):
            atr_f = None
        try:
            sup_f = float(support) if support is not None else None
        except (TypeError, ValueError):
            sup_f = None
        try:
            res_f = float(resistance) if resistance is not None else None
        except (TypeError, ValueError):
            res_f = None
        return price_f, atr_f, sup_f, res_f

    @staticmethod
    def _event_blackout(
        p0_math: Optional[dict[str, Any]],
        resolved_entity: dict | None,
        context_items: list[Any],
    ) -> str | None:
        chunks: list[str] = []
        if isinstance(p0_math, dict):
            chunks.append(str(p0_math.get("event_note") or ""))
            chunks.append(str((p0_math.get("catalyst") or "")))
        if isinstance(resolved_entity, dict):
            chunks.append(str(resolved_entity.get("next_event") or ""))
        for item in context_items or []:
            chunks.append(str(getattr(item, "content", "") or "")[:240])
        blob = " ".join(chunks)
        if _EVENT_RE.search(blob):
            return (
                "Results/event language is in context — skip a point forecast and "
                "use wider if/then bands until the event prints."
            )
        return None

    @staticmethod
    def _bands(
        price: float | None,
        atr_val: float | None,
        support: float | None,
        resistance: float | None,
        event_blackout: str | None,
    ) -> dict[str, str]:
        if not isinstance(price, (int, float)) or price <= 0:
            return {}
        width = atr_val if isinstance(atr_val, (int, float)) and atr_val > 0 else price * 0.012
        if event_blackout:
            width *= 1.6
        intra = (price - width, price + width)
        swing = (price - 1.5 * width, price + 1.5 * width)
        medium = (price - 2.5 * width, price + 2.5 * width)
        hold = (
            f"stay inside ₹{swing[0]:.1f}–₹{swing[1]:.1f}"
            + (f" while ₹{support:.1f} holds" if isinstance(support, (int, float)) else "")
        )
        brk = (
            f"next band toward ₹{price - 2.0 * width:.1f}"
            + (f" if ₹{support:.1f} gives way" if isinstance(support, (int, float)) else " if nearby support gives way")
        )
        if isinstance(resistance, (int, float)):
            hold += f"; upside check ₹{resistance:.1f}"
        return {
            "intraday_band": f"₹{intra[0]:.1f}–₹{intra[1]:.1f}",
            "swing_band": f"₹{swing[0]:.1f}–₹{swing[1]:.1f}",
            "medium_band": f"₹{medium[0]:.1f}–₹{medium[1]:.1f}",
            "support_hold": hold,
            "support_break": brk,
        }

    def predict(
        self,
        context_items: list[KnowledgeItem],
        deterministic_note: str = "",
        resolved_entity: dict | None = None,
        trade_plan: dict | None = None,
        p0_math: dict | None = None,
    ) -> PredictionSignals:
        """Generate multi-horizon scenario signals from indicators + structure."""
        bullish, bearish, key_signals = self._score_items(context_items)
        kb_net = float(bullish - bearish)
        ind_net, ind_notes = self._indicator_net(p0_math)
        plan = trade_plan or (p0_math or {}).get("trade_plan") if isinstance(p0_math, dict) else trade_plan
        if not isinstance(plan, dict):
            plan = {}
        action = str(plan.get("action") or "").upper()
        action_net = float(_ACTION_SCORE.get(action, 0.0))
        if action in _ACTION_SCORE:
            ind_notes = [f"Quant stance {action}"] + ind_notes
        # Indicators + trade-plan stance dominate; generic KB phrases nudge lightly.
        net = ind_net + action_net + 0.25 * kb_net
        key_signals = list(ind_notes) + key_signals

        if resolved_entity:
            entity_label = (
                f"{resolved_entity.get('symbol', '')} ({resolved_entity.get('company_name', '')})"
            ).strip()
            if entity_label and entity_label != "()":
                key_signals.insert(0, f"Entity context: {entity_label}")

        rsi = p0_math.get("wilder_rsi_14") if isinstance(p0_math, dict) else None
        intra_net = net
        if isinstance(rsi, (int, float)) and rsi >= 68:
            intra_net -= 1.0
        swing_net = net * 0.85
        mt_net = net * 0.55

        price, atr_val, support, resistance = self._price_and_atr(p0_math, resolved_entity)
        event_note = self._event_blackout(p0_math, resolved_entity, context_items)
        scenarios = self._bands(price, atr_val, support, resistance, event_note)

        def _lo_hi(key: str) -> tuple[float | None, float | None]:
            text = scenarios.get(key) or ""
            nums = re.findall(r"₹([0-9]+(?:\.[0-9]+)?)", text)
            if len(nums) >= 2:
                return float(nums[0]), float(nums[1])
            return None, None

        intra_lo, intra_hi = _lo_hi("intraday_band")
        swing_lo, swing_hi = _lo_hi("swing_band")
        mt_lo, mt_hi = _lo_hi("medium_band")

        calc_context = deterministic_note.split("\n")[0] if deterministic_note else ""
        if ind_notes:
            calc_context = (calc_context + "; " if calc_context else "") + "; ".join(ind_notes[:3])

        if event_note:
            key_signals.append(event_note)

        intraday_dir, intraday_prob = self._direction_and_prob(intra_net)
        swing_dir, swing_prob = self._direction_and_prob(swing_net)
        mt_dir, mt_prob = self._direction_and_prob(mt_net)

        overall_confidence = max(
            0.10,
            min(0.70, 0.22 + 0.05 * len(ind_notes) + 0.04 * min(len(context_items or []), 6)),
        )
        if (p0_math or {}).get("data_quality", {}).get("degraded"):
            overall_confidence = min(overall_confidence, 0.45)
        if event_note:
            overall_confidence = min(overall_confidence, 0.40)
        if not ind_notes:
            overall_confidence = min(overall_confidence, 0.48)

        if net > 1.2:
            advice = "BUY"
        elif net < -1.2:
            advice = "SELL"
        else:
            advice = "HOLD"
        if event_note and advice == "BUY":
            advice = "HOLD"

        return PredictionSignals(
            intraday=self._build_signal(intraday_dir, intraday_prob, "Intraday", calc_context, intra_lo, intra_hi),
            swing=self._build_signal(swing_dir, swing_prob, "Swing (1-5 days)", calc_context, swing_lo, swing_hi),
            medium_term=self._build_signal(mt_dir, mt_prob, "Medium-term (1-3 months)", calc_context, mt_lo, mt_hi),
            key_signals=tuple(key_signals[:6]),
            overall_confidence=round(overall_confidence, 4),
            scenarios=scenarios,
            event_blackout=event_note,
            advice=advice,
        )
