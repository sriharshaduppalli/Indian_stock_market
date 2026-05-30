from __future__ import annotations

import re


class DeterministicCalculator:
    @staticmethod
    def cagr(start: float, end: float, years: float) -> float:
        if start <= 0 or end <= 0 or years <= 0:
            raise ValueError("Invalid inputs for CAGR")
        return ((end / start) ** (1 / years) - 1) * 100

    @staticmethod
    def absolute_return(buy: float, sell: float) -> float:
        if buy <= 0 or sell < 0:
            raise ValueError("Invalid inputs for return")
        return ((sell - buy) / buy) * 100


class PandasTaIndicatorCalculator:
    _INDICATOR_KEYWORDS = ("rsi", "sma", "ema", "macd", "bollinger", "bbands")
    _PRICE_SERIES_PATTERN = re.compile(r"(?:prices?|series)\s*[:=]?\s*([0-9,\.\s-]+)", flags=re.IGNORECASE)
    _NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")

    @classmethod
    def indicator_requested(cls, query: str) -> bool:
        q = query.lower()
        return any(keyword in q for keyword in cls._INDICATOR_KEYWORDS)

    @classmethod
    def indicator_note(cls, query: str) -> str | None:
        q = query.lower()
        indicator = cls._indicator_from_query(q)
        if not indicator:
            return None
        try:
            import pandas as pd  # type: ignore
            import pandas_ta as ta  # type: ignore
        except Exception:
            return "pandas-ta indicator unavailable: install optional dependencies (pandas and pandas-ta)."
        prices = cls._price_series_from_query(query)
        if len(prices) < 5:
            return "pandas-ta indicator unavailable: provide at least 5 price points using 'prices ...'."
        close = pd.Series(prices, dtype="float64")
        period = cls._period_from_query(q)
        try:
            if indicator == "rsi":
                result = ta.rsi(close, length=period)
                value = cls._latest_value(result)
                return f"pandas-ta RSI({period}) is {value:.2f} from provided prices."
            if indicator == "sma":
                result = ta.sma(close, length=period)
                value = cls._latest_value(result)
                return f"pandas-ta SMA({period}) is {value:.2f} from provided prices."
            if indicator == "ema":
                result = ta.ema(close, length=period)
                value = cls._latest_value(result)
                return f"pandas-ta EMA({period}) is {value:.2f} from provided prices."
            if indicator == "macd":
                result = ta.macd(close, fast=12, slow=26, signal=9)
                if result is None or result.empty:
                    raise ValueError("No MACD values available")
                latest = result.dropna().tail(1)
                if latest.empty:
                    raise ValueError("No MACD values available")
                macd = float(latest.iloc[0].iloc[0])
                signal = float(latest.iloc[0].iloc[1])
                hist = float(latest.iloc[0].iloc[2])
                return (
                    "pandas-ta MACD is "
                    f"{macd:.2f} (signal {signal:.2f}, histogram {hist:.2f}) from provided prices."
                )
            if indicator == "bbands":
                result = ta.bbands(close, length=period, std=2.0)
                if result is None or result.empty:
                    raise ValueError("No Bollinger values available")
                latest = result.dropna().tail(1)
                if latest.empty:
                    raise ValueError("No Bollinger values available")
                lower = float(latest.iloc[0].iloc[0])
                middle = float(latest.iloc[0].iloc[1])
                upper = float(latest.iloc[0].iloc[2])
                return (
                    "pandas-ta Bollinger Bands are "
                    f"lower {lower:.2f}, middle {middle:.2f}, upper {upper:.2f}."
                )
        except Exception:
            return "pandas-ta indicator unavailable: unable to derive indicator from provided prices."
        return None

    @classmethod
    def _indicator_from_query(cls, query_lower: str) -> str | None:
        if "rsi" in query_lower:
            return "rsi"
        if "sma" in query_lower:
            return "sma"
        if "ema" in query_lower:
            return "ema"
        if "macd" in query_lower:
            return "macd"
        if "bollinger" in query_lower or "bbands" in query_lower:
            return "bbands"
        return None

    @classmethod
    def _price_series_from_query(cls, query: str) -> list[float]:
        match = cls._PRICE_SERIES_PATTERN.search(query)
        segment = match.group(1) if match else query
        return [float(m.group(0)) for m in cls._NUMBER_PATTERN.finditer(segment)]

    @staticmethod
    def _period_from_query(query_lower: str) -> int:
        period_match = re.search(r"(?:period|window|length)\s*(\d+)", query_lower)
        if period_match:
            return max(2, int(period_match.group(1)))
        indicator_match = re.search(r"(?:rsi|sma|ema|bbands?)\s*(\d+)", query_lower)
        if indicator_match:
            return max(2, int(indicator_match.group(1)))
        return 14

    @staticmethod
    def _latest_value(values) -> float:
        if values is None:
            raise ValueError("Indicator result missing")
        latest = values.dropna()
        if latest.empty:
            raise ValueError("Indicator result missing")
        return float(latest.iloc[-1])
