from __future__ import annotations

import math
import re
from typing import Any


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

    @staticmethod
    def pe(price: float, eps: float) -> float:
        if eps == 0:
            raise ValueError("EPS cannot be zero")
        return price / eps

    @staticmethod
    def pb(price: float, book_value: float) -> float:
        if book_value == 0:
            raise ValueError("Book value cannot be zero")
        return price / book_value

    @staticmethod
    def peg(pe: float, growth_pct: float) -> float:
        if growth_pct == 0:
            raise ValueError("Growth cannot be zero")
        return pe / growth_pct

    @staticmethod
    def sharpe(excess_return_pct: float, volatility_pct: float) -> float:
        if volatility_pct == 0:
            raise ValueError("Volatility cannot be zero")
        return excess_return_pct / volatility_pct

    @staticmethod
    def drawdown(peak: float, trough: float) -> float:
        if peak <= 0:
            raise ValueError("Peak must be positive")
        return ((trough - peak) / peak) * 100


def _ema_series(values: list[float], length: int) -> list[float]:
    if length <= 1 or not values:
        return list(values)
    alpha = 2.0 / (length + 1.0)
    out: list[float] = []
    ema = values[0]
    for price in values:
        ema = alpha * price + (1.0 - alpha) * ema
        out.append(ema)
    return out


def _sma_last(values: list[float], length: int) -> float:
    if len(values) < length:
        raise ValueError("Not enough prices for SMA")
    window = values[-length:]
    return sum(window) / float(length)


def _rsi_last(closes: list[float], length: int = 14) -> float:
    if len(closes) <= length:
        raise ValueError("Not enough prices for RSI")
    gains = 0.0
    losses = 0.0
    for i in range(-length, 0):
        change = closes[i] - closes[i - 1]
        if change >= 0:
            gains += change
        else:
            losses -= change
    avg_gain = gains / length
    avg_loss = losses / length
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def _atr_last(highs: list[float], lows: list[float], closes: list[float], length: int = 14) -> float:
    if len(closes) <= length:
        raise ValueError("Not enough bars for ATR")
    trs: list[float] = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        trs.append(tr)
    window = trs[-length:]
    return sum(window) / float(length)


def _macd_last(closes: list[float]) -> tuple[float, float, float]:
    if len(closes) < 35:
        raise ValueError("Not enough prices for MACD")
    ema12 = _ema_series(closes, 12)
    ema26 = _ema_series(closes, 26)
    macd_line = [a - b for a, b in zip(ema12, ema26)]
    signal_line = _ema_series(macd_line, 9)
    macd = macd_line[-1]
    signal = signal_line[-1]
    hist = macd - signal
    return macd, signal, hist


def _bollinger_last(closes: list[float], length: int = 20, std_mult: float = 2.0) -> tuple[float, float, float]:
    if len(closes) < length:
        raise ValueError("Not enough prices for Bollinger")
    window = closes[-length:]
    middle = sum(window) / float(length)
    variance = sum((x - middle) ** 2 for x in window) / float(length)
    std = math.sqrt(variance)
    return middle - std_mult * std, middle, middle + std_mult * std


class PandasTaIndicatorCalculator:
    """Technical indicators with live NSE OHLCV auto-fetch.

    Backend preference:
      1) pandas-ta (best on Python < 3.14 / Render)
      2) `ta` library
      3) pure numpy-style fallbacks (always available)

    Users can paste `prices ...` or ask e.g. `RSI of RELIANCE`.
    """

    _INDICATOR_KEYWORDS = (
        "rsi",
        "sma",
        "ema",
        "macd",
        "bollinger",
        "bbands",
        "atr",
        "stoch",
        "stochastic",
        "adx",
        "vwap",
        "obv",
    )
    _PRICE_SERIES_PATTERN = re.compile(
        r"(?:prices?|series)\s*[:=]?\s*([0-9,\.\s-]+)", flags=re.IGNORECASE
    )
    _NUMBER_PATTERN = re.compile(r"-?\d+(?:\.\d+)?")
    _SYMBOL_PATTERN = re.compile(
        r"\b(?:of|for|on)\s+([A-Z][A-Z0-9.&-]{1,15})\b"
        r"|\b([A-Z]{2,15}(?:\.[A-Z]{1,3})?)\s+(?:rsi|sma|ema|macd|atr|adx|vwap|obv|bollinger|stoch)",
        flags=re.IGNORECASE,
    )
    _INDEX_ALIASES = {
        "NIFTY": "NIFTY50",
        "NIFTY50": "NIFTY50",
        "SENSEX": "SENSEX",
        "BANKNIFTY": "BANKNIFTY",
    }
    # Legacy tickers → current Yahoo/NSE symbols (demergers, renames).
    _SYMBOL_ALIASES = {
        "TATAMOTORS": "TMPV",
        "ADANITRANS": "ADANIENSOL",
        "GMRINFRA": "GMRAIRPORT",
        "MINDAIND": "UNOMINDA",
        "MOTHERSUMI": "MOTHERSON",
        "L&TFH": "LTF",
        "IIFLWAM": "360ONE",
        "IBULHSGFIN": "SAMMAANCAP",
        "ADANIWILMAR": "AWL",
        "HPCL": "HINDPETRO",
        "ZOMATO": "ETERNAL",
    }

    @classmethod
    def indicator_requested(cls, query: str) -> bool:
        q = query.lower()
        return any(keyword in q for keyword in cls._INDICATOR_KEYWORDS)

    @classmethod
    def indicator_note(cls, query: str, symbol_hint: str | None = None) -> str | None:
        q = query.lower()
        indicator = cls._indicator_from_query(q)
        if not indicator:
            return None

        prices = cls._price_series_from_query(query)
        source_label = "provided prices"
        symbol = (symbol_hint or cls._symbol_from_query(query) or "").strip().upper()
        symbol = cls._INDEX_ALIASES.get(symbol, symbol)
        symbol = cls._SYMBOL_ALIASES.get(symbol, symbol)

        opens: list[float] = []
        highs: list[float] = []
        lows: list[float] = []
        closes: list[float] = []
        volumes: list[float] = []

        if len(prices) >= 5:
            closes = prices
            highs = prices
            lows = prices
            opens = prices
            volumes = [0.0] * len(prices)
        elif symbol:
            frame = cls._load_ohlcv_frame(symbol)
            if frame is None:
                return (
                    f"Indicator unavailable: could not fetch OHLCV for {symbol}. "
                    "Try again or provide prices ..."
                )
            opens = list(frame["open"])
            highs = list(frame["high"])
            lows = list(frame["low"])
            closes = list(frame["close"])
            volumes = list(frame.get("volume") or [0.0] * len(closes))
            source_label = f"live market history for {symbol}"
        else:
            return (
                "Indicator unavailable: name a symbol (e.g. RSI of RELIANCE) "
                "or provide at least 5 price points using 'prices ...'."
            )

        period = cls._period_from_query(q)
        if closes:
            period = min(period, max(2, len(closes) - 1))
        # Prefer pandas-ta → ta → pure math.
        note = cls._compute_with_pandas_ta(
            indicator, opens, highs, lows, closes, volumes, period, source_label
        )
        if note:
            return note
        note = cls._compute_with_ta_lib(
            indicator, opens, highs, lows, closes, volumes, period, source_label
        )
        if note:
            return note
        return cls._compute_with_fallback(
            indicator, highs, lows, closes, volumes, period, source_label
        )

    @classmethod
    def _compute_with_pandas_ta(
        cls,
        indicator: str,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        period: int,
        source_label: str,
    ) -> str | None:
        try:
            import pandas as pd  # type: ignore
            import pandas_ta as pta  # type: ignore
        except Exception:
            return None
        close = pd.Series(closes, dtype="float64")
        high = pd.Series(highs, dtype="float64")
        low = pd.Series(lows, dtype="float64")
        volume = pd.Series(volumes, dtype="float64")
        try:
            if indicator == "rsi":
                value = cls._latest_value(pta.rsi(close, length=period))
                return f"pandas-ta RSI({period}) is {value:.2f} from {source_label}."
            if indicator == "sma":
                value = cls._latest_value(pta.sma(close, length=period))
                return f"pandas-ta SMA({period}) is {value:.2f} from {source_label}."
            if indicator == "ema":
                value = cls._latest_value(pta.ema(close, length=period))
                return f"pandas-ta EMA({period}) is {value:.2f} from {source_label}."
            if indicator == "macd":
                result = pta.macd(close, fast=12, slow=26, signal=9)
                latest = result.dropna().tail(1)
                macd = float(latest.iloc[0].iloc[0])
                signal = float(latest.iloc[0].iloc[1])
                hist = float(latest.iloc[0].iloc[2])
                return (
                    f"pandas-ta MACD is {macd:.2f} (signal {signal:.2f}, histogram {hist:.2f}) "
                    f"from {source_label}."
                )
            if indicator == "bbands":
                result = pta.bbands(close, length=period, std=2.0)
                latest = result.dropna().tail(1)
                lower = float(latest.iloc[0].iloc[0])
                middle = float(latest.iloc[0].iloc[1])
                upper = float(latest.iloc[0].iloc[2])
                return (
                    f"pandas-ta Bollinger Bands are lower {lower:.2f}, middle {middle:.2f}, "
                    f"upper {upper:.2f} from {source_label}."
                )
            if indicator == "atr":
                value = cls._latest_value(pta.atr(high=high, low=low, close=close, length=period))
                return f"pandas-ta ATR({period}) is {value:.2f} from {source_label}."
            if indicator == "stoch":
                result = pta.stoch(high=high, low=low, close=close)
                latest = result.dropna().tail(1)
                k = float(latest.iloc[0].iloc[0])
                d = float(latest.iloc[0].iloc[1]) if latest.shape[1] > 1 else k
                return f"pandas-ta Stochastic is %K {k:.2f}, %D {d:.2f} from {source_label}."
            if indicator == "adx":
                result = pta.adx(high=high, low=low, close=close, length=period)
                latest = result.dropna().tail(1)
                adx = float(latest.iloc[0].iloc[0])
                return f"pandas-ta ADX({period}) is {adx:.2f} from {source_label}."
            if indicator == "vwap":
                typical = (high + low + close) / 3.0
                cum_vol = volume.cumsum().replace(0, float("nan"))
                vwap_series = (typical * volume).cumsum() / cum_vol
                value = cls._latest_value(vwap_series)
                return f"pandas-ta VWAP (window approx) is {value:.2f} from {source_label}."
            if indicator == "obv":
                value = cls._latest_value(pta.obv(close=close, volume=volume))
                return f"pandas-ta OBV is {value:.2f} from {source_label}."
        except Exception:
            return None
        return None

    @classmethod
    def _compute_with_ta_lib(
        cls,
        indicator: str,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        period: int,
        source_label: str,
    ) -> str | None:
        try:
            import pandas as pd  # type: ignore
            from ta.momentum import RSIIndicator, StochasticOscillator  # type: ignore
            from ta.trend import MACD, EMAIndicator, SMAIndicator, ADXIndicator  # type: ignore
            from ta.volatility import AverageTrueRange, BollingerBands  # type: ignore
            from ta.volume import OnBalanceVolumeIndicator  # type: ignore
        except Exception:
            return None
        close = pd.Series(closes, dtype="float64")
        high = pd.Series(highs, dtype="float64")
        low = pd.Series(lows, dtype="float64")
        volume = pd.Series(volumes, dtype="float64")
        try:
            if indicator == "rsi":
                value = float(RSIIndicator(close=close, window=period).rsi().dropna().iloc[-1])
                return f"ta RSI({period}) is {value:.2f} from {source_label}."
            if indicator == "sma":
                value = float(SMAIndicator(close=close, window=period).sma_indicator().dropna().iloc[-1])
                return f"ta SMA({period}) is {value:.2f} from {source_label}."
            if indicator == "ema":
                value = float(EMAIndicator(close=close, window=period).ema_indicator().dropna().iloc[-1])
                return f"ta EMA({period}) is {value:.2f} from {source_label}."
            if indicator == "macd":
                macd_ind = MACD(close=close)
                macd = float(macd_ind.macd().dropna().iloc[-1])
                signal = float(macd_ind.macd_signal().dropna().iloc[-1])
                hist = float(macd_ind.macd_diff().dropna().iloc[-1])
                return (
                    f"ta MACD is {macd:.2f} (signal {signal:.2f}, histogram {hist:.2f}) "
                    f"from {source_label}."
                )
            if indicator == "bbands":
                bb = BollingerBands(close=close, window=period, window_dev=2)
                lower = float(bb.bollinger_lband().dropna().iloc[-1])
                middle = float(bb.bollinger_mavg().dropna().iloc[-1])
                upper = float(bb.bollinger_hband().dropna().iloc[-1])
                return (
                    f"ta Bollinger Bands are lower {lower:.2f}, middle {middle:.2f}, "
                    f"upper {upper:.2f} from {source_label}."
                )
            if indicator == "atr":
                value = float(
                    AverageTrueRange(high=high, low=low, close=close, window=period)
                    .average_true_range()
                    .dropna()
                    .iloc[-1]
                )
                return f"ta ATR({period}) is {value:.2f} from {source_label}."
            if indicator == "stoch":
                stoch = StochasticOscillator(high=high, low=low, close=close)
                k = float(stoch.stoch().dropna().iloc[-1])
                d = float(stoch.stoch_signal().dropna().iloc[-1])
                return f"ta Stochastic is %K {k:.2f}, %D {d:.2f} from {source_label}."
            if indicator == "adx":
                value = float(
                    ADXIndicator(high=high, low=low, close=close, window=period).adx().dropna().iloc[-1]
                )
                return f"ta ADX({period}) is {value:.2f} from {source_label}."
            if indicator == "vwap":
                typical = (high + low + close) / 3.0
                cum_vol = volume.cumsum().replace(0, float("nan"))
                value = float(((typical * volume).cumsum() / cum_vol).dropna().iloc[-1])
                return f"ta VWAP (window approx) is {value:.2f} from {source_label}."
            if indicator == "obv":
                value = float(
                    OnBalanceVolumeIndicator(close=close, volume=volume)
                    .on_balance_volume()
                    .dropna()
                    .iloc[-1]
                )
                return f"ta OBV is {value:.2f} from {source_label}."
        except Exception:
            return None
        return None

    @classmethod
    def _compute_with_fallback(
        cls,
        indicator: str,
        highs: list[float],
        lows: list[float],
        closes: list[float],
        volumes: list[float],
        period: int,
        source_label: str,
    ) -> str | None:
        try:
            if indicator == "rsi":
                value = _rsi_last(closes, period)
                return f"builtin RSI({period}) is {value:.2f} from {source_label}."
            if indicator == "sma":
                value = _sma_last(closes, period)
                return f"builtin SMA({period}) is {value:.2f} from {source_label}."
            if indicator == "ema":
                value = _ema_series(closes, period)[-1]
                return f"builtin EMA({period}) is {value:.2f} from {source_label}."
            if indicator == "macd":
                macd, signal, hist = _macd_last(closes)
                return (
                    f"builtin MACD is {macd:.2f} (signal {signal:.2f}, histogram {hist:.2f}) "
                    f"from {source_label}."
                )
            if indicator == "bbands":
                lower, middle, upper = _bollinger_last(closes, length=max(period, 20))
                return (
                    f"builtin Bollinger Bands are lower {lower:.2f}, middle {middle:.2f}, "
                    f"upper {upper:.2f} from {source_label}."
                )
            if indicator == "atr":
                value = _atr_last(highs, lows, closes, period)
                return f"builtin ATR({period}) is {value:.2f} from {source_label}."
            if indicator == "vwap":
                if not volumes or sum(volumes) <= 0:
                    return "VWAP needs volume — ask VWAP of SYMBOL with live history."
                num = 0.0
                den = 0.0
                for h, l, c, v in zip(highs, lows, closes, volumes):
                    typical = (h + l + c) / 3.0
                    num += typical * v
                    den += v
                value = num / den if den else 0.0
                return f"builtin VWAP (window approx) is {value:.2f} from {source_label}."
            if indicator == "obv":
                obv = 0.0
                for i in range(1, len(closes)):
                    if closes[i] > closes[i - 1]:
                        obv += volumes[i]
                    elif closes[i] < closes[i - 1]:
                        obv -= volumes[i]
                return f"builtin OBV is {obv:.2f} from {source_label}."
            if indicator in {"stoch", "adx"}:
                return (
                    f"{indicator.upper()} needs pandas-ta or the `ta` package "
                    "(install from requirements). RSI/SMA/EMA/MACD/ATR still work via builtin math."
                )
        except Exception:
            return "Indicator unavailable: unable to derive value from market data."
        return None

    @classmethod
    def _load_ohlcv_frame(cls, symbol: str) -> dict[str, list[float]] | None:
        symbol = cls._SYMBOL_ALIASES.get((symbol or "").upper(), (symbol or "").upper())
        # Prefer BYSEL market_data when running inside the app backend.
        try:
            from app.market_data import fetch_quote_history

            candles = fetch_quote_history(symbol, period="3mo", interval="1d")
            if candles:
                return {
                    "open": [float(c.get("open") or 0) for c in candles],
                    "high": [float(c.get("high") or 0) for c in candles],
                    "low": [float(c.get("low") or 0) for c in candles],
                    "close": [float(c.get("close") or 0) for c in candles],
                    "volume": [float(c.get("volume") or 0) for c in candles],
                }
        except Exception:
            pass

        try:
            import yfinance as yf  # type: ignore

            yahoo = None
            try:
                from app.market_data import _yf_ticker

                yahoo = _yf_ticker(symbol)
            except Exception:
                yahoo = None
            tickers: list[str] = []
            if yahoo:
                tickers.append(yahoo)
            if symbol.startswith("^"):
                tickers.append(symbol)
            else:
                tickers.extend([f"{symbol}.NS", f"{symbol}.BO"])
            hist = None
            for ticker in tickers:
                hist = yf.Ticker(ticker).history(period="3mo", interval="1d", auto_adjust=False)
                if hist is not None and not hist.empty:
                    break
            if hist is None or hist.empty:
                return None
            return {
                "open": [float(x) for x in hist["Open"].tolist()],
                "high": [float(x) for x in hist["High"].tolist()],
                "low": [float(x) for x in hist["Low"].tolist()],
                "close": [float(x) for x in hist["Close"].tolist()],
                "volume": [float(x) for x in hist["Volume"].tolist()],
            }
        except Exception:
            return None

    @classmethod
    def _indicator_from_query(cls, query_lower: str) -> str | None:
        if "rsi" in query_lower:
            return "rsi"
        if "macd" in query_lower:
            return "macd"
        if "bollinger" in query_lower or "bbands" in query_lower:
            return "bbands"
        if "stoch" in query_lower:
            return "stoch"
        if "adx" in query_lower:
            return "adx"
        if "atr" in query_lower:
            return "atr"
        if "vwap" in query_lower:
            return "vwap"
        if "obv" in query_lower:
            return "obv"
        if "sma" in query_lower:
            return "sma"
        if "ema" in query_lower:
            return "ema"
        return None

    @classmethod
    def _symbol_from_query(cls, query: str) -> str | None:
        # Prefer explicit "of/for/on SYMBOL" or "SYMBOL RSI" patterns.
        of_match = re.search(
            r"\b(?:of|for|on)\s+([A-Za-z][A-Za-z0-9.&-]{1,15})\b",
            query,
            flags=re.IGNORECASE,
        )
        if of_match:
            candidate = of_match.group(1).upper()
            if candidate.lower() not in cls._SYMBOL_STOPWORDS:
                return cls._INDEX_ALIASES.get(candidate, candidate)

        lead_match = re.search(
            r"\b([A-Za-z]{2,15}(?:\.[A-Za-z]{1,3})?)\s+"
            r"(?:rsi|sma|ema|macd|atr|adx|vwap|obv|bollinger|stoch)",
            query,
            flags=re.IGNORECASE,
        )
        if lead_match:
            candidate = lead_match.group(1).upper()
            if candidate.lower() not in cls._SYMBOL_STOPWORDS:
                return cls._INDEX_ALIASES.get(candidate, candidate)

        # Last resort: uppercase-looking tokens that are not English filler.
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9.&-]{1,15}\b", query)
        for token in tokens:
            if token.lower() in cls._SYMBOL_STOPWORDS:
                continue
            # Prefer all-caps ticker style in mixed queries.
            if token.isupper() and len(token) >= 2:
                return cls._INDEX_ALIASES.get(token, token)
        return None

    _SYMBOL_STOPWORDS = {
        "rsi", "sma", "ema", "macd", "atr", "adx", "vwap", "obv", "what", "is", "the",
        "of", "for", "on", "calculate", "show", "me", "current", "latest", "bollinger",
        "bands", "stochastic", "stoch", "indicator", "value", "please", "tell", "a",
        "an", "to", "my", "now", "right", "today", "about", "how", "do", "does",
        "can", "you", "give", "get", "find", "compute", "formula", "equation",
        "meaning", "define", "explain", "stock", "market", "india", "indian", "nse",
        "bse", "price", "prices", "series", "period", "window", "length",
        "sentiment", "news", "mood", "investor", "headline", "headlines", "analysis",
        "bullish", "bearish", "neutral", "factor", "factors",
        "buy", "sell", "hold", "wait", "trim", "accumulate", "action",
        "paper", "practice", "legend", "direct", "answer",
        "beginner", "beginners", "teach", "learn", "habit", "habits",
        "educational", "education",
        "nav", "sip", "ter", "cas", "fomo", "sgb", "drhp", "etf",
    }

    @classmethod
    def _price_series_from_query(cls, query: str) -> list[float]:
        match = cls._PRICE_SERIES_PATTERN.search(query)
        if not match:
            return []
        segment = match.group(1)
        return [float(m.group(0)) for m in cls._NUMBER_PATTERN.finditer(segment)]

    @staticmethod
    def _period_from_query(query_lower: str) -> int:
        period_match = re.search(r"(?:period|window|length)\s*(\d+)", query_lower)
        if period_match:
            return max(2, int(period_match.group(1)))
        # "50 EMA" puts the length before the name. "EMA 50" puts it after.
        leading = re.search(
            r"\b(\d{1,3})\s*(?:day|period|length)?\s*(?:rsi|sma|ema|dma|atr|adx)\b",
            query_lower,
        )
        if leading:
            return max(2, int(leading.group(1)))
        indicator_match = re.search(r"(?:rsi|sma|ema|bbands?|atr|adx)\s*(\d+)", query_lower)
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
