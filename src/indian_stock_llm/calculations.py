from __future__ import annotations


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
