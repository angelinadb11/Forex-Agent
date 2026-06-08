from __future__ import annotations

from typing import Any

import pandas as pd

from agents.base import Agent, AgentResult, Direction


def calculate_rsi(closes: pd.Series, period: int = 14) -> float:
    """Calculate the latest RSI value using Wilder's smoothing."""
    if len(closes) < period + 1:
        raise ValueError(f"Need at least {period + 1} closes to calculate RSI({period})")

    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]

    if pd.isna(value):
        raise ValueError("RSI calculation returned no value")

    return float(value)


def _extract_closes(context: dict[str, Any]) -> pd.Series:
    candles = context.get("candles", [])
    if not candles:
        raise ValueError("No candle data in context")

    closes = [float(candle["close"]) for candle in candles if "close" in candle]
    if not closes:
        raise ValueError("Candle data missing close prices")

    return pd.Series(closes, dtype=float)


class RSIAgent(Agent):
    """Relative Strength Index analysis agent."""

    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
    ) -> None:
        self.period = period
        self.oversold = oversold
        self.overbought = overbought

    @property
    def name(self) -> str:
        return "rsi"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        symbol = context.get("symbol", "UNKNOWN")
        timeframe = context.get("metadata", {}).get("timeframe", "unknown")

        try:
            closes = _extract_closes(context)
            rsi = calculate_rsi(closes, period=self.period)
        except ValueError as exc:
            return AgentResult(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=str(exc),
            )

        direction, confidence, reason = self._evaluate_rsi(
            rsi=rsi,
            symbol=symbol,
            timeframe=timeframe,
        )
        return AgentResult(direction=direction, confidence=confidence, reason=reason)

    def _evaluate_rsi(
        self,
        rsi: float,
        symbol: str,
        timeframe: str,
    ) -> tuple[Direction, float, str]:
        if rsi <= self.oversold:
            distance = self.oversold - rsi
            confidence = min(1.0, 0.5 + distance / 50)
            return (
                Direction.LONG,
                round(confidence, 2),
                f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} oversold (<={self.oversold})",
            )

        if rsi >= self.overbought:
            distance = rsi - self.overbought
            confidence = min(1.0, 0.5 + distance / 50)
            return (
                Direction.SHORT,
                round(confidence, 2),
                f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} overbought (>={self.overbought})",
            )

        midpoint_distance = abs(rsi - 50)
        confidence = round(min(0.4, midpoint_distance / 100), 2)
        return (
            Direction.NEUTRAL,
            confidence,
            f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} neutral ({self.oversold}-{self.overbought})",
        )
