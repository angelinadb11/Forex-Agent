from __future__ import annotations

from typing import Any

import pandas as pd

from agents.base import Agent, AgentResult, Direction

RSI_GATE_LONG_MIN = 35.0
RSI_GATE_SHORT_MAX = 65.0


def extract_rsi_value(result: AgentResult) -> float | None:
    """Parse RSI value from an agent result reason string."""
    marker = "RSI("
    if marker not in result.reason:
        return None
    try:
        fragment = result.reason.split("=", 1)[1]
        value_text = fragment.split()[0]
        return float(value_text)
    except (IndexError, ValueError):
        return None


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


def _resolve_trend_direction(context: dict[str, Any]) -> Direction | None:
    trend = context.get("trend_direction")
    if isinstance(trend, Direction):
        return trend
    if trend is None:
        return None
    normalized = str(trend).strip().lower()
    if normalized in {"long", "bullish"}:
        return Direction.LONG
    if normalized in {"short", "bearish"}:
        return Direction.SHORT
    if normalized == "neutral":
        return Direction.NEUTRAL
    return None


class RSIAgent(Agent):
    """RSI confirmation agent aligned with the higher-timeframe trend."""

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
        trend_direction = _resolve_trend_direction(context)

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
            trend_direction=trend_direction,
            symbol=symbol,
            timeframe=timeframe,
        )
        return AgentResult(direction=direction, confidence=confidence, reason=reason)

    def _evaluate_rsi(
        self,
        rsi: float,
        trend_direction: Direction | None,
        symbol: str,
        timeframe: str,
    ) -> tuple[Direction, float, str]:
        if trend_direction is None:
            return (
                Direction.NEUTRAL,
                0.0,
                (
                    f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} ignored "
                    "(H1 trend unavailable)"
                ),
            )

        if rsi <= self.oversold:
            if trend_direction == Direction.LONG:
                distance = self.oversold - rsi
                confidence = round(min(0.45, 0.30 + distance / 100), 2)
                return (
                    Direction.LONG,
                    confidence,
                    (
                        f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} oversold "
                        f"confirms bullish H1 trend"
                    ),
                )
            return (
                Direction.NEUTRAL,
                0.0,
                (
                    f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} oversold "
                    "ignored against bearish H1 trend"
                ),
            )

        if rsi >= self.overbought:
            if trend_direction == Direction.SHORT:
                distance = rsi - self.overbought
                confidence = round(min(0.45, 0.30 + distance / 100), 2)
                return (
                    Direction.SHORT,
                    confidence,
                    (
                        f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} overbought "
                        f"confirms bearish H1 trend"
                    ),
                )
            return (
                Direction.NEUTRAL,
                0.0,
                (
                    f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} overbought "
                    "ignored against bullish H1 trend"
                ),
            )

        midpoint_distance = abs(rsi - 50)
        confidence = round(min(0.15, midpoint_distance / 200), 2)
        return (
            Direction.NEUTRAL,
            confidence,
            (
                f"{symbol} {timeframe} RSI({self.period})={rsi:.2f} neutral "
                f"({self.oversold}-{self.overbought}), no trend confirmation"
            ),
        )
