from __future__ import annotations

from enum import Enum
from typing import Any

import pandas as pd

from agents.base import Agent, AgentResult, Direction

TREND_TIMEFRAME = "1h"
H4_TIMEFRAME = "4h"
TREND_CANDLE_MIN = 200
TREND_H4_CANDLE_MIN = 200
DEFAULT_EMA_FAST = 50
DEFAULT_EMA_SLOW = 200


class TrendBias(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


def calculate_ema(closes: pd.Series, period: int) -> float:
    if len(closes) < period:
        raise ValueError(f"Need at least {period} closes to calculate EMA({period})")

    ema = closes.ewm(span=period, adjust=False).mean()
    value = ema.iloc[-1]
    if pd.isna(value):
        raise ValueError(f"EMA({period}) calculation returned no value")
    return float(value)


def _extract_closes(
    context: dict[str, Any],
    key: str,
    *,
    min_candles: int,
    label: str,
) -> pd.Series:
    candles = context.get(key, [])
    if not candles:
        raise ValueError(f"No {label} candle data in context")

    closes = [float(candle["close"]) for candle in candles if "close" in candle]
    if len(closes) < min_candles:
        raise ValueError(
            f"Need at least {min_candles} {label} closes for trend analysis, "
            f"got {len(closes)}"
        )
    return pd.Series(closes, dtype=float)


def _extract_h1_closes(context: dict[str, Any]) -> pd.Series:
    return _extract_closes(
        context,
        "h1_candles",
        min_candles=TREND_CANDLE_MIN,
        label="H1",
    )


def _extract_h4_closes(context: dict[str, Any]) -> pd.Series:
    return _extract_closes(
        context,
        "h4_candles",
        min_candles=TREND_H4_CANDLE_MIN,
        label="H4",
    )


def classify_trend_from_emas(price: float, ema50: float, ema200: float) -> TrendBias:
    ema_low = min(ema50, ema200)
    ema_high = max(ema50, ema200)

    if ema_low < price < ema_high:
        return TrendBias.NEUTRAL
    if price > ema50:
        return TrendBias.BULLISH
    if price < ema50:
        return TrendBias.BEARISH
    return TrendBias.NEUTRAL


def evaluate_h1_trend(
    closes: pd.Series,
    *,
    ema_fast: int = DEFAULT_EMA_FAST,
    ema_slow: int = DEFAULT_EMA_SLOW,
) -> tuple[TrendBias, float, float, float, str]:
    """Classify H1 trend from close prices and EMA50/EMA200."""
    price = float(closes.iloc[-1])
    ema50 = calculate_ema(closes, ema_fast)
    ema200 = calculate_ema(closes, ema_slow)
    bias = classify_trend_from_emas(price, ema50, ema200)

    if bias == TrendBias.NEUTRAL:
        detail = f"price {price:.2f} between EMA50 ({ema50:.2f}) and EMA200 ({ema200:.2f})"
    elif bias == TrendBias.BULLISH:
        detail = f"price {price:.2f} above EMA50 ({ema50:.2f})"
    else:
        detail = f"price {price:.2f} below EMA50 ({ema50:.2f})"

    return bias, price, ema50, ema200, detail


def trend_bias_to_direction(bias: TrendBias) -> Direction:
    if bias == TrendBias.BULLISH:
        return Direction.LONG
    if bias == TrendBias.BEARISH:
        return Direction.SHORT
    return Direction.NEUTRAL


class TrendFilterAgent(Agent):
    """Higher-timeframe trend filter using H1 EMA50/EMA200."""

    def __init__(
        self,
        ema_fast: int = DEFAULT_EMA_FAST,
        ema_slow: int = DEFAULT_EMA_SLOW,
    ) -> None:
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow

    @property
    def name(self) -> str:
        return "trend_filter"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        symbol = context.get("symbol", "UNKNOWN")

        try:
            closes = _extract_h1_closes(context)
            bias, price, ema50, ema200, detail = evaluate_h1_trend(
                closes,
                ema_fast=self.ema_fast,
                ema_slow=self.ema_slow,
            )
        except ValueError as exc:
            return AgentResult(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=f"{symbol} H1 trend unavailable: {exc}",
            )

        direction = trend_bias_to_direction(bias)
        confidence = 0.80 if direction != Direction.NEUTRAL else 0.0
        reason = (
            f"{symbol} {TREND_TIMEFRAME} trend {bias.value.upper()}: {detail} "
            f"[EMA50={ema50:.2f}, EMA200={ema200:.2f}]"
        )
        return AgentResult(direction=direction, confidence=confidence, reason=reason)


def analyze_h4_trend(context: dict[str, Any]) -> AgentResult:
    """Classify H4 trend using the same EMA50/EMA200 rules as H1."""
    symbol = context.get("symbol", "UNKNOWN")

    try:
        closes = _extract_h4_closes(context)
        bias, price, ema50, ema200, detail = evaluate_h1_trend(closes)
    except ValueError as exc:
        return AgentResult(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason=f"{symbol} H4 trend unavailable: {exc}",
        )

    direction = trend_bias_to_direction(bias)
    confidence = 0.80 if direction != Direction.NEUTRAL else 0.0
    reason = (
        f"{symbol} {H4_TIMEFRAME} trend {bias.value.upper()}: {detail} "
        f"[EMA50={ema50:.2f}, EMA200={ema200:.2f}]"
    )
    return AgentResult(direction=direction, confidence=confidence, reason=reason)
