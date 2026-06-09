from __future__ import annotations

from typing import Any

import pandas as pd

from agents.base import Direction

BB_PERIOD = 20
BB_STD_MULT = 2.0


def calculate_bollinger_bands(
    closes: pd.Series,
    period: int = BB_PERIOD,
    std_mult: float = BB_STD_MULT,
) -> tuple[float, float, float]:
    """Return (lower, middle, upper) Bollinger Bands for the latest close."""
    if len(closes) < period:
        raise ValueError(
            f"Need at least {period} closes to calculate Bollinger Bands({period})"
        )

    window = closes.rolling(window=period)
    middle = window.mean().iloc[-1]
    std = window.std(ddof=0).iloc[-1]

    if pd.isna(middle) or pd.isna(std):
        raise ValueError("Bollinger Bands calculation returned no value")

    upper = float(middle + std_mult * std)
    lower = float(middle - std_mult * std)
    return lower, float(middle), upper


def _extract_closes(context: dict[str, Any]) -> pd.Series:
    candles = context.get("candles", [])
    closes = [float(candle["close"]) for candle in candles if "close" in candle]
    return pd.Series(closes, dtype=float)


def evaluate_bb_gate(
    context: dict[str, Any] | None,
    direction: Direction,
    *,
    period: int = BB_PERIOD,
    std_mult: float = BB_STD_MULT,
) -> str | None:
    """Block entries at band extremes (mean-reversion risk).

    Returns a block message, or None when the trade is allowed.
    LONG is blocked at/above the upper band, SHORT at/below the lower band.
    """
    if context is None:
        return None

    closes = _extract_closes(context)
    if len(closes) < period:
        return None

    try:
        lower, _, upper = calculate_bollinger_bands(
            closes,
            period=period,
            std_mult=std_mult,
        )
    except ValueError:
        return None

    close = float(closes.iloc[-1])

    if direction == Direction.LONG and close >= upper:
        return (
            f"NO TRADE: price {close:.2f} at/above upper Bollinger Band "
            f"{upper:.2f} (overextended for LONG)"
        )
    if direction == Direction.SHORT and close <= lower:
        return (
            f"NO TRADE: price {close:.2f} at/below lower Bollinger Band "
            f"{lower:.2f} (overextended for SHORT)"
        )
    return None
