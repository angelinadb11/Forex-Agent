from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import pandas as pd

from agents.base import Agent, AgentResult, Direction


class MarketStructure(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"


@dataclass(frozen=True)
class SwingPoint:
    index: int
    price: float


@dataclass(frozen=True)
class OrderBlock:
    kind: str
    index: int
    low: float
    high: float


@dataclass(frozen=True)
class FairValueGap:
    kind: str
    index: int
    low: float
    high: float
    filled: bool


@dataclass(frozen=True)
class SMCAnalysis:
    structure: MarketStructure
    bos: str | None
    choch: str | None
    order_block: OrderBlock | None
    fair_value_gap: FairValueGap | None
    current_price: float


def _candles_to_dataframe(context: dict[str, Any]) -> pd.DataFrame:
    candles = context.get("candles", [])
    if not candles:
        raise ValueError("No candle data in context")

    required = {"open", "high", "low", "close"}
    rows: list[dict[str, float]] = []
    for candle in candles:
        if not required.issubset(candle):
            continue
        rows.append(
            {
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
            }
        )

    if len(rows) < 20:
        raise ValueError("Need at least 20 candles for SMC analysis")

    return pd.DataFrame(rows)


def _find_swing_points(df: pd.DataFrame, lookback: int = 2) -> tuple[list[SwingPoint], list[SwingPoint]]:
    swing_highs: list[SwingPoint] = []
    swing_lows: list[SwingPoint] = []

    for index in range(lookback, len(df) - lookback):
        high = df.iloc[index]["high"]
        low = df.iloc[index]["low"]

        if all(high > df.iloc[index - offset]["high"] for offset in range(1, lookback + 1)):
            if all(high > df.iloc[index + offset]["high"] for offset in range(1, lookback + 1)):
                swing_highs.append(SwingPoint(index=index, price=float(high)))

        if all(low < df.iloc[index - offset]["low"] for offset in range(1, lookback + 1)):
            if all(low < df.iloc[index + offset]["low"] for offset in range(1, lookback + 1)):
                swing_lows.append(SwingPoint(index=index, price=float(low)))

    return swing_highs, swing_lows


def _detect_market_structure(
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> MarketStructure:
    if len(swing_highs) < 2 or len(swing_lows) < 2:
        return MarketStructure.RANGING

    recent_highs = swing_highs[-2:]
    recent_lows = swing_lows[-2:]

    higher_high = recent_highs[-1].price > recent_highs[-2].price
    higher_low = recent_lows[-1].price > recent_lows[-2].price
    lower_high = recent_highs[-1].price < recent_highs[-2].price
    lower_low = recent_lows[-1].price < recent_lows[-2].price

    if higher_high and higher_low:
        return MarketStructure.BULLISH
    if lower_high and lower_low:
        return MarketStructure.BEARISH
    return MarketStructure.RANGING


def _detect_bos_choch(
    df: pd.DataFrame,
    structure: MarketStructure,
    swing_highs: list[SwingPoint],
    swing_lows: list[SwingPoint],
) -> tuple[str | None, str | None]:
    if not swing_highs or not swing_lows:
        return None, None

    last_close = float(df.iloc[-1]["close"])
    last_swing_high = swing_highs[-1].price
    last_swing_low = swing_lows[-1].price

    bos: str | None = None
    choch: str | None = None

    if last_close > last_swing_high:
        if structure == MarketStructure.BULLISH:
            bos = "bullish"
        elif structure == MarketStructure.BEARISH:
            choch = "bullish"
        else:
            bos = "bullish"
    elif last_close < last_swing_low:
        if structure == MarketStructure.BEARISH:
            bos = "bearish"
        elif structure == MarketStructure.BULLISH:
            choch = "bearish"
        else:
            bos = "bearish"

    return bos, choch


def _is_displacement_candle(df: pd.DataFrame, index: int, threshold: float = 1.5) -> bool:
    body = abs(df.iloc[index]["close"] - df.iloc[index]["open"])
    avg_body = (df["close"] - df["open"]).abs().rolling(20).mean().iloc[index]
    if pd.isna(avg_body) or avg_body == 0:
        return False
    return body >= avg_body * threshold


def _find_order_block(df: pd.DataFrame, lookback: int = 80) -> OrderBlock | None:
    start = max(0, len(df) - lookback)

    for index in range(len(df) - 2, start, -1):
        if not _is_displacement_candle(df, index):
            continue

        displacement = df.iloc[index]
        if displacement["close"] <= displacement["open"]:
            continue

        for ob_index in range(index - 1, start, -1):
            candle = df.iloc[ob_index]
            if candle["close"] < candle["open"]:
                return OrderBlock(
                    kind="bullish",
                    index=ob_index,
                    low=float(min(candle["open"], candle["close"])),
                    high=float(max(candle["open"], candle["close"])),
                )

    for index in range(len(df) - 2, start, -1):
        if not _is_displacement_candle(df, index):
            continue

        displacement = df.iloc[index]
        if displacement["close"] >= displacement["open"]:
            continue

        for ob_index in range(index - 1, start, -1):
            candle = df.iloc[ob_index]
            if candle["close"] > candle["open"]:
                return OrderBlock(
                    kind="bearish",
                    index=ob_index,
                    low=float(min(candle["open"], candle["close"])),
                    high=float(max(candle["open"], candle["close"])),
                )

    return None


def _find_recent_fair_value_gap(df: pd.DataFrame, lookback: int = 30) -> FairValueGap | None:
    start = max(2, len(df) - lookback)
    current_price = float(df.iloc[-1]["close"])

    for index in range(len(df) - 1, start - 1, -1):
        first = df.iloc[index - 2]
        third = df.iloc[index]

        if first["high"] < third["low"]:
            gap = FairValueGap(
                kind="bullish",
                index=index,
                low=float(first["high"]),
                high=float(third["low"]),
                filled=current_price <= first["high"],
            )
            if not gap.filled:
                return gap

        if first["low"] > third["high"]:
            gap = FairValueGap(
                kind="bearish",
                index=index,
                low=float(third["high"]),
                high=float(first["low"]),
                filled=current_price >= first["low"],
            )
            if not gap.filled:
                return gap

    return None


def _price_in_zone(price: float, low: float, high: float) -> bool:
    return low <= price <= high


def analyze_smc(df: pd.DataFrame, swing_lookback: int = 2) -> SMCAnalysis:
    """Run Smart Money Concepts analysis on OHLC data."""
    swing_highs, swing_lows = _find_swing_points(df, lookback=swing_lookback)
    structure = _detect_market_structure(swing_highs, swing_lows)
    bos, choch = _detect_bos_choch(df, structure, swing_highs, swing_lows)

    order_block = _find_order_block(df)
    fair_value_gap = _find_recent_fair_value_gap(df)
    current_price = float(df.iloc[-1]["close"])

    return SMCAnalysis(
        structure=structure,
        bos=bos,
        choch=choch,
        order_block=order_block,
        fair_value_gap=fair_value_gap,
        current_price=current_price,
    )


class SMCAgent(Agent):
    """Smart Money Concepts analysis agent."""

    def __init__(self, swing_lookback: int = 2) -> None:
        self.swing_lookback = swing_lookback

    @property
    def name(self) -> str:
        return "smc"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        symbol = context.get("symbol", "UNKNOWN")
        timeframe = context.get("metadata", {}).get("timeframe", "unknown")

        try:
            df = _candles_to_dataframe(context)
            analysis = analyze_smc(df, swing_lookback=self.swing_lookback)
        except ValueError as exc:
            return AgentResult(
                direction=Direction.NEUTRAL,
                confidence=0.0,
                reason=str(exc),
            )

        direction, confidence, reason = self._evaluate_analysis(
            analysis=analysis,
            symbol=symbol,
            timeframe=timeframe,
        )
        return AgentResult(direction=direction, confidence=confidence, reason=reason)

    def _evaluate_analysis(
        self,
        analysis: SMCAnalysis,
        symbol: str,
        timeframe: str,
    ) -> tuple[Direction, float, str]:
        bullish_score = 0.0
        bearish_score = 0.0
        reasons: list[str] = []

        if analysis.structure == MarketStructure.BULLISH:
            bullish_score += 0.25
            reasons.append("bullish structure (HH/HL)")
        elif analysis.structure == MarketStructure.BEARISH:
            bearish_score += 0.25
            reasons.append("bearish structure (LH/LL)")

        if analysis.bos == "bullish":
            bullish_score += 0.25
            reasons.append("bullish BOS")
        elif analysis.bos == "bearish":
            bearish_score += 0.25
            reasons.append("bearish BOS")

        if analysis.choch == "bullish":
            bullish_score += 0.2
            reasons.append("bullish ChoCH")
        elif analysis.choch == "bearish":
            bearish_score += 0.2
            reasons.append("bearish ChoCH")

        if analysis.order_block and _price_in_zone(
            analysis.current_price,
            analysis.order_block.low,
            analysis.order_block.high,
        ):
            if analysis.order_block.kind == "bullish":
                bullish_score += 0.2
                reasons.append("price in bullish order block")
            else:
                bearish_score += 0.2
                reasons.append("price in bearish order block")

        if analysis.fair_value_gap:
            if analysis.fair_value_gap.kind == "bullish":
                bullish_score += 0.15
                reasons.append("unfilled bullish FVG nearby")
            else:
                bearish_score += 0.15
                reasons.append("unfilled bearish FVG nearby")

        if bullish_score > bearish_score and bullish_score >= 0.35:
            confidence = round(min(1.0, bullish_score), 2)
            reason = f"{symbol} {timeframe} SMC: " + ", ".join(reasons)
            return Direction.LONG, confidence, reason

        if bearish_score > bullish_score and bearish_score >= 0.35:
            confidence = round(min(1.0, bearish_score), 2)
            reason = f"{symbol} {timeframe} SMC: " + ", ".join(reasons)
            return Direction.SHORT, confidence, reason

        confidence = round(max(bullish_score, bearish_score, 0.1), 2)
        summary = ", ".join(reasons) if reasons else "no clear SMC confluence"
        return (
            Direction.NEUTRAL,
            confidence,
            f"{symbol} {timeframe} SMC: {summary}",
        )


def smc_result_has_structure_conflict(result: AgentResult) -> bool:
    """True when HH/HL structure conflicts with bearish ChoCH, or LH/LL vs bullish ChoCH."""
    reason_lower = result.reason.lower()
    bullish_structure = "bullish structure (hh/hl)" in reason_lower
    bearish_structure = "bearish structure (lh/ll)" in reason_lower
    bullish_choch = "bullish choch" in reason_lower
    bearish_choch = "bearish choch" in reason_lower
    return (bullish_structure and bearish_choch) or (bearish_structure and bullish_choch)
