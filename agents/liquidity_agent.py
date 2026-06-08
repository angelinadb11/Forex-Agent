from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from agents.base import Agent, AgentResult, Direction


@dataclass(frozen=True)
class SwingPoint:
    index: int
    price: float


@dataclass(frozen=True)
class EqualLevel:
    level: float
    count: int


@dataclass(frozen=True)
class LiquidityAnalysis:
    equal_highs: tuple[EqualLevel, ...]
    equal_lows: tuple[EqualLevel, ...]
    buy_side_liquidity: float | None
    sell_side_liquidity: float | None
    liquidity_sweep: Literal["bullish", "bearish"] | None
    stop_hunt: Literal["bullish", "bearish"] | None
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
        raise ValueError("Need at least 20 candles for liquidity analysis")

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


def _prices_equal(left: float, right: float, tolerance_pct: float) -> bool:
    midpoint = (left + right) / 2
    if midpoint == 0:
        return False
    return abs(left - right) / midpoint <= tolerance_pct


def _find_equal_levels(
    points: list[SwingPoint],
    tolerance_pct: float = 0.001,
) -> list[EqualLevel]:
    if len(points) < 2:
        return []

    clusters: list[EqualLevel] = []
    used: set[int] = set()

    for index, anchor in enumerate(points):
        if index in used:
            continue

        cluster = [anchor]
        used.add(index)
        for other_index, other in enumerate(points):
            if other_index in used:
                continue
            if _prices_equal(anchor.price, other.price, tolerance_pct):
                cluster.append(other)
                used.add(other_index)

        if len(cluster) >= 2:
            level = sum(point.price for point in cluster) / len(cluster)
            clusters.append(EqualLevel(level=round(level, 5), count=len(cluster)))

    return clusters


def _nearest_level_above(price: float, levels: list[float]) -> float | None:
    above = [level for level in levels if level > price]
    return min(above) if above else None


def _nearest_level_below(price: float, levels: list[float]) -> float | None:
    below = [level for level in levels if level < price]
    return max(below) if below else None


def _detect_recent_sweep(
    df: pd.DataFrame,
    buy_level: float | None,
    sell_level: float | None,
    lookback: int = 5,
) -> Literal["bullish", "bearish"] | None:
    start = max(0, len(df) - lookback)
    for index in range(len(df) - 1, start - 1, -1):
        candle = df.iloc[index]
        if sell_level is not None and candle["low"] < sell_level and candle["close"] > sell_level:
            return "bullish"
        if buy_level is not None and candle["high"] > buy_level and candle["close"] < buy_level:
            return "bearish"
    return None


def _level_matches_equal_cluster(level: float, clusters: list[EqualLevel], tolerance_pct: float) -> bool:
    return any(_prices_equal(level, cluster.level, tolerance_pct) for cluster in clusters)


def _relevant_equal_levels(
    clusters: list[EqualLevel],
    current_price: float,
    *,
    above: bool,
    limit: int = 3,
) -> list[EqualLevel]:
    if above:
        filtered = [cluster for cluster in clusters if cluster.level > current_price]
        filtered.sort(key=lambda cluster: cluster.level)
    else:
        filtered = [cluster for cluster in clusters if cluster.level < current_price]
        filtered.sort(key=lambda cluster: cluster.level, reverse=True)
    return filtered[:limit]


def analyze_liquidity(
    df: pd.DataFrame,
    swing_lookback: int = 2,
    equal_tolerance_pct: float = 0.001,
    sweep_lookback: int = 5,
) -> LiquidityAnalysis:
    swing_highs, swing_lows = _find_swing_points(df, lookback=swing_lookback)
    equal_highs = _find_equal_levels(swing_highs, tolerance_pct=equal_tolerance_pct)
    equal_lows = _find_equal_levels(swing_lows, tolerance_pct=equal_tolerance_pct)
    current_price = float(df.iloc[-1]["close"])

    high_levels = [point.price for point in swing_highs]
    low_levels = [point.price for point in swing_lows]
    if equal_highs:
        high_levels.extend(cluster.level for cluster in equal_highs)
    if equal_lows:
        low_levels.extend(cluster.level for cluster in equal_lows)

    buy_side_liquidity = _nearest_level_above(current_price, high_levels)
    sell_side_liquidity = _nearest_level_below(current_price, low_levels)

    liquidity_sweep = _detect_recent_sweep(
        df,
        buy_side_liquidity,
        sell_side_liquidity,
        lookback=sweep_lookback,
    )
    stop_hunt: Literal["bullish", "bearish"] | None = None

    if liquidity_sweep == "bullish" and sell_side_liquidity is not None:
        if _level_matches_equal_cluster(sell_side_liquidity, equal_lows, equal_tolerance_pct):
            stop_hunt = "bullish"
    elif liquidity_sweep == "bearish" and buy_side_liquidity is not None:
        if _level_matches_equal_cluster(buy_side_liquidity, equal_highs, equal_tolerance_pct):
            stop_hunt = "bearish"

    return LiquidityAnalysis(
        equal_highs=tuple(equal_highs),
        equal_lows=tuple(equal_lows),
        buy_side_liquidity=buy_side_liquidity,
        sell_side_liquidity=sell_side_liquidity,
        liquidity_sweep=liquidity_sweep,
        stop_hunt=stop_hunt,
        current_price=current_price,
    )


class LiquidityAgent(Agent):
    """Liquidity pool and sweep analysis agent."""

    def __init__(
        self,
        swing_lookback: int = 2,
        equal_tolerance_pct: float = 0.001,
        sweep_lookback: int = 5,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.equal_tolerance_pct = equal_tolerance_pct
        self.sweep_lookback = sweep_lookback

    @property
    def name(self) -> str:
        return "liquidity"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        symbol = context.get("symbol", "UNKNOWN")
        timeframe = context.get("metadata", {}).get("timeframe", "unknown")

        try:
            df = _candles_to_dataframe(context)
            analysis = analyze_liquidity(
                df,
                swing_lookback=self.swing_lookback,
                equal_tolerance_pct=self.equal_tolerance_pct,
                sweep_lookback=self.sweep_lookback,
            )
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
        analysis: LiquidityAnalysis,
        symbol: str,
        timeframe: str,
    ) -> tuple[Direction, float, str]:
        bullish_score = 0.0
        bearish_score = 0.0
        reasons: list[str] = []

        if analysis.equal_highs:
            bearish_score += 0.08
            relevant = _relevant_equal_levels(list(analysis.equal_highs), analysis.current_price, above=True)
            if relevant:
                levels = ", ".join(f"{level.level} (x{level.count})" for level in relevant)
                reasons.append(f"equal highs at {levels}")

        if analysis.equal_lows:
            bullish_score += 0.08
            relevant = _relevant_equal_levels(list(analysis.equal_lows), analysis.current_price, above=False)
            if relevant:
                levels = ", ".join(f"{level.level} (x{level.count})" for level in relevant)
                reasons.append(f"equal lows at {levels}")

        if analysis.buy_side_liquidity is not None:
            bearish_score += 0.05
            reasons.append(f"buy-side liquidity above at {analysis.buy_side_liquidity:.5f}")

        if analysis.sell_side_liquidity is not None:
            bullish_score += 0.05
            reasons.append(f"sell-side liquidity below at {analysis.sell_side_liquidity:.5f}")

        if analysis.liquidity_sweep == "bullish":
            bullish_score += 0.30
            reasons.append("bullish liquidity sweep (SSL taken)")
        elif analysis.liquidity_sweep == "bearish":
            bearish_score += 0.30
            reasons.append("bearish liquidity sweep (BSL taken)")

        if analysis.stop_hunt == "bullish":
            bullish_score += 0.25
            reasons.append("stop hunt at equal lows")
        elif analysis.stop_hunt == "bearish":
            bearish_score += 0.25
            reasons.append("stop hunt at equal highs")

        if bullish_score > bearish_score and bullish_score >= 0.35:
            confidence = round(min(1.0, bullish_score), 2)
            reason = f"{symbol} {timeframe} Liquidity: " + ", ".join(reasons)
            return Direction.LONG, confidence, reason

        if bearish_score > bullish_score and bearish_score >= 0.35:
            confidence = round(min(1.0, bearish_score), 2)
            reason = f"{symbol} {timeframe} Liquidity: " + ", ".join(reasons)
            return Direction.SHORT, confidence, reason

        confidence = round(max(bullish_score, bearish_score, 0.1), 2)
        summary = ", ".join(reasons) if reasons else "no clear liquidity setup"
        return (
            Direction.NEUTRAL,
            confidence,
            f"{symbol} {timeframe} Liquidity: {summary}",
        )
