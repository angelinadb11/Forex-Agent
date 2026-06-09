from __future__ import annotations

from typing import Any

from agents.base import Agent, AgentResult, Direction
from agents.zone_helpers import (
    FVGZone,
    count_fvgs_at_level,
    price_inside_zone,
    resolve_trend_direction,
    resolve_zone_snapshot,
)

MAX_FVG_AGE = 20
MIN_FVG_SIZE_PIPS = 10.0


class FVGAgent(Agent):
    """Fair Value Gap agent — unfilled imbalance zones aligned with H1 trend."""

    @property
    def name(self) -> str:
        return "fvg"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        symbol = str(context.get("symbol", "UNKNOWN"))
        timeframe = context.get("metadata", {}).get("timeframe", "unknown")
        trend_direction = resolve_trend_direction(context)

        try:
            current_price, pip_size, fvgs, _ = resolve_zone_snapshot(
                context,
                symbol,
                max_fvg_age=MAX_FVG_AGE,
            )
        except ValueError as exc:
            return AgentResult(Direction.NEUTRAL, 0.0, str(exc))

        active = [
            fvg
            for fvg in fvgs
            if not fvg.filled and fvg.age_candles <= MAX_FVG_AGE
        ]
        if not active:
            return AgentResult(
                Direction.NEUTRAL,
                0.0,
                f"{symbol} {timeframe} FVG: no active unfilled gaps within {MAX_FVG_AGE} candles",
            )

        best_score = 0.0
        best_direction = Direction.NEUTRAL
        best_reason = ""

        for fvg in active:
            direction = (
                Direction.LONG if fvg.direction == "bullish" else Direction.SHORT
            )
            confidence, reason = self._score_fvg(
                fvg=fvg,
                fvgs=fvgs,
                current_price=current_price,
                trend_direction=trend_direction,
                pip_size=pip_size,
                symbol=symbol,
                timeframe=timeframe,
            )
            if confidence > best_score:
                best_score = confidence
                best_direction = direction
                best_reason = reason

        if best_score == 0.0:
            return AgentResult(
                Direction.NEUTRAL,
                0.0,
                f"{symbol} {timeframe} FVG: active gaps found but no qualifying setup",
            )

        return AgentResult(
            direction=best_direction,
            confidence=round(min(1.0, best_score), 2),
            reason=best_reason,
        )

    def _score_fvg(
        self,
        *,
        fvg: FVGZone,
        fvgs: list[FVGZone],
        current_price: float,
        trend_direction: Direction | None,
        pip_size: float,
        symbol: str,
        timeframe: str,
    ) -> tuple[float, str]:
        if fvg.filled or fvg.age_candles > MAX_FVG_AGE:
            return 0.0, ""

        confidence = 0.0
        reasons: list[str] = []

        if price_inside_zone(current_price, fvg.gap_low, fvg.gap_high):
            confidence += 0.30
            reasons.append("price inside FVG")

        expected_trend = (
            Direction.LONG if fvg.direction == "bullish" else Direction.SHORT
        )
        if trend_direction == expected_trend:
            confidence += 0.25
            reasons.append("FVG aligns with H1 trend")

        if fvg.size_pips > MIN_FVG_SIZE_PIPS:
            confidence += 0.15
            reasons.append(f"FVG size {fvg.size_pips:.1f} pips")

        if count_fvgs_at_level(fvgs, fvg, pip_size) >= 2:
            confidence += 0.20
            reasons.append("stacked FVG at level")

        if confidence == 0.0:
            return 0.0, ""

        bias = "bullish" if fvg.direction == "bullish" else "bearish"
        detail = ", ".join(reasons)
        reason = (
            f"{symbol} {timeframe} FVG: {bias} gap "
            f"[{fvg.gap_low:.2f}-{fvg.gap_high:.2f}], age {fvg.age_candles}c, {detail}"
        )
        return confidence, reason
