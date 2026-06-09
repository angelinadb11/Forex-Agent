from __future__ import annotations

from typing import Any

from agents.base import Agent, AgentResult, Direction
from agents.zone_helpers import (
    OrderBlockZone,
    ZoneCatalog,
    fvg_confirms_order_block,
    resolve_bar_index,
    resolve_trend_direction,
    resolve_zone_snapshot,
)

MAX_OB_AGE = 30
MIN_IMPULSE_PIPS = 15.0


class OrderBlockAgent(Agent):
    """Order block agent — retests of institutional candles before impulse moves."""

    @property
    def name(self) -> str:
        return "order_block"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        catalog = context.get("zone_catalog")
        if isinstance(catalog, ZoneCatalog):
            return self._analyze_from_catalog(context, catalog)
        return self._analyze_incremental(context)

    def _analyze_from_catalog(
        self,
        context: dict[str, Any],
        catalog: ZoneCatalog,
    ) -> AgentResult:
        symbol = str(context.get("symbol", "UNKNOWN"))
        timeframe = context.get("metadata", {}).get("timeframe", "unknown")
        trend_direction = resolve_trend_direction(context)
        bar_index = resolve_bar_index(context)

        active_blocks = catalog.active_obs_by_bar[bar_index]
        retesting = catalog.obs_retesting_at(bar_index)
        fvgs = catalog.unfilled_fvgs_at(bar_index, max_age=MAX_OB_AGE)

        if not active_blocks:
            return AgentResult(
                Direction.NEUTRAL,
                0.0,
                f"{symbol} {timeframe} OB: no active blocks within {MAX_OB_AGE} candles",
            )
        if not retesting:
            return AgentResult(
                Direction.NEUTRAL,
                0.0,
                f"{symbol} {timeframe} OB: blocks found but price not retesting zone",
            )

        best_score = 0.0
        best_direction = Direction.NEUTRAL
        best_reason = ""

        for block in retesting:
            direction = (
                Direction.LONG if block.direction == "bullish" else Direction.SHORT
            )
            confidence, reason = self._score_order_block(
                block=block,
                fvgs=fvgs,
                trend_direction=trend_direction,
                pip_size=catalog.pip_size,
                symbol=symbol,
                timeframe=timeframe,
            )
            if confidence > best_score:
                best_score = confidence
                best_direction = direction
                best_reason = reason

        return AgentResult(
            direction=best_direction,
            confidence=round(min(1.0, best_score), 2),
            reason=best_reason,
        )

    def _analyze_incremental(self, context: dict[str, Any]) -> AgentResult:
        symbol = str(context.get("symbol", "UNKNOWN"))
        timeframe = context.get("metadata", {}).get("timeframe", "unknown")
        trend_direction = resolve_trend_direction(context)

        try:
            current_price, pip_size, fvgs, order_blocks = resolve_zone_snapshot(
                context,
                symbol,
                max_ob_age=MAX_OB_AGE,
                min_impulse_pips=MIN_IMPULSE_PIPS,
            )
        except ValueError as exc:
            return AgentResult(Direction.NEUTRAL, 0.0, str(exc))

        retesting = [
            block
            for block in order_blocks
            if block.age_candles <= MAX_OB_AGE
            and block.zone_low <= current_price <= block.zone_high
        ]
        if not order_blocks:
            return AgentResult(
                Direction.NEUTRAL,
                0.0,
                f"{symbol} {timeframe} OB: no active blocks within {MAX_OB_AGE} candles",
            )
        if not retesting:
            return AgentResult(
                Direction.NEUTRAL,
                0.0,
                f"{symbol} {timeframe} OB: blocks found but price not retesting zone",
            )

        best_score = 0.0
        best_direction = Direction.NEUTRAL
        best_reason = ""

        for block in retesting:
            direction = (
                Direction.LONG if block.direction == "bullish" else Direction.SHORT
            )
            confidence, reason = self._score_order_block(
                block=block,
                fvgs=fvgs,
                trend_direction=trend_direction,
                pip_size=pip_size,
                symbol=symbol,
                timeframe=timeframe,
            )
            if confidence > best_score:
                best_score = confidence
                best_direction = direction
                best_reason = reason

        return AgentResult(
            direction=best_direction,
            confidence=round(min(1.0, best_score), 2),
            reason=best_reason,
        )

    def _score_order_block(
        self,
        *,
        block: OrderBlockZone,
        fvgs: tuple | list,
        trend_direction: Direction | None,
        pip_size: float,
        symbol: str,
        timeframe: str,
    ) -> tuple[float, str]:
        confidence = 0.35
        reasons = ["price retesting OB zone"]

        expected_trend = (
            Direction.LONG if block.direction == "bullish" else Direction.SHORT
        )
        if trend_direction == expected_trend:
            confidence += 0.25
            reasons.append("OB aligns with H1 trend")

        if block.impulse_pips >= MIN_IMPULSE_PIPS:
            confidence += 0.20
            reasons.append(f"impulse {block.impulse_pips:.1f} pips")

        if fvg_confirms_order_block(block, list(fvgs), pip_size):
            confidence += 0.20
            reasons.append("FVG confirms OB")

        bias = "bullish" if block.direction == "bullish" else "bearish"
        detail = ", ".join(reasons)
        reason = (
            f"{symbol} {timeframe} OB: {bias} block "
            f"[{block.zone_low:.2f}-{block.zone_high:.2f}], age {block.age_candles}c, {detail}"
        )
        return confidence, reason
