from __future__ import annotations

import time
from typing import Any, Callable

from agents.base import Direction
from strategy.runner import run_agents
from strategy.structure_weakness import (
    STRUCTURE_CHECK_INTERVAL_SECONDS,
    h1_trend_flipped_against_trade,
    resolve_latest_candle_open_time,
)

ContextFetcher = Callable[[str, str], dict[str, Any]]


def sl_at_or_better_than_breakeven(trade) -> bool:
    """Return True when SL is already at entry or better (locked profit)."""
    if trade.direction == Direction.LONG:
        return trade.stop_loss >= trade.entry
    if trade.direction == Direction.SHORT:
        return trade.stop_loss <= trade.entry
    return False


def should_check_trend_breakeven(
    trade,
    *,
    candle_open_time: float | None,
    now_monotonic: float,
) -> bool:
    if trade.closed or trade.tp1_hit or trade.trend_warning_sent:
        return False
    if sl_at_or_better_than_breakeven(trade):
        return False
    if trade.entry_trend_direction is None or not trade.timeframe:
        return False

    if candle_open_time is not None:
        return trade.last_trend_candle_open_time != candle_open_time

    if (
        trade.last_trend_check_monotonic
        and now_monotonic - trade.last_trend_check_monotonic
        < STRUCTURE_CHECK_INTERVAL_SECONDS
    ):
        return False

    return True


def assess_trend_breakeven_alert(
    trade,
    *,
    current_trend_direction: Direction | None,
) -> bool:
    """Return True when H1 trend flipped against the trade before TP1."""
    return h1_trend_flipped_against_trade(
        trade.direction,
        trade.entry_trend_direction,
        current_trend_direction,
    )


class TrendBreakevenAlertChecker:
    """Detects H1 trend flips on open trades and recommends moving SL to entry."""

    def __init__(self, context_fetcher: ContextFetcher) -> None:
        self.context_fetcher = context_fetcher

    def analyze(
        self,
        trade,
        *,
        now_monotonic: float | None = None,
    ) -> bool:
        if trade.closed or not trade.timeframe:
            return False

        now = now_monotonic if now_monotonic is not None else time.monotonic()
        m15_context = self.context_fetcher(trade.symbol, trade.timeframe)
        candle_open_time = resolve_latest_candle_open_time(m15_context)
        if not should_check_trend_breakeven(
            trade,
            candle_open_time=candle_open_time,
            now_monotonic=now,
        ):
            return False

        current_results = run_agents(m15_context)
        trend = current_results.get("trend_filter")
        current_trend_direction = trend.direction if trend is not None else None
        should_alert = assess_trend_breakeven_alert(
            trade,
            current_trend_direction=current_trend_direction,
        )

        trade.last_trend_check_monotonic = now
        if candle_open_time is not None:
            trade.last_trend_candle_open_time = candle_open_time

        return should_alert
