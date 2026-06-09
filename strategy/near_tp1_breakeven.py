from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from agents.base import Direction
from strategy.structure_weakness import (
    STRUCTURE_CHECK_INTERVAL_SECONDS,
    bos_against_trade_on_m15,
    entry_zone_broken,
    resolve_entry_rsi,
    resolve_latest_candle_open_time,
    rsi_sharp_reversal_against_trade,
)
from strategy.trend_breakeven_alert import sl_at_or_better_than_breakeven

ContextFetcher = Callable[[str, str], dict[str, Any]]

NEAR_TP1_MIN_PROGRESS_R = 1.2


def favorable_progress_r(
    direction: Direction,
    *,
    entry: float,
    risk: float,
    high: float,
    low: float,
) -> float:
    if risk <= 0:
        return 0.0
    if direction == Direction.LONG:
        return max(0.0, (high - entry) / risk)
    if direction == Direction.SHORT:
        return max(0.0, (entry - low) / risk)
    return 0.0


@dataclass(frozen=True)
class NearTp1ReversalAssessment:
    should_move_sl_to_entry: bool
    peak_progress_r: float = 0.0
    met_conditions: tuple[str, ...] = ()


def assess_m15_reversal_conditions(
    trade_direction: Direction,
    *,
    m15_context: dict[str, Any],
    entry_zone_low: float | None,
    entry_zone_high: float | None,
    entry_rsi: float | None,
    previous_rsi: float | None,
) -> tuple[bool, tuple[str, ...]]:
    """Return True when M15 shows reversal against the open trade direction."""
    if trade_direction == Direction.NEUTRAL:
        return False, ()

    candles = m15_context.get("candles", [])
    if not candles:
        return False, ()

    current_price = float(candles[-1]["close"])
    current_rsi = resolve_entry_rsi(m15_context)
    conditions: list[str] = []

    if bos_against_trade_on_m15(m15_context, trade_direction):
        conditions.append("bos_against")

    if current_rsi is not None and rsi_sharp_reversal_against_trade(
        trade_direction,
        entry_rsi=entry_rsi,
        current_rsi=current_rsi,
        previous_rsi=previous_rsi,
    ):
        conditions.append("rsi_reversal")

    if entry_zone_broken(
        current_price,
        zone_low=entry_zone_low,
        zone_high=entry_zone_high,
        trade_direction=trade_direction,
    ):
        conditions.append("entry_zone_break")

    met = tuple(conditions)
    return bool(met), met


def assess_near_tp1_reversal(
    trade_direction: Direction,
    *,
    peak_progress_r: float,
    tp1_hit: bool,
    sl_at_breakeven: bool,
    m15_context: dict[str, Any],
    entry_zone_low: float | None,
    entry_zone_high: float | None,
    entry_rsi: float | None,
    previous_rsi: float | None,
) -> NearTp1ReversalAssessment:
    """Recommend SL at entry when trade reached >=1.2R and M15 shows reversal."""
    if (
        trade_direction == Direction.NEUTRAL
        or tp1_hit
        or sl_at_breakeven
        or peak_progress_r + 1e-9 < NEAR_TP1_MIN_PROGRESS_R
    ):
        return NearTp1ReversalAssessment(
            should_move_sl_to_entry=False,
            peak_progress_r=peak_progress_r,
        )

    active, conditions = assess_m15_reversal_conditions(
        trade_direction,
        m15_context=m15_context,
        entry_zone_low=entry_zone_low,
        entry_zone_high=entry_zone_high,
        entry_rsi=entry_rsi,
        previous_rsi=previous_rsi,
    )

    return NearTp1ReversalAssessment(
        should_move_sl_to_entry=active,
        peak_progress_r=peak_progress_r,
        met_conditions=conditions,
    )


def should_check_near_tp1_breakeven(
    trade,
    *,
    candle_open_time: float | None,
    now_monotonic: float,
) -> bool:
    if trade.closed or trade.tp1_hit or trade.near_tp1_warning_sent:
        return False
    if sl_at_or_better_than_breakeven(trade):
        return False
    if getattr(trade, "peak_progress_r", 0.0) + 1e-9 < NEAR_TP1_MIN_PROGRESS_R:
        return False
    if not trade.timeframe:
        return False

    if candle_open_time is not None:
        return trade.last_near_tp1_candle_open_time != candle_open_time

    if (
        trade.last_near_tp1_check_monotonic
        and now_monotonic - trade.last_near_tp1_check_monotonic
        < STRUCTURE_CHECK_INTERVAL_SECONDS
    ):
        return False

    return True


class NearTp1BreakevenChecker:
    """Detects near-TP1 reversals and recommends moving SL to entry."""

    def __init__(self, context_fetcher: ContextFetcher) -> None:
        self.context_fetcher = context_fetcher

    def analyze(
        self,
        trade,
        *,
        now_monotonic: float | None = None,
    ) -> NearTp1ReversalAssessment | None:
        if trade.closed or not trade.timeframe:
            return None

        now = now_monotonic if now_monotonic is not None else time.monotonic()
        m15_context = self.context_fetcher(trade.symbol, trade.timeframe)
        candle_open_time = resolve_latest_candle_open_time(m15_context)
        if not should_check_near_tp1_breakeven(
            trade,
            candle_open_time=candle_open_time,
            now_monotonic=now,
        ):
            return None

        assessment = assess_near_tp1_reversal(
            trade.direction,
            peak_progress_r=getattr(trade, "peak_progress_r", 0.0),
            tp1_hit=trade.tp1_hit,
            sl_at_breakeven=sl_at_or_better_than_breakeven(trade),
            m15_context=m15_context,
            entry_zone_low=trade.entry_zone_low,
            entry_zone_high=trade.entry_zone_high,
            entry_rsi=trade.entry_rsi,
            previous_rsi=trade.last_rsi,
        )

        trade.last_near_tp1_check_monotonic = now
        if candle_open_time is not None:
            trade.last_near_tp1_candle_open_time = candle_open_time
        current_rsi = resolve_entry_rsi(m15_context)
        if current_rsi is not None:
            trade.last_rsi = current_rsi

        return assessment
