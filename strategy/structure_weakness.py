from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd

from agents.base import Direction
from agents.rsi_agent import calculate_rsi
from agents.smc_agent import _candles_to_dataframe, analyze_smc
from agents.zone_helpers import price_inside_zone, resolve_bar_index, resolve_zone_snapshot
from strategy.runner import run_agents

ContextFetcher = Callable[[str, str], dict[str, Any]]

STRUCTURE_WARNING_MESSAGES: tuple[str, ...] = (
    "⚠️ Структура слабшає — розглянь захист позиції",
    "⚠️ Ринок показує ознаки розвороту — SL на беззбиток?",
    "⚠️ Сигнал під тиском — стеж за позицією",
)

MAX_STRUCTURE_WARNINGS = 2
MIN_WEAKNESS_CONDITIONS = 2
STRUCTURE_CHECK_INTERVAL_SECONDS = 900.0
RSI_REVERSAL_DELTA = 12.0
RSI_CANDLE_DELTA = 8.0
RSI_LONG_WEAK_LEVEL = 45.0
RSI_SHORT_WEAK_LEVEL = 55.0


@dataclass(frozen=True)
class EntryZone:
    zone_low: float
    zone_high: float
    kind: str


@dataclass(frozen=True)
class StructureWeaknessAssessment:
    should_warn: bool
    met_conditions: tuple[str, ...] = ()
    message: str | None = None


def pick_structure_warning_message(rng: random.Random | None = None) -> str:
    source = rng if rng is not None else random
    return source.choice(STRUCTURE_WARNING_MESSAGES)


def _trend_supports_trade(trend: Direction, trade_direction: Direction) -> bool:
    if trade_direction == Direction.LONG:
        return trend == Direction.LONG
    if trade_direction == Direction.SHORT:
        return trend == Direction.SHORT
    return False


def resolve_entry_zone(
    context: dict[str, Any],
    direction: Direction,
    entry_price: float,
) -> EntryZone | None:
    """Return the OB or FVG zone that contained the entry price."""
    if direction == Direction.NEUTRAL:
        return None

    expected = "bullish" if direction == Direction.LONG else "bearish"
    symbol = str(context.get("symbol", "UNKNOWN"))
    catalog = context.get("zone_catalog")

    if catalog is not None:
        bar_index = resolve_bar_index(context)
        for block in catalog.obs_retesting_at(bar_index):
            if block.direction != expected:
                continue
            if price_inside_zone(entry_price, block.zone_low, block.zone_high):
                return EntryZone(block.zone_low, block.zone_high, "ob")

        for fvg in catalog.unfilled_fvgs_at(bar_index):
            if fvg.direction != expected:
                continue
            if price_inside_zone(entry_price, fvg.gap_low, fvg.gap_high):
                return EntryZone(fvg.gap_low, fvg.gap_high, "fvg")
        return None

    try:
        _, _, fvgs, order_blocks = resolve_zone_snapshot(context, symbol)
    except ValueError:
        return None

    for block in order_blocks:
        if block.direction != expected:
            continue
        if price_inside_zone(entry_price, block.zone_low, block.zone_high):
            return EntryZone(block.zone_low, block.zone_high, "ob")

    for fvg in fvgs:
        if fvg.direction != expected or fvg.filled:
            continue
        if price_inside_zone(entry_price, fvg.gap_low, fvg.gap_high):
            return EntryZone(fvg.gap_low, fvg.gap_high, "fvg")

    return None


def resolve_entry_rsi(context: dict[str, Any]) -> float | None:
    candles = context.get("candles", [])
    if not candles:
        return None
    closes = [float(candle["close"]) for candle in candles if "close" in candle]
    if len(closes) < 15:
        return None
    try:
        return calculate_rsi(pd.Series(closes, dtype=float))
    except ValueError:
        return None


def resolve_latest_candle_open_time(context: dict[str, Any]) -> float | None:
    candles = context.get("candles", [])
    if not candles:
        return None
    last_candle = candles[-1]
    open_time = last_candle.get("open_time")
    if open_time is None:
        return None
    return float(open_time)


def enrich_trade_entry_context(trade, context: dict[str, Any]) -> None:
    """Store entry zone and RSI snapshot used by structure weakness checks."""
    zone = resolve_entry_zone(context, trade.direction, trade.entry)
    if zone is not None:
        trade.entry_zone_low = zone.zone_low
        trade.entry_zone_high = zone.zone_high
        trade.entry_zone_kind = zone.kind

    entry_rsi = resolve_entry_rsi(context)
    if entry_rsi is not None:
        trade.entry_rsi = entry_rsi


def bos_against_trade_on_m15(context: dict[str, Any], trade_direction: Direction) -> bool:
    if trade_direction == Direction.NEUTRAL:
        return False

    try:
        df = _candles_to_dataframe(context)
        analysis = analyze_smc(df)
    except ValueError:
        return False

    if trade_direction == Direction.LONG:
        return analysis.bos == "bearish"
    return analysis.bos == "bullish"


def h1_trend_flipped_against_trade(
    trade_direction: Direction,
    entry_trend: Direction | None,
    current_trend: Direction | None,
) -> bool:
    if (
        trade_direction == Direction.NEUTRAL
        or entry_trend is None
        or current_trend is None
    ):
        return False

    if not _trend_supports_trade(entry_trend, trade_direction):
        return False

    return not _trend_supports_trade(current_trend, trade_direction)


def rsi_sharp_reversal_against_trade(
    trade_direction: Direction,
    *,
    entry_rsi: float | None,
    current_rsi: float,
    previous_rsi: float | None,
) -> bool:
    if entry_rsi is None or trade_direction == Direction.NEUTRAL:
        return False

    if trade_direction == Direction.LONG:
        if current_rsi > RSI_LONG_WEAK_LEVEL:
            return False
        drop_from_entry = entry_rsi - current_rsi
        drop_from_previous = (
            previous_rsi - current_rsi if previous_rsi is not None else 0.0
        )
        return (
            drop_from_entry >= RSI_REVERSAL_DELTA
            or drop_from_previous >= RSI_CANDLE_DELTA
        )

    if current_rsi < RSI_SHORT_WEAK_LEVEL:
        return False
    rise_from_entry = current_rsi - entry_rsi
    rise_from_previous = (
        current_rsi - previous_rsi if previous_rsi is not None else 0.0
    )
    return (
        rise_from_entry >= RSI_REVERSAL_DELTA
        or rise_from_previous >= RSI_CANDLE_DELTA
    )


def entry_zone_broken(
    price: float,
    *,
    zone_low: float | None,
    zone_high: float | None,
    trade_direction: Direction,
) -> bool:
    if zone_low is None or zone_high is None or trade_direction == Direction.NEUTRAL:
        return False

    if trade_direction == Direction.LONG:
        return price < zone_low
    return price > zone_high


def assess_structure_weakness(
    trade,
    *,
    m15_context: dict[str, Any],
    current_results: dict[str, Any],
    current_rsi: float | None,
    rng: random.Random | None = None,
) -> StructureWeaknessAssessment:
    """Return True when at least two structural weakness conditions are met."""
    if trade.closed or trade.structure_warning_count >= MAX_STRUCTURE_WARNINGS:
        return StructureWeaknessAssessment(should_warn=False)

    current_price = float(m15_context["candles"][-1]["close"])
    current_trend = current_results.get("trend_filter")
    current_trend_direction = (
        current_trend.direction if current_trend is not None else None
    )

    conditions: list[str] = []

    if bos_against_trade_on_m15(m15_context, trade.direction):
        conditions.append("bos_against")

    if h1_trend_flipped_against_trade(
        trade.direction,
        trade.entry_trend_direction,
        current_trend_direction,
    ):
        conditions.append("h1_trend_flip")

    if current_rsi is not None and rsi_sharp_reversal_against_trade(
        trade.direction,
        entry_rsi=trade.entry_rsi,
        current_rsi=current_rsi,
        previous_rsi=trade.last_rsi,
    ):
        conditions.append("rsi_reversal")

    if entry_zone_broken(
        current_price,
        zone_low=trade.entry_zone_low,
        zone_high=trade.entry_zone_high,
        trade_direction=trade.direction,
    ):
        conditions.append("entry_zone_break")

    if len(conditions) < MIN_WEAKNESS_CONDITIONS:
        return StructureWeaknessAssessment(
            should_warn=False,
            met_conditions=tuple(conditions),
        )

    return StructureWeaknessAssessment(
        should_warn=True,
        met_conditions=tuple(conditions),
        message=pick_structure_warning_message(rng),
    )


def should_run_structure_check(
    trade,
    *,
    candle_open_time: float | None,
    now_monotonic: float,
) -> bool:
    if trade.closed or trade.structure_warning_count >= MAX_STRUCTURE_WARNINGS:
        return False

    if candle_open_time is not None:
        return trade.last_structure_candle_open_time != candle_open_time

    if (
        trade.last_structure_check_monotonic
        and now_monotonic - trade.last_structure_check_monotonic
        < STRUCTURE_CHECK_INTERVAL_SECONDS
    ):
        return False

    return True


class StructureWeaknessChecker:
    """Checks open trades once per M15 candle for structural weakness."""

    def __init__(self, context_fetcher: ContextFetcher) -> None:
        self.context_fetcher = context_fetcher

    def analyze(
        self,
        trade,
        *,
        now_monotonic: float,
        rng: random.Random | None = None,
    ) -> StructureWeaknessAssessment | None:
        if trade.closed or not trade.timeframe:
            return None

        m15_context = self.context_fetcher(trade.symbol, trade.timeframe)
        candle_open_time = resolve_latest_candle_open_time(m15_context)
        if not should_run_structure_check(
            trade,
            candle_open_time=candle_open_time,
            now_monotonic=now_monotonic,
        ):
            return None

        current_results = run_agents(m15_context)
        current_rsi = resolve_entry_rsi(m15_context)
        assessment = assess_structure_weakness(
            trade,
            m15_context=m15_context,
            current_results=current_results,
            current_rsi=current_rsi,
            rng=rng,
        )

        trade.last_structure_check_monotonic = now_monotonic
        if candle_open_time is not None:
            trade.last_structure_candle_open_time = candle_open_time
        if current_rsi is not None:
            trade.last_rsi = current_rsi

        return assessment
