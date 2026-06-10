"""VIP 2 scalp: ICT Turtle Soup (M5, XAUUSD).

Failed breakout: sweep 10-30 pips through a reference level, same candle
closes back inside. Market entry at sweep close, SL behind sweep wick,
TP1=1.5R / TP2=2.5R / TP3=opposite liquidity.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

from agents.base import Direction
from agents.liquidity_agent import _find_equal_levels, _find_swing_points
from backtest.engine import candle_timestamp
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, align_trade_signal_direction
from strategy.scalp_mode import ScalpAnalysisResult, ScalpPublishGate
from strategy.sweep_fvg_scalp import (
    RefLevel,
    asia_range_for_day,
    build_timestamps,
    liquidity_pool_levels,
    prev_hour_range,
)
from tracking.trade_pnl import pip_size_for_symbol

TURTLE_SOUP_TIMEFRAME = "5m-vip2"
TURTLE_SOUP_SYMBOLS = frozenset({"XAUUSD"})
TURTLE_SOUP_CANDLE_LIMIT = 700

MIN_PIERCE_PIPS = 10.0
MAX_PIERCE_PIPS = 30.0
STOP_BUFFER_PIPS = 2.0
MIN_SL_PIPS = 5.0
MAX_SL_PIPS = 30.0
TP1_R = 1.5
TP2_R = 2.5
TP3_FALLBACK_R = 3.5
LOCAL_SWING_LOOKBACK = 120

TURTLE_MAX_SIGNALS_PER_DAY = 5
TURTLE_MIN_INTERVAL_SECONDS = 600

VIP2_SIGNAL_TAG = "VIP2 Turtle Soup"
VIP2_SESSION_START_HOUR = 8
VIP2_SESSION_END_HOUR = 16
ASIA_FLAT_END_HOUR = 6


def is_vip2_core_session(current: datetime) -> bool:
    """VIP2 quality window: 08:00-16:00 UTC, skip Asia flat (00:00-06:00)."""
    if current.tzinfo is None:
        current = current.replace(tzinfo=timezone.utc)
    else:
        current = current.astimezone(timezone.utc)
    if current.hour < ASIA_FLAT_END_HOUR:
        return False
    return VIP2_SESSION_START_HOUR <= current.hour < VIP2_SESSION_END_HOUR


@dataclass(frozen=True)
class TurtleSoupSetup:
    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    level: float
    level_label: str
    sl_pips: float
    reason: str


def is_turtle_soup_scalp_enabled(symbol: str) -> bool:
    return resolve_symbol(symbol).display in TURTLE_SOUP_SYMBOLS


def build_turtle_soup_gate() -> ScalpPublishGate:
    return ScalpPublishGate(
        min_interval_seconds=TURTLE_MIN_INTERVAL_SECONDS,
        max_signals_per_day=TURTLE_MAX_SIGNALS_PER_DAY,
    )


def local_swing_levels(candles: list, index: int) -> list[RefLevel]:
    import pandas as pd

    start = max(0, index - LOCAL_SWING_LOOKBACK)
    window = candles[start:index]
    if len(window) < 7:
        return []

    df = pd.DataFrame(
        {
            "open": [c["open"] for c in window],
            "high": [c["high"] for c in window],
            "low": [c["low"] for c in window],
            "close": [c["close"] for c in window],
        }
    )
    swing_highs, swing_lows = _find_swing_points(df, lookback=2)
    levels: list[RefLevel] = []
    if swing_highs:
        latest = max(swing_highs, key=lambda point: point.index)
        levels.append(RefLevel(latest.price, "high", "swing-high"))
    if swing_lows:
        latest = max(swing_lows, key=lambda point: point.index)
        levels.append(RefLevel(latest.price, "low", "swing-low"))
    return levels


def prev_day_range(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    target_date = timestamps[index].date() - timedelta(days=1)
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index, -1, -1):
        ts = timestamps[j]
        if ts.date() == target_date:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
        elif ts.date() < target_date:
            break
    if not highs:
        return None
    return max(highs), min(lows)


def reference_levels_for_bar(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> list[RefLevel]:
    levels: list[RefLevel] = []
    asia = asia_range_for_day(candles, timestamps, index)
    if asia:
        levels.append(RefLevel(asia[0], "high", "asia-high"))
        levels.append(RefLevel(asia[1], "low", "asia-low"))
    hour = prev_hour_range(candles, timestamps, index)
    if hour:
        levels.append(RefLevel(hour[0], "high", "hour-high"))
        levels.append(RefLevel(hour[1], "low", "hour-low"))
    prev_day = prev_day_range(candles, timestamps, index)
    if prev_day:
        levels.append(RefLevel(prev_day[0], "high", "prev-day-high"))
        levels.append(RefLevel(prev_day[1], "low", "prev-day-low"))
    levels.extend(local_swing_levels(candles, index))
    levels.extend(liquidity_pool_levels(candles, index))
    return levels


def opposite_liquidity_tp3(
    *,
    direction: Direction,
    entry: float,
    risk: float,
    levels: list[RefLevel],
    pip: float,
) -> float:
    if direction == Direction.LONG:
        candidates = [
            ref.price
            for ref in levels
            if ref.kind == "high" and ref.price > entry + risk * 0.5
        ]
        if candidates:
            return max(min(candidates), entry + TP2_R * risk + pip)
        return entry + TP3_FALLBACK_R * risk
    candidates = [
        ref.price
        for ref in levels
        if ref.kind == "low" and ref.price < entry - risk * 0.5
    ]
    if candidates:
        return min(max(candidates), entry - TP2_R * risk - pip)
    return entry - TP3_FALLBACK_R * risk


def detect_turtle_soup_setup(
    candles: list[dict[str, Any]],
    *,
    index: int | None = None,
    symbol: str = "XAUUSD",
) -> tuple[TurtleSoupSetup | None, str]:
    if len(candles) < 3:
        return None, "NO VIP2: need at least 3 M5 candles"

    sweep_index = len(candles) - 1 if index is None else index
    if sweep_index < 1:
        return None, "NO VIP2: insufficient history"

    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    timestamps = build_timestamps(candles)
    sweep = candles[sweep_index]
    pre = candles[sweep_index - 1]
    levels = reference_levels_for_bar(candles, timestamps, sweep_index)
    if not levels:
        return None, "NO VIP2: no reference levels"

    min_pierce = MIN_PIERCE_PIPS * pip
    max_pierce = MAX_PIERCE_PIPS * pip

    for ref in levels:
        if ref.kind == "low":
            pierce = ref.price - sweep["low"]
            if not (
                min_pierce <= pierce <= max_pierce
                and sweep["close"] > ref.price
                and pre["low"] >= ref.price
            ):
                continue
            entry = float(sweep["close"])
            stop = float(sweep["low"]) - STOP_BUFFER_PIPS * pip
            if entry <= stop:
                continue
            risk = entry - stop
            sl_pips = risk / pip
            if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
                return None, f"NO VIP2: SL {sl_pips:.1f} pips outside {MIN_SL_PIPS}-{MAX_SL_PIPS}"
            tp1 = entry + TP1_R * risk
            tp2 = entry + TP2_R * risk
            tp3 = opposite_liquidity_tp3(
                direction=Direction.LONG,
                entry=entry,
                risk=risk,
                levels=levels,
                pip=pip,
            )
            return TurtleSoupSetup(
                direction=Direction.LONG,
                entry=entry,
                stop_loss=stop,
                tp1=tp1,
                tp2=tp2,
                tp3=tp3,
                level=ref.price,
                level_label=ref.label,
                sl_pips=sl_pips,
                reason=(
                    f"{VIP2_SIGNAL_TAG} LONG: {ref.label} @ {ref.price:.2f}, "
                    f"close entry {entry:.2f}, SL {sl_pips:.1f} pips"
                ),
            ), "VIP2 setup detected"

        pierce = sweep["high"] - ref.price
        if not (
            min_pierce <= pierce <= max_pierce
            and sweep["close"] < ref.price
            and pre["high"] <= ref.price
        ):
            continue
        entry = float(sweep["close"])
        stop = float(sweep["high"]) + STOP_BUFFER_PIPS * pip
        if entry >= stop:
            continue
        risk = stop - entry
        sl_pips = risk / pip
        if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
            return None, f"NO VIP2: SL {sl_pips:.1f} pips outside {MIN_SL_PIPS}-{MAX_SL_PIPS}"
        tp1 = entry - TP1_R * risk
        tp2 = entry - TP2_R * risk
        tp3 = opposite_liquidity_tp3(
            direction=Direction.SHORT,
            entry=entry,
            risk=risk,
            levels=levels,
            pip=pip,
        )
        return TurtleSoupSetup(
            direction=Direction.SHORT,
            entry=entry,
            stop_loss=stop,
            tp1=tp1,
            tp2=tp2,
            tp3=tp3,
            level=ref.price,
            level_label=ref.label,
            sl_pips=sl_pips,
            reason=(
                f"{VIP2_SIGNAL_TAG} SHORT: {ref.label} @ {ref.price:.2f}, "
                f"close entry {entry:.2f}, SL {sl_pips:.1f} pips"
            ),
        ), "VIP2 setup detected"

    return None, "NO VIP2: no turtle soup on last M5 candle"


def build_turtle_soup_signal(
    setup: TurtleSoupSetup,
    symbol: str,
    *,
    deposit: float = 200.0,
) -> TradeSignal:
    from config.sl_config import calculate_lot_size_for_symbol

    display = resolve_symbol(symbol).display
    signal = TradeSignal(
        direction=setup.direction,
        entry=setup.entry,
        stop_loss=setup.stop_loss,
        tp1=setup.tp1,
        tp2=setup.tp2,
        tp3=setup.tp3,
        confidence=0.68,
        reason=setup.reason,
        lot_size=calculate_lot_size_for_symbol(deposit, display),
    )
    return align_trade_signal_direction(signal)


def analyze_turtle_soup_scalp_symbol(
    symbol: str,
    *,
    provider: MarketDataProvider,
    publish_gate: ScalpPublishGate | None = None,
) -> tuple[TradeSignal | None, dict[str, Any] | None, ScalpAnalysisResult]:
    display = resolve_symbol(symbol).display

    def rejected(message: str, context: dict[str, Any] | None = None):
        return None, context, ScalpAnalysisResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            message=message,
        )

    if not is_turtle_soup_scalp_enabled(display):
        return rejected(f"NO VIP2: disabled for {display}")

    gate = publish_gate or build_turtle_soup_gate()
    context = provider.to_context(
        display,
        "5m",
        limit=TURTLE_SOUP_CANDLE_LIMIT,
        include_h4_trend=False,
    )
    timestamp = context.get("timestamp")
    if isinstance(timestamp, datetime) and not is_vip2_core_session(timestamp):
        return rejected("NO VIP2: outside 08-16 UTC core session", context)

    setup, reason = detect_turtle_soup_setup(
        context.get("candles", []),
        symbol=display,
    )
    if setup is None:
        return rejected(reason, context)

    if isinstance(timestamp, datetime):
        allowed, block_reason = gate.can_publish(display, timestamp)
        if not allowed:
            return rejected(block_reason or "NO VIP2: publish gate blocked", context)

    signal = build_turtle_soup_signal(setup, display)
    if isinstance(timestamp, datetime):
        gate.record(display, timestamp)

    return signal, context, ScalpAnalysisResult(
        approved=True,
        direction=setup.direction,
        confidence=signal.confidence,
        message="VIP2 Turtle Soup signal approved",
    )
