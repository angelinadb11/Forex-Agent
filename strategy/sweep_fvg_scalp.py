"""Premium VIP scalp: Liquidity Sweep + FVG retest (M5, XAUUSD).

Detects session/pool level sweep with FVG on the impulse candle, entry at the
FVG edge (limit retest), stop behind sweep wick, TP1=1R / TP2=2R partial.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

from agents.base import Direction
from agents.liquidity_agent import _find_equal_levels, _find_swing_points
from agents.session_agent import is_london_or_new_york_session
from backtest.engine import candle_timestamp
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, align_trade_signal_direction
from strategy.scalp_mode import ScalpAnalysisResult, ScalpPublishGate
from tracking.trade_pnl import pip_size_for_symbol

SWEEP_FVG_TIMEFRAME = "5m"
SWEEP_FVG_SYMBOLS = frozenset({"XAUUSD"})
SWEEP_FVG_CANDLE_LIMIT = 700

MIN_PIERCE_PIPS = 5.0
MAX_PIERCE_PIPS = 60.0
STOP_BUFFER_PIPS = 3.0
MIN_SL_PIPS = 5.0
MAX_SL_PIPS = 40.0
TP2_R = 2.0
ASIA_START_HOUR = 0
ASIA_END_HOUR = 8
POOL_LOOKBACK = 240
MIN_POOL_TOUCHES = 2
SWING_LOOKBACK = 2
EQUAL_TOLERANCE_PCT = 0.0008

PREMIUM_MAX_SIGNALS_PER_DAY = 3
PREMIUM_MIN_INTERVAL_SECONDS = 900

LevelMode = Literal["asia", "pools", "all"]
DEFAULT_LEVEL_MODE: LevelMode = "all"

VIP_SIGNAL_TAG = "VIP Sweep+FVG"


@dataclass(frozen=True)
class RefLevel:
    price: float
    kind: Literal["high", "low"]
    label: str


@dataclass(frozen=True)
class SweepFvgSetup:
    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    level: float
    level_label: str
    sl_pips: float
    reason: str


def is_sweep_fvg_scalp_enabled(symbol: str) -> bool:
    return resolve_symbol(symbol).display in SWEEP_FVG_SYMBOLS


def build_premium_scalp_gate() -> ScalpPublishGate:
    return ScalpPublishGate(
        min_interval_seconds=PREMIUM_MIN_INTERVAL_SECONDS,
        max_signals_per_day=PREMIUM_MAX_SIGNALS_PER_DAY,
    )


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def build_timestamps(candles: list) -> list[datetime]:
    return [_utc(candle_timestamp(candles, i)) for i in range(len(candles))]


def asia_range_for_day(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    now = timestamps[index]
    if now.hour < ASIA_END_HOUR:
        return None
    day = now.date()
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index, -1, -1):
        ts = timestamps[j]
        if ts.date() != day:
            if ts.date() < day:
                break
            continue
        if ASIA_START_HOUR <= ts.hour < ASIA_END_HOUR:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
    if not highs:
        return None
    return max(highs), min(lows)


def prev_hour_range(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    now = timestamps[index]
    target_hour = (
        now.replace(minute=0, second=0, microsecond=0).timestamp() - 3600
    )
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index, -1, -1):
        ts = timestamps[j].replace(minute=0, second=0, microsecond=0).timestamp()
        if ts == target_hour:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
        elif ts < target_hour:
            break
    if not highs:
        return None
    return max(highs), min(lows)


def liquidity_pool_levels(
    candles: list,
    index: int,
    *,
    lookback: int = POOL_LOOKBACK,
) -> list[RefLevel]:
    import pandas as pd

    start = max(0, index - lookback)
    window = candles[start:index]
    if len(window) < 60:
        return []

    df = pd.DataFrame(
        {
            "open": [c["open"] for c in window],
            "high": [c["high"] for c in window],
            "low": [c["low"] for c in window],
            "close": [c["close"] for c in window],
        }
    )
    swing_highs, swing_lows = _find_swing_points(df, lookback=SWING_LOOKBACK)
    equal_highs = _find_equal_levels(swing_highs, tolerance_pct=EQUAL_TOLERANCE_PCT)
    equal_lows = _find_equal_levels(swing_lows, tolerance_pct=EQUAL_TOLERANCE_PCT)

    levels: list[RefLevel] = []
    for cluster in equal_highs:
        if cluster.count >= MIN_POOL_TOUCHES:
            levels.append(RefLevel(cluster.level, "high", f"pool-high x{cluster.count}"))
    for cluster in equal_lows:
        if cluster.count >= MIN_POOL_TOUCHES:
            levels.append(RefLevel(cluster.level, "low", f"pool-low x{cluster.count}"))
    return levels


def reference_levels_for_bar(
    candles: list,
    timestamps: list[datetime],
    index: int,
    mode: LevelMode,
) -> list[RefLevel]:
    levels: list[RefLevel] = []
    if mode in ("asia", "all"):
        asia = asia_range_for_day(candles, timestamps, index)
        if asia:
            levels.append(RefLevel(asia[0], "high", "asia-high"))
            levels.append(RefLevel(asia[1], "low", "asia-low"))
        hour = prev_hour_range(candles, timestamps, index)
        if hour:
            levels.append(RefLevel(hour[0], "high", "hour-high"))
            levels.append(RefLevel(hour[1], "low", "hour-low"))
    if mode in ("pools", "all"):
        levels.extend(liquidity_pool_levels(candles, index))
    return levels


def detect_sweep_fvg_setup(
    candles: list[dict[str, Any]],
    *,
    index: int | None = None,
    level_mode: LevelMode = DEFAULT_LEVEL_MODE,
    symbol: str = "XAUUSD",
) -> tuple[SweepFvgSetup | None, str]:
    """Detect sweep+FVG on confirm candle ``index`` (default: last candle)."""
    if len(candles) < 4:
        return None, "NO VIP: need at least 4 M5 candles"

    confirm_index = len(candles) - 1 if index is None else index
    if confirm_index < 3:
        return None, "NO VIP: insufficient history"

    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    timestamps = build_timestamps(candles)

    sweep = candles[confirm_index - 1]
    pre = candles[confirm_index - 2]
    confirm = candles[confirm_index]
    levels = reference_levels_for_bar(candles, timestamps, confirm_index - 1, level_mode)
    if not levels:
        return None, "NO VIP: no reference levels (asia/hour/pools)"

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
            if confirm["low"] <= pre["high"]:
                continue
            entry = float(confirm["low"])
            stop = float(sweep["low"]) - STOP_BUFFER_PIPS * pip
            if entry <= stop:
                continue
            risk = entry - stop
            sl_pips = risk / pip
            if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
                return None, f"NO VIP: SL {sl_pips:.1f} pips outside {MIN_SL_PIPS}-{MAX_SL_PIPS}"
            return SweepFvgSetup(
                direction=Direction.LONG,
                entry=entry,
                stop_loss=stop,
                tp1=entry + risk,
                tp2=entry + TP2_R * risk,
                level=ref.price,
                level_label=ref.label,
                sl_pips=sl_pips,
                reason=(
                    f"{VIP_SIGNAL_TAG} LONG: {ref.label} @ {ref.price:.2f}, "
                    f"FVG retest {entry:.2f}, SL {sl_pips:.1f} pips"
                ),
            ), "VIP setup detected"

        pierce = sweep["high"] - ref.price
        if not (
            min_pierce <= pierce <= max_pierce
            and sweep["close"] < ref.price
            and pre["high"] <= ref.price
        ):
            continue
        if confirm["high"] >= pre["low"]:
            continue
        entry = float(confirm["high"])
        stop = float(sweep["high"]) + STOP_BUFFER_PIPS * pip
        if entry >= stop:
            continue
        risk = stop - entry
        sl_pips = risk / pip
        if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
            return None, f"NO VIP: SL {sl_pips:.1f} pips outside {MIN_SL_PIPS}-{MAX_SL_PIPS}"
        return SweepFvgSetup(
            direction=Direction.SHORT,
            entry=entry,
            stop_loss=stop,
            tp1=entry - risk,
            tp2=entry - TP2_R * risk,
            level=ref.price,
            level_label=ref.label,
            sl_pips=sl_pips,
            reason=(
                f"{VIP_SIGNAL_TAG} SHORT: {ref.label} @ {ref.price:.2f}, "
                f"FVG retest {entry:.2f}, SL {sl_pips:.1f} pips"
            ),
        ), "VIP setup detected"

    return None, "NO VIP: no sweep+FVG on last M5 candle"


def build_sweep_fvg_signal(
    setup: SweepFvgSetup,
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
        tp3=setup.tp2,
        confidence=0.72,
        reason=setup.reason,
        lot_size=calculate_lot_size_for_symbol(deposit, display),
    )
    return align_trade_signal_direction(signal)


def analyze_sweep_fvg_scalp_symbol(
    symbol: str,
    *,
    provider: MarketDataProvider,
    publish_gate: ScalpPublishGate | None = None,
    level_mode: LevelMode = DEFAULT_LEVEL_MODE,
) -> tuple[TradeSignal | None, dict[str, Any] | None, ScalpAnalysisResult]:
    display = resolve_symbol(symbol).display

    def rejected(message: str, context: dict[str, Any] | None = None):
        return None, context, ScalpAnalysisResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            message=message,
        )

    if not is_sweep_fvg_scalp_enabled(display):
        return rejected(f"NO VIP: disabled for {display}")

    gate = publish_gate or build_premium_scalp_gate()
    context = provider.to_context(
        display,
        SWEEP_FVG_TIMEFRAME,
        limit=SWEEP_FVG_CANDLE_LIMIT,
        include_h4_trend=False,
    )
    timestamp = context.get("timestamp")
    if isinstance(timestamp, datetime) and not is_london_or_new_york_session(timestamp):
        return rejected("NO VIP: outside London/NY session", context)

    setup, reason = detect_sweep_fvg_setup(
        context.get("candles", []),
        level_mode=level_mode,
        symbol=display,
    )
    if setup is None:
        return rejected(reason, context)

    if isinstance(timestamp, datetime):
        allowed, block_reason = gate.can_publish(display, timestamp)
        if not allowed:
            return rejected(block_reason or "NO VIP: publish gate blocked", context)

    signal = build_sweep_fvg_signal(setup, display)
    if isinstance(timestamp, datetime):
        gate.record(display, timestamp)

    return signal, context, ScalpAnalysisResult(
        approved=True,
        direction=setup.direction,
        confidence=signal.confidence,
        message="VIP Sweep+FVG signal approved",
    )
