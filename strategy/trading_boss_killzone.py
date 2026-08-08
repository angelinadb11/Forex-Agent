"""Trading Boss: Liquidity Sweep + OB/FVG in London/NY Killzones.

Multi-agent pipeline (bias → liquidity sweep → structure → execution) with
weighted confidence scoring. Main channel only — scalp streams are unchanged.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import pandas as pd
import requests

from agents.base import AgentResult, Direction
from agents.smc_agent import (
    MarketStructure,
    _candles_to_dataframe,
    _find_order_block,
    _find_recent_fair_value_gap,
    _find_swing_points,
    analyze_smc,
)
from config.sl_config import calculate_lot_size_for_symbol
from config.symbols import resolve_symbol
from data import MarketDataProvider
from news.models import NewsAction
from signal_generator import (
    MAIN_MIN_RR_TO_TP1,
    TP_STEP_R,
    TradeSignal,
    align_trade_signal_direction,
    build_take_profit_levels,
    calculate_atr,
    planned_rr_to_target,
)

MIN_RR_TP2 = MAIN_MIN_RR_TO_TP1 + TP_STEP_R
from strategy.signal_filter import (
    FilterResult,
    SignalFilter,
    resolve_min_confidence,
)
from strategy.sweep_fvg_scalp import (
    RefLevel,
    asia_range_for_day,
    build_timestamps,
    liquidity_pool_levels,
    reference_levels_for_bar,
)
from tracking.trade_pnl import pip_size_for_symbol

LOGGER = logging.getLogger(__name__)

# Legacy aliases (standard profile defaults).
TRADING_BOSS_TIMEFRAME = "15m"
STRUCTURE_TIMEFRAME = "5m"
SWEEP_TIMEFRAME = "5m"


@dataclass(frozen=True)
class KillzoneProfile:
    """Timeframe and risk preset for Trading Boss killzone setups."""

    name: str
    context_tf: str
    sweep_tf: str
    htf_filter_tf: str
    sweep_reclaim_bars: int
    sweep_lookback_bars: int
    sl_atr_buffer_mult: float
    min_wick_atr_mult: float
    max_sl_pips: dict[str, float]
    min_sl_pips: dict[str, float]


KILLZONE_PROFILE_STANDARD = KillzoneProfile(
    name="standard",
    context_tf="15m",
    sweep_tf="5m",
    htf_filter_tf="15m",
    sweep_reclaim_bars=3,
    sweep_lookback_bars=80,
    sl_atr_buffer_mult=0.35,
    min_wick_atr_mult=0.15,
    max_sl_pips={"XAUUSD": 80.0, "default": 150.0},
    min_sl_pips={"XAUUSD": 15.0, "default": 20.0},
)

KILLZONE_PROFILE_PRECISION = KillzoneProfile(
    name="precision",
    context_tf="5m",
    sweep_tf="1m",
    htf_filter_tf="5m",
    sweep_reclaim_bars=4,
    sweep_lookback_bars=120,
    sl_atr_buffer_mult=0.25,
    min_wick_atr_mult=0.08,
    max_sl_pips={"XAUUSD": 100.0, "BTCUSDT": 250.0, "default": 80.0},
    min_sl_pips={"XAUUSD": 10.0, "BTCUSDT": 80.0, "default": 10.0},
)

DEFAULT_KILLZONE_PROFILE = KILLZONE_PROFILE_PRECISION


def resolve_killzone_profile(profile_name: str | None = None) -> KillzoneProfile:
    """Return Trading Boss profile from env/config (default: precision / smaller SL)."""
    key = (profile_name or os.getenv("TRADING_BOSS_KILLZONE_PROFILE", "precision")).strip().lower()
    if key in {"standard", "legacy", "m15"}:
        return KILLZONE_PROFILE_STANDARD
    return KILLZONE_PROFILE_PRECISION


def _profile_sl_limits(symbol: str, profile: KillzoneProfile) -> tuple[float, float]:
    display = resolve_symbol(symbol).display
    max_pips = profile.max_sl_pips.get(display, profile.max_sl_pips["default"])
    min_pips = profile.min_sl_pips.get(display, profile.min_sl_pips["default"])
    return min_pips, max_pips


def is_fresh_sweep(sweep: SweepEvent, candle_count: int) -> bool:
    """Sweep reclaim is recent enough to treat as a new setup."""
    return sweep.reclaim_index >= candle_count - FRESH_SWEEP_MAX_LAG_BARS


def resolve_killzone_min_confidence(
    symbol: str,
    *,
    default: float,
) -> float:
    try:
        display = resolve_symbol(symbol).display
    except ValueError:
        display = symbol.upper()
    return KILLZONE_MIN_CONFIDENCE.get(display, default)

# Kyiv UTC+3 → UTC killzones
LONDON_KILLZONE_START = (6, 0)
LONDON_KILLZONE_END = (8, 0)
NY_KILLZONE_START = (11, 30)
NY_KILLZONE_END = (13, 30)
ASIAN_KILLZONE_START = (0, 0)
ASIAN_KILLZONE_END = (3, 0)

SWEEP_RECLAIM_BARS = 3
SWEEP_LOOKBACK_BARS = 80
MIN_WICK_ATR_MULT = 0.15
SL_ATR_BUFFER_MULT = 0.35
COUNTER_BIAS_WEIGHT = 0.22
ALIGNMENT_BONUS = 0.10
KILLZONE_DECISION_MIN_SCORE = 0.30
FRESH_SWEEP_MAX_LAG_BARS = 3
KILLZONE_MIN_CONFIDENCE: dict[str, float] = {
    "XAUUSD": 0.42,
}

KILLZONE_AGENT_WEIGHTS: dict[str, float] = {
    "bias": 0.25,
    "liquidity": 0.25,
    "structure": 0.30,
    "session": 0.10,
    "execution": 0.10,
}

ZoneKind = Literal["OB", "FVG"]


@dataclass(frozen=True)
class KillzoneWindow:
    label: str
    start_hour: int
    start_minute: int
    end_hour: int
    end_minute: int


KILLZONE_WINDOWS = (
    KillzoneWindow("London", *LONDON_KILLZONE_START, *LONDON_KILLZONE_END),
    KillzoneWindow("NY", *NY_KILLZONE_START, *NY_KILLZONE_END),
)


@dataclass(frozen=True)
class KillzoneFrequencySettings:
    """Runtime frequency preset for main-channel Killzone signals."""

    label: str
    max_signals_per_window: int
    include_asian: bool


def resolve_killzone_frequency(frequency: str | None = None) -> KillzoneFrequencySettings:
    key = (frequency or os.getenv("TRADING_BOSS_KILLZONE_FREQUENCY", "balanced")).strip().lower()
    if key in {"daily", "day", "high"}:
        return KillzoneFrequencySettings("daily", 2, True)
    return KillzoneFrequencySettings("balanced", 1, False)


def resolve_trading_boss_dual_tier(enabled: str | None = None) -> bool:
    raw = (enabled or os.getenv("TRADING_BOSS_DUAL_TIER", "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class KillzoneTier:
    """Trading Boss signal tier (main channel only)."""

    label: str
    telegram_header: str
    max_signals_per_window: int
    include_asian: bool
    min_confidence: float
    require_strong_choch: bool
    require_ob_fvg_zone: bool
    require_bias_alignment: bool
    telegram_min_confidence: float


KILLZONE_TIER_ACTIVE = KillzoneTier(
    label="active",
    telegram_header="🔥 TRADING BOSS ACTIVE",
    max_signals_per_window=2,
    include_asian=True,
    min_confidence=0.42,
    require_strong_choch=False,
    require_ob_fvg_zone=False,
    require_bias_alignment=False,
    telegram_min_confidence=0.42,
)

KILLZONE_TIER_SELECT = KillzoneTier(
    label="select",
    telegram_header="💎 TRADING BOSS SELECT",
    max_signals_per_window=1,
    include_asian=False,
    min_confidence=0.55,
    require_strong_choch=True,
    require_ob_fvg_zone=True,
    require_bias_alignment=True,
    telegram_min_confidence=0.55,
)


def resolve_killzone_tiers(*, dual_tier: bool | None = None) -> tuple[KillzoneTier, ...]:
    if dual_tier if dual_tier is not None else resolve_trading_boss_dual_tier():
        return (KILLZONE_TIER_SELECT, KILLZONE_TIER_ACTIVE)
    return (KILLZONE_TIER_ACTIVE,) if resolve_killzone_frequency().label == "daily" else (
        KILLZONE_TIER_SELECT,
    )


def get_killzone_windows(*, include_asian: bool | None = None) -> tuple[KillzoneWindow, ...]:
    if include_asian is None:
        include_asian = resolve_killzone_frequency().include_asian
    windows: list[KillzoneWindow] = []
    if include_asian:
        windows.append(KillzoneWindow("Asian", *ASIAN_KILLZONE_START, *ASIAN_KILLZONE_END))
    windows.extend(KILLZONE_WINDOWS)
    return tuple(windows)


@dataclass(frozen=True)
class BiasAnalysis:
    direction: Direction
    zone: Literal["premium", "discount", "equilibrium"]
    confidence: float
    reason: str


@dataclass(frozen=True)
class SweepEvent:
    direction: Direction
    level: RefLevel
    sweep_index: int
    sweep_extreme: float
    reclaim_index: int
    wick_depth: float
    confidence: float
    reason: str


@dataclass(frozen=True)
class StructureSetup:
    direction: Direction
    entry: float
    zone_low: float
    zone_high: float
    zone_kind: ZoneKind
    confidence: float
    reason: str
    choch_confirmed: bool = False
    choch_only: bool = False


@dataclass(frozen=True)
class KillzoneSetup:
    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confidence: float
    reason: str
    sweep: SweepEvent
    structure: StructureSetup
    bias: BiasAnalysis


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def _minutes_since_midnight(ts: datetime) -> int:
    ts = _utc(ts)
    return ts.hour * 60 + ts.minute


def _window_contains(ts: datetime, window: KillzoneWindow) -> bool:
    current = _minutes_since_midnight(ts)
    start = window.start_hour * 60 + window.start_minute
    end = window.end_hour * 60 + window.end_minute
    return start <= current < end


def active_killzone(
    ts: datetime | None,
    *,
    include_asian: bool | None = None,
) -> KillzoneWindow | None:
    if ts is None:
        return None
    for window in get_killzone_windows(include_asian=include_asian):
        if _window_contains(ts, window):
            return window
    return None


def active_killzone_for_tier(ts: datetime | None, tier: KillzoneTier) -> KillzoneWindow | None:
    return active_killzone(ts, include_asian=tier.include_asian)


def is_killzone_session(
    ts: datetime | None,
    *,
    include_asian: bool | None = None,
) -> tuple[bool, str]:
    window = active_killzone(ts, include_asian=include_asian)
    if window is None:
        if include_asian if include_asian is not None else resolve_killzone_frequency().include_asian:
            return False, "поза Killzone (Asia 03:00–06:00 / London 09:00–11:00 / NY 14:30–16:30 Kyiv)"
        return False, "поза Killzone (London 09:00–11:00 / NY 14:30–16:30 Kyiv)"
    return True, f"{window.label} Killzone активна"


def is_killzone_session_for_tier(ts: datetime | None, tier: KillzoneTier) -> tuple[bool, str]:
    return is_killzone_session(ts, include_asian=tier.include_asian)


def is_in_trading_boss_killzone_union(ts: datetime | None) -> bool:
    """True when inside any tier window (Asian + London + NY)."""
    return active_killzone(ts, include_asian=True) is not None


def killzone_window_key(
    ts: datetime | None,
    *,
    include_asian: bool | None = None,
) -> tuple[str, str] | None:
    """Unique slot per calendar day and Killzone window."""
    window = active_killzone(ts, include_asian=include_asian)
    if window is None or ts is None:
        return None
    return (_utc(ts).date().isoformat(), window.label)


@dataclass
class KillzoneWindowGate:
    """Limit main-channel signals per Killzone window per day."""

    max_signals_per_window: int = 1
    include_asian: bool = False
    tier_label: str = "main"
    _counts: dict[tuple[str, str], int] = field(default_factory=dict)

    @classmethod
    def for_tier(cls, tier: KillzoneTier) -> KillzoneWindowGate:
        return cls(
            max_signals_per_window=tier.max_signals_per_window,
            include_asian=tier.include_asian,
            tier_label=tier.label,
        )

    def can_take(self, ts: datetime | None) -> tuple[bool, str]:
        key = killzone_window_key(ts, include_asian=self.include_asian)
        if key is None:
            return False, f"TB {self.tier_label}: поза Killzone"
        used = self._counts.get(key, 0)
        if used >= self.max_signals_per_window:
            return (
                False,
                f"TB {self.tier_label} slot full ({key[1]} {key[0]}, {used}/{self.max_signals_per_window})",
            )
        return True, "ok"

    def record(self, ts: datetime | None) -> None:
        key = killzone_window_key(ts, include_asian=self.include_asian)
        if key is not None:
            self._counts[key] = self._counts.get(key, 0) + 1


def prev_day_range(
    candles: list[dict[str, Any]],
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    current_day = timestamps[index].date()
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index - 1, -1, -1):
        day = timestamps[j].date()
        if day >= current_day:
            continue
        if day < current_day - timedelta(days=1):
            break
        highs.append(float(candles[j]["high"]))
        lows.append(float(candles[j]["low"]))
    if not highs:
        return None
    return max(highs), min(lows)


def build_killzone_liquidity_levels(
    candles: list[dict[str, Any]],
    timestamps: list[datetime],
    index: int,
) -> list[RefLevel]:
    levels = reference_levels_for_bar(candles, timestamps, index, "all")
    day_range = prev_day_range(candles, timestamps, index)
    if day_range:
        levels.append(RefLevel(day_range[0], "high", "prev-day-high"))
        levels.append(RefLevel(day_range[1], "low", "prev-day-low"))
    return levels


def _htf_close_for_ltf_bar(
    ltf_ts: datetime,
    htf_candles: list[dict[str, Any]],
    htf_timestamps: list[datetime],
) -> float | None:
    target = _utc(ltf_ts)
    for j in range(len(htf_timestamps) - 1, -1, -1):
        ts = htf_timestamps[j]
        if ts <= target:
            return float(htf_candles[j]["close"])
    return None


def _premium_discount(
    price: float,
    swing_high: float,
    swing_low: float,
) -> Literal["premium", "discount", "equilibrium"]:
    midpoint = (swing_high + swing_low) / 2
    span = swing_high - swing_low
    if span <= 0:
        return "equilibrium"
    if price >= midpoint + span * 0.05:
        return "premium"
    if price <= midpoint - span * 0.05:
        return "discount"
    return "equilibrium"


def analyze_bias(
    h1_candles: list[dict[str, Any]],
    h4_candles: list[dict[str, Any]] | None,
) -> BiasAnalysis:
    h1_df = _candles_to_dataframe({"candles": h1_candles})
    h1_smc = analyze_smc(h1_df, swing_lookback=2)
    swing_highs, swing_lows = _find_swing_points(h1_df, lookback=2)
    price = float(h1_df.iloc[-1]["close"])

    zone = "equilibrium"
    if swing_highs and swing_lows:
        zone = _premium_discount(price, swing_highs[-1].price, swing_lows[-1].price)

    bullish = 0.0
    bearish = 0.0
    reasons: list[str] = []

    if h1_smc.choch == "bullish":
        bullish += 0.45
        reasons.append("H1 bullish CHoCH")
    elif h1_smc.bos == "bullish":
        bullish += 0.35
        reasons.append("H1 bullish BOS")
    elif h1_smc.structure == MarketStructure.BULLISH:
        bullish += 0.25
        reasons.append("H1 bullish structure")

    if h1_smc.choch == "bearish":
        bearish += 0.45
        reasons.append("H1 bearish CHoCH")
    elif h1_smc.bos == "bearish":
        bearish += 0.35
        reasons.append("H1 bearish BOS")
    elif h1_smc.structure == MarketStructure.BEARISH:
        bearish += 0.25
        reasons.append("H1 bearish structure")

    if h4_candles:
        h4_df = _candles_to_dataframe({"candles": h4_candles})
        h4_smc = analyze_smc(h4_df, swing_lookback=2)
        if h4_smc.choch == "bullish" or h4_smc.bos == "bullish":
            bullish += 0.25
            reasons.append("H4 bullish BOS/CHoCH")
        elif h4_smc.choch == "bearish" or h4_smc.bos == "bearish":
            bearish += 0.25
            reasons.append("H4 bearish BOS/CHoCH")

    if zone == "discount":
        bullish += 0.10
        reasons.append("ціна в discount")
    elif zone == "premium":
        bearish += 0.10
        reasons.append("ціна в premium")

    if bullish > bearish and bullish >= 0.30:
        direction = Direction.LONG
        confidence = min(0.90, bullish)
    elif bearish > bullish and bearish >= 0.30:
        direction = Direction.SHORT
        confidence = min(0.90, bearish)
    else:
        direction = Direction.NEUTRAL
        confidence = max(bullish, bearish, 0.25)

    reason = ", ".join(reasons) if reasons else "HTF bias unclear"
    return BiasAnalysis(
        direction=direction,
        zone=zone,
        confidence=round(confidence, 2),
        reason=reason,
    )


def _sweep_wick_depth(candle: dict[str, Any], level: RefLevel) -> float:
    if level.kind == "low":
        return max(0.0, float(level.price) - float(candle["low"]))
    return max(0.0, float(candle["high"]) - float(level.price))


def detect_liquidity_sweep(
    sweep_candles: list[dict[str, Any]],
    sweep_timestamps: list[datetime],
    *,
    htf_candles: list[dict[str, Any]],
    htf_timestamps: list[datetime],
    bias: BiasAnalysis,
    atr: float,
    profile: KillzoneProfile = DEFAULT_KILLZONE_PROFILE,
    recent_bars_only: int | None = None,
) -> SweepEvent | None:
    reclaim_bars = profile.sweep_reclaim_bars
    if len(sweep_candles) < reclaim_bars + 5:
        return None

    search_end = len(sweep_candles) - 1
    search_start = max(0, search_end - profile.sweep_lookback_bars)
    if recent_bars_only is not None:
        search_start = max(search_start, search_end - recent_bars_only)

    best: SweepEvent | None = None
    for i in range(search_start, search_end):
        levels = build_killzone_liquidity_levels(
            sweep_candles,
            sweep_timestamps,
            max(0, i - 1),
        )
        sweep = sweep_candles[i]
        prev = sweep_candles[i - 1] if i > 0 else None
        if prev is None:
            continue

        htf_close = _htf_close_for_ltf_bar(
            sweep_timestamps[i],
            htf_candles,
            htf_timestamps,
        )

        for ref in levels:
            wick = _sweep_wick_depth(sweep, ref)
            if wick < profile.min_wick_atr_mult * atr:
                continue

            if ref.kind == "low":
                if not (sweep["low"] < ref.price and sweep["close"] > ref.price):
                    continue
                if prev["low"] < ref.price:
                    continue
                direction = Direction.LONG
            else:
                if not (sweep["high"] > ref.price and sweep["close"] < ref.price):
                    continue
                if prev["high"] > ref.price:
                    continue
                direction = Direction.SHORT

            counter_bias = bias.direction not in (Direction.NEUTRAL, direction)

            reclaim_index = i
            for j in range(i, min(i + 1 + reclaim_bars, len(sweep_candles))):
                close = float(sweep_candles[j]["close"])
                if direction == Direction.LONG and close > ref.price:
                    reclaim_index = j
                    break
                if direction == Direction.SHORT and close < ref.price:
                    reclaim_index = j
                    break
            else:
                continue

            wick_quality = min(1.0, wick / (0.5 * atr))
            confidence = round(0.45 + 0.35 * wick_quality, 2)
            if counter_bias:
                confidence = round(confidence * 0.65, 2)
            if htf_close is not None:
                if direction == Direction.LONG and htf_close < ref.price:
                    confidence = round(confidence * 0.85, 2)
                elif direction == Direction.SHORT and htf_close > ref.price:
                    confidence = round(confidence * 0.85, 2)
            event = SweepEvent(
                direction=direction,
                level=ref,
                sweep_index=i,
                sweep_extreme=float(sweep["low"] if direction == Direction.LONG else sweep["high"]),
                reclaim_index=reclaim_index,
                wick_depth=wick,
                confidence=confidence,
                reason=(
                    f"sweep {ref.label} @ {ref.price:.2f}, "
                    f"wick {wick:.2f}, reclaim bar +{reclaim_index - i}"
                ),
            )
            if best is None or reclaim_index >= best.reclaim_index:
                best = event

    return best


def _zone_matches_direction(kind: str, direction: Direction) -> bool:
    if direction == Direction.LONG:
        return kind == "bullish"
    if direction == Direction.SHORT:
        return kind == "bearish"
    return False


def find_structure_setup(
    candles_m5: list[dict[str, Any]],
    sweep: SweepEvent,
) -> StructureSetup | None:
    slice_candles = candles_m5[sweep.reclaim_index :]
    if len(slice_candles) < 12:
        start = max(0, min(sweep.sweep_index, len(candles_m5) - 12))
        slice_candles = candles_m5[start:]
    if len(slice_candles) < 5:
        return None

    df = pd.DataFrame(
        {
            "open": [float(c["open"]) for c in slice_candles],
            "high": [float(c["high"]) for c in slice_candles],
            "low": [float(c["low"]) for c in slice_candles],
            "close": [float(c["close"]) for c in slice_candles],
        }
    )
    analysis = analyze_smc(df, swing_lookback=2)
    choch_ok = False
    if sweep.direction == Direction.LONG:
        choch_ok = analysis.choch == "bullish" or analysis.bos == "bullish"
    else:
        choch_ok = analysis.choch == "bearish" or analysis.bos == "bearish"

    ob = _find_order_block(df, lookback=max(10, len(df) - 1))
    fvg = _find_recent_fair_value_gap(df, lookback=max(10, len(df) - 1))

    zones: list[tuple[float, float, ZoneKind, float]] = []
    if ob and _zone_matches_direction(ob.kind, sweep.direction):
        zones.append((ob.low, ob.high, "OB", 0.55))
    if fvg and _zone_matches_direction(fvg.kind, sweep.direction) and not fvg.filled:
        zones.append((fvg.low, fvg.high, "FVG", 0.50))

    if not zones:
        if not choch_ok:
            return None
        current_close = float(df.iloc[-1]["close"])
        pad = max(0.5, sweep.wick_depth * 0.35)
        return StructureSetup(
            direction=sweep.direction,
            entry=current_close,
            zone_low=current_close - pad,
            zone_high=current_close + pad,
            zone_kind="OB",
            confidence=0.48,
            reason=f"CHoCH-only {sweep.direction.value}, no OB/FVG",
            choch_confirmed=choch_ok,
            choch_only=True,
        )

    zone_low, zone_high, zone_kind, base = max(zones, key=lambda item: item[3])
    current_close = float(df.iloc[-1]["close"])
    if zone_low <= current_close <= zone_high:
        entry = current_close
        entry_mode = "close in zone"
    else:
        entry = (zone_low + zone_high) / 2
        entry_mode = "limit mid-zone"

    confidence = base + (0.20 if choch_ok else 0.08)
    if not choch_ok:
        confidence -= 0.08

    reason = (
        f"{zone_kind} {sweep.direction.value} {zone_low:.2f}-{zone_high:.2f}, "
        f"{'CHoCH ok' if choch_ok else 'CHoCH weak'}, {entry_mode}"
    )
    return StructureSetup(
        direction=sweep.direction,
        entry=entry,
        zone_low=zone_low,
        zone_high=zone_high,
        zone_kind=zone_kind,
        confidence=round(max(0.2, min(0.95, confidence)), 2),
        reason=reason,
        choch_confirmed=choch_ok,
        choch_only=False,
    )


def _next_liquidity_target(
    direction: Direction,
    entry: float,
    risk: float,
    levels: list[RefLevel],
) -> float | None:
    if direction == Direction.LONG:
        candidates = [level.price for level in levels if level.kind == "high" and level.price > entry]
        if candidates:
            return min(candidates)
    else:
        candidates = [level.price for level in levels if level.kind == "low" and level.price < entry]
        if candidates:
            return max(candidates)
    return None


def build_killzone_setup(
    *,
    sweep: SweepEvent,
    structure: StructureSetup,
    sweep_candles: list[dict[str, Any]],
    sweep_timestamps: list[datetime],
    atr: float,
    symbol: str,
    profile: KillzoneProfile = DEFAULT_KILLZONE_PROFILE,
) -> KillzoneSetup | None:
    pip = pip_size_for_symbol(symbol) or 1.0
    direction = sweep.direction
    entry = structure.entry

    buffer = max(profile.sl_atr_buffer_mult * atr, 2.0 * pip)
    if direction == Direction.LONG:
        stop_loss = sweep.sweep_extreme - buffer
        if stop_loss >= entry:
            stop_loss = entry - max(5.0 * pip, buffer)
    else:
        stop_loss = sweep.sweep_extreme + buffer
        if stop_loss <= entry:
            stop_loss = entry + max(5.0 * pip, buffer)

    risk = abs(entry - stop_loss)
    if risk <= 0:
        return None

    sl_pips = risk / pip
    min_sl_pips, max_sl_pips = _profile_sl_limits(symbol, profile)
    if sl_pips > max_sl_pips or sl_pips < min_sl_pips:
        return None

    tp1, tp2_default, tp3_default = build_take_profit_levels(
        direction,
        entry,
        risk,
        min_rr_tp1=MAIN_MIN_RR_TO_TP1,
    )

    levels = build_killzone_liquidity_levels(
        sweep_candles,
        sweep_timestamps,
        len(sweep_candles) - 1,
    )
    liquidity_tp = _next_liquidity_target(direction, entry, risk, levels)
    min_tp2 = entry + MIN_RR_TP2 * risk if direction == Direction.LONG else entry - MIN_RR_TP2 * risk
    if liquidity_tp is not None:
        if direction == Direction.LONG:
            tp2 = max(tp2_default, liquidity_tp)
            tp3 = max(tp3_default, tp2 + risk)
        else:
            tp2 = min(tp2_default, liquidity_tp)
            tp3 = min(tp3_default, tp2 - risk)
    else:
        tp2 = tp2_default
        tp3 = tp3_default

    if planned_rr_to_target(entry, tp2, risk) + 1e-6 < MIN_RR_TP2:
        tp2 = min_tp2
        tp3 = tp2 + risk if direction == Direction.LONG else tp2 - risk

    confidence = round(
        min(
            0.95,
            sweep.confidence * 0.35
            + structure.confidence * 0.45
            + 0.20,
        ),
        2,
    )
    reason = (
        f"Killzone {profile.name} {direction.value.upper()}: {sweep.reason} | "
        f"{structure.reason} | SL {sl_pips:.1f} pips, "
        f"TP1 {planned_rr_to_target(entry, tp1, risk):.1f}R / "
        f"TP2 {planned_rr_to_target(entry, tp2, risk):.1f}R"
    )
    return KillzoneSetup(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        confidence=confidence,
        reason=reason,
        sweep=sweep,
        structure=structure,
        bias=BiasAnalysis(Direction.NEUTRAL, "equilibrium", 0.0, ""),
    )


def run_killzone_agents(
    *,
    bias: BiasAnalysis,
    sweep: SweepEvent | None,
    structure: StructureSetup | None,
    timestamp: datetime | None,
    setup: KillzoneSetup | None,
) -> dict[str, AgentResult]:
    in_kz, kz_reason = (
        is_killzone_session(timestamp, include_asian=True)
        if resolve_trading_boss_dual_tier()
        else is_killzone_session(timestamp)
    )
    session_conf = 0.85 if in_kz else 0.20
    session_dir = setup.direction if setup else Direction.NEUTRAL

    bias_result = AgentResult(
        direction=bias.direction,
        confidence=bias.confidence,
        reason=f"Bias: {bias.reason} ({bias.zone})",
    )
    if sweep is None:
        liquidity_result = AgentResult(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="Liquidity: sweep не знайдено",
        )
    else:
        liquidity_result = AgentResult(
            direction=sweep.direction,
            confidence=sweep.confidence,
            reason=f"Liquidity: {sweep.reason}",
        )
    if structure is None:
        structure_result = AgentResult(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="Structure: CHoCH/OB/FVG не знайдено",
        )
    else:
        structure_result = AgentResult(
            direction=structure.direction,
            confidence=structure.confidence,
            reason=f"Structure: {structure.reason}",
        )
    session_result = AgentResult(
        direction=session_dir if in_kz else Direction.NEUTRAL,
        confidence=session_conf,
        reason=f"Session: {kz_reason}",
    )
    if setup is None:
        execution_result = AgentResult(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="Execution: setup не побудовано",
        )
    else:
        execution_result = AgentResult(
            direction=setup.direction,
            confidence=setup.confidence,
            reason=f"Execution: entry {setup.entry:.2f}, SL {setup.stop_loss:.2f}",
        )
    return {
        "bias": bias_result,
        "liquidity": liquidity_result,
        "structure": structure_result,
        "session": session_result,
        "execution": execution_result,
    }


def compute_killzone_decision(
    results: dict[str, AgentResult],
    *,
    bias: BiasAnalysis,
    in_killzone: bool,
) -> tuple[Direction, float]:
    long_score = 0.0
    short_score = 0.0
    for name, weight in KILLZONE_AGENT_WEIGHTS.items():
        result = results[name]
        if result.direction == Direction.LONG:
            long_score += result.confidence * weight
        elif result.direction == Direction.SHORT:
            short_score += result.confidence * weight

    if long_score > short_score and long_score >= KILLZONE_DECISION_MIN_SCORE:
        direction = Direction.LONG
        confidence = long_score
    elif short_score > long_score and short_score >= KILLZONE_DECISION_MIN_SCORE:
        direction = Direction.SHORT
        confidence = short_score
    else:
        return Direction.NEUTRAL, max(long_score, short_score)

    if bias.direction not in (Direction.NEUTRAL, direction):
        confidence = min(confidence, COUNTER_BIAS_WEIGHT)
    elif (
        bias.direction == direction
        and results["liquidity"].direction == direction
        and results["structure"].direction == direction
        and in_killzone
    ):
        confidence = min(1.0, confidence + ALIGNMENT_BONUS)

    if not in_killzone:
        confidence = max(0.0, confidence - 0.20)

    return direction, round(confidence, 2)


def evaluate_killzone_tier_filter(
    *,
    tier: KillzoneTier,
    signal_filter: SignalFilter,
    results: dict[str, AgentResult],
    setup: KillzoneSetup | None,
    direction: Direction,
    confidence: float,
    symbol: str,
    timestamp: datetime | None,
    bias: BiasAnalysis,
) -> FilterResult:
    tier_tag = tier.label.upper()
    if direction == Direction.NEUTRAL:
        return FilterResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: neutral killzone decision",
        )

    in_kz, kz_msg = is_killzone_session_for_tier(timestamp, tier)
    if not in_kz:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: {kz_msg}",
        )

    if setup is None:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: setup не зібрано",
        )

    if results["liquidity"].direction != direction:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: liquidity sweep не підтверджує напрямок",
        )
    if results["structure"].direction != direction:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: structure (CHoCH/OB/FVG) не підтверджує напрямок",
        )

    if tier.require_strong_choch and not setup.structure.choch_confirmed:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: CHoCH не підтверджено (потрібен сильний setup)",
        )

    if tier.require_ob_fvg_zone and setup.structure.choch_only:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: потрібен OB/FVG, не CHoCH-only",
        )

    if tier.require_bias_alignment and bias.direction not in (Direction.NEUTRAL, direction):
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE [{tier_tag}]: HTF bias не в напрямку угоди",
        )

    min_conf = tier.min_confidence
    if tier.label == "active":
        min_conf = resolve_killzone_min_confidence(
            symbol,
            default=resolve_min_confidence(
                symbol,
                default=signal_filter.min_confidence,
            ),
        )
    if confidence + 1e-6 < min_conf:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=(
                f"NO TRADE [{tier_tag}]: confidence {confidence:.2f} "
                f"below minimum {min_conf:.2f}"
            ),
        )

    news_warning = None
    if signal_filter.news_gate is not None and timestamp is not None:
        news = signal_filter.news_gate.evaluate(symbol, timestamp)
        if news.action == NewsAction.BLOCK:
            return FilterResult(
                approved=False,
                direction=direction,
                confidence=confidence,
                message=f"NO TRADE [{tier_tag}]: high-impact news window ({news.event.name if news.event else 'macro'})",
            )
        if news.action == NewsAction.WARN:
            news_warning = news.message

    return FilterResult(
        approved=True,
        direction=direction,
        confidence=confidence,
        message=f"TB {tier_tag} approved ({kz_msg})",
        news_warning=news_warning,
    )


def evaluate_killzone_filter(
    *,
    signal_filter: SignalFilter,
    results: dict[str, AgentResult],
    direction: Direction,
    confidence: float,
    symbol: str,
    timestamp: datetime | None,
) -> FilterResult:
    if direction == Direction.NEUTRAL:
        return FilterResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=confidence,
            message="NO TRADE: neutral killzone decision",
        )

    in_kz, kz_msg = is_killzone_session(timestamp)
    if not in_kz:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=f"NO TRADE: {kz_msg}",
        )

    if results["liquidity"].direction != direction:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message="NO TRADE: liquidity sweep не підтверджує напрямок",
        )
    if results["structure"].direction != direction:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message="NO TRADE: structure (CHoCH/OB/FVG) не підтверджує напрямок",
        )

    min_conf = resolve_killzone_min_confidence(
        symbol,
        default=resolve_min_confidence(
            symbol,
            default=signal_filter.min_confidence,
        ),
    )
    if confidence + 1e-6 < min_conf:
        return FilterResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=(
                f"NO TRADE: confidence {confidence:.2f} "
                f"below minimum {min_conf:.2f}"
            ),
        )

    news_warning = None
    if signal_filter.news_gate is not None and timestamp is not None:
        news = signal_filter.news_gate.evaluate(symbol, timestamp)
        if news.action == NewsAction.BLOCK:
            return FilterResult(
                approved=False,
                direction=direction,
                confidence=confidence,
                message=f"NO TRADE: high-impact news window ({news.event.name if news.event else 'macro'})",
            )
        if news.action == NewsAction.WARN:
            news_warning = news.message

    return FilterResult(
        approved=True,
        direction=direction,
        confidence=confidence,
        message=f"Killzone setup approved ({kz_msg})",
        news_warning=news_warning,
    )


def build_killzone_signal(
    setup: KillzoneSetup,
    symbol: str,
    *,
    confidence: float,
    deposit: float = 200.0,
) -> TradeSignal:
    display = resolve_symbol(symbol).display
    signal = TradeSignal(
        direction=setup.direction,
        entry=setup.entry,
        stop_loss=setup.stop_loss,
        tp1=setup.tp1,
        tp2=setup.tp2,
        tp3=setup.tp3,
        confidence=confidence,
        reason=setup.reason,
        lot_size=calculate_lot_size_for_symbol(deposit, display),
    )
    return align_trade_signal_direction(signal)


@dataclass(frozen=True)
class KillzoneTierOutcome:
    tier: KillzoneTier
    filter_result: FilterResult
    signal: TradeSignal | None


@dataclass(frozen=True)
class KillzoneScanResult:
    results: dict[str, AgentResult] | None
    context: dict | None
    setup: KillzoneSetup | None
    direction: Direction
    confidence: float
    tier_outcomes: tuple[KillzoneTierOutcome, ...]
    primary_filter: FilterResult

    @property
    def approved_tiers(self) -> tuple[KillzoneTierOutcome, ...]:
        return tuple(
            outcome
            for outcome in self.tier_outcomes
            if outcome.filter_result.approved and outcome.signal is not None
        )


def analyze_trading_boss_killzone_symbol(
    symbol: str,
    *,
    provider: MarketDataProvider,
    timeframe: str,
    candle_limit: int,
    signal_filter: SignalFilter,
    logger: logging.Logger,
    profile: KillzoneProfile | None = None,
    dual_tier: bool | None = None,
) -> KillzoneScanResult:
    """Main-channel analysis: Liquidity Sweep + OB/FVG in Killzones."""
    killzone_profile = profile or resolve_killzone_profile()
    symbol_def = resolve_symbol(symbol)
    display_symbol = symbol_def.display
    sweep_limit = max(candle_limit, 400 if killzone_profile.sweep_tf == "5m" else 600)

    try:
        context_htf = provider.to_context(
            display_symbol,
            killzone_profile.htf_filter_tf,
            limit=candle_limit,
        )
        context_sweep = provider.to_context(
            display_symbol,
            killzone_profile.sweep_tf,
            limit=sweep_limit,
            include_h4_trend=True,
        )
    except requests.RequestException as exc:
        logger.error("Market data fetch failed for %s: %s", display_symbol, exc)
        return KillzoneScanResult(
            results=None,
            context=None,
            setup=None,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            tier_outcomes=(),
            primary_filter=FilterResult(
                approved=False,
                direction=Direction.NEUTRAL,
                confidence=0.0,
                message=f"NO TRADE: market data unavailable ({exc})",
            ),
        )
    timestamp = context_sweep.get("timestamp")
    if isinstance(timestamp, datetime):
        timestamp = _utc(timestamp)

    htf_candles = context_htf.get("candles", [])
    sweep_candles = context_sweep.get("candles", [])
    h1_candles = context_sweep.get("h1_candles") or context_htf.get("h1_candles")
    h4_candles = context_sweep.get("h4_candles")

    context = {
        **context_sweep,
        "candles_htf": htf_candles,
        "metadata": {
            **context_sweep.get("metadata", {}),
            "timeframe": killzone_profile.sweep_tf,
            "killzone_profile": killzone_profile.name,
        },
    }

    logger.info(
        "Killzone scan %s [%s]: HTF=%s/%s sweep=%s/%s H1=%s H4=%s",
        display_symbol,
        killzone_profile.name,
        killzone_profile.htf_filter_tf,
        len(htf_candles),
        killzone_profile.sweep_tf,
        len(sweep_candles),
        len(h1_candles or []),
        len(h4_candles or []),
    )

    if not h1_candles or len(sweep_candles) < 30:
        results = run_killzone_agents(
            bias=BiasAnalysis(Direction.NEUTRAL, "equilibrium", 0.0, "no H1 data"),
            sweep=None,
            structure=None,
            timestamp=timestamp,
            setup=None,
        )
        filter_result = FilterResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            message="NO TRADE: insufficient HTF/LTF data",
        )
        return KillzoneScanResult(
            results=results,
            context=context,
            setup=None,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            tier_outcomes=(),
            primary_filter=filter_result,
        )

    bias = analyze_bias(h1_candles, h4_candles)
    sweep_timestamps = build_timestamps(sweep_candles)
    htf_timestamps = build_timestamps(htf_candles)
    df_sweep = _candles_to_dataframe({"candles": sweep_candles})
    atr = calculate_atr(df_sweep, period=14)

    sweep = detect_liquidity_sweep(
        sweep_candles,
        sweep_timestamps,
        htf_candles=htf_candles,
        htf_timestamps=htf_timestamps,
        bias=bias,
        atr=atr,
        profile=killzone_profile,
    )
    structure = find_structure_setup(sweep_candles, sweep) if sweep else None
    setup = (
        build_killzone_setup(
            sweep=sweep,
            structure=structure,
            sweep_candles=sweep_candles,
            sweep_timestamps=sweep_timestamps,
            atr=atr,
            symbol=display_symbol,
            profile=killzone_profile,
        )
        if sweep and structure
        else None
    )
    if setup is not None:
        setup = KillzoneSetup(
            direction=setup.direction,
            entry=setup.entry,
            stop_loss=setup.stop_loss,
            tp1=setup.tp1,
            tp2=setup.tp2,
            tp3=setup.tp3,
            confidence=setup.confidence,
            reason=setup.reason,
            sweep=setup.sweep,
            structure=setup.structure,
            bias=bias,
        )

    results = run_killzone_agents(
        bias=bias,
        sweep=sweep,
        structure=structure,
        timestamp=timestamp,
        setup=setup,
    )
    use_dual_tier = dual_tier if dual_tier is not None else resolve_trading_boss_dual_tier()
    in_kz = (
        is_in_trading_boss_killzone_union(timestamp)
        if use_dual_tier
        else is_killzone_session(timestamp)[0]
    )
    direction, confidence = compute_killzone_decision(results, bias=bias, in_killzone=in_kz)

    tier_outcomes: list[KillzoneTierOutcome] = []
    for tier in resolve_killzone_tiers(dual_tier=use_dual_tier):
        tier_filter = evaluate_killzone_tier_filter(
            tier=tier,
            signal_filter=signal_filter,
            results=results,
            setup=setup,
            direction=direction,
            confidence=confidence,
            symbol=display_symbol,
            timestamp=timestamp,
            bias=bias,
        )
        tier_signal = None
        if tier_filter.approved and setup is not None:
            tier_signal = build_killzone_signal(
                setup,
                display_symbol,
                confidence=tier_filter.confidence,
            )
            logger.info(
                "%s | SIGNAL [%s] | %s | entry=%.2f sl=%.2f tp1=%.2f tp2=%.2f tp3=%.2f | %s",
                display_symbol,
                tier.label.upper(),
                tier_signal.direction.value,
                tier_signal.entry,
                tier_signal.stop_loss,
                tier_signal.tp1,
                tier_signal.tp2,
                tier_signal.tp3,
                tier_signal.reason,
            )
        tier_outcomes.append(
            KillzoneTierOutcome(
                tier=tier,
                filter_result=tier_filter,
                signal=tier_signal,
            )
        )

    primary_filter = tier_outcomes[0].filter_result if tier_outcomes else FilterResult(
        approved=False,
        direction=direction,
        confidence=confidence,
        message="NO TRADE: no tier configured",
    )

    for name, result in results.items():
        logger.info(
            "%s | %s | %s | confidence=%.2f | %s",
            display_symbol,
            name,
            result.direction.value,
            result.confidence,
            result.reason,
        )
    logger.info(
        "%s | FINAL | %s | confidence=%.2f",
        display_symbol,
        direction.value,
        confidence,
    )

    return KillzoneScanResult(
        results=results,
        context=context,
        setup=setup,
        direction=direction,
        confidence=confidence,
        tier_outcomes=tuple(tier_outcomes),
        primary_filter=primary_filter,
    )
