"""Liquidity-sweep scalp strategy (M5, small stop).

Setup: price sweeps a pool of equal lows/highs (stop hunt) and closes back
beyond the level on the same candle. Entry at the close of the sweep candle,
stop just past the sweep wick, TP1 = 1R, TP2 = nearest opposite liquidity
pool (or 2R fallback).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agents.base import Direction
from agents.liquidity_agent import (
    EqualLevel,
    _find_equal_levels,
    _find_swing_points,
)
from agents.session_agent import is_london_or_new_york_session
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, align_trade_signal_direction, calculate_atr
from strategy.scalp_mode import ScalpAnalysisResult, ScalpPublishGate
from tracking.trade_pnl import pip_size_for_symbol

# Production config selected by backtest (17 days M1 XAUUSD):
# volume >=1.3x + Stoch RSI extreme -> +7.18R, 57 trades, WR 52.6%.
LIQUIDITY_SCALP_TIMEFRAME = "1m"
LIQUIDITY_SCALP_SYMBOLS = frozenset({"XAUUSD"})
LIQUIDITY_SCALP_CANDLE_LIMIT = 300
LIQUIDITY_SCALP_DETECTION_WINDOW = 240

MIN_SL_PIPS = 5.0
MAX_SL_PIPS = 25.0
SL_BUFFER_ATR_MULT = 0.25
MIN_POOL_TOUCHES = 2
SWING_LOOKBACK = 2
EQUAL_TOLERANCE_PCT = 0.0008
TP2_MIN_R = 1.5
TP2_MAX_R = 3.5
TP2_FALLBACK_R = 2.0
BASE_CONFIDENCE = 0.60
CONFIDENCE_PER_EXTRA_TOUCH = 0.05
MAX_CONFIDENCE = 0.75
MIN_CANDLES = 60
H1_TREND_EMA_PERIOD = 50

LIQUIDITY_SCALP_MAX_SIGNALS_PER_DAY = 6
# 5 minutes between signals: after a fast stop/BE the next setup is allowed
# within ~2-3 minutes of the close (the gate counts from signal time, and an
# open trade blocks the slot anyway).
LIQUIDITY_SCALP_MIN_INTERVAL_SECONDS = 300


STOCH_RSI_PERIOD = 14
STOCH_RSI_SMOOTH_K = 3
STOCH_RSI_OVERSOLD = 20.0
STOCH_RSI_OVERBOUGHT = 80.0
VOLUME_SMA_PERIOD = 20


@dataclass(frozen=True)
class LiquidityScalpConfig:
    """Tunable quality filters for the sweep detection."""

    min_pool_touches: int = MIN_POOL_TOUCHES
    require_directional_close: bool = False
    min_wick_atr_mult: float = 0.0
    require_h1_trend: bool = False
    require_heiken_ashi: bool = False
    require_volume_spike: bool = False
    min_volume_ratio: float = 1.5
    require_stoch_rsi: bool = False
    min_sl_pips: float = MIN_SL_PIPS
    max_sl_pips: float = MAX_SL_PIPS


DEFAULT_LIQUIDITY_SCALP_CONFIG = LiquidityScalpConfig(
    require_volume_spike=True,
    min_volume_ratio=1.3,
    require_stoch_rsi=True,
)


def is_liquidity_scalp_enabled(symbol: str) -> bool:
    return resolve_symbol(symbol).display in LIQUIDITY_SCALP_SYMBOLS


def build_liquidity_scalp_gate() -> ScalpPublishGate:
    return ScalpPublishGate(
        min_interval_seconds=LIQUIDITY_SCALP_MIN_INTERVAL_SECONDS,
        max_signals_per_day=LIQUIDITY_SCALP_MAX_SIGNALS_PER_DAY,
    )


@dataclass(frozen=True)
class LiquiditySweepSetup:
    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    pool_level: float
    pool_touches: int
    sweep_extreme: float
    sl_pips: float
    reason: str

    @property
    def risk(self) -> float:
        return abs(self.entry - self.stop_loss)


def _candles_to_dataframe(candles: list[dict[str, Any]]):
    import pandas as pd

    rows = [
        {
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        }
        for c in candles
        if {"open", "high", "low", "close"}.issubset(c)
    ]
    return pd.DataFrame(rows)


def _pool_clusters(
    candles: list[dict[str, Any]],
) -> tuple[list[EqualLevel], list[EqualLevel]]:
    """Equal highs/lows clusters from history EXCLUDING the trigger candle."""
    df = _candles_to_dataframe(candles[:-1])
    swing_highs, swing_lows = _find_swing_points(df, lookback=SWING_LOOKBACK)
    equal_highs = _find_equal_levels(swing_highs, tolerance_pct=EQUAL_TOLERANCE_PCT)
    equal_lows = _find_equal_levels(swing_lows, tolerance_pct=EQUAL_TOLERANCE_PCT)
    return equal_highs, equal_lows


def _select_swept_pool(
    clusters: list[EqualLevel],
    *,
    candle: dict[str, Any],
    prev_candle: dict[str, Any],
    direction: Direction,
    config: LiquidityScalpConfig,
    atr: float,
) -> EqualLevel | None:
    """Return the pool freshly swept by ``candle`` (wick beyond, close back)."""
    min_wick = config.min_wick_atr_mult * atr
    candidates: list[EqualLevel] = []
    for cluster in clusters:
        if cluster.count < config.min_pool_touches:
            continue
        level = cluster.level
        if direction == Direction.LONG:
            swept = candle["low"] < level and candle["close"] > level
            fresh = prev_candle["low"] >= level
            deep_enough = (level - candle["low"]) >= min_wick
        else:
            swept = candle["high"] > level and candle["close"] < level
            fresh = prev_candle["high"] <= level
            deep_enough = (candle["high"] - level) >= min_wick
        if swept and fresh and deep_enough:
            candidates.append(cluster)

    if not candidates:
        return None
    # The deepest swept pool defines the wick extreme protection.
    if direction == Direction.LONG:
        return min(candidates, key=lambda c: c.level)
    return max(candidates, key=lambda c: c.level)


def heiken_ashi_direction(candles: list[dict[str, Any]], lookback: int = 50) -> Direction:
    """Color of the last Heiken Ashi candle (LONG=green, SHORT=red)."""
    window = candles[-lookback:]
    if len(window) < 2:
        return Direction.NEUTRAL

    ha_open = (float(window[0]["open"]) + float(window[0]["close"])) / 2
    ha_close = (
        float(window[0]["open"])
        + float(window[0]["high"])
        + float(window[0]["low"])
        + float(window[0]["close"])
    ) / 4
    for candle in window[1:]:
        ha_open = (ha_open + ha_close) / 2
        ha_close = (
            float(candle["open"])
            + float(candle["high"])
            + float(candle["low"])
            + float(candle["close"])
        ) / 4

    if ha_close > ha_open:
        return Direction.LONG
    if ha_close < ha_open:
        return Direction.SHORT
    return Direction.NEUTRAL


def volume_spike_ratio(candles: list[dict[str, Any]]) -> float | None:
    """Volume of the last candle relative to the SMA of prior volumes."""
    volumes = [float(c["volume"]) for c in candles if "volume" in c]
    if len(volumes) < VOLUME_SMA_PERIOD + 1:
        return None
    baseline = volumes[-(VOLUME_SMA_PERIOD + 1) : -1]
    average = sum(baseline) / len(baseline)
    if average <= 0:
        return None
    return volumes[-1] / average


def stoch_rsi_k(candles: list[dict[str, Any]]) -> float | None:
    """Stochastic RSI %K (0-100) for the last candle."""
    import pandas as pd

    closes = pd.Series([float(c["close"]) for c in candles])
    if len(closes) < STOCH_RSI_PERIOD * 3:
        return None

    delta = closes.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / STOCH_RSI_PERIOD, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / STOCH_RSI_PERIOD, adjust=False).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi = 100 - 100 / (1 + rs)

    rsi_min = rsi.rolling(STOCH_RSI_PERIOD).min()
    rsi_max = rsi.rolling(STOCH_RSI_PERIOD).max()
    span = (rsi_max - rsi_min).replace(0, float("nan"))
    stoch = (rsi - rsi_min) / span * 100
    k = stoch.rolling(STOCH_RSI_SMOOTH_K).mean().iloc[-1]
    if k != k:  # NaN
        return None
    return float(k)


def h1_trend_direction(h1_candles: list[dict[str, Any]] | None) -> Direction:
    """Direction of H1 trend by price vs EMA50; NEUTRAL when unavailable."""
    if not h1_candles or len(h1_candles) < H1_TREND_EMA_PERIOD:
        return Direction.NEUTRAL

    import pandas as pd

    closes = pd.Series([float(c["close"]) for c in h1_candles])
    ema = closes.ewm(span=H1_TREND_EMA_PERIOD, adjust=False).mean().iloc[-1]
    price = closes.iloc[-1]
    if price > ema:
        return Direction.LONG
    if price < ema:
        return Direction.SHORT
    return Direction.NEUTRAL


def _opposite_liquidity_target(
    *,
    direction: Direction,
    entry: float,
    risk: float,
    equal_highs: list[EqualLevel],
    equal_lows: list[EqualLevel],
) -> float:
    """TP2 at the nearest opposite pool within [1.5R, 3.5R], else 2R."""
    if direction == Direction.LONG:
        levels = sorted(c.level for c in equal_highs if c.level > entry)
    else:
        levels = sorted(
            (c.level for c in equal_lows if c.level < entry), reverse=True
        )

    for level in levels:
        distance_r = abs(level - entry) / risk
        if TP2_MIN_R <= distance_r <= TP2_MAX_R:
            return level
        if distance_r > TP2_MAX_R:
            break

    if direction == Direction.LONG:
        return entry + TP2_FALLBACK_R * risk
    return entry - TP2_FALLBACK_R * risk


def detect_liquidity_sweep_setup(
    candles: list[dict[str, Any]],
    symbol: str,
    *,
    config: LiquidityScalpConfig = DEFAULT_LIQUIDITY_SCALP_CONFIG,
    h1_candles: list[dict[str, Any]] | None = None,
) -> tuple[LiquiditySweepSetup | None, str]:
    """Detect a liquidity sweep on the last closed candle.

    Returns (setup, reason). ``setup`` is None when no valid sweep exists and
    ``reason`` explains why.
    """
    if len(candles) < MIN_CANDLES:
        return None, f"NO SCALP: need at least {MIN_CANDLES} M5 candles"

    display = resolve_symbol(symbol).display
    pip_size = pip_size_for_symbol(display) or 1.0

    candle = candles[-1]
    prev_candle = candles[-2]
    equal_highs, equal_lows = _pool_clusters(candles)

    df = _candles_to_dataframe(candles)
    try:
        atr = calculate_atr(df)
    except Exception:
        return None, "NO SCALP: ATR unavailable"
    buffer = atr * SL_BUFFER_ATR_MULT

    long_pool = _select_swept_pool(
        equal_lows,
        candle=candle,
        prev_candle=prev_candle,
        direction=Direction.LONG,
        config=config,
        atr=atr,
    )
    short_pool = _select_swept_pool(
        equal_highs,
        candle=candle,
        prev_candle=prev_candle,
        direction=Direction.SHORT,
        config=config,
        atr=atr,
    )

    if long_pool is not None and short_pool is not None:
        return None, "NO SCALP: both-side sweep on one candle (indecision)"
    if long_pool is None and short_pool is None:
        return None, "NO SCALP: no fresh liquidity sweep on last candle"

    sweep_direction = Direction.LONG if long_pool is not None else Direction.SHORT

    if config.require_directional_close:
        if sweep_direction == Direction.LONG and candle["close"] < candle["open"]:
            return None, "NO SCALP: sweep candle closed bearish (weak reclaim)"
        if sweep_direction == Direction.SHORT and candle["close"] > candle["open"]:
            return None, "NO SCALP: sweep candle closed bullish (weak reclaim)"

    if config.require_h1_trend:
        trend = h1_trend_direction(h1_candles)
        if trend != sweep_direction:
            return None, (
                "NO SCALP: H1 trend does not confirm sweep "
                f"(sweep={sweep_direction.value}, trend={trend.value})"
            )

    if config.require_heiken_ashi:
        ha_direction = heiken_ashi_direction(candles)
        if ha_direction != sweep_direction:
            return None, (
                "NO SCALP: Heiken Ashi candle does not confirm sweep "
                f"(sweep={sweep_direction.value}, HA={ha_direction.value})"
            )

    if config.require_volume_spike:
        ratio = volume_spike_ratio(candles)
        if ratio is None:
            return None, "NO SCALP: volume data unavailable"
        if ratio < config.min_volume_ratio:
            return None, (
                f"NO SCALP: sweep volume {ratio:.2f}x below "
                f"required {config.min_volume_ratio:.2f}x average"
            )

    if config.require_stoch_rsi:
        k = stoch_rsi_k(candles)
        if k is None:
            return None, "NO SCALP: Stoch RSI unavailable"
        if sweep_direction == Direction.LONG and k > STOCH_RSI_OVERSOLD:
            return None, f"NO SCALP: Stoch RSI {k:.0f} not oversold (<={STOCH_RSI_OVERSOLD:.0f})"
        if sweep_direction == Direction.SHORT and k < STOCH_RSI_OVERBOUGHT:
            return None, (
                f"NO SCALP: Stoch RSI {k:.0f} not overbought (>={STOCH_RSI_OVERBOUGHT:.0f})"
            )

    if long_pool is not None:
        direction = Direction.LONG
        pool = long_pool
        sweep_extreme = float(candle["low"])
        entry = float(candle["close"])
        stop_loss = sweep_extreme - buffer
        min_risk = config.min_sl_pips * pip_size
        if entry - stop_loss < min_risk:
            stop_loss = entry - min_risk
    else:
        direction = Direction.SHORT
        pool = short_pool  # type: ignore[assignment]
        sweep_extreme = float(candle["high"])
        entry = float(candle["close"])
        stop_loss = sweep_extreme + buffer
        min_risk = config.min_sl_pips * pip_size
        if stop_loss - entry < min_risk:
            stop_loss = entry + min_risk

    risk = abs(entry - stop_loss)
    sl_pips = risk / pip_size
    if sl_pips > config.max_sl_pips:
        return None, (
            f"NO SCALP: sweep stop {sl_pips:.1f} pips exceeds max {config.max_sl_pips:.0f}"
        )

    if direction == Direction.LONG:
        tp1 = entry + risk
    else:
        tp1 = entry - risk
    tp2 = _opposite_liquidity_target(
        direction=direction,
        entry=entry,
        risk=risk,
        equal_highs=equal_highs,
        equal_lows=equal_lows,
    )

    side = "SSL" if direction == Direction.LONG else "BSL"
    reason = (
        f"Liquidity scalp {direction.value.upper()}: {side} sweep at "
        f"{pool.level:.2f} (x{pool.count}), wick {sweep_extreme:.2f}, "
        f"SL {sl_pips:.1f} pips, TP 1R/{abs(tp2 - entry) / risk:.1f}R"
    )

    setup = LiquiditySweepSetup(
        direction=direction,
        entry=entry,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        pool_level=pool.level,
        pool_touches=pool.count,
        sweep_extreme=sweep_extreme,
        sl_pips=sl_pips,
        reason=reason,
    )
    return setup, reason


def setup_confidence(setup: LiquiditySweepSetup) -> float:
    extra = max(0, setup.pool_touches - MIN_POOL_TOUCHES)
    return round(
        min(MAX_CONFIDENCE, BASE_CONFIDENCE + extra * CONFIDENCE_PER_EXTRA_TOUCH), 2
    )


def build_liquidity_scalp_signal(
    setup: LiquiditySweepSetup,
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
        confidence=setup_confidence(setup),
        reason=setup.reason,
        lot_size=calculate_lot_size_for_symbol(deposit, display),
    )
    return align_trade_signal_direction(signal)


def analyze_liquidity_scalp_symbol(
    symbol: str,
    *,
    provider: MarketDataProvider,
    publish_gate: ScalpPublishGate | None = None,
    candle_limit: int = LIQUIDITY_SCALP_CANDLE_LIMIT,
    config: LiquidityScalpConfig = DEFAULT_LIQUIDITY_SCALP_CONFIG,
) -> tuple[TradeSignal | None, dict[str, Any] | None, ScalpAnalysisResult]:
    """Live liquidity-scalp analysis matching the analyze_scalp interface."""
    display = resolve_symbol(symbol).display

    def rejected(message: str, context: dict[str, Any] | None = None):
        return None, context, ScalpAnalysisResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            message=message,
        )

    if not is_liquidity_scalp_enabled(display):
        return rejected(f"NO SCALP: liquidity scalp disabled for {display}")

    gate = publish_gate or build_liquidity_scalp_gate()
    context = provider.to_context(
        display,
        LIQUIDITY_SCALP_TIMEFRAME,
        limit=candle_limit,
        include_h4_trend=False,
    )
    timestamp = context.get("timestamp")
    if isinstance(timestamp, datetime) and not is_london_or_new_york_session(timestamp):
        return rejected("NO SCALP: outside London/NY session", context)

    candles = context.get("candles", [])[-LIQUIDITY_SCALP_DETECTION_WINDOW:]
    setup, reason = detect_liquidity_sweep_setup(
        candles,
        display,
        config=config,
        h1_candles=context.get("h1_candles"),
    )
    if setup is None:
        return rejected(reason, context)

    if isinstance(timestamp, datetime):
        allowed, block_reason = gate.can_publish(display, timestamp)
        if not allowed:
            return rejected(block_reason or "NO SCALP: publish gate blocked", context)

    signal = build_liquidity_scalp_signal(setup, display)
    if isinstance(timestamp, datetime):
        gate.record(display, timestamp)

    return signal, context, ScalpAnalysisResult(
        approved=True,
        direction=setup.direction,
        confidence=signal.confidence,
        message="Liquidity scalp signal approved",
    )
