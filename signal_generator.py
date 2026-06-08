from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from agents.base import Direction
from agents.smc_agent import _candles_to_dataframe, _find_swing_points
from config.sl_config import (
    SYMBOL_SL_RULES,
    calculate_lot_size_for_symbol,
    get_sl_config,
)
from config.symbols import resolve_symbol
from signal_geometry import (
    coerce_direction,
    infer_direction_from_levels,
    validate_trade_levels,
)

DEFAULT_SWING_LOOKBACK = 5
DEFAULT_ATR_PERIOD = 14
DEFAULT_ATR_BUFFER_MULTIPLIER = 0.3
DEFAULT_DEPOSIT = 200.0
DEFAULT_LOT_SIZE = calculate_lot_size_for_symbol(DEFAULT_DEPOSIT, "XAUUSD")


@dataclass(frozen=True)
class TradeSignal:
    """Risk-based trade signal with entry, stop loss, and take profit levels."""

    direction: Direction
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confidence: float
    reason: str
    lot_size: float = DEFAULT_LOT_SIZE

    def __post_init__(self) -> None:
        resolved = infer_direction_from_levels(
            self.entry,
            self.stop_loss,
            self.tp1,
            self.tp3,
        )
        validate_trade_levels(
            self.entry,
            self.stop_loss,
            self.tp1,
            self.tp2,
            self.tp3,
            resolved,
        )
        if self.direction != resolved:
            raise ValueError(
                f"TradeSignal.direction ({self.direction.value}) "
                f"does not match price geometry ({resolved.value})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "direction": self.direction.value,
            "entry": self.entry,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "tp3": self.tp3,
            "confidence": self.confidence,
            "reason": self.reason,
            "lot_size": self.lot_size,
        }

    def format(self) -> str:
        lines = [
            "=== TRADE SIGNAL ===",
            f"Direction: {self.direction.value.upper()}",
            f"Entry: {self.entry:.2f}",
            f"Stop Loss: {self.stop_loss:.2f}",
            f"TP1: {self.tp1:.2f}",
            f"TP2: {self.tp2:.2f}",
            f"TP3: {self.tp3:.2f}",
            f"Lot Size: {self.lot_size:.2f}",
            f"Confidence: {self.confidence:.2f}",
            f"Reason: {self.reason}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class SLValidationResult:
    signal: TradeSignal | None = None
    rejection_reason: str | None = None

    @property
    def approved(self) -> bool:
        return self.signal is not None


def calculate_lot_size(deposit: float, symbol: str = "XAUUSD") -> float:
    """Return fixed lot size for the given deposit and symbol."""
    return calculate_lot_size_for_symbol(deposit, symbol)


def resolve_signal_direction(signal: TradeSignal) -> Direction:
    """Return the direction implied by the signal's price levels."""
    return infer_direction_from_levels(
        signal.entry,
        signal.stop_loss,
        signal.tp1,
        signal.tp3,
    )


def align_trade_signal_direction(signal: TradeSignal) -> TradeSignal:
    """Ensure TradeSignal.direction matches entry/SL/TP geometry."""
    resolved = resolve_signal_direction(signal)
    validate_trade_levels(
        signal.entry,
        signal.stop_loss,
        signal.tp1,
        signal.tp2,
        signal.tp3,
        resolved,
    )

    if signal.direction == resolved:
        return signal

    return TradeSignal(
        direction=resolved,
        entry=signal.entry,
        stop_loss=signal.stop_loss,
        tp1=signal.tp1,
        tp2=signal.tp2,
        tp3=signal.tp3,
        confidence=signal.confidence,
        reason=signal.reason,
        lot_size=signal.lot_size,
    )


def calculate_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD) -> float:
    """Calculate the latest ATR value using Wilder's smoothing."""
    if len(df) < period + 1:
        raise ValueError(f"Need at least {period + 1} candles to calculate ATR({period})")

    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    value = atr.iloc[-1]
    if pd.isna(value):
        raise ValueError("ATR calculation returned no value")
    return float(value)


def price_distance_pips(distance: float, pip_size: float) -> float:
    return abs(distance) / pip_size


class SignalGenerator:
    """Builds R-multiple trade signals from market context and direction."""

    def __init__(
        self,
        swing_lookback: int = DEFAULT_SWING_LOOKBACK,
        atr_period: int = DEFAULT_ATR_PERIOD,
        atr_buffer_multiplier: float = DEFAULT_ATR_BUFFER_MULTIPLIER,
        deposit: float = DEFAULT_DEPOSIT,
    ) -> None:
        self.swing_lookback = swing_lookback
        self.atr_period = atr_period
        self.atr_buffer_multiplier = atr_buffer_multiplier
        self.deposit = deposit

    def generate(
        self,
        context: dict[str, Any],
        direction: Direction,
        confidence: float,
        reason: str = "",
    ) -> SLValidationResult:
        direction = coerce_direction(direction)
        if direction == Direction.NEUTRAL:
            return SLValidationResult(
                rejection_reason="Cannot generate a trade signal for NEUTRAL direction",
            )

        symbol = str(context.get("symbol", "UNKNOWN")).upper()
        try:
            display_symbol = resolve_symbol(symbol).display
        except ValueError:
            display_symbol = symbol

        try:
            df = _candles_to_dataframe(context)
            entry = float(df.iloc[-1]["close"])
            atr = calculate_atr(df, period=self.atr_period)
            atr_buffer = atr * self.atr_buffer_multiplier
            swing_highs, swing_lows = _find_swing_points(df, lookback=self.swing_lookback)
        except ValueError as exc:
            return SLValidationResult(rejection_reason=str(exc))

        if direction == Direction.LONG:
            if not swing_lows:
                return SLValidationResult(
                    rejection_reason="No swing low found for LONG stop loss",
                )
            swing_price = swing_lows[-1].price
            stop_loss = swing_price - atr_buffer
            if stop_loss >= entry:
                return SLValidationResult(
                    rejection_reason="Buffered stop loss must be below entry for LONG signal",
                )
        else:
            if not swing_highs:
                return SLValidationResult(
                    rejection_reason="No swing high found for SHORT stop loss",
                )
            swing_price = swing_highs[-1].price
            stop_loss = swing_price + atr_buffer
            if stop_loss <= entry:
                return SLValidationResult(
                    rejection_reason="Buffered stop loss must be above entry for SHORT signal",
                )

        return self.validate_sl(
            symbol=display_symbol,
            direction=direction,
            entry=entry,
            swing_price=swing_price,
            stop_loss=stop_loss,
            confidence=confidence,
            reason=reason,
            atr_buffer=atr_buffer,
        )

    def validate_sl(
        self,
        *,
        symbol: str,
        direction: Direction,
        entry: float,
        swing_price: float,
        stop_loss: float,
        confidence: float,
        reason: str = "",
        atr_buffer: float | None = None,
    ) -> SLValidationResult:
        """Validate SL distance and build the final trade signal."""
        direction = coerce_direction(direction)
        sl_config = get_sl_config(symbol.upper())

        if direction == Direction.LONG and stop_loss >= entry:
            return SLValidationResult(
                rejection_reason="Stop loss must be below entry for LONG signal",
            )
        if direction == Direction.SHORT and stop_loss <= entry:
            return SLValidationResult(
                rejection_reason="Stop loss must be above entry for SHORT signal",
            )

        risk = abs(entry - stop_loss)
        lot_size = calculate_lot_size_for_symbol(self.deposit, symbol)

        if sl_config is not None:
            sl_pips = price_distance_pips(risk, sl_config.pip_size)
            if sl_pips < sl_config.min_sl_pips:
                return SLValidationResult(
                    rejection_reason=(
                        f"SL rejected: {sl_pips:.1f} pips "
                        f"below minimum {sl_config.min_sl_pips:.0f} pips for {symbol}"
                    ),
                )

            if sl_pips > sl_config.max_sl_pips:
                return SLValidationResult(
                    rejection_reason=(
                        f"SL rejected: {sl_pips:.1f} pips "
                        f"exceeds maximum {sl_config.max_sl_pips:.0f} pips for {symbol}"
                    ),
                )

        buffer_note = ""
        if atr_buffer is not None:
            buffer_note = f", ATR buffer {atr_buffer:.2f}"

        if direction == Direction.LONG:
            signal_reason = reason or (
                f"LONG signal: SL at swing low {swing_price:.2f}{buffer_note}, "
                f"risk {risk:.2f} ({lot_size:.2f} lot), targets at 1R/2R/3R"
            )
            signal = TradeSignal(
                direction=Direction.LONG,
                entry=entry,
                stop_loss=stop_loss,
                tp1=entry + risk,
                tp2=entry + risk * 2,
                tp3=entry + risk * 3,
                confidence=confidence,
                reason=signal_reason,
                lot_size=lot_size,
            )
        else:
            signal_reason = reason or (
                f"SHORT signal: SL at swing high {swing_price:.2f}{buffer_note}, "
                f"risk {risk:.2f} ({lot_size:.2f} lot), targets at 1R/2R/3R"
            )
            signal = TradeSignal(
                direction=Direction.SHORT,
                entry=entry,
                stop_loss=stop_loss,
                tp1=entry - risk,
                tp2=entry - risk * 2,
                tp3=entry - risk * 3,
                confidence=confidence,
                reason=signal_reason,
                lot_size=lot_size,
            )

        return SLValidationResult(signal=align_trade_signal_direction(signal))

    def print_signal(
        self,
        context: dict[str, Any],
        direction: Direction,
        confidence: float,
        reason: str = "",
    ) -> TradeSignal | None:
        if direction == Direction.NEUTRAL:
            print("=== TRADE SIGNAL ===")
            print("No trade signal generated (NEUTRAL decision)")
            return None

        result = self.generate(context, direction, confidence, reason)
        if result.signal is None:
            print("=== TRADE SIGNAL ===")
            print(result.rejection_reason or "Signal rejected")
            return None

        print(result.signal.format())
        return result.signal
