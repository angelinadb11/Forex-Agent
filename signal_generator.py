from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agents.base import Direction
from agents.smc_agent import _candles_to_dataframe, _find_swing_points
from signal_geometry import (
    coerce_direction,
    infer_direction_from_levels,
    validate_trade_levels,
)


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
            f"Confidence: {self.confidence:.2f}",
            f"Reason: {self.reason}",
        ]
        return "\n".join(lines)


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
    )


class SignalGenerator:
    """Builds R-multiple trade signals from market context and direction."""

    def __init__(self, swing_lookback: int = 2) -> None:
        self.swing_lookback = swing_lookback

    def generate(
        self,
        context: dict[str, Any],
        direction: Direction,
        confidence: float,
        reason: str = "",
    ) -> TradeSignal:
        direction = coerce_direction(direction)
        if direction == Direction.NEUTRAL:
            raise ValueError("Cannot generate a trade signal for NEUTRAL direction")

        df = _candles_to_dataframe(context)
        entry = float(df.iloc[-1]["close"])
        swing_highs, swing_lows = _find_swing_points(df, lookback=self.swing_lookback)

        if direction == Direction.LONG:
            signal = self._build_long_signal(
                entry=entry,
                swing_lows=swing_lows,
                confidence=confidence,
                reason=reason,
            )
        else:
            signal = self._build_short_signal(
                entry=entry,
                swing_highs=swing_highs,
                confidence=confidence,
                reason=reason,
            )

        return align_trade_signal_direction(signal)

    def _build_long_signal(
        self,
        entry: float,
        swing_lows: list,
        confidence: float,
        reason: str,
    ) -> TradeSignal:
        if not swing_lows:
            raise ValueError("No swing low found for LONG stop loss")

        stop_loss = swing_lows[-1].price
        if stop_loss >= entry:
            raise ValueError("Last swing low must be below entry for LONG signal")

        risk = entry - stop_loss
        signal_reason = reason or (
            f"LONG signal: stop loss at last swing low ({stop_loss:.2f}), "
            f"risk {risk:.2f}, targets at 1R/2R/3R"
        )

        return TradeSignal(
            direction=Direction.LONG,
            entry=entry,
            stop_loss=stop_loss,
            tp1=entry + risk,
            tp2=entry + risk * 2,
            tp3=entry + risk * 3,
            confidence=confidence,
            reason=signal_reason,
        )

    def _build_short_signal(
        self,
        entry: float,
        swing_highs: list,
        confidence: float,
        reason: str,
    ) -> TradeSignal:
        if not swing_highs:
            raise ValueError("No swing high found for SHORT stop loss")

        stop_loss = swing_highs[-1].price
        if stop_loss <= entry:
            raise ValueError("Last swing high must be above entry for SHORT signal")

        risk = stop_loss - entry
        signal_reason = reason or (
            f"SHORT signal: stop loss at last swing high ({stop_loss:.2f}), "
            f"risk {risk:.2f}, targets at 1R/2R/3R"
        )

        return TradeSignal(
            direction=Direction.SHORT,
            entry=entry,
            stop_loss=stop_loss,
            tp1=entry - risk,
            tp2=entry - risk * 2,
            tp3=entry - risk * 3,
            confidence=confidence,
            reason=signal_reason,
        )

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

        signal = self.generate(context, direction, confidence, reason)
        print(signal.format())
        return signal
