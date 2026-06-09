from __future__ import annotations

from typing import Protocol


class _StopLossTrade(Protocol):
    result: str
    tp1_hit: bool
    pnl_r: float
    exit_price: float
    stop_loss: float
    entry: float
    risk: float


class _StopLossRecord(Protocol):
    result: str | None
    tp1_hit: bool
    entry: float
    stop_loss: float


FULL_STOP_TOLERANCE_RATIO = 1e-4


def _exit_at_initial_stop(
    *,
    exit_price: float,
    initial_stop_loss: float,
    risk: float,
) -> bool:
    tolerance = max(1e-9, abs(risk) * FULL_STOP_TOLERANCE_RATIO)
    return abs(exit_price - initial_stop_loss) <= tolerance


def is_full_stop_loss(
    trade: _StopLossTrade,
    *,
    initial_stop_loss: float | None = None,
) -> bool:
    """Return True only for a full loss at the original stop, not BE or trailed exits."""
    initial_sl = (
        initial_stop_loss
        if initial_stop_loss is not None
        else trade.stop_loss
    )
    risk = trade.risk if trade.risk else abs(trade.entry - initial_sl)
    return is_full_stop_loss_from_values(
        result=trade.result,
        tp1_hit=trade.tp1_hit,
        pnl_r=trade.pnl_r,
        exit_price=trade.exit_price,
        entry=trade.entry,
        initial_stop_loss=initial_sl,
        risk=risk,
    )


def is_full_stop_loss_from_values(
    *,
    result: str,
    tp1_hit: bool,
    pnl_r: float,
    exit_price: float,
    entry: float,
    initial_stop_loss: float,
    risk: float,
) -> bool:
    if result == "breakeven" or tp1_hit:
        return False
    if result != "stop_loss" or pnl_r >= -1e-9:
        return False
    return _exit_at_initial_stop(
        exit_price=exit_price,
        initial_stop_loss=initial_stop_loss,
        risk=risk,
    )


def is_full_stop_loss_record(trade: _StopLossRecord) -> bool:
    """Classify live/history trades without simulated exit metadata."""
    if trade.result == "breakeven" or trade.tp1_hit:
        return False
    if trade.result != "stop_loss":
        return False

    risk = abs(trade.entry - trade.stop_loss)
    if risk <= 0:
        return False

    tolerance = max(1e-9, risk * FULL_STOP_TOLERANCE_RATIO)
    return abs(trade.stop_loss - trade.entry) > tolerance
