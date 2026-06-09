from __future__ import annotations

from agents.base import Direction


def stop_loss_hit(*, direction: Direction, high: float, low: float, stop_loss: float) -> bool:
    if direction == Direction.LONG:
        return low <= stop_loss
    return high >= stop_loss


def take_profit_hit(*, direction: Direction, high: float, low: float, tp_price: float) -> bool:
    if direction == Direction.LONG:
        return high >= tp_price
    return low <= tp_price
