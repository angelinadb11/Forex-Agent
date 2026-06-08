from __future__ import annotations

from agents.base import Direction


def coerce_direction(value: Direction | str) -> Direction:
    """Normalize direction values from enums, strings, or labels."""
    if isinstance(value, Direction):
        return value

    normalized = str(value).strip().lower()
    if normalized in {"long", "buy"}:
        return Direction.LONG
    if normalized in {"short", "sell"}:
        return Direction.SHORT
    if normalized == "neutral":
        return Direction.NEUTRAL
    raise ValueError(f"Invalid direction: {value!r}")


def infer_direction_from_levels(
    entry: float,
    stop_loss: float,
    tp1: float,
    tp3: float,
) -> Direction:
    """Infer trade direction from entry, stop loss, and take-profit geometry."""
    if stop_loss < entry and tp1 > entry and tp3 > entry:
        return Direction.LONG
    if stop_loss > entry and tp1 < entry and tp3 < entry:
        return Direction.SHORT
    raise ValueError(
        "Trade levels do not match a valid LONG or SHORT setup: "
        f"entry={entry:.2f}, sl={stop_loss:.2f}, tp1={tp1:.2f}, tp3={tp3:.2f}"
    )


def validate_trade_levels(
    entry: float,
    stop_loss: float,
    tp1: float,
    tp2: float,
    tp3: float,
    direction: Direction,
) -> None:
    """Ensure stop loss and take profits match the declared direction."""
    if direction == Direction.LONG:
        if not (stop_loss < entry < tp1 <= tp2 <= tp3):
            raise ValueError(
                "LONG signal requires SL below entry and TPs above entry "
                f"(entry={entry:.2f}, sl={stop_loss:.2f}, tp1={tp1:.2f})"
            )
        return

    if direction == Direction.SHORT:
        if not (stop_loss > entry > tp1 >= tp2 >= tp3):
            raise ValueError(
                "SHORT signal requires SL above entry and TPs below entry "
                f"(entry={entry:.2f}, sl={stop_loss:.2f}, tp1={tp1:.2f})"
            )
        return

    raise ValueError("Cannot validate trade levels for NEUTRAL direction")


def format_trade_side(direction: Direction) -> str:
    """Return BUY/SELL label for a resolved trade direction."""
    return "BUY" if direction == Direction.LONG else "SELL"
