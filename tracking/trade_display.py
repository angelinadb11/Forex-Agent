from __future__ import annotations

from agents.base import Direction
from signal_generator import TradeSignal, resolve_signal_direction
from signal_geometry import format_trade_side


def _format_reason(reason: str) -> str:
    cleaned = reason.replace(" | ", " + ")
    for token in ("BTCUSDT ", "XAUUSDT ", "XAUUSD ", "DJ30 ", "15m ", "SMC: ", "RSI(14)=", "Session: "):
        cleaned = cleaned.replace(token, "")
    return cleaned.strip(" +")


from tracking.console import safe_print


def format_trade_signal(symbol: str, signal: TradeSignal) -> str:
    """Format a generated signal for console display."""
    direction = resolve_signal_direction(signal)
    side = format_trade_side(direction)
    header_emoji = "🟢" if direction == Direction.LONG else "🔴"
    header = f"{header_emoji} {symbol} {side}"
    confidence_pct = min(100, int(round(signal.confidence * 100)))
    reason = _format_reason(signal.reason)

    lines = [
        header,
        "",
        f"Entry: {signal.entry:.2f}",
        f"SL: {signal.stop_loss:.2f}",
        "",
        f"TP1: {signal.tp1:.2f}",
        f"TP2: {signal.tp2:.2f}",
        f"TP3: {signal.tp3:.2f}",
        "",
        f"Confidence: {confidence_pct}%",
        "",
        "Reason:",
        reason,
        "",
        "⸻",
    ]
    return "\n".join(lines)


def print_trade_signal(symbol: str, signal: TradeSignal) -> None:
    safe_print(format_trade_signal(symbol, signal))
