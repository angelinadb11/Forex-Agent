from __future__ import annotations

import random

from agents.base import AgentResult, Direction
from signal_generator import TradeSignal, resolve_signal_direction

BULLISH_ANALYSIS_PHRASES = (
    "Liquidity sweep confirmed.",
    "Bullish structure remains intact.",
    "Buyers stepped in after the sweep.",
    "Momentum remains to the upside.",
    "Looking for continuation higher.",
    "Market structure favors longs.",
)

BEARISH_ANALYSIS_PHRASES = (
    "Liquidity sweep confirmed.",
    "Bearish structure remains intact.",
    "Sellers stepped in after the sweep.",
    "Momentum remains to the downside.",
    "Looking for continuation lower.",
    "Market structure favors shorts.",
)


def select_analysis_phrases(direction: Direction, *, count: int | None = None) -> list[str]:
    """Return 1-2 varied analysis sentences for the trade direction."""
    pool = BULLISH_ANALYSIS_PHRASES if direction == Direction.LONG else BEARISH_ANALYSIS_PHRASES
    phrase_count = random.randint(1, 2) if count is None else count
    phrase_count = max(1, min(phrase_count, len(pool)))
    return random.sample(pool, phrase_count)


def summarize_analysis_sentences(
    results: dict[str, AgentResult],
    direction: Direction,
) -> list[str]:
    """Pick 1-2 natural-language analysis sentences for Telegram."""
    del results
    return select_analysis_phrases(direction)


def format_trade_signal(
    symbol: str,
    signal: TradeSignal,
    timeframe: str,
    results: dict[str, AgentResult] | None = None,
    news_warning: str | None = None,
) -> str:
    direction = resolve_signal_direction(signal)
    direction_label = direction.value.upper()
    analysis = summarize_analysis_sentences(results or {}, direction)

    lines = [
        f"{symbol} {direction_label}",
        "",
        f"Entry: {signal.entry:.2f}",
        f"SL: {signal.stop_loss:.2f}",
        "",
        f"✅ TP1: {signal.tp1:.2f}",
        f"✅ TP2: {signal.tp2:.2f}",
        f"✅ TP3: {signal.tp3:.2f}",
        "",
        f"TF: {timeframe}",
        "",
        *analysis,
    ]
    if news_warning:
        lines.extend(["", news_warning])
    return "\n".join(lines)


def format_trade_result(symbol: str, direction: Direction, event: str) -> str:
    """Format a TP or stop-loss result message for Telegram."""
    direction_label = direction.value.upper()
    event_formats = {
        "tp1": ("TP1:", "✅ TP1 HIT"),
        "tp2": ("TP2:", "✅✅ TP2 HIT"),
        "tp3": ("TP3:", "✅✅✅ TP3 HIT 🔥"),
        "stop_loss": ("Stop loss:", "🔴 STOP LOSS HIT"),
    }

    label, headline = event_formats[event]
    return f"{label}\n{headline}\n\n{symbol} {direction_label}"


def format_trade_update(symbol: str, direction: Direction, event: str) -> str:
    """Format a trade monitor update for TP/SL events."""
    return format_trade_result(symbol, direction, event)


def format_trade_update_warning(
    symbol: str,
    direction: Direction,
    reasons: list[str],
) -> str:
    """Format a Level 1 trade update warning for active positions."""
    direction_label = direction.value.upper()
    lines = [
        "⚠️ TRADE UPDATE",
        "",
        f"{symbol} {direction_label}",
        "",
        *reasons,
        "",
        "Monitor position closely.",
    ]
    return "\n".join(lines)


def format_high_risk_update(
    symbol: str,
    direction: Direction,
    reasons: list[str],
) -> str:
    """Format a Level 2 high-risk trade update warning."""
    direction_label = direction.value.upper()
    lines = [
        "⚠️ HIGH RISK UPDATE",
        "",
        f"{symbol} {direction_label}",
        "",
        *reasons,
        "",
        "Consider closing the position manually.",
    ]
    return "\n".join(lines)


def format_agent_result(agent_name: str, result: AgentResult) -> str:
    direction = result.direction.value.upper()
    confidence_pct = min(100, int(round(result.confidence * 100)))
    return (
        f"📡 {agent_name.upper()}\n"
        f"Bias: {direction}\n"
        f"Confidence: {confidence_pct}%"
    )


def format_agent_summary(results: dict[str, AgentResult]) -> str:
    lines = ["📊 Agent Summary"]
    labels = {
        "smc": "SMC",
        "liquidity": "Liquidity",
        "rsi": "RSI",
        "session": "Session",
    }
    for name, result in results.items():
        label = labels.get(name, name.upper())
        confidence_pct = min(100, int(round(result.confidence * 100)))
        lines.append(
            f"• {label}: {result.direction.value.upper()} ({confidence_pct}%)"
        )
    return "\n".join(lines)
