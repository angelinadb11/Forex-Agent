from __future__ import annotations

import random

from agents.base import AgentResult, Direction
from signal_generator import TradeSignal, resolve_signal_direction
from tracking.trade_pnl import (
    DEFAULT_LOT_SIZE,
    distance_to_sl_pips,
    pip_size_for_symbol,
    pips_to_dollars,
    price_distance_pips,
    signed_pips_long,
    signed_pips_short,
)

BULLISH_ANALYSIS_PHRASES = (
    "Свіп ліквідності підтверджено.",
    "Бичача структура збережена.",
    "Покупці активні після свіпу.",
    "Імпульс залишається вгору.",
    "Очікуємо продовження росту.",
    "Структура ринку на користь лонгу.",
)

BEARISH_ANALYSIS_PHRASES = (
    "Свіп ліквідності підтверджено.",
    "Ведмежа структура збережена.",
    "Продавці активні після свіпу.",
    "Імпульс залишається вниз.",
    "Очікуємо продовження падіння.",
    "Структура ринку на користь шорту.",
)

SCALP_BULLISH_PHRASES = (
    "Ліквідність підтверджена.",
    "Структура бичача.",
    "Покупці активні після свіпу.",
    "Імпульс залишається вгору.",
)

SCALP_BEARISH_PHRASES = (
    "Ліквідність підтверджена.",
    "Структура ведмежа.",
    "Продавці активні після свіпу.",
    "Імпульс залишається вниз.",
)


def _append_warnings(lines: list[str], *warnings: str | None) -> None:
    for warning in warnings:
        if warning:
            lines.extend(["", warning])


def format_scalp_direction_label(direction: Direction) -> str:
    if direction == Direction.LONG:
        return "КУПИТИ"
    if direction == Direction.SHORT:
        return "ПРОДАТИ"
    return direction.value.upper()


def summarize_scalp_analysis_sentences(direction: Direction) -> list[str]:
    pool = SCALP_BULLISH_PHRASES if direction == Direction.LONG else SCALP_BEARISH_PHRASES
    return random.sample(pool, min(2, len(pool)))


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


def format_scalp_trade_signal(
    symbol: str,
    signal: TradeSignal,
    results: dict[str, AgentResult] | None = None,
    news_warning: str | None = None,
    off_hours_warning: str | None = None,
    h4_mismatch_warning: str | None = None,
    timeframe: str = "5m",
) -> str:
    direction = resolve_signal_direction(signal)
    direction_label = format_scalp_direction_label(direction)
    analysis = summarize_scalp_analysis_sentences(direction)
    risk = abs(signal.entry - signal.stop_loss)
    tp1_r = abs(signal.tp1 - signal.entry) / risk if risk else 0.0
    tp2_r = abs(signal.tp2 - signal.entry) / risk if risk else 0.0

    lines = [
        "⚡ СКАЛЬП",
        "",
        f"{symbol} {direction_label}",
        "",
        f"Вхід: {signal.entry:.2f}",
        f"Стоп: {signal.stop_loss:.2f}",
        "",
        f"✅ ТП1: {signal.tp1:.2f} ({tp1_r:.0f}R)",
        f"✅ ТП2: {signal.tp2:.2f} ({tp2_r:.0f}R)",
        "",
        f"ТФ: {format_timeframe_label(timeframe)}",
        "",
        *analysis,
    ]
    _append_warnings(lines, off_hours_warning, h4_mismatch_warning, news_warning)
    return "\n".join(lines)


def format_premium_scalp_trade_signal(
    symbol: str,
    signal: TradeSignal,
    *,
    timeframe: str = "5m",
) -> str:
    """VIP / Premium Sweep+FVG scalp signal."""
    direction = resolve_signal_direction(signal)
    direction_label = format_scalp_direction_label(direction)
    risk = abs(signal.entry - signal.stop_loss)
    tp1_r = abs(signal.tp1 - signal.entry) / risk if risk else 0.0
    tp2_r = abs(signal.tp2 - signal.entry) / risk if risk else 0.0

    lines = [
        "⭐ VIP ПРЕМІUM",
        "Sweep + FVG | ліміт на край FVG",
        "",
        f"{symbol} {direction_label}",
        "",
        f"Вхід (ліміт): {signal.entry:.2f}",
        f"Стоп: {signal.stop_loss:.2f}",
        "",
        f"✅ ТП1: {signal.tp1:.2f} ({tp1_r:.0f}R) — закрити 50%, SL у вхід",
        f"✅ ТП2: {signal.tp2:.2f} ({tp2_r:.0f}R)",
        "",
        f"ТФ: {format_timeframe_label(timeframe)}",
        "",
        "Рівні: Азія / година / пули ліквідності",
        "Після ТП1 — перенеси стоп у беззбиток.",
    ]
    return "\n".join(lines)


def format_turtle_soup_scalp_trade_signal(
    symbol: str,
    signal: TradeSignal,
    *,
    timeframe: str = "5m",
) -> str:
    """VIP 2 — ICT Turtle Soup scalp signal."""
    direction = resolve_signal_direction(signal)
    direction_label = format_scalp_direction_label(direction)
    risk = abs(signal.entry - signal.stop_loss)
    tp1_r = abs(signal.tp1 - signal.entry) / risk if risk else 0.0
    tp2_r = abs(signal.tp2 - signal.entry) / risk if risk else 0.0
    tp3_r = abs(signal.tp3 - signal.entry) / risk if risk else 0.0

    lines = [
        "🐢 VIP 2 — TURTLE SOUP",
        "ICT failed breakout | вхід на close sweep-свічки",
        "",
        f"{symbol} {direction_label}",
        "",
        f"Вхід: {signal.entry:.2f}",
        f"Стоп: {signal.stop_loss:.2f}",
        "",
        f"✅ ТП1: {signal.tp1:.2f} ({tp1_r:.1f}R) — закрити 50%, SL у вхід",
        f"✅ ТП2: {signal.tp2:.2f} ({tp2_r:.1f}R)",
        f"✅ ТП3: {signal.tp3:.2f} ({tp3_r:.1f}R) — ліквідність",
        "",
        f"ТФ: {format_timeframe_label(timeframe)}",
        "",
        "Рівні: Азія / день / година / swing / пули",
        "Сесія: 08:00–16:00 UTC (London core)",
        "Після ТП1 — перенеси стоп у беззбиток.",
    ]
    return "\n".join(lines)


def format_trade_signal(
    symbol: str,
    signal: TradeSignal,
    timeframe: str,
    results: dict[str, AgentResult] | None = None,
    news_warning: str | None = None,
    off_hours_warning: str | None = None,
    h4_mismatch_warning: str | None = None,
    *,
    tier_header: str | None = None,
) -> str:
    direction = resolve_signal_direction(signal)
    direction_label = format_trade_direction_label(direction)
    analysis = summarize_analysis_sentences(results or {}, direction)

    lines: list[str] = []
    if tier_header:
        lines.extend([tier_header, ""])
    lines.extend([
        f"{symbol} {direction_label}",
        "",
        f"Вхід: {signal.entry:.2f}",
        f"Стоп: {signal.stop_loss:.2f}",
        "",
        f"✅ ТП1: {signal.tp1:.2f}",
        f"✅ ТП2: {signal.tp2:.2f}",
        f"✅ ТП3: {signal.tp3:.2f}",
        "",
        f"ТФ: {format_timeframe_label(timeframe)}",
        "",
        *analysis,
    ])
    _append_warnings(lines, off_hours_warning, h4_mismatch_warning, news_warning)
    return "\n".join(lines)


def format_tradingview_alert(
    alert: "TradingViewAlert",
) -> str:
    """Format a TradingView indicator alert for the main Trading Boss channel."""
    from webhook.tradingview import TradingViewAlert as _Alert

    if not isinstance(alert, _Alert):
        raise TypeError("alert must be TradingViewAlert")

    def _level(value: float | None) -> str:
        return f"{value:.2f}" if value is not None else "—"

    def _action_label() -> str:
        if alert.direction == Direction.LONG:
            return "BUY"
        if alert.direction == Direction.SHORT:
            return "SELL"
        return "NEUTRAL"

    def _header_emoji() -> str:
        if alert.direction == Direction.LONG:
            return "🟢"
        if alert.direction == Direction.SHORT:
            return "🔴"
        return "⚪"

    def _timeframe_display() -> str:
        tf = alert.timeframe.strip()
        if tf.endswith("m") or tf.endswith("h"):
            return tf
        return format_timeframe_label(tf)

    source = f" ({alert.source})" if alert.source else ""
    lines = [
        f"{_header_emoji()} {alert.symbol} {_action_label()}{source}",
        "",
        f"Enter: {_level(alert.entry)}",
        f"SL: {_level(alert.stop_loss)}",
        f"TP1: {_level(alert.tp1)}",
        f"TP2: {_level(alert.tp2)}",
        f"TP3: {_level(alert.tp3)}",
        "",
        f"⏱ Таймфрейм: {_timeframe_display()}",
    ]
    return "\n".join(lines)


def format_trading_boss_trade_signal(
    symbol: str,
    signal: TradeSignal,
    timeframe: str,
    *,
    tier_header: str,
    results: dict[str, AgentResult] | None = None,
    news_warning: str | None = None,
    off_hours_warning: str | None = None,
    h4_mismatch_warning: str | None = None,
) -> str:
    """Trading Boss main channel signal with tier label (Active / Select)."""
    return format_trade_signal(
        symbol,
        signal,
        timeframe,
        results,
        news_warning,
        off_hours_warning,
        h4_mismatch_warning,
        tier_header=tier_header,
    )


def format_near_tp1_breakeven_warning(
    *,
    direction: Direction,
    current_price: float,
    entry: float,
    peak_progress_r: float,
    conditions: tuple[str, ...] = (),
) -> str:
    reason_map = {
        "bos_against": "M15 BOS проти позиції",
        "rsi_reversal": "RSI розвертається",
        "entry_zone_break": "Пробій entry zone",
    }
    reasons = [reason_map.get(item, item) for item in conditions]
    reason_text = ", ".join(reasons) if reasons else "ознаки розвороту на M15"

    return "\n".join(
        [
            "⚠️ УВАГА — Майже TP1, ринок розвертається",
            f"└ Прогрес: {peak_progress_r:.2f}R (TP1 ще не взято)",
            f"└ Причина: {reason_text}",
            f"└ Поточна ціна: {current_price:.2f}",
            f"└ Точка входу: {entry:.2f}",
            "",
            "📌 Що робити зараз:",
            "• Перенеси SL на точку входу (беззбиток)",
            "  — ціна вже дала ~1.2R+, не віддавай повний стоп",
            "• Не додавай до позиції",
            "• Чекай TP1 або вихід по беззбитку",
        ]
    )


def format_trade_result(symbol: str, direction: Direction, event: str) -> str:
    """Format a TP, breakeven, or stop-loss result message for Telegram."""
    direction_label = format_trade_direction_label(direction)
    event_formats = {
        "tp1": ("ТП1:", "✅ ТП1 ДОСЯГНУТО"),
        "tp2": ("ТП2:", "✅✅ ТП2 ДОСЯГНУТО"),
        "tp3": ("ТП3:", "✅✅✅ ТП3 ДОСЯГНУТО 🔥"),
        "stop_loss": ("Стоп:", "🔴 СТОП-ЛОСС"),
        "breakeven": ("Беззбиток:", "⚪ БЕЗЗБИТОК"),
    }

    label, headline = event_formats[event]
    return f"{label}\n{headline}\n\n{symbol} {direction_label}"


def format_breakeven_reply(
    *,
    direction: Direction,
    entry: float,
    exit_price: float,
) -> str:
    direction_label = format_trade_direction_label(direction)
    return "\n".join(
        [
            "⚪ Закрили в BE",
            f"└ Сигнал: {direction_label} {entry:.2f}",
            f"└ Ціна закриття: {exit_price:.2f}",
            "└ Результат: 0R 👌",
        ]
    )


def format_post_tp_close_reply(
    *,
    symbol: str,
    direction: Direction,
    entry: float,
    exit_price: float,
    tp1: float,
    tp2: float,
    tp2_hit: bool,
) -> str:
    """Remainder stopped out after TP1/TP2 was taken — profit is locked, not BE."""
    direction_label = format_trade_direction_label(direction)
    highest_tp = 2 if tp2_hit else 1
    tp1_pips = trade_move_pips(
        symbol=symbol, direction=direction, entry=entry, price=tp1
    )
    lines = [
        f"✅ Угоду закрито з прибутком (ТП{highest_tp} взято)",
        f"└ Сигнал: {direction_label} {entry:.2f}",
        f"└ ТП1: +{tp1_pips:.1f} піпсів (50% позиції)",
    ]
    if tp2_hit:
        tp2_pips = trade_move_pips(
            symbol=symbol, direction=direction, entry=entry, price=tp2
        )
        lines.append(f"└ ТП2: +{tp2_pips:.1f} піпсів (25% позиції)")
    lines.append(f"└ Решту позиції закрито: {exit_price:.2f}")
    return "\n".join(lines)


def format_trade_update(symbol: str, direction: Direction, event: str) -> str:
    """Format a trade monitor update for TP/SL events."""
    return format_trade_result(symbol, direction, event)


def format_trade_update_warning(
    symbol: str,
    direction: Direction,
    reasons: list[str],
) -> str:
    """Format a Level 1 trade update warning for active positions."""
    direction_label = format_trade_direction_label(direction)
    lines = [
        "⚠️ ОНОВЛЕННЯ ПОЗИЦІЇ",
        "",
        f"{symbol} {direction_label}",
        "",
        *reasons,
        "",
        "Стеж за позицією уважно.",
    ]
    return "\n".join(lines)


def format_high_risk_update(
    symbol: str,
    direction: Direction,
    reasons: list[str],
) -> str:
    """Format a Level 2 high-risk trade update warning."""
    direction_label = format_trade_direction_label(direction)
    lines = [
        "⚠️ ВИСОКИЙ РИЗИК",
        "",
        f"{symbol} {direction_label}",
        "",
        *reasons,
        "",
        "Розглянь ручне закриття позиції.",
    ]
    return "\n".join(lines)


def format_timeframe_label(timeframe: str) -> str:
    if timeframe.startswith("5m"):
        return "5 хв"
    if timeframe.startswith("1m"):
        return "1 хв"
    mapping = {
        "15m": "15 хв",
        "1h": "1 год",
        "4h": "4 год",
    }
    return mapping.get(timeframe, timeframe)


def format_trade_direction_label(direction: Direction) -> str:
    if direction == Direction.LONG:
        return "ЛОНГ"
    if direction == Direction.SHORT:
        return "ШОРТ"
    return direction.value.upper()


def format_trend_change_warning(
    *,
    open_time: str,
    direction: Direction,
    current_price: float,
    entry: float | None = None,
    reason: str = "Зміна тренду H1 проти позиції",
) -> str:
    lines = [
        "⚠️ УВАГА — Зміна тренду",
        f"└ Причина: {reason}",
        f"└ Поточна ціна: {current_price:.2f}",
        "└ TP1 ще не досягнуто",
    ]
    if entry is not None:
        lines.append(f"└ Точка входу: {entry:.2f}")
    lines.extend(
        [
            "",
            "📌 Що робити зараз:",
            "• Перенеси SL на точку входу (беззбиток)",
            "  — тоді при відкаті не буде мінуса",
            "• Не додавай до позиції",
            "• Чекай TP1/TP2 або вихід по беззбитку",
        ]
    )
    return "\n".join(lines)


def format_sl_proximity_warning(
    *,
    current_price: float,
    remaining_pips: float,
) -> str:
    return "\n".join(
        [
            "⚠️ SL близько",
            f"└ Поточна ціна: {current_price:.2f}",
            f"└ До SL залишилось: {remaining_pips:.1f} pips",
            "",
            "📌 Що робити зараз:",
            "• Можеш закрити вручну щоб зменшити збиток",
            "• Не пересувай SL далі — це збільшить ризик",
            "• Чекай рішення ринку",
        ]
    )


def format_stop_loss_reply(
    *,
    result_dollars: float,
) -> str:
    return "\n".join(
        [
            "❌ СТОП-ЛОСС",
            f"└ Результат: -${result_dollars:.2f}",
            "",
            "📌 Що робити зараз:",
            "• Не намагайся відіграти одразу",
            "• Чекай наступного сигналу системи",
            "• Це нормальна частина торгівлі 💪",
            "• Один збиток не вирішує результат місяця",
        ]
    )


def format_take_profit_reply(
    *,
    tp_level: int,
    entry: float,
    tp_price: float,
    move_pips: float,
    tp1: float,
    tp2: float,
    tp3: float,
) -> str:
    move_pips = abs(move_pips)

    if tp_level == 1:
        return "\n".join(
            [
                "✅ ТЕЙК-ПРОФІТ 1",
                f"└ TP1 хітнуло: {tp_price:.2f}",
                f"└ Хід: +{move_pips:.1f} pips",
                "",
                "📌 Що робити зараз:",
                "• Закрий 50% позиції на TP1",
                f"• Перенеси SL на точку входу {entry:.2f} для решти 50%",
                f"• Решта позиції йде до TP2: {tp2:.2f}",
                "• Ти вже зафіксував частину прибутку 🎯",
            ]
        )

    if tp_level == 2:
        return "\n".join(
            [
                "✅ ТЕЙК-ПРОФІТ 2",
                f"└ TP2 хітнуло: {tp_price:.2f}",
                f"└ Хід: +{move_pips:.1f} pips",
                "",
                "📌 Що робити зараз:",
                "• Закрий ще 25% позиції",
                f"• Перенеси SL на TP1: {tp1:.2f} для останніх 25%",
                f"• Решта позиції йде до TP3: {tp3:.2f}",
            ]
        )

    if tp_level == 3:
        return "\n".join(
            [
                "✅ ТЕЙК-ПРОФІТ 3",
                f"└ TP3 хітнуло: {tp_price:.2f}",
                f"└ Хід: +{move_pips:.1f} pips",
                "",
                "📌 Що робити зараз:",
                "• На ваш розсуд — можете закрити всю позицію",
                "• Або тримати далі, якщо SL уже перенесено в плюс",
            ]
        )

    return "\n".join(
        [
            f"✅ ТЕЙК-ПРОФІТ {tp_level}",
            f"└ TP{tp_level} хітнуло: {tp_price:.2f}",
            f"└ Хід: +{move_pips:.1f} pips",
            "",
            "📌 Що робити зараз:",
            "• На ваш розсуд — можете закрити всю позицію",
            "• Або тримати далі, якщо SL уже перенесено в плюс",
        ]
    )


def trade_move_pips(
    *,
    symbol: str,
    direction: Direction,
    entry: float,
    price: float,
) -> float:
    pip_size = pip_size_for_symbol(symbol)
    if pip_size is None:
        return abs(price - entry)
    if direction == Direction.LONG:
        return signed_pips_long(entry, price, pip_size)
    return signed_pips_short(entry, price, pip_size)


def trade_result_dollars(
    *,
    symbol: str,
    pips: float,
    lot_size: float = DEFAULT_LOT_SIZE,
) -> float:
    return pips_to_dollars(symbol, pips, lot_size)


def format_agent_result(agent_name: str, result: AgentResult) -> str:
    direction = format_trade_direction_label(result.direction)
    confidence_pct = min(100, int(round(result.confidence * 100)))
    labels = {
        "bias": "Bias (HTF)",
        "structure": "Структура",
        "execution": "Execution",
        "smc": "SMC",
        "liquidity": "Ліквідність",
        "fvg": "FVG",
        "order_block": "Order Block",
        "rsi": "RSI",
        "session": "Сесія",
    }
    label = labels.get(agent_name, agent_name.upper())
    return (
        f"📡 {label}\n"
        f"Напрям: {direction}\n"
        f"Впевненість: {confidence_pct}%"
    )


def format_agent_summary(results: dict[str, AgentResult]) -> str:
    lines = ["📊 Підсумок агентів"]
    labels = {
        "bias": "Bias (HTF)",
        "structure": "Структура",
        "execution": "Execution",
        "smc": "SMC",
        "liquidity": "Ліквідність",
        "fvg": "FVG",
        "order_block": "Order Block",
        "rsi": "RSI",
        "session": "Сесія",
    }
    for name, result in results.items():
        label = labels.get(name, name.upper())
        confidence_pct = min(100, int(round(result.confidence * 100)))
        direction = format_trade_direction_label(result.direction)
        lines.append(f"• {label}: {direction} ({confidence_pct}%)")
    return "\n".join(lines)
