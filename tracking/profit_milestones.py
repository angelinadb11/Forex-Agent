from __future__ import annotations

import random
from dataclasses import dataclass

from agents.base import Direction
from config.symbols import resolve_symbol
from tracking.trade_pnl import pip_size_for_symbol, signed_pips_long, signed_pips_short

MILESTONE_TIERS: tuple[int, ...] = (20, 50, 80, 100)
CRYPTO_INDEX_TIER_MULTIPLIER = 3
CRYPTO_INDEX_SYMBOLS = frozenset({"BTCUSDT", "DJ30"})

MILESTONE_MESSAGE_TEMPLATES: dict[int, tuple[str, ...]] = {
    20: (
        "📈 +{pips} піпсів — йде!",
        "📈 Вже +{pips} — тримаємо!",
    ),
    50: (
        "🔥 +{pips} піпсів!",
        "🔥 Красиво йде +{pips}!",
    ),
    80: (
        "😈 +{pips} піпсів — монстр!",
        "😈 +{pips} піпсів! Ось це рух!",
    ),
    100: (
        "🚀 Сотка +{pips} піпсів!!!",
        "🚀 +{pips} піпсів! Хто в позиції — красавці!",
    ),
}


@dataclass(frozen=True)
class ProfitMilestone:
    tier: int
    threshold_pips: float


def _tier_multiplier(symbol: str) -> int:
    try:
        display = resolve_symbol(symbol).display
    except ValueError:
        display = symbol.upper()
    if display in CRYPTO_INDEX_SYMBOLS:
        return CRYPTO_INDEX_TIER_MULTIPLIER
    return 1


def profit_milestones_for_symbol(symbol: str) -> tuple[ProfitMilestone, ...]:
    multiplier = _tier_multiplier(symbol)
    return tuple(
        ProfitMilestone(tier=tier, threshold_pips=tier * multiplier)
        for tier in MILESTONE_TIERS
    )


def _format_milestone_message(tier: int, threshold_pips: int) -> tuple[str, ...]:
    pips = int(threshold_pips)
    if tier == 100 and pips == 100:
        return (
            "🚀 Сотка +100 піпсів!!!",
            "🚀 +100 піпсів! Хто в позиції — красавці!",
        )
    return tuple(template.format(pips=pips) for template in MILESTONE_MESSAGE_TEMPLATES[tier])


def pick_profit_milestone_message(
    milestone: ProfitMilestone,
    *,
    rng: random.Random | None = None,
) -> str:
    source = rng if rng is not None else random
    options = _format_milestone_message(milestone.tier, int(milestone.threshold_pips))
    return source.choice(options)


def favorable_price(direction: Direction, *, high: float, low: float) -> float:
    if direction == Direction.LONG:
        return high
    return low


def profit_pips_for_trade(
    *,
    symbol: str,
    direction: Direction,
    entry: float,
    price: float,
) -> float:
    pip_size = pip_size_for_symbol(symbol)
    if pip_size is None:
        if direction == Direction.LONG:
            return price - entry
        if direction == Direction.SHORT:
            return entry - price
        return 0.0
    if direction == Direction.LONG:
        return signed_pips_long(entry, price, pip_size)
    if direction == Direction.SHORT:
        return signed_pips_short(entry, price, pip_size)
    return 0.0


def pending_profit_milestone_messages(
    trade,
    *,
    high: float,
    low: float,
    rng: random.Random | None = None,
) -> list[str]:
    """Return motivational messages for newly reached profit tiers."""
    if trade.tp1_hit or trade.tp2_hit or trade.tp3_hit:
        return []

    price = favorable_price(trade.direction, high=high, low=low)
    profit_pips = profit_pips_for_trade(
        symbol=trade.symbol,
        direction=trade.direction,
        entry=trade.entry,
        price=price,
    )
    if profit_pips <= 0:
        return []

    sent_tiers = set(getattr(trade, "profit_milestones_sent", []) or [])
    messages: list[str] = []

    for milestone in profit_milestones_for_symbol(trade.symbol):
        if milestone.tier in sent_tiers:
            continue
        if profit_pips < milestone.threshold_pips:
            continue
        messages.append(pick_profit_milestone_message(milestone, rng=rng))
        sent_tiers.add(milestone.tier)

    if messages:
        trade.profit_milestones_sent = sorted(sent_tiers)

    return messages
