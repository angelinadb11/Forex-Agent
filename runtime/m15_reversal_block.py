from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from agents.base import Direction
from config.symbols import resolve_symbol
from runtime.dedup import DedupDecision
from signal_generator import TradeSignal, resolve_signal_direction
from strategy.near_tp1_breakeven import assess_m15_reversal_conditions
from strategy.structure_weakness import resolve_entry_rsi

ContextFetcher = Callable[[str, str], dict[str, Any]]


@dataclass
class M15ReversalBlockState:
    direction: Direction
    timeframe: str
    entry_zone_low: float | None
    entry_zone_high: float | None
    entry_rsi: float | None
    last_rsi: float | None = None


class M15ReversalBlockGate:
    """Block repeat signals in the same direction while M15 reversal persists."""

    def __init__(self, context_fetcher: ContextFetcher) -> None:
        self.context_fetcher = context_fetcher
        self._blocks: dict[str, M15ReversalBlockState] = {}

    def register_from_trade(self, trade) -> None:
        display_symbol = resolve_symbol(trade.symbol).display
        self._blocks[display_symbol] = M15ReversalBlockState(
            direction=trade.direction,
            timeframe=trade.timeframe or "15m",
            entry_zone_low=trade.entry_zone_low,
            entry_zone_high=trade.entry_zone_high,
            entry_rsi=trade.entry_rsi,
            last_rsi=trade.last_rsi,
        )

    def seed_from_active_trades(self, trades) -> None:
        for trade in trades:
            if trade.closed:
                continue
            if trade.near_tp1_warning_sent:
                self.register_from_trade(trade)

    def can_publish(
        self,
        symbol: str,
        signal: TradeSignal,
        timeframe: str,
    ) -> DedupDecision:
        display_symbol = resolve_symbol(symbol).display
        state = self._blocks.get(display_symbol)
        if state is None:
            return DedupDecision(allowed=True)

        signal_direction = resolve_signal_direction(signal)
        if signal_direction != state.direction:
            return DedupDecision(allowed=True)

        m15_context = self.context_fetcher(symbol, timeframe or state.timeframe)
        current_rsi = resolve_entry_rsi(m15_context)
        active, conditions = assess_m15_reversal_conditions(
            state.direction,
            m15_context=m15_context,
            entry_zone_low=state.entry_zone_low,
            entry_zone_high=state.entry_zone_high,
            entry_rsi=state.entry_rsi,
            previous_rsi=state.last_rsi,
        )
        if current_rsi is not None:
            state.last_rsi = current_rsi

        if not active:
            del self._blocks[display_symbol]
            return DedupDecision(allowed=True)

        joined = ", ".join(conditions) if conditions else "reversal"
        return DedupDecision(
            allowed=False,
            reason=(
                f"M15 reversal still active for {display_symbol} "
                f"{state.direction.value} ({joined})"
            ),
        )
