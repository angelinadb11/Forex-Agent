from __future__ import annotations

from dataclasses import dataclass

from agents.base import Direction
from backtest.simulator import SimulatedTradeResult
from signal_generator import TradeSignal, resolve_signal_direction
from strategy.near_tp1_breakeven import assess_m15_reversal_conditions
from strategy.structure_weakness import resolve_entry_rsi


@dataclass
class _BlockState:
    direction: Direction
    entry_zone_low: float | None
    entry_zone_high: float | None
    entry_rsi: float | None
    last_rsi: float | None = None


class BacktestM15ReversalBlock:
    """Skip repeat same-direction setups while M15 reversal persists (live gate parity)."""

    def __init__(self) -> None:
        self._state: _BlockState | None = None
        self.blocked_setups = 0

    def register_from_trade(self, trade: SimulatedTradeResult) -> None:
        if not (
            trade.near_tp1_be_triggered
            or (trade.result == "breakeven" and not trade.tp1_hit)
        ):
            return
        self._state = _BlockState(
            direction=Direction(trade.direction),
            entry_zone_low=trade.entry_zone_low,
            entry_zone_high=trade.entry_zone_high,
            entry_rsi=trade.entry_rsi,
            last_rsi=trade.last_rsi,
        )

    def blocks_setup(
        self,
        signal: TradeSignal,
        candles: list[dict[str, float]],
        index: int,
        *,
        symbol: str,
        zone_catalog,
    ) -> bool:
        if self._state is None:
            return False

        signal_direction = resolve_signal_direction(signal)
        if signal_direction != self._state.direction:
            return False

        context = {
            "symbol": symbol,
            "candles": candles[: index + 1],
            "bar_index": index,
        }
        if zone_catalog is not None:
            context["zone_catalog"] = zone_catalog

        current_rsi = resolve_entry_rsi(context)
        active, _ = assess_m15_reversal_conditions(
            self._state.direction,
            m15_context=context,
            entry_zone_low=self._state.entry_zone_low,
            entry_zone_high=self._state.entry_zone_high,
            entry_rsi=self._state.entry_rsi,
            previous_rsi=self._state.last_rsi,
        )
        if current_rsi is not None:
            self._state.last_rsi = current_rsi

        if not active:
            self._state = None
            return False

        self.blocked_setups += 1
        return True
