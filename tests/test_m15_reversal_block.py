import unittest
from unittest.mock import MagicMock

from agents.base import Direction
from signal_generator import TradeSignal
from runtime.m15_reversal_block import M15ReversalBlockGate


class M15ReversalBlockGateTests(unittest.TestCase):
    def _signal(self, direction: Direction) -> TradeSignal:
        if direction == Direction.LONG:
            return TradeSignal(
                direction,
                entry=100.0,
                stop_loss=99.0,
                tp1=101.5,
                tp2=102.5,
                tp3=103.5,
                confidence=0.8,
                reason="test",
            )
        return TradeSignal(
            direction,
            entry=100.0,
            stop_loss=101.0,
            tp1=98.5,
            tp2=97.5,
            tp3=96.5,
            confidence=0.8,
            reason="test",
        )

    def test_blocks_same_direction_while_reversal_active(self) -> None:
        gate = M15ReversalBlockGate(context_fetcher=lambda symbol, tf: {"candles": []})
        trade = MagicMock()
        trade.symbol = "XAUUSD"
        trade.direction = Direction.SHORT
        trade.timeframe = "15m"
        trade.entry_zone_low = 100.0
        trade.entry_zone_high = 101.0
        trade.entry_rsi = 70.0
        trade.last_rsi = 68.0
        gate.register_from_trade(trade)

        with unittest.mock.patch(
            "runtime.m15_reversal_block.assess_m15_reversal_conditions",
            return_value=(True, ("bos_against",)),
        ):
            decision = gate.can_publish("XAUUSD", self._signal(Direction.SHORT), "15m")

        self.assertFalse(decision.allowed)
        self.assertIn("M15 reversal still active", decision.reason or "")

    def test_allows_opposite_direction(self) -> None:
        gate = M15ReversalBlockGate(context_fetcher=lambda symbol, tf: {"candles": []})
        trade = MagicMock()
        trade.symbol = "XAUUSD"
        trade.direction = Direction.SHORT
        trade.timeframe = "15m"
        trade.entry_zone_low = None
        trade.entry_zone_high = None
        trade.entry_rsi = None
        trade.last_rsi = None
        gate.register_from_trade(trade)

        decision = gate.can_publish("XAUUSD", self._signal(Direction.LONG), "15m")
        self.assertTrue(decision.allowed)

    def test_clears_block_when_reversal_gone(self) -> None:
        gate = M15ReversalBlockGate(context_fetcher=lambda symbol, tf: {"candles": []})
        trade = MagicMock()
        trade.symbol = "XAUUSD"
        trade.direction = Direction.SHORT
        trade.timeframe = "15m"
        trade.entry_zone_low = None
        trade.entry_zone_high = None
        trade.entry_rsi = None
        trade.last_rsi = None
        gate.register_from_trade(trade)

        with unittest.mock.patch(
            "runtime.m15_reversal_block.assess_m15_reversal_conditions",
            return_value=(False, ()),
        ):
            decision = gate.can_publish("XAUUSD", self._signal(Direction.SHORT), "15m")

        self.assertTrue(decision.allowed)
        self.assertNotIn("XAUUSD", gate._blocks)


if __name__ == "__main__":
    unittest.main()
