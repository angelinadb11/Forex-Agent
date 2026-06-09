import unittest
from unittest.mock import patch

from agents.base import Direction
from backtest.m15_reversal_block import BacktestM15ReversalBlock
from backtest.simulator import SimulatedTradeResult
from signal_generator import TradeSignal


class BacktestM15ReversalBlockTests(unittest.TestCase):
    def _trade(
        self,
        *,
        direction: Direction = Direction.SHORT,
        result: str = "breakeven",
        tp1_hit: bool = False,
        near_tp1_be_triggered: bool = True,
    ) -> SimulatedTradeResult:
        return SimulatedTradeResult(
            entry_index=10,
            exit_index=20,
            direction=direction.value,
            entry=100.0,
            exit_price=100.0,
            stop_loss=101.0,
            tp1=98.5,
            tp2=97.5,
            tp3=96.5,
            risk=1.0,
            pnl_r=0.0,
            result=result,
            win=False,
            loss=False,
            tp1_hit=tp1_hit,
            tp2_hit=False,
            tp3_hit=False,
            confidence=0.8,
            reason="test",
            near_tp1_be_triggered=near_tp1_be_triggered,
            entry_zone_low=99.0,
            entry_zone_high=100.5,
            entry_rsi=70.0,
            last_rsi=68.0,
        )

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

    def test_registers_after_near_tp1_be_exit(self) -> None:
        block = BacktestM15ReversalBlock()
        block.register_from_trade(self._trade())
        self.assertIsNotNone(block._state)

    def test_skips_same_direction_while_reversal_active(self) -> None:
        block = BacktestM15ReversalBlock()
        block.register_from_trade(self._trade())

        with patch(
            "backtest.m15_reversal_block.assess_m15_reversal_conditions",
            return_value=(True, ("bos_against",)),
        ):
            blocked = block.blocks_setup(
                self._signal(Direction.SHORT),
                [{"high": 101.0, "low": 99.0, "close": 100.5}],
                25,
                symbol="XAUUSD",
                zone_catalog=None,
            )

        self.assertTrue(blocked)
        self.assertEqual(block.blocked_setups, 1)

    def test_clears_block_when_reversal_gone(self) -> None:
        block = BacktestM15ReversalBlock()
        block.register_from_trade(self._trade())

        with patch(
            "backtest.m15_reversal_block.assess_m15_reversal_conditions",
            return_value=(False, ()),
        ):
            blocked = block.blocks_setup(
                self._signal(Direction.SHORT),
                [{"high": 101.0, "low": 99.0, "close": 100.5}],
                25,
                symbol="XAUUSD",
                zone_catalog=None,
            )

        self.assertFalse(blocked)
        self.assertIsNone(block._state)


if __name__ == "__main__":
    unittest.main()
