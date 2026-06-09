import unittest

from agents.base import Direction
from backtest.simulator import TradeManagementMode, TradeSimulator
from signal_generator import TradeSignal


class TradeSimulatorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.simulator = TradeSimulator()
        self.long_signal = TradeSignal(
            Direction.LONG,
            entry=100.0,
            stop_loss=90.0,
            tp1=110.0,
            tp2=120.0,
            tp3=130.0,
            confidence=0.8,
            reason="test",
        )

    def test_legacy_breakeven_after_tp1_is_zero_r(self) -> None:
        candles = [
            {"open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0},
            {"open": 115.0, "high": 115.5, "low": 99.5, "close": 100.0},
        ]
        result = self.simulator.simulate(
            self.long_signal,
            candles,
            entry_index=0,
            mode=TradeManagementMode.LEGACY,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.tp1_hit)
        self.assertEqual(result.result, "breakeven")
        self.assertAlmostEqual(result.pnl_r, 0.0)

    def test_partial_breakeven_after_tp1_locks_half_at_tp1(self) -> None:
        candles = [
            {"open": 100.0, "high": 116.0, "low": 99.0, "close": 115.0},
            {"open": 115.0, "high": 115.5, "low": 99.5, "close": 100.0},
        ]
        result = self.simulator.simulate(
            self.long_signal,
            candles,
            entry_index=0,
            mode=TradeManagementMode.PARTIAL,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.tp1_hit)
        self.assertEqual(result.result, "breakeven")
        self.assertAlmostEqual(result.pnl_r, 0.5)

    def test_partial_full_tp3_reaches_weighted_target(self) -> None:
        candles = [
            {"open": 100.0, "high": 136.0, "low": 99.0, "close": 135.0},
        ]
        result = self.simulator.simulate(
            self.long_signal,
            candles,
            entry_index=0,
            mode=TradeManagementMode.PARTIAL,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.result, "tp3")
        self.assertAlmostEqual(result.pnl_r, 1.75)


    def test_partial_full_stop_is_stop_loss(self) -> None:
        candles = [
            {"open": 100.0, "high": 101.0, "low": 89.0, "close": 90.0},
        ]
        result = self.simulator.simulate(
            self.long_signal,
            candles,
            entry_index=0,
            mode=TradeManagementMode.PARTIAL,
        )
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual(result.result, "stop_loss")
        self.assertAlmostEqual(result.pnl_r, -1.0)


if __name__ == "__main__":
    unittest.main()
