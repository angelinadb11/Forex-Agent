import unittest

import pandas as pd

from agents.base import Direction
from strategy.bollinger_gate import calculate_bollinger_bands, evaluate_bb_gate


def _context_from_closes(closes: list[float]) -> dict:
    return {"candles": [{"close": value} for value in closes]}


class BollingerBandsCalcTests(unittest.TestCase):
    def test_flat_series_bands_collapse_to_price(self):
        closes = pd.Series([100.0] * 25)
        lower, middle, upper = calculate_bollinger_bands(closes)
        self.assertAlmostEqual(lower, 100.0)
        self.assertAlmostEqual(middle, 100.0)
        self.assertAlmostEqual(upper, 100.0)

    def test_requires_enough_closes(self):
        with self.assertRaises(ValueError):
            calculate_bollinger_bands(pd.Series([100.0] * 5))


class BollingerGateTests(unittest.TestCase):
    def _trending_down_closes(self) -> list[float]:
        # Strong down-move: last close far below the lower band.
        return [100.0] * 20 + [99.0, 97.0, 94.0, 90.0, 85.0]

    def _trending_up_closes(self) -> list[float]:
        return [100.0] * 20 + [101.0, 103.0, 106.0, 110.0, 115.0]

    def test_blocks_short_at_lower_band(self):
        context = _context_from_closes(self._trending_down_closes())
        block = evaluate_bb_gate(context, Direction.SHORT)
        self.assertIsNotNone(block)
        self.assertIn("lower Bollinger Band", block)

    def test_allows_long_at_lower_band(self):
        context = _context_from_closes(self._trending_down_closes())
        self.assertIsNone(evaluate_bb_gate(context, Direction.LONG))

    def test_blocks_long_at_upper_band(self):
        context = _context_from_closes(self._trending_up_closes())
        block = evaluate_bb_gate(context, Direction.LONG)
        self.assertIsNotNone(block)
        self.assertIn("upper Bollinger Band", block)

    def test_allows_short_at_upper_band(self):
        context = _context_from_closes(self._trending_up_closes())
        self.assertIsNone(evaluate_bb_gate(context, Direction.SHORT))

    def test_allows_trade_mid_band(self):
        closes = [100.0 + (0.2 if i % 2 else -0.2) for i in range(30)]
        context = _context_from_closes(closes)
        self.assertIsNone(evaluate_bb_gate(context, Direction.LONG))
        self.assertIsNone(evaluate_bb_gate(context, Direction.SHORT))

    def test_no_context_allows_trade(self):
        self.assertIsNone(evaluate_bb_gate(None, Direction.LONG))

    def test_short_history_allows_trade(self):
        context = _context_from_closes([100.0] * 5)
        self.assertIsNone(evaluate_bb_gate(context, Direction.SHORT))


if __name__ == "__main__":
    unittest.main()
