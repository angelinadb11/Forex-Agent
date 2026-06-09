import unittest

import pandas as pd

from agents.base import Direction
from agents.rsi_agent import RSIAgent
from agents.trend_filter_agent import TrendBias, classify_trend_from_emas, evaluate_h1_trend, trend_bias_to_direction


class TrendFilterAgentTests(unittest.TestCase):
    def test_price_above_ema50_is_bullish(self):
        closes = pd.Series([100.0] * 199 + [120.0])
        bias, _, ema50, _, _ = evaluate_h1_trend(closes)
        self.assertEqual(bias, TrendBias.BULLISH)
        self.assertLess(ema50, 120.0)

    def test_price_below_ema50_is_bearish(self):
        closes = pd.Series([120.0] * 199 + [100.0])
        bias, _, ema50, _, _ = evaluate_h1_trend(closes)
        self.assertEqual(bias, TrendBias.BEARISH)
        self.assertGreater(ema50, 100.0)

    def test_price_between_emas_is_neutral(self):
        bias = classify_trend_from_emas(price=110.0, ema50=105.0, ema200=115.0)
        self.assertEqual(bias, TrendBias.NEUTRAL)

    def test_trend_bias_maps_to_direction(self):
        self.assertEqual(trend_bias_to_direction(TrendBias.BULLISH), Direction.LONG)
        self.assertEqual(trend_bias_to_direction(TrendBias.BEARISH), Direction.SHORT)
        self.assertEqual(trend_bias_to_direction(TrendBias.NEUTRAL), Direction.NEUTRAL)


class RSIAgentTrendConfirmationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = RSIAgent()

    def test_oversold_with_bullish_trend_confirms_long(self):
        direction, confidence, reason = self.agent._evaluate_rsi(
            rsi=25.0,
            trend_direction=Direction.LONG,
            symbol="XAUUSD",
            timeframe="15m",
        )
        self.assertEqual(direction, Direction.LONG)
        self.assertGreater(confidence, 0.0)
        self.assertIn("confirms bullish H1 trend", reason)

    def test_oversold_with_bearish_trend_is_ignored(self):
        direction, confidence, reason = self.agent._evaluate_rsi(
            rsi=25.0,
            trend_direction=Direction.SHORT,
            symbol="XAUUSD",
            timeframe="15m",
        )
        self.assertEqual(direction, Direction.NEUTRAL)
        self.assertEqual(confidence, 0.0)
        self.assertIn("ignored", reason)

    def test_overbought_with_bearish_trend_confirms_short(self):
        direction, confidence, reason = self.agent._evaluate_rsi(
            rsi=75.0,
            trend_direction=Direction.SHORT,
            symbol="XAUUSD",
            timeframe="15m",
        )
        self.assertEqual(direction, Direction.SHORT)
        self.assertGreater(confidence, 0.0)
        self.assertIn("confirms bearish H1 trend", reason)


if __name__ == "__main__":
    unittest.main()
