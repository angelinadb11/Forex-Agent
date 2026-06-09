import unittest
from unittest.mock import patch

from agents.base import AgentResult, Direction
from agents.liquidity_agent import (
    MAX_LIQUIDITY_CONFIDENCE,
    LiquidityAgent,
    LiquidityAnalysis,
)


def _sample_context(rows: int = 30) -> dict:
    candles = []
    price = 100.0
    for _ in range(rows):
        candles.append(
            {
                "open": price,
                "high": price + 1.0,
                "low": price - 1.0,
                "close": price + 0.2,
            }
        )
        price += 0.1
    return {
        "symbol": "XAUUSD",
        "metadata": {"timeframe": "15m"},
        "candles": candles,
    }


class LiquidityAgentTests(unittest.TestCase):
    def test_confidence_capped_at_half(self):
        agent = LiquidityAgent()
        analysis = LiquidityAnalysis(
            equal_highs=(),
            equal_lows=(),
            buy_side_liquidity=101.0,
            sell_side_liquidity=99.0,
            liquidity_sweep="bullish",
            stop_hunt="bullish",
            current_price=100.0,
        )
        direction, confidence, _ = agent._evaluate_analysis(
            analysis=analysis,
            symbol="XAUUSD",
            timeframe="15m",
        )
        self.assertEqual(direction, Direction.LONG)
        self.assertLessEqual(confidence, MAX_LIQUIDITY_CONFIDENCE)

    def test_neutral_without_peer_confirmation(self):
        agent = LiquidityAgent()
        context = _sample_context()
        peer_results = {
            "smc": AgentResult(Direction.NEUTRAL, 0.0, "no structure"),
            "fvg": AgentResult(Direction.NEUTRAL, 0.0, "no fvg"),
            "order_block": AgentResult(Direction.NEUTRAL, 0.0, "no ob"),
        }
        analysis = LiquidityAnalysis(
            equal_highs=(),
            equal_lows=(),
            buy_side_liquidity=101.0,
            sell_side_liquidity=99.0,
            liquidity_sweep="bullish",
            stop_hunt=None,
            current_price=100.0,
        )
        direction, _, reason = agent._apply_confirmation(
            direction=Direction.LONG,
            confidence=0.45,
            reason="test",
            analysis=analysis,
            peer_results=peer_results,
            context=context,
        )
        self.assertEqual(direction, Direction.NEUTRAL)
        self.assertIn("no SMC/FVG/OB confirmation", reason)

    def test_sweep_requires_structure_or_zone(self):
        agent = LiquidityAgent()
        context = _sample_context()
        peer_results = {
            "smc": AgentResult(Direction.LONG, 0.40, "bullish trend only"),
            "fvg": AgentResult(Direction.NEUTRAL, 0.0, "no fvg"),
            "order_block": AgentResult(Direction.NEUTRAL, 0.0, "no ob"),
        }
        analysis = LiquidityAnalysis(
            equal_highs=(),
            equal_lows=(),
            buy_side_liquidity=101.0,
            sell_side_liquidity=99.0,
            liquidity_sweep="bullish",
            stop_hunt=None,
            current_price=100.0,
        )
        direction, _, reason = agent._apply_confirmation(
            direction=Direction.LONG,
            confidence=0.45,
            reason="test",
            analysis=analysis,
            peer_results=peer_results,
            context=context,
        )
        self.assertEqual(direction, Direction.NEUTRAL)
        self.assertIn("sweep without BOS/ChoCH or OB/FVG zone", reason)

    @patch("agents.liquidity_agent.price_in_active_entry_zone", return_value=True)
    def test_sweep_confirmed_by_entry_zone(self, _zone_mock):
        agent = LiquidityAgent()
        context = _sample_context()
        peer_results = {
            "smc": AgentResult(Direction.LONG, 0.40, "bullish trend only"),
            "fvg": AgentResult(Direction.NEUTRAL, 0.0, "no fvg"),
            "order_block": AgentResult(Direction.NEUTRAL, 0.0, "no ob"),
        }
        analysis = LiquidityAnalysis(
            equal_highs=(),
            equal_lows=(),
            buy_side_liquidity=101.0,
            sell_side_liquidity=99.0,
            liquidity_sweep="bullish",
            stop_hunt=None,
            current_price=100.0,
        )
        direction, confidence, _ = agent._apply_confirmation(
            direction=Direction.LONG,
            confidence=0.45,
            reason="test",
            analysis=analysis,
            peer_results=peer_results,
            context=context,
        )
        self.assertEqual(direction, Direction.LONG)
        self.assertLessEqual(confidence, MAX_LIQUIDITY_CONFIDENCE)


if __name__ == "__main__":
    unittest.main()
