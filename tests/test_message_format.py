import random
import unittest

from agents.base import AgentResult, Direction
from signal_generator import TradeSignal
from telegram.message_format import (
    BEARISH_ANALYSIS_PHRASES,
    BULLISH_ANALYSIS_PHRASES,
    format_trade_signal,
    select_analysis_phrases,
    summarize_analysis_sentences,
)


class MessageFormatTests(unittest.TestCase):
    def _long_signal(self) -> TradeSignal:
        return TradeSignal(Direction.LONG, 100.0, 90.0, 110.0, 120.0, 130.0, 0.78, "test")

    def test_trade_message_uses_minimal_human_format(self):
        random.seed(7)
        signal = self._long_signal()
        results = {
            "smc": AgentResult(
                Direction.LONG,
                0.55,
                "BTCUSDT 15m SMC: bullish structure (HH/HL), bullish BOS",
            ),
            "liquidity": AgentResult(
                Direction.LONG,
                0.45,
                "BTCUSDT 15m Liquidity: bullish liquidity sweep (SSL taken)",
            ),
        }

        message = format_trade_signal("BTCUSDT", signal, "15m", results)
        analysis_lines = message.splitlines()[11:]

        self.assertEqual(
            message.splitlines()[:10],
            [
                "BTCUSDT ЛОНГ",
                "",
                "Вхід: 100.00",
                "Стоп: 90.00",
                "",
                "✅ ТП1: 110.00",
                "✅ ТП2: 120.00",
                "✅ ТП3: 130.00",
                "",
                "ТФ: 15 хв",
            ],
        )
        self.assertGreaterEqual(len(analysis_lines), 1)
        self.assertLessEqual(len(analysis_lines), 2)
        self.assertNotIn("Confidence", message)
        self.assertNotIn("Analysis:", message)
        self.assertNotIn("•", message)
        self.assertIn("ТФ: 15 хв", message)
        self.assertTrue(all(line in BULLISH_ANALYSIS_PHRASES for line in analysis_lines))

    def test_analysis_phrases_are_randomized_from_pool(self):
        random.seed(1)
        long_phrases = {
            tuple(summarize_analysis_sentences({}, Direction.LONG))
            for _ in range(12)
        }
        self.assertGreater(len(long_phrases), 1)

        random.seed(2)
        short_phrases = select_analysis_phrases(Direction.SHORT, count=2)
        self.assertEqual(len(short_phrases), 2)
        self.assertTrue(all(phrase in BEARISH_ANALYSIS_PHRASES for phrase in short_phrases))
        self.assertFalse(any("101.2" in phrase for phrase in short_phrases))


if __name__ == "__main__":
    unittest.main()
