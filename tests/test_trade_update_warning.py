import unittest

from agents.base import AgentResult, Direction
from strategy.trade_update import (
    LEVEL2_STANDARD_CONFIRMATION_CYCLES,
    WarningLevel,
    assess_trade_update,
    evaluate_level2,
)
from telegram.message_format import format_high_risk_update, format_trade_update_warning


class TradeUpdateWarningTests(unittest.TestCase):
    def _entry_results(self) -> dict[str, AgentResult]:
        return {
            "smc": AgentResult(Direction.LONG, 0.75, "bullish structure"),
            "liquidity": AgentResult(Direction.LONG, 0.70, "bullish sweep"),
            "rsi": AgentResult(Direction.LONG, 0.60, "bullish momentum"),
            "session": AgentResult(Direction.NEUTRAL, 0.50, "London-New York overlap"),
        }

    def test_level1_message_format(self):
        message = format_trade_update_warning(
            "XAUUSD",
            Direction.LONG,
            ["Bullish momentum is weakening."],
        )
        self.assertEqual(
            message,
            "\n".join(
                [
                    "⚠️ TRADE UPDATE",
                    "",
                    "XAUUSD LONG",
                    "",
                    "Bullish momentum is weakening.",
                    "",
                    "Monitor position closely.",
                ]
            ),
        )

    def test_level2_message_format(self):
        message = format_high_risk_update(
            "XAUUSD",
            Direction.LONG,
            [
                "Market structure has flipped bearish.",
                "The original setup is no longer valid.",
            ],
        )
        self.assertIn("⚠️ HIGH RISK UPDATE", message)
        self.assertIn("Market structure has flipped bearish.", message)
        self.assertIn("The original setup is no longer valid.", message)
        self.assertIn("Consider closing the position manually.", message)

    def test_level1_on_soft_smc_weakening(self):
        current = self._entry_results()
        current["smc"] = AgentResult(Direction.NEUTRAL, 0.20, "no clear SMC confluence")

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertEqual(assessment.level, WarningLevel.LEVEL_1)
        self.assertFalse(assessment.level2_instant)
        self.assertFalse(assessment.level2_standard)
        self.assertEqual(assessment.reasons, ("Bullish structure is weakening.",))

    def test_level1_on_rsi_weakening(self):
        current = self._entry_results()
        current["rsi"] = AgentResult(Direction.NEUTRAL, 0.15, "neutral momentum")

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertEqual(assessment.level, WarningLevel.LEVEL_1)
        self.assertEqual(assessment.reasons, ("Bullish momentum is weakening.",))

    def test_dual_opposite_without_third_confirmation_is_standard_only(self):
        current = self._entry_results()
        current["smc"] = AgentResult(Direction.SHORT, 0.70, "bearish structure")
        current["liquidity"] = AgentResult(Direction.SHORT, 0.65, "bearish bias")
        current["rsi"] = AgentResult(Direction.LONG, 0.80, "still bullish momentum")

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertEqual(assessment.level, WarningLevel.NONE)
        self.assertFalse(assessment.level2_instant)
        self.assertTrue(assessment.level2_standard)
        self.assertIn("Market structure has flipped bearish.", assessment.level2_reasons)

    def test_dual_opposite_with_rsi_opposite_is_instant(self):
        current = self._entry_results()
        current["smc"] = AgentResult(Direction.SHORT, 0.70, "bearish structure")
        current["liquidity"] = AgentResult(Direction.SHORT, 0.65, "bearish bias")
        current["rsi"] = AgentResult(Direction.SHORT, 0.55, "bearish momentum")

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertTrue(assessment.level2_instant)
        self.assertFalse(assessment.level2_standard)

    def test_single_smc_opposite_never_triggers_level2(self):
        current = self._entry_results()
        current["smc"] = AgentResult(Direction.SHORT, 0.70, "bearish structure")
        current["liquidity"] = AgentResult(Direction.LONG, 0.50, "mixed liquidity")
        current["rsi"] = AgentResult(Direction.LONG, 0.30, "weak momentum")

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertEqual(assessment.level, WarningLevel.NONE)
        self.assertFalse(assessment.level2_instant)
        self.assertFalse(assessment.level2_standard)

    def test_dual_opposite_with_opposite_sweep_is_instant(self):
        current = self._entry_results()
        current["smc"] = AgentResult(Direction.SHORT, 0.70, "bearish structure")
        current["liquidity"] = AgentResult(
            Direction.SHORT,
            0.55,
            "BTCUSDT 15m Liquidity: bearish liquidity sweep (BSL taken)",
        )
        current["rsi"] = AgentResult(Direction.LONG, 0.40, "still bullish momentum")

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertTrue(assessment.level2_instant)
        self.assertFalse(assessment.level2_standard)

    def test_dual_opposite_with_severe_confidence_is_instant(self):
        current = self._entry_results()
        current["smc"] = AgentResult(Direction.SHORT, 0.70, "bearish structure")
        current["liquidity"] = AgentResult(Direction.SHORT, 0.65, "bearish bias")
        current["rsi"] = AgentResult(Direction.NEUTRAL, 0.05, "neutral")
        current["session"] = AgentResult(Direction.NEUTRAL, 0.0, "off-hours")

        evaluation = evaluate_level2(
            Direction.LONG,
            0.85,
            current,
        )

        self.assertTrue(evaluation.instant)

    def test_standard_confirmation_requires_two_cycles(self):
        self.assertEqual(LEVEL2_STANDARD_CONFIRMATION_CYCLES, 2)

    def test_no_warning_when_context_still_supports_trade(self):
        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            self._entry_results(),
        )

        self.assertEqual(assessment.level, WarningLevel.NONE)
        self.assertEqual(assessment.reasons, ())

    def test_session_end_does_not_trigger_warning(self):
        current = self._entry_results()
        current["session"] = AgentResult(
            Direction.NEUTRAL,
            0.0,
            "Outside London and New York sessions",
        )

        assessment = assess_trade_update(
            Direction.LONG,
            0.85,
            self._entry_results(),
            current,
        )

        self.assertEqual(assessment.level, WarningLevel.NONE)


if __name__ == "__main__":
    unittest.main()
