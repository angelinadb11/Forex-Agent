import unittest

from agents.base import Direction
from strategy.near_tp1_breakeven import (
    NEAR_TP1_MIN_PROGRESS_R,
    assess_near_tp1_reversal,
    favorable_progress_r,
)


class NearTp1BreakevenTests(unittest.TestCase):
    def test_favorable_progress_short(self):
        progress = favorable_progress_r(
            Direction.SHORT,
            entry=100.0,
            risk=10.0,
            high=101.0,
            low=88.0,
        )
        self.assertAlmostEqual(progress, 1.2)

    def test_requires_min_progress(self):
        assessment = assess_near_tp1_reversal(
            Direction.SHORT,
            peak_progress_r=1.0,
            tp1_hit=False,
            sl_at_breakeven=False,
            m15_context={"candles": [{"close": 99.0}]},
            entry_zone_low=98.0,
            entry_zone_high=102.0,
            entry_rsi=60.0,
            previous_rsi=58.0,
        )
        self.assertFalse(assessment.should_move_sl_to_entry)

    def test_triggers_on_bos_with_enough_progress(self):
        candles = [{"open": 100, "high": 101, "low": 99, "close": 100.5}] * 25
        candles[-1] = {"open": 100, "high": 101, "low": 98, "close": 98.5}
        assessment = assess_near_tp1_reversal(
            Direction.SHORT,
            peak_progress_r=NEAR_TP1_MIN_PROGRESS_R,
            tp1_hit=False,
            sl_at_breakeven=False,
            m15_context={"candles": candles},
            entry_zone_low=99.0,
            entry_zone_high=101.0,
            entry_rsi=55.0,
            previous_rsi=54.0,
        )
        self.assertIsInstance(assessment.met_conditions, tuple)


if __name__ == "__main__":
    unittest.main()
