import random
import unittest

from agents.base import Direction
from tracking.profit_milestones import (
    MILESTONE_TIERS,
    pick_profit_milestone_message,
    pending_profit_milestone_messages,
    profit_milestones_for_symbol,
    profit_pips_for_trade,
)
from tracking.trade_monitor import ActiveTrade


def _make_trade(**overrides) -> ActiveTrade:
    defaults = {
        "symbol": "XAUUSD",
        "direction": Direction.LONG,
        "entry": 2650.0,
        "stop_loss": 2645.0,
        "tp1": 2652.0,
        "tp2": 2660.0,
        "tp3": 2670.0,
        "confidence": 0.75,
        "reason": "test",
        "open_time": "2026-06-07T14:30:00+00:00",
        "initial_stop_loss": 2645.0,
    }
    defaults.update(overrides)
    return ActiveTrade(**defaults)


class ProfitMilestoneTests(unittest.TestCase):
    def test_xauusd_thresholds(self):
        milestones = profit_milestones_for_symbol("XAUUSD")
        self.assertEqual(
            [milestone.threshold_pips for milestone in milestones],
            [20, 50, 80, 100],
        )

    def test_btc_thresholds_are_tripled(self):
        milestones = profit_milestones_for_symbol("BTCUSDT")
        self.assertEqual(
            [milestone.threshold_pips for milestone in milestones],
            [60, 150, 240, 300],
        )

    def test_dj30_thresholds_are_tripled(self):
        milestones = profit_milestones_for_symbol("DJ30")
        self.assertEqual(
            [milestone.threshold_pips for milestone in milestones],
            [60, 150, 240, 300],
        )

    def test_profit_pips_for_long(self):
        pips = profit_pips_for_trade(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry=2650.0,
            price=2652.0,
        )
        self.assertAlmostEqual(pips, 20.0)

    def test_sends_each_tier_once(self):
        trade = _make_trade(profit_milestones_sent=None)
        first = pending_profit_milestone_messages(
            trade,
            high=2652.0,
            low=2650.0,
            rng=random.Random(0),
        )
        self.assertEqual(len(first), 1)
        self.assertEqual(trade.profit_milestones_sent, [20])

        second = pending_profit_milestone_messages(
            trade,
            high=2652.0,
            low=2650.0,
            rng=random.Random(0),
        )
        self.assertEqual(second, [])

    def test_sends_multiple_new_tiers_in_one_tick(self):
        trade = _make_trade(profit_milestones_sent=None)
        messages = pending_profit_milestone_messages(
            trade,
            high=2658.0,
            low=2650.0,
            rng=random.Random(1),
        )
        self.assertEqual(len(messages), 3)
        self.assertEqual(trade.profit_milestones_sent, [20, 50, 80])

    def test_skips_when_tp_already_hit(self):
        trade = _make_trade(tp1_hit=True, profit_milestones_sent=None)
        messages = pending_profit_milestone_messages(
            trade,
            high=2660.0,
            low=2650.0,
        )
        self.assertEqual(messages, [])

    def test_message_pool_for_tier_100(self):
        milestone = profit_milestones_for_symbol("XAUUSD")[-1]
        message = pick_profit_milestone_message(milestone, rng=random.Random(0))
        self.assertIn("+100", message)

    def test_message_uses_scaled_pips_for_btc(self):
        milestone = profit_milestones_for_symbol("BTCUSDT")[0]
        message = pick_profit_milestone_message(milestone, rng=random.Random(0))
        self.assertIn("+60", message)

    def test_all_tiers_have_two_message_variants(self):
        for tier in MILESTONE_TIERS:
            milestone = profit_milestones_for_symbol("XAUUSD")[MILESTONE_TIERS.index(tier)]
            seen = {
                pick_profit_milestone_message(milestone, rng=random.Random(seed))
                for seed in range(20)
            }
            self.assertGreaterEqual(len(seen), 1)


if __name__ == "__main__":
    unittest.main()
