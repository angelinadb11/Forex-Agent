import unittest
from unittest.mock import patch

from agents.base import AgentResult, Direction
from strategy.trend_breakeven_alert import (
    TrendBreakevenAlertChecker,
    assess_trend_breakeven_alert,
    should_check_trend_breakeven,
    sl_at_or_better_than_breakeven,
)
from tracking.trade_monitor import ActiveTrade


def _make_trade(**overrides) -> ActiveTrade:
    defaults = {
        "symbol": "XAUUSD",
        "direction": Direction.LONG,
        "entry": 2650.0,
        "stop_loss": 2645.0,
        "tp1": 2655.0,
        "tp2": 2660.0,
        "tp3": 2665.0,
        "confidence": 0.75,
        "reason": "test",
        "open_time": "2026-06-07T14:30:00+00:00",
        "initial_stop_loss": 2645.0,
        "timeframe": "15m",
        "entry_trend_direction": Direction.LONG,
    }
    defaults.update(overrides)
    return ActiveTrade(**defaults)


class TrendBreakevenAlertTests(unittest.TestCase):
    def test_sl_at_breakeven_for_long(self):
        trade = _make_trade(stop_loss=2650.0)
        self.assertTrue(sl_at_or_better_than_breakeven(trade))

    def test_sl_at_breakeven_for_short(self):
        trade = _make_trade(
            direction=Direction.SHORT,
            entry=2650.0,
            stop_loss=2650.0,
            initial_stop_loss=2655.0,
        )
        self.assertTrue(sl_at_or_better_than_breakeven(trade))

    def test_should_not_check_after_tp1(self):
        trade = _make_trade(tp1_hit=True)
        self.assertFalse(
            should_check_trend_breakeven(
                trade,
                candle_open_time=123.0,
                now_monotonic=1000.0,
            )
        )

    def test_should_not_check_when_warning_already_sent(self):
        trade = _make_trade(trend_warning_sent=True)
        self.assertFalse(
            should_check_trend_breakeven(
                trade,
                candle_open_time=123.0,
                now_monotonic=1000.0,
            )
        )

    def test_assess_detects_h1_flip(self):
        trade = _make_trade()
        self.assertTrue(
            assess_trend_breakeven_alert(
                trade,
                current_trend_direction=Direction.SHORT,
            )
        )
        self.assertFalse(
            assess_trend_breakeven_alert(
                trade,
                current_trend_direction=Direction.LONG,
            )
        )

    def test_checker_sends_alert_once_on_flip(self):
        trade = _make_trade()
        context = {
            "symbol": "XAUUSD",
            "candles": [{"open_time": 100.0, "close": 2649.0}],
        }
        checker = TrendBreakevenAlertChecker(lambda _symbol, _tf: context)

        with patch(
            "strategy.trend_breakeven_alert.run_agents",
            return_value={
                "trend_filter": AgentResult(Direction.SHORT, 0.8, "bearish H1"),
            },
        ):
            self.assertTrue(checker.analyze(trade, now_monotonic=1000.0))
            self.assertFalse(checker.analyze(trade, now_monotonic=1001.0))

    def test_checker_skips_when_sl_already_at_entry(self):
        trade = _make_trade(stop_loss=2650.0)
        context = {
            "symbol": "XAUUSD",
            "candles": [{"open_time": 100.0, "close": 2649.0}],
        }
        checker = TrendBreakevenAlertChecker(lambda _symbol, _tf: context)

        with patch(
            "strategy.trend_breakeven_alert.run_agents",
            return_value={
                "trend_filter": AgentResult(Direction.SHORT, 0.8, "bearish H1"),
            },
        ):
            self.assertFalse(checker.analyze(trade, now_monotonic=1000.0))


if __name__ == "__main__":
    unittest.main()
