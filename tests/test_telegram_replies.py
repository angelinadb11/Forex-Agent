import unittest

from agents.base import AgentResult, Direction
from strategy.trade_update import assess_trend_opposes_trade
from telegram.message_format import (
    format_sl_proximity_warning,
    format_stop_loss_reply,
    format_take_profit_reply,
    format_trend_change_warning,
    trade_move_pips,
    trade_result_dollars,
)
from tracking.trade_pnl import distance_to_sl_pips


class TelegramReplyFormatTests(unittest.TestCase):
    def test_trend_warning_format(self):
        message = format_trend_change_warning(
            open_time="2026-06-07T14:30:00+00:00",
            direction=Direction.LONG,
            current_price=2650.50,
        )
        self.assertIn("⚠️ УВАГА — Зміна тренду", message)
        self.assertIn("Різка зміна тренду на H1", message)
        self.assertIn("2650.50", message)
        self.assertIn("📌 Що робити зараз:", message)
        self.assertIn("Не додавай до позиції", message)

    def test_sl_proximity_warning_format(self):
        message = format_sl_proximity_warning(
            current_price=2650.50,
            remaining_pips=8.5,
        )
        self.assertIn("⚠️ SL близько", message)
        self.assertIn("2650.50", message)
        self.assertIn("8.5 pips", message)
        self.assertIn("Не пересувай SL далі", message)

    def test_stop_loss_reply_format(self):
        message = format_stop_loss_reply(result_dollars=1.0)
        self.assertIn("❌ СТОП-ЛОСС", message)
        self.assertIn("-$1.00", message)
        self.assertIn("📌 Що робити зараз:", message)
        self.assertIn("Чекай наступного сигналу системи", message)

    def test_tp1_reply_format(self):
        message = format_take_profit_reply(
            tp_level=1,
            open_time="2026-06-07T14:30:00+00:00",
            direction=Direction.LONG,
            entry=2650.00,
            tp_price=2651.00,
            move_pips=10.0,
            tp1=2651.00,
            tp2=2652.00,
            tp3=2653.00,
        )
        self.assertIn("✅ ТЕЙК-ПРОФІТ 1", message)
        self.assertIn("LONG", message)
        self.assertIn("+10.0 pips", message)
        self.assertIn("2650.00", message)
        self.assertIn("2652.00", message)
        self.assertIn("безризиковій угоді", message)

    def test_tp2_reply_format(self):
        message = format_take_profit_reply(
            tp_level=2,
            open_time="2026-06-07T14:30:00+00:00",
            direction=Direction.SHORT,
            entry=2650.00,
            tp_price=2648.00,
            move_pips=20.0,
            tp1=2649.00,
            tp2=2648.00,
            tp3=2647.00,
        )
        self.assertIn("✅ ТЕЙК-ПРОФІТ 2", message)
        self.assertIn("+20.0 pips", message)
        self.assertIn("2649.00", message)
        self.assertIn("2647.00", message)
        self.assertIn("50% позиції", message)

    def test_xauusd_pip_helpers(self):
        move = trade_move_pips(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry=2650.00,
            price=2651.00,
        )
        self.assertAlmostEqual(move, 10.0)
        self.assertAlmostEqual(
            trade_result_dollars(symbol="XAUUSD", pips=10.0, lot_size=0.01),
            1.0,
        )
        sl_distance = distance_to_sl_pips(
            symbol="XAUUSD",
            direction="long",
            current_price=2650.50,
            stop_loss=2649.50,
        )
        self.assertAlmostEqual(sl_distance, 10.0)


class TrendOpposesTradeTests(unittest.TestCase):
    def test_long_trade_blocked_by_bearish_trend(self):
        results = {
            "trend_filter": AgentResult(Direction.SHORT, 0.8, "bearish"),
        }
        self.assertTrue(assess_trend_opposes_trade(Direction.LONG, results))

    def test_short_trade_allowed_by_bearish_trend(self):
        results = {
            "trend_filter": AgentResult(Direction.SHORT, 0.8, "bearish"),
        }
        self.assertFalse(assess_trend_opposes_trade(Direction.SHORT, results))


if __name__ == "__main__":
    unittest.main()
