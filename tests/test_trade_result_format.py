import unittest

from agents.base import Direction
from telegram.message_format import format_trade_result


class TradeResultFormatTests(unittest.TestCase):
    def test_tp1_format(self):
        message = format_trade_result("BTCUSDT", Direction.LONG, "tp1")
        self.assertEqual(
            message,
            "ТП1:\n✅ ТП1 ДОСЯГНУТО\n\nBTCUSDT ЛОНГ",
        )

    def test_tp2_format(self):
        message = format_trade_result("BTCUSDT", Direction.SHORT, "tp2")
        self.assertEqual(
            message,
            "ТП2:\n✅✅ ТП2 ДОСЯГНУТО\n\nBTCUSDT ШОРТ",
        )

    def test_tp3_format(self):
        message = format_trade_result("BTCUSDT", Direction.LONG, "tp3")
        self.assertEqual(
            message,
            "ТП3:\n✅✅✅ ТП3 ДОСЯГНУТО 🔥\n\nBTCUSDT ЛОНГ",
        )

    def test_stop_loss_format(self):
        message = format_trade_result("BTCUSDT", Direction.LONG, "stop_loss")
        self.assertEqual(
            message,
            "Стоп:\n🔴 СТОП-ЛОСС\n\nBTCUSDT ЛОНГ",
        )

    def test_breakeven_reply_format(self):
        from telegram.message_format import format_breakeven_reply

        message = format_breakeven_reply(
            direction=Direction.LONG,
            entry=2650.00,
            exit_price=2650.00,
        )
        self.assertIn("⚪ Закрили в BE", message)
        self.assertIn("ЛОНГ 2650.00", message)
        self.assertIn("Ціна закриття: 2650.00", message)
        self.assertIn("0R 👌", message)


if __name__ == "__main__":
    unittest.main()
