import unittest

from agents.base import Direction
from telegram.message_format import format_trade_result


class TradeResultFormatTests(unittest.TestCase):
    def test_tp1_format(self):
        message = format_trade_result("BTCUSDT", Direction.LONG, "tp1")
        self.assertEqual(
            message,
            "TP1:\n✅ TP1 HIT\n\nBTCUSDT LONG",
        )

    def test_tp2_format(self):
        message = format_trade_result("BTCUSDT", Direction.SHORT, "tp2")
        self.assertEqual(
            message,
            "TP2:\n✅✅ TP2 HIT\n\nBTCUSDT SHORT",
        )

    def test_tp3_format(self):
        message = format_trade_result("BTCUSDT", Direction.LONG, "tp3")
        self.assertEqual(
            message,
            "TP3:\n✅✅✅ TP3 HIT 🔥\n\nBTCUSDT LONG",
        )

    def test_stop_loss_format(self):
        message = format_trade_result("BTCUSDT", Direction.LONG, "stop_loss")
        self.assertEqual(
            message,
            "Stop loss:\n🔴 STOP LOSS HIT\n\nBTCUSDT LONG",
        )


if __name__ == "__main__":
    unittest.main()
