import unittest

from agents.base import Direction
from signal_generator import TradeSignal, resolve_signal_direction
from signal_geometry import infer_direction_from_levels, validate_trade_levels
from telegram.message_format import format_trade_signal


class SignalGeometryTests(unittest.TestCase):
    def test_long_geometry_resolves_to_long(self):
        direction = infer_direction_from_levels(100.0, 90.0, 110.0, 130.0)
        self.assertEqual(direction, Direction.LONG)

    def test_short_geometry_resolves_to_short(self):
        direction = infer_direction_from_levels(100.0, 110.0, 90.0, 70.0)
        self.assertEqual(direction, Direction.SHORT)

    def test_trade_signal_rejects_mismatched_direction(self):
        with self.assertRaises(ValueError):
            TradeSignal(Direction.SHORT, 100.0, 90.0, 110.0, 120.0, 130.0, 0.75, "bad")

    def test_telegram_header_matches_long_geometry(self):
        signal = TradeSignal(Direction.LONG, 100.0, 90.0, 110.0, 120.0, 130.0, 0.75, "test")
        message = format_trade_signal("BTCUSDT", signal, "15m", None)
        self.assertTrue(message.startswith("BTCUSDT LONG"))
        self.assertEqual(resolve_signal_direction(signal), Direction.LONG)

    def test_telegram_header_matches_short_geometry(self):
        signal = TradeSignal(Direction.SHORT, 100.0, 110.0, 90.0, 80.0, 70.0, 0.75, "test")
        message = format_trade_signal("BTCUSDT", signal, "15m", None)
        self.assertTrue(message.startswith("BTCUSDT SHORT"))
        self.assertEqual(resolve_signal_direction(signal), Direction.SHORT)

    def test_validate_long_levels(self):
        validate_trade_levels(100.0, 90.0, 110.0, 120.0, 130.0, Direction.LONG)

    def test_validate_short_levels(self):
        validate_trade_levels(100.0, 110.0, 90.0, 80.0, 70.0, Direction.SHORT)


if __name__ == "__main__":
    unittest.main()
