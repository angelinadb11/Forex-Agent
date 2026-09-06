"""Tests for TradingView webhook parsing and formatting."""

from __future__ import annotations

import unittest

from agents.base import Direction
from telegram.message_format import format_tradingview_alert
from webhook.tradingview import parse_tradingview_payload


class TradingViewParseTests(unittest.TestCase):
    def test_parses_json_short(self) -> None:
        body = """
        {
          "secret": "abc",
          "symbol": "XAUUSD",
          "action": "SELL",
          "entry": 4378.53,
          "sl": 4380.99,
          "tp1": 4374.84,
          "tp2": 4372.38,
          "timeframe": "5",
          "note": "TB sweep"
        }
        """
        alert = parse_tradingview_payload(body)
        self.assertEqual(alert.symbol, "XAUUSD")
        self.assertEqual(alert.direction, Direction.SHORT)
        self.assertAlmostEqual(alert.entry or 0, 4378.53)
        self.assertAlmostEqual(alert.stop_loss or 0, 4380.99)

    def test_parses_plain_text(self) -> None:
        alert = parse_tradingview_payload("XAUUSD SELL entry 4378 sl 4380")
        self.assertEqual(alert.direction, Direction.SHORT)
        self.assertAlmostEqual(alert.entry or 0, 4378.0)


class TradingViewFormatTests(unittest.TestCase):
    def test_formats_message(self) -> None:
        alert = parse_tradingview_payload(
            '{"symbol":"XAUUSD","action":"BUY","entry":4400,"sl":4390,"tp1":4410,"timeframe":"5m","note":"test"}'
        )
        message = format_tradingview_alert(alert)
        self.assertIn("TRADINGVIEW", message)
        self.assertIn("XAUUSD", message)
        self.assertIn("КУПИТИ", message)
        self.assertIn("4400.00", message)


if __name__ == "__main__":
    unittest.main()
