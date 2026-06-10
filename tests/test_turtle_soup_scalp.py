"""Tests for VIP 2 Turtle Soup scalp."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from agents.base import Direction
from signal_generator import TradeSignal
from strategy.turtle_soup_scalp import (
    VIP2_SIGNAL_TAG,
    detect_turtle_soup_setup,
    is_vip2_core_session,
)
from telegram.message_format import format_turtle_soup_scalp_trade_signal


class TurtleMessageTests(unittest.TestCase):
    def test_vip2_message_header(self) -> None:
        signal = TradeSignal(
            direction=Direction.LONG,
            entry=2650.60,
            stop_loss=2649.00,
            tp1=2652.90,
            tp2=2654.60,
            tp3=2656.00,
            confidence=0.68,
            reason=f"{VIP2_SIGNAL_TAG} LONG",
            lot_size=0.01,
        )
        message = format_turtle_soup_scalp_trade_signal("XAUUSD", signal, timeframe="5m")
        self.assertIn("VIP 2", message)
        self.assertIn("TURTLE SOUP", message)
        self.assertIn("2.5R", message)
        self.assertIn("5 хв", message)


class TurtleDetectionTests(unittest.TestCase):
    def test_graceful_rejection_without_timestamps(self) -> None:
        candles = [
            {"open": 2651.0, "high": 2651.5, "low": 2650.5, "close": 2651.0},
            {"open": 2650.8, "high": 2651.0, "low": 2649.2, "close": 2650.6},
        ]
        setup, reason = detect_turtle_soup_setup(candles, symbol="XAUUSD")
        self.assertIsNone(setup)
        self.assertIn("NO VIP2", reason)


class Vip2SessionTests(unittest.TestCase):
    def test_core_session_window(self) -> None:
        self.assertTrue(
            is_vip2_core_session(datetime(2026, 6, 10, 10, 0, tzinfo=timezone.utc))
        )
        self.assertFalse(
            is_vip2_core_session(datetime(2026, 6, 10, 3, 0, tzinfo=timezone.utc))
        )
        self.assertFalse(
            is_vip2_core_session(datetime(2026, 6, 10, 18, 0, tzinfo=timezone.utc))
        )


if __name__ == "__main__":
    unittest.main()
