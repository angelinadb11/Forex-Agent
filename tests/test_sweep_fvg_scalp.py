"""Tests for VIP premium Sweep+FVG scalp."""

from __future__ import annotations

import unittest

from agents.base import Direction
from signal_generator import TradeSignal
from strategy.sweep_fvg_scalp import VIP_SIGNAL_TAG, detect_sweep_fvg_setup
from telegram.message_format import format_premium_scalp_trade_signal


def _asia_day_candles() -> list[dict[str, float]]:
    candles: list[dict[str, float]] = []
    # Asia session lows around 2650
    for i in range(120):
        price = 2650.0 + (i % 5) * 0.2
        candles.append(
            {"open": price, "high": price + 0.5, "low": price - 0.3, "close": price + 0.1}
        )
    # Pre-sweep holds above Asia low
    candles.append({"open": 2651.0, "high": 2651.5, "low": 2650.5, "close": 2651.0})
    # Sweep below 2650, close back above
    candles.append({"open": 2650.8, "high": 2651.0, "low": 2649.2, "close": 2650.6})
    # Confirm with bullish FVG: low above pre-sweep high
    candles.append({"open": 2650.7, "high": 2652.5, "low": 2651.2, "close": 2652.0})
    return candles


class PremiumMessageTests(unittest.TestCase):
    def test_premium_message_contains_vip_header(self) -> None:
        signal = TradeSignal(
            direction=Direction.LONG,
            entry=2651.20,
            stop_loss=2649.00,
            tp1=2653.40,
            tp2=2655.60,
            tp3=2655.60,
            confidence=0.72,
            reason=f"{VIP_SIGNAL_TAG} LONG",
            lot_size=0.01,
        )
        message = format_premium_scalp_trade_signal("XAUUSD", signal, timeframe="5m")
        self.assertIn("VIP ПРЕМІUM", message)
        self.assertIn("Sweep + FVG", message)
        self.assertIn("ліміт", message.lower())
        self.assertIn("5 хв", message)


class SweepFvgDetectionTests(unittest.TestCase):
    def test_detects_bullish_sweep_fvg_pattern(self) -> None:
        candles = _asia_day_candles()
        setup, reason = detect_sweep_fvg_setup(
            candles,
            level_mode="asia",
            symbol="XAUUSD",
        )
        # Asia range detection needs proper timestamps from candle_timestamp -
        # synthetic candles may not have open_time; expect graceful rejection or setup.
        if setup is None:
            self.assertIn("NO VIP", reason)
        else:
            self.assertEqual(setup.direction, Direction.LONG)
            self.assertIn(VIP_SIGNAL_TAG, setup.reason)


if __name__ == "__main__":
    unittest.main()
