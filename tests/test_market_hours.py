"""Tests for forex weekend market hours."""

from __future__ import annotations

import unittest
import unittest.mock
from datetime import datetime, timezone

from config.market_hours import is_forex_market_open, should_publish_forex_signal


def _utc(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


class ForexMarketHoursTests(unittest.TestCase):
    def test_saturday_closed(self) -> None:
        ok, msg = is_forex_market_open(_utc(2026, 9, 5, 12, 0))
        self.assertFalse(ok)
        self.assertIn("weekend", msg.lower())

    def test_sunday_before_open_closed(self) -> None:
        ok, _ = is_forex_market_open(_utc(2026, 9, 6, 15, 0))
        self.assertFalse(ok)

    def test_sunday_after_open(self) -> None:
        ok, _ = is_forex_market_open(_utc(2026, 9, 6, 23, 0))
        self.assertTrue(ok)

    def test_friday_after_close(self) -> None:
        ok, _ = is_forex_market_open(_utc(2026, 9, 4, 22, 30))
        self.assertFalse(ok)

    def test_friday_before_close_open(self) -> None:
        ok, _ = is_forex_market_open(_utc(2026, 9, 4, 21, 0))
        self.assertTrue(ok)

    def test_weekday_open(self) -> None:
        ok, _ = is_forex_market_open(_utc(2026, 9, 3, 10, 0))
        self.assertTrue(ok)

    def test_can_disable_block(self) -> None:
        with unittest.mock.patch.dict("os.environ", {"FOREX_BLOCK_WEEKENDS": "0"}):
            ok, msg = should_publish_forex_signal(_utc(2026, 9, 5, 12, 0))
        self.assertTrue(ok)
        self.assertIn("disabled", msg.lower())


if __name__ == "__main__":
    unittest.main()
