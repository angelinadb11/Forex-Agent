import unittest
from datetime import datetime, timezone

from agents.base import AgentResult, Direction
from config.symbols import resolve_symbols
from strategy.signal_filter import MIN_CONFIDENCE, SignalFilter


def _passing_agents(direction: Direction = Direction.LONG) -> dict[str, AgentResult]:
    return {
        "smc": AgentResult(direction, 0.40, "smc"),
        "liquidity": AgentResult(direction, 0.35, "liquidity"),
        "rsi": AgentResult(direction, 0.10, "rsi"),
        "session": AgentResult(Direction.NEUTRAL, 0.30, "session"),
    }


class SignalFilterConfidenceTests(unittest.TestCase):
    def test_confidence_below_minimum_blocks_trade(self):
        filter_result = SignalFilter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.69,
            symbol="BTCUSDT",
        )
        self.assertFalse(filter_result.approved)
        self.assertIn("below minimum", filter_result.message)
        self.assertIn("0.69", filter_result.message)

    def test_confidence_at_minimum_allows_trade(self):
        filter_result = SignalFilter().evaluate(
            _passing_agents(),
            Direction.LONG,
            MIN_CONFIDENCE,
            symbol="BTCUSDT",
        )
        self.assertTrue(filter_result.approved)
        self.assertEqual(filter_result.message, "Signal approved")

    def test_confidence_above_minimum_allows_trade(self):
        filter_result = SignalFilter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.85,
            symbol="BTCUSDT",
        )
        self.assertTrue(filter_result.approved)


class SignalFilterSessionConfigTests(unittest.TestCase):
    def test_xauusd_not_blocked_off_hours_by_default(self):
        off_hours = datetime(2026, 6, 7, 23, 26, tzinfo=timezone.utc)
        filter_result = SignalFilter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.75,
            symbol="XAUUSD",
            timestamp=off_hours,
        )

        self.assertTrue(filter_result.approved)

    def test_london_ny_restriction_only_when_configured(self):
        off_hours = datetime(2026, 6, 7, 23, 26, tzinfo=timezone.utc)
        filter_result = SignalFilter(
            london_ny_session_symbols=frozenset({"XAUUSD"}),
        ).evaluate(
            _passing_agents(),
            Direction.LONG,
            0.75,
            symbol="XAUUSD",
            timestamp=off_hours,
        )

        self.assertFalse(filter_result.approved)
        self.assertIn("London or New York session", filter_result.message)


class SymbolConfigTests(unittest.TestCase):
    def test_resolve_symbols_deduplicates_aliases(self):
        symbols = resolve_symbols(["XAUUSD", "XAUUSDT", "BTCUSDT"])
        self.assertEqual(symbols, ("XAUUSD", "BTCUSDT"))


if __name__ == "__main__":
    unittest.main()
