import unittest
from datetime import datetime, timezone

from agents.base import AgentResult, Direction
from news.calendar_provider import CachedEconomicCalendar
from news.event_matcher import classify_tracked_event
from news.models import EconomicEvent, NewsAction, SymbolNewsPolicy
from news.news_gate import NEWS_WARNING_MESSAGE, NewsGate
from strategy.signal_filter import SignalFilter


class FakeCalendarProvider:
    def __init__(self, events: list[EconomicEvent]) -> None:
        self.events = events

    def fetch_events(self, start, end):
        del start, end
        return self.events


class EventMatcherTests(unittest.TestCase):
    def test_classifies_core_macro_events(self):
        self.assertEqual(classify_tracked_event("Core CPI m/m", "US"), "Core CPI")
        self.assertEqual(classify_tracked_event("Non-Farm Payrolls", "USD"), "NFP")
        self.assertEqual(classify_tracked_event("FOMC Statement", "US"), "FOMC")
        self.assertEqual(classify_tracked_event("Fed Interest Rate Decision", "US"), "Interest Rate Decision")
        self.assertEqual(classify_tracked_event("Core PCE Price Index", "US"), "PCE")
        self.assertEqual(classify_tracked_event("Powell Speaks", "US"), "Powell Speech")

    def test_ignores_non_us_events(self):
        self.assertIsNone(classify_tracked_event("CPI y/y", "EU"))


class NewsGateTests(unittest.TestCase):
    def _gate(self, events: list[EconomicEvent]) -> NewsGate:
        calendar = CachedEconomicCalendar([FakeCalendarProvider(events)], cache_ttl_seconds=3600)
        return NewsGate(
            calendar,
            symbol_policies={
                "XAUUSD": SymbolNewsPolicy(action=NewsAction.BLOCK, buffer_minutes=15),
                "BTCUSDT": SymbolNewsPolicy(action=NewsAction.BLOCK, buffer_minutes=15),
                "DJ30": SymbolNewsPolicy(action=NewsAction.WARN, buffer_minutes=15),
            },
        )

    def test_blocks_xauusd_during_news_window(self):
        event_time = datetime(2026, 6, 7, 13, 30, tzinfo=timezone.utc)
        gate = self._gate(
            [
                EconomicEvent(
                    name="Core CPI m/m",
                    label="Core CPI",
                    country="US",
                    event_time=event_time,
                )
            ]
        )
        result = gate.evaluate("XAUUSD", event_time)
        self.assertEqual(result.action, NewsAction.BLOCK)
        self.assertIn("Core CPI", result.message or "")

    def test_warns_dj30_without_blocking(self):
        event_time = datetime(2026, 6, 7, 13, 30, tzinfo=timezone.utc)
        gate = self._gate(
            [
                EconomicEvent(
                    name="Non-Farm Payrolls",
                    label="NFP",
                    country="US",
                    event_time=event_time,
                )
            ]
        )
        result = gate.evaluate("DJ30", event_time)
        self.assertEqual(result.action, NewsAction.WARN)
        self.assertEqual(result.message, NEWS_WARNING_MESSAGE)

    def test_allows_signals_outside_window(self):
        event_time = datetime(2026, 6, 7, 13, 30, tzinfo=timezone.utc)
        gate = self._gate(
            [
                EconomicEvent(
                    name="Core CPI m/m",
                    label="Core CPI",
                    country="US",
                    event_time=event_time,
                )
            ]
        )
        outside = datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc)
        result = gate.evaluate("XAUUSD", outside)
        self.assertEqual(result.action, NewsAction.NONE)


class SignalFilterNewsTests(unittest.TestCase):
    def test_news_block_integrates_with_filter(self):
        event_time = datetime(2026, 6, 7, 13, 30, tzinfo=timezone.utc)
        calendar = CachedEconomicCalendar(
            [
                FakeCalendarProvider(
                    [
                        EconomicEvent(
                            name="FOMC Statement",
                            label="FOMC",
                            country="US",
                            event_time=event_time,
                        )
                    ]
                )
            ]
        )
        gate = NewsGate(
            calendar,
            symbol_policies={"BTCUSDT": SymbolNewsPolicy(action=NewsAction.BLOCK, buffer_minutes=15)},
        )
        agents = {
            "smc": AgentResult(Direction.LONG, 0.40, "smc"),
            "liquidity": AgentResult(Direction.LONG, 0.35, "liquidity"),
            "rsi": AgentResult(Direction.LONG, 0.10, "rsi"),
            "session": AgentResult(Direction.NEUTRAL, 0.30, "session"),
            "trend_filter": AgentResult(Direction.LONG, 0.80, "bullish H1"),
        }
        result = SignalFilter(news_gate=gate).evaluate(
            agents,
            Direction.LONG,
            0.85,
            symbol="BTCUSDT",
            timestamp=event_time,
        )
        self.assertFalse(result.approved)
        self.assertIn("high-impact news window active", result.message)


    def test_dj30_warn_does_not_block_or_reduce_confidence(self):
        event_time = datetime(2026, 6, 7, 13, 30, tzinfo=timezone.utc)
        calendar = CachedEconomicCalendar(
            [
                FakeCalendarProvider(
                    [
                        EconomicEvent(
                            name="Non-Farm Payrolls",
                            label="NFP",
                            country="US",
                            event_time=event_time,
                        )
                    ]
                )
            ]
        )
        gate = NewsGate(
            calendar,
            symbol_policies={"DJ30": SymbolNewsPolicy(action=NewsAction.WARN, buffer_minutes=15)},
        )
        agents = {
            "smc": AgentResult(Direction.LONG, 0.40, "smc"),
            "liquidity": AgentResult(Direction.LONG, 0.35, "liquidity"),
            "rsi": AgentResult(Direction.LONG, 0.10, "rsi"),
            "session": AgentResult(Direction.NEUTRAL, 0.30, "session"),
            "trend_filter": AgentResult(Direction.LONG, 0.80, "bullish H1"),
        }
        result = SignalFilter(news_gate=gate).evaluate(
            agents,
            Direction.LONG,
            0.85,
            symbol="DJ30",
            timestamp=event_time,
        )
        self.assertTrue(result.approved)
        self.assertEqual(result.confidence, 0.85)
        self.assertEqual(result.message, "Signal approved")
        self.assertEqual(result.news_warning, NEWS_WARNING_MESSAGE)


if __name__ == "__main__":
    unittest.main()
