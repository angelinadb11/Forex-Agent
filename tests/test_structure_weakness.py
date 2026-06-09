import random
import unittest
from unittest.mock import patch

from agents.base import AgentResult, Direction
from strategy.structure_weakness import (
    EntryZone,
    MAX_STRUCTURE_WARNINGS,
    STRUCTURE_WARNING_MESSAGES,
    assess_structure_weakness,
    bos_against_trade_on_m15,
    enrich_trade_entry_context,
    entry_zone_broken,
    h1_trend_flipped_against_trade,
    pick_structure_warning_message,
    resolve_entry_zone,
    rsi_sharp_reversal_against_trade,
    should_run_structure_check,
)
from tracking.trade_monitor import ActiveTrade


def _make_trade(**overrides) -> ActiveTrade:
    defaults = {
        "symbol": "XAUUSD",
        "direction": Direction.LONG,
        "entry": 2650.0,
        "stop_loss": 2645.0,
        "tp1": 2655.0,
        "tp2": 2660.0,
        "tp3": 2665.0,
        "confidence": 0.75,
        "reason": "test",
        "open_time": "2026-06-07T14:30:00+00:00",
        "initial_stop_loss": 2645.0,
        "entry_trend_direction": Direction.LONG,
        "entry_rsi": 58.0,
        "entry_zone_low": 2648.0,
        "entry_zone_high": 2652.0,
        "entry_zone_kind": "ob",
    }
    defaults.update(overrides)
    return ActiveTrade(**defaults)


def _m15_context(
    *,
    close: float = 2650.0,
    candles: list[dict[str, float]] | None = None,
) -> dict:
    if candles is None:
        candles = [
            {
                "open_time": 1_000_000.0,
                "open": close,
                "high": close + 1,
                "low": close - 1,
                "close": close,
            }
        ]
    return {"symbol": "XAUUSD", "candles": candles}


class StructureWeaknessTests(unittest.TestCase):
    def test_pick_warning_message_is_from_pool(self):
        rng = random.Random(0)
        message = pick_structure_warning_message(rng)
        self.assertIn(message, STRUCTURE_WARNING_MESSAGES)

    def test_bos_against_long_on_bearish_break(self):
        candles = [{"open": 100.0, "high": 101.0, "low": 99.0, "close": 98.0}] * 25
        context = {"candles": candles}
        analysis = type("Analysis", (), {"bos": "bearish"})()
        with patch("strategy.structure_weakness.analyze_smc", return_value=analysis):
            self.assertTrue(bos_against_trade_on_m15(context, Direction.LONG))
            self.assertFalse(bos_against_trade_on_m15(context, Direction.SHORT))

    def test_h1_trend_flip_requires_supported_entry(self):
        self.assertTrue(
            h1_trend_flipped_against_trade(
                Direction.LONG,
                Direction.LONG,
                Direction.SHORT,
            )
        )
        self.assertFalse(
            h1_trend_flipped_against_trade(
                Direction.LONG,
                Direction.SHORT,
                Direction.SHORT,
            )
        )

    def test_rsi_sharp_reversal_for_long(self):
        self.assertTrue(
            rsi_sharp_reversal_against_trade(
                Direction.LONG,
                entry_rsi=58.0,
                current_rsi=42.0,
                previous_rsi=50.0,
            )
        )
        self.assertFalse(
            rsi_sharp_reversal_against_trade(
                Direction.LONG,
                entry_rsi=58.0,
                current_rsi=52.0,
                previous_rsi=54.0,
            )
        )

    def test_entry_zone_break_for_long(self):
        self.assertTrue(
            entry_zone_broken(
                2647.0,
                zone_low=2648.0,
                zone_high=2652.0,
                trade_direction=Direction.LONG,
            )
        )
        self.assertFalse(
            entry_zone_broken(
                2649.0,
                zone_low=2648.0,
                zone_high=2652.0,
                trade_direction=Direction.LONG,
            )
        )

    def test_assess_warns_when_two_conditions_met(self):
        trade = _make_trade()
        context = _m15_context(close=2647.0)
        results = {
            "trend_filter": AgentResult(Direction.SHORT, 0.8, "bearish H1"),
        }
        assessment = assess_structure_weakness(
            trade,
            m15_context=context,
            current_results=results,
            current_rsi=40.0,
            rng=random.Random(0),
        )
        self.assertTrue(assessment.should_warn)
        self.assertGreaterEqual(len(assessment.met_conditions), 2)
        self.assertIn(assessment.message, STRUCTURE_WARNING_MESSAGES)

    def test_assess_no_warning_with_single_condition(self):
        trade = _make_trade(entry_zone_low=None, entry_zone_high=None)
        context = _m15_context(close=2650.0)
        results = {
            "trend_filter": AgentResult(Direction.LONG, 0.8, "bullish H1"),
        }
        assessment = assess_structure_weakness(
            trade,
            m15_context=context,
            current_results=results,
            current_rsi=40.0,
            rng=random.Random(0),
        )
        self.assertFalse(assessment.should_warn)
        self.assertEqual(assessment.met_conditions, ("rsi_reversal",))

    def test_should_run_once_per_candle(self):
        trade = _make_trade(last_structure_candle_open_time=100.0)
        self.assertFalse(
            should_run_structure_check(
                trade,
                candle_open_time=100.0,
                now_monotonic=0.0,
            )
        )
        self.assertTrue(
            should_run_structure_check(
                trade,
                candle_open_time=200.0,
                now_monotonic=0.0,
            )
        )

    def test_max_two_warnings_per_trade(self):
        trade = _make_trade(structure_warning_count=MAX_STRUCTURE_WARNINGS)
        assessment = assess_structure_weakness(
            trade,
            m15_context=_m15_context(close=2640.0),
            current_results={
                "trend_filter": AgentResult(Direction.SHORT, 0.8, "bearish H1"),
            },
            current_rsi=30.0,
        )
        self.assertFalse(assessment.should_warn)

    def test_resolve_entry_zone_prefers_order_block(self):
        context = {
            "symbol": "XAUUSD",
            "bar_index": 0,
            "candles": [
                {
                    "open": 2650.0,
                    "high": 2651.0,
                    "low": 2649.0,
                    "close": 2650.0,
                }
            ],
            "zone_catalog": type(
                "Catalog",
                (),
                {
                    "obs_retesting_at": lambda _self, _index: [
                        type(
                            "Block",
                            (),
                            {
                                "direction": "bullish",
                                "zone_low": 2648.0,
                                "zone_high": 2652.0,
                            },
                        )()
                    ],
                    "unfilled_fvgs_at": lambda _self, _index: [],
                },
            )(),
        }
        zone = resolve_entry_zone(context, Direction.LONG, 2650.0)
        self.assertEqual(zone, EntryZone(2648.0, 2652.0, "ob"))

    def test_enrich_trade_entry_context(self):
        trade = _make_trade(
            entry=50.0,
            entry_zone_low=None,
            entry_zone_high=None,
            entry_zone_kind=None,
            entry_rsi=None,
        )
        candles = [
            {
                "open": 49.5 + (index % 3),
                "high": 51.0 + (index % 3),
                "low": 48.5 + (index % 3),
                "close": 50.0 + (index % 5) * 0.4,
            }
            for index in range(20)
        ]
        context = {
            "symbol": "XAUUSD",
            "candles": candles,
            "zone_catalog": type(
                "Catalog",
                (),
                {
                    "obs_retesting_at": lambda _self, _index: [
                        type(
                            "Block",
                            (),
                            {
                                "direction": "bullish",
                                "zone_low": 49.5,
                                "zone_high": 50.5,
                            },
                        )()
                    ],
                    "unfilled_fvgs_at": lambda _self, _index: [],
                },
            )(),
            "bar_index": len(candles) - 1,
        }
        enrich_trade_entry_context(trade, context)
        self.assertEqual(trade.entry_zone_low, 49.5)
        self.assertEqual(trade.entry_zone_high, 50.5)
        self.assertEqual(trade.entry_zone_kind, "ob")
        self.assertIsNotNone(trade.entry_rsi)


if __name__ == "__main__":
    unittest.main()
