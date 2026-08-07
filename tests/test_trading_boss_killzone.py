"""Tests for Trading Boss Killzone strategy."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from agents.base import AgentResult, Direction
from strategy.trading_boss_killzone import (
    BiasAnalysis,
    RefLevel,
    SweepEvent,
    _premium_discount,
    active_killzone,
    compute_killzone_decision,
    detect_liquidity_sweep,
    evaluate_killzone_filter,
    is_killzone_session,
    run_killzone_agents,
)
from strategy.signal_filter import SignalFilter


def _ts(hour: int, minute: int = 0) -> datetime:
    return datetime(2026, 6, 10, hour, minute, tzinfo=timezone.utc)


class KillzoneSessionTests(unittest.TestCase):
    def test_london_killzone_utc(self) -> None:
        window = active_killzone(_ts(7, 0))
        self.assertIsNotNone(window)
        self.assertEqual(window.label, "London")

    def test_ny_killzone_utc(self) -> None:
        window = active_killzone(_ts(12, 0))
        self.assertIsNotNone(window)
        self.assertEqual(window.label, "NY")

    def test_outside_killzone(self) -> None:
        self.assertIsNone(active_killzone(_ts(10, 0)))
        ok, msg = is_killzone_session(_ts(10, 0))
        self.assertFalse(ok)
        self.assertIn("Killzone", msg)


class PremiumDiscountTests(unittest.TestCase):
    def test_discount_zone(self) -> None:
        self.assertEqual(_premium_discount(95.0, 120.0, 80.0), "discount")

    def test_premium_zone(self) -> None:
        self.assertEqual(_premium_discount(118.0, 120.0, 80.0), "premium")


class SweepDetectionTests(unittest.TestCase):
    def _build_long_sweep_m5(self) -> tuple[list[dict], list[datetime]]:
        from datetime import timedelta

        candles: list[dict] = []
        timestamps: list[datetime] = []
        base = _ts(6, 30)
        for i in range(100):
            price = 2005.0
            candles.append(
                {
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price,
                }
            )
            timestamps.append(base + timedelta(minutes=i * 5))

        level = 2000.0
        for idx in (30, 50):
            candles[idx] = {
                "open": 2005.0,
                "high": 2005.5,
                "low": level,
                "close": 2004.0,
            }

        sweep_idx = 80
        candles[sweep_idx - 1] = {
            "open": 2001.0,
            "high": 2002.0,
            "low": 2000.5,
            "close": 2001.0,
        }
        candles[sweep_idx] = {
            "open": 2000.8,
            "high": 2002.5,
            "low": 1998.5,
            "close": 2001.2,
        }
        candles[sweep_idx + 1] = {
            "open": 2001.0,
            "high": 2003.0,
            "low": 2000.8,
            "close": 2002.0,
        }
        return candles, timestamps

    def test_detects_bullish_sweep_with_m15_reclaim(self) -> None:
        candles_m5, timestamps_m5 = self._build_long_sweep_m5()
        m15_candles = [{"open": 2000, "high": 2003, "low": 1998, "close": 2001.5}] * 50
        m15_timestamps = [_ts(6, 0)] * 50
        bias = BiasAnalysis(Direction.LONG, "discount", 0.7, "test bias")
        atr = 2.0
        sweep = detect_liquidity_sweep(
            candles_m5,
            timestamps_m5,
            m15_candles=m15_candles,
            m15_timestamps=m15_timestamps,
            bias=bias,
            atr=atr,
            reclaim_bars=3,
        )
        self.assertIsNotNone(sweep)
        assert sweep is not None
        self.assertEqual(sweep.direction, Direction.LONG)


class ConfidenceScoringTests(unittest.TestCase):
    def test_alignment_bonus_in_killzone(self) -> None:
        bias = BiasAnalysis(Direction.LONG, "discount", 0.75, "bullish")
        sweep = SweepEvent(
            direction=Direction.LONG,
            level=RefLevel(2000.0, "low", "pool-low"),
            sweep_index=10,
            sweep_extreme=1999.0,
            reclaim_index=11,
            wick_depth=1.0,
            confidence=0.7,
            reason="test sweep",
        )
        from strategy.trading_boss_killzone import StructureSetup

        structure = StructureSetup(
            direction=Direction.LONG,
            entry=2001.0,
            zone_low=2000.5,
            zone_high=2001.5,
            zone_kind="OB",
            confidence=0.75,
            reason="test structure",
        )
        results = run_killzone_agents(
            bias=bias,
            sweep=sweep,
            structure=structure,
            timestamp=_ts(7, 0),
            setup=None,
        )
        direction, confidence = compute_killzone_decision(
            results,
            bias=bias,
            in_killzone=True,
        )
        self.assertEqual(direction, Direction.LONG)
        self.assertGreaterEqual(confidence, 0.55)

    def test_counter_bias_capped(self) -> None:
        bias = BiasAnalysis(Direction.SHORT, "premium", 0.8, "bearish")
        results = {
            "bias": AgentResult(Direction.SHORT, 0.8, "bearish"),
            "liquidity": AgentResult(Direction.LONG, 0.8, "sweep long"),
            "structure": AgentResult(Direction.LONG, 0.8, "structure long"),
            "session": AgentResult(Direction.LONG, 0.85, "in kz"),
            "execution": AgentResult(Direction.LONG, 0.8, "exec"),
        }
        direction, confidence = compute_killzone_decision(
            results,
            bias=bias,
            in_killzone=True,
        )
        self.assertEqual(direction, Direction.LONG)
        self.assertLessEqual(confidence, 0.15)


class KillzoneFilterTests(unittest.TestCase):
    def test_blocks_outside_killzone(self) -> None:
        signal_filter = SignalFilter(news_gate=None)
        results = run_killzone_agents(
            bias=BiasAnalysis(Direction.LONG, "discount", 0.7, "bias"),
            sweep=None,
            structure=None,
            timestamp=_ts(10, 0),
            setup=None,
        )
        outcome = evaluate_killzone_filter(
            signal_filter=signal_filter,
            results=results,
            direction=Direction.LONG,
            confidence=0.7,
            symbol="XAUUSD",
            timestamp=_ts(10, 0),
        )
        self.assertFalse(outcome.approved)
        self.assertIn("Killzone", outcome.message)


if __name__ == "__main__":
    unittest.main()
