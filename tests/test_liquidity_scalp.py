"""Tests for the liquidity-sweep scalp strategy."""

from __future__ import annotations

import unittest

from agents.base import Direction
from strategy.liquidity_scalp import (
    LiquidityScalpConfig,
    MAX_SL_PIPS,
    MIN_SL_PIPS,
    build_liquidity_scalp_signal,
    detect_liquidity_sweep_setup,
    h1_trend_direction,
    heiken_ashi_direction,
    stoch_rsi_k,
    volume_spike_ratio,
)

PIP = 0.1  # XAUUSD

# Detection-only config: production default enables volume + Stoch RSI filters.
BASE_CONFIG = LiquidityScalpConfig()


def _flat_candle(price: float = 2005.0) -> dict[str, float]:
    return {
        "open": price,
        "high": price + 0.5,
        "low": price - 0.5,
        "close": price,
    }


def _build_long_sweep_candles(
    *,
    pool_level: float = 2000.0,
    sweep_low: float = 1999.0,
    sweep_close: float = 2001.0,
    sweep_open: float = 2000.5,
) -> list[dict[str, float]]:
    """70 flat candles with two equal swing lows and a final sweep candle."""
    candles = [_flat_candle() for _ in range(70)]
    for dip_index in (20, 40):
        candles[dip_index] = {
            "open": 2005.0,
            "high": 2005.5,
            "low": pool_level,
            "close": 2004.0,
        }
    candles[-1] = {
        "open": sweep_open,
        "high": max(sweep_open, sweep_close) + 0.2,
        "low": sweep_low,
        "close": sweep_close,
    }
    return candles


def _build_short_sweep_candles(
    *,
    pool_level: float = 2010.0,
    sweep_high: float = 2011.0,
    sweep_close: float = 2009.0,
    sweep_open: float = 2009.5,
) -> list[dict[str, float]]:
    candles = [_flat_candle() for _ in range(70)]
    for spike_index in (20, 40):
        candles[spike_index] = {
            "open": 2005.0,
            "high": pool_level,
            "low": 2004.5,
            "close": 2006.0,
        }
    candles[-1] = {
        "open": sweep_open,
        "high": sweep_high,
        "low": min(sweep_open, sweep_close) - 0.2,
        "close": sweep_close,
    }
    return candles


class DetectLiquiditySweepTests(unittest.TestCase):
    def test_long_sweep_detected_with_small_stop(self) -> None:
        candles = _build_long_sweep_candles()
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=BASE_CONFIG)

        self.assertIsNotNone(setup, reason)
        assert setup is not None
        self.assertEqual(setup.direction, Direction.LONG)
        self.assertEqual(setup.entry, candles[-1]["close"])
        self.assertLess(setup.stop_loss, candles[-1]["low"])
        self.assertGreaterEqual(setup.sl_pips, MIN_SL_PIPS)
        self.assertLessEqual(setup.sl_pips, MAX_SL_PIPS)
        # TP1 must be exactly 1R.
        risk = setup.entry - setup.stop_loss
        self.assertAlmostEqual(setup.tp1, setup.entry + risk, places=6)

    def test_short_sweep_detected(self) -> None:
        candles = _build_short_sweep_candles()
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=BASE_CONFIG)

        self.assertIsNotNone(setup, reason)
        assert setup is not None
        self.assertEqual(setup.direction, Direction.SHORT)
        self.assertGreater(setup.stop_loss, candles[-1]["high"])
        risk = setup.stop_loss - setup.entry
        self.assertAlmostEqual(setup.tp1, setup.entry - risk, places=6)

    def test_no_setup_without_sweep(self) -> None:
        candles = [_flat_candle() for _ in range(70)]
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=BASE_CONFIG)
        self.assertIsNone(setup)
        self.assertIn("NO SCALP", reason)

    def test_sl_too_wide_rejected(self) -> None:
        # Deep wick far below the pool makes the stop exceed MAX_SL_PIPS.
        candles = _build_long_sweep_candles(sweep_low=1994.0, sweep_close=2001.5)
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=BASE_CONFIG)
        self.assertIsNone(setup)
        self.assertIn("exceeds max", reason)

    def test_min_sl_floor_applied(self) -> None:
        # Shallow wick: raw risk would be tiny, so stop widens to MIN_SL_PIPS.
        candles = _build_long_sweep_candles(sweep_low=1999.9, sweep_close=2000.3)
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=BASE_CONFIG)
        self.assertIsNotNone(setup, reason)
        assert setup is not None
        self.assertGreaterEqual(setup.sl_pips, MIN_SL_PIPS - 1e-9)

    def test_directional_close_filter_rejects_bearish_reclaim(self) -> None:
        candles = _build_long_sweep_candles(sweep_open=2002.0, sweep_close=2001.0)
        config = LiquidityScalpConfig(require_directional_close=True)
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=config)
        self.assertIsNone(setup)
        self.assertIn("weak reclaim", reason)

    def test_h1_trend_filter_blocks_counter_trend(self) -> None:
        candles = _build_long_sweep_candles()
        config = LiquidityScalpConfig(require_h1_trend=True)
        falling_h1 = [
            {"open": 2100.0 - i, "high": 2101.0 - i, "low": 2099.0 - i, "close": 2100.0 - i}
            for i in range(60)
        ]
        setup, reason = detect_liquidity_sweep_setup(
            candles, "XAUUSD", config=config, h1_candles=falling_h1
        )
        self.assertIsNone(setup)
        self.assertIn("H1 trend", reason)

    def test_min_pool_touches_filter(self) -> None:
        candles = _build_long_sweep_candles()
        config = LiquidityScalpConfig(min_pool_touches=3)
        setup, _ = detect_liquidity_sweep_setup(candles, "XAUUSD", config=config)
        self.assertIsNone(setup)


class SignalBuildTests(unittest.TestCase):
    def test_signal_levels_match_setup(self) -> None:
        candles = _build_long_sweep_candles()
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=BASE_CONFIG)
        self.assertIsNotNone(setup, reason)
        assert setup is not None

        signal = build_liquidity_scalp_signal(setup, "XAUUSD")
        self.assertEqual(signal.direction, Direction.LONG)
        self.assertEqual(signal.entry, setup.entry)
        self.assertEqual(signal.stop_loss, setup.stop_loss)
        self.assertEqual(signal.tp1, setup.tp1)
        self.assertEqual(signal.tp2, setup.tp2)
        self.assertEqual(signal.tp3, setup.tp2)
        self.assertGreaterEqual(signal.confidence, 0.6)
        self.assertLessEqual(signal.confidence, 0.75)


class ConfirmationFilterTests(unittest.TestCase):
    def test_heiken_ashi_direction_rising_series(self) -> None:
        rising = [
            {"open": 2000.0 + i, "high": 2001.5 + i, "low": 1999.5 + i, "close": 2001.0 + i}
            for i in range(30)
        ]
        self.assertEqual(heiken_ashi_direction(rising), Direction.LONG)

    def test_heiken_ashi_direction_falling_series(self) -> None:
        falling = [
            {"open": 2030.0 - i, "high": 2030.5 - i, "low": 2028.5 - i, "close": 2029.0 - i}
            for i in range(30)
        ]
        self.assertEqual(heiken_ashi_direction(falling), Direction.SHORT)

    def test_volume_spike_filter(self) -> None:
        candles = _build_long_sweep_candles()
        for candle in candles:
            candle["volume"] = 100.0
        candles[-1]["volume"] = 300.0

        config = LiquidityScalpConfig(require_volume_spike=True, min_volume_ratio=1.5)
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=config)
        self.assertIsNotNone(setup, reason)

        candles[-1]["volume"] = 100.0
        setup, reason = detect_liquidity_sweep_setup(candles, "XAUUSD", config=config)
        self.assertIsNone(setup)
        self.assertIn("volume", reason)

    def test_stoch_rsi_extremes(self) -> None:
        def zigzag(start: float, sign: float) -> list[dict[str, float]]:
            # Trending series with pullbacks so RSI is defined and varying.
            closes = [start]
            for i in range(100):
                step = 2.0 if i % 3 != 2 else -0.8
                closes.append(closes[-1] + sign * step)
            # Strong final push so Stoch RSI sits at the trend's extreme.
            for _ in range(15):
                closes.append(closes[-1] + sign * 2.0)
            return [
                {"open": c, "high": c + 0.5, "low": c - 0.5, "close": c}
                for c in closes
            ]

        rising = zigzag(2000.0, +1.0)
        falling = zigzag(2300.0, -1.0)
        k_rising = stoch_rsi_k(rising)
        k_falling = stoch_rsi_k(falling)
        self.assertIsNotNone(k_rising)
        self.assertIsNotNone(k_falling)
        assert k_rising is not None and k_falling is not None
        self.assertGreater(k_rising, 80.0)
        self.assertLess(k_falling, 20.0)

    def test_volume_ratio_requires_data(self) -> None:
        candles = _build_long_sweep_candles()
        self.assertIsNone(volume_spike_ratio(candles))


class H1TrendTests(unittest.TestCase):
    def test_uptrend_detected(self) -> None:
        rising = [
            {"open": 2000.0 + i, "high": 2001.0 + i, "low": 1999.0 + i, "close": 2000.0 + i}
            for i in range(60)
        ]
        self.assertEqual(h1_trend_direction(rising), Direction.LONG)

    def test_neutral_when_insufficient_data(self) -> None:
        self.assertEqual(h1_trend_direction(None), Direction.NEUTRAL)
        self.assertEqual(h1_trend_direction([_flat_candle()] * 10), Direction.NEUTRAL)


if __name__ == "__main__":
    unittest.main()
