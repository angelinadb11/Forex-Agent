import unittest

import pandas as pd

from agents.base import AgentResult, Direction
from agents.fvg_agent import FVGAgent
from agents.order_block_agent import OrderBlockAgent
from agents.zone_helpers import ZoneCatalog, detect_fvgs, detect_order_blocks
from strategy.runner import AGENT_WEIGHTS, compute_final_decision


def _build_candles(rows: list[tuple[float, float, float, float]]) -> list[dict[str, float]]:
    candles = []
    for index, (open_price, high, low, close) in enumerate(rows):
        candles.append(
            {
                "open_time": float(index * 60_000),
                "open": open_price,
                "high": high,
                "low": low,
                "close": close,
            }
        )
    return candles


def _pad_candles(
    core: list[tuple[float, float, float, float]],
    *,
    count: int = 20,
    fill: tuple[float, float, float, float] = (105, 106, 104.5, 105.5),
) -> list[dict[str, float]]:
    rows = [fill] * (count - len(core)) + core
    return _build_candles(rows)


class FVGDetectionTests(unittest.TestCase):
    def test_detects_bullish_fvg(self) -> None:
        df = pd.DataFrame(
            [
                {"open": 100, "high": 101, "low": 99, "close": 100},
                {"open": 101, "high": 108, "low": 100.5, "close": 107},
                {"open": 112, "high": 114, "low": 112, "close": 113},
            ]
        )
        fvgs = detect_fvgs(df, pip_size=1.0)
        bullish = [fvg for fvg in fvgs if fvg.direction == "bullish"]
        self.assertEqual(len(bullish), 1)
        self.assertAlmostEqual(bullish[0].gap_low, 101.0)
        self.assertAlmostEqual(bullish[0].gap_high, 112.0)


class FVGAgentTests(unittest.TestCase):
    def test_scores_price_inside_active_fvg(self) -> None:
        core = [
            (100, 101, 99, 100),
            (101, 108, 100.5, 107),
            (112, 114, 112, 113),
            (105, 106, 104.5, 105.5),
        ]
        context = {
            "symbol": "XAUUSD",
            "candles": _pad_candles(core),
            "trend_direction": Direction.LONG,
            "metadata": {"timeframe": "15m"},
        }
        result = FVGAgent().analyze(context)
        self.assertEqual(result.direction, Direction.LONG)
        self.assertGreaterEqual(result.confidence, 0.55)


class OrderBlockAgentTests(unittest.TestCase):
    def test_detects_bullish_order_block(self) -> None:
        rows = [
            (100, 101, 99.5, 100.5),
            (100.4, 100.8, 99.8, 100.0),
            (100.1, 100.5, 99.9, 100.2),
            (100.2, 116.0, 100.0, 115.0),
            (115, 116, 114, 115.5),
        ]
        df = pd.DataFrame(
            [
                {
                    "open": row[0],
                    "high": row[1],
                    "low": row[2],
                    "close": row[3],
                }
                for row in rows
            ]
        )
        blocks = detect_order_blocks(df, pip_size=1.0, min_impulse_pips=15.0)
        self.assertTrue(any(block.direction == "bullish" for block in blocks))

    def test_scores_price_retesting_order_block(self) -> None:
        core = [
            (100, 101, 99.5, 100.5),
            (100.4, 100.8, 99.8, 100.0),
            (100.1, 100.5, 99.9, 100.2),
            (100.2, 116.0, 100.0, 115.0),
            (100.1, 100.7, 99.9, 100.3),
        ]
        context = {
            "symbol": "XAUUSD",
            "candles": _pad_candles(core, count=25),
            "trend_direction": Direction.LONG,
            "metadata": {"timeframe": "15m"},
        }
        result = OrderBlockAgent().analyze(context)
        self.assertEqual(result.direction, Direction.LONG)
        self.assertGreaterEqual(result.confidence, 0.55)


class ZoneCatalogTests(unittest.TestCase):
    def test_catalog_precomputes_retesting_order_blocks(self) -> None:
        candles = _pad_candles(
            [
                (100, 101, 99.5, 100.5),
                (100.4, 100.8, 99.8, 100.0),
                (100.1, 100.5, 99.9, 100.2),
                (100.2, 116.0, 100.0, 115.0),
                (100.1, 100.7, 99.9, 100.3),
            ],
            count=30,
        )
        catalog = ZoneCatalog.from_candles(candles, "XAUUSD")
        bar_index = len(candles) - 1
        retesting = catalog.obs_retesting_at(bar_index)
        self.assertTrue(retesting)
        result = OrderBlockAgent().analyze(
            {
                "symbol": "XAUUSD",
                "candles": candles,
                "bar_index": bar_index,
                "zone_catalog": catalog,
                "trend_direction": Direction.LONG,
                "metadata": {"timeframe": "15m"},
            }
        )
        self.assertEqual(result.direction, Direction.LONG)

    def test_catalog_matches_incremental_detection(self) -> None:
        candles = _pad_candles(
            [
                (100, 101, 99, 100),
                (101, 108, 100.5, 107),
                (112, 114, 112, 113),
                (105, 106, 104.5, 105.5),
            ],
            count=30,
        )
        catalog = ZoneCatalog.from_candles(candles, "XAUUSD")
        bar_index = len(candles) - 1
        df = pd.DataFrame(
            [
                {
                    "open": candle["open"],
                    "high": candle["high"],
                    "low": candle["low"],
                    "close": candle["close"],
                }
                for candle in candles
            ]
        )

        catalog_fvgs = catalog.fvgs_at(bar_index)
        direct_fvgs = detect_fvgs(df.iloc[: bar_index + 1], catalog.pip_size)
        self.assertEqual(len(catalog_fvgs), len(direct_fvgs))
        if catalog_fvgs:
            self.assertEqual(catalog_fvgs[-1].filled, direct_fvgs[-1].filled)

        catalog_blocks = catalog.order_blocks_at(bar_index)
        direct_blocks = detect_order_blocks(
            df.iloc[: bar_index + 1],
            catalog.pip_size,
            min_impulse_pips=15.0,
        )
        self.assertEqual(len(catalog_blocks), len(direct_blocks))


class WeightedDecisionTests(unittest.TestCase):
    def test_weighted_scores_use_chief_analyst_weights(self) -> None:
        results = {
            name: AgentResult(Direction.LONG, 1.0, name)
            for name in AGENT_WEIGHTS
        }
        results["trend_filter"] = AgentResult(Direction.LONG, 0.8, "trend")
        direction, confidence, long_score, short_score = compute_final_decision(results)
        self.assertEqual(direction, Direction.LONG)
        self.assertAlmostEqual(long_score, sum(AGENT_WEIGHTS.values()))
        self.assertEqual(short_score, 0.0)
        self.assertAlmostEqual(confidence, 1.0)

    def test_caps_confidence_when_core_agents_disagree(self) -> None:
        results = {
            "smc": AgentResult(Direction.LONG, 1.0, "smc"),
            "liquidity": AgentResult(Direction.LONG, 1.0, "liquidity"),
            "fvg": AgentResult(Direction.LONG, 1.0, "fvg"),
            "order_block": AgentResult(Direction.LONG, 1.0, "order_block"),
            "rsi": AgentResult(Direction.SHORT, 1.0, "rsi"),
            "trend_filter": AgentResult(Direction.LONG, 0.8, "trend"),
        }
        direction, confidence, long_score, _ = compute_final_decision(results)
        self.assertEqual(direction, Direction.LONG)
        self.assertAlmostEqual(long_score, 0.9)
        self.assertAlmostEqual(confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
