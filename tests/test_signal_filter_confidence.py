import unittest
from datetime import datetime, timezone

from agents.base import AgentResult, Direction
from config.symbols import resolve_symbols
from strategy.runner import compute_final_decision
from strategy.signal_filter import (
    FILTER_PROFILE_B,
    FILTER_PROFILE_C,
    MIN_CONFIDENCE,
    XAUUSD_MIN_CONFIDENCE,
    SignalFilter,
)


def _legacy_filter() -> SignalFilter:
    return SignalFilter(require_h4_h1_alignment=False, require_entry_zone=False)


def _passing_agents(direction: Direction = Direction.LONG) -> dict[str, AgentResult]:
    trend_direction = direction if direction != Direction.NEUTRAL else Direction.NEUTRAL
    return {
        "smc": AgentResult(direction, 0.40, "smc"),
        "liquidity": AgentResult(direction, 0.35, "liquidity"),
        "fvg": AgentResult(direction, 0.30, "fvg"),
        "order_block": AgentResult(Direction.NEUTRAL, 0.0, "ob"),
        "rsi": AgentResult(direction, 0.10, "rsi"),
        "session": AgentResult(Direction.NEUTRAL, 0.30, "session"),
        "trend_filter": AgentResult(trend_direction, 0.80, "trend"),
        "h4_trend_filter": AgentResult(trend_direction, 0.80, "h4 trend"),
    }


class SignalFilterConfidenceTests(unittest.TestCase):
    def test_confidence_below_minimum_blocks_trade(self):
        filter_result = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.59,
            symbol="BTCUSDT",
        )
        self.assertFalse(filter_result.approved)
        self.assertIn("below minimum", filter_result.message)
        self.assertIn("0.59", filter_result.message)

    def test_confidence_at_minimum_allows_trade(self):
        filter_result = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            MIN_CONFIDENCE,
            symbol="BTCUSDT",
        )
        self.assertTrue(filter_result.approved)
        self.assertEqual(filter_result.message, "Signal approved")

    def test_confidence_above_minimum_allows_trade(self):
        filter_result = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.85,
            symbol="BTCUSDT",
        )
        self.assertTrue(filter_result.approved)

    def test_xauusd_uses_lower_confidence_threshold(self):
        filter_result = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.52,
            symbol="XAUUSD",
        )
        self.assertTrue(filter_result.approved)

        blocked = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.49,
            symbol="XAUUSD",
        )
        self.assertFalse(blocked.approved)
        self.assertIn("below minimum", blocked.message)
        self.assertIn(f"{XAUUSD_MIN_CONFIDENCE:.2f}", blocked.message)

    def test_btcusdt_keeps_default_confidence_threshold(self):
        filter_result = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.59,
            symbol="BTCUSDT",
        )
        self.assertFalse(filter_result.approved)


class SignalFilterPrimaryAgreementTests(unittest.TestCase):
    def test_single_primary_agent_blocks_trade(self):
        agents = _passing_agents(Direction.LONG)
        agents["liquidity"] = AgentResult(Direction.SHORT, 0.35, "liquidity")
        agents["fvg"] = AgentResult(Direction.NEUTRAL, 0.0, "fvg")

        filter_result = _legacy_filter().evaluate(
            agents,
            Direction.LONG,
            0.85,
            symbol="XAUUSD",
        )

        self.assertFalse(filter_result.approved)
        self.assertIn("primary agents agree", filter_result.message)

    def test_two_primary_agents_with_trend_allows_trade(self):
        agents = _passing_agents(Direction.LONG)
        agents["fvg"] = AgentResult(Direction.NEUTRAL, 0.0, "fvg")

        filter_result = _legacy_filter().evaluate(
            agents,
            Direction.LONG,
            0.75,
            symbol="XAUUSD",
        )

        self.assertTrue(filter_result.approved)


class SignalFilterTrendTests(unittest.TestCase):
    def test_bullish_trend_blocks_short(self):
        agents = _passing_agents(Direction.SHORT)
        agents["trend_filter"] = AgentResult(Direction.LONG, 0.80, "bullish H1")
        agents["smc"] = AgentResult(Direction.SHORT, 0.40, "smc")
        agents["liquidity"] = AgentResult(Direction.SHORT, 0.35, "liquidity")
        agents["fvg"] = AgentResult(Direction.SHORT, 0.30, "fvg")

        filter_result = _legacy_filter().evaluate(
            agents,
            Direction.SHORT,
            0.85,
            symbol="XAUUSD",
        )

        self.assertFalse(filter_result.approved)
        self.assertIn("BULLISH", filter_result.message)
        self.assertIn("SHORT", filter_result.message)

    def test_bearish_trend_blocks_long(self):
        agents = _passing_agents(Direction.LONG)
        agents["trend_filter"] = AgentResult(Direction.SHORT, 0.80, "bearish H1")

        filter_result = _legacy_filter().evaluate(
            agents,
            Direction.LONG,
            0.85,
            symbol="XAUUSD",
        )

        self.assertFalse(filter_result.approved)
        self.assertIn("BEARISH", filter_result.message)
        self.assertIn("LONG", filter_result.message)

    def test_neutral_trend_does_not_confirm(self):
        agents = _passing_agents(Direction.LONG)
        agents["trend_filter"] = AgentResult(Direction.NEUTRAL, 0.0, "neutral H1")
        agents["h4_trend_filter"] = AgentResult(Direction.LONG, 0.80, "h4 long")

        filter_result = SignalFilter(require_entry_zone=False).evaluate(
            agents,
            Direction.LONG,
            0.85,
            symbol="XAUUSD",
        )

        self.assertFalse(filter_result.approved)
        self.assertIn("TrendFilter does not confirm", filter_result.message)

    def test_h4_h1_mismatch_blocks_trade(self):
        agents = _passing_agents(Direction.LONG)
        agents["h4_trend_filter"] = AgentResult(Direction.SHORT, 0.80, "h4 short")

        filter_result = SignalFilter(require_entry_zone=False).evaluate(
            agents,
            Direction.LONG,
            0.85,
            symbol="XAUUSD",
        )

        self.assertFalse(filter_result.approved)
        self.assertIn("H4/H1", filter_result.message)


class SignalFilterSessionConfigTests(unittest.TestCase):
    def test_xauusd_not_blocked_off_hours_by_default(self):
        off_hours = datetime(2026, 6, 7, 23, 26, tzinfo=timezone.utc)
        filter_result = _legacy_filter().evaluate(
            _passing_agents(),
            Direction.LONG,
            0.75,
            symbol="XAUUSD",
            timestamp=off_hours,
        )

        self.assertTrue(filter_result.approved)

    def test_london_ny_off_hours_adds_warning_without_blocking(self):
        off_hours = datetime(2026, 6, 7, 23, 26, tzinfo=timezone.utc)
        filter_result = SignalFilter(
            london_ny_session_symbols=frozenset({"XAUUSD"}),
            require_h4_h1_alignment=False,
            require_entry_zone=False,
        ).evaluate(
            _passing_agents(),
            Direction.LONG,
            0.75,
            symbol="XAUUSD",
            timestamp=off_hours,
        )

        self.assertTrue(filter_result.approved)
        self.assertEqual(filter_result.off_hours_warning, "⚠️ Off-hours")


class ConsensusDecisionTests(unittest.TestCase):
    def test_requires_two_primary_agents_and_trend(self):
        results = _passing_agents(Direction.LONG)
        results["fvg"] = AgentResult(Direction.NEUTRAL, 0.0, "fvg")
        direction, confidence, _, _ = compute_final_decision(results)
        self.assertEqual(direction, Direction.LONG)
        self.assertAlmostEqual(confidence, 0.20)

    def test_neutral_when_trend_does_not_confirm(self):
        results = _passing_agents(Direction.LONG)
        results["trend_filter"] = AgentResult(Direction.NEUTRAL, 0.0, "neutral")
        direction, confidence, _, _ = compute_final_decision(results)
        self.assertEqual(direction, Direction.NEUTRAL)
        self.assertEqual(confidence, 0.0)


class SymbolConfigTests(unittest.TestCase):
    def test_resolve_symbols_deduplicates_aliases(self):
        symbols = resolve_symbols(["XAUUSD", "XAUUSDT", "BTCUSDT"])
        self.assertEqual(symbols, ("XAUUSD", "BTCUSDT"))


class SmcConflictFilterTests(unittest.TestCase):
    def test_blocks_bullish_structure_with_bearish_choch(self):
        agents = _passing_agents(Direction.SHORT)
        agents["smc"] = AgentResult(
            Direction.SHORT,
            0.35,
            "XAUUSD 15m SMC: bullish structure (HH/HL), bearish ChoCH",
        )
        filter_result = SignalFilter(
            require_h4_h1_alignment=False,
            require_entry_zone=False,
            block_smc_structure_conflict=True,
        ).evaluate(agents, Direction.SHORT, 0.75, symbol="XAUUSD")

        self.assertFalse(filter_result.approved)
        self.assertIn("SMC structure conflict", filter_result.message)

    def test_allows_aligned_structure_without_choch_conflict(self):
        agents = _passing_agents(Direction.SHORT)
        agents["smc"] = AgentResult(
            Direction.SHORT,
            0.40,
            "XAUUSD 15m SMC: bearish structure (LH/LL), bearish BOS",
        )
        filter_result = SignalFilter(
            require_h4_h1_alignment=False,
            require_entry_zone=False,
            block_smc_structure_conflict=True,
        ).evaluate(agents, Direction.SHORT, 0.75, symbol="XAUUSD")

        self.assertTrue(filter_result.approved)


class SoftEntryZoneFilterTests(unittest.TestCase):
    def test_profile_b_enables_zone_cluster_and_rsi_gate(self):
        profile_filter = SignalFilter.from_profile(FILTER_PROFILE_B)
        self.assertTrue(profile_filter.use_zone_cluster)
        self.assertTrue(profile_filter.use_rsi_gate)
        self.assertFalse(profile_filter.h4_soft_mode)

    def test_profile_c_enables_h4_soft_mode(self):
        profile_filter = SignalFilter.from_profile(FILTER_PROFILE_C)
        self.assertTrue(profile_filter.h4_soft_mode)

    def test_profile_d_adds_smc_conflict_and_disables_btcusdt(self):
        from strategy.signal_filter import FILTER_PROFILE_D, profile_symbols
        from config.symbols import DEFAULT_SYMBOLS

        profile_filter = SignalFilter.from_profile(FILTER_PROFILE_D)
        self.assertTrue(profile_filter.use_zone_cluster)
        self.assertTrue(profile_filter.use_rsi_gate)
        self.assertTrue(profile_filter.block_smc_structure_conflict)
        self.assertEqual(FILTER_PROFILE_D.disabled_symbols, frozenset({"BTCUSDT"}))
        self.assertNotIn("BTCUSDT", profile_symbols(FILTER_PROFILE_D, DEFAULT_SYMBOLS))


class ZoneClusterTests(unittest.TestCase):
    def test_fvg_and_ob_count_as_single_zone_vote(self):
        from strategy.runner import resolve_zone_cluster, count_primary_agreement
        from strategy.decision_config import ZONE_RSI_DECISION_CONFIG

        results = _passing_agents(Direction.LONG)
        results["fvg"] = AgentResult(Direction.LONG, 0.30, "fvg")
        results["order_block"] = AgentResult(Direction.LONG, 0.90, "ob")
        zone = resolve_zone_cluster(results["fvg"], results["order_block"])
        self.assertEqual(zone.confidence, 0.90)
        self.assertEqual(
            count_primary_agreement(results, Direction.LONG, config=ZONE_RSI_DECISION_CONFIG),
            3,
        )


class RsiGateFilterTests(unittest.TestCase):
    def test_blocks_long_when_rsi_below_35(self):
        agents = _passing_agents(Direction.LONG)
        agents["rsi"] = AgentResult(
            Direction.NEUTRAL,
            0.0,
            "XAUUSD 15m RSI(14)=34.50 neutral",
        )
        filter_result = SignalFilter(
            require_h4_h1_alignment=False,
            require_entry_zone=False,
            use_rsi_gate=True,
        ).evaluate(agents, Direction.LONG, 0.75, symbol="XAUUSD")

        self.assertFalse(filter_result.approved)
        self.assertIn("RSI", filter_result.message)


if __name__ == "__main__":
    unittest.main()
