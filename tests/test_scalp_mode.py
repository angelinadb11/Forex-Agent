import unittest
from datetime import datetime, timezone

from agents.base import AgentResult, Direction
from signal_generator import SignalGenerator, TradeSignal
from strategy.scalp_mode import (
    ScalpPublishGate,
    evaluate_scalp_alignment,
    evaluate_scalp_direction_alignment,
    is_scalp_enabled,
    passes_scalp_confidence,
)
from telegram.message_format import format_scalp_trade_signal


def _directional_agents(
    direction: Direction,
    *,
    high_confidence: bool = False,
) -> dict[str, AgentResult]:
    trend = direction if direction != Direction.NEUTRAL else Direction.LONG
    score = 0.85 if high_confidence else 0.40
    return {
        "smc": AgentResult(direction, score, "smc"),
        "liquidity": AgentResult(direction, score, "liquidity"),
        "fvg": AgentResult(direction, score, "fvg"),
        "order_block": AgentResult(direction, score if high_confidence else 0.0, "ob"),
        "rsi": AgentResult(direction, 0.10, "rsi"),
        "session": AgentResult(Direction.NEUTRAL, 0.30, "session"),
        "trend_filter": AgentResult(trend, 0.80, "trend"),
    }


class ScalpModeTests(unittest.TestCase):
    def test_scalp_enabled_only_for_xauusd(self):
        self.assertTrue(is_scalp_enabled("XAUUSD"))
        self.assertFalse(is_scalp_enabled("BTCUSDT"))
        self.assertFalse(is_scalp_enabled("DJ30"))

    def test_alignment_requires_matching_m15_direction(self):
        m5 = _directional_agents(Direction.LONG)
        m15 = _directional_agents(Direction.SHORT)
        m15["trend_filter"] = AgentResult(Direction.SHORT, 0.80, "trend")

        self.assertIsNone(evaluate_scalp_alignment(m5, m15))

    def test_alignment_passes_when_m5_m15_and_h1_match(self):
        m5 = _directional_agents(Direction.LONG, high_confidence=True)
        m15 = _directional_agents(Direction.LONG, high_confidence=True)

        alignment = evaluate_scalp_alignment(m5, m15)
        self.assertIsNotNone(alignment)
        direction, confidence = alignment
        self.assertEqual(direction, Direction.LONG)
        self.assertGreater(confidence, 0.0)

    def test_confidence_threshold_blocks_low_m5_score(self):
        m5 = _directional_agents(Direction.LONG)
        m5["fvg"] = AgentResult(Direction.NEUTRAL, 0.0, "fvg")
        m15 = _directional_agents(Direction.LONG)

        self.assertIsNotNone(evaluate_scalp_direction_alignment(m5, m15))
        self.assertIsNone(evaluate_scalp_alignment(m5, m15))
        self.assertFalse(passes_scalp_confidence(0.55))

    def test_publish_gate_enforces_hourly_gap_and_daily_cap(self):
        gate = ScalpPublishGate()
        first = datetime(2026, 6, 8, 10, 0, tzinfo=timezone.utc)
        second = datetime(2026, 6, 8, 10, 30, tzinfo=timezone.utc)

        allowed, _ = gate.can_publish("XAUUSD", first)
        self.assertTrue(allowed)
        gate.record("XAUUSD", first)

        blocked, reason = gate.can_publish("XAUUSD", second)
        self.assertFalse(blocked)
        self.assertIn("minimum interval", reason or "")

        later = datetime(2026, 6, 8, 11, 5, tzinfo=timezone.utc)
        allowed, _ = gate.can_publish("XAUUSD", later)
        self.assertTrue(allowed)

    def test_publish_gate_limits_four_signals_per_day(self):
        gate = ScalpPublishGate()
        base = datetime(2026, 6, 8, 8, 0, tzinfo=timezone.utc)
        for offset in (0, 2, 4, 6):
            moment = base.replace(hour=8 + offset)
            allowed, _ = gate.can_publish("XAUUSD", moment)
            self.assertTrue(allowed)
            gate.record("XAUUSD", moment)

        blocked, reason = gate.can_publish(
            "XAUUSD",
            datetime(2026, 6, 8, 20, 0, tzinfo=timezone.utc),
        )
        self.assertFalse(blocked)
        self.assertIn("daily limit", reason or "")

    def test_generate_scalp_caps_sl_at_twenty_pips(self):
        generator = SignalGenerator()
        candles = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0},
        ] * 24
        candles[-1] = {"open": 100.0, "high": 101.0, "low": 70.0, "close": 100.0}
        context = {"symbol": "XAUUSD", "candles": candles}

        result = generator.generate_scalp(context, Direction.LONG, 0.55)
        self.assertIsNotNone(result.signal)
        signal = result.signal
        assert signal is not None
        risk_pips = abs(signal.entry - signal.stop_loss) / 0.10
        self.assertAlmostEqual(risk_pips, 20.0, places=1)
        self.assertAlmostEqual(signal.tp1, signal.entry + (signal.entry - signal.stop_loss), places=4)
        self.assertAlmostEqual(
            signal.tp2,
            signal.entry + 2 * (signal.entry - signal.stop_loss),
            places=4,
        )
        self.assertEqual(signal.tp3, signal.tp2)

    def test_scalp_telegram_message_contains_label(self):
        signal = TradeSignal(
            direction=Direction.LONG,
            entry=2650.50,
            stop_loss=2648.50,
            tp1=2652.50,
            tp2=2654.50,
            tp3=2654.50,
            confidence=0.55,
            reason="Scalp setup",
        )
        message = format_scalp_trade_signal("XAUUSD", signal, _directional_agents(Direction.LONG))
        self.assertIn("⚡ СКАЛЬП", message)
        self.assertIn("XAUUSD КУПИТИ", message)
        self.assertIn("Вхід: 2650.50", message)
        self.assertIn("Стоп: 2648.50", message)
        self.assertIn("✅ ТП1: 2652.50 (1R)", message)
        self.assertIn("✅ ТП2: 2654.50 (2R)", message)
        self.assertIn("ТФ: 5 хв", message)
        self.assertNotIn("TP3", message)
        self.assertNotIn("Entry:", message)


if __name__ == "__main__":
    unittest.main()
