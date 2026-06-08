import unittest

from agents.base import Direction
from runtime.dedup import SignalDedupGate
from signal_generator import TradeSignal


class SignalDedupGateTests(unittest.TestCase):
    def _signal(self, entry: float = 100.0, sl: float = 90.0) -> TradeSignal:
        return TradeSignal(Direction.LONG, entry, sl, 110.0, 120.0, 130.0, 0.80, "test")

    def test_blocks_when_symbol_already_open(self):
        gate = SignalDedupGate(signal_cooldown_minutes=0)
        decision = gate.can_publish("BTCUSDT", self._signal(), {"BTCUSDT"})
        self.assertFalse(decision.allowed)
        self.assertIn("open trade", decision.reason or "")

    def test_blocks_duplicate_fingerprint(self):
        gate = SignalDedupGate(signal_cooldown_minutes=0)
        signal = self._signal()
        gate.record_published("BTCUSDT", signal)
        decision = gate.can_publish("BTCUSDT", signal, set())
        self.assertFalse(decision.allowed)
        self.assertIn("duplicate setup", decision.reason or "")

    def test_allows_new_setup_after_fingerprint_changes(self):
        gate = SignalDedupGate(signal_cooldown_minutes=0)
        gate.record_published("BTCUSDT", self._signal(entry=100.0, sl=90.0))
        decision = gate.can_publish("BTCUSDT", self._signal(entry=101.0, sl=90.0), set())
        self.assertTrue(decision.allowed)


if __name__ == "__main__":
    unittest.main()
