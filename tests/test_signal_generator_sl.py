import unittest

from agents.base import Direction
from config.sl_config import SYMBOL_SL_CONFIG, calculate_lot_size, get_sl_config
from signal_generator import (
    DEFAULT_DEPOSIT,
    SignalGenerator,
    planned_rr_to_target,
    price_distance_pips,
)


class SignalGeneratorSLTests(unittest.TestCase):
    def setUp(self) -> None:
        self.generator = SignalGenerator(deposit=DEFAULT_DEPOSIT)

    def test_calculate_lot_size_scales_with_deposit(self) -> None:
        self.assertEqual(calculate_lot_size(100.0), 0.01)
        self.assertEqual(calculate_lot_size(200.0), 0.02)

    def test_symbol_sl_config_loaded_for_all_instruments(self) -> None:
        self.assertIn("XAUUSD", SYMBOL_SL_CONFIG)
        self.assertIn("DJ30", SYMBOL_SL_CONFIG)
        self.assertIn("BTCUSDT", SYMBOL_SL_CONFIG)
        self.assertEqual(get_sl_config("XAUUSD").max_sl_pips, 150)
        self.assertEqual(get_sl_config("DJ30").max_sl_pips, 300)
        self.assertEqual(get_sl_config("BTCUSDT").max_sl_pips, 500)

    def test_validate_sl_accepts_xauusd_within_pip_range(self) -> None:
        config = get_sl_config("XAUUSD")
        assert config is not None
        entry = 2650.00
        stop_loss = entry - (50 * config.pip_size)

        result = self.generator.validate_sl(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry=entry,
            swing_price=stop_loss,
            stop_loss=stop_loss,
            confidence=0.80,
        )

        self.assertIsNotNone(result.signal)
        self.assertEqual(result.signal.lot_size, 0.02)
        self.assertAlmostEqual(
            planned_rr_to_target(
                result.signal.entry,
                result.signal.tp1,
                abs(result.signal.entry - result.signal.stop_loss),
            ),
            1.5,
        )

    def test_validate_sl_sets_standard_targets(self) -> None:
        config = get_sl_config("XAUUSD")
        assert config is not None
        entry = 2650.00
        risk = 50 * config.pip_size
        stop_loss = entry - risk

        result = self.generator.validate_sl(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry=entry,
            swing_price=stop_loss,
            stop_loss=stop_loss,
            confidence=0.80,
        )
        assert result.signal is not None
        self.assertAlmostEqual(result.signal.tp1, entry + 1.5 * risk)
        self.assertAlmostEqual(result.signal.tp2, entry + 2.5 * risk)
        self.assertAlmostEqual(result.signal.tp3, entry + 3.5 * risk)

    def test_validate_sl_rejects_sl_too_far_for_xauusd(self) -> None:
        config = get_sl_config("XAUUSD")
        assert config is not None
        entry = 2650.00
        stop_loss = entry - (160 * config.pip_size)

        result = self.generator.validate_sl(
            symbol="XAUUSD",
            direction=Direction.LONG,
            entry=entry,
            swing_price=stop_loss,
            stop_loss=stop_loss,
            confidence=0.80,
        )

        self.assertIsNone(result.signal)
        self.assertIn("exceeds maximum", result.rejection_reason or "")

    def test_validate_sl_rejects_sl_too_close_for_btcusdt(self) -> None:
        config = get_sl_config("BTCUSDT")
        assert config is not None
        entry = 65000.0
        stop_loss = entry - (50 * config.pip_size)

        result = self.generator.validate_sl(
            symbol="BTCUSDT",
            direction=Direction.LONG,
            entry=entry,
            swing_price=stop_loss,
            stop_loss=stop_loss,
            confidence=0.80,
        )

        self.assertIsNone(result.signal)
        self.assertIn("below minimum", result.rejection_reason or "")

    def test_validate_sl_accepts_dj30_within_range(self) -> None:
        config = get_sl_config("DJ30")
        assert config is not None
        entry = 42000.0
        stop_loss = entry - (120 * config.pip_size)

        result = self.generator.validate_sl(
            symbol="DJ30",
            direction=Direction.LONG,
            entry=entry,
            swing_price=stop_loss,
            stop_loss=stop_loss,
            confidence=0.80,
        )

        self.assertIsNotNone(result.signal)
        sl_pips = price_distance_pips(entry - stop_loss, config.pip_size)
        self.assertEqual(sl_pips, 120.0)


if __name__ == "__main__":
    unittest.main()
