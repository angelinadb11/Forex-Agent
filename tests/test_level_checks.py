import unittest

from agents.base import Direction
from signal_generator import TradeSignal
from tracking.level_checks import stop_loss_hit, take_profit_hit
from tracking.trade_monitor import ActiveTrade, TradeMonitor


class LevelChecksTests(unittest.TestCase):
    def test_long_sl_uses_candle_low(self) -> None:
        self.assertFalse(
            stop_loss_hit(
                direction=Direction.LONG,
                high=101.0,
                low=91.0,
                stop_loss=90.0,
            )
        )
        self.assertTrue(
            stop_loss_hit(
                direction=Direction.LONG,
                high=101.0,
                low=89.5,
                stop_loss=90.0,
            )
        )

    def test_short_sl_uses_candle_high(self) -> None:
        self.assertFalse(
            stop_loss_hit(
                direction=Direction.SHORT,
                high=109.0,
                low=99.0,
                stop_loss=110.0,
            )
        )
        self.assertTrue(
            stop_loss_hit(
                direction=Direction.SHORT,
                high=110.5,
                low=99.0,
                stop_loss=110.0,
            )
        )

    def test_long_tp_uses_candle_high(self) -> None:
        self.assertFalse(
            take_profit_hit(
                direction=Direction.LONG,
                high=109.0,
                low=100.0,
                tp_price=110.0,
            )
        )
        self.assertTrue(
            take_profit_hit(
                direction=Direction.LONG,
                high=110.5,
                low=100.0,
                tp_price=110.0,
            )
        )


class TradeMonitorCandleTests(unittest.TestCase):
    def test_long_does_not_stop_out_on_close_only(self) -> None:
        signal = TradeSignal(
            Direction.LONG,
            entry=100.0,
            stop_loss=90.0,
            tp1=110.0,
            tp2=120.0,
            tp3=130.0,
            confidence=0.8,
            reason="test",
        )
        trade = ActiveTrade.from_signal("XAUUSD", signal, timeframe="15m")
        monitor = TradeMonitor(price_fetcher=lambda _symbol: 89.0)

        monitor._evaluate_candle(trade, high=101.0, low=91.0)

        self.assertFalse(trade.closed)
        self.assertIsNone(trade.result)

    def test_long_stops_out_when_candle_low_hits_sl(self) -> None:
        signal = TradeSignal(
            Direction.LONG,
            entry=100.0,
            stop_loss=90.0,
            tp1=110.0,
            tp2=120.0,
            tp3=130.0,
            confidence=0.8,
            reason="test",
        )
        trade = ActiveTrade.from_signal("XAUUSD", signal, timeframe="15m")
        monitor = TradeMonitor(price_fetcher=lambda _symbol: 95.0)

        monitor._evaluate_candle(trade, high=96.0, low=89.0)

        self.assertTrue(trade.closed)
        self.assertEqual(trade.result, "stop_loss")


if __name__ == "__main__":
    unittest.main()
