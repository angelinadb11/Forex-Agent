"""Tests for MT5 bridge market data provider (main channel only)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

from config.settings import Settings
from data.market_data import MarketDataProvider, build_main_market_data_provider, build_scalp_market_data_provider
from data.providers.mt5_bridge_provider import Mt5BridgeProvider


class Mt5BridgeProviderTests(unittest.TestCase):
    def test_parses_candle_response(self) -> None:
        provider = Mt5BridgeProvider("http://127.0.0.1:8765", broker_symbol="XAUUSD")
        payload = {
            "ok": True,
            "candles": [
                {
                    "open_time": 1710000000000,
                    "open": 2340.1,
                    "high": 2341.0,
                    "low": 2339.5,
                    "close": 2340.8,
                    "volume": 120,
                }
            ],
        }

        with patch.object(provider, "_request", return_value=payload):
            candles = provider.get_market_data("XAUUSD", "15m", limit=1)

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0]["close"], 2340.8)

    def test_current_price_from_bridge(self) -> None:
        provider = Mt5BridgeProvider("http://127.0.0.1:8765", broker_symbol="XAUUSD.m")
        with patch.object(provider, "_request", return_value={"ok": True, "price": 3340.15}):
            price = provider.get_current_price("XAUUSD")
        self.assertAlmostEqual(price, 3340.15)
        self.assertEqual(provider.broker_symbol_for("XAUUSDT"), "XAUUSD.m")


class Mt5RoutingTests(unittest.TestCase):
    def test_main_provider_uses_mt5_when_enabled(self) -> None:
        settings = Settings(
            mt5_bridge_url="http://127.0.0.1:8765",
            mt5_symbol="XAUUSD",
        )
        with patch.dict(
            os.environ,
            {"MAIN_XAUUSD_USE_MT5": "1", "MAIN_XAUUSD_USE_OANDA": "0"},
            clear=False,
        ):
            main_provider = build_main_market_data_provider(settings)
        self.assertEqual(main_provider.data_source("XAUUSD"), "mt5")

    def test_scalp_provider_always_binance(self) -> None:
        scalp_provider = build_scalp_market_data_provider()
        self.assertEqual(scalp_provider.data_source("XAUUSD"), "binance")

    def test_main_mt5_does_not_change_scalp_provider(self) -> None:
        settings = Settings(mt5_bridge_url="http://127.0.0.1:8765")
        with patch.dict(os.environ, {"MAIN_XAUUSD_USE_MT5": "1"}, clear=False):
            main_provider = build_main_market_data_provider(settings)
        scalp_provider = build_scalp_market_data_provider()
        self.assertEqual(main_provider.data_source("XAUUSD"), "mt5")
        self.assertEqual(scalp_provider.data_source("XAUUSD"), "binance")

    @patch("data.providers.mt5_bridge_provider.requests.get")
    def test_main_provider_fetches_mt5_candles(self, mock_get: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "ok": True,
            "candles": [
                {
                    "open_time": 1710000000000,
                    "open": 1,
                    "high": 2,
                    "low": 0.5,
                    "close": 1.5,
                    "volume": 10,
                }
            ],
        }
        mock_get.return_value = mock_response

        provider = MarketDataProvider(
            symbol_provider_overrides={"XAUUSD": "mt5"},
            mt5_bridge_url="http://127.0.0.1:8765",
            mt5_symbol="XAUUSD",
        )
        candles = provider.get_market_data("XAUUSD", "15m", limit=1)
        self.assertEqual(candles[0]["close"], 1.5)
        called_url = mock_get.call_args.args[0]
        self.assertIn("/candles", called_url)


if __name__ == "__main__":
    unittest.main()
