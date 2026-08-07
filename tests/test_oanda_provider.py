"""Tests for OANDA market data provider."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from data.market_data import MarketDataProvider, build_main_market_data_provider
from data.providers.oanda_provider import OandaProvider
from config.settings import Settings


class OandaProviderTests(unittest.TestCase):
    def test_parses_candle_response(self) -> None:
        provider = OandaProvider("test-token", account_id="101-001-1", env="practice")
        payload = {
            "candles": [
                {
                    "complete": True,
                    "time": "2026-06-10T10:15:00.000000000Z",
                    "volume": 120,
                    "mid": {"o": "3340.1", "h": "3341.0", "l": "3339.5", "c": "3340.8"},
                }
            ]
        }

        with patch.object(provider, "_request", return_value=payload):
            candles = provider.get_market_data("XAUUSD", "15m", limit=1)

        self.assertEqual(len(candles), 1)
        self.assertEqual(candles[0]["open"], 3340.1)
        self.assertEqual(candles[0]["close"], 3340.8)
        self.assertGreater(candles[0]["open_time"], 0)

    def test_current_price_uses_pricing_when_account_configured(self) -> None:
        provider = OandaProvider("test-token", account_id="101-001-1", env="practice")
        pricing_payload = {
            "prices": [
                {
                    "bids": [{"price": "3340.00"}],
                    "asks": [{"price": "3340.20"}],
                }
            ]
        }

        with patch.object(provider, "_request", return_value=pricing_payload):
            price = provider.get_current_price("XAUUSD")

        self.assertAlmostEqual(price, 3340.10)

    def test_maps_xauusdt_to_xau_usd(self) -> None:
        provider = OandaProvider("test-token")
        self.assertEqual(provider.instrument_for("XAUUSDT"), "XAU_USD")


class IndexProviderGoldTests(unittest.TestCase):
    def test_xauusdt_resolves_to_yahoo_gold(self) -> None:
        from data.providers.index_provider import IndexProvider

        provider = IndexProvider()
        self.assertEqual(provider.normalize_symbol("XAUUSDT"), "XAUUSD")
        self.assertEqual(provider.index_name("XAUUSDT"), "Gold Spot / US Dollar")


class MarketDataProviderRoutingTests(unittest.TestCase):
    def test_main_provider_routes_xauusd_to_oanda(self) -> None:
        settings = Settings(oanda_api_key="test-token", oanda_account_id="101-001-1")
        provider = build_main_market_data_provider(settings)
        self.assertEqual(provider.data_source("XAUUSD"), "oanda")
        self.assertEqual(provider.data_source("BTCUSDT"), "binance")

    def test_main_provider_falls_back_to_yahoo_without_oanda(self) -> None:
        settings = Settings(oanda_api_key="")
        provider = build_main_market_data_provider(settings)
        self.assertEqual(provider.data_source("XAUUSD"), "yahoo_finance")
        self.assertEqual(provider.data_source("BTCUSDT"), "binance")

    def test_scalp_provider_keeps_binance_for_xauusd(self) -> None:
        provider = MarketDataProvider()
        self.assertEqual(provider.data_source("XAUUSD"), "binance")

    @patch("data.providers.oanda_provider.requests.request")
    def test_main_provider_fetches_oanda_candles(self, mock_request: MagicMock) -> None:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "candles": [
                {
                    "complete": True,
                    "time": "2026-06-10T10:15:00.000000000Z",
                    "volume": 10,
                    "mid": {"o": "1", "h": "2", "l": "0.5", "c": "1.5"},
                }
            ]
        }
        mock_request.return_value = mock_response

        provider = MarketDataProvider(
            symbol_provider_overrides={"XAUUSD": "oanda"},
            oanda_api_key="token",
        )
        candles = provider.get_market_data("XAUUSD", "15m", limit=1)
        self.assertEqual(candles[0]["close"], 1.5)
        called_url = mock_request.call_args.args[1]
        self.assertIn("/v3/instruments/XAU_USD/candles", called_url)


if __name__ == "__main__":
    unittest.main()
