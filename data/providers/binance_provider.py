from __future__ import annotations

from typing import Any

import requests

from data.providers.base import BaseDataProvider, Candle, DEFAULT_LIMIT

BINANCE_SPOT_API = "https://api.binance.com"
BINANCE_FUTURES_API = "https://fapi.binance.com"

SYMBOL_CONFIG: dict[str, dict[str, str]] = {
    "BTCUSDT": {
        "api_base": BINANCE_SPOT_API,
        "klines_path": "/api/v3/klines",
        "market": "spot",
    },
    "XAUUSDT": {
        "api_base": BINANCE_FUTURES_API,
        "klines_path": "/fapi/v1/klines",
        "market": "futures",
    },
}


class BinanceProvider(BaseDataProvider):
    """Loads crypto OHLC data from Binance public APIs."""

    @property
    def supported_symbols(self) -> tuple[str, ...]:
        return tuple(SYMBOL_CONFIG.keys())

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        symbol = self.normalize_symbol(symbol)
        timeframe = self.validate_timeframe(timeframe)
        self.validate_limit(limit)

        if symbol not in SYMBOL_CONFIG:
            supported = ", ".join(self.supported_symbols)
            raise ValueError(f"Unsupported Binance symbol '{symbol}'. Use one of: {supported}")

        raw_candles = self._fetch_klines(symbol, timeframe, limit)
        return [self._parse_candle(candle) for candle in raw_candles]

    def market_type(self, symbol: str) -> str:
        symbol = self.normalize_symbol(symbol)
        return SYMBOL_CONFIG[symbol]["market"]

    def get_historical_market_data(
        self,
        symbol: str,
        timeframe: str,
        total_candles: int = 1500,
    ) -> list[Candle]:
        """Fetch paginated historical candles for backtesting."""
        symbol = self.normalize_symbol(symbol)
        timeframe = self.validate_timeframe(timeframe)

        if symbol not in SYMBOL_CONFIG:
            supported = ", ".join(self.supported_symbols)
            raise ValueError(f"Unsupported Binance symbol '{symbol}'. Use one of: {supported}")
        if total_candles < 1:
            raise ValueError("total_candles must be at least 1")

        collected: list[Candle] = []
        end_time: int | None = None

        while len(collected) < total_candles:
            batch_size = min(1000, total_candles - len(collected))
            raw_batch = self._fetch_klines(
                symbol,
                timeframe,
                batch_size,
                end_time=end_time,
            )
            if not raw_batch:
                break

            parsed_batch = [self._parse_candle(candle) for candle in raw_batch]
            collected = parsed_batch + collected
            end_time = int(raw_batch[0][0]) - 1

            if len(raw_batch) < batch_size:
                break

        return collected[-total_candles:]

    def _fetch_klines(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        end_time: int | None = None,
    ) -> list[list[Any]]:
        config = SYMBOL_CONFIG[symbol]
        url = f"{config['api_base']}{config['klines_path']}"
        params: dict[str, Any] = {
            "symbol": symbol,
            "interval": timeframe,
            "limit": limit,
        }
        if end_time is not None:
            params["endTime"] = end_time

        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_candle(raw: list[Any]) -> Candle:
        return {
            "open_time": float(raw[0]),
            "open": float(raw[1]),
            "high": float(raw[2]),
            "low": float(raw[3]),
            "close": float(raw[4]),
        }

    def get_current_price(self, symbol: str) -> float:
        symbol = self.normalize_symbol(symbol)
        if symbol not in SYMBOL_CONFIG:
            supported = ", ".join(self.supported_symbols)
            raise ValueError(f"Unsupported Binance symbol '{symbol}'. Use one of: {supported}")

        config = SYMBOL_CONFIG[symbol]
        if config["market"] == "spot":
            url = f"{config['api_base']}/api/v3/ticker/price"
        else:
            url = f"{config['api_base']}/fapi/v1/ticker/price"

        response = requests.get(url, params={"symbol": symbol}, timeout=30)
        response.raise_for_status()
        return float(response.json()["price"])
