from __future__ import annotations

from typing import Any

import requests

from data.providers.base import BaseDataProvider, Candle, DEFAULT_LIMIT

SUPPORTED_MT5_SYMBOLS = ("XAUUSD", "XAUUSDT")


class Mt5BridgeProvider(BaseDataProvider):
    """Loads XAUUSD OHLC from a local MT5 bridge (Moneta Markets terminal on Windows)."""

    def __init__(
        self,
        bridge_url: str,
        *,
        broker_symbol: str = "XAUUSD",
        bridge_token: str = "",
        timeout: float = 30.0,
    ) -> None:
        base = bridge_url.strip().rstrip("/")
        if not base:
            raise ValueError("MT5_BRIDGE_URL is required for Mt5BridgeProvider")
        self.bridge_url = base
        self.broker_symbol = broker_symbol.strip() or "XAUUSD"
        self.bridge_token = bridge_token.strip()
        self.timeout = timeout

    @property
    def supported_symbols(self) -> tuple[str, ...]:
        return SUPPORTED_MT5_SYMBOLS

    def broker_symbol_for(self, symbol: str) -> str:
        symbol = self.normalize_symbol(symbol)
        if symbol in SUPPORTED_MT5_SYMBOLS:
            return self.broker_symbol
        return symbol

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        timeframe = self.validate_timeframe(timeframe)
        self.validate_limit(limit)
        broker_symbol = self.broker_symbol_for(symbol)
        payload = self._request(
            "/candles",
            params={
                "symbol": broker_symbol,
                "timeframe": timeframe,
                "limit": limit,
            },
        )
        candles = self._parse_candles(payload)
        if not candles:
            raise ValueError(f"MT5 bridge returned no candles for {broker_symbol} {timeframe}")
        return candles[-limit:]

    def get_current_price(self, symbol: str) -> float:
        broker_symbol = self.broker_symbol_for(symbol)
        payload = self._request("/price", params={"symbol": broker_symbol})
        price = payload.get("price")
        if price is None:
            raise ValueError(f"MT5 bridge price response missing 'price' for {broker_symbol}")
        return float(price)

    def get_historical_market_data(
        self,
        symbol: str,
        timeframe: str,
        total_candles: int = 1500,
    ) -> list[Candle]:
        timeframe = self.validate_timeframe(timeframe)
        if total_candles < 1:
            raise ValueError("total_candles must be at least 1")

        broker_symbol = self.broker_symbol_for(symbol)
        payload = self._request(
            "/candles",
            params={
                "symbol": broker_symbol,
                "timeframe": timeframe,
                "limit": min(total_candles, 1000),
                "historical": "1",
            },
        )
        candles = self._parse_candles(payload)
        return candles[-total_candles:]

    def _request(self, path: str, *, params: dict[str, Any] | None = None) -> dict[str, Any]:
        url = f"{self.bridge_url}{path}"
        headers: dict[str, str] = {}
        if self.bridge_token:
            headers["Authorization"] = f"Bearer {self.bridge_token}"

        response = requests.get(url, params=params, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected MT5 bridge response format")
        if payload.get("ok") is False:
            raise ValueError(str(payload.get("error", "MT5 bridge request failed")))
        return payload

    @staticmethod
    def _parse_candles(payload: dict[str, Any]) -> list[Candle]:
        raw_candles = payload.get("candles", [])
        if not isinstance(raw_candles, list):
            raise ValueError("MT5 bridge candles payload must be a list")

        parsed: list[Candle] = []
        for item in raw_candles:
            if not isinstance(item, dict):
                continue
            parsed.append(
                {
                    "open_time": float(item["open_time"]),
                    "open": float(item["open"]),
                    "high": float(item["high"]),
                    "low": float(item["low"]),
                    "close": float(item["close"]),
                    "volume": float(item.get("volume", 0)),
                }
            )
        return parsed
