from __future__ import annotations

from typing import Any
from urllib.parse import quote

import requests

from data.providers.base import BaseDataProvider, Candle, DEFAULT_LIMIT

YAHOO_CHART_API = "https://query1.finance.yahoo.com/v8/finance/chart"

XAUUSD_YAHOO_TICKERS = ("GC=F", "MGC=F")

INDEX_SYMBOLS: dict[str, dict[str, str]] = {
    "DJ30": {
        "yahoo_ticker": "^DJI",
        "name": "Dow Jones Industrial Average",
    },
    "NAS100": {
        "yahoo_ticker": "^NDX",
        "name": "Nasdaq 100",
    },
    "EURUSD": {
        "yahoo_ticker": "EURUSD=X",
        "name": "Euro / US Dollar",
    },
    "GBPUSD": {
        "yahoo_ticker": "GBPUSD=X",
        "name": "British Pound / US Dollar",
    },
    "XAUUSD": {
        "yahoo_ticker": "GC=F",
        "name": "COMEX Gold Futures (GC=F)",
    },
}

TIMEFRAME_RANGE: dict[str, str] = {
    "1m": "7d",
    "5m": "1mo",
    "15m": "1mo",
    "1h": "6mo",
    "4h": "6mo",
}


class IndexProvider(BaseDataProvider):
    """Loads index OHLC data from Yahoo Finance public chart API."""

    @property
    def supported_symbols(self) -> tuple[str, ...]:
        return tuple(INDEX_SYMBOLS.keys())

    def normalize_symbol(self, symbol: str) -> str:
        symbol = super().normalize_symbol(symbol)
        if symbol == "XAUUSDT":
            return "XAUUSD"
        return symbol

    def _yahoo_tickers_for(self, symbol: str) -> tuple[str, ...]:
        symbol = self.normalize_symbol(symbol)
        if symbol == "XAUUSD":
            return XAUUSD_YAHOO_TICKERS
        return (INDEX_SYMBOLS[symbol]["yahoo_ticker"],)

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        symbol = self.normalize_symbol(symbol)
        timeframe = self.validate_timeframe(timeframe)
        self.validate_limit(limit)

        if symbol not in INDEX_SYMBOLS:
            supported = ", ".join(self.supported_symbols)
            raise ValueError(f"Unsupported index symbol '{symbol}'. Use one of: {supported}")

        range_param = self._range_for_limit(timeframe, limit)
        last_error: requests.HTTPError | None = None
        for ticker in self._yahoo_tickers_for(symbol):
            try:
                raw = self._fetch_chart(ticker, timeframe, range_param)
                candles = self._parse_chart_response(raw)
            except requests.HTTPError as exc:
                last_error = exc
                continue

            if len(candles) < limit:
                return candles
            return candles[-limit:]

        if last_error is not None:
            raise last_error
        raise ValueError(f"Yahoo Finance returned no candles for {symbol}")

    @staticmethod
    def _range_for_limit(timeframe: str, limit: int) -> str:
        if timeframe == "15m":
            if limit > 1500:
                return "60d"
            if limit > 500:
                return "60d"
        return TIMEFRAME_RANGE[timeframe]

    def index_name(self, symbol: str) -> str:
        symbol = self.normalize_symbol(symbol)
        return INDEX_SYMBOLS[symbol]["name"]

    def _fetch_chart(self, ticker: str, interval: str, range_param: str) -> dict[str, Any]:
        url = f"{YAHOO_CHART_API}/{quote(ticker, safe='')}"
        params = {
            "interval": interval,
            "range": range_param,
        }
        headers = {"User-Agent": "Mozilla/5.0"}

        response = requests.get(url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        payload = response.json()
        result = payload.get("chart", {}).get("result")
        if not result:
            raise ValueError(f"Yahoo Finance returned no chart data for {ticker}")
        return payload

    @staticmethod
    def _parse_chart_response(payload: dict[str, Any]) -> list[Candle]:
        result = payload["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]

        candles: list[Candle] = []
        for index in range(len(result["timestamp"])):
            open_price = quote["open"][index]
            high_price = quote["high"][index]
            low_price = quote["low"][index]
            close_price = quote["close"][index]

            if None in (open_price, high_price, low_price, close_price):
                continue

            candles.append(
                {
                    "open_time": float(result["timestamp"][index]) * 1000,
                    "open": float(open_price),
                    "high": float(high_price),
                    "low": float(low_price),
                    "close": float(close_price),
                }
            )

        return candles

    def get_current_price(self, symbol: str) -> float:
        symbol = self.normalize_symbol(symbol)
        if symbol not in INDEX_SYMBOLS:
            supported = ", ".join(self.supported_symbols)
            raise ValueError(f"Unsupported index symbol '{symbol}'. Use one of: {supported}")

        last_error: requests.HTTPError | None = None
        for ticker in self._yahoo_tickers_for(symbol):
            try:
                payload = self._fetch_chart(ticker, "1m", "1d")
                return float(payload["chart"]["result"][0]["meta"]["regularMarketPrice"])
            except requests.HTTPError as exc:
                last_error = exc
                continue

        if last_error is not None:
            raise last_error
        raise ValueError(f"Yahoo Finance returned no price for {symbol}")
