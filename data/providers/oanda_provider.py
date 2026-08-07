from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from data.providers.base import BaseDataProvider, Candle, DEFAULT_LIMIT

OANDA_PRACTICE_API = "https://api-fxpractice.oanda.com"
OANDA_LIVE_API = "https://api-fxtrade.oanda.com"

INSTRUMENT_MAP: dict[str, str] = {
    "XAUUSD": "XAU_USD",
    "XAUUSDT": "XAU_USD",
}

GRANULARITY_MAP: dict[str, str] = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "1h": "H1",
    "4h": "H4",
}


class OandaProvider(BaseDataProvider):
    """Loads XAUUSD OHLC and pricing from OANDA v20 REST API."""

    def __init__(
        self,
        api_key: str,
        *,
        account_id: str = "",
        env: str = "practice",
        timeout: float = 30.0,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OANDA_API_KEY is required for OandaProvider")
        self.api_key = api_key.strip()
        self.account_id = account_id.strip()
        self.env = env.strip().lower() or "practice"
        self.timeout = timeout
        self.api_base = OANDA_LIVE_API if self.env == "live" else OANDA_PRACTICE_API

    @property
    def supported_symbols(self) -> tuple[str, ...]:
        return tuple(INSTRUMENT_MAP.keys())

    def instrument_for(self, symbol: str) -> str:
        symbol = self.normalize_symbol(symbol)
        instrument = INSTRUMENT_MAP.get(symbol)
        if instrument is None:
            supported = ", ".join(self.supported_symbols)
            raise ValueError(f"Unsupported OANDA symbol '{symbol}'. Use one of: {supported}")
        return instrument

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        instrument = self.instrument_for(symbol)
        timeframe = self.validate_timeframe(timeframe)
        self.validate_limit(limit)
        granularity = GRANULARITY_MAP[timeframe]

        url = f"{self.api_base}/v3/instruments/{instrument}/candles"
        params = {
            "granularity": granularity,
            "count": limit,
            "price": "M",
        }
        payload = self._request("GET", url, params=params)
        candles = payload.get("candles", [])
        parsed = [self._parse_candle(item) for item in candles if item.get("complete", True)]
        if not parsed:
            raise ValueError(f"OANDA returned no candles for {instrument} {timeframe}")
        return parsed[-limit:]

    def get_current_price(self, symbol: str) -> float:
        instrument = self.instrument_for(symbol)
        if self.account_id:
            url = f"{self.api_base}/v3/accounts/{self.account_id}/pricing"
            payload = self._request("GET", url, params={"instruments": instrument})
            prices = payload.get("prices", [])
            if prices:
                return self._mid_from_price_entry(prices[0])

        candles = self.get_market_data(symbol, "1m", limit=1)
        return float(candles[-1]["close"])

    def get_historical_market_data(
        self,
        symbol: str,
        timeframe: str,
        total_candles: int = 1500,
    ) -> list[Candle]:
        instrument = self.instrument_for(symbol)
        timeframe = self.validate_timeframe(timeframe)
        if total_candles < 1:
            raise ValueError("total_candles must be at least 1")

        granularity = GRANULARITY_MAP[timeframe]
        collected: list[Candle] = []
        to_time: str | None = None

        while len(collected) < total_candles:
            batch_size = min(500, total_candles - len(collected))
            url = f"{self.api_base}/v3/instruments/{instrument}/candles"
            params: dict[str, Any] = {
                "granularity": granularity,
                "count": batch_size,
                "price": "M",
            }
            if to_time is not None:
                params["to"] = to_time

            payload = self._request("GET", url, params=params)
            raw_batch = payload.get("candles", [])
            if not raw_batch:
                break

            parsed_batch = [
                self._parse_candle(item)
                for item in raw_batch
                if item.get("complete", True)
            ]
            if not parsed_batch:
                break

            collected = parsed_batch + collected
            oldest_time = raw_batch[0]["time"]
            to_time = oldest_time

            if len(raw_batch) < batch_size:
                break

        return collected[-total_candles:]

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = requests.request(
            method,
            url,
            headers=self._headers(),
            params=params,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("Unexpected OANDA response format")
        return payload

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _parse_candle(raw: dict[str, Any]) -> Candle:
        price = raw.get("mid") or raw.get("bid") or raw.get("ask")
        if price is None:
            raise ValueError("OANDA candle missing price data")

        time_text = str(raw["time"])
        open_time_ms = OandaProvider._parse_oanda_time(time_text)
        return {
            "open_time": float(open_time_ms),
            "open": float(price["o"]),
            "high": float(price["h"]),
            "low": float(price["l"]),
            "close": float(price["c"]),
            "volume": float(raw.get("volume", 0)),
        }

    @staticmethod
    def _parse_oanda_time(time_text: str) -> int:
        normalized = time_text.replace("Z", "+00:00")
        if "." in normalized:
            base, rest = normalized.split(".", 1)
            tz_sep = "+" if "+" in rest else "-"
            if tz_sep in rest:
                fraction, tz = rest.split(tz_sep, 1)
                tz = tz_sep + tz
            else:
                fraction, tz = rest, "+00:00"
            fraction = (fraction + "000000")[:6]
            normalized = f"{base}.{fraction}{tz}"
        dt = datetime.fromisoformat(normalized)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    @staticmethod
    def _mid_from_price_entry(entry: dict[str, Any]) -> float:
        bids = entry.get("bids") or []
        asks = entry.get("asks") or []
        if bids and asks:
            bid = float(bids[0]["price"])
            ask = float(asks[0]["price"])
            return (bid + ask) / 2
        closeout_bid = entry.get("closeoutBid")
        closeout_ask = entry.get("closeoutAsk")
        if closeout_bid is not None and closeout_ask is not None:
            return (float(closeout_bid) + float(closeout_ask)) / 2
        raise ValueError("OANDA pricing entry missing bid/ask")
