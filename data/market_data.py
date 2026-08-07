from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from config.symbols import SUPPORTED_SYMBOLS, SUPPORTED_TIMEFRAMES, SymbolDefinition, resolve_symbol, resolve_timeframe
from config.settings import Settings
from data.providers.base import (
    DEFAULT_LIMIT,
    BaseDataProvider,
    Candle,
)
from data.providers.binance_provider import BinanceProvider
from data.providers.index_provider import IndexProvider
from data.providers.oanda_provider import OandaProvider


@dataclass(frozen=True)
class MarketSnapshot:
    """Point-in-time market state passed to agents."""

    symbol: str
    timestamp: datetime
    candles: list[Candle]
    metadata: dict[str, Any]


class MarketDataProvider:
    """Routes symbol requests to the correct market data backend."""

    def __init__(
        self,
        *,
        symbol_provider_overrides: dict[str, str] | None = None,
        oanda_api_key: str = "",
        oanda_account_id: str = "",
        oanda_env: str = "practice",
    ) -> None:
        self._symbol_provider_overrides: dict[str, str] = {}
        for symbol, provider_name in (symbol_provider_overrides or {}).items():
            display = resolve_symbol(symbol).display
            self._symbol_provider_overrides[display] = provider_name

        self._providers: dict[str, BaseDataProvider] = {
            "binance": BinanceProvider(),
            "index": IndexProvider(),
        }
        if oanda_api_key.strip():
            self._providers["oanda"] = OandaProvider(
                api_key=oanda_api_key,
                account_id=oanda_account_id,
                env=oanda_env,
            )

    def resolve(self, symbol: str):
        return resolve_symbol(symbol)

    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        symbol_def, provider = self._resolve_provider(symbol)
        return provider.get_market_data(symbol_def.data_symbol, timeframe, limit)

    def fetch_snapshot(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> MarketSnapshot:
        symbol_def, provider = self._resolve_provider(symbol)
        candles = provider.get_market_data(symbol_def.data_symbol, timeframe, limit)

        metadata: dict[str, Any] = {
            "timeframe": timeframe,
            "candle_count": len(candles),
            "display_symbol": symbol_def.display,
            "data_symbol": symbol_def.data_symbol,
        }

        if isinstance(provider, BinanceProvider):
            metadata["source"] = "binance"
            metadata["market"] = provider.market_type(symbol_def.data_symbol)
        elif isinstance(provider, OandaProvider):
            metadata["source"] = "oanda"
            metadata["market"] = "cfd"
            metadata["instrument"] = provider.instrument_for(symbol_def.data_symbol)
        elif isinstance(provider, IndexProvider):
            metadata["source"] = "yahoo_finance"
            metadata["market"] = "index"
            metadata["index_name"] = provider.index_name(symbol_def.data_symbol)

        return MarketSnapshot(
            symbol=symbol_def.display,
            timestamp=datetime.now(timezone.utc),
            candles=candles,
            metadata=metadata,
        )

    def to_context(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
        *,
        include_h1_trend: bool = True,
        h1_limit: int = 250,
        include_h4_trend: bool = True,
        h4_limit: int = 250,
    ) -> dict[str, Any]:
        snapshot = self.fetch_snapshot(symbol, timeframe, limit)
        context: dict[str, Any] = {
            "symbol": snapshot.symbol,
            "timestamp": snapshot.timestamp,
            "candles": snapshot.candles,
            "metadata": snapshot.metadata,
        }
        if include_h1_trend:
            context["h1_candles"] = self.get_market_data(symbol, "1h", h1_limit)
        if include_h4_trend:
            context["h4_candles"] = self.get_market_data(symbol, "4h", h4_limit)
        return context

    def data_source(self, symbol: str) -> str:
        _, provider = self._resolve_provider(symbol)
        if isinstance(provider, BinanceProvider):
            return "binance"
        if isinstance(provider, OandaProvider):
            return "oanda"
        return "yahoo_finance"

    def get_current_price(self, symbol: str) -> float:
        symbol_def, provider = self._resolve_provider(symbol)
        return provider.get_current_price(symbol_def.data_symbol)

    def display_symbol(self, symbol: str) -> str:
        return resolve_symbol(symbol).display

    def get_historical_market_data(
        self,
        symbol: str,
        timeframe: str,
        total_candles: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        """Load historical candles for backtesting across all supported symbols."""
        symbol_def, provider = self._resolve_provider(symbol)
        resolve_timeframe(timeframe)

        if isinstance(provider, BinanceProvider):
            return provider.get_historical_market_data(
                symbol_def.data_symbol,
                timeframe,
                total_candles,
            )
        if isinstance(provider, OandaProvider):
            return provider.get_historical_market_data(
                symbol_def.data_symbol,
                timeframe,
                total_candles,
            )

        return provider.get_market_data(
            symbol_def.data_symbol,
            timeframe,
            total_candles,
        )

    def _resolve_provider(self, symbol: str) -> tuple[SymbolDefinition, BaseDataProvider]:
        symbol_def = resolve_symbol(symbol)
        provider_key = self._symbol_provider_overrides.get(
            symbol_def.display,
            symbol_def.provider,
        )
        provider = self._providers.get(provider_key)
        if provider is None:
            provider = self._providers[symbol_def.provider]
        return symbol_def, provider


def build_scalp_market_data_provider() -> MarketDataProvider:
    """Default provider for SPACE/scalp streams (Binance XAUUSDT)."""
    return MarketDataProvider()


def build_main_market_data_provider(settings: Settings) -> MarketDataProvider:
    """Main Trading Boss provider — OANDA for XAUUSD when API key is configured."""
    if not settings.oanda_api_key.strip():
        return build_scalp_market_data_provider()

    return MarketDataProvider(
        symbol_provider_overrides={"XAUUSD": "oanda"},
        oanda_api_key=settings.oanda_api_key,
        oanda_account_id=settings.oanda_account_id,
        oanda_env=settings.oanda_env,
    )


def get_market_data(
    symbol: str,
    timeframe: str,
    limit: int = DEFAULT_LIMIT,
) -> list[Candle]:
    return MarketDataProvider().get_market_data(symbol, timeframe, limit)


def print_market_data_test(
    symbol: str = "BTCUSDT",
    timeframe: str = "15m",
    limit: int = DEFAULT_LIMIT,
) -> None:
    provider = MarketDataProvider()
    symbol_def = provider.resolve(symbol)
    candles = provider.get_market_data(symbol, timeframe, limit)
    last_candle = candles[-1]

    print(f"Symbol: {symbol_def.display}")
    print(f"Data symbol: {symbol_def.data_symbol}")
    print(f"Source: {provider.data_source(symbol)}")
    print(f"Timeframe: {timeframe}")
    print(f"Last candle: {last_candle}")
    print(f"Number of candles loaded: {len(candles)}")


if __name__ == "__main__":
    for test_symbol in SUPPORTED_SYMBOLS:
        for test_timeframe in SUPPORTED_TIMEFRAMES:
            print()
            print_market_data_test(test_symbol, test_timeframe)
