from __future__ import annotations

from abc import ABC, abstractmethod

from config.symbols import SUPPORTED_TIMEFRAMES

Candle = dict[str, float]
DEFAULT_LIMIT = 500

SYMBOL_ALIASES = {
    "US30": "DJ30",
}


class BaseDataProvider(ABC):
    """Base interface for market data providers."""

    @property
    @abstractmethod
    def supported_symbols(self) -> tuple[str, ...]:
        pass

    @abstractmethod
    def get_market_data(
        self,
        symbol: str,
        timeframe: str,
        limit: int = DEFAULT_LIMIT,
    ) -> list[Candle]:
        pass

    def normalize_symbol(self, symbol: str) -> str:
        symbol = symbol.upper()
        return SYMBOL_ALIASES.get(symbol, symbol)

    def validate_timeframe(self, timeframe: str) -> str:
        if timeframe not in SUPPORTED_TIMEFRAMES:
            supported = ", ".join(SUPPORTED_TIMEFRAMES)
            raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {supported}")
        return timeframe

    def validate_limit(self, limit: int) -> None:
        if limit < 1 or limit > 1000:
            raise ValueError("limit must be between 1 and 1000")
