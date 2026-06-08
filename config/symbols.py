from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULT_CONFIG = {
    "symbol": "BTCUSDT",
    "symbols": ["BTCUSDT", "XAUUSD", "DJ30"],
    "timeframe": "15m",
    "candle_limit": 500,
    "london_ny_session_symbols": [],
    "session_confidence_symbols": [],
}


@dataclass(frozen=True)
class SymbolDefinition:
    """User-facing symbol mapped to a provider-specific data symbol."""

    display: str
    data_symbol: str
    provider: str


SYMBOL_DEFINITIONS: dict[str, SymbolDefinition] = {
    "BTCUSDT": SymbolDefinition("BTCUSDT", "BTCUSDT", "binance"),
    "XAUUSD": SymbolDefinition("XAUUSD", "XAUUSDT", "binance"),
    "XAUUSDT": SymbolDefinition("XAUUSD", "XAUUSDT", "binance"),
    "DJ30": SymbolDefinition("DJ30", "DJ30", "index"),
    "US30": SymbolDefinition("DJ30", "DJ30", "index"),
}

SUPPORTED_SYMBOLS = ("BTCUSDT", "XAUUSD", "DJ30")
DEFAULT_SYMBOLS = ("BTCUSDT", "XAUUSD", "DJ30")
SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m")
SYMBOL_ALIASES = ("US30", "XAUUSDT")


def resolve_symbol(symbol: str) -> SymbolDefinition:
    key = symbol.upper()
    if key not in SYMBOL_DEFINITIONS:
        supported = ", ".join(SUPPORTED_SYMBOLS)
        raise ValueError(f"Unsupported symbol '{symbol}'. Use one of: {supported}")
    return SYMBOL_DEFINITIONS[key]


def resolve_symbols(symbols: list[str] | tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Return unique display symbols in config order."""
    source = DEFAULT_SYMBOLS if symbols is None else symbols
    seen: set[str] = set()
    resolved: list[str] = []
    for symbol in source:
        display = resolve_symbol(str(symbol).upper()).display
        if display in seen:
            continue
        seen.add(display)
        resolved.append(display)
    if not resolved:
        raise ValueError("At least one symbol must be configured")
    return tuple(resolved)


def resolve_timeframe(timeframe: str) -> str:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(SUPPORTED_TIMEFRAMES)
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {supported}")
    return timeframe


def _normalize_symbol_list(raw_symbols: object, fallback: tuple[str, ...]) -> list[str]:
    if raw_symbols is None:
        return list(fallback)
    if not isinstance(raw_symbols, list) or not raw_symbols:
        return list(fallback)
    return [str(symbol).upper() for symbol in raw_symbols]


def load_config_file() -> dict:
    if not CONFIG_FILE.exists():
        return DEFAULT_CONFIG.copy()

    with CONFIG_FILE.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = DEFAULT_CONFIG.copy()
    config.update(payload)

    config["symbols"] = list(
        resolve_symbols(_normalize_symbol_list(config.get("symbols"), DEFAULT_SYMBOLS))
    )
    config["symbol"] = resolve_symbol(str(config["symbol"]).upper()).display
    config["london_ny_session_symbols"] = [
        resolve_symbol(str(symbol).upper()).display
        for symbol in config.get("london_ny_session_symbols", [])
    ]
    config["session_confidence_symbols"] = [
        resolve_symbol(str(symbol).upper()).display
        for symbol in config.get("session_confidence_symbols", [])
    ]
    resolve_timeframe(str(config["timeframe"]))

    return config


def save_default_config_file() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if CONFIG_FILE.exists():
        return

    with CONFIG_FILE.open("w", encoding="utf-8") as handle:
        json.dump(DEFAULT_CONFIG, handle, indent=2)
