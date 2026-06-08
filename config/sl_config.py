from __future__ import annotations

from dataclasses import dataclass

from config.symbols import resolve_symbol

DEPOSIT_PER_LOT_UNIT = 100.0

SYMBOL_SL_CONFIG_RAW: dict[str, dict[str, float]] = {
    "XAUUSD": {
        "min_sl_pips": 20,
        "max_sl_pips": 150,
        "pip_value": 0.10,
        "lot_per_100": 0.01,
    },
    "DJ30": {
        "min_sl_pips": 50,
        "max_sl_pips": 300,
        "pip_value": 1.0,
        "lot_per_100": 0.01,
    },
    "BTCUSDT": {
        "min_sl_pips": 100,
        "max_sl_pips": 500,
        "pip_value": 1.0,
        "lot_per_100": 0.01,
    },
}

# Dollar P/L per pip at 1.0 lot (used for Telegram result formatting).
PIP_DOLLAR_VALUE_PER_LOT: dict[str, float] = {
    "XAUUSD": 10.0,
    "DJ30": 1.0,
    "BTCUSDT": 1.0,
}


@dataclass(frozen=True)
class SymbolSLConfig:
    min_sl_pips: float
    max_sl_pips: float
    pip_size: float
    lot_per_100: float
    pip_value_per_lot: float


def _build_symbol_sl_config(symbol: str, raw: dict[str, float]) -> SymbolSLConfig:
    return SymbolSLConfig(
        min_sl_pips=float(raw["min_sl_pips"]),
        max_sl_pips=float(raw["max_sl_pips"]),
        pip_size=float(raw["pip_value"]),
        lot_per_100=float(raw["lot_per_100"]),
        pip_value_per_lot=PIP_DOLLAR_VALUE_PER_LOT.get(symbol, float(raw["pip_value"])),
    )


SYMBOL_SL_CONFIG: dict[str, SymbolSLConfig] = {
    symbol: _build_symbol_sl_config(symbol, raw)
    for symbol, raw in SYMBOL_SL_CONFIG_RAW.items()
}

# Backward-compatible alias used by trade_pnl and legacy imports.
SymbolSLRules = SymbolSLConfig
SYMBOL_SL_RULES = SYMBOL_SL_CONFIG


def get_sl_config(symbol: str) -> SymbolSLConfig | None:
    """Return SL config for a display symbol (XAUUSD, DJ30, BTCUSDT)."""
    try:
        display = resolve_symbol(symbol).display
    except ValueError:
        display = symbol.upper()
    return SYMBOL_SL_CONFIG.get(display)


def calculate_lot_size(deposit: float, lot_per_100: float = 0.01) -> float:
    """Return fixed lot size: lot_per_100 for each DEPOSIT_PER_LOT_UNIT deposit."""
    if deposit <= 0:
        raise ValueError("deposit must be positive")
    return round(deposit / DEPOSIT_PER_LOT_UNIT * lot_per_100, 2)


def calculate_lot_size_for_symbol(deposit: float, symbol: str) -> float:
    config = get_sl_config(symbol)
    lot_per_100 = config.lot_per_100 if config is not None else 0.01
    return calculate_lot_size(deposit, lot_per_100)
