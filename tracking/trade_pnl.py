from __future__ import annotations

from config.sl_config import SYMBOL_SL_CONFIG, calculate_lot_size_for_symbol, get_sl_config

DEFAULT_LOT_SIZE = calculate_lot_size_for_symbol(200.0, "XAUUSD")


def pip_size_for_symbol(symbol: str) -> float | None:
    config = get_sl_config(symbol)
    return config.pip_size if config else None


def price_distance_pips(distance: float, pip_size: float) -> float:
    return abs(distance) / pip_size


def pips_to_dollars(symbol: str, pips: float, lot_size: float = DEFAULT_LOT_SIZE) -> float:
    config = get_sl_config(symbol)
    if config is None:
        return 0.0
    return abs(pips) * config.pip_value_per_lot * lot_size


def signed_pips_long(entry: float, price: float, pip_size: float) -> float:
    return (price - entry) / pip_size


def signed_pips_short(entry: float, price: float, pip_size: float) -> float:
    return (entry - price) / pip_size


def distance_to_sl_pips(
    *,
    symbol: str,
    direction: str,
    current_price: float,
    stop_loss: float,
) -> float:
    pip_size = pip_size_for_symbol(symbol)
    if pip_size is None:
        return abs(current_price - stop_loss)
    if direction == "long":
        return max(0.0, (current_price - stop_loss) / pip_size)
    return max(0.0, (stop_loss - current_price) / pip_size)
