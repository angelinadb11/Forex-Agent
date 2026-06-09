from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from agents.base import Direction
from config.sl_config import get_sl_config


@dataclass(frozen=True)
class FVGZone:
    direction: Literal["bullish", "bearish"]
    middle_index: int
    gap_low: float
    gap_high: float
    age_candles: int
    size_pips: float
    filled: bool


@dataclass(frozen=True)
class OrderBlockZone:
    direction: Literal["bullish", "bearish"]
    candle_index: int
    zone_low: float
    zone_high: float
    impulse_pips: float
    age_candles: int


@dataclass(frozen=True)
class _FVGDefinition:
    direction: Literal["bullish", "bearish"]
    middle_index: int
    gap_low: float
    gap_high: float
    size_pips: float
    fill_index: int | None


@dataclass(frozen=True)
class _OrderBlockDefinition:
    direction: Literal["bullish", "bearish"]
    candle_index: int
    zone_low: float
    zone_high: float
    impulse_pips: float


@dataclass(frozen=True)
class ZoneCatalog:
    """Precomputed FVG/OB zones for fast per-bar lookups during backtests."""

    pip_size: float
    closes: tuple[float, ...]
    lows: tuple[float, ...]
    highs: tuple[float, ...]
    fvg_defs: tuple[_FVGDefinition, ...]
    ob_defs: tuple[_OrderBlockDefinition, ...]
    active_fvgs_by_bar: tuple[tuple[FVGZone, ...], ...]
    active_obs_by_bar: tuple[tuple[OrderBlockZone, ...], ...]
    unfilled_fvgs_by_bar: tuple[tuple[FVGZone, ...], ...]
    obs_retesting_by_bar: tuple[tuple[OrderBlockZone, ...], ...]

    @classmethod
    def from_candles(
        cls,
        candles: list[dict[str, float]],
        symbol: str,
        *,
        min_impulse_pips: float = 15.0,
        max_impulse_candles: int = 3,
        max_fvg_age: int = 20,
        max_ob_age: int = 30,
    ) -> ZoneCatalog:
        rows: list[dict[str, float]] = []
        for candle in candles:
            if not {"open", "high", "low", "close"}.issubset(candle):
                continue
            rows.append(
                {
                    "open": float(candle["open"]),
                    "high": float(candle["high"]),
                    "low": float(candle["low"]),
                    "close": float(candle["close"]),
                }
            )

        if not rows:
            raise ValueError("No candle data for zone catalog")

        df = pd.DataFrame(rows)
        pip_size = pip_size_for_symbol(symbol)
        fvg_defs = tuple(_build_fvg_definitions(df, pip_size))
        ob_defs = tuple(
            _build_order_block_definitions(
                df,
                pip_size,
                min_impulse_pips=min_impulse_pips,
                max_impulse_candles=max_impulse_candles,
            )
        )
        bar_count = len(df)
        active_fvgs_by_bar = _build_active_fvgs_by_bar(
            fvg_defs,
            bar_count,
            max_age=max_fvg_age,
        )
        active_obs_by_bar = _build_active_obs_by_bar(
            ob_defs,
            bar_count,
            max_age=max_ob_age,
            min_impulse_pips=min_impulse_pips,
        )
        closes = tuple(float(value) for value in df["close"])
        unfilled_fvgs_by_bar = tuple(
            tuple(fvg for fvg in bar_fvgs if not fvg.filled)
            for bar_fvgs in active_fvgs_by_bar
        )
        obs_retesting_by_bar = tuple(
            tuple(
                block
                for block in bar_blocks
                if price_inside_zone(closes[bar_index], block.zone_low, block.zone_high)
            )
            for bar_index, bar_blocks in enumerate(active_obs_by_bar)
        )
        return cls(
            pip_size=pip_size,
            closes=closes,
            lows=tuple(float(value) for value in df["low"]),
            highs=tuple(float(value) for value in df["high"]),
            fvg_defs=fvg_defs,
            ob_defs=ob_defs,
            active_fvgs_by_bar=active_fvgs_by_bar,
            active_obs_by_bar=active_obs_by_bar,
            unfilled_fvgs_by_bar=unfilled_fvgs_by_bar,
            obs_retesting_by_bar=obs_retesting_by_bar,
        )

    def close_at(self, bar_index: int) -> float:
        return self.closes[bar_index]

    def fvgs_at(self, bar_index: int, *, max_age: int = 20) -> list[FVGZone]:
        if bar_index < 0 or bar_index >= len(self.active_fvgs_by_bar):
            return []
        fvgs = self.active_fvgs_by_bar[bar_index]
        if max_age >= 20:
            return list(fvgs)
        return [fvg for fvg in fvgs if fvg.age_candles <= max_age]

    def unfilled_fvgs_at(self, bar_index: int, *, max_age: int = 20) -> tuple[FVGZone, ...]:
        if bar_index < 0 or bar_index >= len(self.unfilled_fvgs_by_bar):
            return ()
        fvgs = self.unfilled_fvgs_by_bar[bar_index]
        if max_age >= 20:
            return fvgs
        return tuple(fvg for fvg in fvgs if fvg.age_candles <= max_age)

    def obs_retesting_at(self, bar_index: int) -> tuple[OrderBlockZone, ...]:
        if bar_index < 0 or bar_index >= len(self.obs_retesting_by_bar):
            return ()
        return self.obs_retesting_by_bar[bar_index]

    def order_blocks_at(
        self,
        bar_index: int,
        *,
        max_age: int = 30,
        min_impulse_pips: float = 15.0,
    ) -> list[OrderBlockZone]:
        if bar_index < 0 or bar_index >= len(self.active_obs_by_bar):
            return []
        blocks = self.active_obs_by_bar[bar_index]
        if max_age >= 30 and min_impulse_pips <= 15.0:
            return list(blocks)
        return [
            block
            for block in blocks
            if block.age_candles <= max_age and block.impulse_pips >= min_impulse_pips
        ]

    def order_blocks_in_price_zone(
        self,
        bar_index: int,
        price: float,
        *,
        max_age: int = 30,
        min_impulse_pips: float = 15.0,
    ) -> list[OrderBlockZone]:
        return [
            block
            for block in self.order_blocks_at(
                bar_index,
                max_age=max_age,
                min_impulse_pips=min_impulse_pips,
            )
            if price_inside_zone(price, block.zone_low, block.zone_high)
        ]


def _build_fvg_definitions(df: pd.DataFrame, pip_size: float) -> list[_FVGDefinition]:
    definitions: list[_FVGDefinition] = []

    for index in range(1, len(df) - 1):
        prev_high = float(df.iloc[index - 1]["high"])
        prev_low = float(df.iloc[index - 1]["low"])
        next_low = float(df.iloc[index + 1]["low"])
        next_high = float(df.iloc[index + 1]["high"])

        if next_low > prev_high:
            gap_low = prev_high
            gap_high = next_low
            fill_index = _find_bullish_fvg_fill_index(df, index, gap_low)
            definitions.append(
                _FVGDefinition(
                    direction="bullish",
                    middle_index=index,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    size_pips=price_to_pips(gap_high - gap_low, pip_size),
                    fill_index=fill_index,
                )
            )

        if next_high < prev_low:
            gap_low = next_high
            gap_high = prev_low
            fill_index = _find_bearish_fvg_fill_index(df, index, gap_high)
            definitions.append(
                _FVGDefinition(
                    direction="bearish",
                    middle_index=index,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    size_pips=price_to_pips(gap_high - gap_low, pip_size),
                    fill_index=fill_index,
                )
            )

    return definitions


def _find_bullish_fvg_fill_index(
    df: pd.DataFrame,
    middle_index: int,
    gap_low: float,
) -> int | None:
    for candle_index in range(middle_index + 2, len(df)):
        if float(df.iloc[candle_index]["low"]) <= gap_low:
            return candle_index
    return None


def _find_bearish_fvg_fill_index(
    df: pd.DataFrame,
    middle_index: int,
    gap_high: float,
) -> int | None:
    for candle_index in range(middle_index + 2, len(df)):
        if float(df.iloc[candle_index]["high"]) >= gap_high:
            return candle_index
    return None


def _build_order_block_definitions(
    df: pd.DataFrame,
    pip_size: float,
    *,
    min_impulse_pips: float,
    max_impulse_candles: int,
) -> list[_OrderBlockDefinition]:
    blocks: list[_OrderBlockDefinition] = []

    for index in range(max(0, len(df) - max_impulse_candles)):
        candle = df.iloc[index]
        is_bearish = float(candle["close"]) < float(candle["open"])
        is_bullish = float(candle["close"]) > float(candle["open"])

        for impulse_candles in range(1, max_impulse_candles + 1):
            end_index = index + impulse_candles
            if end_index >= len(df):
                break

            window = df.iloc[index + 1 : end_index + 1]
            impulse_high = float(window["high"].max())
            impulse_low = float(window["low"].min())

            if is_bearish:
                impulse_pips = price_to_pips(
                    impulse_high - float(candle["low"]),
                    pip_size,
                )
                if impulse_pips >= min_impulse_pips:
                    blocks.append(
                        _OrderBlockDefinition(
                            direction="bullish",
                            candle_index=index,
                            zone_low=float(candle["low"]),
                            zone_high=float(candle["high"]),
                            impulse_pips=impulse_pips,
                        )
                    )
                    break

            if is_bullish:
                impulse_pips = price_to_pips(
                    float(candle["high"]) - impulse_low,
                    pip_size,
                )
                if impulse_pips >= min_impulse_pips:
                    blocks.append(
                        _OrderBlockDefinition(
                            direction="bearish",
                            candle_index=index,
                            zone_low=float(candle["low"]),
                            zone_high=float(candle["high"]),
                            impulse_pips=impulse_pips,
                        )
                    )
                    break

    return blocks


def _build_active_fvgs_by_bar(
    fvg_defs: tuple[_FVGDefinition, ...],
    bar_count: int,
    *,
    max_age: int,
) -> tuple[tuple[FVGZone, ...], ...]:
    active: list[list[FVGZone]] = [[] for _ in range(bar_count)]
    for definition in fvg_defs:
        start = definition.middle_index + 1
        end = min(bar_count - 1, definition.middle_index + max_age)
        if start > end:
            continue
        for bar_index in range(start, end + 1):
            active[bar_index].append(_fvg_from_definition(definition, bar_index))
    return tuple(tuple(zones) for zones in active)


def _build_active_obs_by_bar(
    ob_defs: tuple[_OrderBlockDefinition, ...],
    bar_count: int,
    *,
    max_age: int,
    min_impulse_pips: float,
) -> tuple[tuple[OrderBlockZone, ...], ...]:
    active: list[list[OrderBlockZone]] = [[] for _ in range(bar_count)]
    for definition in ob_defs:
        if definition.impulse_pips < min_impulse_pips:
            continue
        start = definition.candle_index
        end = min(bar_count - 1, definition.candle_index + max_age)
        if start > end:
            continue
        for bar_index in range(start, end + 1):
            active[bar_index].append(
                _order_block_from_definition(definition, bar_index)
            )
    return tuple(tuple(blocks) for blocks in active)


def _fvg_is_active(
    definition: _FVGDefinition,
    bar_index: int,
    *,
    max_age: int,
) -> bool:
    if bar_index < definition.middle_index + 1:
        return False
    return bar_index - definition.middle_index <= max_age


def _fvg_from_definition(definition: _FVGDefinition, bar_index: int) -> FVGZone:
    filled = (
        definition.fill_index is not None
        and definition.fill_index <= bar_index
    )
    return FVGZone(
        direction=definition.direction,
        middle_index=definition.middle_index,
        gap_low=definition.gap_low,
        gap_high=definition.gap_high,
        age_candles=bar_index - definition.middle_index,
        size_pips=definition.size_pips,
        filled=filled,
    )


def _order_block_is_active(
    definition: _OrderBlockDefinition,
    bar_index: int,
    *,
    max_age: int,
    min_impulse_pips: float,
) -> bool:
    if bar_index < definition.candle_index:
        return False
    if bar_index - definition.candle_index > max_age:
        return False
    return definition.impulse_pips >= min_impulse_pips


def _order_block_from_definition(
    definition: _OrderBlockDefinition,
    bar_index: int,
) -> OrderBlockZone:
    return OrderBlockZone(
        direction=definition.direction,
        candle_index=definition.candle_index,
        zone_low=definition.zone_low,
        zone_high=definition.zone_high,
        impulse_pips=definition.impulse_pips,
        age_candles=bar_index - definition.candle_index,
    )


def resolve_bar_index(context: dict[str, Any]) -> int:
    if "bar_index" in context:
        return int(context["bar_index"])
    candles = context.get("candles", [])
    if not candles:
        raise ValueError("No candle data in context")
    return len(candles) - 1


def resolve_zone_snapshot(
    context: dict[str, Any],
    symbol: str,
    *,
    max_fvg_age: int = 20,
    max_ob_age: int = 30,
    min_impulse_pips: float = 15.0,
) -> tuple[float, float, list[FVGZone], list[OrderBlockZone]]:
    catalog = context.get("zone_catalog")
    if isinstance(catalog, ZoneCatalog):
        bar_index = resolve_bar_index(context)
        current_price = catalog.close_at(bar_index)
        fvgs = [
            fvg
            for fvg in catalog.fvgs_at(bar_index, max_age=max_fvg_age)
            if not fvg.filled
        ]
        order_blocks = catalog.order_blocks_at(
            bar_index,
            max_age=max_ob_age,
            min_impulse_pips=min_impulse_pips,
        )
        return current_price, catalog.pip_size, fvgs, order_blocks

    df = candles_to_dataframe(context)
    pip_size = pip_size_for_symbol(symbol)
    current_price = float(df.iloc[-1]["close"])
    bar_index = len(df) - 1
    fvgs = detect_fvgs(df, pip_size)
    order_blocks = detect_order_blocks(
        df,
        pip_size,
        min_impulse_pips=min_impulse_pips,
    )

    fvgs = [
        FVGZone(
            direction=fvg.direction,
            middle_index=fvg.middle_index,
            gap_low=fvg.gap_low,
            gap_high=fvg.gap_high,
            age_candles=bar_index - fvg.middle_index,
            size_pips=fvg.size_pips,
            filled=fvg.filled,
        )
        for fvg in fvgs
        if bar_index - fvg.middle_index <= max_fvg_age
    ]
    order_blocks = [
        OrderBlockZone(
            direction=block.direction,
            candle_index=block.candle_index,
            zone_low=block.zone_low,
            zone_high=block.zone_high,
            impulse_pips=block.impulse_pips,
            age_candles=bar_index - block.candle_index,
        )
        for block in order_blocks
        if bar_index - block.candle_index <= max_ob_age
    ]
    return current_price, pip_size, fvgs, order_blocks


def candles_to_dataframe(context: dict[str, Any], *, min_candles: int = 20) -> pd.DataFrame:
    candles = context.get("candles", [])
    if not candles:
        raise ValueError("No candle data in context")

    rows: list[dict[str, float]] = []
    for candle in candles:
        if not {"open", "high", "low", "close"}.issubset(candle):
            continue
        rows.append(
            {
                "open": float(candle["open"]),
                "high": float(candle["high"]),
                "low": float(candle["low"]),
                "close": float(candle["close"]),
            }
        )

    if len(rows) < min_candles:
        raise ValueError(f"Need at least {min_candles} candles for zone analysis")

    return pd.DataFrame(rows)


def resolve_trend_direction(context: dict[str, Any]) -> Direction | None:
    trend = context.get("trend_direction")
    if isinstance(trend, Direction):
        return trend
    if trend is None:
        return None
    normalized = str(trend).strip().lower()
    if normalized in {"long", "bullish"}:
        return Direction.LONG
    if normalized in {"short", "bearish"}:
        return Direction.SHORT
    if normalized == "neutral":
        return Direction.NEUTRAL
    return None


def pip_size_for_symbol(symbol: str) -> float:
    try:
        display = symbol.upper()
        config = get_sl_config(display)
        if config is not None:
            return config.pip_size
    except ValueError:
        pass
    return 1.0


def price_to_pips(distance: float, pip_size: float) -> float:
    if pip_size <= 0:
        return abs(distance)
    return abs(distance) / pip_size


def price_inside_zone(price: float, zone_low: float, zone_high: float) -> bool:
    return zone_low <= price <= zone_high


def price_within_zone_tolerance(
    price: float,
    zone_low: float,
    zone_high: float,
    tolerance: float,
) -> bool:
    """Return True when price is inside the zone or within ``tolerance`` of its bounds."""
    if price_inside_zone(price, zone_low, zone_high):
        return True
    if tolerance <= 0:
        return False
    if price < zone_low:
        return zone_low - price <= tolerance
    return price - zone_high <= tolerance


def _entry_zone_atr_tolerance(
    context: dict[str, Any],
    atr_tolerance_multiplier: float | None,
) -> float:
    if atr_tolerance_multiplier is None or atr_tolerance_multiplier <= 0:
        return 0.0
    from signal_generator import DEFAULT_ATR_PERIOD, calculate_atr

    df = candles_to_dataframe(context, min_candles=DEFAULT_ATR_PERIOD + 1)
    return calculate_atr(df, period=DEFAULT_ATR_PERIOD) * atr_tolerance_multiplier


def zones_overlap(
    low_a: float,
    high_a: float,
    low_b: float,
    high_b: float,
    *,
    tolerance: float = 0.0,
) -> bool:
    return low_a - tolerance <= high_b and low_b - tolerance <= high_a


def detect_fvgs(df: pd.DataFrame, pip_size: float) -> list[FVGZone]:
    fvgs: list[FVGZone] = []

    for index in range(1, len(df) - 1):
        prev_high = float(df.iloc[index - 1]["high"])
        prev_low = float(df.iloc[index - 1]["low"])
        next_low = float(df.iloc[index + 1]["low"])
        next_high = float(df.iloc[index + 1]["high"])

        if next_low > prev_high:
            gap_low = prev_high
            gap_high = next_low
            filled = any(
                float(df.iloc[candle_index]["low"]) <= gap_low
                for candle_index in range(index + 2, len(df))
            )
            fvgs.append(
                FVGZone(
                    direction="bullish",
                    middle_index=index,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    age_candles=len(df) - 1 - index,
                    size_pips=price_to_pips(gap_high - gap_low, pip_size),
                    filled=filled,
                )
            )

        if next_high < prev_low:
            gap_low = next_high
            gap_high = prev_low
            filled = any(
                float(df.iloc[candle_index]["high"]) >= gap_high
                for candle_index in range(index + 2, len(df))
            )
            fvgs.append(
                FVGZone(
                    direction="bearish",
                    middle_index=index,
                    gap_low=gap_low,
                    gap_high=gap_high,
                    age_candles=len(df) - 1 - index,
                    size_pips=price_to_pips(gap_high - gap_low, pip_size),
                    filled=filled,
                )
            )

    return fvgs


def count_fvgs_at_level(fvgs: list[FVGZone], target: FVGZone, pip_size: float) -> int:
    tolerance = pip_size * 10
    count = 0
    for fvg in fvgs:
        if fvg.direction != target.direction or fvg.filled:
            continue
        if zones_overlap(
            fvg.gap_low,
            fvg.gap_high,
            target.gap_low,
            target.gap_high,
            tolerance=tolerance,
        ):
            count += 1
    return count


def detect_order_blocks(
    df: pd.DataFrame,
    pip_size: float,
    *,
    min_impulse_pips: float = 15.0,
    max_impulse_candles: int = 3,
) -> list[OrderBlockZone]:
    blocks: list[OrderBlockZone] = []

    for index in range(max(0, len(df) - max_impulse_candles)):
        candle = df.iloc[index]
        is_bearish = float(candle["close"]) < float(candle["open"])
        is_bullish = float(candle["close"]) > float(candle["open"])

        for impulse_candles in range(1, max_impulse_candles + 1):
            end_index = index + impulse_candles
            if end_index >= len(df):
                break

            window = df.iloc[index + 1 : end_index + 1]
            impulse_high = float(window["high"].max())
            impulse_low = float(window["low"].min())

            if is_bearish:
                impulse_pips = price_to_pips(
                    impulse_high - float(candle["low"]),
                    pip_size,
                )
                if impulse_pips >= min_impulse_pips:
                    blocks.append(
                        OrderBlockZone(
                            direction="bullish",
                            candle_index=index,
                            zone_low=float(candle["low"]),
                            zone_high=float(candle["high"]),
                            impulse_pips=impulse_pips,
                            age_candles=len(df) - 1 - index,
                        )
                    )
                    break

            if is_bullish:
                impulse_pips = price_to_pips(
                    float(candle["high"]) - impulse_low,
                    pip_size,
                )
                if impulse_pips >= min_impulse_pips:
                    blocks.append(
                        OrderBlockZone(
                            direction="bearish",
                            candle_index=index,
                            zone_low=float(candle["low"]),
                            zone_high=float(candle["high"]),
                            impulse_pips=impulse_pips,
                            age_candles=len(df) - 1 - index,
                        )
                    )
                    break

    return blocks


def fvg_confirms_order_block(
    block: OrderBlockZone,
    fvgs: list[FVGZone],
    pip_size: float,
) -> bool:
    tolerance = pip_size * 10
    expected = "bullish" if block.direction == "bullish" else "bearish"
    for fvg in fvgs:
        if fvg.direction != expected or fvg.filled:
            continue
        if zones_overlap(
            block.zone_low,
            block.zone_high,
            fvg.gap_low,
            fvg.gap_high,
            tolerance=tolerance,
        ):
            return True
    return False


def _direction_to_zone_bias(direction: Direction) -> str:
    return "bullish" if direction == Direction.LONG else "bearish"


def price_in_active_entry_zone(
    context: dict[str, Any],
    direction: Direction,
    *,
    atr_tolerance_multiplier: float | None = None,
) -> bool:
    """Return True when price is inside, or near, an active OB or unfilled FVG."""
    if direction == Direction.NEUTRAL:
        return False

    expected = _direction_to_zone_bias(direction)
    tolerance = _entry_zone_atr_tolerance(context, atr_tolerance_multiplier)
    catalog = context.get("zone_catalog")

    if isinstance(catalog, ZoneCatalog):
        bar_index = resolve_bar_index(context)
        price = catalog.close_at(bar_index)
        if tolerance <= 0:
            for block in catalog.obs_retesting_at(bar_index):
                if block.direction == expected:
                    return True
            for fvg in catalog.unfilled_fvgs_at(bar_index):
                if fvg.direction != expected:
                    continue
                if price_inside_zone(price, fvg.gap_low, fvg.gap_high):
                    return True
            return False

        for block in catalog.order_blocks_at(bar_index):
            if block.direction != expected:
                continue
            if price_within_zone_tolerance(
                price,
                block.zone_low,
                block.zone_high,
                tolerance,
            ):
                return True
        for fvg in catalog.unfilled_fvgs_at(bar_index):
            if fvg.direction != expected:
                continue
            if price_within_zone_tolerance(
                price,
                fvg.gap_low,
                fvg.gap_high,
                tolerance,
            ):
                return True
        return False

    try:
        symbol = str(context.get("symbol", "UNKNOWN"))
        current_price, _, fvgs, order_blocks = resolve_zone_snapshot(context, symbol)
    except ValueError:
        return False

    for block in order_blocks:
        if block.direction != expected:
            continue
        if price_within_zone_tolerance(
            current_price,
            block.zone_low,
            block.zone_high,
            tolerance,
        ):
            return True

    for fvg in fvgs:
        if fvg.direction != expected or fvg.filled:
            continue
        if price_within_zone_tolerance(
            current_price,
            fvg.gap_low,
            fvg.gap_high,
            tolerance,
        ):
            return True

    return False
