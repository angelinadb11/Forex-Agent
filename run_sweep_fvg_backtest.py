"""Backtest: Liquidity Sweep + FVG retest entry (XAUUSD scalp).

Setup (per user's spec):
1. Reference levels: Asia session high/low (00:00-08:00 UTC) and previous
   full clock-hour high/low.
2. Sweep: candle pierces the level with its wick and closes back inside.
3. Reversal impulse leaves an FVG (3-candle gap) -> limit order at the FVG
   edge, stop behind the sweep wick + small buffer.
4. TP1 = 1R (close 50%, SL to BE), TP2 = 2R. London/NY sessions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal

from agents.liquidity_agent import _find_equal_levels, _find_swing_points
from agents.session_agent import is_london_or_new_york_session
from backtest.engine import candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, align_trade_signal_direction
from strategy.scalp_mode import ScalpPublishGate
from tracking.console import configure_console_encoding
from tracking.trade_outcome import is_full_stop_loss
from tracking.trade_pnl import pip_size_for_symbol

WARMUP = 600  # need enough history for Asia range of the current day
PROGRESS_EVERY = 2000

MIN_PIERCE_PIPS = 5.0
MAX_PIERCE_PIPS = 60.0
STOP_BUFFER_PIPS = 3.0
MIN_SL_PIPS = 5.0
MAX_SL_PIPS = 40.0
TP2_R = 2.0
FILL_WINDOW_CANDLES = 30
ASIA_START_HOUR = 0
ASIA_END_HOUR = 8
POOL_LOOKBACK = 240
MIN_POOL_TOUCHES = 2
SWING_LOOKBACK = 2
EQUAL_TOLERANCE_PCT = 0.0008

LevelMode = Literal["asia", "pools", "all"]


@dataclass(frozen=True)
class RefLevel:
    price: float
    kind: Literal["high", "low"]
    label: str


@dataclass
class SweepFvgStats:
    no_setup: int = 0
    no_fvg: int = 0
    sl_invalid: int = 0
    off_session: int = 0
    not_filled: int = 0
    rate_limited: int = 0
    slot_busy: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    period_start: str = ""
    period_end: str = ""

    @property
    def total_signals(self) -> int:
        return len(self.trades)

    @property
    def tp1_wins(self) -> int:
        return sum(1 for t in self.trades if t.tp1_hit)

    @property
    def win_rate(self) -> float:
        return self.tp1_wins / self.total_signals * 100 if self.trades else 0.0

    @property
    def tp2_full_wins(self) -> int:
        return sum(1 for t in self.trades if t.tp2_hit)

    @property
    def tp1_then_be(self) -> int:
        return sum(
            1 for t in self.trades
            if t.tp1_hit and not t.tp2_hit and t.result == "breakeven"
        )

    @property
    def full_stops(self) -> int:
        return sum(1 for t in self.trades if is_full_stop_loss(t))

    @property
    def total_r(self) -> float:
        return sum(t.pnl_r for t in self.trades)


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def build_timestamps(candles: list) -> list[datetime]:
    return [_utc(candle_timestamp(candles, i)) for i in range(len(candles))]


def asia_range_for_day(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    """High/low of today's completed Asia session (00:00-08:00 UTC)."""
    now = timestamps[index]
    if now.hour < ASIA_END_HOUR:
        return None
    day = now.date()
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index, -1, -1):
        ts = timestamps[j]
        if ts.date() != day:
            if ts.date() < day:
                break
            continue
        if ASIA_START_HOUR <= ts.hour < ASIA_END_HOUR:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
    if not highs:
        return None
    return max(highs), min(lows)


def prev_hour_range(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    """High/low of the previous full clock hour."""
    now = timestamps[index]
    target_hour = (now.replace(minute=0, second=0, microsecond=0)
                   .timestamp() - 3600)
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index, -1, -1):
        ts = timestamps[j].replace(minute=0, second=0, microsecond=0).timestamp()
        if ts == target_hour:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
        elif ts < target_hour:
            break
    if not highs:
        return None
    return max(highs), min(lows)


def liquidity_pool_levels(
    candles: list,
    index: int,
    *,
    lookback: int = POOL_LOOKBACK,
) -> list[RefLevel]:
    import pandas as pd

    start = max(0, index - lookback)
    window = candles[start:index]
    if len(window) < 60:
        return []

    df = pd.DataFrame(
        {
            "open": [c["open"] for c in window],
            "high": [c["high"] for c in window],
            "low": [c["low"] for c in window],
            "close": [c["close"] for c in window],
        }
    )
    swing_highs, swing_lows = _find_swing_points(df, lookback=SWING_LOOKBACK)
    equal_highs = _find_equal_levels(swing_highs, tolerance_pct=EQUAL_TOLERANCE_PCT)
    equal_lows = _find_equal_levels(swing_lows, tolerance_pct=EQUAL_TOLERANCE_PCT)

    levels: list[RefLevel] = []
    for cluster in equal_highs:
        if cluster.count >= MIN_POOL_TOUCHES:
            levels.append(RefLevel(cluster.level, "high", f"pool-high x{cluster.count}"))
    for cluster in equal_lows:
        if cluster.count >= MIN_POOL_TOUCHES:
            levels.append(RefLevel(cluster.level, "low", f"pool-low x{cluster.count}"))
    return levels


def reference_levels_for_bar(
    candles: list,
    timestamps: list[datetime],
    index: int,
    mode: LevelMode,
) -> list[RefLevel]:
    levels: list[RefLevel] = []
    if mode in ("asia", "all"):
        asia = asia_range_for_day(candles, timestamps, index)
        if asia:
            levels.append(RefLevel(asia[0], "high", "asia-high"))
            levels.append(RefLevel(asia[1], "low", "asia-low"))
        hour = prev_hour_range(candles, timestamps, index)
        if hour:
            levels.append(RefLevel(hour[0], "high", "hour-high"))
            levels.append(RefLevel(hour[1], "low", "hour-low"))
    if mode in ("pools", "all"):
        levels.extend(liquidity_pool_levels(candles, index))
    return levels


@dataclass(frozen=True)
class PendingOrder:
    signal_index: int
    direction: str  # "long" | "short"
    entry: float
    stop: float
    level: float


def detect_sweep_fvg(
    candles: list,
    timestamps: list[datetime],
    index: int,
    pip: float,
    stats: SweepFvgStats,
    *,
    level_mode: LevelMode = "asia",
) -> PendingOrder | None:
    """Detect at candle ``index`` (FVG confirm candle, sweep = index-1)."""
    if index < 3:
        return None
    sweep = candles[index - 1]
    pre = candles[index - 2]
    confirm = candles[index]

    levels = reference_levels_for_bar(candles, timestamps, index - 1, level_mode)
    if not levels:
        stats.no_setup += 1
        return None

    min_pierce = MIN_PIERCE_PIPS * pip
    max_pierce = MAX_PIERCE_PIPS * pip
    saw_sweep = False
    saw_sweep_no_fvg = False

    for ref in levels:
        if ref.kind == "low":
            pierce = ref.price - sweep["low"]
            if not (
                min_pierce <= pierce <= max_pierce
                and sweep["close"] > ref.price
                and pre["low"] >= ref.price
            ):
                continue
            saw_sweep = True
            if confirm["low"] > pre["high"]:
                entry = confirm["low"]
                stop = sweep["low"] - STOP_BUFFER_PIPS * pip
                if entry > stop:
                    return PendingOrder(index, "long", entry, stop, ref.price)
            saw_sweep_no_fvg = True
            continue

        pierce = sweep["high"] - ref.price
        if not (
            min_pierce <= pierce <= max_pierce
            and sweep["close"] < ref.price
            and pre["high"] <= ref.price
        ):
            continue
        saw_sweep = True
        if confirm["high"] < pre["low"]:
            entry = confirm["high"]
            stop = sweep["high"] + STOP_BUFFER_PIPS * pip
            if entry < stop:
                return PendingOrder(index, "short", entry, stop, ref.price)
        saw_sweep_no_fvg = True

    if saw_sweep_no_fvg:
        stats.no_fvg += 1
    elif saw_sweep:
        stats.no_fvg += 1
    else:
        stats.no_setup += 1
    return None


def build_signal(order: PendingOrder, symbol: str) -> TradeSignal | None:
    from agents.base import Direction
    from config.sl_config import calculate_lot_size_for_symbol

    risk = abs(order.entry - order.stop)
    pip = pip_size_for_symbol(symbol) or 1.0
    sl_pips = risk / pip
    if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
        return None

    if order.direction == "long":
        direction = Direction.LONG
        tp1 = order.entry + risk
        tp2 = order.entry + TP2_R * risk
    else:
        direction = Direction.SHORT
        tp1 = order.entry - risk
        tp2 = order.entry - TP2_R * risk

    signal = TradeSignal(
        direction=direction,
        entry=order.entry,
        stop_loss=order.stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp2,
        confidence=0.65,
        reason=(
            f"Sweep+FVG {order.direction.upper()}: level {order.level:.2f}, "
            f"SL {sl_pips:.1f} pips, retest entry"
        ),
        lot_size=calculate_lot_size_for_symbol(200.0, symbol),
    )
    return align_trade_signal_direction(signal)


def find_fill_index(
    candles: list,
    order: PendingOrder,
    fill_window: int,
) -> int | None:
    """First candle index where the limit order is touched."""
    start = order.signal_index + 1
    end = min(len(candles) - 1, order.signal_index + fill_window)
    for j in range(start, end + 1):
        if order.direction == "long" and candles[j]["low"] <= order.entry:
            return j
        if order.direction == "short" and candles[j]["high"] >= order.entry:
            return j
    return None


def run(
    candles: list,
    symbol: str,
    *,
    interval_min: int,
    max_day: int,
    fill_window: int,
    level_mode: LevelMode = "asia",
) -> SweepFvgStats:
    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    timestamps = build_timestamps(candles)
    gate = ScalpPublishGate(
        min_interval_seconds=interval_min * 60,
        max_signals_per_day=max_day,
    )
    simulator = TradeSimulator()
    stats = SweepFvgStats()
    open_until = -1

    scan_start = WARMUP
    scan_end = len(candles) - 1
    progress = BacktestScanProgress(
        scan_start,
        scan_end,
        update_every=PROGRESS_EVERY,
        message_template="Оброблено {processed}/{total} свічок...",
        finish_message="Сканування завершено.",
    )

    for index in range(scan_start, scan_end):
        progress.update(index)
        order = detect_sweep_fvg(
            candles, timestamps, index, pip, stats, level_mode=level_mode
        )
        if order is None:
            continue

        ts = timestamps[index]
        if not is_london_or_new_york_session(ts):
            stats.off_session += 1
            continue
        if index <= open_until:
            stats.slot_busy += 1
            continue
        allowed, _ = gate.can_publish(display, ts)
        if not allowed:
            stats.rate_limited += 1
            continue

        signal = build_signal(order, display)
        if signal is None:
            stats.sl_invalid += 1
            continue

        fill_index = find_fill_index(candles, order, fill_window)
        if fill_index is None:
            stats.not_filled += 1
            continue

        simulated = simulator.simulate(
            signal,
            candles[fill_index:],
            entry_index=fill_index - 1,
            mode=TradeManagementMode.PARTIAL,
        )
        if simulated is None:
            continue

        stats.trades.append(simulated)
        gate.record(display, ts)
        open_until = simulated.exit_index

    progress.finish()
    stats.period_start = timestamps[scan_start].isoformat()
    stats.period_end = timestamps[scan_end].isoformat()
    return stats


def print_report(stats: SweepFvgStats, tf: str, level_mode: LevelMode) -> None:
    start = datetime.fromisoformat(stats.period_start)
    end = datetime.fromisoformat(stats.period_end)
    days = max((end - start).total_seconds() / 86_400, 0.01)
    print()
    print(f"=== Sweep + FVG Backtest (XAUUSD, {tf}, levels={level_mode}) ===")
    print("Рівні: Азія (00-08 UTC) + попередня година | sweep 5-60 піпс | вхід на FVG")
    print(f"Період: {stats.period_start[:16]} -> {stats.period_end[:16]} (~{days:.1f} днів)")
    print()
    print(f"Без сетапу:             {stats.no_setup}")
    print(f"Sweep без FVG:          {stats.no_fvg}")
    print(f"SL поза лімітами:       {stats.sl_invalid}")
    print(f"Поза сесією:            {stats.off_session}")
    print(f"Лімітник не наповнено:  {stats.not_filled}")
    print(f"Слот/частота:           {stats.slot_busy + stats.rate_limited}")
    print()
    print(f"Угод:                   {stats.total_signals} ({stats.total_signals / days:.2f}/день)")
    if stats.total_signals:
        print(
            f"Win rate (TP1):         {stats.tp1_wins}/{stats.total_signals} "
            f"({stats.win_rate:.1f}%)"
        )
        print(f"  ТП1 + ТП2 (повний):   {stats.tp2_full_wins} (+1.5R)")
        print(f"  ТП1 -> БЕ:            {stats.tp1_then_be} (+0.5R)")
        print(f"  Повний стоп:          {stats.full_stops} (-1R)")
    print(f"Total R:                {stats.total_r:+.2f}R")
    if stats.total_signals:
        print(f"Середнє на угоду:       {stats.total_r / stats.total_signals:+.2f}R")


def main() -> None:
    import argparse

    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Sweep + FVG backtest")
    parser.add_argument("--tf", default="5m", choices=["1m", "5m"])
    parser.add_argument("--candles", type=int, default=6000)
    parser.add_argument("--interval-min", type=int, default=5)
    parser.add_argument("--max-day", type=int, default=6)
    parser.add_argument("--fill-window", type=int, default=FILL_WINDOW_CANDLES)
    parser.add_argument("--levels", default="asia", choices=["asia", "pools", "all"])
    args = parser.parse_args()

    symbol = "XAUUSD"
    needed = WARMUP + args.candles + 1
    provider = MarketDataProvider()
    print(f"Завантаження {symbol} {args.tf} x{needed}...", flush=True)
    candles = provider.get_historical_market_data(symbol, args.tf, needed)
    print(f"Отримано {len(candles)} свічок", flush=True)

    stats = run(
        candles,
        symbol,
        interval_min=args.interval_min,
        max_day=args.max_day,
        fill_window=args.fill_window,
        level_mode=args.levels,
    )
    print_report(stats, args.tf, args.levels)


if __name__ == "__main__":
    main()
