"""Backtest: ICT Turtle Soup scalp (XAUUSD).

Failed breakout / liquidity sweep reversal:
1. Price sweeps a reference level (10-30 pips).
2. Same candle closes back inside the range (core filter).
3. Entry at candle close or FVG retest; SL behind sweep wick (max 30 pips).
4. TP1=1.5R, TP2=2.5R, TP3=nearest opposite liquidity (or 3.5R fallback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Literal

from agents.base import Direction
from agents.liquidity_agent import _find_equal_levels, _find_swing_points
from agents.session_agent import is_london_or_new_york_session
from backtest.engine import candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from config.sl_config import calculate_lot_size_for_symbol
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, align_trade_signal_direction
from strategy.scalp_mode import ScalpPublishGate
from tracking.console import configure_console_encoding
from tracking.trade_outcome import is_full_stop_loss
from tracking.trade_pnl import pip_size_for_symbol

WARMUP = 800
PROGRESS_EVERY = 2000
FILL_WINDOW = 20

MIN_PIERCE_PIPS = 10.0
MAX_PIERCE_PIPS = 30.0
STOP_BUFFER_PIPS = 2.0
MIN_SL_PIPS = 5.0
MAX_SL_PIPS = 30.0
TP1_R = 1.5
TP2_R = 2.5
TP3_FALLBACK_R = 3.5

ASIA_START_HOUR = 0
ASIA_END_HOUR = 8
POOL_LOOKBACK = 240
MIN_POOL_TOUCHES = 2
SWING_LOOKBACK = 2
EQUAL_TOLERANCE_PCT = 0.0008
LOCAL_SWING_LOOKBACK = 120

EntryMode = Literal["close", "fvg"]
LevelMode = Literal["core", "all"]
SessionMode = Literal["london_ny", "ict_windows", "core_08_16"]


@dataclass(frozen=True)
class RefLevel:
    price: float
    kind: Literal["high", "low"]
    label: str


@dataclass(frozen=True)
class TurtleSetup:
    direction: Literal["long", "short"]
    entry: float
    stop: float
    level: RefLevel
    signal_index: int
    entry_mode: EntryMode


@dataclass
class TurtleSoupStats:
    no_setup: int = 0
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
            1
            for t in self.trades
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
    now = timestamps[index]
    target_hour = (
        now.replace(minute=0, second=0, microsecond=0).timestamp() - 3600
    )
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


def prev_day_range(
    candles: list,
    timestamps: list[datetime],
    index: int,
) -> tuple[float, float] | None:
    target_date = timestamps[index].date() - timedelta(days=1)
    highs: list[float] = []
    lows: list[float] = []
    for j in range(index, -1, -1):
        ts = timestamps[j]
        if ts.date() == target_date:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
        elif ts.date() < target_date:
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


def local_swing_levels(candles: list, index: int) -> list[RefLevel]:
    import pandas as pd

    start = max(0, index - LOCAL_SWING_LOOKBACK)
    window = candles[start:index]
    if len(window) < 2 * SWING_LOOKBACK + 3:
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
    levels: list[RefLevel] = []
    if swing_highs:
        latest = max(swing_highs, key=lambda p: p.index)
        levels.append(RefLevel(latest.price, "high", "swing-high"))
    if swing_lows:
        latest = max(swing_lows, key=lambda p: p.index)
        levels.append(RefLevel(latest.price, "low", "swing-low"))
    return levels


def reference_levels_for_bar(
    candles: list,
    timestamps: list[datetime],
    index: int,
    mode: LevelMode,
) -> list[RefLevel]:
    levels: list[RefLevel] = []
    asia = asia_range_for_day(candles, timestamps, index)
    if asia:
        levels.append(RefLevel(asia[0], "high", "asia-high"))
        levels.append(RefLevel(asia[1], "low", "asia-low"))
    hour = prev_hour_range(candles, timestamps, index)
    if hour:
        levels.append(RefLevel(hour[0], "high", "hour-high"))
        levels.append(RefLevel(hour[1], "low", "hour-low"))
    prev_day = prev_day_range(candles, timestamps, index)
    if prev_day:
        levels.append(RefLevel(prev_day[0], "high", "prev-day-high"))
        levels.append(RefLevel(prev_day[1], "low", "prev-day-low"))
    levels.extend(local_swing_levels(candles, index))
    if mode == "all":
        levels.extend(liquidity_pool_levels(candles, index))
    return levels


def in_ict_session_window(ts: datetime) -> bool:
    minute_of_day = ts.hour * 60 + ts.minute
    windows = (
        (0, 120),        # 00:00-02:00 UTC
        (8 * 60, 10 * 60),  # 08:00-10:00 UTC
        (13 * 60 + 30, 15 * 60 + 30),  # 13:30-15:30 UTC
    )
    return any(start <= minute_of_day < end for start, end in windows)


def in_core_session_window(ts: datetime) -> bool:
    """08:00-16:00 UTC; skip Asia flat 00:00-06:00."""
    ts = _utc(ts)
    if 0 <= ts.hour < 6:
        return False
    minute = ts.hour * 60 + ts.minute
    return 8 * 60 <= minute < 16 * 60


def in_session(ts: datetime, mode: SessionMode) -> bool:
    if mode == "ict_windows":
        return in_ict_session_window(ts)
    if mode == "core_08_16":
        return in_core_session_window(ts)
    return is_london_or_new_york_session(ts)


def opposite_liquidity_tp3(
    setup: TurtleSetup,
    candles: list,
    timestamps: list[datetime],
    risk: float,
    level_mode: LevelMode,
) -> float:
    levels = reference_levels_for_bar(
        candles, timestamps, setup.signal_index, level_mode
    )
    if setup.direction == "long":
        candidates = [
            ref.price
            for ref in levels
            if ref.kind == "high" and ref.price > setup.entry + risk * 0.5
        ]
        if candidates:
            return min(candidates)
        return setup.entry + TP3_FALLBACK_R * risk
    candidates = [
        ref.price
        for ref in levels
        if ref.kind == "low" and ref.price < setup.entry - risk * 0.5
    ]
    if candidates:
        return max(candidates)
    return setup.entry - TP3_FALLBACK_R * risk


def detect_turtle_soup(
    candles: list,
    timestamps: list[datetime],
    index: int,
    pip: float,
    stats: TurtleSoupStats,
    *,
    level_mode: LevelMode,
    entry_mode: EntryMode,
) -> TurtleSetup | None:
    if index < 2:
        return None

    if entry_mode == "close":
        sweep = candles[index]
        pre = candles[index - 1]
        check_index = index
    else:
        # FVG confirm candle at index; sweep at index-1
        sweep = candles[index - 1]
        pre = candles[index - 2]
        confirm = candles[index]
        check_index = index - 1

    levels = reference_levels_for_bar(
        candles, timestamps, check_index, level_mode
    )
    if not levels:
        stats.no_setup += 1
        return None

    min_pierce = MIN_PIERCE_PIPS * pip
    max_pierce = MAX_PIERCE_PIPS * pip
    saw_sweep = False

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
            stop = sweep["low"] - STOP_BUFFER_PIPS * pip
            if entry_mode == "close":
                entry = sweep["close"]
                if entry <= stop:
                    continue
                return TurtleSetup(
                    "long", entry, stop, ref, index, entry_mode
                )
            if confirm["low"] > pre["high"]:
                entry = confirm["low"]
                if entry <= stop:
                    continue
                return TurtleSetup(
                    "long", entry, stop, ref, index, entry_mode
                )
            continue

        pierce = sweep["high"] - ref.price
        if not (
            min_pierce <= pierce <= max_pierce
            and sweep["close"] < ref.price
            and pre["high"] <= ref.price
        ):
            continue
        saw_sweep = True
        stop = sweep["high"] + STOP_BUFFER_PIPS * pip
        if entry_mode == "close":
            entry = sweep["close"]
            if entry >= stop:
                continue
            return TurtleSetup(
                "short", entry, stop, ref, index, entry_mode
            )
        if confirm["high"] < pre["low"]:
            entry = confirm["high"]
            if entry >= stop:
                continue
            return TurtleSetup(
                "short", entry, stop, ref, index, entry_mode
            )

    if not saw_sweep:
        stats.no_setup += 1
    return None


def build_signal(
    setup: TurtleSetup,
    candles: list,
    timestamps: list[datetime],
    symbol: str,
    level_mode: LevelMode,
) -> TradeSignal | None:
    risk = abs(setup.entry - setup.stop)
    pip = pip_size_for_symbol(symbol) or 1.0
    sl_pips = risk / pip
    if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
        return None

    tp3 = opposite_liquidity_tp3(
        setup, candles, timestamps, risk, level_mode
    )
    if setup.direction == "long":
        direction = Direction.LONG
        tp1 = setup.entry + TP1_R * risk
        tp2 = setup.entry + TP2_R * risk
        tp3 = max(
            tp3,
            setup.entry + TP3_FALLBACK_R * risk,
            tp2 + pip,
        )
    else:
        direction = Direction.SHORT
        tp1 = setup.entry - TP1_R * risk
        tp2 = setup.entry - TP2_R * risk
        tp3 = min(
            tp3,
            setup.entry - TP3_FALLBACK_R * risk,
            tp2 - pip,
        )

    signal = TradeSignal(
        direction=direction,
        entry=setup.entry,
        stop_loss=setup.stop,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        confidence=0.68,
        reason=(
            f"Turtle Soup {setup.direction.upper()} @ {setup.level.label} "
            f"{setup.level.price:.2f}, SL {sl_pips:.1f} pips, {setup.entry_mode}"
        ),
        lot_size=calculate_lot_size_for_symbol(200.0, symbol),
    )
    return align_trade_signal_direction(signal)


def find_fill_index(
    candles: list,
    setup: TurtleSetup,
    fill_window: int,
) -> int | None:
    if setup.entry_mode == "close":
        return setup.signal_index
    start = setup.signal_index + 1
    end = min(len(candles) - 1, setup.signal_index + fill_window)
    for j in range(start, end + 1):
        if setup.direction == "long" and candles[j]["low"] <= setup.entry:
            return j
        if setup.direction == "short" and candles[j]["high"] >= setup.entry:
            return j
    return None


def run(
    candles: list,
    symbol: str,
    *,
    interval_min: int,
    max_day: int,
    entry_mode: EntryMode,
    level_mode: LevelMode,
    session_mode: SessionMode,
) -> TurtleSoupStats:
    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    timestamps = build_timestamps(candles)
    gate = ScalpPublishGate(
        min_interval_seconds=interval_min * 60,
        max_signals_per_day=max_day,
    )
    simulator = TradeSimulator()
    stats = TurtleSoupStats()
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
        setup = detect_turtle_soup(
            candles,
            timestamps,
            index,
            pip,
            stats,
            level_mode=level_mode,
            entry_mode=entry_mode,
        )
        if setup is None:
            continue

        ts = timestamps[setup.signal_index]
        if not in_session(ts, session_mode):
            stats.off_session += 1
            continue
        if setup.signal_index <= open_until:
            stats.slot_busy += 1
            continue
        allowed, _ = gate.can_publish(display, ts)
        if not allowed:
            stats.rate_limited += 1
            continue

        signal = build_signal(setup, candles, timestamps, display, level_mode)
        if signal is None:
            stats.sl_invalid += 1
            continue

        fill_index = find_fill_index(candles, setup, FILL_WINDOW)
        if fill_index is None:
            stats.not_filled += 1
            continue

        simulated = simulator.simulate(
            signal,
            candles[fill_index:],
            entry_index=fill_index,
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


def period_days(stats: TurtleSoupStats) -> float:
    if not stats.period_start or not stats.period_end:
        return 1.0
    start = datetime.fromisoformat(stats.period_start)
    end = datetime.fromisoformat(stats.period_end)
    return max((end - start).total_seconds() / 86_400, 0.01)


def print_report(
    stats: TurtleSoupStats,
    *,
    tf: str,
    entry_mode: EntryMode,
    level_mode: LevelMode,
    session_mode: SessionMode,
) -> None:
    days = period_days(stats)
    print()
    print(
        f"=== ICT Turtle Soup ({tf}, entry={entry_mode}, "
        f"levels={level_mode}, session={session_mode}) ==="
    )
    print(
        "Sweep 10-30 pips + close back inside | SL за хвіст | "
        "TP1=1.5R / TP2=2.5R / TP3=ліквідність"
    )
    print(
        f"Період: {stats.period_start[:16]} -> {stats.period_end[:16]} "
        f"(~{days:.1f} днів)"
    )
    print()
    print(f"Без сетапу:             {stats.no_setup}")
    print(f"SL поза лімітами:       {stats.sl_invalid}")
    print(f"Поза сесією:            {stats.off_session}")
    print(f"Лімітник не наповнено:  {stats.not_filled}")
    print(f"Слот/частота:           {stats.slot_busy + stats.rate_limited}")
    print()
    print(
        f"Угод:                   {stats.total_signals} "
        f"({stats.total_signals / days:.2f}/день)"
    )
    if stats.total_signals:
        print(
            f"Win rate (TP1):         {stats.tp1_wins}/{stats.total_signals} "
            f"({stats.win_rate:.1f}%)"
        )
        print(f"  ТП1 + ТП2+:           {stats.tp2_full_wins}")
        print(f"  ТП1 -> БЕ:            {stats.tp1_then_be}")
        print(f"  Повний стоп:          {stats.full_stops}")
    print(f"Total R:                {stats.total_r:+.2f}R")
    if stats.total_signals:
        print(f"Середнє на угоду:       {stats.total_r / stats.total_signals:+.2f}R")


def main() -> None:
    import argparse

    configure_console_encoding()
    parser = argparse.ArgumentParser(description="ICT Turtle Soup backtest")
    parser.add_argument("--tf", default="5m", choices=["1m", "5m"])
    parser.add_argument("--candles", type=int, default=6000)
    parser.add_argument("--interval-min", type=int, default=5)
    parser.add_argument("--max-day", type=int, default=7)
    parser.add_argument(
        "--entry",
        default="close",
        choices=["close", "fvg"],
        help="market at sweep close or FVG retest",
    )
    parser.add_argument(
        "--levels",
        default="core",
        choices=["core", "all"],
        help="core=asia/hour/prev-day/swing; all=+pools",
    )
    parser.add_argument(
        "--session",
        default="london_ny",
        choices=["london_ny", "ict_windows"],
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="run M1/M5 x close/fvg x session matrix",
    )
    args = parser.parse_args()

    symbol = "XAUUSD"
    needed = WARMUP + args.candles + 1
    provider = MarketDataProvider()

    def load(tf: str, candles: int) -> list:
        need = WARMUP + candles + 1
        print(f"Завантаження {symbol} {tf} x{need}...", flush=True)
        data = provider.get_historical_market_data(symbol, tf, need)
        print(f"Отримано {len(data)} свічок", flush=True)
        return data

    if args.compare:
        candle_counts = {"5m": 6000, "1m": 25000}
        configs = [
            ("close", "core", "london_ny"),
            ("close", "all", "london_ny"),
            ("close", "core", "ict_windows"),
            ("fvg", "core", "london_ny"),
            ("fvg", "all", "london_ny"),
        ]
        print("\n=== Turtle Soup — порівняння конфігів ===\n")
        print(
            f"{'TF':<4} {'Entry':<6} {'Lvls':<5} {'Sess':<10} "
            f"{'Угод':>5} {'/день':>6} {'WR':>7} {'Стоп':>5} {'TotalR':>8}"
        )
        print("-" * 72)
        for tf, count in candle_counts.items():
            candles = load(tf, count)
            for entry_mode, level_mode, session_mode in configs:
                stats = run(
                    candles,
                    symbol,
                    interval_min=args.interval_min,
                    max_day=args.max_day,
                    entry_mode=entry_mode,
                    level_mode=level_mode,
                    session_mode=session_mode,
                )
                days = period_days(stats)
                wr = f"{stats.win_rate:.0f}%" if stats.total_signals else "—"
                print(
                    f"{tf:<4} {entry_mode:<6} {level_mode:<5} {session_mode:<10} "
                    f"{stats.total_signals:>5} "
                    f"{stats.total_signals / days:>6.2f} {wr:>7} "
                    f"{stats.full_stops:>5} {stats.total_r:>+7.2f}R",
                    flush=True,
                )
        return

    candles = load(args.tf, args.candles)
    stats = run(
        candles,
        symbol,
        interval_min=args.interval_min,
        max_day=args.max_day,
        entry_mode=args.entry,
        level_mode=args.levels,
        session_mode=args.session,
    )
    print_report(
        stats,
        tf=args.tf,
        entry_mode=args.entry,
        level_mode=args.levels,
        session_mode=args.session,
    )


if __name__ == "__main__":
    main()
