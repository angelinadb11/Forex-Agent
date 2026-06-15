"""Asian range fake breakout + engulfing (XAUUSD, M15).

Asia session 00:00-08:00 UTC. Fake breakout of range high/low with close back
inside, confirmed by engulfing candle. London/NY sessions only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agents.base import Direction
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

WARMUP = 200
PROGRESS_EVERY = 500
ASIA_START = 0
ASIA_END = 8
STOP_BUFFER_PIPS = 3.0
MIN_SL_PIPS = 10.0
MAX_SL_PIPS = 40.0
TP2_R = 2.0


@dataclass
class AsianFakeStats:
    no_range: int = 0
    no_fake: int = 0
    no_engulf: int = 0
    sl_invalid: int = 0
    off_session: int = 0
    rate_limited: int = 0
    slot_busy: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    period_start: str = ""
    period_end: str = ""


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def asia_range(candles: list, timestamps: list[datetime], index: int):
    day = timestamps[index].date()
    if timestamps[index].hour < ASIA_END:
        return None
    highs, lows = [], []
    for j in range(index, -1, -1):
        ts = timestamps[j]
        if ts.date() != day:
            break
        if ASIA_START <= ts.hour < ASIA_END:
            highs.append(candles[j]["high"])
            lows.append(candles[j]["low"])
    if not highs:
        return None
    return max(highs), min(lows)


def is_bullish_engulf(prev: dict, curr: dict) -> bool:
    return (
        curr["close"] > curr["open"]
        and prev["close"] < prev["open"]
        and curr["close"] >= prev["open"]
        and curr["open"] <= prev["close"]
    )


def is_bearish_engulf(prev: dict, curr: dict) -> bool:
    return (
        curr["close"] < curr["open"]
        and prev["close"] > prev["open"]
        and curr["close"] <= prev["open"]
        and curr["open"] >= prev["close"]
    )


def detect_setup(candles: list, timestamps: list[datetime], index: int, pip: float):
    if index < 3:
        return None
    fake = candles[index - 2]
    pre = candles[index - 3]
    engulf = candles[index - 1]
    confirm_index = index - 1

    asia = asia_range(candles, timestamps, index - 1)
    if asia is None:
        return None
    hi, lo = asia

    # LONG: fake sweep below Asia low, engulf confirms.
    if (
        fake["low"] < lo
        and fake["close"] > lo
        and pre["low"] >= lo
        and is_bullish_engulf(fake, engulf)
    ):
        entry = engulf["close"]
        stop = fake["low"] - STOP_BUFFER_PIPS * pip
        if entry > stop:
            return Direction.LONG, entry, stop, confirm_index

    if (
        fake["high"] > hi
        and fake["close"] < hi
        and pre["high"] <= hi
        and is_bearish_engulf(fake, engulf)
    ):
        entry = engulf["close"]
        stop = fake["high"] + STOP_BUFFER_PIPS * pip
        if entry < stop:
            return Direction.SHORT, entry, stop, confirm_index

    return None


def run(candles: list, symbol: str, *, interval_min: int, max_day: int) -> AsianFakeStats:
    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    timestamps = [_utc(candle_timestamp(candles, i)) for i in range(len(candles))]
    gate = ScalpPublishGate(min_interval_seconds=interval_min * 60, max_signals_per_day=max_day)
    simulator = TradeSimulator()
    stats = AsianFakeStats()
    open_until = -1

    scan_start = WARMUP
    scan_end = len(candles) - 1
    progress = BacktestScanProgress(
        scan_start,
        scan_end,
        update_every=PROGRESS_EVERY,
        message_template="Оброблено {processed}/{total} M15 свічок...",
        finish_message="Сканування завершено.",
    )

    for index in range(scan_start, scan_end):
        progress.update(index)
        ts = timestamps[index]
        if not is_london_or_new_york_session(ts):
            continue
        if index <= open_until:
            stats.slot_busy += 1
            continue

        setup = detect_setup(candles, timestamps, index, pip)
        if setup is None:
            stats.no_fake += 1
            continue

        direction, entry, stop, entry_index = setup
        risk = abs(entry - stop)
        sl_pips = risk / pip
        if sl_pips < MIN_SL_PIPS or sl_pips > MAX_SL_PIPS:
            stats.sl_invalid += 1
            continue

        allowed, _ = gate.can_publish(display, ts)
        if not allowed:
            stats.rate_limited += 1
            continue

        if direction == Direction.LONG:
            tp1 = entry + risk
            tp2 = entry + TP2_R * risk
        else:
            tp1 = entry - risk
            tp2 = entry - TP2_R * risk

        from config.sl_config import calculate_lot_size_for_symbol

        signal = align_trade_signal_direction(
            TradeSignal(
                direction=direction,
                entry=entry,
                stop_loss=stop,
                tp1=tp1,
                tp2=tp2,
                tp3=tp2,
                confidence=0.65,
                reason="Asian fake breakout + engulfing",
                lot_size=calculate_lot_size_for_symbol(200.0, display),
            )
        )
        simulated = simulator.simulate(
            signal,
            candles[entry_index + 1 :],
            entry_index=entry_index,
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


def print_report(stats: AsianFakeStats) -> None:
    start = datetime.fromisoformat(stats.period_start)
    end = datetime.fromisoformat(stats.period_end)
    days = max((end - start).total_seconds() / 86_400, 0.01)
    tp1 = sum(1 for t in stats.trades if t.tp1_hit)
    tp2 = sum(1 for t in stats.trades if t.tp2_hit)
    be_after = sum(
        1 for t in stats.trades if t.tp1_hit and not t.tp2_hit and t.result == "breakeven"
    )
    stops = sum(1 for t in stats.trades if is_full_stop_loss(t))
    total_r = sum(t.pnl_r for t in stats.trades)

    print()
    print("=== Asian Fake Breakout Backtest (XAUUSD, M15) ===")
    print(f"Період: {stats.period_start[:16]} -> {stats.period_end[:16]} (~{days:.1f} днів)")
    print(f"Угод: {len(stats.trades)} ({len(stats.trades)/days:.2f}/день)")
    if stats.trades:
        print(f"Win rate TP1: {tp1}/{len(stats.trades)} ({tp1/len(stats.trades)*100:.1f}%)")
        print(f"  TP1+TP2: {tp2} | TP1->BE: {be_after} | Стопи: {stops}")
    print(f"Total R: {total_r:+.2f}R")


def main() -> None:
    import argparse

    configure_console_encoding()
    parser = argparse.ArgumentParser()
    parser.add_argument("--candles", type=int, default=2000)
    args = parser.parse_args()

    needed = WARMUP + args.candles + 1
    provider = MarketDataProvider()
    print(f"Завантаження XAUUSD M15 x{needed}...", flush=True)
    candles = provider.get_historical_market_data("XAUUSD", "15m", needed)
    stats = run(candles, "XAUUSD", interval_min=30, max_day=4)
    print_report(stats)


if __name__ == "__main__":
    main()
