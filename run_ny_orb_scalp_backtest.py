"""NY Open Range Breakout scalp backtest (XAUUSD, M1).

Range: first 5 minutes of NY cash open (13:30-13:34 UTC, EDT).
Breakout candle: body >= 0.8 * ATR(5) and body >= 60% of candle range.
Entry at breakout close, stop at opposite range edge, TP1=1R / TP2=2R partial.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timezone

from agents.base import Direction
from agents.session_agent import is_london_or_new_york_session
from backtest.engine import candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, align_trade_signal_direction, calculate_atr
from strategy.scalp_mode import ScalpPublishGate
from tracking.console import configure_console_encoding
from tracking.trade_outcome import is_full_stop_loss
from tracking.trade_pnl import pip_size_for_symbol

WARMUP = 200
PROGRESS_EVERY = 2000
NY_OPEN = time(13, 30)
NY_RANGE_END = time(13, 35)
ATR_PERIOD = 5
BODY_ATR_MULT = 0.8
MIN_BODY_RATIO = 0.60
TP2_R = 2.0


@dataclass
class OrbStats:
    no_range: int = 0
    no_breakout: int = 0
    weak_body: int = 0
    off_session: int = 0
    rate_limited: int = 0
    slot_busy: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    period_start: str = ""
    period_end: str = ""


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def ny_range_for_day(
    candles: list,
    timestamps: list[datetime],
    day,
) -> tuple[float, float, int] | None:
    """Return (high, low, last_range_index) for NY open range on ``day``."""
    highs: list[float] = []
    lows: list[float] = []
    last_index = -1
    for index, ts in enumerate(timestamps):
        if ts.date() != day:
            continue
        t = ts.time()
        if NY_OPEN <= t < NY_RANGE_END:
            highs.append(candles[index]["high"])
            lows.append(candles[index]["low"])
            last_index = index
    if not highs:
        return None
    return max(highs), min(lows), last_index


def is_valid_breakout_candle(candle: dict, atr: float) -> bool:
    body = abs(candle["close"] - candle["open"])
    span = candle["high"] - candle["low"]
    if span <= 0:
        return False
    return body >= BODY_ATR_MULT * atr and body / span >= MIN_BODY_RATIO


def run(candles: list, symbol: str, *, interval_min: int, max_day: int) -> OrbStats:
    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    timestamps = [_utc(candle_timestamp(candles, i)) for i in range(len(candles))]
    gate = ScalpPublishGate(min_interval_seconds=interval_min * 60, max_signals_per_day=max_day)
    simulator = TradeSimulator()
    stats = OrbStats()
    open_until = -1
    traded_days: set = set()

    scan_start = WARMUP
    scan_end = len(candles) - 1
    progress = BacktestScanProgress(
        scan_start,
        scan_end,
        update_every=PROGRESS_EVERY,
        message_template="Оброблено {processed}/{total} M1 свічок...",
        finish_message="Сканування завершено.",
    )

    for index in range(scan_start, scan_end):
        progress.update(index)
        ts = timestamps[index]
        if not is_london_or_new_york_session(ts):
            continue
        if ts.time() < NY_RANGE_END:
            continue

        ny = ny_range_for_day(candles, timestamps, ts.date())
        if ny is None or index <= ny[2]:
            stats.no_range += 1
            continue
        if ts.date() in traded_days:
            continue

        range_high, range_low, range_end_index = ny
        if index <= open_until:
            stats.slot_busy += 1
            continue

        import pandas as pd

        history = candles[max(0, index - 50) : index + 1]
        df = pd.DataFrame(history)
        try:
            atr = calculate_atr(df, period=ATR_PERIOD)
        except Exception:
            continue

        candle = candles[index]
        direction: Direction | None = None
        if candle["close"] > range_high and is_valid_breakout_candle(candle, atr):
            direction = Direction.LONG
            entry = candle["close"]
            stop = range_low
        elif candle["close"] < range_low and is_valid_breakout_candle(candle, atr):
            direction = Direction.SHORT
            entry = candle["close"]
            stop = range_high
        else:
            stats.weak_body += 1
            continue

        risk = abs(entry - stop)
        if risk <= 0 or risk / pip > 80:
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
                reason=f"NY ORB {direction.value.upper()} breakout",
                lot_size=calculate_lot_size_for_symbol(200.0, display),
            )
        )
        simulated = simulator.simulate(
            signal,
            candles[index + 1 :],
            entry_index=index,
            mode=TradeManagementMode.PARTIAL,
        )
        if simulated is None:
            continue

        stats.trades.append(simulated)
        gate.record(display, ts)
        traded_days.add(ts.date())
        open_until = simulated.exit_index

    progress.finish()
    stats.period_start = timestamps[scan_start].isoformat()
    stats.period_end = timestamps[scan_end].isoformat()
    return stats


def print_report(stats: OrbStats) -> None:
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
    print("=== NY ORB Scalp Backtest (XAUUSD, M1) ===")
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
    parser.add_argument("--candles", type=int, default=25000)
    args = parser.parse_args()

    needed = WARMUP + args.candles + 1
    provider = MarketDataProvider()
    print(f"Завантаження XAUUSD M1 x{needed}...", flush=True)
    candles = provider.get_historical_market_data("XAUUSD", "1m", needed)
    stats = run(candles, "XAUUSD", interval_min=5, max_day=2)
    print_report(stats)


if __name__ == "__main__":
    main()
