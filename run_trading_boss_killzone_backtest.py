"""30-day backtest for Trading Boss Killzone (XAUUSD, London/NY windows only)."""

from __future__ import annotations

import argparse
import bisect
from dataclasses import dataclass, field

from agents.base import Direction
from backtest.progress import BacktestScanProgress
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal, calculate_atr
from strategy.signal_filter import SignalFilter
from strategy.sweep_fvg_scalp import build_timestamps
from strategy.trading_boss_killzone import (
    KillzoneSetup,
    active_killzone,
    analyze_bias,
    build_killzone_setup,
    build_killzone_signal,
    compute_killzone_decision,
    detect_liquidity_sweep,
    evaluate_killzone_filter,
    find_structure_setup,
    get_killzone_windows,
    is_fresh_sweep,
    KillzoneWindowGate,
    resolve_killzone_frequency,
    resolve_killzone_profile,
    run_killzone_agents,
)
from tracking.console import configure_console_encoding
from tracking.trade_pnl import pip_size_for_symbol

DAYS = 30
WARMUP_M1 = 600
M1_PER_DAY = 24 * 60
PROGRESS_EVERY = 500
FILL_WINDOW = 10


@dataclass
class KillzoneBacktestStats:
    bars_scanned: int = 0
    no_sweep: int = 0
    stale_sweep: int = 0
    no_structure: int = 0
    no_setup: int = 0
    filter_blocked: int = 0
    neutral_decision: int = 0
    not_filled: int = 0
    slot_busy: int = 0
    killzone_slot_used: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    period_start: str = ""
    period_end: str = ""
    profile_name: str = "precision"
    frequency_label: str = "balanced"

    @property
    def total_signals(self) -> int:
        return len(self.trades)

    @property
    def tp1_wins(self) -> int:
        return sum(1 for trade in self.trades if trade.tp1_hit)

    @property
    def total_r(self) -> float:
        return sum(trade.pnl_r for trade in self.trades)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.tp1_wins / self.total_signals * 100.0


def _open_times_ms(candles: list[dict]) -> list[float]:
    return [float(candle.get("open_time", 0)) for candle in candles]


def _slice_end_index(open_times: list[float], as_of_ms: float) -> int:
    return bisect.bisect_right(open_times, as_of_ms) - 1


def _slice_as_of(
    candles: list[dict],
    open_times: list[float],
    end_index: int,
    *,
    limit: int,
) -> list[dict]:
    if end_index < 0:
        return []
    start = max(0, end_index + 1 - limit)
    return candles[start : end_index + 1]


def _candles_to_dataframe(candles: list[dict]) -> object:
    from agents.smc_agent import _candles_to_dataframe as smc_df

    return smc_df({"candles": candles})


def _entry_filled(candle: dict, entry: float) -> bool:
    return float(candle["low"]) <= entry <= float(candle["high"])


def _find_fill_index(
    candles: list[dict],
    start: int,
    entry: float,
    window: int,
) -> int | None:
    end = min(len(candles), start + window)
    for index in range(start, end):
        if _entry_filled(candles[index], entry):
            return index
    return None


def evaluate_bar(
    *,
    symbol: str,
    profile,
    signal_filter: SignalFilter,
    m1_candles: list[dict],
    m5_candles: list[dict],
    h1_candles: list[dict],
    h4_candles: list[dict],
    m1_open_times: list[float],
    m5_open_times: list[float],
    h1_open_times: list[float],
    h4_open_times: list[float],
    m1_timestamps: list,
    m5_timestamps: list,
    index: int,
    ts,
) -> tuple[TradeSignal | None, str]:
    as_of_ms = m1_open_times[index]
    sweep_limit = max(600, profile.sweep_lookback_bars + 50)
    htf_limit = 500

    m1_end = index
    m5_end = _slice_end_index(m5_open_times, as_of_ms)
    h1_end = _slice_end_index(h1_open_times, as_of_ms)
    h4_end = _slice_end_index(h4_open_times, as_of_ms)

    sweep_candles = _slice_as_of(m1_candles, m1_open_times, m1_end, limit=sweep_limit)
    htf_candles = _slice_as_of(m5_candles, m5_open_times, m5_end, limit=htf_limit)
    h1_slice = _slice_as_of(h1_candles, h1_open_times, h1_end, limit=250)
    h4_slice = _slice_as_of(h4_candles, h4_open_times, h4_end, limit=250)

    if len(sweep_candles) < 30 or len(h1_slice) < 20:
        return None, "insufficient_data"

    bias = analyze_bias(h1_slice, h4_slice)
    sweep_start = max(0, m1_end + 1 - len(sweep_candles))
    sweep_timestamps = m1_timestamps[sweep_start : m1_end + 1]
    htf_start = max(0, m5_end + 1 - len(htf_candles))
    htf_timestamps = m5_timestamps[htf_start : m5_end + 1]
    atr = calculate_atr(_candles_to_dataframe(sweep_candles), period=14)

    sweep = detect_liquidity_sweep(
        sweep_candles,
        sweep_timestamps,
        htf_candles=htf_candles,
        htf_timestamps=htf_timestamps,
        bias=bias,
        atr=atr,
        profile=profile,
        recent_bars_only=8,
    )
    if sweep is None:
        return None, "no_sweep"
    if not is_fresh_sweep(sweep, len(sweep_candles)):
        return None, "stale_sweep"

    structure = find_structure_setup(sweep_candles, sweep)
    if structure is None:
        return None, "no_structure"

    setup = build_killzone_setup(
        sweep=sweep,
        structure=structure,
        sweep_candles=sweep_candles,
        sweep_timestamps=sweep_timestamps,
        atr=atr,
        symbol=symbol,
        profile=profile,
    )
    if setup is None:
        return None, "no_setup"

    setup = KillzoneSetup(
        direction=setup.direction,
        entry=setup.entry,
        stop_loss=setup.stop_loss,
        tp1=setup.tp1,
        tp2=setup.tp2,
        tp3=setup.tp3,
        confidence=setup.confidence,
        reason=setup.reason,
        sweep=setup.sweep,
        structure=setup.structure,
        bias=bias,
    )

    results = run_killzone_agents(
        bias=bias,
        sweep=sweep,
        structure=structure,
        timestamp=ts,
        setup=setup,
    )
    direction, confidence = compute_killzone_decision(
        results,
        bias=bias,
        in_killzone=True,
    )
    if direction == Direction.NEUTRAL:
        return None, "neutral"

    filter_result = evaluate_killzone_filter(
        signal_filter=signal_filter,
        results=results,
        direction=direction,
        confidence=confidence,
        symbol=symbol,
        timestamp=ts,
    )
    if not filter_result.approved:
        return None, "filter_blocked"

    signal = build_killzone_signal(
        setup,
        symbol,
        confidence=filter_result.confidence,
    )
    return signal, "ok"


def run_backtest(
    *,
    symbol: str,
    m1_candles: list[dict],
    m5_candles: list[dict],
    h1_candles: list[dict],
    h4_candles: list[dict],
    profile_name: str | None = None,
    frequency_name: str | None = None,
    days: int = DAYS,
    warmup: int = WARMUP_M1,
    step: int = 5,
) -> KillzoneBacktestStats:
    display = resolve_symbol(symbol).display
    profile = resolve_killzone_profile(profile_name)
    frequency = resolve_killzone_frequency(frequency_name)
    signal_filter = SignalFilter(news_gate=None)
    simulator = TradeSimulator()
    stats = KillzoneBacktestStats(
        profile_name=profile.name,
        frequency_label=frequency.label,
    )
    killzone_gate = KillzoneWindowGate(
        max_signals_per_window=frequency.max_signals_per_window,
    )

    m1_timestamps = build_timestamps(m1_candles)
    m5_timestamps = build_timestamps(m5_candles)
    m1_open_times = _open_times_ms(m1_candles)
    m5_open_times = _open_times_ms(m5_candles)
    h1_open_times = _open_times_ms(h1_candles)
    h4_open_times = _open_times_ms(h4_candles)

    test_bars = days * M1_PER_DAY
    scan_start = max(warmup, len(m1_candles) - test_bars - 1)
    scan_end = len(m1_candles) - 2
    killzone_windows = get_killzone_windows(include_asian=frequency.include_asian)

    def _in_killzone(ts) -> bool:
        from strategy.trading_boss_killzone import _window_contains

        return any(_window_contains(ts, window) for window in killzone_windows)

    killzone_indices = [
        index
        for index in range(scan_start, scan_end + 1, max(1, step))
        if _in_killzone(m1_timestamps[index])
    ]
    print(f"Killzone bars to scan: {len(killzone_indices)} (step={max(1, step)} M1)", flush=True)
    open_until = -1

    progress = BacktestScanProgress(
        0,
        len(killzone_indices),
        update_every=PROGRESS_EVERY,
        message_template="Killzone bars {processed}/{total}...",
        finish_message="Killzone scan complete.",
    )

    for step, index in enumerate(killzone_indices):
        progress.update(step)
        ts = m1_timestamps[index]
        if index <= open_until:
            stats.slot_busy += 1
            continue

        slot_ok, _ = killzone_gate.can_take(ts)
        if not slot_ok:
            stats.killzone_slot_used += 1
            continue

        stats.bars_scanned += 1
        signal, reason = evaluate_bar(
            symbol=display,
            profile=profile,
            signal_filter=signal_filter,
            m1_candles=m1_candles,
            m5_candles=m5_candles,
            h1_candles=h1_candles,
            h4_candles=h4_candles,
            m1_open_times=m1_open_times,
            m5_open_times=m5_open_times,
            h1_open_times=h1_open_times,
            h4_open_times=h4_open_times,
            m1_timestamps=m1_timestamps,
            m5_timestamps=m5_timestamps,
            index=index,
            ts=ts,
        )
        if signal is None:
            if reason == "no_sweep":
                stats.no_sweep += 1
            elif reason == "stale_sweep":
                stats.stale_sweep += 1
            elif reason == "no_structure":
                stats.no_structure += 1
            elif reason == "no_setup":
                stats.no_setup += 1
            elif reason == "filter_blocked":
                stats.filter_blocked += 1
            elif reason == "neutral":
                stats.neutral_decision += 1
            continue

        fill_index = _find_fill_index(m1_candles, index, signal.entry, FILL_WINDOW)
        if fill_index is None:
            stats.not_filled += 1
            continue

        simulated = simulator.simulate(
            signal,
            m1_candles[fill_index + 1 :],
            entry_index=fill_index,
            mode=TradeManagementMode.PARTIAL,
            symbol=display,
        )
        if simulated is None:
            continue

        stats.trades.append(simulated)
        open_until = simulated.exit_index
        killzone_gate.record(ts)

    progress.finish()
    stats.period_start = m1_timestamps[scan_start].isoformat()
    stats.period_end = m1_timestamps[scan_end].isoformat()
    return stats


def download_dataset(
    provider: MarketDataProvider,
    *,
    days: int,
    warmup: int,
) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    m1_needed = days * M1_PER_DAY + warmup + 10
    m5_needed = days * 288 + warmup // 5 + 50
    h1_needed = days * 24 + 300
    h4_needed = days * 6 + 100

    print(f"Завантаження XAUUSD M1 x{m1_needed}...", flush=True)
    m1 = provider.get_historical_market_data("XAUUSD", "1m", m1_needed)
    print(f"Завантаження XAUUSD M5 x{m5_needed}...", flush=True)
    m5 = provider.get_historical_market_data("XAUUSD", "5m", m5_needed)
    print(f"Завантаження XAUUSD H1 x{h1_needed}...", flush=True)
    h1 = provider.get_historical_market_data("XAUUSD", "1h", h1_needed)
    print(f"Завантаження XAUUSD H4 x{h4_needed}...", flush=True)
    h4 = provider.get_historical_market_data("XAUUSD", "4h", h4_needed)
    return m1, m5, h1, h4


def print_report(stats: KillzoneBacktestStats, *, days: int, frequency: str = "balanced") -> None:
    pip = pip_size_for_symbol("XAUUSD") or 0.1
    print()
    print(f"=== Trading Boss Killzone Backtest (XAUUSD, {days} days) ===")
    print(f"Profile: {stats.profile_name} | freq={stats.frequency_label} | partial TP 50/25/25")
    print(f"Data: Binance XAUUSDT | period {stats.period_start[:16]} -> {stats.period_end[:16]}")
    print()
    print(f"Killzone M1 bars scanned:   {stats.bars_scanned}")
    print(f"No fresh sweep:             {stats.no_sweep}")
    print(f"Stale sweep (dedup):        {stats.stale_sweep}")
    print(f"No structure:               {stats.no_structure}")
    print(f"Setup rejected (SL/RR):     {stats.no_setup}")
    print(f"Filter blocked:             {stats.filter_blocked}")
    print(f"Neutral decision:           {stats.neutral_decision}")
    print(f"Limit not filled:           {stats.not_filled}")
    print(f"Slot busy (overlap):        {stats.slot_busy}")
    print(f"Killzone slot used:         {stats.killzone_slot_used}")
    print()
    print(f"Trades executed:            {stats.total_signals}")
    if stats.total_signals:
        avg_sl = sum(abs(t.entry - t.stop_loss) / pip for t in stats.trades) / stats.total_signals
        print(
            f"TP1 hit:                    {stats.tp1_wins}/{stats.total_signals} "
            f"({stats.win_rate:.1f}%)"
        )
        print(f"Total R:                    {stats.total_r:+.2f}R")
        print(f"Avg R/trade:                {stats.total_r / stats.total_signals:+.2f}R")
        print(f"Avg SL:                     {avg_sl:.1f} pips")
        print()
        print("Trades:")
        for trade in stats.trades:
            sl_pips = abs(trade.entry - trade.stop_loss) / pip
            print(
                f"  {trade.direction.upper()} | SL {sl_pips:.1f}p | "
                f"{trade.result} | {trade.pnl_r:+.2f}R | {trade.reason[:72]}"
            )
    else:
        print("Total R:                    +0.00R")


def main() -> None:
    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Trading Boss Killzone 30-day backtest")
    parser.add_argument("--days", type=int, default=DAYS)
    parser.add_argument("--profile", default="precision", choices=["precision", "standard"])
    parser.add_argument(
        "--frequency",
        default="balanced",
        choices=["balanced", "daily"],
        help="balanced=1 signal/window; daily=2/window + Asian killzone",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=5,
        help="Scan every N M1 bars inside Killzone (5 ≈ production scan cadence)",
    )
    args = parser.parse_args()

    provider = MarketDataProvider()
    m1, m5, h1, h4 = download_dataset(provider, days=args.days, warmup=WARMUP_M1)
    print(
        f"Отримано M1={len(m1)} M5={len(m5)} H1={len(h1)} H4={len(h4)}",
        flush=True,
    )

    stats = run_backtest(
        symbol="XAUUSD",
        m1_candles=m1,
        m5_candles=m5,
        h1_candles=h1,
        h4_candles=h4,
        profile_name=args.profile,
        frequency_name=args.frequency,
        days=args.days,
        warmup=WARMUP_M1,
        step=args.step,
    )
    print_report(stats, days=args.days, frequency=args.frequency)


if __name__ == "__main__":
    main()
