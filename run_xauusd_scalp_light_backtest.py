"""Light scalp backtest (500 M5 candles) for XAUUSD."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.base import Direction
from agents.zone_helpers import ZoneCatalog
from backtest.engine import candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import SignalGenerator, TradeSignal
from strategy.runner import (
    TREND_CANDLE_MIN,
    build_context,
    build_signal_reason,
    run_agents,
    slice_candles_as_of,
)
from strategy.scalp_mode import (
    M15_CONFIRM_TIMEFRAME,
    SCALP_TIMEFRAME,
    ScalpPublishGate,
    evaluate_scalp_direction_alignment,
    is_scalp_enabled,
    passes_scalp_confidence,
)
from tracking.console import configure_console_encoding

WARMUP = 100
LIGHT_SCAN_CANDLES = 500
M5_NEEDED = WARMUP + LIGHT_SCAN_CANDLES + 1
M15_BUFFER = 300
H1_BUFFER = 400
PROGRESS_EVERY = 100


@dataclass
class ScalpSetup:
    entry_index: int
    signal: TradeSignal


@dataclass
class ScalpBacktestStats:
    setup_candidates: int = 0
    alignment_failed: int = 0
    confidence_rejected: int = 0
    rate_limit_rejected: int = 0
    generation_failed: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    period_start: str = ""
    period_end: str = ""

    @property
    def total_signals(self) -> int:
        return len(self.trades)

    @property
    def tp1_wins(self) -> int:
        return sum(1 for trade in self.trades if trade.tp1_hit)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.tp1_wins / self.total_signals * 100

    @property
    def total_r(self) -> float:
        return sum(trade.pnl_r for trade in self.trades)


def slice_m5_window(candles: list, *, scan_candles: int = LIGHT_SCAN_CANDLES) -> list:
    needed = WARMUP + scan_candles + 1
    if len(candles) < needed:
        raise RuntimeError(f"Need {needed} M5 candles, got {len(candles)}")
    return candles[-needed:]


def scan_scalp_setups(
    *,
    symbol: str,
    m5_candles: list,
    m15_candles: list,
    h1_candles: list,
    progress_every: int = PROGRESS_EVERY,
    progress_template: str = "Оброблено {processed}/{total} M5 свічок...",
    progress_finish: str = "Сканування завершено.",
) -> tuple[list[ScalpSetup], ScalpBacktestStats]:
    display = resolve_symbol(symbol).display
    generator = SignalGenerator()
    publish_gate = ScalpPublishGate()
    zone_catalog = ZoneCatalog.from_candles(m5_candles, display)
    setups: list[ScalpSetup] = []
    stats = ScalpBacktestStats()

    scan_start = WARMUP
    scan_end = len(m5_candles) - 1
    progress = BacktestScanProgress(
        scan_start,
        scan_end,
        update_every=progress_every,
        message_template=progress_template,
        finish_message=progress_finish,
    )

    for index in range(scan_start, scan_end):
        progress.update(index)
        timestamp = candle_timestamp(m5_candles, index)
        m5_history = m5_candles[: index + 1]
        m15_history = slice_candles_as_of(m15_candles, timestamp)
        h1_history = slice_candles_as_of(h1_candles, timestamp, limit=TREND_CANDLE_MIN)

        m5_context = build_context(
            symbol=display,
            candles=m5_history,
            timeframe=SCALP_TIMEFRAME,
            timestamp=timestamp,
            h1_candles=h1_history,
        )
        m5_context["zone_catalog"] = zone_catalog
        m5_context["bar_index"] = index

        m15_context = build_context(
            symbol=display,
            candles=m15_history,
            timeframe=M15_CONFIRM_TIMEFRAME,
            timestamp=timestamp,
            h1_candles=h1_history,
        )
        if m15_history:
            m15_context["zone_catalog"] = ZoneCatalog.from_candles(m15_history, display)
            m15_context["bar_index"] = len(m15_history) - 1

        m5_results = run_agents(m5_context)
        m15_results = run_agents(m15_context)
        alignment = evaluate_scalp_direction_alignment(m5_results, m15_results)
        if alignment is None:
            stats.alignment_failed += 1
            continue

        direction, confidence = alignment
        if not passes_scalp_confidence(confidence):
            stats.confidence_rejected += 1
            continue

        allowed, _ = publish_gate.can_publish(display, timestamp)
        if not allowed:
            stats.rate_limit_rejected += 1
            continue

        generation = generator.generate_scalp(
            m5_context,
            direction,
            confidence,
            build_signal_reason(m5_results, direction),
        )
        if generation.signal is None:
            stats.generation_failed += 1
            continue

        setups.append(ScalpSetup(entry_index=index, signal=generation.signal))
        publish_gate.record(display, timestamp)

    progress.finish()
    stats.setup_candidates = len(setups)
    if scan_end > scan_start:
        stats.period_start = candle_timestamp(m5_candles, scan_start).isoformat()
        stats.period_end = candle_timestamp(m5_candles, scan_end).isoformat()
    return setups, stats


def simulate_scalp_setups(
    setups: list[ScalpSetup],
    m5_candles: list,
) -> list[SimulatedTradeResult]:
    simulator = TradeSimulator()
    trades: list[SimulatedTradeResult] = []
    open_until_index = -1

    for setup in setups:
        if setup.entry_index <= open_until_index:
            continue
        simulated = simulator.simulate(
            setup.signal,
            m5_candles[setup.entry_index + 1 :],
            entry_index=setup.entry_index,
            mode=TradeManagementMode.PARTIAL,
        )
        if simulated is None:
            continue
        trades.append(simulated)
        open_until_index = simulated.exit_index

    return trades


def run_symbol_scalp_light_backtest(
    symbol: str,
    *,
    verbose: bool = True,
    progress_every: int = PROGRESS_EVERY,
) -> ScalpBacktestStats:
    display = resolve_symbol(symbol).display
    if not is_scalp_enabled(display):
        if verbose:
            print(f"\n=== {display} SCALP (M5) — disabled ===", flush=True)
        return ScalpBacktestStats()

    provider = MarketDataProvider()

    if verbose:
        print(f"\n=== {display} SCALP (M5) ===", flush=True)
        print(f"Завантаження {display} M5 x{M5_NEEDED}...", flush=True)
    m5_candles = provider.get_historical_market_data(display, "5m", M5_NEEDED)
    if verbose:
        print(f"Завантаження {display} M15 x{M15_BUFFER}...", flush=True)
    m15_candles = provider.get_historical_market_data(display, "15m", M15_BUFFER)
    if verbose:
        print(f"Завантаження {display} H1 x{H1_BUFFER}...", flush=True)
    h1_candles = provider.get_historical_market_data(display, "1h", H1_BUFFER)

    m5_window = slice_m5_window(m5_candles, scan_candles=LIGHT_SCAN_CANDLES)
    progress_prefix = f"[{display} M5] "
    setups, stats = scan_scalp_setups(
        symbol=display,
        m5_candles=m5_window,
        m15_candles=m15_candles,
        h1_candles=h1_candles,
        progress_every=progress_every,
        progress_template=progress_prefix + "Оброблено {processed}/{total}...",
        progress_finish=progress_prefix + "Сканування завершено.",
    )
    stats.trades = simulate_scalp_setups(setups, m5_window)
    return stats


def run_xauusd_scalp_light_backtest() -> ScalpBacktestStats:
    return run_symbol_scalp_light_backtest("XAUUSD")


def print_report(stats: ScalpBacktestStats) -> None:
    print()
    print("=== XAUUSD Scalp Backtest (500 M5 candles) ===")
    print("M5 signal + M15 confirm + H1 trend | SL max 20 pips | TP1=1R TP2=2R")
    print("Filters: M5 confidence >= 60% | 1h gap | max 4/day")
    if stats.period_start and stats.period_end:
        print(f"Period: {stats.period_start} -> {stats.period_end}")
    print()
    print(f"Signal candidates:     {stats.setup_candidates}")
    print(f"Alignment rejected:    {stats.alignment_failed}")
    print(f"Confidence rejected:   {stats.confidence_rejected}")
    print(f"Rate limit rejected:   {stats.rate_limit_rejected}")
    print(f"Generation rejected:   {stats.generation_failed}")
    print()
    print(f"Trades:                {stats.total_signals}")
    print(
        f"Win rate TP1:          "
        f"{stats.tp1_wins}/{stats.total_signals} ({stats.win_rate:.1f}%)"
        if stats.total_signals
        else "Win rate TP1:          —"
    )
    print(f"Total R:               {stats.total_r:+.2f}R")


def main() -> None:
    configure_console_encoding()
    print(f"Light scalp backtest — {LIGHT_SCAN_CANDLES} M5 candles", flush=True)
    stats = run_xauusd_scalp_light_backtest()
    print_report(stats)


if __name__ == "__main__":
    main()
