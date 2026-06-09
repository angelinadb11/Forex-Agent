"""Light backtest (500 M15 candles) for XAUUSD, BTCUSDT, and DJ30."""

from __future__ import annotations

from dataclasses import dataclass

from backtest.engine import BacktestConfig, candle_timestamp
from config.symbols import DEFAULT_SYMBOLS, resolve_symbol
from data import MarketDataProvider
from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import (
    LIGHT_SCAN_CANDLES,
    WARMUP,
    BacktestRunStats,
    LocalDataBacktestEngine,
    slice_m15_window,
)
from strategy.runner import slice_candles_as_of
from tracking.console import configure_console_encoding

H1_BUFFER = 400
H4_BUFFER = 250
SYMBOL_LOCAL_FILES = {
    "XAUUSD": XAUUSD_M15_30D_FILE,
}


@dataclass(frozen=True)
class SymbolLightResult:
    symbol: str
    stats: BacktestRunStats
    period_start: str
    period_end: str


def load_symbol_candles(symbol: str) -> tuple[list, list, list]:
    needed_m15 = WARMUP + LIGHT_SCAN_CANDLES + 1
    local_file = SYMBOL_LOCAL_FILES.get(symbol)
    provider = MarketDataProvider()

    if local_file is not None and local_file.exists():
        m15_candles = load_candles(local_file, "15m")
        h1_candles = load_candles(local_file, "1h")
    else:
        print(
            f"  Downloading {symbol} M15 x{needed_m15}, H1 x{H1_BUFFER} "
            f"from {provider.data_source(symbol)}...",
            flush=True,
        )
        m15_candles = provider.get_historical_market_data(symbol, "15m", needed_m15)
        h1_candles = provider.get_historical_market_data(symbol, "1h", H1_BUFFER)

    h4_candles = provider.get_historical_market_data(symbol, "4h", H4_BUFFER)

    if len(m15_candles) < needed_m15:
        raise RuntimeError(
            f"{symbol}: need {needed_m15} M15 candles, got {len(m15_candles)}"
        )

    m15_window = slice_m15_window(m15_candles, scan_candles=LIGHT_SCAN_CANDLES)
    return m15_window, h1_candles, h4_candles


def run_symbol_light_backtest(symbol: str) -> SymbolLightResult:
    display = resolve_symbol(symbol).display
    print(f"\n=== {display} ===", flush=True)
    print(f"Scanning {LIGHT_SCAN_CANDLES} candles...", flush=True)

    m15_candles, h1_candles, h4_candles = load_symbol_candles(display)
    progress_prefix = f"[{display}] "
    engine = LocalDataBacktestEngine(
        BacktestConfig(
            symbol=display,
            timeframe="15m",
            total_candles=len(m15_candles),
            warmup_candles=WARMUP,
        ),
        m15_candles=m15_candles,
        h1_candles=h1_candles,
        h4_candles=h4_candles,
        progress_every=100,
        progress_template=progress_prefix + "Оброблено {processed}/{total}...",
        progress_finish=progress_prefix + "Сканування завершено.",
        label=f"light {LIGHT_SCAN_CANDLES} candles",
    )
    _, partial_stats = engine.run_comparison()
    period_start = partial_stats.period_start
    period_end = partial_stats.period_end
    return SymbolLightResult(
        symbol=display,
        stats=partial_stats,
        period_start=period_start,
        period_end=period_end,
    )


def print_symbol_summary(result: SymbolLightResult) -> None:
    stats = result.stats
    win_rate = stats.win_rate if stats.total_signals else 0.0
    print(f"  Period:       {result.period_start} -> {result.period_end}")
    print(f"  Trades:       {stats.total_signals}")
    print(f"  Win rate TP1: {stats.tp1_wins}/{stats.total_signals} ({win_rate:.1f}%)")
    print(f"  Total R:      {stats.total_r:+.2f}R")


def print_combined_report(results: list[SymbolLightResult]) -> None:
    print()
    print("=== Light backtest summary (500 candles, partial logic) ===")
    print(f"Chief Analyst: 2/4 agents + H1/H4 trend + OB/FVG entry, min 60%")
    print()
    print(f"{'Symbol':<10} {'Trades':>7} {'TP1 WR':>10} {'Total R':>10}")
    print("-" * 42)

    total_trades = 0
    total_tp1 = 0
    total_r = 0.0

    for result in results:
        stats = result.stats
        win_rate = stats.win_rate if stats.total_signals else 0.0
        tp1_label = (
            f"{stats.tp1_wins}/{stats.total_signals} ({win_rate:.0f}%)"
            if stats.total_signals
            else "—"
        )
        print(
            f"{result.symbol:<10} {stats.total_signals:>7} {tp1_label:>10} "
            f"{stats.total_r:>+9.2f}R"
        )
        total_trades += stats.total_signals
        total_tp1 += stats.tp1_wins
        total_r += stats.total_r

    combined_wr = (total_tp1 / total_trades * 100) if total_trades else 0.0
    combined_tp1 = (
        f"{total_tp1}/{total_trades} ({combined_wr:.0f}%)"
        if total_trades
        else "—"
    )
    print("-" * 42)
    print(
        f"{'TOTAL':<10} {total_trades:>7} {combined_tp1:>10} {total_r:>+9.2f}R"
    )


def main() -> None:
    configure_console_encoding()
    print(
        f"Light multi-symbol backtest — {LIGHT_SCAN_CANDLES} M15 candles per symbol",
        flush=True,
    )
    print(f"Symbols: {', '.join(DEFAULT_SYMBOLS)}", flush=True)

    results: list[SymbolLightResult] = []
    for symbol in DEFAULT_SYMBOLS:
        results.append(run_symbol_light_backtest(symbol))
        print_symbol_summary(results[-1])

    print_combined_report(results)


if __name__ == "__main__":
    main()
