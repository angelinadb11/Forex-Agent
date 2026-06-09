"""Full 30-day M15 backtest for XAUUSD, BTCUSDT, and DJ30."""

from __future__ import annotations

from dataclasses import dataclass

from backtest.engine import BacktestConfig
from config.symbols import DEFAULT_SYMBOLS, resolve_symbol
from data import MarketDataProvider
from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import (
    DAYS,
    H1_CANDLES,
    M15_CANDLES_PER_DAY,
    WARMUP,
    BacktestRunStats,
    LocalDataBacktestEngine,
)
from tracking.console import configure_console_encoding

PROGRESS_EVERY = 200
SYMBOL_LOCAL_FILES = {
    "XAUUSD": XAUUSD_M15_30D_FILE,
}
FULL_M15_NEEDED = WARMUP + DAYS * M15_CANDLES_PER_DAY + 1


@dataclass(frozen=True)
class SymbolBacktestResult:
    symbol: str
    stats: BacktestRunStats
    scan_candles: int
    data_source: str


def load_full_symbol_candles(symbol: str) -> tuple[list, list, str]:
    local_file = SYMBOL_LOCAL_FILES.get(symbol)

    if local_file is not None and local_file.exists():
        m15_candles = load_candles(local_file, "15m")
        h1_candles = load_candles(local_file, "1h")
        source = f"local:{local_file.name}"
    else:
        provider = MarketDataProvider()
        source = provider.data_source(symbol)
        print(
            f"  Loading {symbol} M15 x{FULL_M15_NEEDED}, H1 x{H1_CANDLES} "
            f"from {source}...",
            flush=True,
        )
        m15_candles = provider.get_historical_market_data(
            symbol,
            "15m",
            FULL_M15_NEEDED,
        )
        h1_candles = provider.get_historical_market_data(symbol, "1h", H1_CANDLES)

    min_required = WARMUP + M15_CANDLES_PER_DAY + 1
    if len(m15_candles) < min_required:
        raise RuntimeError(
            f"{symbol}: need at least {min_required} M15 candles, got {len(m15_candles)}"
        )

    return m15_candles, h1_candles, source


def run_symbol_backtest(symbol: str) -> SymbolBacktestResult:
    display = resolve_symbol(symbol).display
    print(f"\n=== {display} (30-day backtest) ===", flush=True)

    m15_candles, h1_candles, source = load_full_symbol_candles(display)
    scan_candles = max(0, len(m15_candles) - WARMUP - 1)
    print(f"Scanning {scan_candles} candles...", flush=True)

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
        progress_every=PROGRESS_EVERY,
        progress_template=progress_prefix + "Оброблено {processed}/{total}...",
        progress_finish=progress_prefix + "Сканування завершено.",
        label=f"{DAYS} days",
    )
    _, partial_stats = engine.run_comparison(data_file=source)
    return SymbolBacktestResult(
        symbol=display,
        stats=partial_stats,
        scan_candles=scan_candles,
        data_source=source,
    )


def _tp1_label(stats: BacktestRunStats) -> str:
    if not stats.total_signals:
        return "—"
    return f"{stats.tp1_wins}/{stats.total_signals} ({stats.win_rate:.0f}%)"


def print_symbol_detail(result: SymbolBacktestResult) -> None:
    stats = result.stats
    print()
    print(f"--- {result.symbol} ---")
    print(f"  Data:           {result.data_source}")
    print(f"  Scan candles:   {result.scan_candles}")
    if stats.period_start and stats.period_end:
        print(f"  Period:         {stats.period_start} -> {stats.period_end}")
    print(f"  Candidates:     {stats.setup_candidates}")
    print(f"  Trades:         {stats.total_signals}")
    print(f"  Win rate TP1:   {_tp1_label(stats)}")
    print(f"  Total R:        {stats.total_r:+.2f}R")
    print(f"  Avg R/trade:    {stats.avg_r_per_trade:+.2f}R")
    print(f"  Trend blocked:  {stats.trend_blocked}")
    print(f"  Other blocked:  {stats.other_filter_blocked}")


def print_summary_table(results: list[SymbolBacktestResult]) -> None:
    print()
    print("=== 30-day backtest summary (partial logic) ===")
    print("Chief Analyst: 2/4 primary agents + trend confirm, min confidence 60%")
    print()
    print(
        f"{'Symbol':<10} {'Scan':>6} {'Trades':>7} {'TP1 WR':>12} "
        f"{'Total R':>10} {'Avg R':>8}"
    )
    print("-" * 58)

    total_trades = 0
    total_tp1 = 0
    total_r = 0.0

    for result in results:
        stats = result.stats
        print(
            f"{result.symbol:<10} {result.scan_candles:>6} {stats.total_signals:>7} "
            f"{_tp1_label(stats):>12} {stats.total_r:>+9.2f}R "
            f"{stats.avg_r_per_trade:>+7.2f}R"
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
    avg_r = total_r / total_trades if total_trades else 0.0
    total_scan = sum(result.scan_candles for result in results)

    print("-" * 58)
    print(
        f"{'TOTAL':<10} {total_scan:>6} {total_trades:>7} "
        f"{combined_tp1:>12} {total_r:>+9.2f}R {avg_r:>+7.2f}R"
    )


def main() -> None:
    configure_console_encoding()
    print(
        f"Multi-symbol 30-day backtest — progress every {PROGRESS_EVERY} candles",
        flush=True,
    )
    print(f"Symbols: {', '.join(DEFAULT_SYMBOLS)}", flush=True)

    results: list[SymbolBacktestResult] = []
    for symbol in DEFAULT_SYMBOLS:
        results.append(run_symbol_backtest(symbol))
        print_symbol_detail(results[-1])

    print_summary_table(results)


if __name__ == "__main__":
    main()
