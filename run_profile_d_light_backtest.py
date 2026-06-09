"""Profile D light backtest (500 M15 candles) for XAUUSD + DJ30."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtest.engine import BacktestConfig
from config.symbols import resolve_symbol
from data import MarketDataProvider
from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import (
    LIGHT_SCAN_CANDLES,
    WARMUP,
    BacktestRunStats,
    LocalDataBacktestEngine,
    slice_m15_window,
)
from signal_generator import MIN_RR_TO_TP1, TP_STEP_R
from strategy.signal_filter import FILTER_PROFILE_D, SignalFilter, profile_symbols
from tracking.console import configure_console_encoding

H1_BUFFER = 400
H4_BUFFER = 250
SYMBOL_LOCAL_FILES = {"XAUUSD": XAUUSD_M15_30D_FILE}
BACKTEST_SYMBOLS = profile_symbols(FILTER_PROFILE_D, ("XAUUSD", "DJ30"))


@dataclass(frozen=True)
class ProfileDLightResult:
    symbol: str
    stats: BacktestRunStats
    period_days: float


def load_symbol_data(symbol: str) -> tuple[list, list, list]:
    needed_m15 = WARMUP + LIGHT_SCAN_CANDLES + 1
    local_file = SYMBOL_LOCAL_FILES.get(symbol)
    provider = MarketDataProvider()

    if local_file is not None and local_file.exists():
        m15_candles = load_candles(local_file, "15m")
        h1_candles = load_candles(local_file, "1h")
    else:
        print(
            f"  Loading {symbol} M15/H1 from {provider.data_source(symbol)}...",
            flush=True,
        )
        m15_candles = provider.get_historical_market_data(symbol, "15m", needed_m15)
        h1_candles = provider.get_historical_market_data(symbol, "1h", H1_BUFFER)

    print(f"  Loading {symbol} H4 x{H4_BUFFER}...", flush=True)
    h4_candles = provider.get_historical_market_data(symbol, "4h", H4_BUFFER)

    if len(m15_candles) < needed_m15:
        raise RuntimeError(
            f"{symbol}: need {needed_m15} M15 candles, got {len(m15_candles)}"
        )

    m15_window = slice_m15_window(m15_candles, scan_candles=LIGHT_SCAN_CANDLES)
    return m15_window, h1_candles, h4_candles


def period_days(stats: BacktestRunStats) -> float:
    if not stats.period_start or not stats.period_end:
        return LIGHT_SCAN_CANDLES * 15 / (60 * 24)
    start = datetime.fromisoformat(stats.period_start)
    end = datetime.fromisoformat(stats.period_end)
    days = (end - start).total_seconds() / 86_400
    return max(days, 15 / (60 * 24))


def run_symbol(symbol: str) -> ProfileDLightResult:
    display = resolve_symbol(symbol).display
    print(f"\n=== {display} | Profile D ===", flush=True)
    print(f"Scanning {LIGHT_SCAN_CANDLES} candles...", flush=True)

    m15_candles, h1_candles, h4_candles = load_symbol_data(display)
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
        progress_finish=progress_prefix + "Готово.",
        signal_filter=SignalFilter.from_profile(FILTER_PROFILE_D),
    )
    _, partial_stats = engine.run_comparison()
    days = period_days(partial_stats)
    return ProfileDLightResult(symbol=display, stats=partial_stats, period_days=days)


def _wr(stats: BacktestRunStats) -> str:
    if not stats.total_signals:
        return "—"
    return f"{stats.win_rate:.0f}%"


def print_report(results: list[ProfileDLightResult]) -> None:
    tp3 = MIN_RR_TO_TP1 + 2 * TP_STEP_R
    print()
    print("=== Profile D light backtest (500 M15 candles) ===")
    print(
        f"TP: {MIN_RR_TO_TP1:.1f}R / {MIN_RR_TO_TP1 + TP_STEP_R:.1f}R / {tp3:.1f}R | "
        "Partial 50/25/25 | BE at entry ≠ SL"
    )
    print(f"Symbols: {', '.join(BACKTEST_SYMBOLS)} (BTC excluded)")
    if results:
        print(f"Period: ~{results[0].period_days:.1f} days")
    print()
    print(
        f"{'Інструмент':<12} {'Угод/день':>10} {'WR':>8} "
        f"{'Стопів':>8} {'BE':>6} {'Total R':>10}"
    )
    print("-" * 62)

    total_trades = 0
    total_tp1 = 0
    total_stops = 0
    total_be = 0
    total_r = 0.0
    max_days = max((item.period_days for item in results), default=0.0)

    for item in results:
        stats = item.stats
        spd = stats.total_signals / item.period_days if item.period_days else 0.0
        print(
            f"{item.symbol:<12} {spd:>10.2f} {_wr(stats):>8} "
            f"{stats.stop_losses:>8} {stats.breakeven_exits:>6} "
            f"{stats.total_r:>+9.2f}R"
        )
        total_trades += stats.total_signals
        total_tp1 += stats.tp1_wins
        total_stops += stats.stop_losses
        total_be += stats.breakeven_exits
        total_r += stats.total_r

    combined_spd = total_trades / max_days if max_days else 0.0
    combined_wr = (total_tp1 / total_trades * 100) if total_trades else 0.0
    combined_wr_label = f"{combined_wr:.0f}%" if total_trades else "—"
    print("-" * 62)
    print(
        f"{'РАЗОМ':<12} {combined_spd:>10.2f} {combined_wr_label:>8} "
        f"{total_stops:>8} {total_be:>6} {total_r:>+9.2f}R"
    )


def main() -> None:
    configure_console_encoding()
    print(
        f"Profile D light backtest — {LIGHT_SCAN_CANDLES} M15 candles",
        flush=True,
    )

    results = [run_symbol(symbol) for symbol in BACKTEST_SYMBOLS]
    print_report(results)


if __name__ == "__main__":
    main()
