"""Compare filter variants A, B, and D on 500 M15 candles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from backtest.engine import BacktestConfig
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
from strategy.signal_filter import (
    FILTER_PROFILE_A,
    FILTER_PROFILE_B,
    FILTER_PROFILE_D,
    FilterProfile,
    SignalFilter,
    profile_symbols,
)
from tracking.console import configure_console_encoding

H1_BUFFER = 400
H4_BUFFER = 250
SYMBOL_LOCAL_FILES = {"XAUUSD": XAUUSD_M15_30D_FILE}

COMPARE_PROFILES = (
    ("A", FILTER_PROFILE_A),
    ("B", FILTER_PROFILE_B),
    ("D", FILTER_PROFILE_D),
)


@dataclass(frozen=True)
class VariantRunResult:
    profile: FilterProfile
    symbol: str
    stats: BacktestRunStats
    period_days: float

    @property
    def signals_per_day(self) -> float:
        if self.period_days <= 0:
            return 0.0
        return self.stats.total_signals / self.period_days


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


def run_variant(
    symbol: str,
    profile: FilterProfile,
    *,
    m15_candles: list,
    h1_candles: list,
    h4_candles: list,
) -> VariantRunResult:
    display = resolve_symbol(symbol).display
    progress_prefix = f"[{display}|{profile.label}] "
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
        signal_filter=SignalFilter.from_profile(profile),
    )
    _, partial_stats = engine.run_comparison()
    days = period_days(partial_stats)
    return VariantRunResult(
        profile=profile,
        symbol=display,
        stats=partial_stats,
        period_days=days,
    )


def _wr(stats: BacktestRunStats) -> str:
    if not stats.total_signals:
        return "—"
    return f"{stats.tp1_wins}/{stats.total_signals} ({stats.win_rate:.0f}%)"


def aggregate(results: list[VariantRunResult]) -> tuple[int, int, float, float, float]:
    trades = sum(item.stats.total_signals for item in results)
    wins = sum(item.stats.tp1_wins for item in results)
    total_r = sum(item.stats.total_r for item in results)
    days = max(item.period_days for item in results) if results else 0.0
    wr = (wins / trades * 100) if trades else 0.0
    spd = (trades / days) if days else 0.0
    return trades, wins, wr, total_r, spd


def print_comparison(
    results_by_label: dict[str, list[VariantRunResult]],
) -> None:
    print()
    print("=== A / B / D backtest (500 M15 candles) ===")
    sample = next(iter(results_by_label.values()))
    print(f"Період: ~{sample[0].period_days:.1f} днів")
    for label, profile in COMPARE_PROFILES:
        symbols = profile_symbols(profile, DEFAULT_SYMBOLS)
        disabled = ", ".join(sorted(profile.disabled_symbols)) or "—"
        print(
            f"{label}: {profile.description} | symbols: {', '.join(symbols)} "
            f"| disabled: {disabled}"
        )
    print()
    print(f"{'Варіант':<10} {'Сигналів/день':>14} {'WR':>14} {'Total R':>10}")
    print("-" * 52)

    for label, _profile in COMPARE_PROFILES:
        results = results_by_label[label]
        trades, wins, wr, total_r, spd = aggregate(results)
        wr_label = f"{wins}/{trades} ({wr:.0f}%)" if trades else "—"
        print(f"{label:<10} {spd:>14.2f} {wr_label:>14} {total_r:>+9.2f}R")

    print()
    print("Per symbol (signals/day | WR):")
    print(f"{'Symbol':<10} {'A/day':>8} {'A WR':>12} {'B/day':>8} {'B WR':>12} {'D/day':>8} {'D WR':>12}")
    print("-" * 74)
    for symbol in DEFAULT_SYMBOLS:
        display = resolve_symbol(symbol).display
        cells: list[str] = [f"{display:<10}"]
        for label, profile in COMPARE_PROFILES:
            if display in profile.disabled_symbols:
                cells.extend([f"{'—':>8}", f"{'disabled':>12}"])
                continue
            item = next(result for result in results_by_label[label] if result.symbol == display)
            cells.extend([f"{item.signals_per_day:>8.2f}", f"{_wr(item.stats):>12}"])
        print(" ".join(cells))


def main() -> None:
    configure_console_encoding()
    print("A / B / D filter comparison — 500 M15 candles per symbol", flush=True)
    print(f"Universe: {', '.join(DEFAULT_SYMBOLS)}", flush=True)

    results_by_label: dict[str, list[VariantRunResult]] = {
        label: [] for label, _ in COMPARE_PROFILES
    }

    for symbol in DEFAULT_SYMBOLS:
        display = resolve_symbol(symbol).display
        print(f"\n=== {display} ===", flush=True)
        m15, h1, h4 = load_symbol_data(display)

        for label, profile in COMPARE_PROFILES:
            if display in profile.disabled_symbols:
                print(f"  Skipping variant {label} (disabled for {display})", flush=True)
                continue

            print(f"  Running variant {label}...", flush=True)
            result = run_variant(display, profile, m15_candles=m15, h1_candles=h1, h4_candles=h4)
            results_by_label[label].append(result)
            print(
                f"  {label}: signals={result.stats.total_signals}, "
                f"{result.signals_per_day:.2f}/day, WR={_wr(result.stats)}, "
                f"R={result.stats.total_r:+.2f}"
            )

    print_comparison(results_by_label)


if __name__ == "__main__":
    main()
