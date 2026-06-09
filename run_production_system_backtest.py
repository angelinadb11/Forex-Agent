"""Backtest of the LIVE production configuration.

Mirrors main.py: Profile D SignalFilter (zone cluster, RSI gate, SMC
conflict block, H4/H1 alignment, entry zone, London/NY session gate),
partial 50/25/25 management with near-TP1 BE and the M15 reversal block,
across production symbols (BTCUSDT disabled by Profile D).
News gate is omitted (no historical news data).
"""

from __future__ import annotations

from datetime import datetime

from backtest.engine import BacktestConfig
from backtest.simulator import TradeManagementMode
from config.settings import load_settings
from config.symbols import resolve_symbol
from run_profile_d_light_backtest import load_symbol_data
from run_xauusd_30d_backtest import (
    LIGHT_SCAN_CANDLES,
    WARMUP,
    BacktestRunStats,
    LocalDataBacktestEngine,
    candle_timestamp,
)
from strategy.signal_filter import FILTER_PROFILE_D, SignalFilter, profile_symbols
from tracking.console import configure_console_encoding


def run_symbol(
    symbol: str,
    *,
    london_ny_session_symbols: frozenset[str],
    session_confidence_symbols: frozenset[str],
) -> tuple[BacktestRunStats, float]:
    display = resolve_symbol(symbol).display
    print(f"\n=== {display} | діюча конфігурація ===", flush=True)

    m15_candles, h1_candles, h4_candles = load_symbol_data(display)
    signal_filter = SignalFilter.from_profile(
        FILTER_PROFILE_D,
        london_ny_session_symbols=london_ny_session_symbols,
        session_confidence_symbols=session_confidence_symbols,
    )
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
        progress_template=f"[{display}] Оброблено {{processed}}/{{total}}...",
        progress_finish=f"[{display}] Готово.",
        signal_filter=signal_filter,
    )

    setups, scan_stats = engine._scan_candles(m15_candles)
    stats = engine._simulate_setups(
        setups,
        m15_candles,
        TradeManagementMode.PARTIAL_NEAR_TP1_BE,
        enable_m15_reversal_block=True,
    )
    stats.setup_candidates = len(setups)
    stats.neutral_decisions = scan_stats.neutral_decisions
    stats.trend_blocked = scan_stats.trend_blocked
    stats.other_filter_blocked = scan_stats.other_filter_blocked
    stats.rr_blocked = scan_stats.rr_blocked
    stats.sl_blocked = scan_stats.sl_blocked

    days = LIGHT_SCAN_CANDLES * 15 / (60 * 24)
    if len(m15_candles) > WARMUP:
        start = candle_timestamp(m15_candles, WARMUP)
        end = candle_timestamp(m15_candles, len(m15_candles) - 1)
        days = max((end - start).total_seconds() / 86_400, 15 / (60 * 24))
    return stats, days


def main() -> None:
    configure_console_encoding()
    settings = load_settings()
    symbols = profile_symbols(FILTER_PROFILE_D, settings.symbols)

    print("=== Бектест діючої продакшн-системи (500 M15) ===")
    print(f"Символи: {', '.join(symbols)} (Profile D, BTCUSDT вимкнений)")
    print(
        "Фільтр: Profile D (zone cluster + RSI gate + SMC conflict + H4/H1 + зона входу"
        + (
            f" + сесії London/NY для {', '.join(sorted(settings.london_ny_session_symbols))}"
            if settings.london_ny_session_symbols
            else ""
        )
        + ")"
    )
    print("Менеджмент: partial 50/25/25 + near-TP1 BE + M15-блок\n")

    results: list[tuple[str, BacktestRunStats, float]] = []
    for symbol in symbols:
        try:
            stats, days = run_symbol(
                symbol,
                london_ny_session_symbols=settings.london_ny_session_symbols,
                session_confidence_symbols=settings.session_confidence_symbols,
            )
            results.append((symbol, stats, days))
        except Exception as exc:
            print(f"  {symbol}: FAILED — {type(exc).__name__}: {exc}")

    print()
    print(
        f"{'Інструмент':<12} {'Угод':>6} {'Угод/день':>10} {'WR':>8} "
        f"{'Стопів':>8} {'BE':>6} {'M15 блок':>9} {'Total R':>10}"
    )
    print("-" * 78)
    total_r = 0.0
    total_trades = 0
    for symbol, stats, days in results:
        spd = stats.total_signals / days if days else 0.0
        wr = f"{stats.win_rate:.0f}%" if stats.total_signals else "—"
        total_r += stats.total_r
        total_trades += stats.total_signals
        print(
            f"{symbol:<12} {stats.total_signals:>6} {spd:>10.2f} {wr:>8} "
            f"{stats.stop_losses:>8} {stats.breakeven_exits:>6} "
            f"{stats.m15_reversal_blocked:>9} {stats.total_r:>+9.2f}R"
        )
    print("-" * 78)
    print(
        f"{'РАЗОМ':<12} {total_trades:>6} {'':>10} {'':>8} {'':>8} {'':>6} "
        f"{'':>9} {total_r:>+9.2f}R"
    )


if __name__ == "__main__":
    main()
