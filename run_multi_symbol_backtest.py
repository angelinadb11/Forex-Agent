"""Profile D light backtest (500 M15) across all candidate symbols."""

from __future__ import annotations

from run_profile_d_light_backtest import _wr, period_days, run_symbol
from tracking.console import configure_console_encoding

SYMBOLS = ("XAUUSD", "DJ30", "NAS100", "EURUSD", "GBPUSD")


def main() -> None:
    configure_console_encoding()
    print("=== Profile D — мульти-символьний бектест (500 M15) ===\n")

    results = []
    for symbol in SYMBOLS:
        try:
            results.append(run_symbol(symbol))
        except Exception as exc:
            print(f"  {symbol}: FAILED — {type(exc).__name__}: {exc}")

    print()
    print(
        f"{'Інструмент':<12} {'Угод':>6} {'Угод/день':>10} {'WR':>8} "
        f"{'Стопів':>8} {'BE':>6} {'Total R':>10}"
    )
    print("-" * 70)
    total_r = 0.0
    total_trades = 0
    for result in results:
        stats = result.stats
        days = result.period_days
        spd = stats.total_signals / days if days else 0.0
        total_r += stats.total_r
        total_trades += stats.total_signals
        print(
            f"{result.symbol:<12} {stats.total_signals:>6} {spd:>10.2f} "
            f"{_wr(stats):>8} {stats.stop_losses:>8} "
            f"{stats.breakeven_exits:>6} {stats.total_r:>+9.2f}R"
        )
    print("-" * 70)
    print(f"{'РАЗОМ':<12} {total_trades:>6} {'':>10} {'':>8} {'':>8} {'':>6} {total_r:>+9.2f}R")


if __name__ == "__main__":
    main()
