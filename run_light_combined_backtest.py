"""Light backtest (500 candles) — M15 signals + M5 scalp (XAUUSD only)."""

from __future__ import annotations

from dataclasses import dataclass

from config.symbols import DEFAULT_SYMBOLS
from run_light_multi_backtest import run_symbol_light_backtest
from run_xauusd_scalp_light_backtest import run_symbol_scalp_light_backtest
from tracking.console import configure_console_encoding


@dataclass(frozen=True)
class CombinedSymbolResult:
    symbol: str
    m15_trades: int
    m15_tp1_wins: int
    m15_wr: float
    m15_r: float
    m5_trades: int
    m5_tp1_wins: int
    m5_wr: float
    m5_r: float

    @property
    def total_r(self) -> float:
        return self.m15_r + self.m5_r


def _wr_label(wins: int, trades: int, win_rate: float) -> str:
    if trades == 0:
        return "—"
    return f"{wins}/{trades} ({win_rate:.0f}%)"


def run_combined_backtest() -> list[CombinedSymbolResult]:
    results: list[CombinedSymbolResult] = []

    for symbol in DEFAULT_SYMBOLS:
        m15 = run_symbol_light_backtest(symbol)
        scalp = run_symbol_scalp_light_backtest(symbol, progress_every=100)
        results.append(
            CombinedSymbolResult(
                symbol=m15.symbol,
                m15_trades=m15.stats.total_signals,
                m15_tp1_wins=m15.stats.tp1_wins,
                m15_wr=m15.stats.win_rate,
                m15_r=m15.stats.total_r,
                m5_trades=scalp.total_signals,
                m5_tp1_wins=scalp.tp1_wins,
                m5_wr=scalp.win_rate,
                m5_r=scalp.total_r,
            )
        )

    return results


def print_combined_table(results: list[CombinedSymbolResult]) -> None:
    print()
    print("=== Combined light backtest (500 candles) ===")
    print("M15: Chief Analyst filters | M5 scalp: XAUUSD only, 60% conf, 1h gap, max 4/day")
    print()
    print(
        f"{'Інструмент':<10} {'M15 угоди':>10} {'M15 WR':>12} "
        f"{'M5 угоди':>10} {'M5 WR':>12} {'Total R':>10}"
    )
    print("-" * 68)

    total_m15 = 0
    total_m15_wins = 0
    total_m5 = 0
    total_m5_wins = 0
    total_r = 0.0

    for row in results:
        total_m15 += row.m15_trades
        total_m15_wins += row.m15_tp1_wins
        total_m5 += row.m5_trades
        total_m5_wins += row.m5_tp1_wins
        total_r += row.total_r

        print(
            f"{row.symbol:<10} {row.m15_trades:>10} "
            f"{_wr_label(row.m15_tp1_wins, row.m15_trades, row.m15_wr):>12} "
            f"{row.m5_trades:>10} "
            f"{_wr_label(row.m5_tp1_wins, row.m5_trades, row.m5_wr):>12} "
            f"{row.total_r:>+9.2f}R"
        )

    m15_total_wr = (total_m15_wins / total_m15 * 100) if total_m15 else 0.0
    m5_total_wr = (total_m5_wins / total_m5 * 100) if total_m5 else 0.0
    print("-" * 68)
    print(
        f"{'TOTAL':<10} {total_m15:>10} "
        f"{_wr_label(total_m15_wins, total_m15, m15_total_wr):>12} "
        f"{total_m5:>10} "
        f"{_wr_label(total_m5_wins, total_m5, m5_total_wr):>12} "
        f"{total_r:>+9.2f}R"
    )


def main() -> None:
    configure_console_encoding()
    print("Combined M15 + M5 scalp backtest — 500 candles per symbol", flush=True)
    print(f"Symbols: {', '.join(DEFAULT_SYMBOLS)}", flush=True)
    results = run_combined_backtest()
    print_combined_table(results)


if __name__ == "__main__":
    main()
