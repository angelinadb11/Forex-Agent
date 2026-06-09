"""Compare Profile D baseline vs Bollinger Band gate on XAUUSD (500 M15)."""

from __future__ import annotations

from dataclasses import replace

from backtest.engine import BacktestConfig, candle_timestamp
from backtest.simulator import TradeManagementMode
from run_profile_d_light_backtest import load_symbol_data, period_days
from run_xauusd_30d_backtest import LocalDataBacktestEngine, WARMUP
from strategy.signal_filter import FILTER_PROFILE_D, SignalFilter
from tracking.console import configure_console_encoding


def _wr(stats) -> str:
    if not stats.total_signals:
        return "—"
    return f"{stats.win_rate:.0f}%"


def _build_engine(m15, h1, h4, signal_filter: SignalFilter) -> LocalDataBacktestEngine:
    return LocalDataBacktestEngine(
        BacktestConfig(
            symbol="XAUUSD",
            timeframe="15m",
            total_candles=len(m15),
            warmup_candles=WARMUP,
        ),
        m15_candles=m15,
        h1_candles=h1,
        h4_candles=h4,
        signal_filter=signal_filter,
    )


def main() -> None:
    configure_console_encoding()
    print("BB gate comparison — Profile D, XAUUSD, 500 M15 candles\n")

    m15, h1, h4 = load_symbol_data("XAUUSD")

    profile_bb = replace(FILTER_PROFILE_D, use_bb_gate=True)

    baseline_engine = _build_engine(
        m15, h1, h4, SignalFilter.from_profile(FILTER_PROFILE_D)
    )
    bb_engine = _build_engine(
        m15, h1, h4, SignalFilter.from_profile(profile_bb)
    )

    base_setups, _ = baseline_engine._scan_candles(m15)
    bb_setups, _ = bb_engine._scan_candles(m15)

    baseline = baseline_engine._simulate_setups(
        base_setups, m15, TradeManagementMode.PARTIAL
    )
    bb_stats = bb_engine._simulate_setups(
        bb_setups, m15, TradeManagementMode.PARTIAL
    )
    days = period_days(baseline)

    print(
        f"Period: ~{days:.1f} days | Setups: baseline {len(base_setups)} "
        f"vs BB gate {len(bb_setups)}"
    )
    print()
    print(
        f"{'Варіант':<28} {'Угод/день':>10} {'WR':>8} "
        f"{'Стопів':>8} {'BE':>6} {'Total R':>10}"
    )
    print("-" * 78)

    for label, stats in (
        ("Baseline (Profile D)", baseline),
        ("+ BB gate (20, 2.0)", bb_stats),
    ):
        spd = stats.total_signals / days if days else 0.0
        print(
            f"{label:<28} {spd:>10.2f} {_wr(stats):>8} "
            f"{stats.stop_losses:>8} {stats.breakeven_exits:>6} "
            f"{stats.total_r:>+9.2f}R"
        )

    delta_r = bb_stats.total_r - baseline.total_r
    delta_stops = bb_stats.stop_losses - baseline.stop_losses
    print("-" * 78)
    print(
        f"{'Δ (BB gate − baseline)':<28} {'':>10} {'':>8} "
        f"{delta_stops:>+8} {'':>6} {delta_r:>+9.2f}R"
    )
    print()

    base_keys = {trade.entry_index for trade in baseline.trades}
    bb_keys = {trade.entry_index for trade in bb_stats.trades}

    removed = [t for t in baseline.trades if t.entry_index not in bb_keys]
    added = [t for t in bb_stats.trades if t.entry_index not in base_keys]

    if removed:
        print("Угоди, які BB gate відсік (як вони закінчились у baseline):")
        for trade in removed:
            ts = candle_timestamp(m15, trade.entry_index)
            print(
                f"  {ts.strftime('%m-%d %H:%M')} {trade.direction} "
                f"entry={trade.entry:.2f} -> {trade.result} {trade.pnl_r:+.2f}R"
            )
    if added:
        print("Нові угоди, що з'явились через звільнені слоти:")
        for trade in added:
            ts = candle_timestamp(m15, trade.entry_index)
            print(
                f"  {ts.strftime('%m-%d %H:%M')} {trade.direction} "
                f"entry={trade.entry:.2f} -> {trade.result} {trade.pnl_r:+.2f}R"
            )
    if not removed and not added:
        print("BB gate не змінив склад угод на цьому періоді.")


if __name__ == "__main__":
    main()
