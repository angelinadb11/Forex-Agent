"""Compare baseline vs near-TP1 breakeven rule on Profile D XAUUSD (500 M15)."""

from __future__ import annotations

from backtest.engine import BacktestConfig
from backtest.simulator import TradeManagementMode
from run_profile_d_light_backtest import load_symbol_data, period_days
from run_xauusd_30d_backtest import LocalDataBacktestEngine, WARMUP
from strategy.signal_filter import FILTER_PROFILE_D, SignalFilter
from tracking.console import configure_console_encoding


def _wr(stats) -> str:
    if not stats.total_signals:
        return "—"
    return f"{stats.win_rate:.0f}%"


def main() -> None:
    configure_console_encoding()
    print("Near-TP1 BE comparison — Profile D, XAUUSD, 500 M15 candles\n")

    m15, h1, h4 = load_symbol_data("XAUUSD")
    engine = LocalDataBacktestEngine(
        BacktestConfig(
            symbol="XAUUSD",
            timeframe="15m",
            total_candles=len(m15),
            warmup_candles=WARMUP,
        ),
        m15_candles=m15,
        h1_candles=h1,
        h4_candles=h4,
        signal_filter=SignalFilter.from_profile(FILTER_PROFILE_D),
    )

    setups, _ = engine._scan_candles(m15)
    baseline = engine._simulate_setups(
        setups,
        m15,
        TradeManagementMode.PARTIAL,
    )
    near_tp1 = engine._simulate_setups(
        setups,
        m15,
        TradeManagementMode.PARTIAL_NEAR_TP1_BE,
    )
    near_tp1_blocked = engine._simulate_setups(
        setups,
        m15,
        TradeManagementMode.PARTIAL_NEAR_TP1_BE,
        enable_m15_reversal_block=True,
    )
    days = period_days(baseline)

    print(f"Period: ~{days:.1f} days | Setups: {len(setups)}")
    print()
    print(
        f"{'Варіант':<32} {'Угод/день':>10} {'WR':>8} "
        f"{'Стопів':>8} {'BE':>6} {'Total R':>10}"
    )
    print("-" * 82)

    rows = (
        ("Baseline (без near-TP1 BE)", baseline, None),
        ("+ Near-TP1 BE (>=1.2R + M15)", near_tp1, None),
        (
            "+ Near-TP1 BE + M15 block",
            near_tp1_blocked,
            near_tp1_blocked.m15_reversal_blocked,
        ),
    )
    for label, stats, blocked in rows:
        spd = stats.total_signals / days if days else 0.0
        suffix = f"  [{blocked} skip]" if blocked else ""
        print(
            f"{label:<32} {spd:>10.2f} {_wr(stats):>8} "
            f"{stats.stop_losses:>8} {stats.breakeven_exits:>6} "
            f"{stats.total_r:>+9.2f}R{suffix}"
        )

    delta_r = near_tp1_blocked.total_r - baseline.total_r
    delta_stops = near_tp1_blocked.stop_losses - baseline.stop_losses
    print("-" * 82)
    print(
        f"{'Δ (blocked − baseline)':<32} {'':>10} {'':>8} "
        f"{delta_stops:>+8} {'':>6} {delta_r:>+9.2f}R"
    )
    print()

    print("Per-trade diff (baseline vs near-TP1 blocked):")
    print(f"{'#':<3} {'Baseline':<12} {'Near-TP1+block':<14} {'ΔR':>8}")
    print("-" * 44)
    for index, (base, updated) in enumerate(
        zip(baseline.trades, near_tp1_blocked.trades),
        start=1,
    ):
        if base.result != updated.result or abs(base.pnl_r - updated.pnl_r) > 1e-6:
            print(
                f"{index:<3} {base.result}/{base.pnl_r:+.2f}R{'':<4} "
                f"{updated.result}/{updated.pnl_r:+.2f}R{'':<6} "
                f"{updated.pnl_r - base.pnl_r:+.2f}R"
            )
    if len(baseline.trades) != len(near_tp1_blocked.trades):
        print(
            f"\nTrade count: baseline {len(baseline.trades)} | "
            f"near-TP1+block {len(near_tp1_blocked.trades)} | "
            f"skipped {near_tp1_blocked.m15_reversal_blocked}"
        )


if __name__ == "__main__":
    main()
