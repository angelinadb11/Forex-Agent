"""Compare liquidity scalp with different max SL limits (XAUUSD M1 prod filters)."""

from __future__ import annotations

from run_liquidity_scalp_backtest import scan_and_simulate, period_days, WARMUP
from strategy.liquidity_scalp import LiquidityScalpConfig
from strategy.scalp_mode import ScalpPublishGate
from data import MarketDataProvider
from tracking.console import configure_console_encoding


def main() -> None:
    configure_console_encoding()
    symbol = "XAUUSD"
    needed = WARMUP + 25000 + 1
    provider = MarketDataProvider()
    print(f"Loading {symbol} M1 x{needed}...", flush=True)
    m1 = provider.get_historical_market_data(symbol, "1m", needed)
    print("Loading H1 x400...", flush=True)
    h1 = provider.get_historical_market_data(symbol, "1h", 400)
    print(f"Got {len(m1)} M1 candles\n", flush=True)

    variants: list[tuple[str, float, float]] = [
        ("prod max 25 (current)", 5, 25),
        ("max SL 30", 5, 30),
        ("max SL 40", 5, 40),
        ("max SL 50", 5, 50),
        ("max SL 60", 5, 60),
        ("SL only 30-60", 30, 60),
    ]

    print(
        f"{'Variant':<24} {'Trades':>6} {'/day':>6} {'WR':>7} "
        f"{'Stops':>6} {'BE':>4} {'TotalR':>9} {'R/tr':>7} {'Rejected':>9}"
    )
    print("-" * 88)

    for label, sl_min, sl_max in variants:
        config = LiquidityScalpConfig(
            require_volume_spike=True,
            min_volume_ratio=1.5,
            require_stoch_rsi=True,
            min_sl_pips=sl_min,
            max_sl_pips=sl_max,
        )
        gate = ScalpPublishGate(min_interval_seconds=300, max_signals_per_day=6)
        stats = scan_and_simulate(
            m1,
            symbol,
            config=config,
            h1_candles=h1,
            quiet=True,
            gate=gate,
        )
        days = period_days(stats)
        wr = f"{stats.win_rate:.0f}%" if stats.total_signals else "—"
        avg = stats.total_r / stats.total_signals if stats.total_signals else 0.0
        print(
            f"{label:<24} {stats.total_signals:>6} "
            f"{stats.total_signals / days:>6.2f} {wr:>7} "
            f"{stats.full_stops:>6} {stats.tp1_then_be:>4} "
            f"{stats.total_r:>+8.2f}R {avg:>+6.2f}R "
            f"{stats.sl_too_wide:>9}",
            flush=True,
        )


if __name__ == "__main__":
    main()
