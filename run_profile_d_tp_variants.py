"""Re-simulate Profile D XAUUSD trades with alternate TP/partial settings."""

from __future__ import annotations

from dataclasses import replace

from agents.base import Direction
from backtest.engine import candle_timestamp
from backtest.simulator import TradeManagementMode, TradeSimulator
from run_light_filter_comparison import load_symbol_data
from run_xauusd_30d_backtest import BacktestConfig, LocalDataBacktestEngine, WARMUP
from signal_generator import TradeSignal, resolve_signal_direction
from strategy.signal_filter import FILTER_PROFILE_D, SignalFilter
from tracking.console import configure_console_encoding
from tracking.level_checks import stop_loss_hit, take_profit_hit


def simulate_partial_custom(
    signal: TradeSignal,
    future_candles: list,
    entry_index: int,
    *,
    f1: float,
    f2: float,
    f3: float,
):
    if not future_candles:
        return None

    stop_loss = signal.stop_loss
    risk = abs(signal.entry - signal.stop_loss)
    if risk == 0:
        return None

    tp1_hit = False
    tp2_hit = False
    tp3_hit = False
    position_remaining = 1.0
    cumulative_r = 0.0
    direction = resolve_signal_direction(signal)
    simulator = TradeSimulator()

    for offset, candle in enumerate(future_candles):
        exit_index = entry_index + offset + 1
        high = candle["high"]
        low = candle["low"]

        if direction == Direction.LONG:
            if stop_loss_hit(
                direction=Direction.LONG,
                high=high,
                low=low,
                stop_loss=stop_loss,
            ):
                sl_r = (stop_loss - signal.entry) / risk
                pnl_r = cumulative_r + position_remaining * sl_r
                return simulator._build_result(
                    signal,
                    entry_index,
                    exit_index,
                    stop_loss,
                    pnl_r,
                    "stop_loss",
                    tp1_hit,
                    tp2_hit,
                    tp3_hit,
                    TradeManagementMode.PARTIAL,
                )

            if not tp1_hit and take_profit_hit(
                direction=Direction.LONG,
                high=high,
                low=low,
                tp_price=signal.tp1,
            ):
                tp1_hit = True
                cumulative_r += f1 * ((signal.tp1 - signal.entry) / risk)
                position_remaining -= f1
                stop_loss = signal.entry

            if not tp2_hit and take_profit_hit(
                direction=Direction.LONG,
                high=high,
                low=low,
                tp_price=signal.tp2,
            ):
                tp2_hit = True
                cumulative_r += f2 * ((signal.tp2 - signal.entry) / risk)
                position_remaining -= f2
                stop_loss = signal.tp1

            if take_profit_hit(
                direction=Direction.LONG,
                high=high,
                low=low,
                tp_price=signal.tp3,
            ):
                tp3_hit = True
                cumulative_r += f3 * ((signal.tp3 - signal.entry) / risk)
                return simulator._build_result(
                    signal,
                    entry_index,
                    exit_index,
                    signal.tp3,
                    cumulative_r,
                    "tp3",
                    tp1_hit,
                    tp2_hit,
                    tp3_hit,
                    TradeManagementMode.PARTIAL,
                )
        else:
            if stop_loss_hit(
                direction=Direction.SHORT,
                high=high,
                low=low,
                stop_loss=stop_loss,
            ):
                sl_r = (signal.entry - stop_loss) / risk
                pnl_r = cumulative_r + position_remaining * sl_r
                return simulator._build_result(
                    signal,
                    entry_index,
                    exit_index,
                    stop_loss,
                    pnl_r,
                    "stop_loss",
                    tp1_hit,
                    tp2_hit,
                    tp3_hit,
                    TradeManagementMode.PARTIAL,
                )

            if not tp1_hit and take_profit_hit(
                direction=Direction.SHORT,
                high=high,
                low=low,
                tp_price=signal.tp1,
            ):
                tp1_hit = True
                cumulative_r += f1 * ((signal.entry - signal.tp1) / risk)
                position_remaining -= f1
                stop_loss = signal.entry

            if not tp2_hit and take_profit_hit(
                direction=Direction.SHORT,
                high=high,
                low=low,
                tp_price=signal.tp2,
            ):
                tp2_hit = True
                cumulative_r += f2 * ((signal.entry - signal.tp2) / risk)
                position_remaining -= f2
                stop_loss = signal.tp1

            if take_profit_hit(
                direction=Direction.SHORT,
                high=high,
                low=low,
                tp_price=signal.tp3,
            ):
                tp3_hit = True
                cumulative_r += f3 * ((signal.entry - signal.tp3) / risk)
                return simulator._build_result(
                    signal,
                    entry_index,
                    exit_index,
                    signal.tp3,
                    cumulative_r,
                    "tp3",
                    tp1_hit,
                    tp2_hit,
                    tp3_hit,
                    TradeManagementMode.PARTIAL,
                )

    return None


def rescale_signal(signal: TradeSignal, rr1: float, rr2: float, rr3: float) -> TradeSignal:
    direction = resolve_signal_direction(signal)
    risk = abs(signal.entry - signal.stop_loss)
    if direction == Direction.LONG:
        return replace(
            signal,
            tp1=signal.entry + rr1 * risk,
            tp2=signal.entry + rr2 * risk,
            tp3=signal.entry + rr3 * risk,
        )
    return replace(
        signal,
        tp1=signal.entry - rr1 * risk,
        tp2=signal.entry - rr2 * risk,
        tp3=signal.entry - rr3 * risk,
    )


def select_non_overlapping(setups, candles):
    open_until = -1
    selected = []
    simulator = TradeSimulator()
    for setup in setups:
        if setup.entry_index <= open_until:
            continue
        simulated = simulator.simulate(
            setup.signal,
            candles[setup.entry_index + 1 :],
            setup.entry_index,
        )
        if simulated is None:
            continue
        selected.append(setup)
        open_until = simulated.exit_index
    return selected


def breakeven_wr_tp1_be(f1: float, rr1: float) -> float:
    """WR needed if wins pay f1*rr1 and losses pay -1R (TP1 then BE)."""
    return 100.0 / (1.0 + f1 * rr1)


def main() -> None:
    configure_console_encoding()
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
        progress_every=500,
        signal_filter=SignalFilter.from_profile(FILTER_PROFILE_D),
    )
    setups, _ = engine._scan_candles(m15)
    selected = select_non_overlapping(setups, m15)

    variants = [
        (
            "1",
            "TP1=1.5R TP2=2.5R TP3=3.5R | partial 50/25/25",
            (1.5, 2.5, 3.5),
            (0.50, 0.25, 0.25),
        ),
        (
            "2",
            "TP1=1R TP2=2R TP3=3R | partial 70/20/10",
            (1.0, 2.0, 3.0),
            (0.70, 0.20, 0.10),
        ),
    ]

    print("=== Profile D XAUUSD — TP variant test (same setups) ===")
    print(f"Trades: {len(selected)}")
    print()
    print(f"{'Var':<5} {'Breakeven WR':>14} {'Actual TP1 WR':>16} {'Total R':>10}")
    print("-" * 50)

    for code, label, rr, fractions in variants:
        rr1, rr2, rr3 = rr
        f1, f2, f3 = fractions
        trades = []
        for setup in selected:
            signal = rescale_signal(setup.signal, rr1, rr2, rr3)
            trade = simulate_partial_custom(
                signal,
                m15[setup.entry_index + 1 :],
                setup.entry_index,
                f1=f1,
                f2=f2,
                f3=f3,
            )
            if trade is not None:
                trades.append(trade)

        total_r = sum(trade.pnl_r for trade in trades)
        tp1_wins = sum(1 for trade in trades if trade.tp1_hit)
        wr = tp1_wins / len(trades) * 100 if trades else 0.0
        be = breakeven_wr_tp1_be(f1, rr1)
        print(f"{code:<5} {be:>13.1f}% {tp1_wins}/{len(trades)} ({wr:.0f}%)".rjust(28) + f" {total_r:>+9.2f}R")

        print(f"      {label}")
        for index, trade in enumerate(trades, 1):
            timestamp = candle_timestamp(m15, trade.entry_index).isoformat()[:16]
            print(
                f"      {index}. {timestamp} "
                f"TP1={'Y' if trade.tp1_hit else 'N'} "
                f"TP2={'Y' if trade.tp2_hit else 'N'} "
                f"TP3={'Y' if trade.tp3_hit else 'N'} "
                f"-> {trade.pnl_r:+.2f}R"
            )
        print()

    baseline = []
    simulator = TradeSimulator()
    for setup in selected:
        trade = simulator.simulate(
            setup.signal,
            m15[setup.entry_index + 1 :],
            setup.entry_index,
        )
        if trade is not None:
            baseline.append(trade)
    base_r = sum(trade.pnl_r for trade in baseline)
    base_wr = sum(1 for trade in baseline if trade.tp1_hit)
    print(
        f"Baseline (1R, 50/25/25): breakeven WR {breakeven_wr_tp1_be(0.5, 1.0):.1f}% | "
        f"actual {base_wr}/{len(baseline)} | Total R {base_r:+.2f}R"
    )


if __name__ == "__main__":
    main()
