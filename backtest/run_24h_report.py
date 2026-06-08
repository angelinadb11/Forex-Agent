"""Run 24-hour M15 backtests with per-symbol SL configuration."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from agents.base import Direction
from backtest.engine import candle_timestamp
from backtest.simulator import TradeSimulator
from config.sl_config import get_sl_config
from config.symbols import SUPPORTED_SYMBOLS, resolve_symbol
from data import MarketDataProvider
from signal_generator import SignalGenerator, price_distance_pips
from strategy.runner import (
    build_context,
    build_signal_reason,
    compute_final_decision,
    run_agents,
    slice_candles_as_of,
)
from strategy.signal_filter import SignalFilter

M15_CANDLES_24H = 96
WARMUP_CANDLES = 100
H1_LIMIT = 250


@dataclass
class Backtest24hStats:
    symbol: str
    timeframe: str
    sl_min_pips: float | None
    sl_max_pips: float | None
    window_start: str
    window_end: str
    bars_analyzed: int
    non_neutral_decisions: int
    blocked_trend_filter: int
    blocked_other_filter: int
    blocked_sl_validation: int
    total_signals: int
    long_signals: int
    short_signals: int
    tp1_hits: int
    win_rate_tp1_pct: float
    avg_sl_pips: float
    avg_rr: float
    trades: list[dict]
    trend_filter_blocks: list[dict]


def _is_trend_block(message: str) -> bool:
    return "H1 trend" in message and ("BULLISH" in message or "BEARISH" in message)


def run_24h_backtest(symbol: str, timeframe: str = "15m") -> Backtest24hStats:
    symbol_def = resolve_symbol(symbol)
    sl_config = get_sl_config(symbol_def.display)
    provider = MarketDataProvider()
    signal_filter = SignalFilter()
    signal_generator = SignalGenerator()
    simulator = TradeSimulator()

    total_needed = WARMUP_CANDLES + M15_CANDLES_24H + 10
    candles = provider.get_historical_market_data(symbol_def.display, timeframe, total_needed)
    h1_candles = provider.get_historical_market_data(
        symbol_def.display,
        "1h",
        max(total_needed, H1_LIMIT),
    )

    if len(candles) < WARMUP_CANDLES + M15_CANDLES_24H:
        raise RuntimeError(
            f"Not enough M15 candles for {symbol_def.display}: got {len(candles)}, "
            f"need at least {WARMUP_CANDLES + M15_CANDLES_24H}"
        )

    eval_start = len(candles) - M15_CANDLES_24H
    window_start = candle_timestamp(candles, eval_start)
    window_end = candle_timestamp(candles, len(candles) - 1)

    blocked_trend = 0
    blocked_other = 0
    blocked_sl = 0
    non_neutral = 0
    trades: list[dict] = []
    trend_blocks: list[dict] = []
    open_until_index = -1
    pip_size = sl_config.pip_size if sl_config else 1.0

    for index in range(eval_start, len(candles) - 1):
        if index <= open_until_index:
            continue

        timestamp = candle_timestamp(candles, index)
        context = build_context(
            symbol=symbol_def.display,
            candles=candles[: index + 1],
            timeframe=timeframe,
            timestamp=timestamp,
            h1_candles=slice_candles_as_of(h1_candles, timestamp, limit=H1_LIMIT),
        )

        agent_results = run_agents(context)
        direction, confidence, _, _ = compute_final_decision(agent_results)

        if direction != Direction.NEUTRAL:
            non_neutral += 1

        filter_result = signal_filter.evaluate(
            agent_results,
            direction,
            confidence,
            symbol=symbol_def.display,
            timestamp=timestamp,
        )

        if not filter_result.approved:
            if direction != Direction.NEUTRAL and _is_trend_block(filter_result.message):
                blocked_trend += 1
                trend_blocks.append(
                    {
                        "time": timestamp.isoformat(),
                        "attempted_direction": direction.value.upper(),
                        "h1_trend": agent_results["trend_filter"].direction.value.upper(),
                        "confidence": round(confidence, 2),
                        "reason": filter_result.message,
                    }
                )
            elif direction != Direction.NEUTRAL:
                blocked_other += 1
            continue

        generation = signal_generator.generate(
            context,
            filter_result.direction,
            filter_result.confidence,
            build_signal_reason(agent_results, filter_result.direction),
        )
        if generation.signal is None:
            blocked_sl += 1
            continue

        signal = generation.signal
        simulated = simulator.simulate(signal, candles[index + 1 :], entry_index=index)
        if simulated is None:
            continue

        sl_pips = price_distance_pips(abs(signal.entry - signal.stop_loss), pip_size)
        rr = abs(signal.tp1 - signal.entry) / abs(signal.entry - signal.stop_loss)

        trades.append(
            {
                "time": timestamp.isoformat(),
                "direction": simulated.direction,
                "entry": simulated.entry,
                "stop_loss": simulated.stop_loss,
                "tp1": simulated.tp1,
                "sl_pips": round(sl_pips, 2),
                "planned_rr_tp1": round(rr, 2),
                "result": simulated.result,
                "tp1_hit": simulated.tp1_hit,
                "pnl_r": round(simulated.pnl_r, 2),
                "confidence": simulated.confidence,
                "lot_size": signal.lot_size,
            }
        )
        open_until_index = simulated.exit_index

    total_signals = len(trades)
    long_signals = sum(1 for trade in trades if trade["direction"] == Direction.LONG.value)
    short_signals = sum(1 for trade in trades if trade["direction"] == Direction.SHORT.value)
    tp1_hits = sum(1 for trade in trades if trade["tp1_hit"])
    win_rate = (tp1_hits / total_signals * 100) if total_signals else 0.0
    avg_sl = sum(trade["sl_pips"] for trade in trades) / total_signals if total_signals else 0.0
    avg_rr = sum(trade["planned_rr_tp1"] for trade in trades) / total_signals if total_signals else 0.0

    return Backtest24hStats(
        symbol=symbol_def.display,
        timeframe=timeframe,
        sl_min_pips=sl_config.min_sl_pips if sl_config else None,
        sl_max_pips=sl_config.max_sl_pips if sl_config else None,
        window_start=window_start.isoformat(),
        window_end=window_end.isoformat(),
        bars_analyzed=M15_CANDLES_24H,
        non_neutral_decisions=non_neutral,
        blocked_trend_filter=blocked_trend,
        blocked_other_filter=blocked_other,
        blocked_sl_validation=blocked_sl,
        total_signals=total_signals,
        long_signals=long_signals,
        short_signals=short_signals,
        tp1_hits=tp1_hits,
        win_rate_tp1_pct=round(win_rate, 1),
        avg_sl_pips=round(avg_sl, 2),
        avg_rr=round(avg_rr, 2),
        trades=trades,
        trend_filter_blocks=trend_blocks,
    )


def print_report(stats: Backtest24hStats) -> None:
    print(f"=== {stats.symbol} M15 Backtest — Last 24 Hours ===")
    if stats.sl_min_pips is not None and stats.sl_max_pips is not None:
        print(f"SL range: {stats.sl_min_pips:.0f}-{stats.sl_max_pips:.0f} pips")
    print(f"Window: {stats.window_start} -> {stats.window_end}")
    print(f"Bars analyzed: {stats.bars_analyzed}")
    print()
    print(f"Non-neutral decisions: {stats.non_neutral_decisions}")
    print(f"Blocked by TrendFilter: {stats.blocked_trend_filter}")
    print(f"Blocked by other filters: {stats.blocked_other_filter}")
    print(f"Blocked by SL validation: {stats.blocked_sl_validation}")
    print()
    print(f"Total signals (executed): {stats.total_signals}")
    print(f"  LONG:  {stats.long_signals}")
    print(f"  SHORT: {stats.short_signals}")
    print()
    print(f"TP1 hit rate: {stats.tp1_hits}/{stats.total_signals} ({stats.win_rate_tp1_pct}%)")
    print(f"Average SL: {stats.avg_sl_pips} pips")
    print(f"Average RR (planned TP1): {stats.avg_rr}R")
    print()

    if stats.trend_filter_blocks:
        print(f"TrendFilter blocks ({len(stats.trend_filter_blocks)}):")
        for block in stats.trend_filter_blocks[:5]:
            print(
                f"  {block['time']} | tried {block['attempted_direction']} | "
                f"H1={block['h1_trend']} | {block['reason']}"
            )
        if len(stats.trend_filter_blocks) > 5:
            print(f"  ... and {len(stats.trend_filter_blocks) - 5} more")
        print()

    if stats.trades:
        print("Trades:")
        for trade in stats.trades:
            print(
                f"  {trade['time']} | {trade['direction'].upper()} | "
                f"entry={trade['entry']:.2f} sl={trade['stop_loss']:.2f} "
                f"({trade['sl_pips']} pips) | {trade['result']} | "
                f"TP1={'yes' if trade['tp1_hit'] else 'no'} | {trade['pnl_r']}R"
            )
    print()


def run_xauusd_24h_backtest() -> Backtest24hStats:
    return run_24h_backtest("XAUUSD")


if __name__ == "__main__":
    results: dict[str, dict] = {}
    for symbol in SUPPORTED_SYMBOLS:
        stats = run_24h_backtest(symbol)
        print_report(stats)
        results[symbol] = asdict(stats)

    output = Path(__file__).resolve().parent.parent / "backtest_24h_all_symbols.json"
    output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved to: {output}")
