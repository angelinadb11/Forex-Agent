from __future__ import annotations

from backtest.metrics import BacktestSummary
from backtest.simulator import SimulatedTradeResult


def format_trade(trade: SimulatedTradeResult, index: int) -> str:
    return (
        f"Trade #{index} {trade.direction.upper()}\n"
        f"  Entry: {trade.entry:.2f}\n"
        f"  Stop Loss: {trade.stop_loss:.2f}\n"
        f"  TP1: {trade.tp1:.2f}\n"
        f"  TP2: {trade.tp2:.2f}\n"
        f"  TP3: {trade.tp3:.2f}\n"
        f"  Result: {trade.result} | PnL: {trade.pnl_r:+.2f}R | "
        f"Confidence: {trade.confidence:.0%}"
    )


def format_backtest_report(
    symbol: str,
    timeframe: str,
    candles_tested: int,
    trades: list[SimulatedTradeResult],
    summary: BacktestSummary,
) -> str:
    lines = [
        "=" * 40,
        "BACKTEST REPORT",
        "=" * 40,
        f"Symbol: {symbol}",
        f"Timeframe: {timeframe}",
        f"Candles tested: {candles_tested}",
        "",
    ]

    if trades:
        lines.append("--- Signals ---")
        for index, trade in enumerate(trades, start=1):
            lines.append(format_trade(trade, index))
            lines.append("")
    else:
        lines.append("No trades generated.")
        lines.append("")

    lines.extend(
        [
            "--- Performance ---",
            f"Total Trades: {summary.total_trades}",
            f"Win Rate: {summary.win_rate:.2f}%",
            f"TP1 Hit Rate: {summary.tp1_hit_rate:.2f}%",
            f"TP2 Hit Rate: {summary.tp2_hit_rate:.2f}%",
            f"TP3 Hit Rate: {summary.tp3_hit_rate:.2f}%",
            f"Average R:R: {summary.average_risk_reward:.2f}R",
            f"Maximum Drawdown: {summary.maximum_drawdown:.2f}R",
        ]
    )
    return "\n".join(lines)


def format_multi_timeframe_report(
    symbol: str,
    results: list[tuple[str, list[SimulatedTradeResult], BacktestSummary, int]],
) -> str:
    sections = [
        format_backtest_report(symbol, timeframe, candles, trades, summary)
        for timeframe, trades, summary, candles in results
    ]
    return "\n\n".join(sections)
