from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from backtest.simulator import SimulatedTradeResult


@dataclass
class BacktestSummary:
    total_trades: int
    wins: int
    losses: int
    win_rate: float
    tp1_hit_rate: float
    tp2_hit_rate: float
    tp3_hit_rate: float
    average_risk_reward: float
    profit_factor: float
    maximum_drawdown: float
    tp1_hits: int
    tp2_hits: int
    tp3_hits: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class BacktestMetrics:
    """Calculates aggregate backtest performance metrics."""

    def summarize(self, trades: list[SimulatedTradeResult]) -> BacktestSummary:
        if not trades:
            return BacktestSummary(0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0)

        total = len(trades)
        wins = sum(1 for trade in trades if trade.win)
        losses = sum(1 for trade in trades if trade.loss)
        tp1_hits = sum(1 for trade in trades if trade.tp1_hit)
        tp2_hits = sum(1 for trade in trades if trade.tp2_hit)
        tp3_hits = sum(1 for trade in trades if trade.tp3_hit)

        pnl_values = [trade.pnl_r for trade in trades]
        gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))

        profit_factor = gross_profit / gross_loss if gross_loss > 0 else gross_profit
        average_rr = sum(pnl_values) / total
        win_rate = (wins / total) * 100

        return BacktestSummary(
            total_trades=total,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            tp1_hit_rate=(tp1_hits / total) * 100,
            tp2_hit_rate=(tp2_hits / total) * 100,
            tp3_hit_rate=(tp3_hits / total) * 100,
            average_risk_reward=average_rr,
            profit_factor=profit_factor,
            maximum_drawdown=self._maximum_drawdown(pnl_values),
            tp1_hits=tp1_hits,
            tp2_hits=tp2_hits,
            tp3_hits=tp3_hits,
        )

    @staticmethod
    def _maximum_drawdown(pnl_values: list[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_drawdown = 0.0

        for pnl in pnl_values:
            equity += pnl
            peak = max(peak, equity)
            max_drawdown = max(max_drawdown, peak - equity)

        return max_drawdown
