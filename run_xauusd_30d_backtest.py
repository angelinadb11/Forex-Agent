"""Run XAUUSD M15 backtest for the last 30 days with detailed blocking stats."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.base import Direction
from backtest.engine import BacktestConfig, BacktestEngine, candle_timestamp
from backtest.simulator import SimulatedTradeResult
from config.sl_config import get_sl_config
from signal_generator import price_distance_pips
from strategy.runner import (
    build_context,
    build_signal_reason,
    compute_final_decision,
    run_agents,
    slice_candles_as_of,
)
from strategy.signal_filter import SignalFilter
from tracking.console import configure_console_encoding

DAYS = 30
M15_CANDLES_PER_DAY = 96
WARMUP = 100
TOTAL_CANDLES = DAYS * M15_CANDLES_PER_DAY + WARMUP
H1_CANDLES = DAYS * 24 + 250
XAUUSD_PIP_SIZE = get_sl_config("XAUUSD").pip_size  # type: ignore[union-attr]


@dataclass
class BacktestRunStats:
    trend_blocked: int = 0
    sl_blocked: int = 0
    other_filter_blocked: int = 0
    neutral_decisions: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)

    @property
    def total_signals(self) -> int:
        return len(self.trades)

    @property
    def long_signals(self) -> int:
        return sum(1 for trade in self.trades if trade.direction == "long")

    @property
    def short_signals(self) -> int:
        return sum(1 for trade in self.trades if trade.direction == "short")

    @property
    def tp1_wins(self) -> int:
        return sum(1 for trade in self.trades if trade.tp1_hit)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.tp1_wins / self.total_signals * 100

    @property
    def avg_sl_pips(self) -> float:
        if not self.trades:
            return 0.0
        values = [
            price_distance_pips(trade.risk, XAUUSD_PIP_SIZE)
            for trade in self.trades
        ]
        return sum(values) / len(values)

    @property
    def avg_rr(self) -> float:
        """Average reward-to-risk to TP1 (theoretical target R)."""
        if not self.trades:
            return 0.0
        values = [
            abs(trade.tp1 - trade.entry) / trade.risk
            for trade in self.trades
            if trade.risk > 0
        ]
        return sum(values) / len(values) if values else 0.0


class DetailedBacktestEngine(BacktestEngine):
    def run_with_stats(self) -> BacktestRunStats:
        candles = self.market_data.get_historical_market_data(
            self.config.symbol,
            self.config.timeframe,
            self.config.total_candles,
        )
        stats = self._simulate_trades_with_stats(candles)
        return stats

    def _simulate_trades_with_stats(
        self,
        candles: list[dict[str, float]],
    ) -> BacktestRunStats:
        stats = BacktestRunStats()
        open_until_index = -1
        h1_candles = self.market_data.get_historical_market_data(
            self.config.symbol,
            "1h",
            H1_CANDLES,
        )

        for index in range(self.config.warmup_candles, len(candles) - 1):
            if index <= open_until_index:
                continue

            history = candles[: index + 1]
            timestamp = candle_timestamp(candles, index)
            context = build_context(
                symbol=self.symbol_def.display,
                candles=history,
                timeframe=self.config.timeframe,
                timestamp=timestamp,
                h1_candles=slice_candles_as_of(h1_candles, timestamp),
            )

            agent_results = run_agents(context)
            direction, confidence, _, _ = compute_final_decision(agent_results)

            if direction == Direction.NEUTRAL:
                stats.neutral_decisions += 1
                continue

            filter_result = self.signal_filter.evaluate(
                agent_results,
                direction,
                confidence,
                symbol=self.symbol_def.display,
                timestamp=context.get("timestamp"),
            )

            if not filter_result.approved:
                if SignalFilter._evaluate_trend_hard_block(
                    agent_results.get("trend_filter"),
                    direction,
                ):
                    stats.trend_blocked += 1
                else:
                    stats.other_filter_blocked += 1
                continue

            generation = self.signal_generator.generate(
                context,
                filter_result.direction,
                filter_result.confidence,
                build_signal_reason(agent_results, filter_result.direction),
            )
            if generation.signal is None:
                stats.sl_blocked += 1
                continue

            signal = generation.signal
            future = candles[index + 1 :]
            simulated = self.simulator.simulate(signal, future, entry_index=index)
            if simulated is None:
                continue

            stats.trades.append(simulated)
            open_until_index = simulated.exit_index

        return stats


def print_stats(stats: BacktestRunStats) -> None:
    period_start = f"last {DAYS} days"
    print(f"=== XAUUSD M15 Backtest ({period_start}) ===")
    print(f"Candles loaded: {TOTAL_CANDLES} ({WARMUP} warmup + {DAYS * M15_CANDLES_PER_DAY} test)")
    print()
    print(f"Total signals (executed):     {stats.total_signals}")
    print(f"  LONG:                       {stats.long_signals}")
    print(f"  SHORT:                      {stats.short_signals}")
    print()
    print(f"Blocked by TrendFilter:       {stats.trend_blocked}")
    print(f"Blocked by SL validation:     {stats.sl_blocked}")
    print(f"Blocked by other filters:     {stats.other_filter_blocked}")
    print(f"Neutral decisions (skipped):  {stats.neutral_decisions}")
    print()
    print(f"Win rate (TP1 hit):           {stats.win_rate:.1f}% ({stats.tp1_wins}/{stats.total_signals})")
    print(f"Average SL:                   {stats.avg_sl_pips:.1f} pips")
    print(f"Average RR (to TP1):          {stats.avg_rr:.2f}R")


def main() -> None:
    configure_console_encoding()
    engine = DetailedBacktestEngine(
        BacktestConfig(
            symbol="XAUUSD",
            timeframe="15m",
            total_candles=TOTAL_CANDLES,
            warmup_candles=WARMUP,
        )
    )
    print("Loading historical data and running backtest...")
    print()
    stats = engine.run_with_stats()
    print_stats(stats)


if __name__ == "__main__":
    main()
