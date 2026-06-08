from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import SignalGenerator
from strategy.runner import (
    build_context,
    build_signal_reason,
    compute_final_decision,
    format_agents_agreement,
    run_agents,
)
from strategy.signal_filter import SignalFilter

from backtest.metrics import BacktestMetrics, BacktestSummary
from backtest.report import format_backtest_report, format_multi_timeframe_report
from backtest.simulator import SimulatedTradeResult, TradeSimulator
from tracking.signal_csv import SIGNALS_CSV_FILE, SignalCsvRow, SignalCsvStore

BACKTEST_RESULTS_FILE = Path(__file__).resolve().parent.parent / "backtest_results.json"
DEFAULT_WARMUP = 100
DEFAULT_BTC_TIMEFRAMES = ("1m", "5m", "15m")


@dataclass
class BacktestConfig:
    symbol: str = "BTCUSDT"
    timeframe: str = "15m"
    total_candles: int = 1500
    warmup_candles: int = DEFAULT_WARMUP


def candle_timestamp(candles: list[dict[str, float]], index: int) -> datetime:
    candle = candles[index]
    open_time = candle.get("open_time")
    if open_time is not None:
        return datetime.fromtimestamp(open_time / 1000, tz=timezone.utc)
    return datetime.now(timezone.utc)


class BacktestEngine:
    """Runs the multi-agent strategy over historical candles."""

    def __init__(
        self,
        config: BacktestConfig | None = None,
        results_file: Path = BACKTEST_RESULTS_FILE,
        signal_filter: SignalFilter | None = None,
    ) -> None:
        self.config = config or BacktestConfig()
        self.symbol_def = resolve_symbol(self.config.symbol)
        self.results_file = results_file
        self.market_data = MarketDataProvider()
        self.signal_generator = SignalGenerator()
        self.signal_filter = signal_filter or SignalFilter()
        self.simulator = TradeSimulator()
        self.metrics = BacktestMetrics()
        self.signal_csv = SignalCsvStore()

    def run(self) -> dict[str, Any]:
        candles = self.market_data.get_historical_market_data(
            self.config.symbol,
            self.config.timeframe,
            self.config.total_candles,
        )
        trades = self._simulate_trades(candles)
        summary = self.metrics.summarize(trades)
        payload = self._build_payload(candles, trades, summary)
        self._export(payload)
        return payload

    def run_timeframes(self, timeframes: tuple[str, ...] | list[str]) -> dict[str, Any]:
        """Run backtests for multiple timeframes and return a combined payload."""
        runs: list[dict[str, Any]] = []
        report_sections: list[tuple[str, list[SimulatedTradeResult], BacktestSummary, int]] = []

        for timeframe in timeframes:
            self.config = BacktestConfig(
                symbol=self.config.symbol,
                timeframe=timeframe,
                total_candles=self.config.total_candles,
                warmup_candles=self.config.warmup_candles,
            )
            payload = self.run()
            runs.append(payload)
            summary = BacktestSummary(**payload["summary"])
            trades = [SimulatedTradeResult(**trade) for trade in payload["trades"]]
            report_sections.append((timeframe, trades, summary, payload["candles_tested"]))

        combined = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol_def.display,
            "data_symbol": self.symbol_def.data_symbol,
            "timeframes": list(timeframes),
            "runs": runs,
        }
        self._export(combined)
        combined["report"] = format_multi_timeframe_report(self.symbol_def.display, report_sections)
        return combined

    def run_btcusdt_suite(
        self,
        timeframes: tuple[str, ...] = DEFAULT_BTC_TIMEFRAMES,
        total_candles: int = 1500,
    ) -> dict[str, Any]:
        self.config = BacktestConfig(
            symbol="BTCUSDT",
            timeframe=timeframes[0],
            total_candles=total_candles,
            warmup_candles=self.config.warmup_candles,
        )
        return self.run_timeframes(timeframes)

    def _simulate_trades(self, candles: list[dict[str, float]]) -> list[SimulatedTradeResult]:
        trades: list[SimulatedTradeResult] = []
        open_until_index = -1

        for index in range(self.config.warmup_candles, len(candles) - 1):
            if index <= open_until_index:
                continue

            history = candles[: index + 1]
            context = build_context(
                symbol=self.symbol_def.display,
                candles=history,
                timeframe=self.config.timeframe,
                timestamp=candle_timestamp(candles, index),
            )

            agent_results = run_agents(context)
            direction, confidence, _, _ = compute_final_decision(agent_results)
            filter_result = self.signal_filter.evaluate(
                agent_results,
                direction,
                confidence,
                symbol=self.symbol_def.display,
                timestamp=context.get("timestamp"),
            )

            if not filter_result.approved:
                continue

            try:
                signal = self.signal_generator.generate(
                    context,
                    filter_result.direction,
                    filter_result.confidence,
                    build_signal_reason(agent_results, filter_result.direction),
                )
            except ValueError:
                continue

            future = candles[index + 1 :]
            simulated = self.simulator.simulate(signal, future, entry_index=index)
            if simulated is None:
                continue

            trades.append(simulated)
            open_until_index = simulated.exit_index
            self.signal_csv.append(
                SignalCsvRow(
                    date=candle_timestamp(candles, index).isoformat(),
                    symbol=self.symbol_def.display,
                    direction=simulated.direction,
                    entry=simulated.entry,
                    sl=simulated.stop_loss,
                    tp1=simulated.tp1,
                    tp2=simulated.tp2,
                    tp3=simulated.tp3,
                    result=simulated.result,
                    profit_loss=simulated.pnl_r,
                    confidence=simulated.confidence,
                    agents_agreement=format_agents_agreement(
                        agent_results,
                        filter_result.direction,
                    ),
                )
            )

        return trades

    def _build_payload(
        self,
        candles: list[dict[str, float]],
        trades: list[SimulatedTradeResult],
        summary: BacktestSummary,
    ) -> dict[str, Any]:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol_def.display,
            "data_symbol": self.symbol_def.data_symbol,
            "timeframe": self.config.timeframe,
            "candles_tested": len(candles),
            "warmup_candles": self.config.warmup_candles,
            "summary": summary.to_dict(),
            "trades": [trade.to_dict() for trade in trades],
        }

    def _export(self, payload: dict[str, Any]) -> None:
        with self.results_file.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def print_report(self, payload: dict[str, Any]) -> None:
        if "runs" in payload:
            print(payload["report"])
            print()
            print(f"Results exported to: {self.results_file}")
            print(f"Signals exported to: {SIGNALS_CSV_FILE}")
            return

        trades = [SimulatedTradeResult(**trade) for trade in payload["trades"]]
        summary = BacktestSummary(**payload["summary"])
        report = format_backtest_report(
            symbol=payload["symbol"],
            timeframe=payload["timeframe"],
            candles_tested=payload["candles_tested"],
            trades=trades,
            summary=summary,
        )
        print(report)
        print()
        print(f"Results exported to: {self.results_file}")
        print(f"Signals exported to: {SIGNALS_CSV_FILE}")
