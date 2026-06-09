"""Download XAUUSD M15 history (30 days), save locally, and run backtest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

from agents.base import Direction
from agents.zone_helpers import ZoneCatalog
from backtest.engine import BacktestConfig, BacktestEngine, candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.m15_reversal_block import BacktestM15ReversalBlock
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from config.sl_config import get_sl_config
from data import MarketDataProvider
from data.historical_store import (
    XAUUSD_M15_30D_FILE,
    load_candles,
    load_historical_dataset,
    save_historical_dataset,
)
from data.providers.base import Candle
from signal_generator import MIN_RR_TO_TP1, TradeSignal, price_distance_pips, planned_rr_to_target
from strategy.runner import (
    build_context,
    build_signal_reason,
    compute_final_decision,
    run_agents,
    slice_candles_as_of,
)
from strategy.runner import TREND_H4_CANDLE_MIN
from strategy.signal_filter import SignalFilter
from tracking.console import configure_console_encoding

DAYS = 30
M15_CANDLES_PER_DAY = 96
WARMUP = 100
LIGHT_SCAN_CANDLES = 500
TOTAL_CANDLES = DAYS * M15_CANDLES_PER_DAY + WARMUP
H1_CANDLES = DAYS * 24 + 250
XAUUSD_PIP_SIZE = get_sl_config("XAUUSD").pip_size  # type: ignore[union-attr]


@dataclass
class BacktestRunStats:
    trend_blocked: int = 0
    sl_blocked: int = 0
    rr_blocked: int = 0
    other_filter_blocked: int = 0
    neutral_decisions: int = 0
    setup_candidates: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    data_file: str = ""
    m15_candles: int = 0
    h1_candles: int = 0
    period_start: str = ""
    period_end: str = ""
    m15_reversal_blocked: int = 0

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
    def tp2_hits(self) -> int:
        return sum(1 for trade in self.trades if trade.tp2_hit)

    @property
    def tp3_hits(self) -> int:
        return sum(1 for trade in self.trades if trade.tp3_hit)

    @property
    def tp2_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.tp2_hits / self.total_signals * 100

    @property
    def tp3_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.tp3_hits / self.total_signals * 100

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
    def avg_rr_tp1(self) -> float:
        if not self.trades:
            return 0.0
        values = [
            planned_rr_to_target(trade.entry, trade.tp1, trade.risk)
            for trade in self.trades
            if trade.risk > 0
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_rr_tp2(self) -> float:
        if not self.trades:
            return 0.0
        values = [
            planned_rr_to_target(trade.entry, trade.tp2, trade.risk)
            for trade in self.trades
            if trade.risk > 0
        ]
        return sum(values) / len(values) if values else 0.0

    @property
    def avg_rr(self) -> float:
        return self.avg_rr_tp1

    @property
    def total_r(self) -> float:
        return sum(trade.pnl_r for trade in self.trades)

    @property
    def stop_losses(self) -> int:
        from tracking.trade_outcome import is_full_stop_loss

        return sum(1 for trade in self.trades if is_full_stop_loss(trade))

    @property
    def breakeven_exits(self) -> int:
        """Pure 0R exits: stop at entry before TP1. After-TP1 closes are wins."""
        return sum(
            1
            for trade in self.trades
            if trade.result == "breakeven" and not trade.tp1_hit
        )

    @property
    def avg_r_per_trade(self) -> float:
        if not self.trades:
            return 0.0
        return self.total_r / self.total_signals


@dataclass
class TradeSetup:
    entry_index: int
    signal: TradeSignal


class LocalDataBacktestEngine(BacktestEngine):
    def __init__(
        self,
        config: BacktestConfig,
        *,
        m15_candles: list[Candle],
        h1_candles: list[Candle],
        h4_candles: list[Candle] | None = None,
        progress_every: int = 50,
        progress_template: str = "Processed {processed}/{total} candles...",
        progress_finish: str = "Scan complete.",
        label: str = f"{DAYS} days",
        signal_filter: SignalFilter | None = None,
    ) -> None:
        super().__init__(config, signal_filter=signal_filter)
        self._m15_candles = m15_candles
        self._h1_candles = h1_candles
        self._h4_candles = h4_candles or []
        self._progress_every = progress_every
        self._progress_template = progress_template
        self._progress_finish = progress_finish
        self._label = label
        self._zone_catalog = ZoneCatalog.from_candles(
            m15_candles,
            self.symbol_def.display,
        )

    def run_with_stats(
        self,
        *,
        data_file: str = "",
        management_mode: TradeManagementMode = TradeManagementMode.PARTIAL,
    ) -> BacktestRunStats:
        setups, scan_stats = self._scan_candles(self._m15_candles)
        stats = self._simulate_setups(setups, self._m15_candles, management_mode)
        stats.setup_candidates = len(setups)
        stats.trend_blocked = scan_stats.trend_blocked
        stats.sl_blocked = scan_stats.sl_blocked
        stats.rr_blocked = scan_stats.rr_blocked
        stats.other_filter_blocked = scan_stats.other_filter_blocked
        stats.neutral_decisions = scan_stats.neutral_decisions
        stats.data_file = data_file
        stats.m15_candles = len(self._m15_candles)
        stats.h1_candles = len(self._h1_candles)
        if len(self._m15_candles) > self.config.warmup_candles:
            test_start = self.config.warmup_candles
            stats.period_start = candle_timestamp(
                self._m15_candles, test_start
            ).isoformat()
            stats.period_end = candle_timestamp(
                self._m15_candles, len(self._m15_candles) - 1
            ).isoformat()
        return stats

    def run_comparison(self, *, data_file: str = "") -> tuple[BacktestRunStats, BacktestRunStats]:
        setups, scan_stats = self._scan_candles(self._m15_candles)
        legacy_stats = self._simulate_setups(
            setups,
            self._m15_candles,
            TradeManagementMode.LEGACY,
        )
        partial_stats = self._simulate_setups(
            setups,
            self._m15_candles,
            TradeManagementMode.PARTIAL,
        )
        for stats in (legacy_stats, partial_stats):
            stats.setup_candidates = len(setups)
            stats.trend_blocked = scan_stats.trend_blocked
            stats.sl_blocked = scan_stats.sl_blocked
            stats.rr_blocked = scan_stats.rr_blocked
            stats.other_filter_blocked = scan_stats.other_filter_blocked
            stats.neutral_decisions = scan_stats.neutral_decisions
            stats.data_file = data_file
            stats.m15_candles = len(self._m15_candles)
            stats.h1_candles = len(self._h1_candles)
            if len(self._m15_candles) > self.config.warmup_candles:
                test_start = self.config.warmup_candles
                stats.period_start = candle_timestamp(
                    self._m15_candles, test_start
                ).isoformat()
                stats.period_end = candle_timestamp(
                    self._m15_candles, len(self._m15_candles) - 1
                ).isoformat()
        return legacy_stats, partial_stats

    def _scan_candles(self, candles: list[Candle]) -> tuple[list[TradeSetup], BacktestRunStats]:
        setups: list[TradeSetup] = []
        stats = BacktestRunStats()
        h1_candles = self._h1_candles
        h4_candles = self._h4_candles
        scan_start = self.config.warmup_candles
        scan_end = len(candles) - 1
        progress = BacktestScanProgress(
            scan_start,
            scan_end,
            update_every=self._progress_every,
            message_template=self._progress_template,
            finish_message=self._progress_finish,
        )

        for index in range(scan_start, scan_end):
            progress.update(index)
            history = candles[: index + 1]
            timestamp = candle_timestamp(candles, index)
            context = build_context(
                symbol=self.symbol_def.display,
                candles=history,
                timeframe=self.config.timeframe,
                timestamp=timestamp,
                h1_candles=slice_candles_as_of(h1_candles, timestamp),
                h4_candles=slice_candles_as_of(
                    h4_candles,
                    timestamp,
                    limit=TREND_H4_CANDLE_MIN,
                )
                if h4_candles
                else None,
            )
            context["zone_catalog"] = self._zone_catalog
            context["bar_index"] = index

            agent_results = run_agents(context)
            direction, confidence, _, _ = compute_final_decision(
                agent_results,
                self.signal_filter.decision_config,
            )

            if direction == Direction.NEUTRAL:
                stats.neutral_decisions += 1
                continue

            filter_result = self.signal_filter.evaluate(
                agent_results,
                direction,
                confidence,
                symbol=self.symbol_def.display,
                timestamp=context.get("timestamp"),
                context=context,
            )
            if not filter_result.approved:
                if SignalFilter._evaluate_trend_hard_block(
                    agent_results.get("trend_filter"),
                    direction,
                ):
                    stats.trend_blocked += 1
                elif (
                    self.signal_filter.require_h4_h1_alignment
                    and filter_result.message.startswith("NO TRADE: H4")
                ):
                    stats.trend_blocked += 1
                elif (
                    self.signal_filter.require_entry_zone
                    and "entry zone" in filter_result.message
                ):
                    stats.other_filter_blocked += 1
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
                reason = generation.rejection_reason or ""
                if reason.startswith("RR rejected"):
                    stats.rr_blocked += 1
                else:
                    stats.sl_blocked += 1
                continue

            setups.append(TradeSetup(entry_index=index, signal=generation.signal))

        progress.finish()
        return setups, stats

    def _simulate_setups(
        self,
        setups: list[TradeSetup],
        candles: list[Candle],
        management_mode: TradeManagementMode,
        *,
        enable_m15_reversal_block: bool = False,
    ) -> BacktestRunStats:
        stats = BacktestRunStats()
        open_until_index = -1
        simulator = TradeSimulator()
        use_near_tp1_be = management_mode == TradeManagementMode.PARTIAL_NEAR_TP1_BE
        block = BacktestM15ReversalBlock() if enable_m15_reversal_block else None

        for setup in setups:
            index = setup.entry_index
            if index <= open_until_index:
                continue

            if block is not None and block.blocks_setup(
                setup.signal,
                candles,
                index,
                symbol=self.symbol_def.display,
                zone_catalog=self._zone_catalog,
            ):
                continue

            simulated = simulator.simulate(
                setup.signal,
                candles[index + 1 :],
                entry_index=index,
                mode=management_mode,
                all_candles=candles if use_near_tp1_be else None,
                zone_catalog=self._zone_catalog if use_near_tp1_be else None,
                symbol=self.symbol_def.display,
            )
            if simulated is None:
                continue

            stats.trades.append(simulated)
            open_until_index = simulated.exit_index
            if block is not None:
                block.register_from_trade(simulated)

        if block is not None:
            stats.m15_reversal_blocked = block.blocked_setups

        return stats


def print_management_summary(label: str, stats: BacktestRunStats) -> None:
    print(f"--- {label} ---")
    print(f"Trades:                       {stats.total_signals}")
    print(
        f"TP1 / TP2 / TP3:              "
        f"{stats.tp1_wins}/{stats.tp2_hits}/{stats.tp3_hits}"
    )
    print(f"Total result (30 days):       {stats.total_r:+.2f}R")
    print(f"Average per trade:            {stats.avg_r_per_trade:+.2f}R")


def print_comparison(
    legacy_stats: BacktestRunStats,
    partial_stats: BacktestRunStats,
    *,
    label: str = f"{DAYS} days",
    test_candles: int | None = None,
) -> None:
    test_count = test_candles or DAYS * M15_CANDLES_PER_DAY
    print(f"=== XAUUSD M15 Backtest ({label}, local data) ===")
    print(f"Data file: {legacy_stats.data_file}")
    print(
        f"Candles: {legacy_stats.m15_candles} M15 ({WARMUP} warmup + "
        f"{test_count} test), {legacy_stats.h1_candles} H1"
    )
    if legacy_stats.period_start and legacy_stats.period_end:
        print(f"Test window: {legacy_stats.period_start} -> {legacy_stats.period_end}")
    print()
    print(f"Signal candidates (before overlap): {legacy_stats.setup_candidates}")
    print(f"Blocked by TrendFilter:       {legacy_stats.trend_blocked}")
    print(f"Blocked by SL validation:     {legacy_stats.sl_blocked}")
    print(f"Blocked by min RR (<{MIN_RR_TO_TP1:.1f}R): {legacy_stats.rr_blocked}")
    print(f"Blocked by other filters:     {legacy_stats.other_filter_blocked}")
    print(f"Neutral decisions (skipped):  {legacy_stats.neutral_decisions}")
    print()
    print("=== Trade management comparison ===")
    print()
    print_management_summary("Old logic (100% position, SL -> entry at TP1)", legacy_stats)
    print()
    print_management_summary(
        "New logic (50% at TP1, +25% at TP2, 25% to TP3)",
        partial_stats,
    )
    print()
    delta_total = partial_stats.total_r - legacy_stats.total_r
    delta_avg = partial_stats.avg_r_per_trade - legacy_stats.avg_r_per_trade
    print(f"Improvement total:            {delta_total:+.2f}R")
    print(f"Improvement avg/trade:        {delta_avg:+.2f}R")
    print()
    print(f"Average SL:                   {partial_stats.avg_sl_pips:.1f} pips")
    print(f"Average planned RR to TP1:    {partial_stats.avg_rr_tp1:.2f}R")
    print(f"Average planned RR to TP2:    {partial_stats.avg_rr_tp2:.2f}R")

    if partial_stats.trades:
        print()
        print("New logic trades:")
        for trade in partial_stats.trades:
            sl_pips = price_distance_pips(trade.risk, XAUUSD_PIP_SIZE)
            legacy_trade = next(
                (
                    item
                    for item in legacy_stats.trades
                    if item.entry_index == trade.entry_index
                ),
                None,
            )
            legacy_r = legacy_trade.pnl_r if legacy_trade else 0.0
            print(
                f"  {trade.direction.upper()} | entry={trade.entry:.2f} "
                f"({sl_pips:.1f} pips SL) | {trade.result} | "
                f"old={legacy_r:+.2f}R new={trade.pnl_r:+.2f}R"
            )


def download_xauusd_dataset(
    *,
    days: int = DAYS,
    warmup: int = WARMUP,
    output_file: Path = XAUUSD_M15_30D_FILE,
    force: bool = False,
) -> Path:
    if output_file.exists() and not force:
        return output_file

    m15_needed = days * M15_CANDLES_PER_DAY + warmup
    h1_needed = days * 24 + 250

    provider = MarketDataProvider()
    print(
        f"Downloading XAUUSD (XAUUSDT) from {provider.data_source('XAUUSD')} "
        f"— M15 x{m15_needed}, H1 x{h1_needed}..."
    )

    m15_candles = provider.get_historical_market_data("XAUUSD", "15m", m15_needed)
    h1_candles = provider.get_historical_market_data("XAUUSD", "1h", h1_needed)

    if len(m15_candles) < warmup + days * M15_CANDLES_PER_DAY:
        raise RuntimeError(
            f"Not enough M15 candles: got {len(m15_candles)}, need "
            f"{warmup + days * M15_CANDLES_PER_DAY}"
        )

    saved = save_historical_dataset(
        output_file,
        symbol="XAUUSD",
        data_symbol="XAUUSDT",
        source=provider.data_source("XAUUSD"),
        period_days=days,
        candles_by_timeframe={
            "15m": m15_candles,
            "1h": h1_candles,
        },
        metadata={
            "warmup_candles": warmup,
            "m15_requested": m15_needed,
            "h1_requested": h1_needed,
        },
    )
    print(f"Saved {len(m15_candles)} M15 + {len(h1_candles)} H1 candles to: {saved}")
    return saved


def print_stats(stats: BacktestRunStats) -> None:
    print(f"=== XAUUSD M15 Backtest (last {DAYS} days, local data) ===")
    print(f"Data file: {stats.data_file}")
    print(
        f"Candles: {stats.m15_candles} M15 ({WARMUP} warmup + "
        f"{DAYS * M15_CANDLES_PER_DAY} test), {stats.h1_candles} H1"
    )
    if stats.period_start and stats.period_end:
        print(f"Test window: {stats.period_start} -> {stats.period_end}")
    print()
    print(f"Total signals (executed):     {stats.total_signals}")
    print(f"  LONG:                       {stats.long_signals}")
    print(f"  SHORT:                      {stats.short_signals}")
    print()
    print(f"Blocked by TrendFilter:       {stats.trend_blocked}")
    print(f"Blocked by SL validation:     {stats.sl_blocked}")
    print(f"Blocked by min RR (<{MIN_RR_TO_TP1:.1f}R): {stats.rr_blocked}")
    print(f"Blocked by other filters:     {stats.other_filter_blocked}")
    print(f"Neutral decisions (skipped):  {stats.neutral_decisions}")
    print()
    print("--- Take Profit breakdown ---")
    print(
        f"TP1 hit:                      {stats.tp1_wins}/{stats.total_signals} "
        f"({stats.win_rate:.1f}%)"
    )
    print(
        f"TP2 hit:                      {stats.tp2_hits}/{stats.total_signals} "
        f"({stats.tp2_rate:.1f}%)"
    )
    print(
        f"TP3 hit:                      {stats.tp3_hits}/{stats.total_signals} "
        f"({stats.tp3_rate:.1f}%)"
    )
    print()
    print(f"Average planned RR to TP1:    {stats.avg_rr_tp1:.2f}R (min {MIN_RR_TO_TP1:.1f}R)")
    print(f"Average planned RR to TP2:    {stats.avg_rr_tp2:.2f}R")
    print(f"Average SL:                   {stats.avg_sl_pips:.1f} pips")

    if stats.trades:
        print()
        print("Trades:")
        for trade in stats.trades:
            sl_pips = price_distance_pips(trade.risk, XAUUSD_PIP_SIZE)
            print(
                f"  {trade.direction.upper()} | entry={trade.entry:.2f} "
                f"sl={trade.stop_loss:.2f} ({sl_pips:.1f} pips) | "
                f"{trade.result} | TP1={'yes' if trade.tp1_hit else 'no'} | "
                f"TP2={'yes' if trade.tp2_hit else 'no'} | "
                f"TP3={'yes' if trade.tp3_hit else 'no'} | "
                f"{trade.pnl_r:.2f}R"
            )


def slice_m15_window(
    m15_candles: list[Candle],
    *,
    scan_candles: int,
    warmup: int = WARMUP,
) -> list[Candle]:
    needed = warmup + scan_candles + 1
    if len(m15_candles) < needed:
        raise SystemExit(
            f"Not enough M15 candles for window: need {needed}, got {len(m15_candles)}"
        )
    return m15_candles[-needed:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download XAUUSD M15 30-day history and run backtest"
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=XAUUSD_M15_30D_FILE,
        help=f"Local JSON dataset path (default: {XAUUSD_M15_30D_FILE.name})",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download and save data, skip backtest",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Re-download even if local file exists",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use existing local file without downloading",
    )
    parser.add_argument(
        "--light",
        action="store_true",
        help=(
            f"Fast backtest on last {LIGHT_SCAN_CANDLES} M15 candles "
            f"(instead of full {DAYS}-day window)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    configure_console_encoding()
    args = parse_args()

    data_file = args.data_file
    if args.skip_download:
        if not data_file.exists():
            raise SystemExit(f"Local data file not found: {data_file}")
        print(f"Using existing data: {data_file}")
    else:
        data_file = download_xauusd_dataset(
            output_file=data_file,
            force=args.force_download,
        )

    if args.download_only:
        dataset = load_historical_dataset(data_file)
        m15 = dataset["timeframes"]["15m"]
        print()
        print("Download complete.")
        print(f"  M15: {m15['candle_count']} candles")
        print(f"       {m15['first_open_time']} -> {m15['last_open_time']}")
        return

    m15_candles = load_candles(data_file, "15m")
    h1_candles = load_candles(data_file, "1h")

    if args.light:
        m15_candles = slice_m15_window(m15_candles, scan_candles=LIGHT_SCAN_CANDLES)
        first_ts = candle_timestamp(m15_candles, 0)
        h1_candles = slice_candles_as_of(h1_candles, first_ts, limit=len(h1_candles))
        progress_every = 100
        progress_template = "Оброблено {processed}/{total}..."
        progress_finish = "Сканування завершено."
        label = f"light {LIGHT_SCAN_CANDLES} candles"
        test_candles = LIGHT_SCAN_CANDLES
    else:
        progress_every = 50
        progress_template = "Processed {processed}/{total} candles..."
        progress_finish = "Scan complete."
        label = f"last {DAYS} days"
        test_candles = max(0, len(m15_candles) - WARMUP - 1)

    print()
    print("Running backtest on local historical data...")
    print(f"Scanning {test_candles} candles...", flush=True)
    print()

    engine = LocalDataBacktestEngine(
        BacktestConfig(
            symbol="XAUUSD",
            timeframe="15m",
            total_candles=len(m15_candles),
            warmup_candles=WARMUP,
        ),
        m15_candles=m15_candles,
        h1_candles=h1_candles,
        progress_every=progress_every,
        progress_template=progress_template,
        progress_finish=progress_finish,
        label=label,
    )
    legacy_stats, partial_stats = engine.run_comparison(data_file=str(data_file))
    print_comparison(
        legacy_stats,
        partial_stats,
        label=label,
        test_candles=test_candles,
    )


if __name__ == "__main__":
    main()
