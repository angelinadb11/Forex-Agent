"""Backtest of the liquidity-sweep scalp strategy (XAUUSD, M5)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from backtest.engine import candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.simulator import SimulatedTradeResult, TradeManagementMode, TradeSimulator
from agents.session_agent import is_london_or_new_york_session
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import TradeSignal
from strategy.liquidity_scalp import (
    DEFAULT_LIQUIDITY_SCALP_CONFIG,
    LiquidityScalpConfig,
    build_liquidity_scalp_gate,
    build_liquidity_scalp_signal,
    detect_liquidity_sweep_setup,
)
from strategy.runner import slice_candles_as_of
from tracking.console import configure_console_encoding
from tracking.trade_outcome import is_full_stop_loss

WARMUP = 100
SCAN_CANDLES = 6000  # ~3 тижні M5
DETECTION_WINDOW = 240  # 20 годин історії для пошуку пулів
PROGRESS_EVERY = 500


@dataclass
class LiquidityScalpStats:
    no_sweep: int = 0
    off_session: int = 0
    sl_too_wide: int = 0
    rate_limited: int = 0
    slot_busy: int = 0
    trades: list[SimulatedTradeResult] = field(default_factory=list)
    period_start: str = ""
    period_end: str = ""

    @property
    def total_signals(self) -> int:
        return len(self.trades)

    @property
    def tp1_wins(self) -> int:
        return sum(1 for trade in self.trades if trade.tp1_hit)

    @property
    def win_rate(self) -> float:
        if not self.trades:
            return 0.0
        return self.tp1_wins / self.total_signals * 100

    @property
    def full_stops(self) -> int:
        return sum(1 for trade in self.trades if is_full_stop_loss(trade))

    @property
    def breakeven_exits(self) -> int:
        return sum(
            1
            for trade in self.trades
            if trade.result == "breakeven" and not trade.tp1_hit
        )

    @property
    def tp2_full_wins(self) -> int:
        return sum(1 for trade in self.trades if trade.tp2_hit)

    @property
    def tp1_then_be(self) -> int:
        return sum(
            1
            for trade in self.trades
            if trade.tp1_hit and not trade.tp2_hit and trade.result == "breakeven"
        )

    @property
    def other_exits(self) -> int:
        return (
            self.total_signals
            - self.tp2_full_wins
            - self.tp1_then_be
            - self.full_stops
            - self.breakeven_exits
        )

    @property
    def avg_duration_candles(self) -> float:
        if not self.trades:
            return 0.0
        return sum(t.exit_index - t.entry_index for t in self.trades) / len(self.trades)

    @property
    def total_r(self) -> float:
        return sum(trade.pnl_r for trade in self.trades)


@dataclass(frozen=True)
class ScalpTradeSetup:
    entry_index: int
    signal: TradeSignal


def scan_and_simulate(
    m5_candles: list,
    symbol: str = "XAUUSD",
    *,
    config: LiquidityScalpConfig = DEFAULT_LIQUIDITY_SCALP_CONFIG,
    h1_candles: list | None = None,
    quiet: bool = False,
    gate=None,
) -> LiquidityScalpStats:
    display = resolve_symbol(symbol).display
    gate = gate if gate is not None else build_liquidity_scalp_gate()
    simulator = TradeSimulator()
    stats = LiquidityScalpStats()
    open_until_index = -1

    scan_start = WARMUP
    scan_end = len(m5_candles) - 1
    progress = BacktestScanProgress(
        scan_start,
        scan_end,
        update_every=10_000_000 if quiet else PROGRESS_EVERY,
        message_template="Оброблено {processed}/{total} M5 свічок...",
        finish_message="" if quiet else "Сканування завершено.",
    )

    for index in range(scan_start, scan_end):
        progress.update(index)
        timestamp = candle_timestamp(m5_candles, index)

        window_start = max(0, index + 1 - DETECTION_WINDOW)
        window = m5_candles[window_start : index + 1]
        h1_history = (
            slice_candles_as_of(h1_candles, timestamp) if h1_candles else None
        )
        setup, reason = detect_liquidity_sweep_setup(
            window,
            display,
            config=config,
            h1_candles=h1_history,
        )
        if setup is None:
            if "exceeds max" in reason:
                stats.sl_too_wide += 1
            else:
                stats.no_sweep += 1
            continue

        if not is_london_or_new_york_session(timestamp):
            stats.off_session += 1
            continue

        if index <= open_until_index:
            stats.slot_busy += 1
            continue

        allowed, _ = gate.can_publish(display, timestamp)
        if not allowed:
            stats.rate_limited += 1
            continue

        signal = build_liquidity_scalp_signal(setup, display)
        simulated = simulator.simulate(
            signal,
            m5_candles[index + 1 :],
            entry_index=index,
            mode=TradeManagementMode.PARTIAL,
        )
        if simulated is None:
            continue

        stats.trades.append(simulated)
        gate.record(display, timestamp)
        open_until_index = simulated.exit_index

    progress.finish()
    if scan_end > scan_start:
        stats.period_start = candle_timestamp(m5_candles, scan_start).isoformat()
        stats.period_end = candle_timestamp(m5_candles, scan_end).isoformat()
    return stats


def period_days(stats: LiquidityScalpStats) -> float:
    if not stats.period_start or not stats.period_end:
        return SCAN_CANDLES * 5 / (60 * 24)
    start = datetime.fromisoformat(stats.period_start)
    end = datetime.fromisoformat(stats.period_end)
    return max((end - start).total_seconds() / 86_400, 5 / (60 * 24))


def print_report(stats: LiquidityScalpStats) -> None:
    days = period_days(stats)
    print()
    print("=== Liquidity Scalp Backtest (XAUUSD) ===")
    print(
        "Sweep equal lows/highs + reclaim | SL за хвіст + ATR-буфер | "
        "TP1=1R (50%, стоп у БЕ), TP2=ліквідність/2R"
    )
    print("Сесії London/NY (ліміти гейта і SL — у конфігу вище)")
    if stats.period_start:
        print(f"Період: {stats.period_start[:16]} -> {stats.period_end[:16]} (~{days:.1f} днів)")
    print()
    print(f"Без sweep-сетапу:       {stats.no_sweep}")
    print(f"SL занадто широкий:     {stats.sl_too_wide}")
    print(f"Поза сесією:            {stats.off_session}")
    print(f"Зайнятий слот:          {stats.slot_busy}")
    print(f"Ліміт частоти:          {stats.rate_limited}")
    print()
    print(f"Угод:                   {stats.total_signals} ({stats.total_signals / days:.2f}/день)")
    if stats.total_signals:
        print(
            f"Win rate (TP1):         {stats.tp1_wins}/{stats.total_signals} "
            f"({stats.win_rate:.1f}%)"
        )
    print()
    print(f"  ТП1 + ТП2 (повний):   {stats.tp2_full_wins} (+1.5R кожна)")
    print(f"  ТП1 -> БЕ (решта):    {stats.tp1_then_be} (+0.5R кожна)")
    print(f"  Повний стоп (-1R):    {stats.full_stops}")
    print(f"  БЕ до ТП1 (0R):       {stats.breakeven_exits}")
    if stats.other_exits:
        print(f"  Інші виходи:          {stats.other_exits}")
    print()
    print(f"Середня тривалість:     {stats.avg_duration_candles:.0f} свічок")
    print(f"Total R:                {stats.total_r:+.2f}R")
    if stats.total_signals:
        print(f"Середнє на угоду:       {stats.total_r / stats.total_signals:+.2f}R")


VARIANTS: list[tuple[str, LiquidityScalpConfig]] = [
    ("A: базовий", LiquidityScalpConfig()),
    ("B: пул >=3 дотиків", LiquidityScalpConfig(min_pool_touches=3)),
    ("C: сильний reclaim", LiquidityScalpConfig(require_directional_close=True)),
    ("D: хвіст >=0.3 ATR", LiquidityScalpConfig(min_wick_atr_mult=0.3)),
    (
        "E: C+D",
        LiquidityScalpConfig(require_directional_close=True, min_wick_atr_mult=0.3),
    ),
    (
        "F: C+D+H1 тренд",
        LiquidityScalpConfig(
            require_directional_close=True,
            min_wick_atr_mult=0.3,
            require_h1_trend=True,
        ),
    ),
    (
        "G: B+C+D+H1 тренд",
        LiquidityScalpConfig(
            min_pool_touches=3,
            require_directional_close=True,
            min_wick_atr_mult=0.3,
            require_h1_trend=True,
        ),
    ),
]


def main() -> None:
    import argparse

    from strategy.scalp_mode import ScalpPublishGate
    from strategy.liquidity_scalp import (
        LIQUIDITY_SCALP_MAX_SIGNALS_PER_DAY,
    )

    configure_console_encoding()
    parser = argparse.ArgumentParser(description="Liquidity scalp backtest")
    parser.add_argument("--min-touches", type=int, default=None)
    parser.add_argument("--wick", type=float, default=None, help="min wick / ATR")
    parser.add_argument("--trend", action="store_true", help="require H1 trend")
    parser.add_argument("--close", action="store_true", help="require directional close")
    parser.add_argument("--ha", action="store_true", help="require Heiken Ashi confirm")
    parser.add_argument("--volume", type=float, default=None, help="min volume ratio")
    parser.add_argument("--stoch", action="store_true", help="require Stoch RSI extreme")
    parser.add_argument("--bb", action="store_true", help="require sweep wick beyond Bollinger Band")
    parser.add_argument("--tf", default="5m", choices=["1m", "5m"], help="timeframe")
    parser.add_argument("--candles", type=int, default=None, help="scan candles")
    parser.add_argument("--interval-min", type=int, default=30, help="min gap, minutes")
    parser.add_argument("--max-day", type=int, default=LIQUIDITY_SCALP_MAX_SIGNALS_PER_DAY)
    parser.add_argument("--sl-min", type=float, default=None, help="min SL pips")
    parser.add_argument("--sl-max", type=float, default=None, help="max SL pips")
    parser.add_argument(
        "--drop-last",
        type=int,
        default=0,
        help="drop N most recent candles (out-of-sample check)",
    )
    args = parser.parse_args()

    filters_used = (
        args.min_touches is not None
        or args.wick is not None
        or args.trend
        or args.close
        or args.ha
        or args.volume is not None
        or args.stoch
        or args.bb
        or args.sl_min is not None
        or args.sl_max is not None
    )
    single_config: LiquidityScalpConfig | None = None
    if filters_used or args.tf == "1m":
        default = DEFAULT_LIQUIDITY_SCALP_CONFIG
        single_config = LiquidityScalpConfig(
            min_pool_touches=args.min_touches if args.min_touches is not None else 2,
            require_directional_close=args.close,
            min_wick_atr_mult=args.wick if args.wick is not None else 0.0,
            require_h1_trend=args.trend,
            require_heiken_ashi=args.ha,
            require_volume_spike=args.volume is not None,
            min_volume_ratio=args.volume if args.volume is not None else 1.5,
            require_stoch_rsi=args.stoch,
            require_bollinger=args.bb,
            min_sl_pips=args.sl_min if args.sl_min is not None else default.min_sl_pips,
            max_sl_pips=args.sl_max if args.sl_max is not None else default.max_sl_pips,
        )

    symbol = "XAUUSD"
    scan_candles = args.candles if args.candles is not None else SCAN_CANDLES
    needed = WARMUP + scan_candles + 1 + args.drop_last
    provider = MarketDataProvider()
    print(f"Завантаження {symbol} {args.tf} x{needed}...", flush=True)
    m5_candles = provider.get_historical_market_data(symbol, args.tf, needed)
    print(f"Завантаження {symbol} H1 x400...", flush=True)
    h1_candles = provider.get_historical_market_data(symbol, "1h", 400)
    print(f"Отримано {len(m5_candles)} {args.tf} свічок", flush=True)
    if args.drop_last:
        m5_candles = m5_candles[: -args.drop_last]
        print(f"Відкинуто останні {args.drop_last} свічок", flush=True)

    if single_config is not None:
        gate = ScalpPublishGate(
            min_interval_seconds=args.interval_min * 60,
            max_signals_per_day=args.max_day,
        )
        print(f"Конфіг: {single_config}", flush=True)
        print(
            f"Гейт: інтервал {args.interval_min} хв, макс {args.max_day}/день",
            flush=True,
        )
        stats = scan_and_simulate(
            m5_candles,
            symbol,
            config=single_config,
            h1_candles=h1_candles,
            quiet=True,
            gate=gate,
        )
        print_report(stats)
        return

    print("\n=== Порівняння варіантів фільтрів ===\n", flush=True)
    print(
        f"{'Варіант':<22} {'Угод':>5} {'/день':>6} {'WR':>7} "
        f"{'Стопів':>7} {'Total R':>9} {'R/угоду':>8}"
    )
    print("-" * 70)

    best: tuple[str, LiquidityScalpConfig, LiquidityScalpStats] | None = None
    for label, config in VARIANTS:
        stats = scan_and_simulate(
            m5_candles, symbol, config=config, h1_candles=h1_candles, quiet=True
        )
        days = period_days(stats)
        wr = f"{stats.win_rate:.0f}%" if stats.total_signals else "—"
        avg = stats.total_r / stats.total_signals if stats.total_signals else 0.0
        print(
            f"{label:<22} {stats.total_signals:>5} "
            f"{stats.total_signals / days:>6.2f} {wr:>7} "
            f"{stats.full_stops:>7} {stats.total_r:>+8.2f}R {avg:>+7.2f}R",
            flush=True,
        )
        if best is None or stats.total_r > best[2].total_r:
            best = (label, config, stats)

    if best is not None:
        print()
        print(f"=== Найкращий варіант: {best[0]} ===")
        print_report(best[2])


if __name__ == "__main__":
    main()
