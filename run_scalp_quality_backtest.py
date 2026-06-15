"""Compare scalp quality filters: session core 08-16 UTC, min pierce 10p, news block."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from agents.session_agent import is_london_or_new_york_session
from backtest.engine import candle_timestamp
from backtest.progress import BacktestScanProgress
from backtest.simulator import TradeManagementMode, TradeSimulator
from config.symbols import resolve_symbol
from data import MarketDataProvider
from news import build_news_gate
from news.models import NewsAction
from run_liquidity_scalp_backtest import (
    DETECTION_WINDOW,
    LiquidityScalpStats,
    period_days,
    WARMUP,
)
from strategy.liquidity_scalp import (
    DEFAULT_LIQUIDITY_SCALP_CONFIG,
    build_liquidity_scalp_gate,
    build_liquidity_scalp_signal,
    detect_liquidity_sweep_setup,
)
from strategy.runner import slice_candles_as_of
from strategy.scalp_mode import ScalpPublishGate
from tracking.console import configure_console_encoding
from tracking.trade_pnl import pip_size_for_symbol

SCAN_CANDLES = 25000


def _utc(ts: datetime) -> datetime:
    return ts.replace(tzinfo=timezone.utc) if ts.tzinfo is None else ts.astimezone(timezone.utc)


def in_core_london_ny_window(ts: datetime) -> bool:
    """08:00-16:00 UTC (user's quality window)."""
    ts = _utc(ts)
    minute = ts.hour * 60 + ts.minute
    return 8 * 60 <= minute < 16 * 60


def in_asia_flat(ts: datetime) -> bool:
    """00:00-06:00 UTC — skip flat Asia."""
    ts = _utc(ts)
    return 0 <= ts.hour < 6


def session_allowed(ts: datetime, mode: str) -> bool:
    if in_asia_flat(ts):
        return False
    if mode == "core_08_16":
        return in_core_london_ny_window(ts)
    return is_london_or_new_york_session(ts)


@dataclass
class QualityStats(LiquidityScalpStats):
    weak_sweep: int = 0
    news_blocked: int = 0


def scan_with_quality_filters(
    candles: list,
    symbol: str = "XAUUSD",
    *,
    session_mode: str = "london_ny",
    min_pierce_pips: float | None = None,
    use_news_block: bool = False,
    news_buffer_minutes: int = 30,
    max_signals_per_day: int = 6,
    interval_minutes: int = 5,
    h1_candles: list | None = None,
) -> QualityStats:
    display = resolve_symbol(symbol).display
    pip = pip_size_for_symbol(display) or 1.0
    gate = ScalpPublishGate(
        min_interval_seconds=interval_minutes * 60,
        max_signals_per_day=max_signals_per_day,
    )
    news_gate = None
    if use_news_block:
        news_gate = build_news_gate(
            enabled=True,
            buffer_minutes=news_buffer_minutes,
        )

    simulator = TradeSimulator()
    stats = QualityStats()
    open_until = -1
    scan_start = WARMUP
    scan_end = len(candles) - 1
    progress = BacktestScanProgress(
        scan_start,
        scan_end,
        update_every=10_000_000,
        message_template="",
        finish_message="",
    )

    for index in range(scan_start, scan_end):
        progress.update(index)
        timestamp = candle_timestamp(candles, index)
        if not session_allowed(timestamp, session_mode):
            stats.off_session += 1
            continue

        if use_news_block and news_gate is not None:
            news = news_gate.evaluate(display, timestamp)
            if news.action == NewsAction.BLOCK:
                stats.news_blocked += 1
                continue

        window_start = max(0, index + 1 - DETECTION_WINDOW)
        window = candles[window_start : index + 1]
        h1_history = slice_candles_as_of(h1_candles, timestamp) if h1_candles else None
        setup, reason = detect_liquidity_sweep_setup(
            window,
            display,
            config=DEFAULT_LIQUIDITY_SCALP_CONFIG,
            h1_candles=h1_history,
        )
        if setup is None:
            if "exceeds max" in reason:
                stats.sl_too_wide += 1
            else:
                stats.no_sweep += 1
            continue

        if min_pierce_pips is not None:
            pierce = abs(setup.sweep_extreme - setup.pool_level) / pip
            if pierce < min_pierce_pips:
                stats.weak_sweep += 1
                continue

        if index <= open_until:
            stats.slot_busy += 1
            continue

        allowed, _ = gate.can_publish(display, timestamp)
        if not allowed:
            stats.rate_limited += 1
            continue

        signal = build_liquidity_scalp_signal(setup, display)
        simulated = simulator.simulate(
            signal,
            candles[index + 1 :],
            entry_index=index,
            mode=TradeManagementMode.PARTIAL,
        )
        if simulated is None:
            continue

        stats.trades.append(simulated)
        gate.record(display, timestamp)
        open_until = simulated.exit_index

    if scan_end > scan_start:
        stats.period_start = candle_timestamp(candles, scan_start).isoformat()
        stats.period_end = candle_timestamp(candles, scan_end).isoformat()
    return stats


def run_turtle_variant(
    candles: list,
    *,
    session_mode: str,
    max_day: int = 7,
) -> tuple[int, float, float, int, int]:
    from run_turtle_soup_backtest import run as run_turtle

    level_mode = "all"
    entry_mode = "close"
    session_map = {"london_ny": "london_ny", "core_08_16": "core_08_16"}
    stats = run_turtle(
        candles,
        "XAUUSD",
        interval_min=5,
        max_day=max_day,
        entry_mode=entry_mode,
        level_mode=level_mode,
        session_mode=session_map.get(session_mode, "london_ny"),  # type: ignore[arg-type]
    )
    days = max(
        (
            datetime.fromisoformat(stats.period_end)
            - datetime.fromisoformat(stats.period_start)
        ).total_seconds()
        / 86_400,
        0.01,
    )
    return (
        stats.total_signals,
        stats.total_signals / days,
        stats.win_rate,
        stats.full_stops,
        stats.total_r,
    )


def print_row(label: str, stats: QualityStats) -> None:
    days = period_days(stats)
    wr = f"{stats.win_rate:.0f}%" if stats.total_signals else "—"
    avg = stats.total_r / stats.total_signals if stats.total_signals else 0.0
    print(
        f"{label:<28} {stats.total_signals:>5} {stats.total_signals / days:>6.2f} "
        f"{wr:>7} {stats.full_stops:>5} {stats.tp1_then_be:>4} "
        f"{stats.total_r:>+8.2f}R {avg:>+6.2f}R",
        flush=True,
    )


def main() -> None:
    configure_console_encoding()
    symbol = "XAUUSD"
    needed = WARMUP + SCAN_CANDLES + 1
    provider = MarketDataProvider()
    print(f"Завантаження {symbol} M1 x{needed}...", flush=True)
    m1 = provider.get_historical_market_data(symbol, "1m", needed)
    print(f"Завантаження {symbol} H1 x400...", flush=True)
    h1 = provider.get_historical_market_data(symbol, "1h", 400)
    print(f"Отримано {len(m1)} M1 свічок\n", flush=True)

    print("=== Liquidity Scalp M1 (prod vol+StochRSI) — фільтри якості ===")
    print(
        f"{'Варіант':<28} {'Угод':>5} {'/день':>6} {'WR':>7} "
        f"{'Стоп':>5} {'BE':>4} {'TotalR':>9} {'R/уг':>7}"
    )
    print("-" * 82)

    variants = [
        ("A: prod (London/NY)", dict(session_mode="london_ny", max_signals_per_day=6)),
        ("B: loose 15/day", dict(session_mode="london_ny", max_signals_per_day=15)),
        ("C: вікно 08-16 UTC", dict(session_mode="core_08_16", max_signals_per_day=15)),
        (
            "D: 08-16 + pierce>=10",
            dict(session_mode="core_08_16", min_pierce_pips=10.0, max_signals_per_day=15),
        ),
        (
            "E: D + news 30m",
            dict(
                session_mode="core_08_16",
                min_pierce_pips=10.0,
                use_news_block=True,
                news_buffer_minutes=30,
                max_signals_per_day=15,
            ),
        ),
        (
            "F: TOP (E, max 10/day)",
            dict(
                session_mode="core_08_16",
                min_pierce_pips=10.0,
                use_news_block=True,
                news_buffer_minutes=30,
                max_signals_per_day=10,
            ),
        ),
    ]

    best_label = ""
    best_r = float("-inf")
    best_stats: QualityStats | None = None

    for label, kwargs in variants:
        stats = scan_with_quality_filters(m1, h1_candles=h1, **kwargs)
        print_row(label, stats)
        if stats.total_r > best_r:
            best_r = stats.total_r
            best_label = label
            best_stats = stats

    if best_stats:
        days = period_days(best_stats)
        print()
        print(f"Найкращий M1: {best_label} | {best_stats.total_r:+.2f}R / {days:.1f}d")

    # Turtle Soup M5 — add core session to backtest if missing
    print("\n=== Turtle Soup M5 (close, all levels) ===")
    m5_needed = 800 + 6000
    print(f"Завантаження {symbol} M5 x{m5_needed}...", flush=True)
    m5 = provider.get_historical_market_data(symbol, "5m", m5_needed)
    print(f"{'Варіант':<28} {'Угод':>5} {'/день':>6} {'WR':>7} {'Стоп':>5} {'TotalR':>9}")
    print("-" * 65)

    for label, sess in [("London/NY (prod VIP2)", "london_ny"), ("08-16 UTC core", "core_08_16")]:
        try:
            n, per_day, wr, stops, total_r = run_turtle_variant(
                m5, session_mode=sess, max_day=7
            )
            wr_s = f"{wr:.0f}%" if n else "—"
            print(
                f"{label:<28} {n:>5} {per_day:>6.2f} {wr_s:>7} {stops:>5} {total_r:>+8.2f}R",
                flush=True,
            )
        except Exception as exc:
            print(f"{label:<28}  помилка: {exc}", flush=True)


if __name__ == "__main__":
    main()
