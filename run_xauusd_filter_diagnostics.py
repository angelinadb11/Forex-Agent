"""Diagnose which filters block XAUUSD signals during a light 500-candle scan."""

from __future__ import annotations

from dataclasses import dataclass, field

from agents.base import Direction
from backtest.engine import BacktestConfig, candle_timestamp
from config.symbols import resolve_symbol
from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import (
    LIGHT_SCAN_CANDLES,
    WARMUP,
    LocalDataBacktestEngine,
    slice_m15_window,
)
from strategy.runner import (
    TREND_H4_CANDLE_MIN,
    build_context,
    build_signal_reason,
    compute_final_decision,
    count_primary_agreement,
    run_agents,
    slice_candles_as_of,
    trend_confirms_signal,
)
from strategy.signal_filter import SignalFilter
from tracking.console import configure_console_encoding

H1_BUFFER = 400
H4_BUFFER = 250


@dataclass
class FilterBlockStats:
    candles_scanned: int = 0
    neutral_decisions: int = 0
    neutral_no_primary_agreement: int = 0
    neutral_trend_filter: int = 0
    trend_filter_blocked: int = 0
    h4_filter_blocked: int = 0
    entry_zone_blocked: int = 0
    confidence_blocked: int = 0
    primary_agreement_blocked: int = 0
    sl_validation_blocked: int = 0
    rr_validation_blocked: int = 0
    other_filter_blocked: int = 0
    signals_generated: int = 0
    examples: dict[str, list[str]] = field(default_factory=dict)

    def record_example(self, category: str, message: str, *, limit: int = 3) -> None:
        bucket = self.examples.setdefault(category, [])
        if len(bucket) < limit and message not in bucket:
            bucket.append(message)


def classify_filter_block(message: str) -> str:
    lowered = message.lower()
    if "trend" in lowered and (
        "bullish" in lowered
        or "bearish" in lowered
        or "trendfilter does not confirm" in lowered
        or "trend filter unavailable" in lowered
    ):
        return "trend_filter_blocked"
    if message.startswith("NO TRADE: H4"):
        return "h4_filter_blocked"
    if "entry zone" in lowered:
        return "entry_zone_blocked"
    if "below minimum" in lowered:
        return "confidence_blocked"
    if "primary agents agree" in lowered:
        return "primary_agreement_blocked"
    return "other_filter_blocked"


def scan_xauusd_filter_blocks() -> FilterBlockStats:
    symbol = "XAUUSD"
    display = resolve_symbol(symbol).display
    needed_m15 = WARMUP + LIGHT_SCAN_CANDLES + 1

    m15_candles = load_candles(XAUUSD_M15_30D_FILE, "15m")
    h1_candles = load_candles(XAUUSD_M15_30D_FILE, "1h")
    if len(m15_candles) < needed_m15:
        raise RuntimeError(
            f"{display}: need {needed_m15} M15 candles, got {len(m15_candles)}"
        )

    from data import MarketDataProvider

    provider = MarketDataProvider()
    h4_candles = provider.get_historical_market_data(display, "4h", H4_BUFFER)
    m15_window = slice_m15_window(m15_candles, scan_candles=LIGHT_SCAN_CANDLES)

    engine = LocalDataBacktestEngine(
        BacktestConfig(
            symbol=display,
            timeframe="15m",
            total_candles=len(m15_window),
            warmup_candles=WARMUP,
        ),
        m15_candles=m15_window,
        h1_candles=h1_candles,
        h4_candles=h4_candles,
        progress_every=100,
        progress_template="[XAUUSD diagnostics] {processed}/{total}...",
        progress_finish="[XAUUSD diagnostics] Scan complete.",
        label=f"filter diagnostics {LIGHT_SCAN_CANDLES} candles",
    )

    stats = FilterBlockStats()
    scan_start = engine.config.warmup_candles
    scan_end = len(m15_window) - 1
    stats.candles_scanned = scan_end - scan_start

    for index in range(scan_start, scan_end):
        if (index - scan_start + 1) % 100 == 0 or index == scan_end - 1:
            processed = index - scan_start + 1
            print(
                f"[XAUUSD diagnostics] {processed}/{stats.candles_scanned}...",
                flush=True,
            )

        history = m15_window[: index + 1]
        timestamp = candle_timestamp(m15_window, index)
        context = build_context(
            symbol=display,
            candles=history,
            timeframe="15m",
            timestamp=timestamp,
            h1_candles=slice_candles_as_of(h1_candles, timestamp),
            h4_candles=slice_candles_as_of(
                h4_candles,
                timestamp,
                limit=TREND_H4_CANDLE_MIN,
            ),
        )
        context["zone_catalog"] = engine._zone_catalog
        context["bar_index"] = index

        agent_results = run_agents(context)
        direction, confidence, _, _ = compute_final_decision(agent_results)

        if direction == Direction.NEUTRAL:
            stats.neutral_decisions += 1
            trend = agent_results.get("trend_filter")
            long_agreement = count_primary_agreement(agent_results, Direction.LONG)
            short_agreement = count_primary_agreement(agent_results, Direction.SHORT)
            has_primary = long_agreement >= 2 or short_agreement >= 2
            long_ok = long_agreement >= 2 and trend_confirms_signal(trend, Direction.LONG)
            short_ok = short_agreement >= 2 and trend_confirms_signal(trend, Direction.SHORT)

            if has_primary and not long_ok and not short_ok:
                stats.neutral_trend_filter += 1
                stats.record_example(
                    "neutral_trend_filter",
                    f"{timestamp.isoformat()} | "
                    f"LONG {long_agreement}/4 trend={trend.direction.value if trend else 'n/a'} | "
                    f"SHORT {short_agreement}/4",
                )
            else:
                stats.neutral_no_primary_agreement += 1
            continue

        filter_result = engine.signal_filter.evaluate(
            agent_results,
            direction,
            confidence,
            symbol=display,
            timestamp=context.get("timestamp"),
            context=context,
        )
        if not filter_result.approved:
            category = classify_filter_block(filter_result.message)
            setattr(stats, category, getattr(stats, category) + 1)
            stats.record_example(category, f"{timestamp.isoformat()} | {filter_result.message}")
            continue

        generation = engine.signal_generator.generate(
            context,
            filter_result.direction,
            filter_result.confidence,
            build_signal_reason(agent_results, filter_result.direction),
        )
        if generation.signal is None:
            reason = generation.rejection_reason or "unknown"
            if reason.startswith("RR rejected"):
                stats.rr_validation_blocked += 1
                stats.record_example("rr_validation_blocked", f"{timestamp.isoformat()} | {reason}")
            else:
                stats.sl_validation_blocked += 1
                stats.record_example("sl_validation_blocked", f"{timestamp.isoformat()} | {reason}")
            continue

        stats.signals_generated += 1

    print("[XAUUSD diagnostics] Scan complete.", flush=True)
    return stats


def print_report(stats: FilterBlockStats) -> None:
    period_note = f"{LIGHT_SCAN_CANDLES} M15 candles (~5 days)"
    print()
    print(f"=== XAUUSD filter diagnostics ({period_note}) ===")
    print(f"Candles scanned:              {stats.candles_scanned}")
    print(f"Signals generated:            {stats.signals_generated}")
    print()
    print("--- Before filter (decision layer) ---")
    print(f"Neutral decisions (total):    {stats.neutral_decisions}")
    print(f"  No 2/4 primary agents:      {stats.neutral_no_primary_agreement}")
    print(f"  TrendFilter blocks setup:   {stats.neutral_trend_filter}")
    print()
    print("--- Signal filter blocks ---")
    rows = [
        ("TrendFilter", stats.trend_filter_blocked),
        ("H4 alignment filter", stats.h4_filter_blocked),
        ("OB/FVG entry zone", stats.entry_zone_blocked),
        ("Confidence < 60%", stats.confidence_blocked),
        ("Primary agents (<2/4)", stats.primary_agreement_blocked),
        ("SL validation", stats.sl_validation_blocked),
        ("RR validation", stats.rr_validation_blocked),
        ("Other filters", stats.other_filter_blocked),
    ]
    total_blocks = sum(count for _, count in rows)
    print(f"{'Filter':<26} {'Blocks':>8} {'Share':>8}")
    print("-" * 44)
    for label, count in rows:
        share = (count / total_blocks * 100) if total_blocks else 0.0
        print(f"{label:<26} {count:>8} {share:>7.1f}%")
    print("-" * 44)
    print(f"{'Filter-stage total':<26} {total_blocks:>8}")
    print()

    combined = [
        ("Neutral: no 2/4 agents", stats.neutral_no_primary_agreement),
        ("Neutral: TrendFilter", stats.neutral_trend_filter),
        *rows,
    ]
    grand_total = sum(count for _, count in combined)
    print("--- Biggest blockers (all stages) ---")
    ranked = sorted(combined, key=lambda item: item[1], reverse=True)
    for label, count in ranked:
        if count <= 0:
            continue
        share = count / grand_total * 100 if grand_total else 0.0
        print(f"  {label:<28} {count:>5} ({share:5.1f}%)")

    print()
    print("--- Sample messages ---")
    for category, messages in stats.examples.items():
        if not messages:
            continue
        print(f"{category}:")
        for message in messages:
            print(f"  - {message}")


def main() -> None:
    configure_console_encoding()
    stats = scan_xauusd_filter_blocks()
    print_report(stats)


if __name__ == "__main__":
    main()
