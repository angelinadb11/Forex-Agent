from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from agents.base import AgentResult, Direction
from agents.zone_helpers import ZoneCatalog
from config.symbols import resolve_symbol
from data import MarketDataProvider
from signal_generator import SignalGenerator, TradeSignal
from strategy.runner import (
    build_context,
    build_signal_reason,
    compute_final_decision,
    run_agents,
    slice_candles_as_of,
    trend_confirms_signal,
)

SCALP_TIMEFRAME = "5m"
M15_CONFIRM_TIMEFRAME = "15m"
SCALP_SYMBOLS = frozenset({"XAUUSD"})
SCALP_M5_CANDLE_LIMIT = 500
SCALP_M15_CANDLE_LIMIT = 200
SCALP_MIN_CONFIDENCE = 0.60
SCALP_MIN_CONFIDENCE_PCT = 60
SCALP_MIN_INTERVAL_SECONDS = 3600
SCALP_MAX_SIGNALS_PER_DAY = 4


def is_scalp_enabled(symbol: str) -> bool:
    return resolve_symbol(symbol).display in SCALP_SYMBOLS


@dataclass(frozen=True)
class ScalpAnalysisResult:
    approved: bool
    direction: Direction
    confidence: float
    message: str
    m5_results: dict[str, AgentResult] | None = None
    m15_results: dict[str, AgentResult] | None = None


@dataclass
class ScalpPublishGate:
    """Limits scalp signal frequency for Telegram groups."""

    min_interval_seconds: int = SCALP_MIN_INTERVAL_SECONDS
    max_signals_per_day: int = SCALP_MAX_SIGNALS_PER_DAY
    _last_signal_at: dict[str, datetime] = field(default_factory=dict)
    _daily_counts: dict[tuple[str, str], int] = field(default_factory=dict)

    def can_publish(self, symbol: str, timestamp: datetime) -> tuple[bool, str | None]:
        display = resolve_symbol(symbol).display
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        day_key = timestamp.astimezone(timezone.utc).date().isoformat()
        daily_count = self._daily_counts.get((display, day_key), 0)
        if daily_count >= self.max_signals_per_day:
            return False, (
                f"NO SCALP: daily limit reached ({self.max_signals_per_day}/day for {display})"
            )

        last_signal_at = self._last_signal_at.get(display)
        if last_signal_at is not None:
            elapsed = (timestamp - last_signal_at).total_seconds()
            if elapsed < self.min_interval_seconds:
                remaining_minutes = int(
                    (self.min_interval_seconds - elapsed + 59) // 60
                )
                return False, (
                    "NO SCALP: minimum interval active "
                    f"({remaining_minutes} min remaining for {display})"
                )

        return True, None

    def record(self, symbol: str, timestamp: datetime) -> None:
        display = resolve_symbol(symbol).display
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        normalized = timestamp.astimezone(timezone.utc)
        day_key = normalized.date().isoformat()
        self._last_signal_at[display] = normalized
        key = (display, day_key)
        self._daily_counts[key] = self._daily_counts.get(key, 0) + 1


def passes_scalp_confidence(
    confidence: float,
    *,
    min_confidence: float = SCALP_MIN_CONFIDENCE,
) -> bool:
    return confidence >= min_confidence


def resolve_scalp_m5_confidence(
    m5_results: dict[str, AgentResult],
    direction: Direction,
) -> float:
    _, _, long_score, short_score = compute_final_decision(m5_results)
    if direction == Direction.LONG:
        return long_score
    if direction == Direction.SHORT:
        return short_score
    return 0.0


def evaluate_scalp_direction_alignment(
    m5_results: dict[str, AgentResult],
    m15_results: dict[str, AgentResult],
) -> tuple[Direction, float] | None:
    """Return direction when M5, M15, and H1 align (without confidence gate)."""
    m5_direction, _, _, _ = compute_final_decision(m5_results)
    if m5_direction == Direction.NEUTRAL:
        return None

    m15_direction, _, _, _ = compute_final_decision(m15_results)
    if m15_direction != m5_direction:
        return None

    trend = m5_results.get("trend_filter")
    if not trend_confirms_signal(trend, m5_direction):
        return None

    m5_confidence = resolve_scalp_m5_confidence(m5_results, m5_direction)
    return m5_direction, m5_confidence


def evaluate_scalp_alignment(
    m5_results: dict[str, AgentResult],
    m15_results: dict[str, AgentResult],
    *,
    min_confidence: float = SCALP_MIN_CONFIDENCE,
) -> tuple[Direction, float] | None:
    """Return direction and confidence when all scalp setup rules pass."""
    alignment = evaluate_scalp_direction_alignment(m5_results, m15_results)
    if alignment is None:
        return None

    direction, confidence = alignment
    if not passes_scalp_confidence(confidence, min_confidence=min_confidence):
        return None

    return direction, confidence


def analyze_scalp_symbol(
    symbol: str,
    *,
    provider: MarketDataProvider,
    signal_generator: SignalGenerator | None = None,
    m5_candle_limit: int = SCALP_M5_CANDLE_LIMIT,
    m15_candle_limit: int = SCALP_M15_CANDLE_LIMIT,
    publish_gate: ScalpPublishGate | None = None,
) -> tuple[TradeSignal | None, dict[str, Any] | None, ScalpAnalysisResult]:
    """Analyze a scalp setup on M5 with M15 and H1 confirmation."""
    symbol_def = resolve_symbol(symbol)
    display_symbol = symbol_def.display

    if not is_scalp_enabled(display_symbol):
        return None, None, ScalpAnalysisResult(
            approved=False,
            direction=Direction.NEUTRAL,
            confidence=0.0,
            message=f"NO SCALP: scalp mode disabled for {display_symbol}",
        )

    generator = signal_generator or SignalGenerator()
    gate = publish_gate or ScalpPublishGate()

    m5_context = provider.to_context(display_symbol, SCALP_TIMEFRAME, limit=m5_candle_limit)
    m15_context = provider.to_context(
        display_symbol,
        M15_CONFIRM_TIMEFRAME,
        limit=m15_candle_limit,
        include_h4_trend=False,
    )

    timestamp = m5_context.get("timestamp")
    if isinstance(timestamp, datetime):
        m15_context["h1_candles"] = m5_context.get("h1_candles", m15_context.get("h1_candles"))
        m15_candles = m15_context.get("candles", [])
        if m15_candles:
            m15_context["candles"] = slice_candles_as_of(m15_candles, timestamp)

    m5_candles = m5_context["candles"]
    m5_context["zone_catalog"] = ZoneCatalog.from_candles(m5_candles, display_symbol)
    m5_context["bar_index"] = len(m5_candles) - 1

    m15_candles = m15_context.get("candles", [])
    if m15_candles:
        m15_context["zone_catalog"] = ZoneCatalog.from_candles(m15_candles, display_symbol)
        m15_context["bar_index"] = len(m15_candles) - 1

    m5_results = run_agents(m5_context)
    m15_results = run_agents(m15_context)

    alignment = evaluate_scalp_alignment(m5_results, m15_results)
    if alignment is None:
        m5_direction, _, _, _ = compute_final_decision(m5_results)
        m15_direction, _, _, _ = compute_final_decision(m15_results)
        if m5_direction == Direction.NEUTRAL:
            message = "NO SCALP: M5 agents have no directional setup"
        elif m15_direction != m5_direction:
            message = (
                "NO SCALP: M15 does not confirm M5 direction "
                f"(M5={m5_direction.value.upper()}, M15={m15_direction.value.upper()})"
            )
        elif not trend_confirms_signal(m5_results.get("trend_filter"), m5_direction):
            trend = m5_results.get("trend_filter")
            trend_value = trend.direction.value if trend is not None else "unavailable"
            message = (
                "NO SCALP: H1 trend does not match "
                f"(signal={m5_direction.value.upper()}, H1={trend_value})"
            )
        else:
            m5_confidence = resolve_scalp_m5_confidence(m5_results, m5_direction)
            message = (
                "NO SCALP: M5 confidence "
                f"{m5_confidence:.2f} below minimum "
                f"{SCALP_MIN_CONFIDENCE:.2f} ({SCALP_MIN_CONFIDENCE_PCT}%)"
            )
        return None, None, ScalpAnalysisResult(
            approved=False,
            direction=m5_direction if m5_direction != Direction.NEUTRAL else Direction.NEUTRAL,
            confidence=0.0,
            message=message,
            m5_results=m5_results,
            m15_results=m15_results,
        )

    direction, confidence = alignment
    timestamp = m5_context.get("timestamp")
    if isinstance(timestamp, datetime):
        allowed, block_reason = gate.can_publish(display_symbol, timestamp)
        if not allowed:
            return None, m5_context, ScalpAnalysisResult(
                approved=False,
                direction=direction,
                confidence=confidence,
                message=block_reason or "NO SCALP: publish gate blocked",
                m5_results=m5_results,
                m15_results=m15_results,
            )

    reason = build_signal_reason(m5_results, direction)
    generation = generator.generate_scalp(
        m5_context,
        direction,
        confidence,
        reason,
    )
    if generation.signal is None:
        return None, m5_context, ScalpAnalysisResult(
            approved=False,
            direction=direction,
            confidence=confidence,
            message=generation.rejection_reason or "NO SCALP: signal generation failed",
            m5_results=m5_results,
            m15_results=m15_results,
        )

    if isinstance(timestamp, datetime):
        gate.record(display_symbol, timestamp)

    return generation.signal, m5_context, ScalpAnalysisResult(
        approved=True,
        direction=direction,
        confidence=confidence,
        message="Scalp signal approved",
        m5_results=m5_results,
        m15_results=m15_results,
    )
