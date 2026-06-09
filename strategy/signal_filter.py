from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from agents.base import AgentResult, Direction
from agents.rsi_agent import (
    RSI_GATE_LONG_MIN,
    RSI_GATE_SHORT_MAX,
    extract_rsi_value,
)
from agents.session_agent import is_london_or_new_york_session
from agents.smc_agent import smc_result_has_structure_conflict
from agents.zone_helpers import price_in_active_entry_zone
from config.symbols import resolve_symbol
from news.models import NewsAction
from strategy.decision_config import DecisionConfig, LEGACY_DECISION_CONFIG
from strategy.runner import (
    MIN_PRIMARY_AGREEMENT,
    count_primary_agreement,
    format_primary_agreement,
    primary_vote_count,
    trend_confirms_signal,
)

MIN_CONFIDENCE = 0.60
MIN_CONFIDENCE_PCT = 60
XAUUSD_MIN_CONFIDENCE = 0.50
XAUUSD_MIN_CONFIDENCE_PCT = 50
MIN_SESSION_CONFIDENCE = 0.30

SYMBOL_MIN_CONFIDENCE: dict[str, float] = {
    "XAUUSD": XAUUSD_MIN_CONFIDENCE,
}


@dataclass(frozen=True)
class FilterProfile:
    """Preset Chief Analyst filter configuration for backtests and runtime."""

    label: str
    description: str
    require_h4_h1_alignment: bool
    require_entry_zone: bool
    min_confidence: float
    use_symbol_confidence_overrides: bool = True
    soft_entry_zone_atr: float | None = None
    block_smc_structure_conflict: bool = False
    use_zone_cluster: bool = False
    use_rsi_gate: bool = False
    h4_soft_mode: bool = False
    disabled_symbols: frozenset[str] = frozenset()


FILTER_PROFILE_A = FilterProfile(
    label="A",
    description="Legacy: 4 primary votes, RSI weight, H4 hard block",
    require_h4_h1_alignment=True,
    require_entry_zone=True,
    min_confidence=MIN_CONFIDENCE,
    use_zone_cluster=False,
    use_rsi_gate=False,
    h4_soft_mode=False,
)

FILTER_PROFILE_B = FilterProfile(
    label="B",
    description="Zone cluster + RSI gate + H4 hard block",
    require_h4_h1_alignment=True,
    require_entry_zone=True,
    min_confidence=MIN_CONFIDENCE,
    use_zone_cluster=True,
    use_rsi_gate=True,
    h4_soft_mode=False,
)

FILTER_PROFILE_C = FilterProfile(
    label="C",
    description="Zone cluster + RSI gate + H4 soft warning",
    require_h4_h1_alignment=True,
    require_entry_zone=True,
    min_confidence=MIN_CONFIDENCE,
    use_zone_cluster=True,
    use_rsi_gate=True,
    h4_soft_mode=True,
)

FILTER_PROFILE_D = FilterProfile(
    label="D",
    description="B + SMC conflict filter, BTCUSDT disabled",
    require_h4_h1_alignment=True,
    require_entry_zone=True,
    min_confidence=MIN_CONFIDENCE,
    use_zone_cluster=True,
    use_rsi_gate=True,
    h4_soft_mode=False,
    block_smc_structure_conflict=True,
    disabled_symbols=frozenset({"BTCUSDT"}),
)


def profile_symbols(profile: FilterProfile, symbols: tuple[str, ...]) -> tuple[str, ...]:
    """Return symbols enabled for a profile (respecting ``disabled_symbols``)."""
    disabled = {symbol.upper() for symbol in profile.disabled_symbols}
    return tuple(
        symbol
        for symbol in symbols
        if resolve_symbol(symbol).display not in disabled
    )


def resolve_min_confidence(
    symbol: str | None,
    *,
    default: float = MIN_CONFIDENCE,
) -> float:
    if not symbol:
        return default
    try:
        display = resolve_symbol(symbol).display
    except ValueError:
        display = symbol.upper()
    return SYMBOL_MIN_CONFIDENCE.get(display, default)


def resolve_min_confidence_pct(
    symbol: str | None,
    *,
    default: int = MIN_CONFIDENCE_PCT,
) -> int:
    return int(round(resolve_min_confidence(symbol) * 100)) or default


@dataclass(frozen=True)
class FilterResult:
    approved: bool
    direction: Direction
    confidence: float
    message: str
    news_warning: str | None = None
    off_hours_warning: str | None = None
    h4_mismatch_warning: str | None = None


class SignalFilter:
    """Chief Analyst filter layer — hard blocks before signals are published."""

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        min_session_confidence: float = MIN_SESSION_CONFIDENCE,
        london_ny_session_symbols: frozenset[str] | set[str] | None = None,
        session_confidence_symbols: frozenset[str] | set[str] | None = None,
        news_gate=None,
        *,
        require_h4_h1_alignment: bool = True,
        require_entry_zone: bool = True,
        use_symbol_confidence_overrides: bool = True,
        soft_entry_zone_atr: float | None = None,
        block_smc_structure_conflict: bool = False,
        use_zone_cluster: bool = False,
        use_rsi_gate: bool = False,
        h4_soft_mode: bool = False,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_session_confidence = min_session_confidence
        self.london_ny_session_symbols = frozenset(london_ny_session_symbols or ())
        self.session_confidence_symbols = frozenset(session_confidence_symbols or ())
        self.news_gate = news_gate
        self.require_h4_h1_alignment = require_h4_h1_alignment
        self.require_entry_zone = require_entry_zone
        self.use_symbol_confidence_overrides = use_symbol_confidence_overrides
        self.soft_entry_zone_atr = soft_entry_zone_atr
        self.block_smc_structure_conflict = block_smc_structure_conflict
        self.use_zone_cluster = use_zone_cluster
        self.use_rsi_gate = use_rsi_gate
        self.h4_soft_mode = h4_soft_mode

    @property
    def decision_config(self) -> DecisionConfig:
        return DecisionConfig(
            use_zone_cluster=self.use_zone_cluster,
            use_rsi_gate=self.use_rsi_gate,
        )

    @classmethod
    def from_profile(cls, profile: FilterProfile, **kwargs) -> SignalFilter:
        return cls(
            min_confidence=profile.min_confidence,
            require_h4_h1_alignment=profile.require_h4_h1_alignment,
            require_entry_zone=profile.require_entry_zone,
            use_symbol_confidence_overrides=profile.use_symbol_confidence_overrides,
            soft_entry_zone_atr=profile.soft_entry_zone_atr,
            block_smc_structure_conflict=profile.block_smc_structure_conflict,
            use_zone_cluster=profile.use_zone_cluster,
            use_rsi_gate=profile.use_rsi_gate,
            h4_soft_mode=profile.h4_soft_mode,
            **kwargs,
        )

    def evaluate(
        self,
        results: dict[str, AgentResult],
        final_direction: Direction,
        final_confidence: float,
        *,
        symbol: str | None = None,
        timestamp: datetime | None = None,
        context: dict[str, Any] | None = None,
    ) -> FilterResult:
        session = results["session"]
        trend = results.get("trend_filter")
        symbol_key = symbol.upper() if symbol else None
        config = self.decision_config

        trend_block = self._evaluate_trend_hard_block(trend, final_direction)
        if trend_block is not None:
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=trend_block,
            )

        adjusted_confidence = final_confidence
        off_hours_warning = None
        h4_mismatch_warning = None

        if (
            symbol_key
            and symbol_key in self.london_ny_session_symbols
            and not is_london_or_new_york_session(timestamp)
        ):
            off_hours_warning = "⚠️ Off-hours"

        if final_direction == Direction.NEUTRAL:
            return FilterResult(
                approved=False,
                direction=Direction.NEUTRAL,
                confidence=final_confidence,
                message="NO TRADE: neutral decision",
            )

        agreement = count_primary_agreement(results, final_direction, config=config)
        vote_slots = primary_vote_count(config)
        if agreement < MIN_PRIMARY_AGREEMENT:
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=(
                    "NO TRADE: only "
                    f"{agreement}/{vote_slots} primary agents agree "
                    f"({format_primary_agreement(results, final_direction, config=config)})"
                ),
            )

        if not trend_confirms_signal(trend, final_direction):
            trend_value = trend.direction.value if trend is not None else "unavailable"
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=(
                    "NO TRADE: TrendFilter does not confirm "
                    f"{final_direction.value.upper()} "
                    f"(trend={trend_value})"
                ),
            )

        if self.require_h4_h1_alignment:
            h4_block = self._evaluate_h4_h1_alignment(
                results.get("h4_trend_filter"),
                trend,
                final_direction,
            )
            if h4_block is not None:
                if self.h4_soft_mode:
                    h4_mismatch_warning = "⚠️ H4 mismatch"
                else:
                    return FilterResult(
                        approved=False,
                        direction=final_direction,
                        confidence=final_confidence,
                        message=h4_block,
                    )

        if self.use_rsi_gate:
            rsi_block = self._evaluate_rsi_gate(results, final_direction)
            if rsi_block is not None:
                return FilterResult(
                    approved=False,
                    direction=final_direction,
                    confidence=final_confidence,
                    message=rsi_block,
                )

        if self.block_smc_structure_conflict:
            smc = results.get("smc")
            if smc is not None and smc_result_has_structure_conflict(smc):
                return FilterResult(
                    approved=False,
                    direction=final_direction,
                    confidence=final_confidence,
                    message=(
                        "NO TRADE: SMC structure conflict "
                        "(HH/HL vs bearish ChoCH or LH/LL vs bullish ChoCH)"
                    ),
                )

        if self.require_entry_zone:
            in_zone = context is not None and price_in_active_entry_zone(
                context,
                final_direction,
                atr_tolerance_multiplier=self.soft_entry_zone_atr,
            )
            if not in_zone:
                zone_label = (
                    "active OB/FVG entry zone (soft 0.5 ATR)"
                    if self.soft_entry_zone_atr
                    else "active OB/FVG entry zone"
                )
                return FilterResult(
                    approved=False,
                    direction=final_direction,
                    confidence=final_confidence,
                    message=f"NO TRADE: price not in {zone_label}",
                )

        if self.use_symbol_confidence_overrides:
            min_confidence = resolve_min_confidence(
                symbol_key,
                default=self.min_confidence,
            )
            min_confidence_pct = resolve_min_confidence_pct(symbol_key)
        else:
            min_confidence = self.min_confidence
            min_confidence_pct = int(round(self.min_confidence * 100))
        if adjusted_confidence < min_confidence:
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=adjusted_confidence,
                message=(
                    f"NO TRADE: confidence {adjusted_confidence:.2f} "
                    f"below minimum {min_confidence:.2f} ({min_confidence_pct}%)"
                ),
            )

        if (
            symbol_key
            and symbol_key in self.session_confidence_symbols
            and session.confidence < self.min_session_confidence
        ):
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=(
                    f"NO TRADE: session confidence {session.confidence:.2f} "
                    f"below minimum {self.min_session_confidence:.2f}"
                ),
            )

        news_warning = None
        if symbol_key and self.news_gate is not None:
            news_result = self.news_gate.evaluate(symbol_key, timestamp)
            if news_result.action == NewsAction.BLOCK:
                return FilterResult(
                    approved=False,
                    direction=final_direction,
                    confidence=final_confidence,
                    message=f"NO TRADE: {news_result.message}",
                )
            if news_result.action == NewsAction.WARN and news_result.message:
                news_warning = news_result.message

        return FilterResult(
            approved=True,
            direction=final_direction,
            confidence=adjusted_confidence,
            message="Signal approved",
            news_warning=news_warning,
            off_hours_warning=off_hours_warning,
            h4_mismatch_warning=h4_mismatch_warning,
        )

    @staticmethod
    def _evaluate_rsi_gate(
        results: dict[str, AgentResult],
        final_direction: Direction,
    ) -> str | None:
        rsi = results.get("rsi")
        if rsi is None:
            return None

        rsi_value = extract_rsi_value(rsi)
        if rsi_value is None:
            return None

        if final_direction == Direction.LONG and rsi_value < RSI_GATE_LONG_MIN:
            return (
                f"NO TRADE: RSI {rsi_value:.2f} below {RSI_GATE_LONG_MIN:.0f} "
                "— LONG blocked"
            )

        if final_direction == Direction.SHORT and rsi_value > RSI_GATE_SHORT_MAX:
            return (
                f"NO TRADE: RSI {rsi_value:.2f} above {RSI_GATE_SHORT_MAX:.0f} "
                "— SHORT blocked"
            )

        return None

    @staticmethod
    def _evaluate_trend_hard_block(
        trend: AgentResult | None,
        final_direction: Direction,
    ) -> str | None:
        """Hard Chief Analyst trend filter — not a soft recommendation."""
        if trend is None:
            return "NO TRADE: H1 trend filter unavailable"

        if trend.direction == Direction.LONG and final_direction == Direction.SHORT:
            return "NO TRADE: H1 trend BULLISH — SHORT signals blocked"

        if trend.direction == Direction.SHORT and final_direction == Direction.LONG:
            return "NO TRADE: H1 trend BEARISH — LONG signals blocked"

        return None

    @staticmethod
    def _evaluate_h4_h1_alignment(
        h4_trend: AgentResult | None,
        h1_trend: AgentResult | None,
        final_direction: Direction,
    ) -> str | None:
        if h4_trend is None:
            return "NO TRADE: H4 trend filter unavailable"

        if h1_trend is None:
            return "NO TRADE: H1 trend filter unavailable"

        if h4_trend.direction == Direction.NEUTRAL or h1_trend.direction == Direction.NEUTRAL:
            return (
                "NO TRADE: H4/H1 trend not aligned "
                f"(H4={h4_trend.direction.value}, H1={h1_trend.direction.value})"
            )

        if h4_trend.direction != h1_trend.direction:
            return (
                "NO TRADE: H4/H1 trend mismatch "
                f"(H4={h4_trend.direction.value}, H1={h1_trend.direction.value})"
            )

        if h4_trend.direction != final_direction:
            return (
                "NO TRADE: H4/H1 trend does not match signal "
                f"(trend={h4_trend.direction.value}, signal={final_direction.value})"
            )

        return None
