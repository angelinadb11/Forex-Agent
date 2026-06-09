from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Callable

from agents.base import AgentResult, Direction
from strategy.runner import run_agents

ContextFetcher = Callable[[str, str], dict[str, Any]]

SEVERE_CONFIDENCE_DROP = 0.25
OPPOSITE_SWEEP_MIN_CONFIDENCE = 0.35
LEVEL2_STANDARD_CONFIRMATION_CYCLES = 2


class WarningLevel(IntEnum):
    NONE = 0
    LEVEL_1 = 1
    LEVEL_2 = 2


@dataclass(frozen=True)
class Level2Evaluation:
    instant: bool = False
    standard: bool = False
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class TradeUpdateAssessment:
    """Informational assessment of whether an active trade is weakening."""

    level: WarningLevel
    reasons: tuple[str, ...]
    level2_instant: bool = False
    level2_standard: bool = False
    level2_reasons: tuple[str, ...] = ()

    @property
    def is_weakening(self) -> bool:
        return (
            self.level != WarningLevel.NONE
            or self.level2_instant
            or self.level2_standard
        )


def _opposite(direction: Direction) -> Direction:
    if direction == Direction.LONG:
        return Direction.SHORT
    if direction == Direction.SHORT:
        return Direction.LONG
    return Direction.NEUTRAL


def _direction_score(results: dict[str, AgentResult], direction: Direction) -> float:
    return sum(
        result.confidence
        for result in results.values()
        if result.direction == direction
    )


def _has_opposite_liquidity_sweep(result: AgentResult, trade_direction: Direction) -> bool:
    if result.confidence < OPPOSITE_SWEEP_MIN_CONFIDENCE:
        return False
    reason = result.reason.lower()
    if trade_direction == Direction.LONG:
        return "bearish liquidity sweep" in reason or "bsl taken" in reason
    return "bullish liquidity sweep" in reason or "ssl taken" in reason


def _has_severe_confidence_deterioration(
    trade_direction: Direction,
    entry_confidence: float,
    current_results: dict[str, AgentResult],
) -> bool:
    opposite = _opposite(trade_direction)
    trade_score = _direction_score(current_results, trade_direction)
    opposite_score = _direction_score(current_results, opposite)
    return (
        entry_confidence - trade_score >= SEVERE_CONFIDENCE_DROP
        and opposite_score >= trade_score
    )


def _structure_flip_message(trade_direction: Direction) -> str:
    if trade_direction == Direction.LONG:
        return "Market structure has flipped bearish."
    return "Market structure has flipped bullish."


def _level1_message(agent_name: str, trade_direction: Direction) -> str:
    if agent_name == "smc":
        return (
            "Bullish structure is weakening."
            if trade_direction == Direction.LONG
            else "Bearish structure is weakening."
        )
    if agent_name == "liquidity":
        return (
            "Liquidity support is fading."
            if trade_direction == Direction.LONG
            else "Liquidity pressure is fading."
        )
    return (
        "Bullish momentum is weakening."
        if trade_direction == Direction.LONG
        else "Bearish momentum is weakening."
    )


def _has_third_level2_confirmation(
    trade_direction: Direction,
    entry_confidence: float,
    current_results: dict[str, AgentResult],
) -> bool:
    opposite = _opposite(trade_direction)
    rsi = current_results["rsi"]
    liquidity = current_results["liquidity"]

    if rsi.direction == opposite:
        return True
    if _has_opposite_liquidity_sweep(liquidity, trade_direction):
        return True
    return _has_severe_confidence_deterioration(
        trade_direction,
        entry_confidence,
        current_results,
    )


def _build_level2_reasons(trade_direction: Direction) -> tuple[str, ...]:
    return (
        _structure_flip_message(trade_direction),
        "The original setup is no longer valid.",
    )


def evaluate_level2(
    trade_direction: Direction,
    entry_confidence: float,
    current_results: dict[str, AgentResult],
) -> Level2Evaluation:
    """Hybrid Level 2 criteria: instant, standard (2-cycle), or none."""
    opposite = _opposite(trade_direction)
    smc = current_results["smc"]
    liquidity = current_results["liquidity"]

    dual_core_opposite = smc.direction == opposite and liquidity.direction == opposite
    if not dual_core_opposite:
        return Level2Evaluation()

    reasons = _build_level2_reasons(trade_direction)
    if _has_third_level2_confirmation(trade_direction, entry_confidence, current_results):
        return Level2Evaluation(instant=True, reasons=reasons)

    return Level2Evaluation(standard=True, reasons=reasons)


def _assess_level_1(
    trade_direction: Direction,
    entry_results: dict[str, AgentResult] | None,
    current_results: dict[str, AgentResult],
) -> TradeUpdateAssessment:
    if not entry_results:
        return TradeUpdateAssessment(level=WarningLevel.NONE, reasons=())

    opposite = _opposite(trade_direction)
    for agent_name in ("smc", "liquidity", "rsi"):
        if current_results[agent_name].direction == opposite:
            return TradeUpdateAssessment(level=WarningLevel.NONE, reasons=())

    softened_agents: list[str] = []
    for agent_name in ("smc", "liquidity", "rsi"):
        entry = entry_results[agent_name]
        current = current_results[agent_name]
        if entry.direction == trade_direction and current.direction == Direction.NEUTRAL:
            softened_agents.append(agent_name)

    if not softened_agents:
        return TradeUpdateAssessment(level=WarningLevel.NONE, reasons=())

    for agent_name in ("smc", "liquidity", "rsi"):
        if agent_name in softened_agents:
            return TradeUpdateAssessment(
                level=WarningLevel.LEVEL_1,
                reasons=(_level1_message(agent_name, trade_direction),),
            )

    return TradeUpdateAssessment(level=WarningLevel.NONE, reasons=())


def assess_trade_update(
    trade_direction: Direction,
    entry_confidence: float,
    entry_results: dict[str, AgentResult] | None,
    current_results: dict[str, AgentResult],
) -> TradeUpdateAssessment:
    """Compare current agent context against the original trade setup."""
    if trade_direction == Direction.NEUTRAL:
        return TradeUpdateAssessment(level=WarningLevel.NONE, reasons=())

    level2 = evaluate_level2(trade_direction, entry_confidence, current_results)
    if level2.instant:
        return TradeUpdateAssessment(
            level=WarningLevel.NONE,
            reasons=(),
            level2_instant=True,
            level2_reasons=level2.reasons,
        )
    if level2.standard:
        return TradeUpdateAssessment(
            level=WarningLevel.NONE,
            reasons=(),
            level2_standard=True,
            level2_reasons=level2.reasons,
        )

    return _assess_level_1(trade_direction, entry_results, current_results)


def assess_trade_weakening(
    trade_direction: Direction,
    entry_results: dict[str, AgentResult] | None,
    current_results: dict[str, AgentResult],
    *,
    entry_confidence: float = 0.0,
) -> TradeUpdateAssessment:
    """Backward-compatible alias for trade update assessment."""
    return assess_trade_update(
        trade_direction,
        entry_confidence,
        entry_results,
        current_results,
    )


def assess_trend_opposes_trade(
    trade_direction: Direction,
    current_results: dict[str, AgentResult],
) -> bool:
    """Return True when H1 trend filter opposes the open trade direction."""
    trend = current_results.get("trend_filter")
    if trend is None:
        return False

    if trend.direction == Direction.NEUTRAL:
        return True

    if trade_direction == Direction.LONG and trend.direction == Direction.SHORT:
        return True

    if trade_direction == Direction.SHORT and trend.direction == Direction.LONG:
        return True

    return False


class TradeUpdateChecker:
    """Re-analyzes open trades and emits informational warnings only."""

    def __init__(self, context_fetcher: ContextFetcher) -> None:
        self.context_fetcher = context_fetcher

    def analyze(self, trade) -> TradeUpdateAssessment | None:
        if trade.closed or not trade.timeframe:
            return None

        context = self.context_fetcher(trade.symbol, trade.timeframe)
        current_results = run_agents(context)
        return assess_trade_update(
            trade.direction,
            trade.confidence,
            trade.entry_agent_results,
            current_results,
        )
