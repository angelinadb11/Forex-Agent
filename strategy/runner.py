from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents import LiquidityAgent, RSIAgent, SessionAgent, SMCAgent
from agents.base import Agent, AgentResult, Direction

HIGH_CONFIDENCE_THRESHOLD = 0.80
CAPPED_MAX_CONFIDENCE = 0.70


def build_agents() -> list[Agent]:
    return [
        SMCAgent(),
        LiquidityAgent(),
        RSIAgent(),
        SessionAgent(),
    ]


def build_context(
    symbol: str,
    candles: list[dict[str, float]],
    timeframe: str,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "timestamp": timestamp or datetime.now(timezone.utc),
        "candles": candles,
        "metadata": {
            "timeframe": timeframe,
            "candle_count": len(candles),
        },
    }


def run_agents(context: dict[str, Any]) -> dict[str, AgentResult]:
    results: dict[str, AgentResult] = {}
    for agent in build_agents():
        results[agent.name] = agent.analyze(context)
    return results


def core_agents_agree(results: dict[str, AgentResult], direction: Direction) -> bool:
    """Return True when SMC, Liquidity, and RSI share the same non-neutral direction."""
    smc = results["smc"]
    liquidity = results["liquidity"]
    rsi = results["rsi"]

    core_agents = (smc, liquidity, rsi)
    if any(agent.direction == Direction.NEUTRAL for agent in core_agents):
        return False
    return all(agent.direction == direction for agent in core_agents)


def format_agents_agreement(results: dict[str, AgentResult], direction: Direction) -> str:
    return "Yes" if core_agents_agree(results, direction) else "No"


def _smc_liquidity_rsi_agree(results: dict[str, AgentResult], direction: Direction) -> bool:
    return core_agents_agree(results, direction)


def apply_confidence_cap(
    results: dict[str, AgentResult],
    direction: Direction,
    raw_confidence: float,
) -> float:
    """Allow confidence above 80% only when SMC, Liquidity, and RSI all agree."""
    if raw_confidence <= HIGH_CONFIDENCE_THRESHOLD:
        return round(raw_confidence, 2)
    if _smc_liquidity_rsi_agree(results, direction):
        return round(min(1.0, raw_confidence), 2)
    return CAPPED_MAX_CONFIDENCE


def compute_final_decision(
    results: dict[str, AgentResult],
) -> tuple[Direction, float, float, float]:
    long_score = sum(
        result.confidence for result in results.values() if result.direction == Direction.LONG
    )
    short_score = sum(
        result.confidence for result in results.values() if result.direction == Direction.SHORT
    )

    if long_score > short_score:
        confidence = apply_confidence_cap(results, Direction.LONG, long_score)
        return Direction.LONG, confidence, long_score, short_score
    if short_score > long_score:
        confidence = apply_confidence_cap(results, Direction.SHORT, short_score)
        return Direction.SHORT, confidence, long_score, short_score

    return Direction.NEUTRAL, long_score, long_score, short_score


def build_signal_reason(results: dict[str, AgentResult], direction: Direction) -> str:
    matching = [
        result.reason
        for result in results.values()
        if result.direction == direction
    ]
    if matching:
        return " | ".join(matching)
    return f"{direction.value.upper()} signal from aggregated agent scores"
