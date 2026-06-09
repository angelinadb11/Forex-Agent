from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agents import (
    FVGAgent,
    LiquidityAgent,
    OrderBlockAgent,
    RSIAgent,
    SessionAgent,
    SMCAgent,
)
from agents.base import Agent, AgentResult, Direction
from agents.trend_filter_agent import (
    TREND_CANDLE_MIN,
    TREND_H4_CANDLE_MIN,
    TREND_TIMEFRAME,
    TrendFilterAgent,
    analyze_h4_trend,
)
from strategy.decision_config import DecisionConfig, LEGACY_DECISION_CONFIG

HIGH_CONFIDENCE_THRESHOLD = 0.80
CAPPED_MAX_CONFIDENCE = 0.70
AGENT_WEIGHTS: dict[str, float] = {
    "smc": 0.25,
    "liquidity": 0.25,
    "fvg": 0.20,
    "order_block": 0.20,
    "rsi": 0.10,
}
ZONE_CLUSTER_WEIGHT = 0.40
PRIMARY_AGENT_NAMES = ("smc", "liquidity", "fvg", "order_block")
CLUSTER_PRIMARY_NAMES = ("smc", "liquidity", "zone")
MIN_PRIMARY_AGREEMENT = 2
DECISION_AGENT_NAMES = ("smc", "liquidity", "fvg", "order_block", "rsi")


def build_agents() -> list[Agent]:
    return [
        SMCAgent(),
        FVGAgent(),
        OrderBlockAgent(),
        RSIAgent(),
        SessionAgent(),
    ]


def build_context(
    symbol: str,
    candles: list[dict[str, float]],
    timeframe: str,
    timestamp: datetime | None = None,
    h1_candles: list[dict[str, float]] | None = None,
    h4_candles: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    context: dict[str, Any] = {
        "symbol": symbol,
        "timestamp": timestamp or datetime.now(timezone.utc),
        "candles": candles,
        "metadata": {
            "timeframe": timeframe,
            "candle_count": len(candles),
        },
    }
    if h1_candles is not None:
        context["h1_candles"] = h1_candles
    if h4_candles is not None:
        context["h4_candles"] = h4_candles
    return context


def slice_candles_as_of(
    candles: list[dict[str, float]],
    as_of: datetime,
    *,
    limit: int = TREND_CANDLE_MIN,
) -> list[dict[str, float]]:
    """Return candles whose open time is at or before ``as_of``."""
    as_of_ms = as_of.timestamp() * 1000
    filtered = [
        candle
        for candle in candles
        if candle.get("open_time", 0) <= as_of_ms
    ]
    if len(filtered) > limit:
        return filtered[-limit:]
    return filtered


def run_agents(context: dict[str, Any]) -> dict[str, AgentResult]:
    results: dict[str, AgentResult] = {}

    trend_result = TrendFilterAgent().analyze(context)
    results["trend_filter"] = trend_result
    if context.get("h4_candles"):
        results["h4_trend_filter"] = analyze_h4_trend(context)

    enriched_context = {
        **context,
        "trend_direction": trend_result.direction,
    }

    for agent in build_agents():
        results[agent.name] = agent.analyze(enriched_context)

    liquidity_context = {
        **enriched_context,
        "agent_results": results,
    }
    results["liquidity"] = LiquidityAgent().analyze(liquidity_context)

    return results


def resolve_zone_cluster(
    fvg: AgentResult,
    order_block: AgentResult,
) -> AgentResult:
    """Merge FVG and OB into a single Zone vote (higher confidence wins)."""
    if fvg.confidence >= order_block.confidence:
        winner, source = fvg, "FVG"
    else:
        winner, source = order_block, "OB"
    return AgentResult(
        direction=winner.direction,
        confidence=winner.confidence,
        reason=f"Zone ({source}): {winner.reason}",
    )


def primary_vote_count(config: DecisionConfig) -> int:
    return len(CLUSTER_PRIMARY_NAMES if config.use_zone_cluster else PRIMARY_AGENT_NAMES)


def count_primary_agreement(
    results: dict[str, AgentResult],
    direction: Direction,
    *,
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> int:
    if config.use_zone_cluster:
        zone = resolve_zone_cluster(results["fvg"], results["order_block"])
        votes = 0
        if results["smc"].direction == direction:
            votes += 1
        if results["liquidity"].direction == direction:
            votes += 1
        if zone.direction == direction:
            votes += 1
        return votes

    return sum(
        1
        for name in PRIMARY_AGENT_NAMES
        if results[name].direction == direction
    )


def format_primary_agreement(
    results: dict[str, AgentResult],
    direction: Direction,
    *,
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> str:
    if config.use_zone_cluster:
        labels = {
            "smc": "SMC",
            "liquidity": "Liquidity",
            "zone": "Zone",
        }
        zone = resolve_zone_cluster(results["fvg"], results["order_block"])
        cluster_results = {
            "smc": results["smc"],
            "liquidity": results["liquidity"],
            "zone": zone,
        }
        agreeing = [
            labels[name]
            for name in CLUSTER_PRIMARY_NAMES
            if cluster_results[name].direction == direction
        ]
    else:
        labels = {
            "smc": "SMC",
            "liquidity": "Liquidity",
            "fvg": "FVG",
            "order_block": "OB",
        }
        agreeing = [
            labels[name]
            for name in PRIMARY_AGENT_NAMES
            if results[name].direction == direction
        ]

    if agreeing:
        return ", ".join(agreeing)
    return "none"


def _weighted_direction_scores(
    results: dict[str, AgentResult],
    config: DecisionConfig,
) -> tuple[float, float]:
    long_score = 0.0
    short_score = 0.0

    if config.use_zone_cluster:
        zone = resolve_zone_cluster(results["fvg"], results["order_block"])
        weighted = {
            "smc": (results["smc"], AGENT_WEIGHTS["smc"]),
            "liquidity": (results["liquidity"], AGENT_WEIGHTS["liquidity"]),
            "zone": (zone, ZONE_CLUSTER_WEIGHT),
        }
        for result, weight in weighted.values():
            if result.direction == Direction.LONG:
                long_score += result.confidence * weight
            elif result.direction == Direction.SHORT:
                short_score += result.confidence * weight
    else:
        for name, weight in AGENT_WEIGHTS.items():
            result = results[name]
            if result.direction == Direction.LONG:
                long_score += result.confidence * weight
            elif result.direction == Direction.SHORT:
                short_score += result.confidence * weight

    return round(long_score, 4), round(short_score, 4)


def _consensus_weighted_score(
    results: dict[str, AgentResult],
    direction: Direction,
    config: DecisionConfig,
) -> float:
    if config.use_zone_cluster:
        zone = resolve_zone_cluster(results["fvg"], results["order_block"])
        total = 0.0
        if results["smc"].direction == direction:
            total += results["smc"].confidence * AGENT_WEIGHTS["smc"]
        if results["liquidity"].direction == direction:
            total += results["liquidity"].confidence * AGENT_WEIGHTS["liquidity"]
        if zone.direction == direction:
            total += zone.confidence * ZONE_CLUSTER_WEIGHT
        return total

    return sum(
        results[name].confidence * AGENT_WEIGHTS[name]
        for name in PRIMARY_AGENT_NAMES
        if results[name].direction == direction
    )


def trend_confirms_signal(
    trend: AgentResult | None,
    direction: Direction,
) -> bool:
    if trend is None or direction == Direction.NEUTRAL:
        return False
    return trend.direction == direction


def resolve_consensus_direction(
    results: dict[str, AgentResult],
    *,
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> Direction:
    """Pick direction when >=2 primary agents agree and H1 trend confirms."""
    trend = results.get("trend_filter")
    best_direction = Direction.NEUTRAL
    best_votes = 0
    best_score = -1.0

    for direction in (Direction.LONG, Direction.SHORT):
        votes = count_primary_agreement(results, direction, config=config)
        if votes < MIN_PRIMARY_AGREEMENT:
            continue
        if not trend_confirms_signal(trend, direction):
            continue
        score = _consensus_weighted_score(results, direction, config)
        if votes > best_votes or (votes == best_votes and score > best_score):
            best_direction = direction
            best_votes = votes
            best_score = score

    return best_direction


def core_agents_agree(
    results: dict[str, AgentResult],
    direction: Direction,
    *,
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> bool:
    """Return True when at least two primary agents share the same direction."""
    return count_primary_agreement(results, direction, config=config) >= MIN_PRIMARY_AGREEMENT


def format_agents_agreement(
    results: dict[str, AgentResult],
    direction: Direction,
    *,
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> str:
    return "Yes" if core_agents_agree(results, direction, config=config) else "No"


def _high_confidence_agreement(
    results: dict[str, AgentResult],
    direction: Direction,
    config: DecisionConfig,
) -> bool:
    if config.use_zone_cluster:
        zone = resolve_zone_cluster(results["fvg"], results["order_block"])
        return (
            results["smc"].direction == direction
            and results["liquidity"].direction == direction
            and zone.direction == direction
        )
    return core_agents_agree(results, direction, config=config)


def apply_confidence_cap(
    results: dict[str, AgentResult],
    direction: Direction,
    raw_confidence: float,
    *,
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> float:
    """Allow confidence above 80% only when core primary agents all agree."""
    if raw_confidence <= HIGH_CONFIDENCE_THRESHOLD:
        return round(raw_confidence, 2)
    if _high_confidence_agreement(results, direction, config):
        return round(min(1.0, raw_confidence), 2)
    return CAPPED_MAX_CONFIDENCE


def compute_final_decision(
    results: dict[str, AgentResult],
    config: DecisionConfig = LEGACY_DECISION_CONFIG,
) -> tuple[Direction, float, float, float]:
    long_score, short_score = _weighted_direction_scores(results, config)

    direction = resolve_consensus_direction(results, config=config)
    if direction == Direction.NEUTRAL:
        return Direction.NEUTRAL, 0.0, long_score, short_score

    winning_score = long_score if direction == Direction.LONG else short_score
    confidence = apply_confidence_cap(results, direction, winning_score, config=config)
    return direction, confidence, long_score, short_score


def build_signal_reason(results: dict[str, AgentResult], direction: Direction) -> str:
    matching = [
        result.reason
        for name, result in results.items()
        if name in DECISION_AGENT_NAMES and result.direction == direction
    ]
    if matching:
        return " | ".join(matching)
    return f"{direction.value.upper()} signal from aggregated agent scores"
