from typing import Any

from agents.base import Agent, AgentResult, Direction


class TradeManagerAgent(Agent):
    """Aggregates agent signals and manages trade decisions."""

    @property
    def name(self) -> str:
        return "trade_manager"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        # Trading logic not implemented yet.
        return AgentResult(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="Trade management not implemented",
        )

    def aggregate(self, signals: list[AgentResult]) -> AgentResult:
        """Combine individual agent signals into a final decision."""
        # Trading logic not implemented yet.
        return AgentResult(
            direction=Direction.NEUTRAL,
            confidence=0.0,
            reason="Signal aggregation not implemented",
        )
