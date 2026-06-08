from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any


class Direction(str, Enum):
    """Trade bias returned by an agent."""

    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class AgentResult:
    """Standard output contract for all agents."""

    direction: Direction
    confidence: float
    reason: str

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")


class Agent(ABC):
    """Base class for trading analysis agents."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable agent identifier."""

    @abstractmethod
    def analyze(self, context: dict[str, Any]) -> AgentResult:
        """Produce a directional signal from the given market context."""
