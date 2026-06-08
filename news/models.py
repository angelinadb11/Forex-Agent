from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class NewsAction(str, Enum):
    NONE = "none"
    BLOCK = "block"
    WARN = "warn"


@dataclass(frozen=True)
class EconomicEvent:
    """Normalized high-impact economic event."""

    name: str
    label: str
    country: str
    event_time: datetime


@dataclass(frozen=True)
class SymbolNewsPolicy:
    """Symbol-specific behavior during tracked news windows."""

    action: NewsAction
    buffer_minutes: int = 15


@dataclass(frozen=True)
class NewsEvaluation:
    """Result of checking whether a news window affects new signals."""

    action: NewsAction
    event: EconomicEvent | None = None
    message: str | None = None

    @property
    def in_window(self) -> bool:
        return self.action != NewsAction.NONE
