from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from agents.base import AgentResult, Direction
from agents.session_agent import is_london_or_new_york_session
from news.models import NewsAction

MIN_CONFIDENCE = 0.70
MIN_CONFIDENCE_PCT = 70
MIN_SESSION_CONFIDENCE = 0.30


@dataclass(frozen=True)
class FilterResult:
    approved: bool
    direction: Direction
    confidence: float
    message: str
    news_warning: str | None = None


class SignalFilter:
    """Filters signals before they are sent or simulated."""

    def __init__(
        self,
        min_confidence: float = MIN_CONFIDENCE,
        min_session_confidence: float = MIN_SESSION_CONFIDENCE,
        london_ny_session_symbols: frozenset[str] | set[str] | None = None,
        session_confidence_symbols: frozenset[str] | set[str] | None = None,
        news_gate=None,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_session_confidence = min_session_confidence
        self.london_ny_session_symbols = frozenset(london_ny_session_symbols or ())
        self.session_confidence_symbols = frozenset(session_confidence_symbols or ())
        self.news_gate = news_gate

    def evaluate(
        self,
        results: dict[str, AgentResult],
        final_direction: Direction,
        final_confidence: float,
        *,
        symbol: str | None = None,
        timestamp: datetime | None = None,
    ) -> FilterResult:
        smc = results["smc"]
        liquidity = results["liquidity"]
        session = results["session"]
        symbol_key = symbol.upper() if symbol else None

        if (
            symbol_key
            and symbol_key in self.london_ny_session_symbols
            and not is_london_or_new_york_session(timestamp)
        ):
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=(
                    f"NO TRADE: {symbol_key} signals only allowed during "
                    "London or New York session"
                ),
            )

        if final_direction == Direction.NEUTRAL:
            return FilterResult(
                approved=False,
                direction=Direction.NEUTRAL,
                confidence=final_confidence,
                message="NO TRADE: neutral decision",
            )

        if not self._smc_agrees_with_liquidity(smc, liquidity):
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=(
                    "NO TRADE: SMC and Liquidity do not agree "
                    f"(SMC={smc.direction.value}, Liquidity={liquidity.direction.value})"
                ),
            )

        if smc.direction != final_direction or liquidity.direction != final_direction:
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message="NO TRADE: SMC/Liquidity agreement does not match final direction",
            )

        if final_confidence < self.min_confidence:
            return FilterResult(
                approved=False,
                direction=final_direction,
                confidence=final_confidence,
                message=(
                    f"NO TRADE: confidence {final_confidence:.2f} "
                    f"below minimum {self.min_confidence:.2f} ({MIN_CONFIDENCE_PCT}%)"
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
            confidence=final_confidence,
            message="Signal approved",
            news_warning=news_warning,
        )

    @staticmethod
    def _smc_agrees_with_liquidity(smc: AgentResult, liquidity: AgentResult) -> bool:
        if smc.direction == Direction.NEUTRAL or liquidity.direction == Direction.NEUTRAL:
            return False
        return smc.direction == liquidity.direction
