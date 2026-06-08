from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from config.symbols import resolve_symbol
from news.calendar_provider import (
    FOREX_FACTORY_CALENDAR_URL,
    CachedEconomicCalendar,
    EconomicCalendarProvider,
    FinnhubCalendarProvider,
    ForexFactoryCalendarProvider,
)
from news.models import EconomicEvent, NewsAction, NewsEvaluation, SymbolNewsPolicy

LOGGER = logging.getLogger(__name__)

DEFAULT_BUFFER_MINUTES = 15

DEFAULT_SYMBOL_POLICIES: dict[str, SymbolNewsPolicy] = {
    "XAUUSD": SymbolNewsPolicy(action=NewsAction.BLOCK, buffer_minutes=DEFAULT_BUFFER_MINUTES),
    # Blocking is safer than confidence reduction during US macro windows.
    "BTCUSDT": SymbolNewsPolicy(action=NewsAction.BLOCK, buffer_minutes=DEFAULT_BUFFER_MINUTES),
    "DJ30": SymbolNewsPolicy(action=NewsAction.WARN, buffer_minutes=DEFAULT_BUFFER_MINUTES),
    "US30": SymbolNewsPolicy(action=NewsAction.WARN, buffer_minutes=DEFAULT_BUFFER_MINUTES),
}

NEWS_WARNING_MESSAGE = (
    "⚠️ High-impact news window active.\n"
    "Expected volatility is elevated."
)


def build_calendar_service(
    *,
    finnhub_api_key: str = "",
    calendar_url: str = FOREX_FACTORY_CALENDAR_URL,
    cache_ttl_seconds: int = 3600,
) -> CachedEconomicCalendar:
    providers: list[EconomicCalendarProvider] = []
    if finnhub_api_key.strip():
        providers.append(FinnhubCalendarProvider(finnhub_api_key))

    providers.append(ForexFactoryCalendarProvider(calendar_url=calendar_url))
    return CachedEconomicCalendar(providers, cache_ttl_seconds=cache_ttl_seconds)


class NewsGate:
    """Symbol-specific high-impact news handling for new signal generation only."""

    def __init__(
        self,
        calendar: CachedEconomicCalendar | None,
        *,
        enabled: bool = True,
        symbol_policies: dict[str, SymbolNewsPolicy] | None = None,
        fail_open: bool = True,
    ) -> None:
        self.calendar = calendar
        self.enabled = enabled
        self.symbol_policies = symbol_policies or DEFAULT_SYMBOL_POLICIES
        self.fail_open = fail_open

    def evaluate(
        self,
        symbol: str,
        moment: datetime | None = None,
    ) -> NewsEvaluation:
        if not self.enabled:
            return NewsEvaluation(action=NewsAction.NONE)

        symbol_key = resolve_symbol(symbol).display
        policy = self.symbol_policies.get(symbol_key)
        if policy is None or policy.action == NewsAction.NONE:
            return NewsEvaluation(action=NewsAction.NONE)

        current = moment or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        else:
            current = current.astimezone(timezone.utc)

        if self.calendar is None:
            if self.fail_open:
                LOGGER.warning("News calendar unavailable; allowing signal for %s", symbol_key)
                return NewsEvaluation(action=NewsAction.NONE)
            return NewsEvaluation(
                action=NewsAction.BLOCK,
                message="News calendar unavailable",
            )

        active_event = self._find_active_event(current, policy.buffer_minutes)
        if active_event is None:
            return NewsEvaluation(action=NewsAction.NONE)

        if policy.action == NewsAction.BLOCK:
            return NewsEvaluation(
                action=NewsAction.BLOCK,
                event=active_event,
                message=(
                    f"high-impact news window active ({active_event.label}, "
                    f"±{policy.buffer_minutes}m)"
                ),
            )

        return NewsEvaluation(
            action=NewsAction.WARN,
            event=active_event,
            message=NEWS_WARNING_MESSAGE,
        )

    def _find_active_event(
        self,
        moment: datetime,
        buffer_minutes: int,
    ) -> EconomicEvent | None:
        try:
            events = self.calendar.get_events(moment)
        except Exception as exc:
            LOGGER.warning("News calendar lookup failed: %s", exc)
            if self.fail_open:
                return None
            raise

        window = timedelta(minutes=buffer_minutes)
        for event in events:
            if event.event_time - window <= moment <= event.event_time + window:
                return event
        return None


def build_news_gate(
    *,
    enabled: bool = True,
    buffer_minutes: int = DEFAULT_BUFFER_MINUTES,
    finnhub_api_key: str = "",
    calendar_url: str = FOREX_FACTORY_CALENDAR_URL,
    cache_ttl_seconds: int = 3600,
) -> NewsGate:
    """Create a configured news gate for signal filtering."""
    calendar = build_calendar_service(
        finnhub_api_key=finnhub_api_key,
        calendar_url=calendar_url,
        cache_ttl_seconds=cache_ttl_seconds,
    )
    symbol_policies = {
        symbol: SymbolNewsPolicy(action=policy.action, buffer_minutes=buffer_minutes)
        for symbol, policy in DEFAULT_SYMBOL_POLICIES.items()
    }
    return NewsGate(
        calendar,
        enabled=enabled,
        symbol_policies=symbol_policies,
    )
