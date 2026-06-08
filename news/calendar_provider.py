from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

import requests

from news.event_matcher import classify_tracked_event
from news.models import EconomicEvent

LOGGER = logging.getLogger(__name__)

FINNHUB_CALENDAR_URL = "https://finnhub.io/api/v1/calendar/economic"
FOREX_FACTORY_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
EASTERN = ZoneInfo("America/New_York")


class EconomicCalendarProvider(ABC):
    """Fetch normalized economic calendar events."""

    @abstractmethod
    def fetch_events(self, start: date, end: date) -> list[EconomicEvent]:
        """Return tracked US macro events between start and end inclusive."""


class FinnhubCalendarProvider(EconomicCalendarProvider):
    """Primary provider using Finnhub's economic calendar API."""

    def __init__(self, api_key: str, timeout: float = 20.0) -> None:
        self.api_key = api_key.strip()
        self.timeout = timeout

    def fetch_events(self, start: date, end: date) -> list[EconomicEvent]:
        if not self.api_key:
            return []

        params = {
            "from": start.isoformat(),
            "to": end.isoformat(),
            "token": self.api_key,
        }
        try:
            response = requests.get(
                FINNHUB_CALENDAR_URL,
                params=params,
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except requests.RequestException as exc:
            LOGGER.warning("Finnhub calendar fetch failed: %s", exc)
            return []

        events: list[EconomicEvent] = []
        for item in payload.get("economicCalendar", []):
            parsed = self._parse_item(item)
            if parsed is not None:
                events.append(parsed)
        return events

    @staticmethod
    def _parse_item(item: dict[str, Any]) -> EconomicEvent | None:
        event_name = str(item.get("event", "")).strip()
        country = str(item.get("country", "")).strip()
        label = classify_tracked_event(event_name, country)
        if label is None:
            return None

        raw_time = str(item.get("time", "")).strip()
        if not raw_time:
            return None

        try:
            event_time = datetime.strptime(raw_time, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=timezone.utc
            )
        except ValueError:
            return None

        return EconomicEvent(
            name=event_name,
            label=label,
            country=country,
            event_time=event_time,
        )


class ForexFactoryCalendarProvider(EconomicCalendarProvider):
    """Best-effort fallback using the public Forex Factory weekly JSON feed."""

    def __init__(
        self,
        calendar_url: str = FOREX_FACTORY_CALENDAR_URL,
        timeout: float = 20.0,
    ) -> None:
        self.calendar_url = calendar_url
        self.timeout = timeout

    def fetch_events(self, start: date, end: date) -> list[EconomicEvent]:
        try:
            response = requests.get(self.calendar_url, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError) as exc:
            LOGGER.warning("Forex Factory calendar fetch failed: %s", exc)
            return []

        if not isinstance(payload, list):
            return []

        events: list[EconomicEvent] = []
        for item in payload:
            parsed = self._parse_item(item, start, end)
            if parsed is not None:
                events.append(parsed)
        return events

    @staticmethod
    def _parse_item(
        item: dict[str, Any],
        start: date,
        end: date,
    ) -> EconomicEvent | None:
        title = str(item.get("title", "")).strip()
        country = str(item.get("country", "USD")).strip()
        label = classify_tracked_event(title, country)
        if label is None:
            return None

        event_time = ForexFactoryCalendarProvider._parse_event_time(item)
        if event_time is None:
            return None
        if not (start <= event_time.date() <= end):
            return None

        return EconomicEvent(
            name=title,
            label=label,
            country=country,
            event_time=event_time,
        )

    @staticmethod
    def _parse_event_time(item: dict[str, Any]) -> datetime | None:
        raw_date = str(item.get("date", "")).strip()
        raw_time = str(item.get("time", "")).strip().lower()
        if not raw_date or raw_time in {"", "tentative", "all day", "day 1", "day 2", "day 3"}:
            return None

        if "T" in raw_date:
            try:
                parsed = datetime.fromisoformat(raw_date.replace("Z", "+00:00"))
                return parsed.astimezone(timezone.utc)
            except ValueError:
                return None

        try:
            event_date = date.fromisoformat(raw_date[:10])
        except ValueError:
            return None

        parsed_time = ForexFactoryCalendarProvider._parse_clock(raw_time)
        if parsed_time is None:
            return None

        local_dt = datetime.combine(event_date, parsed_time, tzinfo=EASTERN)
        return local_dt.astimezone(timezone.utc)

    @staticmethod
    def _parse_clock(raw_time: str) -> time | None:
        cleaned = raw_time.strip().lower().replace(" ", "")
        for fmt in ("%I:%M%p", "%H:%M"):
            try:
                return datetime.strptime(cleaned, fmt).time()
            except ValueError:
                continue
        return None


class CachedEconomicCalendar:
    """Cache calendar responses to avoid repeated network calls."""

    def __init__(
        self,
        providers: list[EconomicCalendarProvider],
        cache_ttl_seconds: int = 3600,
    ) -> None:
        self.providers = providers
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache_events: list[EconomicEvent] = []
        self._cache_start: date | None = None
        self._cache_end: date | None = None
        self._cache_loaded_at: datetime | None = None

    def get_events(self, moment: datetime) -> list[EconomicEvent]:
        start = (moment - timedelta(days=1)).date()
        end = (moment + timedelta(days=1)).date()
        if self._cache_is_valid(start, end, moment):
            return self._cache_events

        merged: dict[tuple[str, datetime], EconomicEvent] = {}
        for provider in self.providers:
            for event in provider.fetch_events(start, end):
                merged[(event.label, event.event_time)] = event

        self._cache_events = sorted(
            merged.values(),
            key=lambda event: event.event_time,
        )
        self._cache_start = start
        self._cache_end = end
        self._cache_loaded_at = moment
        return self._cache_events

    def _cache_is_valid(self, start: date, end: date, moment: datetime) -> bool:
        if self._cache_loaded_at is None:
            return False
        if self._cache_start != start or self._cache_end != end:
            return False
        age = moment - self._cache_loaded_at
        return age.total_seconds() < self.cache_ttl_seconds
