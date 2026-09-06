"""Spot forex weekly session hours (XAUUSD and majors).

OTC forex typically closes Friday ~22:00 UTC and reopens Sunday ~22:00 UTC.
"""

from __future__ import annotations

import os
from datetime import datetime, time, timezone

FOREX_WEEKLY_CLOSE_WEEKDAY = 4  # Friday
FOREX_WEEKLY_CLOSE_UTC = time(22, 0)
FOREX_WEEKLY_OPEN_WEEKDAY = 6  # Sunday
FOREX_WEEKLY_OPEN_UTC = time(22, 0)

WEEKEND_CLOSED_MESSAGE = "Forex market closed (weekend)"


def forex_weekend_block_enabled() -> bool:
    raw = os.getenv("FOREX_BLOCK_WEEKENDS", "1").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_forex_market_open(ts: datetime | None = None) -> tuple[bool, str]:
    """Return whether spot forex is inside its weekly trading window (UTC)."""
    now = ts or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    else:
        now = now.astimezone(timezone.utc)

    weekday = now.weekday()
    clock = now.time()

    if weekday == 5:
        return False, WEEKEND_CLOSED_MESSAGE

    if weekday == 6 and clock < FOREX_WEEKLY_OPEN_UTC:
        return False, WEEKEND_CLOSED_MESSAGE

    if weekday == FOREX_WEEKLY_CLOSE_WEEKDAY and clock >= FOREX_WEEKLY_CLOSE_UTC:
        return False, WEEKEND_CLOSED_MESSAGE

    return True, "Forex market open"


def should_publish_forex_signal(ts: datetime | None = None) -> tuple[bool, str]:
    if not forex_weekend_block_enabled():
        return True, "Weekend block disabled"
    return is_forex_market_open(ts)
