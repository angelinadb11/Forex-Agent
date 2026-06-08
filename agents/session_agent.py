from __future__ import annotations

from datetime import datetime, time, timezone
from typing import Any

from agents.base import Agent, AgentResult, Direction

LONDON_START = time(7, 0)
LONDON_END = time(16, 0)
NEW_YORK_START = time(13, 30)
NEW_YORK_END = time(20, 0)

OVERLAP_CONFIDENCE = 0.50
SESSION_CONFIDENCE = 0.30


def _to_utc_time(value: datetime) -> time:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.time()


def _in_time_window(current: time, start: time, end: time) -> bool:
    return start <= current < end


def is_london_or_new_york_session(current: datetime | None = None) -> bool:
    """Return True when London or New York session is active (UTC)."""
    current_dt = current or datetime.now(timezone.utc)
    current_time = _to_utc_time(current_dt)

    in_london = _in_time_window(current_time, LONDON_START, LONDON_END)
    in_new_york = _in_time_window(current_time, NEW_YORK_START, NEW_YORK_END)
    return in_london or in_new_york


def evaluate_session(current: datetime | None = None) -> tuple[str, float, str]:
    """Return session name, confidence, and reason for the given UTC time."""
    current_dt = current or datetime.now(timezone.utc)
    current_time = _to_utc_time(current_dt)

    in_london = _in_time_window(current_time, LONDON_START, LONDON_END)
    in_new_york = _in_time_window(current_time, NEW_YORK_START, NEW_YORK_END)

    if in_london and in_new_york:
        session_name = "London-New York overlap"
        confidence = OVERLAP_CONFIDENCE
        reason = "London (07:00-16:00 UTC) and New York (13:30-20:00 UTC) are both active"
        return session_name, confidence, reason

    if in_london:
        session_name = "London"
        confidence = SESSION_CONFIDENCE
        reason = "Active session window 07:00-16:00 UTC"
        return session_name, confidence, reason

    if in_new_york:
        session_name = "New York"
        confidence = SESSION_CONFIDENCE
        reason = "Active session window 13:30-20:00 UTC"
        return session_name, confidence, reason

    session_name = "Off-hours"
    return session_name, 0.0, "Outside London and New York sessions"


class SessionAgent(Agent):
    """Trading session / time-of-day analysis agent."""

    @property
    def name(self) -> str:
        return "session"

    def analyze(self, context: dict[str, Any]) -> AgentResult:
        timestamp = context.get("timestamp")
        if isinstance(timestamp, datetime):
            session_name, confidence, reason = evaluate_session(timestamp)
        else:
            session_name, confidence, reason = evaluate_session()

        return AgentResult(
            direction=Direction.NEUTRAL,
            confidence=confidence,
            reason=f"Session: {session_name} - {reason}",
        )
