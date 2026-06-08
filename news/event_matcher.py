from __future__ import annotations

import re

TRACKED_EVENT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bcore\s+cpi\b", re.IGNORECASE), "Core CPI"),
    (re.compile(r"\bcpi\b|consumer price index", re.IGNORECASE), "CPI"),
    (re.compile(r"non[-\s]?farm|nonfarm|\bnfp\b|payroll", re.IGNORECASE), "NFP"),
    (re.compile(r"\bfomc\b|federal open market", re.IGNORECASE), "FOMC"),
    (
        re.compile(
            r"interest rate decision|fed interest rate|federal funds rate|fed funds rate",
            re.IGNORECASE,
        ),
        "Interest Rate Decision",
    ),
    (re.compile(r"\bpce\b|personal consumption expenditures", re.IGNORECASE), "PCE"),
    (re.compile(r"\bpowell\b|fed chair", re.IGNORECASE), "Powell Speech"),
)

US_COUNTRY_CODES = frozenset({"US", "USA", "USD", "UNITED STATES"})


def classify_tracked_event(event_name: str, country: str) -> str | None:
    """Return a display label when the event matches a tracked US macro release."""
    if country.strip().upper() not in US_COUNTRY_CODES:
        return None

    for pattern, label in TRACKED_EVENT_PATTERNS:
        if pattern.search(event_name):
            return label
    return None
