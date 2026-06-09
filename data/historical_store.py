from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import PROJECT_ROOT
from data.providers.base import Candle

DEFAULT_HISTORICAL_DIR = PROJECT_ROOT / "data" / "historical"
XAUUSD_M15_30D_FILE = DEFAULT_HISTORICAL_DIR / "xauusd_m15_30d.json"


def candle_open_time_iso(candle: Candle) -> str:
    open_time = candle.get("open_time")
    if open_time is None:
        return ""
    return datetime.fromtimestamp(open_time / 1000, tz=timezone.utc).isoformat()


def timeframe_summary(candles: list[Candle]) -> dict[str, Any]:
    if not candles:
        return {"candle_count": 0, "first_open_time": None, "last_open_time": None}

    return {
        "candle_count": len(candles),
        "first_open_time": candle_open_time_iso(candles[0]),
        "last_open_time": candle_open_time_iso(candles[-1]),
    }


def save_historical_dataset(
    path: Path,
    *,
    symbol: str,
    data_symbol: str,
    source: str,
    period_days: int,
    candles_by_timeframe: dict[str, list[Candle]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    timeframes_payload: dict[str, Any] = {}
    for timeframe, candles in candles_by_timeframe.items():
        timeframes_payload[timeframe] = {
            **timeframe_summary(candles),
            "candles": candles,
        }

    payload = {
        "symbol": symbol,
        "data_symbol": data_symbol,
        "source": source,
        "period_days": period_days,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
        "timeframes": timeframes_payload,
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    return path


def load_historical_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Historical dataset not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    timeframes = payload.get("timeframes", {})
    for timeframe, block in timeframes.items():
        if "candles" not in block:
            raise ValueError(f"Dataset missing candles for timeframe '{timeframe}'")

    return payload


def load_candles(path: Path, timeframe: str) -> list[Candle]:
    payload = load_historical_dataset(path)
    timeframes = payload["timeframes"]
    if timeframe not in timeframes:
        available = ", ".join(sorted(timeframes))
        raise KeyError(
            f"Timeframe '{timeframe}' not in {path.name}. Available: {available}"
        )
    return timeframes[timeframe]["candles"]
