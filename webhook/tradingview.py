from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from agents.base import Direction


@dataclass(frozen=True)
class TradingViewAlert:
    """Normalized alert payload from TradingView webhook message."""

    symbol: str
    direction: Direction
    entry: float | None
    stop_loss: float | None
    tp1: float | None
    tp2: float | None
    tp3: float | None
    timeframe: str
    note: str
    raw_text: str
    secret: str | None = None

    @property
    def has_levels(self) -> bool:
        return self.entry is not None and self.stop_loss is not None


def _parse_direction(raw: object) -> Direction:
    text = str(raw or "").strip().upper()
    if text in {"BUY", "LONG", "BULL", "BULLISH", "КУПИТИ", "КУПІВЛЯ"}:
        return Direction.LONG
    if text in {"SELL", "SHORT", "BEAR", "BEARISH", "ПРОДАТИ", "ПРОДАЖ"}:
        return Direction.SHORT
    return Direction.NEUTRAL


def _parse_float(raw: object) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _normalize_symbol(raw: object) -> str:
    symbol = str(raw or "XAUUSD").strip().upper()
    symbol = symbol.replace("OANDA:", "").replace("BINANCE:", "").replace("FX:", "")
    if symbol.endswith("USDT"):
        symbol = "XAUUSD" if "XAU" in symbol else symbol
    return symbol or "XAUUSD"


def _normalize_timeframe(raw: object) -> str:
    text = str(raw or "").strip().lower()
    if not text:
        return "5m"
    mapping = {
        "1": "1m",
        "5": "5m",
        "15": "15m",
        "60": "1h",
        "240": "4h",
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "1h",
        "4h": "4h",
    }
    return mapping.get(text, text)


def _parse_plain_text(body: str) -> TradingViewAlert:
    direction = Direction.NEUTRAL
    upper = body.upper()
    if any(word in upper for word in (" BUY", " LONG", "КУПИТИ", "BULLISH")):
        direction = Direction.LONG
    elif any(word in upper for word in (" SELL", " SHORT", "ПРОДАТИ", "BEARISH")):
        direction = Direction.SHORT

    symbol_match = re.search(r"\b([A-Z]{3,6}(?:USD|USDT)?)\b", body.upper())
    symbol = symbol_match.group(1) if symbol_match else "XAUUSD"

    def _find_level(label: str) -> float | None:
        pattern = rf"{label}\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)"
        match = re.search(pattern, body, flags=re.IGNORECASE)
        return float(match.group(1)) if match else None

    return TradingViewAlert(
        symbol=_normalize_symbol(symbol),
        direction=direction,
        entry=_find_level("entry") or _find_level("вхід"),
        stop_loss=_find_level("sl") or _find_level("stop"),
        tp1=_find_level("tp1") or _find_level("тп1"),
        tp2=_find_level("tp2") or _find_level("тп2"),
        tp3=_find_level("tp3") or _find_level("тп3"),
        timeframe="5m",
        note=body.strip(),
        raw_text=body.strip(),
    )


def parse_tradingview_payload(body: str) -> TradingViewAlert:
    """Parse TradingView webhook body (JSON or plain text)."""
    text = body.strip()
    if not text:
        raise ValueError("Empty TradingView webhook body")

    if text.startswith("{"):
        payload: dict[str, Any] = json.loads(text)
        direction = _parse_direction(
            payload.get("action")
            or payload.get("direction")
            or payload.get("side")
            or payload.get("signal")
        )
        return TradingViewAlert(
            symbol=_normalize_symbol(payload.get("symbol") or payload.get("ticker")),
            direction=direction,
            entry=_parse_float(payload.get("entry") or payload.get("price")),
            stop_loss=_parse_float(payload.get("sl") or payload.get("stop_loss") or payload.get("stop")),
            tp1=_parse_float(payload.get("tp1") or payload.get("take_profit")),
            tp2=_parse_float(payload.get("tp2")),
            tp3=_parse_float(payload.get("tp3")),
            timeframe=_normalize_timeframe(payload.get("timeframe") or payload.get("interval")),
            note=str(payload.get("note") or payload.get("message") or "").strip(),
            raw_text=text,
            secret=str(payload.get("secret") or payload.get("token") or "").strip() or None,
        )

    return _parse_plain_text(text)
