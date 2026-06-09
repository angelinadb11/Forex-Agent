from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agents.base import AgentResult, Direction
from config.settings import PROJECT_ROOT

ACTIVE_TRADES_FILE = PROJECT_ROOT / "active_trades.json"


def _agent_result_to_dict(result: AgentResult) -> dict[str, Any]:
    return {
        "direction": result.direction.value,
        "confidence": result.confidence,
        "reason": result.reason,
    }


def _agent_result_from_dict(payload: dict[str, Any]) -> AgentResult:
    return AgentResult(
        direction=Direction(payload["direction"]),
        confidence=float(payload["confidence"]),
        reason=str(payload["reason"]),
    )


def _serialize_agent_results(
    results: dict[str, AgentResult] | None,
) -> dict[str, dict[str, Any]] | None:
    if results is None:
        return None
    return {name: _agent_result_to_dict(result) for name, result in results.items()}


def _deserialize_agent_results(
    payload: dict[str, dict[str, Any]] | None,
) -> dict[str, AgentResult] | None:
    if payload is None:
        return None
    return {name: _agent_result_from_dict(item) for name, item in payload.items()}


class ActiveTradesStore:
    """Persists open trades so monitoring survives bot restarts."""

    def __init__(self, file_path: Path = ACTIVE_TRADES_FILE) -> None:
        self.file_path = file_path

    def load(self) -> list:
        from tracking.trade_monitor import ActiveTrade

        if not self.file_path.exists():
            return []

        with self.file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        trades: list[ActiveTrade] = []
        for item in payload:
            data = dict(item)
            entry_agent_results = _deserialize_agent_results(data.pop("entry_agent_results", None))
            entry_trend = data.get("entry_trend_direction")
            if entry_trend is not None:
                data["entry_trend_direction"] = Direction(entry_trend)
            data["direction"] = Direction(data["direction"])
            trades.append(
                ActiveTrade(
                    entry_agent_results=entry_agent_results,
                    **data,
                )
            )
        return [trade for trade in trades if not trade.closed]

    def save(self, trades: list) -> None:
        open_trades = [trade for trade in trades if not trade.closed]
        payload: list[dict[str, Any]] = []
        for trade in open_trades:
            payload.append(
                {
                    "symbol": trade.symbol,
                    "direction": trade.direction.value,
                    "entry": trade.entry,
                    "stop_loss": trade.stop_loss,
                    "tp1": trade.tp1,
                    "tp2": trade.tp2,
                    "tp3": trade.tp3,
                    "confidence": trade.confidence,
                    "reason": trade.reason,
                    "open_time": trade.open_time,
                    "initial_stop_loss": trade.initial_stop_loss,
                    "agents_agreement": trade.agents_agreement,
                    "timeframe": trade.timeframe,
                    "entry_agent_results": _serialize_agent_results(trade.entry_agent_results),
                    "entry_trend_direction": (
                        trade.entry_trend_direction.value
                        if trade.entry_trend_direction is not None
                        else None
                    ),
                    "entry_zone_low": trade.entry_zone_low,
                    "entry_zone_high": trade.entry_zone_high,
                    "entry_zone_kind": trade.entry_zone_kind,
                    "entry_rsi": trade.entry_rsi,
                    "last_rsi": trade.last_rsi,
                    "structure_warning_count": trade.structure_warning_count,
                    "last_structure_candle_open_time": trade.last_structure_candle_open_time,
                    "last_structure_check_monotonic": trade.last_structure_check_monotonic,
                    "telegram_message_id": trade.telegram_message_id,
                    "lot_size": trade.lot_size,
                    "level1_warning_sent": trade.level1_warning_sent,
                    "level2_warning_sent": trade.level2_warning_sent,
                    "level2_streak": trade.level2_streak,
                    "trend_warning_sent": trade.trend_warning_sent,
                    "last_trend_check_monotonic": trade.last_trend_check_monotonic,
                    "last_trend_candle_open_time": trade.last_trend_candle_open_time,
                    "tp1_reply_sent": trade.tp1_reply_sent,
                    "tp2_reply_sent": trade.tp2_reply_sent,
                    "tp3_reply_sent": trade.tp3_reply_sent,
                    "sl_reply_sent": trade.sl_reply_sent,
                    "sl_proximity_warning_sent": trade.sl_proximity_warning_sent,
                    "profit_milestones_sent": trade.profit_milestones_sent or [],
                    "tp1_hit": trade.tp1_hit,
                    "tp2_hit": trade.tp2_hit,
                    "tp3_hit": trade.tp3_hit,
                    "closed": trade.closed,
                    "close_time": trade.close_time,
                    "result": trade.result,
                    "recorded": trade.recorded,
                }
            )

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
