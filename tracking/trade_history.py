from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from config.settings import PROJECT_ROOT
from tracking.console import safe_print

TRADE_HISTORY_FILE = PROJECT_ROOT / "trade_history.json"


@dataclass
class TradeRecord:
    symbol: str
    direction: str
    entry: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    confidence: float
    reason: str
    open_time: str
    close_time: str | None = None
    result: str | None = None
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TradeHistoryStore:
    """Persists closed trades to trade_history.json."""

    def __init__(self, file_path: Path = TRADE_HISTORY_FILE) -> None:
        self.file_path = file_path

    def load(self) -> list[TradeRecord]:
        if not self.file_path.exists():
            return []

        with self.file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        return [TradeRecord(**item) for item in payload]

    def save(self, trades: list[TradeRecord]) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", encoding="utf-8") as handle:
            json.dump([trade.to_dict() for trade in trades], handle, indent=2)

    def add_trade(self, trade: TradeRecord) -> None:
        trades = self.load()
        trades.append(trade)
        self.save(trades)


@dataclass
class TradeStatistics:
    total_trades: int
    win_rate: float
    tp1_hit_rate: float
    tp2_hit_rate: float
    tp3_hit_rate: float

    def format(self) -> str:
        return (
            "=== TRADE STATISTICS ===\n"
            f"Total trades: {self.total_trades}\n"
            f"Win rate: {self.win_rate:.1f}%\n"
            f"TP1 hit rate: {self.tp1_hit_rate:.1f}%\n"
            f"TP2 hit rate: {self.tp2_hit_rate:.1f}%\n"
            f"TP3 hit rate: {self.tp3_hit_rate:.1f}%"
        )


class TradeStatisticsCalculator:
    """Calculates aggregate performance metrics from trade history."""

    def calculate(self, trades: list[TradeRecord]) -> TradeStatistics:
        closed = [trade for trade in trades if trade.close_time is not None]
        total = len(closed)

        if total == 0:
            return TradeStatistics(0, 0.0, 0.0, 0.0, 0.0)

        tp1_hits = sum(1 for trade in closed if trade.tp1_hit)
        tp2_hits = sum(1 for trade in closed if trade.tp2_hit)
        tp3_hits = sum(1 for trade in closed if trade.tp3_hit)
        wins = sum(1 for trade in closed if trade.tp1_hit)

        return TradeStatistics(
            total_trades=total,
            win_rate=(wins / total) * 100,
            tp1_hit_rate=(tp1_hits / total) * 100,
            tp2_hit_rate=(tp2_hits / total) * 100,
            tp3_hit_rate=(tp3_hits / total) * 100,
        )

    def print_statistics(
        self,
        store: TradeHistoryStore | None = None,
        trades: list[TradeRecord] | None = None,
    ) -> TradeStatistics:
        if trades is None:
            store = store or TradeHistoryStore()
            trades = store.load()

        stats = self.calculate(trades)
        safe_print()
        safe_print(stats.format())
        return stats


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
