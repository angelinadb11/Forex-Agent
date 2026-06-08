from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from config.settings import PROJECT_ROOT

SIGNALS_CSV_FILE = PROJECT_ROOT / "signals.csv"

CSV_COLUMNS = [
    "Date",
    "Symbol",
    "Direction",
    "Entry",
    "SL",
    "TP1",
    "TP2",
    "TP3",
    "Result",
    "ProfitLoss",
    "Confidence",
    "AgentsAgreement",
]


@dataclass(frozen=True)
class SignalCsvRow:
    date: str
    symbol: str
    direction: str
    entry: float
    sl: float
    tp1: float
    tp2: float
    tp3: float
    result: str
    profit_loss: float
    confidence: float
    agents_agreement: str

    def to_list(self) -> list[str | float]:
        return [
            self.date,
            self.symbol,
            self.direction.upper(),
            f"{self.entry:.2f}",
            f"{self.sl:.2f}",
            f"{self.tp1:.2f}",
            f"{self.tp2:.2f}",
            f"{self.tp3:.2f}",
            self.result,
            f"{self.profit_loss:.2f}",
            f"{self.confidence * 100:.2f}",
            self.agents_agreement,
        ]


class SignalCsvStore:
    """Appends every signal and result to a CSV file."""

    def __init__(self, file_path: Path = SIGNALS_CSV_FILE) -> None:
        self.file_path = file_path
        self._ensure_header()

    def append(self, row: SignalCsvRow) -> None:
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(row.to_list())

    def _ensure_header(self) -> None:
        if self.file_path.exists():
            return

        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("w", newline="", encoding="utf-8") as handle:
            csv.writer(handle).writerow(CSV_COLUMNS)
