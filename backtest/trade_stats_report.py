"""Detailed trade outcome report for XAUUSD 30-day backtest."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.simulator import TradeManagementMode
from config.sl_config import get_sl_config
from data.historical_store import XAUUSD_M15_30D_FILE, load_candles
from run_xauusd_30d_backtest import (
    DAYS,
    LocalDataBacktestEngine,
    WARMUP,
    BacktestConfig,
)
from signal_generator import price_distance_pips
from tracking.console import configure_console_encoding

LOT_SIZE = 0.01
SYMBOL = "XAUUSD"


@dataclass
class OutcomeBucket:
    label: str
    count: int = 0
    total_r: float = 0.0
    total_dollars: float = 0.0

    @property
    def avg_r(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total_r / self.count


def classify_trade(trade) -> str:
    if not trade.tp1_hit:
        return "sl_full"
    if trade.tp3_hit:
        return "tp3"
    if trade.tp2_hit:
        return "tp2_zone"
    return "be_after_tp1"


def trade_dollars(trade, lot_size: float = LOT_SIZE) -> float:
    config = get_sl_config(SYMBOL)
    assert config is not None
    sl_pips = price_distance_pips(trade.risk, config.pip_size)
    risk_dollars = sl_pips * config.pip_value_per_lot * lot_size
    return trade.pnl_r * risk_dollars


def build_report(trades) -> dict[str, OutcomeBucket]:
    buckets = {
        "sl_full": OutcomeBucket("SL (збиток -1R)"),
        "be_after_tp1": OutcomeBucket("Breakeven після TP1 (0R на решту, 50% зафіксовано)"),
        "tp2_zone": OutcomeBucket("Досягли TP2 (SL на TP1)"),
        "tp3": OutcomeBucket("Досягли TP3"),
    }
    reach_tp1 = OutcomeBucket("Досягли TP1 (хоча б частково)")
    reach_tp2 = OutcomeBucket("Досягли TP2")
    reach_tp3 = OutcomeBucket("Досягли TP3")

    for trade in trades:
        key = classify_trade(trade)
        dollars = trade_dollars(trade)
        buckets[key].count += 1
        buckets[key].total_r += trade.pnl_r
        buckets[key].total_dollars += dollars

        if trade.tp1_hit:
            reach_tp1.count += 1
            reach_tp1.total_r += trade.pnl_r
            reach_tp1.total_dollars += dollars
        if trade.tp2_hit:
            reach_tp2.count += 1
            reach_tp2.total_r += trade.pnl_r
            reach_tp2.total_dollars += dollars
        if trade.tp3_hit:
            reach_tp3.count += 1
            reach_tp3.total_r += trade.pnl_r
            reach_tp3.total_dollars += dollars

    return {
        **buckets,
        "reach_tp1": reach_tp1,
        "reach_tp2": reach_tp2,
        "reach_tp3": reach_tp3,
    }


def print_report(trades) -> None:
    total_r = sum(trade.pnl_r for trade in trades)
    total_dollars = sum(trade_dollars(trade) for trade in trades)
    buckets = build_report(trades)

    print(f"=== Детальна статистика ({len(trades)} угод, {DAYS} днів, lot {LOT_SIZE}) ===")
    print("Логіка: 50% на TP1, +25% на TP2, 25% до TP3")
    print()

    print("--- Результат закриття (взаємовиключні категорії) ---")
    for key in ("sl_full", "be_after_tp1", "tp2_zone", "tp3"):
        bucket = buckets[key]
        print(
            f"{bucket.label}: "
            f"{bucket.count} угод | "
            f"сума {bucket.total_r:+.2f}R | "
            f"avg {bucket.avg_r:+.2f}R | "
            f"${bucket.total_dollars:+.2f}"
        )

    print()
    print("--- Досягнення TP (можуть перетинатись) ---")
    for key in ("reach_tp1", "reach_tp2", "reach_tp3"):
        bucket = buckets[key]
        print(f"{bucket.label}: {bucket.count} угод")

    print()
    print("--- Планові R на рівень (якщо б закрили 100% на TP) ---")
    sl_cfg = get_sl_config(SYMBOL)
    assert sl_cfg is not None
    avg_sl_pips = sum(
        price_distance_pips(trade.risk, sl_cfg.pip_size) for trade in trades
    ) / len(trades)
    one_r_dollars = avg_sl_pips * sl_cfg.pip_value_per_lot * LOT_SIZE
    print(f"Середній SL: {avg_sl_pips:.1f} pips → 1R ≈ ${one_r_dollars:.2f} на {LOT_SIZE} lot")
    print(f"TP1 = 1.5R ≈ ${1.5 * one_r_dollars:.2f}  |  TP2 = 2.5R ≈ ${2.5 * one_r_dollars:.2f}  |  TP3 = 3.5R ≈ ${3.5 * one_r_dollars:.2f}")

    print()
    print("--- Підсумок ---")
    print(f"Загальний результат:          {total_r:+.2f}R")
    print(f"Середній на угоду:            {total_r / len(trades):+.2f}R")
    print(f"Прибуток/збиток ({LOT_SIZE} lot): ${total_dollars:+.2f}")


def main() -> None:
    configure_console_encoding()
    m15 = load_candles(XAUUSD_M15_30D_FILE, "15m")
    h1 = load_candles(XAUUSD_M15_30D_FILE, "1h")

    engine = LocalDataBacktestEngine(
        BacktestConfig(
            symbol=SYMBOL,
            timeframe="15m",
            total_candles=len(m15),
            warmup_candles=WARMUP,
        ),
        m15_candles=m15,
        h1_candles=h1,
    )
    setups, _ = engine._scan_candles(m15)
    stats = engine._simulate_setups(setups, m15, TradeManagementMode.PARTIAL)
    print_report(stats.trades)


if __name__ == "__main__":
    main()
