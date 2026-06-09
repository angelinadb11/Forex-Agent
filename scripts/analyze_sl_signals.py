"""Analyze stop-loss trades from trade_history + agent logs + post-entry price."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from data import MarketDataProvider
from strategy.runner import AGENT_WEIGHTS

LOG_FILE = PROJECT_ROOT / "logs" / "smc-ai-trading-agent.log"
HISTORY_FILE = PROJECT_ROOT / "trade_history.json"
LOCAL_OFFSET = timedelta(hours=3)

AGENT_LINE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| .* \| "
    r"(?:(BTCUSDT|XAUUSD|DJ30) \| )?(\w+) \| (\w+) \| confidence=([\d.]+)"
)
FINAL_LINE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| .* \| "
    r"(?:(BTCUSDT|XAUUSD|DJ30) \| )?FINAL \| (\w+) \| confidence=([\d.]+) \| "
    r"long=([\d.]+) short=([\d.]+)"
)
SIGNAL_LINE = re.compile(
    r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+ \| .* \| "
    r"(?:(BTCUSDT|XAUUSD|DJ30) \| )?SIGNAL \| (\w+) \| entry=([\d.]+) sl=([\d.]+) "
    r"tp1=([\d.]+) tp2=([\d.]+) tp3=([\d.]+)"
)


def parse_log_ts(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is not None:
        return parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def extract_signal_block(symbol: str, open_time_utc: str, log_lines: list[str]) -> dict | None:
    target = parse_utc(open_time_utc)
    local_target = target + LOCAL_OFFSET

    for index, line in enumerate(log_lines):
        match = FINAL_LINE.search(line)
        if match is None:
            continue
        symbol_in_line = match.group(2) or symbol
        if symbol_in_line != symbol:
            continue
        timestamp = parse_log_ts(match.group(1))
        if abs((timestamp - local_target).total_seconds()) > 180:
            continue

        agents: dict[str, dict[str, float | str]] = {}
        for previous in log_lines[max(0, index - 15) : index + 1]:
            agent_match = AGENT_LINE.search(previous)
            if agent_match is None:
                continue
            line_symbol = agent_match.group(2) or symbol
            if line_symbol != symbol:
                continue
            name = agent_match.group(3)
            if name in {"FINAL", "SIGNAL"}:
                continue
            agents[name] = {
                "direction": agent_match.group(4),
                "confidence": float(agent_match.group(5)),
            }

        signal_data = None
        for candidate in log_lines[index : index + 4]:
            signal_match = SIGNAL_LINE.search(candidate)
            if signal_match and (signal_match.group(2) or symbol) == symbol:
                signal_data = {
                    "direction": signal_match.group(3),
                    "entry": float(signal_match.group(4)),
                    "sl": float(signal_match.group(5)),
                    "tp1": float(signal_match.group(6)),
                    "tp2": float(signal_match.group(7)),
                    "tp3": float(signal_match.group(8)),
                }
                break

        direction = match.group(3)
        weighted: list[tuple[float, str, dict]] = []
        for name, payload in agents.items():
            weight = AGENT_WEIGHTS.get(name, 0.0)
            if payload["direction"] != direction:
                continue
            weighted.append((payload["confidence"] * weight, name, payload))
        weighted.sort(reverse=True)

        return {
            "log_time_local": match.group(1),
            "direction": direction,
            "final_confidence": float(match.group(4)),
            "long_score": float(match.group(5)),
            "short_score": float(match.group(6)),
            "agents": agents,
            "top_agent": weighted[0] if weighted else None,
            "signal": signal_data,
        }
    return None


def describe_price_path(
    symbol: str,
    direction: str,
    entry: float,
    stop_loss: float,
    tp1: float,
    open_time_utc: str,
    close_time_utc: str,
) -> str:
    provider = MarketDataProvider()
    timeframe = "15m"
    candles = provider.get_historical_market_data(symbol, timeframe, 200)

    open_ms = parse_utc(open_time_utc).timestamp() * 1000
    close_ms = parse_utc(close_time_utc).timestamp() * 1000

    path_candles = [
        candle
        for candle in candles
        if open_ms <= candle.get("open_time", 0) <= close_ms + 15 * 60_000
    ]
    if not path_candles:
        return "Немає M15 свічок для періоду угоди."

    highs = [float(candle["high"]) for candle in path_candles]
    lows = [float(candle["low"]) for candle in path_candles]
    closes = [float(candle["close"]) for candle in path_candles]
    max_high = max(highs)
    min_low = min(lows)
    last_close = closes[-1]

    if direction == "long":
        mfe = max_high - entry
        mae = entry - min_low
        tp1_reached = max_high >= tp1
        sl_hit = min_low <= stop_loss
        summary = (
            f"Після входу: max {max_high:.2f} (+{mfe:.2f}), min {min_low:.2f} (-{mae:.2f}), "
            f"close періоду {last_close:.2f}. "
        )
        if tp1_reached:
            summary += f"TP1 {tp1:.2f} торкнулися (max high). "
        else:
            summary += f"TP1 {tp1:.2f} не досягнуто (max {max_high:.2f}). "
        if sl_hit:
            summary += f"SL {stop_loss:.2f} пробито (min low {min_low:.2f})."
        return summary

    mfe = entry - min_low
    mae = max_high - entry
    tp1_reached = min_low <= tp1
    sl_hit = max_high >= stop_loss
    summary = (
        f"Після входу: min {min_low:.2f} (+{mfe:.2f} вниз), max {max_high:.2f} (-{mae:.2f} проти), "
        f"close періоду {last_close:.2f}. "
    )
    if tp1_reached:
        summary += f"TP1 {tp1:.2f} торкнулися (min low). "
    else:
        summary += f"TP1 {tp1:.2f} не досягнуто (min {min_low:.2f}). "
    if sl_hit:
        summary += f"SL {stop_loss:.2f} пробито (max high {max_high:.2f})."
    return summary


def main() -> None:
    log_lines = LOG_FILE.read_text(encoding="utf-8").splitlines()
    history = json.loads(HISTORY_FILE.read_text(encoding="utf-8"))

    today = datetime.now().date()
    sl_trades = [
        trade
        for trade in history
        if trade.get("result") == "stop_loss"
    ]

    print(f"=== Stop-loss signals (trade_history + logs) ===")
    print(f"Today: {today.isoformat()}")
    print()

    if not sl_trades:
        print("Stop-loss угод не знайдено.")
        return

    for trade in sl_trades:
        symbol = trade["symbol"]
        direction = trade["direction"]
        block = extract_signal_block(symbol, trade["open_time"], log_lines)

        print("=" * 72)
        print(
            f"{symbol} {direction.upper()} | "
            f"open {trade['open_time']} | close {trade['close_time']}"
        )
        print(
            f"Entry {trade['entry']:.2f} | SL {trade['stop_loss']:.2f} | "
            f"TP1 {trade['tp1']:.2f} | confidence {trade['confidence']:.0%}"
        )
        print(f"Reason: {trade['reason'][:160]}...")

        if block is None:
            print("\nAgent log block: не знайдено в smc-ai-trading-agent.log")
        else:
            print(f"\nAgent log ({block['log_time_local']} local):")
            print(
                f"FINAL {block['direction'].upper()} | confidence {block['final_confidence']:.2f} | "
                f"long={block['long_score']:.2f} short={block['short_score']:.2f}"
            )
            print("\nГолоси агентів:")
            for name in sorted(block["agents"]):
                payload = block["agents"][name]
                weight = AGENT_WEIGHTS.get(name)
                weight_label = f" (weight {weight:.3f})" if weight is not None else ""
                print(
                    f"  {name:12} {payload['direction']:7} "
                    f"conf={payload['confidence']:.2f}{weight_label}"
                )
            if block["top_agent"]:
                contrib, name, payload = block["top_agent"]
                print(
                    f"\nНайвищий weighted score: {name} "
                    f"(conf {payload['confidence']:.2f}, внесок {contrib:.3f})"
                )

        try:
            price_path = describe_price_path(
                symbol,
                direction,
                trade["entry"],
                trade["stop_loss"],
                trade["tp1"],
                trade["open_time"],
                trade["close_time"],
            )
            print(f"\nPrice action: {price_path}")
        except Exception as exc:
            print(f"\nPrice action: не вдалося завантажити ({exc})")
        print()


if __name__ == "__main__":
    main()
