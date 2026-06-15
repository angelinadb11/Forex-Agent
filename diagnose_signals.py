"""Quick runtime diagnostic: why no Telegram signals? Run on VPS: py diagnose_signals.py"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

from data import MarketDataProvider
from strategy.liquidity_scalp import analyze_liquidity_scalp_symbol, build_liquidity_scalp_gate
from strategy.sweep_fvg_scalp import analyze_sweep_fvg_scalp_symbol, build_premium_scalp_gate
from strategy.turtle_soup_scalp import analyze_turtle_soup_scalp_symbol, build_turtle_soup_gate
from telegram.telegram_bot import load_scalp_telegram_credentials, load_telegram_credentials


def _check_env() -> None:
    print("=== ENV ===")
    main_token, main_chat = load_telegram_credentials()
    scalp_token, scalp_chat = load_scalp_telegram_credentials()
    print(f"Main Telegram:     {'OK' if main_token and main_chat else 'MISSING'}")
    print(f"Scalp Telegram:    {'OK' if scalp_token and scalp_chat else 'MISSING'}")
    if scalp_chat:
        print(f"  scalp chat_id:   {scalp_chat}")
    print(f"LOG_LEVEL:         {os.getenv('LOG_LEVEL', 'INFO')}")
    print()


def _check_active_trades() -> None:
    print("=== ACTIVE TRADES ===")
    for name in ("active_trades.json", "scalp_active_trades.json"):
        path = PROJECT_ROOT / name
        if not path.exists():
            print(f"{name}: not found (no open trades persisted)")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"{name}: INVALID JSON — {exc}")
            continue
        open_trades = [t for t in data if not t.get("closed")]
        print(f"{name}: {len(open_trades)} open / {len(data)} total")
        for t in open_trades:
            print(
                f"  - {t.get('symbol')} {t.get('direction')} "
                f"tf={t.get('timeframe')} since {t.get('open_time')}"
            )
    print()


def _check_log_tail() -> None:
    print("=== LOG (last 15 relevant lines) ===")
    log_path = PROJECT_ROOT / "logs" / "smc-ai-trading-agent.log"
    if not log_path.exists():
        print("Log file not found.")
        print()
        return
    keywords = (
        "Continuous mode",
        "published",
        "Scalp skipped",
        "VIP",
        "ERROR",
        "Exception",
        "Traceback",
        "disabled",
        "Trade closed",
    )
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [line for line in lines if any(k in line for k in keywords)]
    for line in hits[-15:]:
        print(line)
    if not hits:
        print("(no matching lines — bot may never have started or log is empty)")
    print()


def _check_live_analysis() -> None:
    print("=== LIVE ANALYSIS (XAUUSD, now) ===")
    provider = MarketDataProvider()
    now = datetime.now(timezone.utc)
    print(f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
    for label, fn, gate in (
        ("M1 Scalp", analyze_liquidity_scalp_symbol, build_liquidity_scalp_gate()),
        ("VIP1", analyze_sweep_fvg_scalp_symbol, build_premium_scalp_gate()),
        ("VIP2", analyze_turtle_soup_scalp_symbol, build_turtle_soup_gate()),
    ):
        try:
            sig, _ctx, res = fn("XAUUSD", provider=provider, publish_gate=gate)
            status = "SIGNAL READY" if sig else res.message
            print(f"  {label}: {status}")
        except Exception as exc:
            print(f"  {label}: ERROR — {exc}")
    print()


def main() -> None:
    print(f"Forex Agent diagnostic | {PROJECT_ROOT}\n")
    _check_env()
    _check_active_trades()
    _check_log_tail()
    _check_live_analysis()
    print("=== EXPECTED FREQUENCY (prod filters, ~3 days backtest) ===")
    print("M1 scalp: ~2/day | VIP1: ~0.2/day | VIP2: ~1/day")
    print("2 days silence = check screen -ls and scalp_active_trades.json first.")


if __name__ == "__main__":
    main()
