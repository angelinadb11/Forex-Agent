#!/usr/bin/env python3
"""MT5 price bridge for Moneta Markets (or any MT5 broker).

Run on Windows with MetaTrader 5 installed, logged into your broker account.
The VPS bot reads live XAUUSD candles/prices from this HTTP service.

Usage (Windows PowerShell):
  pip install MetaTrader5
  python tools/mt5_bridge.py --symbol XAUUSD --host 0.0.0.0 --port 8765

Secure remote access: Tailscale VPN or SSH tunnel; optional --token for auth.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

try:
    import MetaTrader5 as mt5
except ImportError:
    print("Install MetaTrader5: pip install MetaTrader5", file=sys.stderr)
    raise SystemExit(1)

TIMEFRAME_MAP = {
    "1m": mt5.TIMEFRAME_M1,
    "5m": mt5.TIMEFRAME_M5,
    "15m": mt5.TIMEFRAME_M15,
    "1h": mt5.TIMEFRAME_H1,
    "4h": mt5.TIMEFRAME_H4,
}

BRIDGE_SYMBOL = "XAUUSD"
BRIDGE_TOKEN = ""
MT5_INITIALIZED = False


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name, "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def initialize_mt5(
    *,
    terminal_path: str = "",
    login: int = 0,
    password: str = "",
    server: str = "",
) -> None:
    global MT5_INITIALIZED

    kwargs: dict[str, Any] = {}
    if terminal_path.strip():
        kwargs["path"] = terminal_path.strip()
    if login:
        kwargs["login"] = login
    if password.strip():
        kwargs["password"] = password.strip()
    if server.strip():
        kwargs["server"] = server.strip()

    if not mt5.initialize(**kwargs):
        code, message = mt5.last_error()
        raise RuntimeError(f"MT5 initialize failed: {code} {message}")

    info = mt5.terminal_info()
    if info is None:
        raise RuntimeError("MT5 terminal_info() returned None — is the terminal running?")

    MT5_INITIALIZED = True


def ensure_symbol(symbol: str) -> None:
    info = mt5.symbol_info(symbol)
    if info is None:
        raise ValueError(f"Symbol '{symbol}' not found in MT5 Market Watch")
    if not info.visible:
        if not mt5.symbol_select(symbol, True):
            code, message = mt5.last_error()
            raise ValueError(f"Cannot select symbol '{symbol}': {code} {message}")


def fetch_candles(symbol: str, timeframe: str, limit: int, *, historical: bool) -> list[dict[str, Any]]:
    ensure_symbol(symbol)
    if timeframe not in TIMEFRAME_MAP:
        supported = ", ".join(TIMEFRAME_MAP)
        raise ValueError(f"Unsupported timeframe '{timeframe}'. Use one of: {supported}")

    mt5_tf = TIMEFRAME_MAP[timeframe]
    count = max(1, min(limit, 1000))
    if historical:
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)
    else:
        rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, count)

    if rates is None or len(rates) == 0:
        code, message = mt5.last_error()
        raise ValueError(f"No MT5 candles for {symbol} {timeframe}: {code} {message}")

    candles: list[dict[str, Any]] = []
    for row in rates:
        candles.append(
            {
                "open_time": float(int(row["time"]) * 1000),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["tick_volume"]),
            }
        )
    return candles


def fetch_price(symbol: str) -> float:
    ensure_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if tick is None:
        code, message = mt5.last_error()
        raise ValueError(f"No MT5 tick for {symbol}: {code} {message}")
    bid = float(tick.bid)
    ask = float(tick.ask)
    if bid <= 0 or ask <= 0:
        raise ValueError(f"Invalid MT5 tick for {symbol}: bid={bid} ask={ask}")
    return (bid + ask) / 2


def json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


class Mt5BridgeHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))

    def _authorized(self) -> bool:
        if not BRIDGE_TOKEN:
            return True
        auth = self.headers.get("Authorization", "")
        return auth == f"Bearer {BRIDGE_TOKEN}"

    def _reject_unauthorized(self) -> None:
        json_response(self, 401, {"ok": False, "error": "Unauthorized"})

    def do_GET(self) -> None:
        if not self._authorized():
            self._reject_unauthorized()
            return

        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)
        path = parsed.path.rstrip("/") or "/"

        try:
            if path == "/health":
                terminal = mt5.terminal_info()
                account = mt5.account_info()
                json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "symbol": BRIDGE_SYMBOL,
                        "terminal": getattr(terminal, "name", "") if terminal else "",
                        "company": getattr(terminal, "company", "") if terminal else "",
                        "account": getattr(account, "login", 0) if account else 0,
                        "server": getattr(account, "server", "") if account else "",
                    },
                )
                return

            if path == "/price":
                symbol = params.get("symbol", [BRIDGE_SYMBOL])[0]
                price = fetch_price(symbol)
                json_response(self, 200, {"ok": True, "symbol": symbol, "price": price})
                return

            if path == "/candles":
                symbol = params.get("symbol", [BRIDGE_SYMBOL])[0]
                timeframe = params.get("timeframe", ["15m"])[0]
                limit = int(params.get("limit", ["500"])[0])
                historical = params.get("historical", ["0"])[0].strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                candles = fetch_candles(symbol, timeframe, limit, historical=historical)
                json_response(
                    self,
                    200,
                    {
                        "ok": True,
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "candles": candles,
                    },
                )
                return

            json_response(self, 404, {"ok": False, "error": f"Unknown path: {path}"})
        except ValueError as exc:
            json_response(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            json_response(self, 500, {"ok": False, "error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MT5 HTTP bridge for Forex Agent")
    parser.add_argument("--host", default=os.getenv("MT5_BRIDGE_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MT5_BRIDGE_PORT", "8765")))
    parser.add_argument("--symbol", default=os.getenv("MT5_SYMBOL", "XAUUSD"))
    parser.add_argument("--token", default=os.getenv("MT5_BRIDGE_TOKEN", ""))
    parser.add_argument("--terminal-path", default=os.getenv("MT5_TERMINAL_PATH", ""))
    parser.add_argument("--login", type=int, default=int(os.getenv("MT5_LOGIN", "0") or "0"))
    parser.add_argument("--password", default=os.getenv("MT5_PASSWORD", ""))
    parser.add_argument("--server", default=os.getenv("MT5_SERVER", ""))
    return parser.parse_args()


def main() -> None:
    global BRIDGE_SYMBOL, BRIDGE_TOKEN

    if sys.platform != "win32":
        print("MT5 bridge must run on Windows with MetaTrader 5 installed.", file=sys.stderr)
        raise SystemExit(1)

    args = parse_args()
    BRIDGE_SYMBOL = args.symbol.strip() or "XAUUSD"
    BRIDGE_TOKEN = args.token.strip()

    initialize_mt5(
        terminal_path=args.terminal_path,
        login=args.login,
        password=args.password,
        server=args.server,
    )
    ensure_symbol(BRIDGE_SYMBOL)

    server = ThreadingHTTPServer((args.host, args.port), Mt5BridgeHandler)
    print(
        f"MT5 bridge listening on http://{args.host}:{args.port} | symbol={BRIDGE_SYMBOL}",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down MT5 bridge...")
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    main()
