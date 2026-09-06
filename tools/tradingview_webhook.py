#!/usr/bin/env python3
"""TradingView webhook → Telegram (main Trading Boss channel).

Run on VPS behind HTTPS reverse proxy (nginx / Cloudflare Tunnel).

TradingView alert URL example:
  https://YOUR-DOMAIN/tv/webhook?token=YOUR_SECRET

Alert message (JSON recommended):
{
  "secret": "YOUR_SECRET",
  "symbol": "{{ticker}}",
  "action": "SELL",
  "entry": {{close}},
  "sl": 4380.99,
  "tp1": 4374.84,
  "tp2": 4372.38,
  "timeframe": "{{interval}}",
  "note": "TB indicator — sweep"
}
"""

from __future__ import annotations

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse

from dotenv import load_dotenv

from agents.base import Direction
from config.settings import PROJECT_ROOT
from telegram.message_format import format_tradingview_alert
from telegram.telegram_bot import TelegramBot, TelegramError
from webhook.tradingview import parse_tradingview_payload

load_dotenv(PROJECT_ROOT / ".env")

WEBHOOK_SECRET = os.getenv("TRADINGVIEW_WEBHOOK_SECRET", "").strip()
WEBHOOK_HOST = os.getenv("TRADINGVIEW_WEBHOOK_HOST", "127.0.0.1")
WEBHOOK_PORT = int(os.getenv("TRADINGVIEW_WEBHOOK_PORT", "8788"))


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def _authorized(handler: BaseHTTPRequestHandler, body_secret: str | None) -> bool:
    if not WEBHOOK_SECRET:
        return True
    query = parse_qs(urlparse(handler.path).query)
    token = (query.get("token") or query.get("secret") or [""])[0]
    if token == WEBHOOK_SECRET:
        return True
    if body_secret and body_secret == WEBHOOK_SECRET:
        return True
    return False


class TradingViewWebhookHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write("%s - %s\n" % (self.address_string(), format % args))

    def do_POST(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path not in {"/tv/webhook", "/webhook/tradingview", "/"}:
            _json_response(self, 404, {"ok": False, "error": "Not found"})
            return

        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b""
        body = raw.decode("utf-8", errors="replace").strip()

        try:
            alert = parse_tradingview_payload(body)
            if not _authorized(self, alert.secret):
                _json_response(self, 401, {"ok": False, "error": "Unauthorized"})
                return
            if alert.direction == Direction.NEUTRAL:
                _json_response(
                    self,
                    400,
                    {"ok": False, "error": "Direction missing (use BUY/SELL in JSON)"},
                )
                return

            bot = TelegramBot.from_env()
            message = format_tradingview_alert(alert)
            bot.send_message(message)
            _json_response(self, 200, {"ok": True, "symbol": alert.symbol})
        except TelegramError as exc:
            _json_response(self, 502, {"ok": False, "error": str(exc)})
        except (ValueError, json.JSONDecodeError) as exc:
            _json_response(self, 400, {"ok": False, "error": str(exc)})
        except Exception as exc:
            _json_response(self, 500, {"ok": False, "error": str(exc)})

    def do_GET(self) -> None:
        path = urlparse(self.path).path.rstrip("/") or "/"
        if path in {"/health", "/tv/webhook", "/webhook/tradingview", "/"}:
            _json_response(self, 200, {"ok": True, "service": "tradingview-webhook"})
            return
        _json_response(self, 404, {"ok": False, "error": "Not found"})


def main() -> None:
    if not os.getenv("TELEGRAM_BOT_TOKEN", "").strip():
        print("TELEGRAM_BOT_TOKEN is required", file=sys.stderr)
        raise SystemExit(1)
    if not os.getenv("TELEGRAM_CHAT_ID", "").strip():
        print("TELEGRAM_CHAT_ID is required", file=sys.stderr)
        raise SystemExit(1)

    server = ThreadingHTTPServer((WEBHOOK_HOST, WEBHOOK_PORT), TradingViewWebhookHandler)
    print(
        f"TradingView webhook on http://{WEBHOOK_HOST}:{WEBHOOK_PORT}/tv/webhook",
        flush=True,
    )
    if not WEBHOOK_SECRET:
        print("WARNING: TRADINGVIEW_WEBHOOK_SECRET not set — webhook is open", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping TradingView webhook...")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
