from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import requests
from dotenv import load_dotenv

from agents.base import AgentResult, Direction
from config.settings import PROJECT_ROOT
from signal_generator import TradeSignal
from telegram.message_format import (
    format_agent_result,
    format_agent_summary,
    format_trade_signal as build_trade_signal_message,
    format_trade_update as build_trade_update_message,
    format_trade_update_warning as build_trade_update_warning_message,
    format_high_risk_update as build_high_risk_update_message,
)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
CONNECTION_TEST_MESSAGE = "✅ TradingBoss Bot Connected"

UPDATE_CHAT_KEYS = (
    "message",
    "edited_message",
    "channel_post",
    "edited_channel_post",
    "my_chat_member",
    "chat_member",
)


@dataclass(frozen=True)
class TelegramChatInfo:
    chat_id: str
    chat_type: str
    title: str
    username: str | None

    def format_line(self) -> str:
        label = self.title or self.username or "Unknown"
        username = f" (@{self.username})" if self.username else ""
        return (
            f"- [{self.chat_type}] {label}{username} "
            f"=> TELEGRAM_CHAT_ID={self.chat_id}"
        )


class TelegramError(Exception):
    """Raised when Telegram API delivery fails."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body

    def format_full(self) -> str:
        lines = [str(self)]
        if self.status_code is not None:
            lines.append(f"HTTP status: {self.status_code}")
        if self.response_body:
            lines.append(f"Response body: {self.response_body}")
        return "\n".join(lines)


def load_telegram_token() -> str:
    """Load TELEGRAM_BOT_TOKEN from .env and environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")
    return os.getenv("TELEGRAM_BOT_TOKEN", "").strip()


def load_telegram_credentials() -> tuple[str, str]:
    """Load Telegram credentials from .env and environment variables."""
    load_dotenv(PROJECT_ROOT / ".env")

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()
    return token, chat_id


def _telegram_get(token: str, endpoint: str, *, params: dict | None = None) -> dict:
    url = f"https://api.telegram.org/bot{token}/{endpoint}"
    try:
        response = requests.get(url, params=params or {}, timeout=30)
    except requests.RequestException as exc:
        raise TelegramError(f"Telegram {endpoint} request failed: {exc}") from exc

    response_body = response.text
    try:
        payload = response.json()
    except ValueError as exc:
        raise TelegramError(
            f"Telegram {endpoint} returned invalid JSON: {response_body}",
            status_code=response.status_code,
            response_body=response_body,
        ) from exc

    if not response.ok or not payload.get("ok"):
        raise TelegramError(
            f"Telegram {endpoint} error: {payload or response_body}",
            status_code=response.status_code,
            response_body=response_body,
        )
    return payload


def _chat_label(chat: dict) -> str:
    for key in ("title", "first_name", "last_name", "username"):
        value = chat.get(key)
        if value:
            return str(value)
    return "Unknown"


def _chat_from_payload(item: dict) -> TelegramChatInfo | None:
    chat = item.get("chat")
    if not chat:
        return None

    chat_id = chat.get("id")
    chat_type = chat.get("type")
    if chat_id is None or not chat_type:
        return None

    username = chat.get("username")
    return TelegramChatInfo(
        chat_id=str(chat_id),
        chat_type=str(chat_type),
        title=_chat_label(chat),
        username=str(username) if username else None,
    )


class TelegramBot:
    """Telegram notification layer for trade signals."""

    def __init__(self, token: str, chat_id: str) -> None:
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not configured")
        if not chat_id:
            raise ValueError("TELEGRAM_CHAT_ID is not configured")

        self.token = token
        self.chat_id = chat_id

    @classmethod
    def from_env(cls) -> TelegramBot | None:
        """Create a bot from .env credentials, or None if not configured."""
        token, chat_id = load_telegram_credentials()
        if not token or not chat_id:
            return None
        return cls(token=token, chat_id=chat_id)

    def send_message(self, text: str) -> None:
        """Send a plain-text message to the configured chat."""
        url = TELEGRAM_API_URL.format(token=self.token)
        try:
            response = requests.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise TelegramError(f"Telegram request failed: {exc}") from exc

        response_body = response.text
        try:
            payload = response.json()
        except ValueError:
            payload = None

        if not response.ok:
            raise TelegramError(
                f"Telegram HTTP {response.status_code}: {response_body}",
                status_code=response.status_code,
                response_body=response_body,
            )

        if not payload or not payload.get("ok"):
            raise TelegramError(
                f"Telegram API error: {payload or response_body}",
                status_code=response.status_code,
                response_body=response_body,
            )

    def test_connection(self) -> None:
        """Send a connection test message to the configured chat."""
        self.send_message(CONNECTION_TEST_MESSAGE)

    @staticmethod
    def get_bot_info(token: str) -> dict:
        """Return bot metadata from Telegram getMe."""
        payload = _telegram_get(token, "getMe")
        return payload["result"]

    @staticmethod
    def get_updates(token: str, *, limit: int = 100) -> dict:
        """Return the full Telegram getUpdates API payload."""
        return _telegram_get(token, "getUpdates", params={"limit": limit})

    @staticmethod
    def discover_accessible_chats(token: str) -> list[TelegramChatInfo]:
        """Return unique chats visible in recent bot updates."""
        payload = TelegramBot.get_updates(token)

        chats: dict[str, TelegramChatInfo] = {}
        for update in payload.get("result", []):
            for key in UPDATE_CHAT_KEYS:
                item = update.get(key)
                if not item:
                    continue

                chat_info = _chat_from_payload(item)
                if chat_info is not None:
                    chats[chat_info.chat_id] = chat_info
                    continue

                nested_chat = item.get("chat")
                if nested_chat:
                    chat_info = _chat_from_payload({"chat": nested_chat})
                    if chat_info is not None:
                        chats[chat_info.chat_id] = chat_info

        return sorted(chats.values(), key=lambda chat: (chat.chat_type, chat.title))

    @staticmethod
    def discover_chat_ids(token: str) -> list[str]:
        """Return chat IDs seen in recent bot updates (for setup help)."""
        return [chat.chat_id for chat in TelegramBot.discover_accessible_chats(token)]

    @staticmethod
    def format_trade_signal(
        symbol: str,
        signal: TradeSignal,
        timeframe: str,
        results: dict[str, AgentResult] | None = None,
        news_warning: str | None = None,
    ) -> str:
        return build_trade_signal_message(symbol, signal, timeframe, results, news_warning)

    def send_trade_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        timeframe: str,
        agent_results: dict[str, AgentResult] | None = None,
        news_warning: str | None = None,
    ) -> None:
        """Format and send a generated trade signal."""
        if signal.confidence < 0.70:
            raise ValueError("Signal confidence below 70% minimum")

        message = self.format_trade_signal(
            symbol,
            signal,
            timeframe,
            agent_results,
            news_warning,
        )
        self.send_message(message)

    def send_no_trade(self, symbol: str, confidence: float, reason: str = "") -> None:
        """Notify that no trade signal was sent."""
        confidence_pct = min(100, int(round(confidence * 100)))
        lines = [
            "🚫 NO TRADE",
            "",
            symbol,
            f"Confidence: {confidence_pct}%",
        ]
        if reason:
            lines.append(reason)
        self.send_message("\n".join(lines))

    def send_trade_update(self, symbol: str, direction: Direction, event: str) -> None:
        """Send a trade monitor update for TP/SL events."""
        self.send_message(build_trade_update_message(symbol, direction, event))

    def send_trade_update_warning(
        self,
        symbol: str,
        direction: Direction,
        reasons: list[str],
    ) -> None:
        """Send a Level 1 informational warning for an active trade."""
        self.send_message(
            build_trade_update_warning_message(symbol, direction, reasons)
        )

    def send_high_risk_update(
        self,
        symbol: str,
        direction: Direction,
        reasons: list[str],
    ) -> None:
        """Send a Level 2 high-risk informational warning for an active trade."""
        self.send_message(
            build_high_risk_update_message(symbol, direction, reasons)
        )

    def send_agent_result(self, agent_name: str, result: AgentResult) -> None:
        """Format and send a single agent signal."""
        self.send_message(format_agent_result(agent_name, result))

    def send_summary(self, results: dict[str, AgentResult]) -> None:
        """Send a summary of all agent signals."""
        self.send_message(format_agent_summary(results))
