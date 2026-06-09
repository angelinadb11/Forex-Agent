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
    format_scalp_trade_signal as build_scalp_trade_signal_message,
    format_trade_signal as build_trade_signal_message,
    format_trade_update_warning as build_trade_update_warning_message,
    format_high_risk_update as build_high_risk_update_message,
)

TELEGRAM_API_URL = "https://api.telegram.org/bot{token}/sendMessage"
CONNECTION_TEST_MESSAGE = "✅ TradingBoss бот підключено"

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

    def send_message(
        self,
        text: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        """Send a plain-text message to the configured chat."""
        url = TELEGRAM_API_URL.format(token=self.token)
        payload: dict = {
            "chat_id": self.chat_id,
            "text": text,
        }
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id

        try:
            response = requests.post(
                url,
                json=payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise TelegramError(f"Telegram request failed: {exc}") from exc

        response_body = response.text
        try:
            body = response.json()
        except ValueError:
            body = None

        if not response.ok:
            raise TelegramError(
                f"Telegram HTTP {response.status_code}: {response_body}",
                status_code=response.status_code,
                response_body=response_body,
            )

        if not body or not body.get("ok"):
            raise TelegramError(
                f"Telegram API error: {body or response_body}",
                status_code=response.status_code,
                response_body=response_body,
            )

        return int(body["result"]["message_id"])

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
        off_hours_warning: str | None = None,
        h4_mismatch_warning: str | None = None,
    ) -> str:
        return build_trade_signal_message(
            symbol,
            signal,
            timeframe,
            results,
            news_warning,
            off_hours_warning,
            h4_mismatch_warning,
        )

    def send_trade_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        timeframe: str,
        agent_results: dict[str, AgentResult] | None = None,
        news_warning: str | None = None,
        off_hours_warning: str | None = None,
        h4_mismatch_warning: str | None = None,
    ) -> int:
        """Format and send a generated trade signal."""
        if signal.confidence < 0.70:
            raise ValueError("Signal confidence below 70% minimum")

        message = self.format_trade_signal(
            symbol,
            signal,
            timeframe,
            agent_results,
            news_warning,
            off_hours_warning,
            h4_mismatch_warning,
        )
        return self.send_message(message)

    def send_scalp_trade_signal(
        self,
        symbol: str,
        signal: TradeSignal,
        *,
        agent_results: dict[str, AgentResult] | None = None,
        news_warning: str | None = None,
    ) -> int:
        """Format and send a scalp trade signal."""
        message = build_scalp_trade_signal_message(
            symbol,
            signal,
            agent_results,
            news_warning,
        )
        return self.send_message(message)

    def send_trade_reply(
        self,
        text: str,
        *,
        reply_to_message_id: int | None,
    ) -> int:
        """Send a reply linked to the original signal message."""
        return self.send_message(text, reply_to_message_id=reply_to_message_id)

    def send_no_trade(self, symbol: str, confidence: float, reason: str = "") -> None:
        """Notify that no trade signal was sent."""
        confidence_pct = min(100, int(round(confidence * 100)))
        lines = [
            "🚫 БЕЗ УГОДИ",
            "",
            symbol,
            f"Впевненість: {confidence_pct}%",
        ]
        if reason:
            lines.append(reason)
        self.send_message("\n".join(lines))

    def send_profit_milestone_reply(
        self,
        message: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        """Send a motivational profit milestone reply linked to the signal."""
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_structure_weakness_warning(
        self,
        message: str,
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        """Send a structure-based position weakness warning as a reply."""
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_trend_change_warning(
        self,
        *,
        reply_to_message_id: int | None,
        open_time: str,
        direction: Direction,
        current_price: float,
        entry: float | None = None,
        reason: str = "Зміна тренду H1 проти позиції",
    ) -> int:
        from telegram.message_format import format_trend_change_warning

        message = format_trend_change_warning(
            open_time=open_time,
            direction=direction,
            current_price=current_price,
            entry=entry,
            reason=reason,
        )
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_near_tp1_breakeven_warning(
        self,
        *,
        reply_to_message_id: int | None,
        open_time: str,
        direction: Direction,
        current_price: float,
        entry: float,
        peak_progress_r: float,
        conditions: tuple[str, ...] = (),
    ) -> int:
        from telegram.message_format import format_near_tp1_breakeven_warning

        message = format_near_tp1_breakeven_warning(
            direction=direction,
            current_price=current_price,
            entry=entry,
            peak_progress_r=peak_progress_r,
            conditions=conditions,
        )
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_sl_proximity_warning(
        self,
        *,
        reply_to_message_id: int | None,
        current_price: float,
        remaining_pips: float,
    ) -> int:
        from telegram.message_format import format_sl_proximity_warning

        message = format_sl_proximity_warning(
            current_price=current_price,
            remaining_pips=remaining_pips,
        )
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_stop_loss_reply(
        self,
        trade,
        *,
        reply_to_message_id: int | None,
    ) -> int:
        from telegram.message_format import format_stop_loss_reply, trade_move_pips, trade_result_dollars

        move_pips = abs(
            trade_move_pips(
                symbol=trade.symbol,
                direction=trade.direction,
                entry=trade.entry,
                price=trade.initial_stop_loss,
            )
        )
        result_dollars = trade_result_dollars(
            symbol=trade.symbol,
            pips=move_pips,
            lot_size=trade.lot_size,
        )
        message = format_stop_loss_reply(result_dollars=result_dollars)
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_breakeven_reply(
        self,
        trade,
        *,
        reply_to_message_id: int | None,
    ) -> int:
        from telegram.message_format import format_breakeven_reply

        message = format_breakeven_reply(
            direction=trade.direction,
            entry=trade.entry,
            exit_price=trade.stop_loss,
        )
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_post_tp_close_reply(
        self,
        trade,
        *,
        reply_to_message_id: int | None,
    ) -> int:
        from telegram.message_format import format_post_tp_close_reply

        message = format_post_tp_close_reply(
            symbol=trade.symbol,
            direction=trade.direction,
            entry=trade.entry,
            exit_price=trade.stop_loss,
            tp1=trade.tp1,
            tp2=trade.tp2,
            tp2_hit=trade.tp2_hit,
        )
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_take_profit_reply(
        self,
        trade,
        *,
        tp_level: int,
        tp_price: float,
        reply_to_message_id: int | None,
    ) -> int:
        from telegram.message_format import format_take_profit_reply, trade_move_pips

        move_pips = trade_move_pips(
            symbol=trade.symbol,
            direction=trade.direction,
            entry=trade.entry,
            price=tp_price,
        )
        message = format_take_profit_reply(
            tp_level=tp_level,
            open_time=trade.open_time,
            direction=trade.direction,
            entry=trade.entry,
            tp_price=tp_price,
            move_pips=move_pips,
            tp1=trade.tp1,
            tp2=trade.tp2,
            tp3=trade.tp3,
        )
        return self.send_trade_reply(message, reply_to_message_id=reply_to_message_id)

    def send_trade_update_warning(
        self,
        symbol: str,
        direction: Direction,
        reasons: list[str],
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        """Send a Level 1 informational warning for an active trade."""
        return self.send_message(
            build_trade_update_warning_message(symbol, direction, reasons),
            reply_to_message_id=reply_to_message_id,
        )

    def send_high_risk_update(
        self,
        symbol: str,
        direction: Direction,
        reasons: list[str],
        *,
        reply_to_message_id: int | None = None,
    ) -> int:
        """Send a Level 2 high-risk informational warning for an active trade."""
        return self.send_message(
            build_high_risk_update_message(symbol, direction, reasons),
            reply_to_message_id=reply_to_message_id,
        )

    def send_agent_result(self, agent_name: str, result: AgentResult) -> None:
        """Format and send a single agent signal."""
        self.send_message(format_agent_result(agent_name, result))

    def send_summary(self, results: dict[str, AgentResult]) -> None:
        """Send a summary of all agent signals."""
        self.send_message(format_agent_summary(results))
