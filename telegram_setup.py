import json
import sys

from dotenv import load_dotenv

from config.settings import PROJECT_ROOT
from tracking.console import configure_console_encoding, safe_print
from telegram.telegram_bot import TelegramBot, TelegramError, load_telegram_token

load_dotenv(PROJECT_ROOT / ".env")

NO_CHANNEL_INSTRUCTIONS = """
No channel was detected in getUpdates.

To discover your channel chat_id:

1. Open your Telegram channel.
2. Channel Settings -> Administrators -> Add Administrator.
3. Add this bot with "Post Messages" permission.
4. Publish a test message in the channel.
5. Run:
     python telegram_setup.py

Then add the printed ID to .env:
  TELEGRAM_CHAT_ID=-1001234567890
"""


def print_json_section(title: str, payload: dict) -> None:
    safe_print(f"=== {title} ===")
    safe_print(json.dumps(payload, indent=2, ensure_ascii=False))
    safe_print()


def print_channel_chat_ids(channels) -> bool:
    safe_print("CHANNEL CHAT ID:")
    if not channels:
        safe_print("not found")
        safe_print()
        safe_print(NO_CHANNEL_INSTRUCTIONS.strip())
        return False

    for channel in channels:
        safe_print(channel.chat_id)
    safe_print()
    safe_print("Add this value to .env as TELEGRAM_CHAT_ID.")
    return True


def main() -> int:
    configure_console_encoding()
    token = load_telegram_token()

    if not token:
        safe_print("TELEGRAM_BOT_TOKEN is not set in .env")
        safe_print("Add your bot token first:")
        safe_print("  TELEGRAM_BOT_TOKEN=123456789:ABCdef...")
        return 1

    try:
        bot_info = TelegramBot.get_bot_info(token)
        updates_payload = TelegramBot.get_updates(token)
        chats = TelegramBot.discover_accessible_chats(token)
    except TelegramError as exc:
        safe_print("Telegram setup discovery failed.")
        safe_print(exc.format_full())
        return 1

    print_json_section("getMe()", {"ok": True, "result": bot_info})
    print_json_section("getUpdates()", updates_payload)

    channels = [chat for chat in chats if chat.chat_type == "channel"]
    if channels:
        safe_print("=== Detected Channels ===")
        for channel in channels:
            label = channel.title or channel.username or "Unknown"
            username = f" (@{channel.username})" if channel.username else ""
            safe_print(f"- {label}{username} => {channel.chat_id}")
        safe_print()

    print_channel_chat_ids(channels)
    return 0 if channels else 1


if __name__ == "__main__":
    sys.exit(main())
