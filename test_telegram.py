import argparse
import sys

from tracking.console import configure_console_encoding, safe_print
from telegram.telegram_bot import (
    CONNECTION_TEST_MESSAGE,
    TelegramBot,
    TelegramChatInfo,
    TelegramError,
    load_telegram_credentials,
    load_telegram_token,
)


def print_chat_id() -> int:
    configure_console_encoding()
    token, chat_id = load_telegram_credentials()

    if not token:
        safe_print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 1

    if chat_id:
        safe_print(f"TELEGRAM_CHAT_ID={chat_id}")
        return 0

    safe_print("TELEGRAM_CHAT_ID is not set in .env")
    safe_print("Send a message to your bot, then run:")
    safe_print("  python test_telegram.py --discover-chat-id")
    return 1


def discover_chat_id() -> int:
    configure_console_encoding()
    token, _ = load_telegram_credentials()

    if not token:
        safe_print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 1

    try:
        chat_ids = TelegramBot.discover_chat_ids(token)
    except TelegramError as exc:
        safe_print("Failed to discover chat IDs.")
        safe_print(exc.format_full())
        return 1

    if not chat_ids:
        safe_print("No chat IDs found.")
        safe_print("Run: python get_chat_id.py")
        return 1

    safe_print("Discovered chat IDs:")
    for chat_id in chat_ids:
        safe_print(f"  TELEGRAM_CHAT_ID={chat_id}")
    return 0


def test_connection() -> int:
    configure_console_encoding()
    token, chat_id = load_telegram_credentials()

    if not token:
        safe_print("TELEGRAM_BOT_TOKEN is not set in .env")
        return 1

    if not chat_id:
        safe_print("TELEGRAM_CHAT_ID is not set in .env")
        safe_print("Run: python get_chat_id.py")
        return 1

    bot = TelegramBot(token=token, chat_id=chat_id)
    safe_print(f"Sending test message to chat {chat_id}...")
    safe_print(f"Message: {CONNECTION_TEST_MESSAGE}")

    try:
        bot.test_connection()
    except TelegramError as exc:
        safe_print("Telegram connection test failed.")
        safe_print(exc.format_full())
        return 1
    except Exception as exc:
        safe_print(f"Unexpected error: {exc}")
        return 1

    safe_print("Telegram connection test succeeded.")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Telegram bot connection utilities")
    parser.add_argument(
        "--print-chat-id",
        action="store_true",
        help="Print TELEGRAM_CHAT_ID from .env",
    )
    parser.add_argument(
        "--discover-chat-id",
        action="store_true",
        help="List chat IDs from recent bot updates",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.print_chat_id:
        sys.exit(print_chat_id())
    if args.discover_chat_id:
        sys.exit(discover_chat_id())
    sys.exit(test_connection())


if __name__ == "__main__":
    main()
