import sys

from tracking.console import configure_console_encoding, safe_print
from telegram.telegram_bot import TelegramBot, TelegramError, load_telegram_token

NO_CHANNEL_INSTRUCTIONS = """
No Telegram channel was found in recent bot updates.

To discover your channel TELEGRAM_CHAT_ID:

1. Open Telegram and create a channel (or open your existing signal channel).
2. Open Channel Settings -> Administrators -> Add Administrator.
3. Add your bot and grant at least "Post Messages" permission.
4. Publish any message in the channel (for example: "test").
5. Run this script again:
     python get_chat_id.py

Notes:
- Channel IDs usually look like -1001234567890.
- Private chats and groups also appear here if you have messaged the bot directly.
- This script only reads bot metadata and updates. It does not send trading signals.
- After you find the channel ID, add it to .env:
     TELEGRAM_CHAT_ID=-1001234567890
"""


def print_bot_information(bot_info: dict) -> None:
    safe_print("=== Bot Information ===")
    safe_print(f"Bot ID: {bot_info.get('id')}")
    safe_print(f"Username: @{bot_info.get('username')}")
    safe_print(f"Name: {bot_info.get('first_name')}")
    safe_print(f"Can join groups: {bot_info.get('can_join_groups')}")
    safe_print(f"Can read all group messages: {bot_info.get('can_read_all_group_messages')}")
    safe_print(f"Supports inline queries: {bot_info.get('supports_inline_queries')}")
    safe_print()


def print_accessible_chats(chats) -> None:
    safe_print("=== Accessible Chats ===")
    if not chats:
        safe_print("No chats found in recent updates.")
        safe_print()
        return

    for chat in chats:
        safe_print(chat.format_line())
    safe_print()


def print_channel_chat_ids(chats) -> bool:
    channels = [chat for chat in chats if chat.chat_type == "channel"]
    safe_print("=== Channel TELEGRAM_CHAT_ID ===")

    if not channels:
        safe_print("No channel chat_id found.")
        safe_print(NO_CHANNEL_INSTRUCTIONS.strip())
        return False

    for channel in channels:
        safe_print(f"Channel: {channel.title}")
        if channel.username:
            safe_print(f"Username: @{channel.username}")
        safe_print(f"TELEGRAM_CHAT_ID={channel.chat_id}")
        safe_print()

    safe_print("Add the channel ID above to your .env file.")
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
        chats = TelegramBot.discover_accessible_chats(token)
    except TelegramError as exc:
        safe_print("Telegram discovery failed.")
        safe_print(exc.format_full())
        return 1

    print_bot_information(bot_info)
    print_accessible_chats(chats)
    found_channel = print_channel_chat_ids(chats)
    return 0 if found_channel or chats else 1


if __name__ == "__main__":
    sys.exit(main())
