import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from config.symbols import (
    SUPPORTED_SYMBOLS,
    SUPPORTED_TIMEFRAMES,
    load_config_file,
    resolve_symbol,
    resolve_symbols,
    resolve_timeframe,
    save_default_config_file,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    """Application configuration."""

    project_name: str = "smc-ai-trading-agent"
    symbol: str = "BTCUSDT"
    symbols: tuple[str, ...] = ("BTCUSDT", "XAUUSD", "DJ30", "NAS100")
    timeframe: str = "15m"
    candle_limit: int = 500
    london_ny_session_symbols: frozenset[str] = frozenset()
    session_confidence_symbols: frozenset[str] = frozenset()
    news_enabled: bool = True
    news_buffer_minutes: int = 15
    finnhub_api_key: str = ""
    news_calendar_url: str = ""
    runtime_mode: str = "continuous"
    poll_interval_seconds: float = 60.0
    scan_interval_seconds: float = 900.0
    signal_cooldown_minutes: int = 60
    duplicate_entry_tolerance_pct: float = 0.001
    log_dir: Path = LOG_DIR
    log_level: str = "INFO"
    telegram_token: str = ""
    telegram_chat_id: str = ""
    telegram_bot_token: str = ""
    oanda_api_key: str = ""
    oanda_account_id: str = ""
    oanda_env: str = "practice"


def _parse_symbol_list(raw_value: str) -> tuple[str, ...]:
    parts = [part.strip().upper() for part in raw_value.split(",") if part.strip()]
    return resolve_symbols(parts)


def load_settings() -> Settings:
    """Load settings from config.json and environment variables."""
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    save_default_config_file()
    config = load_config_file()

    symbols = resolve_symbols(config["symbols"])
    if os.getenv("TRADING_SYMBOLS"):
        symbols = _parse_symbol_list(os.getenv("TRADING_SYMBOLS", ""))
    elif os.getenv("TRADING_SYMBOL"):
        symbols = (resolve_symbol(os.getenv("TRADING_SYMBOL", config["symbol"]).upper()).display,)

    symbol = symbols[0]
    timeframe = os.getenv("TRADING_TIMEFRAME", config["timeframe"])
    resolve_timeframe(timeframe)
    candle_limit = int(os.getenv("TRADING_CANDLE_LIMIT", config["candle_limit"]))
    news_config = config.get("news", {})
    runtime_config = config.get("runtime", {})

    return Settings(
        symbol=symbol,
        symbols=symbols,
        timeframe=timeframe,
        candle_limit=candle_limit,
        london_ny_session_symbols=frozenset(config.get("london_ny_session_symbols", [])),
        session_confidence_symbols=frozenset(config.get("session_confidence_symbols", [])),
        news_enabled=bool(news_config.get("enabled", True)),
        news_buffer_minutes=int(news_config.get("buffer_minutes", 15)),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY", news_config.get("finnhub_api_key", "")),
        news_calendar_url=os.getenv(
            "NEWS_CALENDAR_URL",
            news_config.get("calendar_url", ""),
        ),
        runtime_mode=str(runtime_config.get("mode", "continuous")),
        poll_interval_seconds=float(
            os.getenv("POLL_INTERVAL_SECONDS", runtime_config.get("poll_interval_seconds", 60))
        ),
        scan_interval_seconds=float(
            os.getenv("SCAN_INTERVAL_SECONDS", runtime_config.get("scan_interval_seconds", 900))
        ),
        signal_cooldown_minutes=int(runtime_config.get("signal_cooldown_minutes", 60)),
        duplicate_entry_tolerance_pct=float(
            runtime_config.get("duplicate_entry_tolerance_pct", 0.001)
        ),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        telegram_token=os.getenv("TELEGRAM_TOKEN", ""),
        telegram_chat_id=os.getenv("TELEGRAM_CHAT_ID", ""),
        telegram_bot_token=os.getenv("TELEGRAM_BOT_TOKEN", ""),
        oanda_api_key=os.getenv("OANDA_API_KEY", ""),
        oanda_account_id=os.getenv("OANDA_ACCOUNT_ID", ""),
        oanda_env=os.getenv("OANDA_ENV", "practice"),
    )


__all__ = ["Settings", "load_settings", "SUPPORTED_SYMBOLS", "SUPPORTED_TIMEFRAMES"]
