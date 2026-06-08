import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from config.settings import Settings


def setup_logging(settings: Settings) -> logging.Logger:
    """Configure console and file logging under logs/."""
    settings.log_dir.mkdir(parents=True, exist_ok=True)

    log_file = settings.log_dir / f"{settings.project_name}.log"
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logging.basicConfig(level=level, handlers=[file_handler, console_handler])

    logger = logging.getLogger(settings.project_name)
    logger.info("Logging initialized: %s", log_file)
    return logger
