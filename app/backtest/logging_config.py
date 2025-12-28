"""Central logging configuration for backtests."""

from __future__ import annotations

import logging
import logging.config
from typing import Any


def configure_logging(*, level: int = logging.INFO) -> None:
    """Configure application-wide logging (console)."""
    config: dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": level,
                "formatter": "standard",
            }
        },
        "root": {"handlers": ["console"], "level": level},
    }
    logging.config.dictConfig(config)
