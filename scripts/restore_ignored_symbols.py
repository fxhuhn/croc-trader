"""Excluded Symbols Status Validator and Restorer.

Queries yfinance for the list of symbols currently in the 'ignored_symbols' table
of the stocks database (due to prior download failures). If valid price history is successfully returned,
the script automatically removes them from the exclusion table so that they will be included
in future daily screener runs.

Usage:
    python script/restore_ignored_symbols.py

Side Effects:
    Queries Yahoo Finance API. Deletes entries from the 'ignored_symbols' table in data/stocks.db.
"""

import logging
import sys
from pathlib import Path

# 1. Setup path so that 'app' can be imported
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.tasks import run_ignored_symbols_check  # noqa: E402

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RestoreIgnoredSymbols")


def check_and_restore_symbols() -> list[str]:
    """Checks all blacklisted/ignored symbols for active market data on Yahoo Finance.

    If valid data is retrieved, the symbol is removed from ignored_symbols
    so that it will be processed again during the regular daily routine.
    """
    logger.info("--- Starting Ignored Symbols Check & Restore ---")

    # Load configuration
    try:
        stocks_db_path = settings.get_path("stocks")
        logger.info("Database Path: %s", stocks_db_path)
    except Exception as error:
        logger.error("Failed to load settings configuration: %s", error)
        sys.exit(1)

    restored_symbols = run_ignored_symbols_check(stocks_db_path)

    logger.info(
        "--- Finish. Restored %d symbols (%s). ---",
        len(restored_symbols),
        ", ".join(sorted(restored_symbols)) if restored_symbols else "none",
    )
    return restored_symbols


if __name__ == "__main__":
    check_and_restore_symbols()
