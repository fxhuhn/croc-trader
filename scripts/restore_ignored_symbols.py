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
from datetime import datetime, timedelta
from pathlib import Path

# 1. Setup path so that 'app' can be imported
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.database.repositories.market import MarketRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402
from app.services.market.provider import YahooDataProvider  # noqa: E402

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("RestoreIgnoredSymbols")


def check_and_restore_symbols() -> None:
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

    # Initialize Repository and Data Provider
    session_factory = DatabaseSession(stocks_db_path)
    repo = MarketRepository(session_factory)
    provider = YahooDataProvider()

    # Ensure tables are initialized
    repo.init_schema()

    # Get the list of ignored symbols
    ignored_symbols = repo.get_ignored_symbols()
    if not ignored_symbols:
        logger.info("No ignored symbols found in database. Exiting.")
        return

    logger.info("Found %d ignored symbols to check.", len(ignored_symbols))

    # Query Yahoo for these symbols using a 10-day history (incremental window)
    start_date = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
    symbols_list = list(ignored_symbols)

    logger.info("Downloading last 10 days of history for check...")
    df_batch, failures = provider.fetch_batch_raw(symbols_list, start_date)

    restored_count = 0

    # Evaluate each symbol
    for symbol in symbols_list:
        # If it failed to download or is missing from columns, it is still invalid
        if symbol in failures:
            logger.info("Symbol %s is still failing (No Data).", symbol)
            continue

        # Extract symbol data and check if we got valid price points
        df_sym = provider.extract_symbol_data(df_batch, symbol)
        if df_sym.empty:
            logger.info("Symbol %s returned empty DataFrame.", symbol)
            continue

        # Standardize columns and drop NaN close values
        df_sym.columns = df_sym.columns.str.lower()
        df_sym = df_sym.dropna(subset=["close"])

        if df_sym.empty:
            logger.info("Symbol %s has no valid closing prices.", symbol)
            continue

        # Valid data retrieved! Remove from ignored symbols table
        logger.info(
            "✓ Valid data found for %s (%d records). Restoring symbol...",
            symbol,
            len(df_sym),
        )
        try:
            repo.remove_ignored_symbol(symbol)
            restored_count += 1
        except Exception as error:
            logger.error("Failed to restore symbol %s in database: %s", symbol, error)

    logger.info(
        "--- Finish. Checked %d symbols, restored %d symbols. ---",
        len(ignored_symbols),
        restored_count,
    )


if __name__ == "__main__":
    check_and_restore_symbols()
