"""Dry-Run Order Generation Tester.

Creates a temporary copy of the active signals database, runs
TradeManager.generate_daily_orders(), and validates the resulting CSV file against
column headers and format rules. Used to verify order structure changes safely
on real data without modifying production databases.

Usage:
    python script/dry_run_orders.py

Side Effects:
    Creates and subsequently deletes a temporary SQLite test database ('data/signals_test.db').
"""

import logging
import shutil
import sys
from pathlib import Path

# Setup project path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.config import settings  # noqa: E402
from app.services.trade_manager.manager import TradeManager  # noqa: E402

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DryRunOrders")

TEST_DB_PATH = project_root / "data" / "signals_test.db"
STOCKS_DB_PATH = project_root / "data" / "stocks.db"


def clean_test_database() -> None:
    """Removes the test database file and associated SQLite files if they exist."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        file_to_remove = TEST_DB_PATH.with_name(TEST_DB_PATH.name + suffix)
        if file_to_remove.exists():
            try:
                file_to_remove.unlink()
            except Exception as e:
                logger.warning(
                    "Failed to remove test database file %s: %s", file_to_remove, e
                )


def verify_generated_csv(csv_path_string: str) -> None:
    """Prints and validates the contents of the generated CSV file."""
    csv_path = Path(csv_path_string)
    if not csv_path.exists():
        logger.error("Generated CSV file does not exist: %s", csv_path)
        sys.exit(1)

    logger.info("CSV file generated successfully at: %s", csv_path)
    print("\n" + "=" * 40 + " GENERATED CSV CONTENTS " + "=" * 40)

    with open(csv_path) as csv_file:
        lines = csv_file.readlines()
        for line in lines:
            print(line.rstrip())

    print("=" * 104 + "\n")

    # Simple column presence and count assertion
    header = lines[0].strip().split(",")
    expected_header = [
        "trade_group_id",
        "bracket_role",
        "symbol",
        "sec_type",
        "exchange",
        "account_id",
        "action",
        "quantity",
        "order_type",
        "target_price",
        "tif",
        "strategy_name",
        "currency",
    ]

    if header != expected_header:
        logger.error(
            "Header mismatch! Found: %s, Expected: %s", header, expected_header
        )
        sys.exit(1)

    logger.info("CSV headers are 100% correct.")


def main() -> None:
    """Main execution orchestrator for the dry run."""
    clean_test_database()

    # Copy live signals database to TEST_DB_PATH
    live_db_path = settings.get_path("signals")
    logger.info("Creating a temporary copy of live signals DB at: %s", TEST_DB_PATH)
    shutil.copy2(live_db_path, TEST_DB_PATH)

    # Initialize TradeManager with our test database
    trade_manager = TradeManager(
        db_path=TEST_DB_PATH, stocks_db_path=STOCKS_DB_PATH, telegram_bot=None
    )

    # Generate daily orders
    csv_file_path = trade_manager.generate_daily_orders()

    if csv_file_path is None:
        logger.error("Order generation returned None! No CSV file was written.")
        sys.exit(1)

    # Verify CSV file
    verify_generated_csv(csv_file_path)

    # Clean up test DB
    clean_test_database()
    logger.info("Dry run completed successfully.")


if __name__ == "__main__":
    main()
