"""Foreground Diagnostics and Test Order Generator.

Performs a dual-stage dry-run check of the TradeManager:
1. Generates live order CSV files directly from the current active database (read-only for the database).
2. Simulates a full daily update process (evaluating exits and entries) on a temporary copy of signals.db
   to print changes and order results before actual background execution.

Usage:
    python script/manual_test_run.py

Side Effects:
    Generates CSV files in the data/orders/ directory. Creates and deletes a temporary database
    ('data/signals_temp_simulation.db') to run the simulation safely.
"""

import logging
import shutil
import sqlite3
import sys
from pathlib import Path

# Setup project path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from app.config import settings  # noqa: E402
from app.services.trade_manager.manager import TradeManager  # noqa: E402

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ManualTestRun")


def get_active_created_count(db_path: Path) -> dict[str, int]:
    """Helper to count active and created trades in the given database."""
    counts = {"ACTIVE": 0, "CREATED": 0, "CLOSED": 0, "INVALID": 0}
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT status, count(*) FROM trades GROUP BY status")
            for status, count in cursor.fetchall():
                if status in counts:
                    counts[status] = count
    except Exception as e:
        logger.error("Error reading database counts: %s", e)
    return counts


def run_live_order_generation() -> Path | None:
    """Runs live order generation (read-only for database) and returns CSV path."""
    logger.info("=== STEP 1: Direct Order Generation from Live Database ===")
    db_path = settings.get_path("signals")
    stocks_db_path = settings.get_path("stocks")

    logger.info("Signals DB: %s", db_path)
    logger.info("Stocks DB: %s", stocks_db_path)

    trade_manager = TradeManager(
        db_path=db_path, stocks_db_path=stocks_db_path, telegram_bot=None
    )

    csv_file_path_str = trade_manager.generate_daily_orders()

    if not csv_file_path_str:
        logger.warning("No orders generated from the live database.")
        return None

    csv_path = Path(csv_file_path_str)
    logger.info("Order CSV successfully generated at: %s", csv_path)
    return csv_path


def _clean_temp_db(db_path: Path) -> None:
    """Removes the temporary database file and any associated SQLite files."""
    for suffix in ("", "-wal", "-shm", "-journal"):
        file_to_remove = db_path.with_name(db_path.name + suffix)
        if file_to_remove.exists():
            try:
                file_to_remove.unlink()
            except Exception as e:
                logger.warning(
                    "Failed to remove temporary database file %s: %s", file_to_remove, e
                )


def run_simulated_daily_process() -> None:
    """Copies signals.db, runs daily process, and generates orders on the copy."""
    logger.info("=== STEP 2: Simulated Daily Process (Entries & Exits) ===")
    live_db_path = settings.get_path("signals")
    stocks_db_path = settings.get_path("stocks")
    temp_db_path = live_db_path.parent / "signals_temp_simulation.db"

    # Clean up any leftover file
    _clean_temp_db(temp_db_path)

    logger.info("Creating a temporary copy of signals.db at: %s", temp_db_path)
    shutil.copy2(live_db_path, temp_db_path)

    try:
        initial_counts = get_active_created_count(temp_db_path)
        logger.info(
            "Initial states in database copy: ACTIVE=%d, CREATED=%d, CLOSED=%d, INVALID=%d",
            initial_counts["ACTIVE"],
            initial_counts["CREATED"],
            initial_counts["CLOSED"],
            initial_counts["INVALID"],
        )

        trade_manager = TradeManager(
            db_path=temp_db_path, stocks_db_path=stocks_db_path, telegram_bot=None
        )

        logger.info("Running daily process (run_daily_process)...")
        trade_manager.run_daily_process()

        post_counts = get_active_created_count(temp_db_path)
        logger.info(
            "Post-process states in database copy: ACTIVE=%d, CREATED=%d, CLOSED=%d, INVALID=%d",
            post_counts["ACTIVE"],
            post_counts["CREATED"],
            post_counts["CLOSED"],
            post_counts["INVALID"],
        )

        # Detect changes
        changes = {k: post_counts[k] - initial_counts[k] for k in initial_counts}
        logger.info("Net changes from simulation run: %s", changes)

        logger.info("Generating daily orders from the post-processed database...")
        csv_file_path_str = trade_manager.generate_daily_orders(
            reference_date="simulation"
        )
        if csv_file_path_str:
            csv_path = Path(csv_file_path_str)
            logger.info("Simulated Order CSV generated at: %s", csv_path)
            print_csv_contents(csv_path, title="SIMULATED CSV CONTENTS")
            csv_path.unlink()
        else:
            logger.info("No orders generated after simulated daily process.")

    finally:
        logger.info("Cleaning up temporary database copy...")
        _clean_temp_db(temp_db_path)


def print_csv_contents(csv_path: Path, title: str = "GENERATED CSV CONTENTS") -> None:
    """Reads and prints the contents of the generated CSV file."""
    if not csv_path.exists():
        logger.error("CSV file does not exist: %s", csv_path)
        return

    print("\n" + "=" * 40 + f" {title} " + "=" * 40)
    with open(csv_path, encoding="utf-8") as csv_file:
        for line in csv_file:
            print(line.rstrip())
    print("=" * (80 + len(title)) + "\n")


def main() -> None:
    """Main execution orchestrator."""
    logger.info("--- Starting Manual Test Run with Today's Data ---")

    # 1. Live generation check (Read-only on DB)
    csv_path = run_live_order_generation()
    if csv_path:
        print_csv_contents(csv_path, title="LIVE GENERATED CSV CONTENTS")

    # 2. Simulation check (Safe write/modify)
    run_simulated_daily_process()

    logger.info("--- Manual Test Run Completed ---")


if __name__ == "__main__":
    main()
