import logging
from datetime import datetime
from pathlib import Path
from flask import Flask
from .services.market.updater import MarketDataUpdater
from .services.market.quality import MarketQualityService
from .database.session import DatabaseSession
from .extensions import cache

logger = logging.getLogger(__name__)


def run_daily_strategy_check(app):
    """Daily strategy check (screener). Triggered by the scheduler."""
    with app.app_context():
        logger.info("⏰ Scheduler: Starting daily strategy check...")

        screener = app.extensions.get("screener_engine")

        if screener:
            try:
                # 1. Run screener (scans for signals)
                screener.run_all(days=0)
            except Exception as e:
                logger.error("Screener job error: %s", e)
        else:
            logger.error("Screener Engine not found!")


def _clear_and_prewarm_cache(app):
    """
    Clears the Flask-Caching cache and preemptively fetches the /trades routes
    to ensure the first user request is instantaneous.
    Runs ONLY in production (debug=False).
    """
    if app.debug:
        logger.info("Skipping cache pre-warming in debug mode.")
        return

    with app.app_context():
        logger.info("🧹 Clearing cache for /trades routes...")
        cache.clear()

        logger.info("🔥 Starting pre-warming of /trades routes...")
        with app.test_client() as client:
            routes = [
                "/analytics",
                "/trades",
                "/trades/croc",
                "/trades/dip-buyer",
                "/trades/turnover",
                "/trades/ndx-momentum",
                "/trades/twopercent",
            ]
            for route in routes:
                try:
                    response = client.get(route)
                    if response.status_code == 200:
                        logger.info("  ✓ Pre-warmed: %s", route)
                    else:
                        logger.warning(
                            "  ✗ Pre-warm failed for %s: Status %s",
                            route,
                            response.status_code,
                        )
                except Exception as e:
                    logger.error("  ✗ Exception during pre-warm of %s: %s", route, e)

        logger.info("✅ Cache pre-warming completed.")


def run_cache_prewarm(app):
    """Triggered by the scheduler to pre-warm the cache for the day.

    Should run after the TradeManager (which runs at 07:00).
    """
    logger.info("⏰ Scheduler: Starting Cache Pre-warming...")
    try:
        _clear_and_prewarm_cache(app)
    except Exception as e:
        logger.error("Cache pre-warming error: %s", e)


def run_market_data_update(db_path: Path):
    """Downloads market data (daily)."""
    logger.info("⏰ Scheduler: Starting market data update...")
    try:
        # Session Factory for Stocks
        session_factory = DatabaseSession(str(db_path))

        # Session Factory for Signals (derived path)
        # Assuming signals.db is in the same directory as stocks.db
        signals_path = db_path.parent / "signals.db"
        signals_session = DatabaseSession(str(signals_path))

        # 1. Updater Init
        updater = MarketDataUpdater(session_factory, signals_session)

        # 2. Run Update
        updater.run_update(full_reload=False)

        # 3. Quality & Gap Check
        quality = MarketQualityService(updater)
        quality.perform_gap_check()

    except Exception as e:
        logger.error("Market data update error: %s", e, exc_info=True)


def run_db_maintenance(db_path: Path):
    """Database maintenance (Sundays)."""
    logger.info("⏰ Scheduler: Starting DB Maintenance...")
    try:
        session_factory = DatabaseSession(str(db_path))
        # Use repository or direct database connection
        with session_factory.connect() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")

        logger.info("DB maintenance completed.")
    except Exception as e:
        logger.error("Maintenance error: %s", e)


def run_db_backup(db_path: Path):
    """Creates a daily backup of the given database.

    Retains only the last 5 backups.
    Uses SQLite VACUUM INTO for safe hot-backups.
    """
    logger.info("⏰ Scheduler: Starting DB backup for %s...", db_path.name)

    try:
        # 1. Define paths
        backup_dir = db_path.parent / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d")
        backup_filename = f"{db_path.name}.{timestamp}"
        backup_file = backup_dir / backup_filename

        # 2. Create backup (VACUUM INTO)
        # Use direct connection for VACUUM INTO
        # Since VACUUM INTO is a SQL statement, we need a connection
        session_factory = DatabaseSession(str(db_path))

        # Check if backup already exists to avoid error or overwrite
        if backup_file.exists():
            logger.warning("Backup %s already exists. Overwriting...", backup_filename)
            backup_file.unlink()

        with session_factory.connect() as conn:
            # SAFETY NOTE: VACUUM INTO does not support parameterized queries
            # (SQLite limitation). The path is constructed from controlled inputs
            # (db_path.name + timestamp), not from user input.
            backup_path_string = str(backup_file.resolve())
            conn.execute(f"VACUUM INTO '{backup_path_string}'")

        logger.info("Backup successfully created: %s", backup_file)

        # 3. Retention policy: keep only the last 5
        all_backups = sorted(backup_dir.glob(f"{db_path.name}.*"))

        # If more than 5, delete the oldest
        keep_count = 5
        if len(all_backups) > keep_count:
            files_to_delete = all_backups[:-keep_count]
            for file_to_delete in files_to_delete:
                try:
                    file_to_delete.unlink()
                    logger.info("Old backup deleted: %s", file_to_delete.name)
                except Exception as delete_error:
                    logger.error(
                        "Error deleting %s: %s", file_to_delete.name, delete_error
                    )

    except Exception as e:
        logger.error("DB backup error: %s", e, exc_info=True)


def run_order_generation(app: Flask) -> None:
    """Generates the daily order CSV files for created trades.

    Args:
        app: The Flask application instance.
    """
    with app.app_context():
        logger.info("⏰ Scheduler: Starting daily order generation...")
        trade_manager = app.extensions.get("trade_manager")
        if trade_manager:
            try:
                order_file_path = trade_manager.generate_daily_orders()
                if order_file_path:
                    logger.info("Order generation successful: %s", order_file_path)
                else:
                    logger.info("ℹ️ No orders to generate.")
            except Exception as e:
                logger.error("Error during order generation: %s", e, exc_info=True)
        else:
            logger.error("TradeManager not found!")
