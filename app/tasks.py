import logging
from datetime import datetime
from pathlib import Path

from flask import Flask

from .config import ConfigManager
from .database.session import DatabaseSession
from .extensions import cache
from .services.market.quality import MarketQualityService
from .services.market.updater import MarketDataUpdater
from .services.telegram import TelegramBot

logger = logging.getLogger(__name__)


def run_daily_strategy_check(app: Flask) -> None:
    """Daily strategy check (screener). Triggered by the scheduler.

    Args:
        app: The Flask application instance.
    """
    with app.app_context():
        logger.info("⏰ Scheduler: Starting daily strategy check...")

        screener_engine = app.extensions.get("screener_engine")

        if not screener_engine:
            logger.error("Screener Engine not found!")
            return

        try:
            screener_engine.run_all(days=0)
        except Exception as error:
            logger.error("Screener job error: %s", error)


def run_cache_prewarm(app: Flask) -> None:
    """Triggered by the scheduler to pre-warm the cache for the day.

    Args:
        app: The Flask application instance.
    """
    logger.info("⏰ Scheduler: Starting Cache Pre-warming...")
    try:
        _clear_and_prewarm_cache(app)
    except Exception as error:
        logger.error("Cache pre-warming error: %s", error)


def run_market_data_update(
    db_path: Path, telegram_bot: TelegramBot | None = None
) -> None:
    """Downloads market data (daily).

    Args:
        db_path: Path to stocks database file.
        telegram_bot: Optional TelegramBot notification instance.
    """
    logger.info("⏰ Scheduler: Starting market data update...")
    try:
        if telegram_bot is None:
            try:
                from flask import current_app, has_app_context

                if has_app_context():
                    telegram_bot = current_app.extensions.get("telegram")
            except Exception:
                telegram_bot = None

        if telegram_bot is None:
            try:
                config_manager = ConfigManager()
                telegram_config = config_manager.app.telegram
                telegram_bot = TelegramBot(
                    token=telegram_config.token,
                    chat_id=telegram_config.chat_id,
                    enabled=telegram_config.enabled,
                )
            except Exception as config_error:
                logger.debug(
                    "Could not initialize TelegramBot from ConfigManager: %s",
                    config_error,
                )

        session_factory = DatabaseSession(str(db_path))
        signals_path = db_path.parent / "signals.db"
        signals_session = DatabaseSession(str(signals_path))

        updater = MarketDataUpdater(session_factory, signals_session)
        updater.run_update(full_reload=False)

        quality_service = MarketQualityService(updater, telegram_bot=telegram_bot)
        quality_service.perform_gap_check()
        quality_service.check_last_trading_day_completeness()

    except Exception as error:
        logger.error("Market data update error: %s", error, exc_info=True)


def run_db_maintenance(db_path: Path) -> None:
    """Database maintenance routine (VACUUM & ANALYZE).

    Args:
        db_path: Path to target database file.
    """
    logger.info("⏰ Scheduler: Starting DB Maintenance...")
    try:
        session_factory = DatabaseSession(str(db_path))
        with session_factory.connect() as database_connection:
            database_connection.execute("VACUUM")
            database_connection.execute("ANALYZE")

        logger.info("DB maintenance completed.")
    except Exception as error:
        logger.error("Maintenance error: %s", error)


def run_db_backup(db_path: Path) -> None:
    """Creates a daily backup of the database with a 5-file retention policy.

    Args:
        db_path: Path to database file to back up.
    """
    logger.info("⏰ Scheduler: Starting DB backup for %s...", db_path.name)

    try:
        backup_directory = db_path.parent / "backup"
        backup_directory.mkdir(parents=True, exist_ok=True)

        current_date_string = datetime.now().strftime("%Y-%m-%d")
        backup_file_name = f"{db_path.name}.{current_date_string}"
        backup_file_path = backup_directory / backup_file_name

        if backup_file_path.exists():
            logger.warning("Backup %s already exists. Overwriting...", backup_file_name)
            backup_file_path.unlink()

        session_factory = DatabaseSession(str(db_path))
        backup_path_string = str(backup_file_path.resolve())

        with session_factory.connect() as database_connection:
            database_connection.execute("VACUUM INTO ?", (backup_path_string,))

        logger.info("Backup successfully created: %s", backup_file_path)
        _enforce_backup_retention(backup_directory, db_path.name, max_backups=5)

    except Exception as error:
        logger.error("DB backup error: %s", error, exc_info=True)


def run_order_generation(app: Flask) -> None:
    """Generates daily order CSV files for created trades.

    Args:
        app: The Flask application instance.
    """
    with app.app_context():
        logger.info("⏰ Scheduler: Starting daily order generation...")
        trade_manager = app.extensions.get("trade_manager")
        if not trade_manager:
            logger.error("TradeManager not found!")
            return

        try:
            order_file_path = trade_manager.generate_daily_orders()
            if order_file_path:
                logger.info("Order generation successful: %s", order_file_path)
            else:
                logger.info("ℹ️ No orders to generate.")
        except Exception as error:
            logger.error("Error during order generation: %s", error, exc_info=True)


def _clear_and_prewarm_cache(app: Flask) -> None:
    """Clears Flask-Caching and warms routes in production mode.

    Args:
        app: The Flask application instance.
    """
    import os
    import sys

    if (
        app.debug
        or app.testing
        or "pytest" in sys.modules
        or os.environ.get("PYTEST_CURRENT_TEST")
    ):
        logger.info("Skipping cache pre-warming in debug/testing mode.")
        return

    with app.app_context():
        logger.info("🧹 Clearing cache for registered routes...")
        cache.clear()
        _prewarm_target_routes(app)


def _prewarm_target_routes(app: Flask) -> None:
    """Executes pre-warm GET requests for registered routes.

    Args:
        app: Active Flask instance.
    """
    target_routes: list[str] = [
        "/analytics",
        "/trades",
        "/trades/croc",
        "/trades/dip-buyer",
        "/trades/turnover",
        "/trades/ndx-momentum",
        "/trades/twopercent",
        "/trades/tgim",
        "/trades/bridge-scout",
        "/trades/bounce-bandit",
        "/screener",
        "/screener/croc",
        "/screener/dip-buyer",
        "/screener/turnover",
        "/screener/twopercent",
        "/screener/ndx-momentum",
        "/screener/tgim",
        "/screener/bridge-scout",
        "/screener/bounce-bandit",
    ]

    logger.info("🔥 Starting pre-warming of registered routes...")
    with app.test_client() as test_client:
        for route_path in target_routes:
            _warm_single_route(test_client, route_path)
    logger.info("✅ Cache pre-warming completed.")


def _warm_single_route(test_client: object, route_path: str) -> None:
    """Executes single pre-warm request.

    Args:
        test_client: Flask test client instance.
        route_path: Relative route path.
    """
    try:
        response = test_client.get(route_path)  # type: ignore[attr-defined]
        if response.status_code == 200:
            logger.info("  ✓ Pre-warmed: %s", route_path)
            return

        logger.warning(
            "  ✗ Pre-warm failed for %s: Status %s",
            route_path,
            response.status_code,
        )
    except Exception as error:
        logger.error("  ✗ Exception during pre-warm of %s: %s", route_path, error)


def _enforce_backup_retention(
    backup_directory: Path, database_name: str, max_backups: int
) -> None:
    """Deletes oldest backup files exceeding the max retention policy.

    Args:
        backup_directory: Directory path containing backups.
        database_name: Base name of database.
        max_backups: Maximum count of recent backups to keep.
    """
    all_backups = sorted(backup_directory.glob(f"{database_name}.*"))
    if len(all_backups) <= max_backups:
        return

    files_to_delete = all_backups[:-max_backups]
    for target_file in files_to_delete:
        try:
            target_file.unlink()
            logger.info("Old backup deleted: %s", target_file.name)
        except Exception as delete_error:
            logger.error("Error deleting %s: %s", target_file.name, delete_error)
