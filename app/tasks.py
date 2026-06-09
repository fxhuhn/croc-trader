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
    """
    Täglicher Strategie-Check (Screener).
    Wird vom Scheduler aufgerufen.
    """
    with app.app_context():
        logger.info("⏰ Scheduler: Starte täglichen Strategie-Check...")

        screener = app.extensions.get("screener_engine")

        if screener:
            try:
                # 1. Screener laufen lassen (Scannt nach Signalen)
                screener.run_all(days=0)
            except Exception as e:
                logger.error(f"Fehler im Screener Job: {e}")
        else:
            logger.error("Screener Engine nicht gefunden!")


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
        logger.info("🧹 Leere Cache für /trades Routen...")
        cache.clear()

        logger.info("🔥 Starte Pre-warming der /trades Routen...")
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
                        logger.info(f"  ✓ Pre-warmed: {route}")
                    else:
                        logger.warning(
                            f"  ✗ Pre-warm fehlgeschlagen für {route}: Status {response.status_code}"
                        )
                except Exception as e:
                    logger.error(f"  ✗ Exception beim Pre-warm von {route}: {e}")

        logger.info("✅ Cache Pre-warming abgeschlossen.")


def run_cache_prewarm(app):
    """
    Wird vom Scheduler aufgerufen, um den Cache für den Tag zu pre-warmen.
    Sollte nach dem TradeManager (der um 07:00 läuft) ausgeführt werden.
    """
    logger.info("⏰ Scheduler: Starte Cache Pre-warming...")
    try:
        _clear_and_prewarm_cache(app)
    except Exception as e:
        logger.error(f"Fehler beim Cache Pre-warming: {e}")


def run_market_data_update(db_path: Path):
    """
    Lädt Marktdaten herunter (Täglich).
    """
    logger.info("⏰ Scheduler: Starte Marktdaten-Update...")
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
        logger.error(f"Fehler im Marktdaten-Update: {e}", exc_info=True)


def run_db_maintenance(db_path: Path):
    """
    Datenbank-Pflege (Sonntags).
    """
    logger.info("⏰ Scheduler: Starte DB Maintenance...")
    try:
        session_factory = DatabaseSession(str(db_path))
        # Nutze Repo oder direkte Session
        with session_factory.connect() as conn:
            conn.execute("VACUUM")
            conn.execute("ANALYZE")

        logger.info("DB Maintenance fertig.")
    except Exception as e:
        logger.error(f"Maintenance Error: {e}")


def run_db_backup(db_path: Path):
    """
    Erstellt ein tägliches Backup der übergebenen Datenbank.
    Behält nur die letzten 5 Backups.
    Verwendet SQLite VACUUM INTO für sichere Hot-Backups.
    """
    logger.info(f"⏰ Scheduler: Starte DB Backup für {db_path.name}...")

    try:
        # 1. Pfade definieren
        backup_dir = db_path.parent / "backup"
        backup_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y-%m-%d")
        backup_filename = f"{db_path.name}.{timestamp}"
        backup_file = backup_dir / backup_filename

        # 2. Backup erstellen (VACUUM INTO)
        # Wir nutzen eine direkte Connection für VACUUM INTO
        # Da VACUUM INTO ein SQL statement ist, brauchen wir eine Connection
        session_factory = DatabaseSession(str(db_path))

        # Check if backup already exists to avoid error or overwrite
        if backup_file.exists():
            logger.warning(
                f"Backup {backup_filename} existiert bereits. Überschreibe..."
            )
            backup_file.unlink()

        with session_factory.connect() as conn:
            # SAFETY NOTE: VACUUM INTO does not support parameterized queries
            # (SQLite limitation). The path is constructed from controlled inputs
            # (db_path.name + timestamp), not from user input.
            backup_path_string = str(backup_file.resolve())
            conn.execute(f"VACUUM INTO '{backup_path_string}'")

        logger.info(f"Backup erfolgreich erstellt: {backup_file}")

        # 3. Retention Policy: Nur die letzten 5 behalten
        # Wir suchen alle Dateien die mit db_path.name beginnen
        all_backups = sorted(backup_dir.glob(f"{db_path.name}.*"))

        # Wenn mehr als 5, die ältesten löschen
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
        logger.error(f"Fehler beim DB Backup: {e}", exc_info=True)


def run_order_generation(app: Flask) -> None:
    """Generates the daily order CSV files for created trades.

    Args:
        app: The Flask application instance.
    """
    with app.app_context():
        logger.info("⏰ Scheduler: Starte tägliche Order-Generierung...")
        trade_manager = app.extensions.get("trade_manager")
        if trade_manager:
            try:
                order_file_path = trade_manager.generate_daily_orders()
                if order_file_path:
                    logger.info("Order generation successful: %s", order_file_path)
                else:
                    logger.info("ℹ️ Keine Orders zu generieren.")
            except Exception as e:
                logger.error("Fehler bei der Order-Generierung: %s", e, exc_info=True)
        else:
            logger.error("TradeManager nicht gefunden!")
