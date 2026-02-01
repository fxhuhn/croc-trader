import logging
from pathlib import Path
from .services.market.updater import MarketDataUpdater
from .services.market.quality import MarketQualityService
from .database.session import DatabaseSession

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

def run_market_data_update(db_path: Path):
    """
    Lädt Marktdaten herunter (Täglich).
    """
    logger.info("⏰ Scheduler: Starte Marktdaten-Update...")
    try:
        # Session Factory
        session_factory = DatabaseSession(str(db_path))
        
        # 1. Updater Init
        updater = MarketDataUpdater(session_factory)
        
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