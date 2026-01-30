import logging
from pathlib import Path
from .services.market_data import MarketDataService, DataValidator

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
        # Service instanziieren (nutzt intern eigene Session)
        svc = MarketDataService(db_path)
        
        # 1. Update
        svc.update_market_data(full_reload=False)
        
        # 2. Validierung
        val = DataValidator(svc)
        val.run_checks() # Simple Checks
        
        # 3. Gap Check
        svc.perform_gap_check()
        
    except Exception as e:
        logger.error(f"Fehler im Marktdaten-Update: {e}")

def run_db_maintenance(db_path: Path):
    """
    Datenbank-Pflege (Sonntags).
    """
    logger.info("⏰ Scheduler: Starte DB Maintenance...")
    try:
        svc = MarketDataService(db_path)
        # SQLite Optimierung
        svc.repo.execute("VACUUM")
        svc.repo.execute("ANALYZE")
        logger.info("DB Maintenance fertig.")
    except Exception as e:
        logger.error(f"Maintenance Error: {e}")