import logging
from datetime import datetime
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
            logger.warning(f"Backup {backup_filename} existiert bereits. Überschreibe...")
            backup_file.unlink()

        with session_factory.connect() as conn:
            # VACUUM INTO erwartet einen Dateipfad als String literal
            conn.execute(f"VACUUM INTO '{str(backup_file)}'")
            
        logger.info(f"Backup erfolgreich erstellt: {backup_file}")
        
        # 3. Retention Policy: Nur die letzten 5 behalten
        # Wir suchen alle Dateien die mit db_path.name beginnen
        all_backups = sorted(backup_dir.glob(f"{db_path.name}.*"))
        
        # Wenn mehr als 5, die ältesten löschen
        keep_count = 5
        if len(all_backups) > keep_count:
            files_to_delete = all_backups[:-keep_count]
            for f in files_to_delete:
                try:
                    f.unlink()
                    logger.info(f"Altes Backup gelöscht: {f.name}")
                except Exception as del_err:
                    logger.error(f"Fehler beim Löschen von {f.name}: {del_err}")
                    
    except Exception as e:
        logger.error(f"Fehler beim DB Backup: {e}", exc_info=True)