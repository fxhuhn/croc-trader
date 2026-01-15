import csv
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path

from ..config import settings  # NEU
from ..models import CrocSignal, SignalStat

# Import aus dem gleichen Package
from .database import SignalDatabase

logger = logging.getLogger(__name__)


class BackgroundWorker:
    """Verarbeitet Webhooks aus der Queue."""

    def __init__(self, db_path: Path, batch_size: int = 40, timeout: int = 5):
        self.queue: queue.Queue = queue.Queue()
        self.db_path = db_path
        self.batch_size = batch_size
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="Webhook-Worker"
        )

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Webhook Worker gestartet.")

    def enqueue(self, signal: CrocSignal):
        self.queue.put(signal)

    def _run(self):
        db = SignalDatabase(self.db_path)
        while not self._stop_event.is_set():
            batch = []
            start_time = time.time()

            # Queue leeren Loop
            while len(batch) < self.batch_size:
                time_left = self.timeout - (time.time() - start_time)
                if time_left <= 0 and batch:
                    break

                try:
                    wait = max(0.1, time_left) if batch else 1.0
                    item = self.queue.get(timeout=wait)
                    batch.append(item)
                except queue.Empty:
                    if batch:
                        break
                    continue

            if batch:
                db.save_many(batch)
                for _ in range(len(batch)):
                    self.queue.task_done()


class CsvImportWorker:
    """Überwacht Ordner auf neue CSV Stats."""

    def __init__(self, data_folder: Path, db_path: Path, check_interval: int = 60):
        self.data_folder = data_folder
        self.db_path = db_path
        self.check_interval = check_interval

        # NEU: Dateiname aus Config
        self.target_file = settings.get_path("stats_import")

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CSV-Importer"
        )

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()
            logger.info(f"CSV Watcher gestartet. Ziel: {self.target_file}")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if self.target_file.exists() and self.target_file.is_file():
                    self._process_file()
                    self._cleanup_old_files()  # Auto-Cleanup
            except Exception as e:
                logger.error(f"CSV Worker Loop Error: {e}")

            self._stop_event.wait(self.check_interval)

    def _process_file(self):
        logger.info("Importiere Statistik...")
        db = SignalDatabase(self.db_path)
        try:
            stats = []
            with open(self.target_file, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "signal" in row:
                        stats.append(SignalStat(**row))

            if stats:
                db.replace_stats(stats)
                db.optimize()  # VACUUM nach Import
                self._mark_processed(".imported")
            else:
                self._mark_processed(".empty")

        except Exception as e:
            logger.error(f"Import Fehler: {e}", exc_info=True)
            self._mark_processed(".error")

    def _mark_processed(self, suffix: str):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{self.target_file.name}{suffix}_{ts}"
        try:
            self.target_file.rename(self.data_folder / new_name)
        except OSError as e:
            logger.error(f"Rename failed: {e}")

    def _cleanup_old_files(self, days=30):
        limit = time.time() - (days * 86400)
        # Suche nach pattern basierend auf dem Dateinamen
        pattern = f"{self.target_file.name}.*"
        for file in self.data_folder.glob(pattern):
            try:
                if file.stat().st_mtime < limit:
                    file.unlink()
            except Exception as e:
                logger.error(f"Fehler beim Löschen von {file}: {e}")

    def backup(self, backup_folder: Path):
        """Erstellt ein Hot-Backup der Datenbank im laufenden Betrieb."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_folder / f"signals_backup_{timestamp}.db"

        try:
            # Verbindung zur Source DB
            source_conn = self._get_conn()

            # Verbindung zur Backup DB Datei
            dest_conn = sqlite3.connect(backup_file)

            # SQLite interne Backup Funktion (kopiert sicher Page für Page)
            with source_conn, dest_conn:
                source_conn.backup(dest_conn)

            source_conn.close()
            dest_conn.close()

            logger.info(f"Datenbank Backup erstellt: {backup_file}")

            # Optional: Alte Backups löschen (z.B. älter als 7 Tage)
            self._cleanup_backups(backup_folder)

        except Exception as e:
            logger.error(f"Backup fehlgeschlagen: {e}")

    def _cleanup_backups(self, folder: Path, keep=7):
        # todo: Missing Backup Function
        logger.warning("Cleanup Backups nicht implementiert")
