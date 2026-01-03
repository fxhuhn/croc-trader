import csv
import logging
import queue
import sqlite3
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import List

from .models import CrocSignal, SignalStat

logger = logging.getLogger(__name__)


class SignalDatabase:
    """Handhabt SQLite Verbindungen und Migrationen."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)

        # WICHTIG: Timeouts erhöhen (falls DB kurz gesperrt ist)
        conn.execute("PRAGMA busy_timeout = 3000;")

        # WICHTIG: WAL Mode für bessere Concurrency
        conn.execute("PRAGMA journal_mode = WAL;")

        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Erstellt Tabellen und führt einfache Migrationen durch."""
        base_schema = """
        CREATE TABLE IF NOT EXISTS signals (
            symbol TEXT NOT NULL, timeframe TEXT NOT NULL, signal TEXT NOT NULL,
            timestamp TEXT NOT NULL, close REAL, high REAL, low REAL, wuk REAL,
            status TEXT, kerze TEXT, trend TEXT, setter TEXT, welle TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_signals_filter ON signals(symbol, timeframe, signal, timestamp DESC);
        """

        # Spalten die existieren müssen (Name: Typ)
        required_columns = {
            "wolke": "TEXT",
            "rsi": "REAL",
            "sma_200": "REAL",
            "sma_20": "REAL",
            "strategy_id": "TEXT",
            "reference": "TEXT",
            "exchange": "TEXT",
        }

        try:
            with self._get_conn() as conn:
                conn.executescript(base_schema)
                # Migration check
                cursor = conn.execute("PRAGMA table_info(signals)")
                existing_cols = {row["name"] for row in cursor.fetchall()}

                for col_name, col_def in required_columns.items():
                    if col_name not in existing_cols:
                        logger.info(f"Migration: Füge Spalte '{col_name}' hinzu.")
                        conn.execute(
                            f"ALTER TABLE signals ADD COLUMN {col_name} {col_def}"
                        )
                conn.commit()
        except sqlite3.Error as e:
            logger.critical(f"DB Init failed: {e}")
            raise

        # NEU: Tabelle für Statistiken
        stats_schema = """
        CREATE TABLE IF NOT EXISTS signal_stats (
            signal TEXT, symbol TEXT, timeframe TEXT, exchange TEXT, level TEXT,
            total REAL, win REAL, loss REAL, rejected REAL,
            wolke TEXT, welle TEXT, trend TEXT, setter TEXT,

            -- Automatische Spalten (werden nicht inserted!)
            win_rate REAL GENERATED ALWAYS AS (
                CASE WHEN (win+loss) > 0 THEN ROUND((win / (win+loss)) * 100, 2) ELSE 0 END
            ) VIRTUAL,

            loss_rate REAL GENERATED ALWAYS AS (
                CASE WHEN (win+loss) > 0 THEN ROUND((loss / (win+loss)) * 100, 2) ELSE 0 END
            ) VIRTUAL,

            updated_at TEXT
        );
        -- Index für schnelle Abfragen (z.B. Gib mir Stats für Symbol X)
        CREATE INDEX IF NOT EXISTS idx_stats_lookup
        ON signal_stats(symbol, signal, timeframe);
        """
        try:
            with self._get_conn() as conn:
                conn.executescript(stats_schema)  # Das neue Schema
                conn.commit()
        except sqlite3.Error as e:
            logger.critical(f"DB Init failed: {e}")
            raise

    def save_many(self, signals: List[CrocSignal]) -> int:
        """Batch Insert für hohe Performance."""
        if not signals:
            return 0

        sql = """
        INSERT INTO signals
        (symbol, timeframe, timestamp, signal, close, high, low, wuk,
         status, kerze, wolke, trend, setter, welle,
         rsi, sma_200, sma_20, strategy_id, reference, exchange)
        VALUES
        (:symbol, :timeframe, :timestamp, :signal, :close, :high, :low, :wuk,
         :status, :kerze, :wolke, :trend, :setter, :welle,
         :rsi, :sma_200, :sma_20, :strategy_id, :reference, :exchange)
        """
        try:
            with self._get_conn() as conn:
                conn.executemany(sql, [s.to_db_row() for s in signals])
                conn.commit()
            return len(signals)
        except sqlite3.Error as e:
            logger.error(f"DB Batch Insert Error: {e}")
            return 0

    def replace_stats(self, stats: List["SignalStat"]) -> int:
        """
        Löscht ALLE alten Statistiken und fügt die neuen ein (Atomic Transaction).
        """
        if not stats:
            return 0

        sql_insert = """
        INSERT INTO signal_stats
        (signal, symbol, timeframe, exchange, level, total, win, loss, rejected,
         wolke, welle, trend, setter, updated_at)
        VALUES
        (:signal, :symbol, :timeframe, :exchange, :level, :total, :win, :loss, :rejected,
         :wolke, :welle, :trend, :setter, :updated_at)
        """

        try:
            with self._get_conn() as conn:
                # 1. Alte Daten löschen
                conn.execute("DELETE FROM signal_stats")

                # 2. Neue Daten einfügen
                conn.executemany(sql_insert, [s.to_db_row() for s in stats])

                # 3. Commit passiert automatisch durch 'with' Kontext, wenn kein Fehler auftritt
                conn.commit()

            return len(stats)
        except sqlite3.Error as e:
            logger.error(f"Stats Replacement Error: {e}")
            raise

    def optimize(self):
        """Führt Wartungsarbeiten durch (Vacuuming)."""
        try:
            with self._get_conn() as conn:
                # VACUUM kann nicht innerhalb einer Transaktion laufen,
                # daher nutzen wir isolation_level=None für Autocommit Mode
                conn.isolation_level = None
                conn.execute("VACUUM")
            logger.info("Datenbank erfolgreich optimiert (VACUUM).")
        except sqlite3.Error as e:
            logger.error(f"Fehler bei DB-Optimierung: {e}")


class BackgroundWorker:
    """Verwaltet die Queue und den Hintergrund-Thread."""

    def __init__(self, db_path: Path, batch_size: int = 40, timeout: int = 5):
        self.queue: queue.Queue = queue.Queue()
        self.db_path = db_path
        self.batch_size = batch_size
        self.timeout = timeout
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()
            logger.info("Background Worker gestartet.")

    def enqueue(self, signal: CrocSignal):
        self.queue.put(signal)

    def _run(self):
        # DB Verbindung wird erst im Thread aufgebaut (Thread-Safety)
        db = SignalDatabase(self.db_path)

        while not self._stop_event.is_set():
            batch = []
            start_time = time.time()

            # Time & Count Strategie
            while len(batch) < self.batch_size:
                time_left = self.timeout - (time.time() - start_time)
                if time_left <= 0 and batch:
                    break  # Timeout erreicht, schreiben

                try:
                    # Wenn timeout <= 0 und batch leer ist, warten wir etwas länger,
                    # um Busy-Looping zu vermeiden
                    wait = max(0.1, time_left) if batch else 1.0
                    item = self.queue.get(timeout=wait)
                    batch.append(item)
                except queue.Empty:
                    if batch:
                        break  # Timeout beim Warten auf nächstes Item -> Schreiben
                    continue  # Queue leer, weiter warten

            if batch:
                count = db.save_many(batch)
                logger.debug(f"Worker: {count} Signale gespeichert.")
                for _ in range(len(batch)):
                    self.queue.task_done()


class CsvImportWorker:
    """
    Überwacht das Datenverzeichnis auf neue Statistik-CSVs und importiert diese automatisch.
    """

    def __init__(self, data_folder: Path, db_path: Path, check_interval: int = 60):
        self.data_folder = data_folder
        self.db_path = db_path
        self.check_interval = check_interval

        # Dateiname, auf den wir warten
        self.target_file = self.data_folder / "croc_statistik.csv"

        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="CSV-Importer"
        )

    def start(self):
        if not self._thread.is_alive():
            self._thread.start()
            logger.info(f"CSV Import Worker gestartet. Überwache: {self.target_file}")

    def _run(self):
        while not self._stop_event.is_set():
            try:
                if self.target_file.exists() and self.target_file.is_file():
                    self._process_file()
            except Exception as e:
                logger.error(f"Fehler im CSV Import Worker Loop: {e}", exc_info=True)

            # Wartezeit bis zum nächsten Check (unterbrechbar)
            self._stop_event.wait(self.check_interval)

            # alte CSV Dateien löschen
            self._cleanup_old_files()

    def _process_file(self):
        logger.info("Neue Statistik-Datei gefunden. Starte Import...")

        # DB Verbindung aufbauen
        db = SignalDatabase(self.db_path)

        try:
            # 1. Datei lesen und parsen
            stats_objects = []
            # Encoding utf-8-sig entfernt mögliches BOM (Byte Order Mark) von Excel
            with open(self.target_file, mode="r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "signal" in row and "symbol" in row:
                        stats_objects.append(SignalStat(**row))

            if not stats_objects:
                logger.warning("Datei war leer oder enthielt keine gültigen Daten.")
                self._mark_as_processed(suffix=".empty")
                return

            logger.info(f" {len(stats_objects)} Datensätz geladen.")

            # 2. In die DB schreiben (Atomic Replace)
            count = db.replace_stats(stats_objects)
            logger.info(f"Import erfolgreich: {count} Datensätze aktualisiert.")

            # 3. Datei umbenennen (als "erledigt" markieren)
            self._mark_as_processed(suffix=".imported")

            # 4. Datenbank aufräumen
            db.optimize()

        except Exception as e:
            logger.error(f"Kritischer Fehler beim CSV Import: {e}", exc_info=True)
            # Im Fehlerfall benennen wir sie auch um, damit der Loop nicht endlos crasht
            self._mark_as_processed(suffix=".error")

    def _mark_as_processed(self, suffix: str):
        """Benennt die Datei um: croc_statistik.csv -> croc_statistik.csv.imported_20231027_1000"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_name = f"{self.target_file.name}{suffix}_{timestamp}"
        destination = self.data_folder / new_name

        try:
            self.target_file.rename(destination)
            logger.info(f"Datei archiviert als: {new_name}")
        except OSError as e:
            logger.error(f"Konnte Datei nicht umbenennen: {e}")

    def _cleanup_old_files(self, days=30):
        """Löscht archivierte Dateien, die älter als 'days' sind."""
        limit = time.time() - (days * 86400)

        # Sucht nach allen Dateien, die wir umbenannt haben
        for file in self.data_folder.glob("croc_statistik.csv.*"):
            try:
                if file.stat().st_mtime < limit:
                    file.unlink()  # Löschen
                    logger.info(f"Alte Datei bereinigt: {file.name}")
            except Exception as e:
                logger.warning(f"Konnte {file.name} nicht löschen: {e}")
