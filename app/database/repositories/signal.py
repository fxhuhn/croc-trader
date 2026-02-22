import json
import logging

from typing import Any

from .base import BaseRepository
from ...const import Strategies, TradeStatus

logger = logging.getLogger(__name__)


class SignalRepository(BaseRepository):
    def init_schema(self):
        """Erstellt Tabellen für Signale (croc), Mappings und migriert Altdaten (ohne Löschen)."""
        with self.session.connect() as conn:
            # 1. Neue Haupt-Tabelle 'croc' erstellen
            # WICHTIG: Wir fügen einen UNIQUE Constraint hinzu, damit wir Duplikate beim
            # mehrfachen Ausführen der Migration verhindern.
            self.execute(
                """
                CREATE TABLE IF NOT EXISTS croc (
                    symbol TEXT NOT NULL, 
                    timeframe TEXT, 
                    signal TEXT,
                    timestamp TEXT, 
                    exchange TEXT,
                    data TEXT, -- JSON Payload
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Verhindert Duplikate: Diese Kombi darf es nur einmal geben
                    UNIQUE(symbol, timeframe, signal, timestamp)
                )
            """,
                conn=conn,
            )

            # 2. KOMPLEXE MIGRATION: Alte 'signals' Struktur in neue 'croc' Struktur
            check_old = self.fetch_one(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='signals'",
                conn=conn,
            )

            if check_old:
                logger.info(
                    "🔄 Migration: 'signals' Tabelle gefunden. Prüfe auf neue Daten..."
                )
                try:
                    # A) Alle alten Daten laden
                    cursor = conn.execute("SELECT * FROM signals")
                    # Wir brauchen die Spaltennamen für das Mapping ins JSON
                    columns = [description[0] for description in cursor.description]
                    rows = cursor.fetchall()

                    if rows:
                        migrated_data = []

                        # B) Zeile für Zeile transformieren
                        for row in rows:
                            # Row in Dict wandeln
                            row_dict = dict(zip(columns, row))

                            # 1. Basis-Felder extrahieren
                            symbol = row_dict.pop("symbol")
                            timeframe = row_dict.pop("timeframe")
                            signal = row_dict.pop("signal")
                            timestamp = row_dict.pop("timestamp")
                            exchange = row_dict.get("exchange")
                            created_at = row_dict.get("created_at")

                            # Cleanup: Felder entfernen, die wir nicht im JSON brauchen
                            row_dict.pop("exchange", None)
                            row_dict.pop("created_at", None)
                            row_dict.pop("dist_sma_20", None)
                            row_dict.pop("dist_sma_200", None)

                            # 2. Der REST kommt in das 'data' JSON Feld
                            json_payload = json.dumps(row_dict, ensure_ascii=False)

                            migrated_data.append(
                                (
                                    symbol,
                                    timeframe,
                                    signal,
                                    timestamp,
                                    exchange,
                                    json_payload,
                                    created_at,
                                )
                            )

                        # C) Massen-Insert mit Duplikat-Schutz
                        # 'INSERT OR IGNORE' überspringt Zeilen, die den UNIQUE Constraint verletzen
                        if migrated_data:
                            conn.executemany(
                                """
                                INSERT OR IGNORE INTO croc (symbol, timeframe, signal, timestamp, exchange, data, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """,
                                migrated_data,
                            )

                            # Wir prüfen, wie viele Zeilen tatsächlich eingefügt wurden (changes()) ist hier ungenau bei Batch,
                            # aber das Log ist beruhigend.
                            logger.info(
                                f"✅ Migration abgeschlossen. {len(migrated_data)} Quell-Datensätze verarbeitet (Duplikate wurden ignoriert)."
                            )

                    # D) Alte Tabelle behalten (Sicherheitsnetz)
                    # self.execute("DROP TABLE signals", conn=conn)
                    # logger.info("🗑️ Alte 'signals' Tabelle gelöscht.")

                except Exception as e:
                    logger.error(
                        f"❌ CRITICAL: Fehler bei der Migration von 'signals' zu 'croc': {e}"
                    )

            # 3. Exchange Mapping Tabelle
            self.execute(
                """
                CREATE TABLE IF NOT EXISTS exchange_mappings (
                    symbol TEXT PRIMARY KEY,
                    exchange TEXT NOT NULL
                )
            """,
                conn=conn,
            )

            self.execute(
                "CREATE INDEX IF NOT EXISTS idx_map_sym ON exchange_mappings(symbol)",
                conn=conn,
            )

            # 4. View aktualisieren (greift nun auf 'croc' zu)
            self.execute("DROP VIEW IF EXISTS view_signals_enriched", conn=conn)

            self.execute(
                """
                CREATE VIEW view_signals_enriched AS
                SELECT 
                    s.rowid as id,
                    s.symbol, s.timeframe, s.signal, s.timestamp, s.data,
                    CASE 
                        WHEN s.exchange IS NULL OR s.exchange = '' OR s.exchange = 'BATS' 
                        THEN COALESCE(m.exchange, 'UNKNOWN')
                        ELSE s.exchange 
                    END as exchange
                FROM croc s
                LEFT JOIN exchange_mappings m ON s.symbol = m.symbol
            """,
                conn=conn,
            )

    def save_signal(self, data: dict[str, Any]) -> int:
        """Speichert Raw Webhook Data in 'croc'."""
        symbol = data.get("symbol") or data.get("ticker", "UNKNOWN")
        timeframe = data.get("timeframe")
        signal_name = data.get("signal") or data.get("strategy")
        exchange = data.get("exchange")
        timestamp = data.get("timestamp") or data.get("date")

        # Use INSERT OR REPLACE to allow updating signals (e.g. monthly rebalance updates)
        sql = """
            INSERT OR REPLACE INTO croc (symbol, timeframe, signal, timestamp, exchange, data)
            VALUES (?, ?, ?, ?, ?, ?)
        """

        cursor = self.execute(
            sql, (symbol, timeframe, signal_name, timestamp, exchange, json.dumps(data))
        )
        return cursor.lastrowid

    def get_unprocessed_signals(self, limit: int = 100) -> list[dict]:
        """Holt unverarbeitete Signale aus der croc-Tabelle."""
        # Note: 'processed' column is not in original schema, assuming it might be added
        # or we just get the latest signals. Given current schema, we'll just get latest.
        sql = "SELECT symbol, signal, timestamp, data FROM croc ORDER BY timestamp DESC LIMIT ?"
        return self.fetch_all(sql, (limit,))

    def get_unique_signal_attributes(self) -> dict[str, set[str]]:
        """Holt alle historisch verfügbaren Werte für Signal und weitere Attribute aus der Datenbank."""
        sql = """
            SELECT 
                signal,
                json_extract(data, '$.status') as status,
                json_extract(data, '$.kerze') as kerze,
                json_extract(data, '$.wolke') as wolke,
                json_extract(data, '$.trend') as trend,
                json_extract(data, '$.setter') as setter,
                json_extract(data, '$.welle') as welle
            FROM croc 
            WHERE data IS NOT NULL OR signal IS NOT NULL
        """
        rows = self.fetch_all(sql)
        
        attributes = {
            "Signal": set(),
            "Status": set(),
            "Kerze": set(),
            "Wolke": set(),
            "Trend": set(),
            "Setter": set(),
            "Welle": set(),
        }
        
        for row in rows:
            for yaml_key, db_key in zip(
                ["Signal", "Status", "Kerze", "Wolke", "Trend", "Setter", "Welle"],
                ["signal", "status", "kerze", "wolke", "trend", "setter", "welle"]
            ):
                val = row[db_key]
                if val is not None:
                    attributes[yaml_key].add(str(val))
                    
        return attributes

    def get_by_timestamp(self, signal_name: str, timestamp: str) -> list[dict]:
        """Holt Signale für einen bestimmten Zeitstempel und Strategie."""
        sql = "SELECT * FROM croc WHERE signal = ? AND timestamp = ?"
        rows = self.fetch_all(sql, (signal_name, timestamp))

        results = []
        for row in rows:
            r = dict(row)
            # Parse signal context from data payload
            if r.get("data"):
                try:
                    # In our SignalRepository, 'data' is the full JSON encoded dict
                    payload = json.loads(r["data"])
                    r["context"] = payload.get("context", {})
                except (json.JSONDecodeError, TypeError):
                    r["context"] = {}
            results.append(r)
        return results

    def get_signal_by_id(self, signal_id: int) -> dict:
        sql = "SELECT * FROM view_signals_enriched WHERE id = ?"
        row = self.fetch_one(sql, (signal_id,))
        return dict(row) if row else None

    def get_signals_by_date(
        self, analysis_date: str = None, days_lookback: int = 0
    ) -> list[dict]:
        """
        Liest Signale aus dem View, gefiltert nach Datum.
        """
        # Basis-Query auf den View
        sql = "SELECT * FROM view_signals_enriched WHERE 1=1"
        params = []

        if analysis_date:
            # Exaktes Datum
            sql += " AND date(timestamp) = ?"
            params.append(analysis_date)
        elif days_lookback > 0:
            # Zeitraum (Lookback)
            import pandas as pd

            start_date = (
                pd.Timestamp.now() - pd.Timedelta(days=days_lookback)
            ).strftime("%Y-%m-%d")
            sql += " AND date(timestamp) >= ?"
            params.append(start_date)

        sql += " ORDER BY timestamp DESC"

        rows = self.fetch_all(sql, tuple(params))
        return [dict(row) for row in rows]

    def get_latest_signal_date(self) -> str | None:
        """Returns the iso-date (YYYY-MM-DD) of the latest signal."""
        sql = "SELECT date(timestamp) as d FROM view_signals_enriched ORDER BY timestamp DESC LIMIT 1"
        row = self.fetch_one(sql)
        return row["d"] if row else None

    def get_trade_candidates(
        self,
        strategy_prefix: str | Strategies | list[str | Strategies],
        limit: int = 100,
        statuses: list[TradeStatus | str] | None = None,
    ) -> list[dict]:
        """
        Holt potenzielle Trades aus der 'trades' Tabelle.
        Parst automatisch das 'signal_context' JSON Feld.

        Args:
            strategy_prefix: z.B. 'Croc' oder Strategies.DipBuyer oder eine Liste ['split_target', 'hold_target']
            limit: Anzahl der Ergebnisse
            statuses: Liste von Status-Enums/Strings (Standard: [TradeStatus.CREATED])
        """
        if statuses is None:
            statuses = [TradeStatus.CREATED]

        # Use string conversion as StrEnum handles this correctly
        status_list = [str(s) for s in statuses]
        status_placeholders = ", ".join("?" for _ in status_list)

        if isinstance(strategy_prefix, (list, tuple)):
            strat_list = [str(s).lower() for s in strategy_prefix]
            strat_placeholders = ", ".join("?" for _ in strat_list)
            strategy_filter = f"LOWER(strategy) IN ({strat_placeholders})"
            params = tuple(status_list) + tuple(strat_list) + (limit,)
        else:
            strategy_filter = "LOWER(strategy) LIKE LOWER(?)"
            params = tuple(status_list) + (f"{strategy_prefix}%", limit)

        sql = f"""
            SELECT * FROM trades 
            WHERE status IN ({status_placeholders})
            AND {strategy_filter}
            ORDER BY created_at DESC 
            LIMIT ?
        """
        rows = self.fetch_all(sql, params)

        results = []
        for row in rows:
            r = dict(row)

            # 1. Context (JSON) parsen
            ctx = {}
            if r.get("signal_context"):
                try:
                    raw = r["signal_context"]
                    if isinstance(raw, str):
                        ctx = json.loads(raw)
                    else:
                        ctx = raw
                except Exception:
                    ctx = {}

            r["context"] = ctx

            # 2. Hilfsfelder für das Template
            r["setup_score"] = ctx.get("setup_score", 0)
            r["market_phase"] = ctx.get("market_phase", "-")

            # 3. Datum für Anzeige formatieren (KORRIGIERT)
            # Prio 1: Wenn Entry schon passiert (sollte bei CREATED nicht sein, aber sicherheitshalber)
            display_ts = r.get("entry_date")

            # Prio 2: Signal-Datum aus dem Context (Das echte Datum!)
            if not display_ts:
                display_ts = ctx.get("date") or ctx.get("setup_date")

            # Prio 3: No fallback to created_at (STRICT RULE)
            if not display_ts:
                display_ts = None

            # String Cleaning (Trennzeichen entfernen bei ISO Format)
            r["display_date"] = (
                str(display_ts).split("T")[0].split(" ")[0] if display_ts else "-"
            )

            results.append(r)

        return results
