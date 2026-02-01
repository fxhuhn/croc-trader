import json
import logging

from .base import BaseRepository

logger = logging.getLogger(__name__)

class SignalRepository(BaseRepository):
    
    def init_schema(self):
        """Erstellt Tabellen für Signale (croc), Mappings und migriert Altdaten (ohne Löschen)."""
        with self.session.connect() as conn:
            # 1. Neue Haupt-Tabelle 'croc' erstellen
            # WICHTIG: Wir fügen einen UNIQUE Constraint hinzu, damit wir Duplikate beim 
            # mehrfachen Ausführen der Migration verhindern.
            self.execute("""
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
            """, conn=conn)
            
            # 2. KOMPLEXE MIGRATION: Alte 'signals' Struktur in neue 'croc' Struktur
            check_old = self.fetch_one("SELECT name FROM sqlite_master WHERE type='table' AND name='signals'", conn=conn)
            
            if check_old:
                logger.info("🔄 Migration: 'signals' Tabelle gefunden. Prüfe auf neue Daten...")
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
                            symbol = row_dict.pop('symbol')
                            timeframe = row_dict.pop('timeframe')
                            signal = row_dict.pop('signal')
                            timestamp = row_dict.pop('timestamp')
                            exchange = row_dict.get('exchange')
                            created_at = row_dict.get('created_at')

                            # Cleanup: Felder entfernen, die wir nicht im JSON brauchen
                            row_dict.pop('exchange', None)
                            row_dict.pop('created_at', None)
                            row_dict.pop('dist_sma_20', None)
                            row_dict.pop('dist_sma_200', None)
                            
                            # 2. Der REST kommt in das 'data' JSON Feld
                            json_payload = json.dumps(row_dict, ensure_ascii=False)
                            
                            migrated_data.append((
                                symbol, timeframe, signal, timestamp, exchange, json_payload, created_at
                            ))
                        
                        # C) Massen-Insert mit Duplikat-Schutz
                        # 'INSERT OR IGNORE' überspringt Zeilen, die den UNIQUE Constraint verletzen
                        if migrated_data:
                            conn.executemany("""
                                INSERT OR IGNORE INTO croc (symbol, timeframe, signal, timestamp, exchange, data, created_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                            """, migrated_data)
                            
                            # Wir prüfen, wie viele Zeilen tatsächlich eingefügt wurden (changes()) ist hier ungenau bei Batch,
                            # aber das Log ist beruhigend.
                            logger.info(f"✅ Migration abgeschlossen. {len(migrated_data)} Quell-Datensätze verarbeitet (Duplikate wurden ignoriert).")
                    
                    # D) Alte Tabelle behalten (Sicherheitsnetz)
                    # self.execute("DROP TABLE signals", conn=conn)
                    # logger.info("🗑️ Alte 'signals' Tabelle gelöscht.") 
                    
                except Exception as e:
                    logger.error(f"❌ CRITICAL: Fehler bei der Migration von 'signals' zu 'croc': {e}")

            # 3. Exchange Mapping Tabelle
            self.execute("""
                CREATE TABLE IF NOT EXISTS exchange_mappings (
                    symbol TEXT PRIMARY KEY,
                    exchange TEXT NOT NULL
                )
            """, conn=conn)
            
            self.execute("CREATE INDEX IF NOT EXISTS idx_map_sym ON exchange_mappings(symbol)", conn=conn)

            # 4. View aktualisieren (greift nun auf 'croc' zu)
            self.execute("DROP VIEW IF EXISTS view_signals_enriched", conn=conn)
            
            self.execute("""
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
            """, conn=conn)

    def save_signal(self, data: dict[str, Any]) -> int:
        """Speichert Raw Webhook Data in 'croc'."""
        symbol = data.get('symbol') or data.get('ticker', 'UNKNOWN')
        timeframe = data.get('timeframe')
        signal_name = data.get('signal') or data.get('strategy')
        exchange = data.get('exchange')
        timestamp = data.get('timestamp') or data.get('date')
        
        # Auch hier: INSERT OR IGNORE verhindert Crashes bei doppelten Webhooks
        sql = """
            INSERT OR IGNORE INTO croc (symbol, timeframe, signal, timestamp, exchange, data)
            VALUES (?, ?, ?, ?, ?, ?)
        """
        
        cursor = self.execute(sql, (
            symbol, timeframe, signal_name, timestamp, exchange, json.dumps(data)
        ))
        return cursor.lastrowid

    def get_unprocessed_signals(self, limit=100) -> list[dict]:
        """Liest aus dem Enriched View."""
        sql = "SELECT * FROM view_signals_enriched ORDER BY timestamp DESC LIMIT ?"
        rows = self.fetch_all(sql, (limit,))
        return [dict(row) for row in rows]
    
    def get_signal_by_id(self, signal_id: int) -> dict:
        sql = "SELECT * FROM view_signals_enriched WHERE id = ?"
        row = self.fetch_one(sql, (signal_id,))
        return dict(row) if row else None

    def get_signals_by_date(self, analysis_date: str = None, days_lookback: int = 0) -> list[dict]:
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
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=days_lookback)).strftime("%Y-%m-%d")
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

    def get_trade_candidates(self, strategy_prefix: str, limit: int = 100) -> list[dict]:
        """
        Holt potenzielle Trades (Status 'CREATED') aus der 'trades' Tabelle.
        Parst automatisch das 'signal_context' JSON Feld.
        
        Args:
            strategy_prefix: z.B. 'Croc' oder 'DipBuyer'
            limit: Anzahl der Ergebnisse
        """
        # SQL: Wir suchen nach Trades, die erstellt, aber noch nicht aktiv sind
        sql = """
            SELECT * FROM trades 
            WHERE status = 'CREATED' 
            AND strategy LIKE ?
            ORDER BY created_at DESC 
            LIMIT ?
        """
        
        rows = self.fetch_all(sql, (f"{strategy_prefix}%", limit))
        
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
            
            r["ctx"] = ctx
            
            # 2. Hilfsfelder für das Template
            r["setup_score"] = ctx.get("setup_score", 0)
            r["market_phase"] = ctx.get("market_phase", "-")
            
            # 3. Datum für Anzeige formatieren (KORRIGIERT)
            # Prio 1: Wenn Entry schon passiert (sollte bei CREATED nicht sein, aber sicherheitshalber)
            display_ts = r.get("entry_date")
            
            # Prio 2: Signal-Datum aus dem Context (Das echte Datum!)
            if not display_ts and ctx.get("date"):
                display_ts = ctx["date"]
            
            # Prio 3: Fallback auf DB-Erstellung (nur wenn context leer)
            if not display_ts:
                display_ts = r.get("created_at")

            # String Cleaning (Trennzeichen entfernen bei ISO Format)
            r["display_date"] = str(display_ts).split("T")[0].split(" ")[0] if display_ts else "-"
            
            # Überschreibe auch created_at für Views, die dieses Feld direkt nutzen
            r["created_at"] = r["display_date"]
            
            results.append(r)
            
        return results    