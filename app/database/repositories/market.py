import logging
import pandas as pd
from .base import BaseRepository

logger = logging.getLogger(__name__)

class MarketRepository(BaseRepository):
    
    def init_schema(self):
        """Erstellt die Tabelle für Marktdaten."""
        with self.session.connect() as conn:
            self.execute("""
                CREATE TABLE IF NOT EXISTS market_prices (
                    symbol TEXT NOT NULL,
                    date TEXT NOT NULL,
                    open REAL,
                    high REAL,
                    low REAL,
                    close REAL,
                    volume INTEGER,
                    provider TEXT NOT NULL DEFAULT 'yahoo',
                    timeframe TEXT NOT NULL DEFAULT '1D',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (symbol, date, timeframe, provider)
                )
            """, conn=conn)

            self.execute("""
                CREATE TABLE IF NOT EXISTS ignored_symbols (
                    symbol TEXT PRIMARY KEY,
                    reason TEXT,
                    ignored_since TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """, conn=conn)            
            
            self.execute("CREATE INDEX IF NOT EXISTS idx_market_date ON market_prices(date)", conn=conn)
            self.execute("CREATE INDEX IF NOT EXISTS idx_market_sym_tf ON market_prices(symbol, timeframe)", conn=conn)

    # --- Blacklist Logic ---
    def ignore_symbol(self, symbol: str, reason: str):
        self.execute("INSERT OR REPLACE INTO ignored_symbols (symbol, reason) VALUES (?, ?)", (symbol, reason))

    def get_ignored_symbols(self) -> set[str]:
        rows = self.fetch_all("SELECT symbol FROM ignored_symbols")
        return {row['symbol'] for row in rows}

    def get_all_known_symbols(self) -> list[str]:
        rows = self.fetch_all("SELECT DISTINCT symbol FROM market_prices")
        return [row['symbol'] for row in rows]

    def get_outdated_symbols(self, reference_date: str, provider: str = 'yahoo') -> list[str]:
        sql = """
            SELECT symbol FROM market_prices
            WHERE provider = ? AND symbol NOT IN (SELECT symbol FROM ignored_symbols)
            GROUP BY symbol HAVING MAX(date) < ?
        """
        rows = self.fetch_all(sql, (provider, reference_date))
        return [row['symbol'] for row in rows]

    def get_symbols_with_missing_history(self, cutoff_date: str) -> list[str]:
        """Findet Symbole, deren Historie erst NACH dem cutoff_date beginnt (zu wenig Daten)."""
        sql = """
            SELECT symbol FROM market_prices
            WHERE symbol NOT IN (SELECT symbol FROM ignored_symbols)
            GROUP BY symbol HAVING MIN(date) > ?
        """
        rows = self.fetch_all(sql, (cutoff_date,))
        return [row['symbol'] for row in rows]

    # --- Data Access Logic (Single Value) ---
    def get_latest_price(self, symbol: str) -> float | None:
        return self.fetch_val("SELECT close FROM market_prices WHERE symbol = ? AND timeframe = '1D' ORDER BY date DESC LIMIT 1", (symbol,))

    def get_trading_days_count(self, symbol: str, start_date: str, end_date: str) -> int:
        s_date = str(start_date).split(" ")[0]
        e_date = str(end_date).split(" ")[0]
        sql = "SELECT COUNT(*) FROM market_prices WHERE symbol = ? AND date >= ? AND date <= ? AND timeframe = '1D'"
        return self.fetch_val(sql, (symbol, s_date, e_date)) or 0
    
    # --- HELPER für Validation ---
    def get_ohlcv(self, symbol: str, date: str) -> dict | None:
        """Holt einen einzelnen OHLCV Datensatz."""
        sql = "SELECT * FROM market_prices WHERE symbol = ? AND date = ? AND timeframe = '1D'"
        row = self.fetch_one(sql, (symbol, date))
        return dict(row) if row else None

    # --- Data Access Logic (Bulk / Pandas) ---
    def get_data_for_lookback(self, start_date: str) -> pd.DataFrame:
        """Lädt alle Daten ab start_date für Pivotisierung."""
        sql = """
            SELECT date, symbol, open, high, low, close, volume
            FROM market_prices
            WHERE date >= ? AND timeframe = '1D'
            ORDER BY date ASC
        """
        with self.session.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=(start_date,))
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"]) # FIX: Type Conversion
            return df

    def get_symbol_history_raw(self, symbol: str, start_date: str) -> pd.DataFrame:
        """Lädt Historie für ein Einzelsymbol (WICHTIG für TradeManager)."""
        sql = """
            SELECT date, open, high, low, close, volume 
            FROM market_prices 
            WHERE symbol = ? AND date >= ? AND timeframe='1D' 
            ORDER BY date ASC
        """
        with self.session.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=(symbol, start_date))
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"]) # FIX: Type Conversion
            return df

    def get_batch_history_raw(self, symbols: list[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Lädt Historie für mehrere Symbole."""
        if not symbols: return pd.DataFrame()
        placeholders = ",".join("?" for _ in symbols)
        sql = f"""
            SELECT symbol, date, open, high, low, close, volume 
            FROM market_prices 
            WHERE symbol IN ({placeholders}) 
            AND date >= ? AND date <= ? AND timeframe='1D' 
            ORDER BY date ASC
        """
        params = symbols + [start_date, end_date]
        with self.session.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"]) # FIX: Type Conversion
            return df

    def save_bulk_prices(self, records: list[tuple]):
        if not records: return
        sql = "INSERT OR REPLACE INTO market_prices (symbol, date, open, high, low, close, volume, provider, timeframe) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        with self.session.connect() as conn:
            conn.executemany(sql, records)