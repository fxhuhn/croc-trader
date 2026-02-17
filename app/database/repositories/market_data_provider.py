import logging
from functools import lru_cache

import pandas as pd

from ..session import DatabaseSession  # NEU: Zentrale Session

logger = logging.getLogger(__name__)

# Type Alias für die Rückgabestruktur (Open, High, Low, Close, Volume als DataFrames)
type MarketDataDict = dict[str, pd.DataFrame]


class MarketDataProvider:
    """
    Zentraler Service für den Zugriff auf Marktdaten (High-Performance Read).
    Nutzt DatabaseSession für konsistente Verbindungen und WAL-Mode.
    """

    def __init__(self, session: DatabaseSession) -> None:
        # Wir speichern die Session, nicht mehr den Pfad
        self.session = session
        self._in_memory_cache: MarketDataDict | None = None
        self._cache_lookback = 0

    def preload_all_data(self, days: int = 1000) -> None:
        """
        Loads ALL market data into memory for high-speed access.
        Critical for backtesting performance.
        """
        logger.info(f"[MarketData] Preloading ALL data into memory (Lines: {days}d)...")
        # Reuse existing logic but store it
        # We assume universe is all symbols in DB or we fetch all.
        try:
             start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
             with self.session.connect() as conn:
                 # Fetch EVERYTHING
                 query = "SELECT date, symbol, open, high, low, close, volume FROM market_prices WHERE date >= ? AND timeframe='1D' ORDER BY date ASC"
                 df = pd.read_sql_query(query, conn, params=(start_date,))
                 
             if not df.empty:
                 self._in_memory_cache = self._pivot_data(df)
                 self._cache_lookback = days
                 logger.info(f"[MarketData] Cache Warm! Loaded {len(df)} rows.")
             else:
                 logger.warning("[MarketData] Preload returned empty.")
        except Exception as e:
            logger.error(f"[MarketData] Preload Failed: {e}")

    @lru_cache(maxsize=4)
    def get_all_daily_data(self, days: int) -> MarketDataDict | None:
        """
        Lädt OHLCV Daten für alle Symbole und pivotiert sie.
        Das Ergebnis wird gecacht (LRU).
        
        :param days: Anzahl der Tage für den Lookback (z.B. 400).
        """
        # Check explicit memory cache first
        if self._in_memory_cache and days <= self._cache_lookback:
            return self._in_memory_cache

        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info(f"[MarketData] Lade Daten aus DB (Lookback: {days}d)...")

        try:
            # NEU: Nutzung der zentralen Session
            # yieldet eine konfigurierte Connection (WAL-Mode aktiv)
            with self.session.connect() as conn:
                # Optimierte Query auf '1D' (entspricht neuem Schema)
                query = """
                    SELECT date, symbol, open, high, low, close, volume
                    FROM market_prices
                    WHERE date >= ? AND timeframe = '1D'
                    ORDER BY date ASC
                """
                # Pandas kann direkt mit dem sqlite3 Connection-Objekt arbeiten
                df = pd.read_sql_query(query, conn, params=(start_date,))
        except Exception as e:
            logger.error(f"[MarketData] DB Fehler: {e}")
            return None

        if df.empty:
            logger.warning("[MarketData] Keine Daten gefunden.")
            return None

        return self._pivot_data(df)

    def get_universe_daily_data(self, symbols: list[str], days: int) -> MarketDataDict | None:
        """
        Lädt Daten nur für eine spezifische Liste von Symbolen (Pre-Filtering).
        Nutzt Memory Cache wenn verfügbar.
        """
        if not symbols:
            return None

        # 1. Try Memory Cache
        if self._in_memory_cache and days <= self._cache_lookback:
            # Filter the pivoted cache for requested symbols
            filtered = {}
            for col, df in self._in_memory_cache.items():
                available_cols = [s for s in symbols if s in df.columns]
                if available_cols:
                    filtered[col] = df[available_cols] # Slice columns
                else:
                     filtered[col] = pd.DataFrame(index=df.index)
            return filtered

        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info(f"[MarketData] Lade Universe-Daten ({len(symbols)} Symbole, {days}d)...")
        
        all_dfs = []
        chunk_size = 500 # Safe limit for SQLite variables
        
        try:
            with self.session.connect() as conn:
                for i in range(0, len(symbols), chunk_size):
                    chunk = symbols[i : i + chunk_size]
                    placeholders = ",".join("?" for _ in chunk)
                    query = f"""
                        SELECT date, symbol, open, high, low, close, volume
                        FROM market_prices
                        WHERE symbol IN ({placeholders})
                          AND date >= ? 
                          AND timeframe = '1D'
                        ORDER BY date ASC
                    """
                    # Params: symbols + start_date
                    params = tuple(chunk) + (start_date,)
                    chunk_df = pd.read_sql_query(query, conn, params=params)
                    if not chunk_df.empty:
                         all_dfs.append(chunk_df)
                         
        except Exception as e:
            logger.error(f"[MarketData] Universe Fetch Fehler: {e}")
            return None
            
        if not all_dfs:
            logger.warning("[MarketData] Keine Daten für Universe gefunden.")
            return None
            
        full_df = pd.concat(all_dfs, ignore_index=True)
        return self._pivot_data(full_df)

    def _pivot_data(self, df: pd.DataFrame) -> MarketDataDict:
        """Helper to pivot raw dataframe into MarketDataDict structure."""
        # Datentyp-Konvertierung
        df["date"] = pd.to_datetime(df["date"])

        # Pivotisierung für vektorisierte Strategien
        # Erzeugt ein Dict mit DataFrames (Index=Date, Columns=Symbols)
        pivoted_data = {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

        # logger.info(f"[MarketData] Daten geladen und pivotisiert ({len(df)} Zeilen).")
        return pivoted_data

    def clear_cache(self) -> None:
        """Leert den internen LRU Cache (z.B. nach einem DB-Update)."""
        self.get_all_daily_data.cache_clear()
        self._in_memory_cache = None
        logger.info("[MarketData] Cache geleert.")

    def get_symbol_history(self, symbol: str, days: int = 400) -> pd.DataFrame:
        """Lädt OHLCV Historie für ein Symbol ohne Pivot. Nutzt Cache wenn möglich."""
        # 1. Try Memory Cache
        if self._in_memory_cache and days <= self._cache_lookback:
            # Reconstruct DataFrame from Pivoted Data
            try:
                data = {}
                # Start date filter
                cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
                
                has_data = False
                for col in ["open", "high", "low", "close", "volume"]:
                    if symbol in self._in_memory_cache[col].columns:
                         series = self._in_memory_cache[col][symbol]
                         data[col] = series[series.index >= cutoff]
                         has_data = True
                
                if has_data:
                    df = pd.DataFrame(data)
                    df.index.name = "date"
                    return df.reset_index()
            except Exception as e:
                logger.warning(f"Failed to extract {symbol} from cache: {e}")
                # Fallback to DB
        
        # Da BaseRepository SQL erlaubt (Layer-Grenze), ist das hier ok.
        with self.session.connect() as conn:
            # end_date not used in query
            start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM market_prices WHERE symbol = ? AND date >= ? AND timeframe='1D' ORDER BY date ASC",
                conn, params=(symbol, start_date)
            )
            if not df.empty:
                df["date"] = pd.to_datetime(df["date"])
            return df

    def get_batch_history(self, symbols: list, days: int = 100, end_date: str = None) -> dict:
        """Lädt Historie für mehrere Symbole."""
        if not symbols:
            return {}
        
        with self.session.connect() as conn:
            if not end_date:
                end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
            start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
            
            placeholders = ",".join("?" for _ in symbols)
            sql = f"""SELECT symbol, date, open, high, low, close, volume FROM market_prices 
                      WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ? AND timeframe='1D' ORDER BY date ASC"""
            
            df = pd.read_sql(sql, conn, params=symbols + [start_date, end_date])
            
        res = {}
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            for sym, grp in df.groupby("symbol"):
                res[sym] = grp
        return res

    def get_available_dates(self, start_date: str, end_date: str) -> list[pd.Timestamp]:
        """Holt eine Liste aller verfügbaren Handelstage im Zeitraum (Fallback für fehlendes SPY)."""
        with self.session.connect() as conn:
            query = """
                SELECT DISTINCT date 
                FROM market_prices 
                WHERE date >= ? AND date <= ? AND timeframe='1D' 
                ORDER BY date ASC
            """
            rows = conn.execute(query, (start_date, end_date)).fetchall()
            
        return [pd.Timestamp(r[0]) for r in rows]

    def get_latest_date(self) -> str | None:
        """
        Gibt das Datum des letzten verfügbaren Datensatzes zurück (Timeframe '1D').
        Dient als 'Global Analysis Date' für den Screener.
        """
        try:
            with self.session.connect() as conn:
                query = "SELECT MAX(date) FROM market_prices WHERE timeframe='1D'"
                row = conn.execute(query).fetchone()
                if row and row[0]:
                    # Schneide Zeitstempel ab falls vorhanden "2026-02-04 00:00:00" -> "2026-02-04"
                    return str(row[0]).split(" ")[0]
        except Exception as e:
            logger.error(f"[MarketData] Konnte aktuellstes Datum nicht ermitteln: {e}")
        return None