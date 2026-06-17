import logging
from functools import lru_cache

import pandas

from ..session import DatabaseSession  # NEU: Zentrale Session

logger = logging.getLogger(__name__)

# Type Alias für die Rückgabestruktur (Open, High, Low, Close, Volume als DataFrames)
type MarketDataDict = dict[str, pandas.DataFrame]


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
        logger.info(
            "[MarketData] Preloading ALL data into memory (Lines: %dd)...", days
        )
        # Reuse existing logic but store it
        # We assume universe is all symbols in DB or we fetch all.
        try:
            start_date = (
                pandas.Timestamp.now() - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")
            with self.session.connect() as connection:
                # Fetch EVERYTHING
                query = "SELECT date, symbol, open, high, low, close, volume FROM market_prices WHERE date >= ? AND timeframe='1D' ORDER BY date ASC"
                df = pandas.read_sql_query(query, connection, params=(start_date,))

            if not df.empty:
                self._in_memory_cache = self._pivot_data(df)
                self._cache_lookback = days
                logger.info("[MarketData] Cache Warm! Loaded %d rows.", len(df))
            else:
                logger.warning("[MarketData] Preload returned empty.")
        except Exception as error:
            logger.error("[MarketData] Preload Failed: %s", error)

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

        start_date = (pandas.Timestamp.now() - pandas.Timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        logger.info("[MarketData] Lade Daten aus DB (Lookback: %dd)...", days)

        try:
            # NEU: Nutzung der zentralen Session
            # yieldet eine konfigurierte Connection (WAL-Mode aktiv)
            with self.session.connect() as connection:
                # Optimierte Query auf '1D' (entspricht neuem Schema)
                query = """
                    SELECT date, symbol, open, high, low, close, volume
                    FROM market_prices
                    WHERE date >= ? AND timeframe = '1D'
                    ORDER BY date ASC
                """
                # Pandas kann direkt mit dem sqlite3 Connection-Objekt arbeiten
                df = pandas.read_sql_query(query, connection, params=(start_date,))
        except Exception as error:
            logger.error("[MarketData] DB Fehler: %s", error)
            return None

        if df.empty:
            logger.warning("[MarketData] Keine Daten gefunden.")
            return None

        return self._pivot_data(df)

    def get_universe_daily_data(
        self, symbols: list[str], days: int
    ) -> MarketDataDict | None:
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
                available_cols = [symbol for symbol in symbols if symbol in df.columns]
                if available_cols:
                    filtered[col] = df[available_cols]  # Slice columns
                else:
                    filtered[col] = pandas.DataFrame(index=df.index)
            return filtered

        start_date = (pandas.Timestamp.now() - pandas.Timedelta(days=days)).strftime(
            "%Y-%m-%d"
        )
        logger.info(
            "[MarketData] Lade Universe-Daten (%d Symbole, %dd)...",
            len(symbols),
            days,
        )

        all_dfs = []
        chunk_size = 500  # Safe limit for SQLite variables

        try:
            with self.session.connect() as connection:
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
                    chunk_df = pandas.read_sql_query(query, connection, params=params)
                    if not chunk_df.empty:
                        all_dfs.append(chunk_df)

        except Exception as error:
            logger.error("[MarketData] Universe Fetch Fehler: %s", error)
            return None

        if not all_dfs:
            logger.warning("[MarketData] Keine Daten für Universe gefunden.")
            return None

        full_df = pandas.concat(all_dfs, ignore_index=True)
        return self._pivot_data(full_df)

    def _pivot_data(self, df: pandas.DataFrame) -> MarketDataDict:
        """Helper to pivot raw dataframe into MarketDataDict structure."""
        # Datentyp-Konvertierung
        df["date"] = pandas.to_datetime(df["date"])

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

    def get_symbol_history(self, symbol: str, days: int = 400) -> pandas.DataFrame:
        """Lädt OHLCV Historie für ein Symbol ohne Pivot. Nutzt Cache wenn möglich."""
        # 1. Try Memory Cache
        if self._in_memory_cache and days <= self._cache_lookback:
            # Reconstruct DataFrame from Pivoted Data
            try:
                data = {}
                # Start date filter
                cutoff = pandas.Timestamp.now() - pandas.Timedelta(days=days)

                has_data = False
                for col in ["open", "high", "low", "close", "volume"]:
                    if symbol in self._in_memory_cache[col].columns:
                        series = self._in_memory_cache[col][symbol]
                        data[col] = series[series.index >= cutoff]
                        has_data = True

                if has_data:
                    df = pandas.DataFrame(data)
                    df.index.name = "date"
                    return df.reset_index()
            except Exception as error:
                logger.warning("Failed to extract %s from cache: %s", symbol, error)
                # Fallback to DB

        # Da BaseRepository SQL erlaubt (Layer-Grenze), ist das hier ok.
        with self.session.connect() as connection:
            # end_date not used in query
            start_date = (
                pandas.Timestamp.now() - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")

            df = pandas.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM market_prices WHERE symbol = ? AND date >= ? AND timeframe='1D' ORDER BY date ASC",
                connection,
                params=(symbol, start_date),
            )
            if not df.empty:
                df["date"] = pandas.to_datetime(df["date"])
            return df

    def get_batch_history(
        self,
        symbols: list[str],
        days: int = 100,
        end_date: str | None = None,
    ) -> dict[str, pandas.DataFrame]:
        """Lädt Historie für mehrere Symbole."""
        if not symbols:
            return {}

        with self.session.connect() as connection:
            if not end_date:
                end_date = pandas.Timestamp.now().strftime("%Y-%m-%d")
            start_date = (
                pandas.Timestamp(end_date) - pandas.Timedelta(days=days)
            ).strftime("%Y-%m-%d")

            placeholders = ",".join("?" for _ in symbols)
            sql = f"""SELECT symbol, date, open, high, low, close, volume FROM market_prices 
                      WHERE symbol IN ({placeholders}) AND date >= ? AND date <= ? AND timeframe='1D' ORDER BY date ASC"""

            df = pandas.read_sql(
                sql, connection, params=symbols + [start_date, end_date]
            )

        result = {}
        if not df.empty:
            df["date"] = pandas.to_datetime(df["date"])
            for symbol, group in df.groupby("symbol"):
                result[symbol] = group
        return result

    def get_available_dates(
        self, start_date: str, end_date: str
    ) -> list[pandas.Timestamp]:
        """Holt eine Liste aller verfügbaren Handelstage im Zeitraum (Fallback für fehlendes SPY)."""
        with self.session.connect() as connection:
            query = """
                SELECT DISTINCT date 
                FROM market_prices 
                WHERE date >= ? AND date <= ? AND timeframe='1D' 
                ORDER BY date ASC
            """
            rows = connection.execute(query, (start_date, end_date)).fetchall()

        return [pandas.Timestamp(row[0]) for row in rows]

    def get_latest_date(self) -> str | None:
        """Gibt das Datum des letzten verfügbaren Datensatzes zurück (Timeframe '1D').

        Dient als 'Global Analysis Date' für den Screener.
        """
        try:
            with self.session.connect() as connection:
                query = "SELECT MAX(date) FROM market_prices WHERE timeframe='1D'"
                row = connection.execute(query).fetchone()
                if row and row[0]:
                    # Schneide Zeitstempel ab falls vorhanden "2026-02-04 00:00:00" -> "2026-02-04"
                    return str(row[0]).split(" ")[0]
        except Exception as error:
            logger.error(
                "[MarketData] Konnte aktuellstes Datum nicht ermitteln: %s", error
            )
        return None
