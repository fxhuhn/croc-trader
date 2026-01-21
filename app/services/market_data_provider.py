import logging
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Type Alias für die Rückgabestruktur (Open, High, Low, Close, Volume als DataFrames)
type MarketDataDict = dict[str, pd.DataFrame]


class MarketDataProvider:
    """
    Zentraler Service für den Zugriff auf Marktdaten.
    Implementiert Caching, um mehrfache Datenbankzugriffe zu vermeiden.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    @lru_cache(maxsize=4)
    def get_all_daily_data(self, days: int) -> MarketDataDict | None:
        """
        Lädt OHLCV Daten für alle Symbole und pivotiert sie.
        Das Ergebnis wird gecacht (LRU).

        :param days: Anzahl der Tage für den Lookback (z.B. 400).
        """
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        logger.info(f"[MarketData] Lade Daten aus DB (Lookback: {days}d)...")

        try:
            with sqlite3.connect(self.db_path) as conn:
                # Optimierte Query
                query = """
                    SELECT date, symbol, open, high, low, close, volume
                    FROM market_prices
                    WHERE date >= ? AND timeframe = '1D'
                    ORDER BY date ASC
                """
                df = pd.read_sql_query(query, conn, params=(start_date,))
        except sqlite3.Error as e:
            logger.error(f"[MarketData] DB Fehler: {e}")
            return None

        if df.empty:
            logger.warning("[MarketData] Keine Daten gefunden.")
            return None

        # Datentyp-Konvertierung
        df["date"] = pd.to_datetime(df["date"])

        # Pivotisierung für vektorisierte Strategien
        # Erzeugt ein Dict mit DataFrames (Index=Date, Columns=Symbols)
        pivoted_data = {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

        logger.info(f"[MarketData] Daten geladen und pivotisiert ({len(df)} Zeilen).")
        return pivoted_data

    def clear_cache(self) -> None:
        """Leert den internen LRU Cache (z.B. nach einem DB-Update)."""
        self.get_all_daily_data.cache_clear()
        logger.info("[MarketData] Cache geleert.")
