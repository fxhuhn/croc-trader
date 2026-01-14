import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .database import SignalDatabase
from .strategy_database import StrategyDatabase
from .telegram import TelegramBot

logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Verarbeitet Signale aus den Screener-Tabellen und überführt sie in
    ausführbare Trades (active_trades).

    FILTER-LOGIK:
    - Nur 'DipBuyer' und 'Moonbag/Moonshot' Strategien werden übertragen.
    - Alle anderen (Webhook, etc.) werden ignoriert.
    """

    def __init__(
        self,
        signals_db_path: Path,
        strategy_db_path: Path,
        telegram_bot: TelegramBot,
        strategies: List[Dict] = None,
    ):
        self.signals_db = SignalDatabase(signals_db_path)
        self.strat_db = StrategyDatabase(strategy_db_path)
        self.telegram = telegram_bot
        self.strategies = strategies or []

    def run_daily_analysis(self, lookback_days: int = 1) -> None:
        """
        Hauptprozess: Scannt Screener nach erlaubten Strategien.
        """
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        total_hits = 0

        # 1. DIP BUYER (Erlaubt)
        total_hits += self._process_dip_buyer(start_date)

        # 2. WEBHOOK STRATEGIEN (DEAKTIVIERT)
        # Gemäß Anforderung werden diese Strategien ignoriert und nicht übertragen.
        # total_hits += self._process_webhook_strategies(start_date)

        # 3. CROC SETUP (Nur Moonbag/Moonshot)
        total_hits += self._process_croc_setup(start_date)

        if total_hits > 0:
            logger.info(
                f"StrategyEngine: {total_hits} Trades verarbeitet (Nur DipBuyer & Moonshot)."
            )

    def _process_dip_buyer(self, start_date: str) -> int:
        sql = f"""
            SELECT date, symbol, entry_limit, atr5
            FROM screener_dip_buyer
            WHERE date >= '{start_date}'
        """
        return self._transfer_to_active_trades(sql, strategy_override="DipBuyer")

    def _process_webhook_strategies(self, start_date: str) -> int:
        """Deaktiviert - wird aktuell nicht aufgerufen."""
        sql = f"""
            SELECT date, symbol, close as entry_price, strategy, 0 as atr5
            FROM screener_webhook
            WHERE date >= '{start_date}'
        """
        return self._transfer_to_active_trades(sql, strategy_column="strategy")

    def _process_croc_setup(self, start_date: str) -> int:
        """
        Verarbeitet Croc-Setups mit striktem Filter auf Moonbag/Moonshot.
        Entry ist das High der Signalkerze (Stop Buy).
        """
        # Wir suchen nach "Moonbag" (DB-Wert) oder "Moonshot" (User-Wording)
        sql = f"""
            SELECT date, symbol, high as entry_price, recommended_strategy as strategy, 0 as atr5
            FROM screener_croc
            WHERE date >= '{start_date}'
            AND (
                recommended_strategy LIKE '%Moonbag%'
                OR recommended_strategy LIKE '%Moonshot%'
            )
        """
        return self._transfer_to_active_trades(sql, strategy_column="strategy")

    def _transfer_to_active_trades(
        self, sql: str, strategy_override: str = None, strategy_column: str = None
    ) -> int:
        """
        Überträgt Signale in active_trades.
        Führt ein UPDATE durch, falls der Trade schon existiert (z.B. Strategienamen-Wechsel).
        """
        hits = 0
        try:
            # Wir nutzen direktes SQLite für präzise Kontrolle über Updates
            with sqlite3.connect(self.signals_db.db_path) as conn:
                conn.row_factory = sqlite3.Row
                df = pd.read_sql_query(sql, conn)

                if df.empty:
                    return 0

                df.columns = df.columns.str.lower()
                cursor = conn.cursor()

                for _, row in df.iterrows():
                    symbol = row["symbol"]
                    date_str = row["date"]

                    # Preis & ATR sicherstellen
                    price = row.get("entry_price") or row.get("entry_limit") or 0.0
                    atr = row.get("atr5") or 0.0

                    # Strategienamen ermitteln
                    if strategy_override:
                        strat_name = strategy_override
                    elif strategy_column and strategy_column in row:
                        strat_name = row[strategy_column]
                    else:
                        strat_name = "Unknown"

                    # 1. Prüfen ob Trade existiert
                    check_sql = "SELECT id, strategy, entry_price FROM active_trades WHERE symbol = ? AND entry_date = ?"
                    existing = cursor.execute(check_sql, (symbol, date_str)).fetchone()

                    if existing:
                        # 2. UPDATE: Wenn Strategie abweicht (z.B. vorher 'CrocSetup', jetzt 'Moonbag (TP5)')
                        trade_id, old_strat, old_price = existing

                        # Nur aktualisieren, wenn noch im Status CREATED (oder Strategie falsch)
                        if old_strat != strat_name or float(old_price) != float(price):
                            update_sql = """
                                UPDATE active_trades
                                SET strategy = ?, entry_price = ?, atr_at_entry = ?
                                WHERE id = ? AND status = 'CREATED'
                            """
                            cursor.execute(
                                update_sql, (strat_name, price, atr, trade_id)
                            )
                            if cursor.rowcount > 0:
                                logger.info(
                                    f"🔄 Trade Update {symbol}: {old_strat} -> {strat_name}"
                                )
                                hits += 1
                    else:
                        # 3. INSERT: Neu anlegen
                        # FIX: Keine 'id' (Auto) und keine 'log' Spalte mehr angeben!
                        insert_sql = """
                            INSERT INTO active_trades
                            (symbol, entry_date, entry_price, atr_at_entry, quantity, status, strategy)
                            VALUES (?, ?, ?, ?, ?, 'CREATED', ?)
                        """
                        try:
                            cursor.execute(
                                insert_sql,
                                (symbol, date_str, price, atr, 1, strat_name),
                            )
                            hits += 1
                        except sqlite3.IntegrityError:
                            pass

                conn.commit()

        except Exception as e:
            logger.error(f"Fehler beim Transfer zu active_trades: {e}")

        return hits

    def send_telegram_report(self):
        pass
