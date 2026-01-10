import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .database import SignalDatabase
from .strategy_database import StrategyDatabase
from .telegram import TelegramBot

logger = logging.getLogger(__name__)


class StrategyEngine:
    def __init__(
        self,
        signals_db_path: Path,
        strategy_db_path: Path,
        telegram_bot: TelegramBot,
        strategies: List[Dict] = None,
    ):
        self.signals_db_path = signals_db_path
        self.strategy_db_path = strategy_db_path
        self.telegram = telegram_bot
        self.strategies = strategies or []

        # Datenbanken initialisieren
        self.strat_db = StrategyDatabase(self.strategy_db_path)
        self.signals_db = SignalDatabase(self.signals_db_path)

    def run_daily_analysis(self, lookback_days=1):
        """
        Führt die Analyse durch und überführt Screener-Treffer in active_trades.
        """
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )
        total_hits = 0

        # 1. DIP BUYER (System Strategie)
        dip_hits = self._process_dip_buyer(start_date)
        total_hits += dip_hits

        # 2. WEBHOOK STRATEGIEN (Dynamisch aus YAML)
        webhook_hits = self._process_yaml_strategies(start_date)
        total_hits += webhook_hits

        if total_hits > 0:
            logger.info(
                f"StrategyEngine: {total_hits} Trades generiert (Dip: {dip_hits}, Webhook: {webhook_hits})."
            )

    def _process_dip_buyer(self, start_date):
        hits = 0
        sql = f"""
            SELECT date, symbol, exchange, timeframe, close, high, entry_limit, atr5
            FROM screener_dip_buyer
            WHERE date >= '{start_date}'
        """

        try:
            with self.signals_db._get_conn() as conn:
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.warning(f"SQL _process_dip_buyer failed: {e}")
            return 0

        if df.empty:
            return 0
        df.columns = df.columns.str.lower()

        for _, row in df.iterrows():
            # Erstelle den operativen Trade in active_trades
            self.signals_db.add_trade(
                symbol=row["symbol"],
                entry_date=row["date"],
                entry_price=row["entry_limit"],
                atr_at_entry=row["atr5"],
                quantity=1,
                strategy="DipBuyer",
            )
            hits += 1
        return hits

    def _process_yaml_strategies(self, start_date):
        hits = 0

        # Wir laden alle Strategie-Treffer aus der Screener-Tabelle
        sql = f"""
            SELECT date, symbol, exchange, timeframe, signal, strategy, close, high, low, rsi, sma_200
            FROM screener_webhook
            WHERE date >= '{start_date}'
        """
        try:
            with self.signals_db._get_conn() as conn:
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.warning(f"SQL _process_yaml_strategies failed: {e}")
            return 0

        if df.empty:
            return 0
        df.columns = df.columns.str.lower()

        for _, row in df.iterrows():
            # WICHTIG: Hier übergeben wir den Strategienamen aus der DB!
            # Bei Webhook-Strategien ist der Entry oft der Close des Signals (oder ein Breakout High)
            # Hier vereinfacht als Close.

            self.signals_db.add_trade(
                symbol=row["symbol"],
                entry_date=row["date"],
                entry_price=row["close"],
                atr_at_entry=0,  # Webhooks haben meist keine ATR vorberechnet, daher 0
                quantity=1,
                strategy=row["strategy"],  # Name z.B. "Rot Lolly"
            )
            hits += 1
        return hits

    def send_telegram_report(self):
        # Optional: Reporting Logik
        pass
