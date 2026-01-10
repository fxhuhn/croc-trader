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
        # Strategien werden jetzt injiziert, nicht mehr selbst geladen
        self.strategies = strategies or []
        self.strat_db = StrategyDatabase(self.strategy_db_path)

    def run_daily_analysis(self, lookback_days=1):
        """
        Führt die Analyse durch.
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
        sig_db = SignalDatabase(self.signals_db_path)
        hits = 0
        strategy_name = "Dip Buyer"

        sql = f"""
            SELECT date, symbol, exchange, timeframe, close, high, entry_limit, atr5
            FROM screener_dip_buyer
            WHERE date >= '{start_date}'
        """

        try:
            with sig_db._get_conn() as conn:
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.warning(f"SQL _process_dip_buyer failed: {e}")
            return 0

        if df.empty:
            return 0
        df.columns = df.columns.str.lower()

        for _, row in df.iterrows():
            if self._create_trade_entry(row, strategy_name, "dip_buyer"):
                hits += 1
        return hits

    def _process_yaml_strategies(self, start_date):
        if not self.strategies:
            return 0

        sig_db = SignalDatabase(self.signals_db_path)
        hits = 0

        for strat in self.strategies:
            name = strat.get("name")
            source = strat.get("source", "webhook")
            # Wir verarbeiten hier nur Webhook-Strategien, da DipBuyer oben separat läuft
            if source != "webhook":
                continue

            sql = f"""
                SELECT date, symbol, exchange, timeframe, signal, close, high, low, rsi, sma_200
                FROM screener_webhook
                WHERE date >= '{start_date}' AND strategy = '{name}'
            """
            try:
                with sig_db._get_conn() as conn:
                    df = pd.read_sql_query(sql, conn)
            except Exception as e:
                logger.warning(f"SQL _process_yaml_strategies failed: {e}")
                continue

            if df.empty:
                continue
            df.columns = df.columns.str.lower()

            for _, row in df.iterrows():
                if self._create_trade_entry(row, name, "webhook"):
                    hits += 1
        return hits

    def _create_trade_entry(self, row, strategy_name, source_type):
        """Erstellt das Trade-Objekt und speichert es."""
        try:
            limit_lmt = None
            limit_stp = None
            stop_loss = 0.0
            take_profit = 0.0
            entry_price_for_risk = 0.0

            # --- FALL A: WEBHOOK (Breakout -> Stop Buy) ---
            if source_type == "webhook":
                limit_stp = row["high"]
                stop_loss = row["low"]

                if limit_stp == 0 or stop_loss == 0:
                    limit_stp = row["close"]
                    stop_loss = row["close"] * 0.95

                entry_price_for_risk = limit_stp
                risk = entry_price_for_risk - stop_loss
                if risk <= 0:
                    risk = entry_price_for_risk * 0.01
                take_profit = entry_price_for_risk + (2 * risk)

            # --- FALL B: DIP BUYER (Pullback -> Limit Buy) ---
            elif source_type == "dip_buyer":
                limit_lmt = row["entry_limit"]
                entry_price_for_risk = limit_lmt
                stop_loss = 0.0
                take_profit = row["high"]
                risk = entry_price_for_risk

            # Positionsgröße (Risk Management)
            qty = 1
            if risk > 0:
                qty = int(100 / risk)  # 100$ Risk Unit
            if qty < 1:
                qty = 1

            trade_data = {
                "date": row["date"],
                "timeframe": row.get("timeframe", "D1"),
                "symbol": row["symbol"],
                "strategy": strategy_name,
                "limit_stp": round(limit_stp, 2) if limit_stp else None,
                "limit_lmt": round(limit_lmt, 2) if limit_lmt else None,
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "qty": qty,
                "status": "PENDING",
            }

            self.strat_db.save_trade(trade_data)
            return True

        except Exception as e:
            logger.error(f"Fehler bei Trade-Erstellung ({strategy_name}): {e}")
            return False

    def send_telegram_report(self):
        df = self.strat_db.get_latest_trades(limit=20)
        if df.empty:
            return

        display_df = df[
            ["symbol", "date", "strategy", "limit_stp", "limit_lmt", "take_profit"]
        ].copy()

        display_df["Entry"] = display_df.apply(
            lambda x: f"STOP {x['limit_stp']:.2f}"
            if pd.notnull(x["limit_stp"])
            else f"LMT {x['limit_lmt']:.2f}",
            axis=1,
        )

        final_df = display_df[["date", "symbol", "strategy", "Entry"]].copy()
        self.telegram.send_dataframe(final_df, title="📋 Strategie Plan")
