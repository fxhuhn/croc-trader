import logging
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ...config import settings
from ..database import SignalDatabase
from .strategies.abstract import BaseTradeStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.moonbag import MoonbagStrategy

logger = logging.getLogger(__name__)


class TradeManager:
    """
    Orchestrator für Trade-Management.
    Delegiert Logik an Strategien (Strategy Pattern).
    """

    def __init__(self, db_path: Path, stocks_db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.stocks_db_path = stocks_db_path
        self.telegram = telegram_bot
        self.orders_dir = settings.get_folder("orders")

        # Strategie-Register
        self.strategies = {
            "dipbuyer": DipBuyerStrategy(),
            "moonbag": MoonbagStrategy(),
            "moonshot": MoonbagStrategy(),  # Alias
            "crocsetup": MoonbagStrategy(),  # Alias
        }

    def run_daily_process(self, investment_per_trade: float = 2000.0) -> None:
        """Hauptprozess: Updates & Order Generierung (01:00 Uhr)."""
        try:
            # 1. Self-Healing: Alte Leichen entfernen bevor wir starten
            self._cleanup_stale_trades()

            # 2. Regulärer Prozess
            self._update_positions_status()
            self._export_orders_to_yaml(investment_per_trade)

        except Exception as error:
            logger.error(f"TradeManager Crash: {error}", exc_info=True)
            # WICHTIG: Fehler auch per Telegram melden!
            if self.telegram:
                self.telegram.send(
                    f"⚠️ **CRITICAL ERROR**: TradeManager ist abgestürzt!\n`{error}`"
                )

    def _cleanup_stale_trades(self) -> None:
        """
        Bereinigt Trades, die hängen geblieben sind (z.B. durch Script-Absturz am Vortag).
        Setzt alte 'CREATED' auf 'MISSED' und uralte 'ACTIVE' auf 'CLOSED'.
        """
        try:
            db = SignalDatabase(self.db_path)
            with db._get_conn() as conn:
                # CREATED älter als 3 Tage -> MISSED
                res_created = conn.execute("""
                    UPDATE active_trades
                    SET status = 'MISSED', exit_reason = 'STALE_CLEANUP', closed_at = CURRENT_TIMESTAMP
                    WHERE status = 'CREATED' AND entry_date < date('now', '-3 days')
                """)

                # ACTIVE älter als 14 Tage -> CLOSED (Sicherheitsnetz, falls Moonbag TimeStop versagt)
                res_active = conn.execute("""
                    UPDATE active_trades
                    SET status = 'CLOSED', exit_reason = 'STALE_CLEANUP', closed_at = CURRENT_TIMESTAMP
                    WHERE status = 'ACTIVE' AND entry_date < date('now', '-14 days')
                """)

                conn.commit()

                if res_created.rowcount > 0 or res_active.rowcount > 0:
                    logger.info(
                        f"DB Cleanup: {res_created.rowcount} veraltete Entries, {res_active.rowcount} veraltete Active Trades bereinigt."
                    )

        except Exception as e:
            logger.error(f"Fehler bei DB-Cleanup: {e}")

    def _update_positions_status(self) -> None:
        """Prüft Fills (CREATED) und Exits (ACTIVE)."""
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()
        if not trades:
            return

        # Marktdaten laden
        market_data = self._fetch_market_data_batch(trades)
        alerts = []

        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in market_data:
                continue

            # Strategie finden
            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue

            # Historie für Symbol
            df_hist = market_data[symbol]

            # Dispatch Status
            msg = None
            if trade["status"] == "CREATED":
                # Entry Check (Letzte Kerze)
                last_candle = df_hist.iloc[-1]
                msg = strat_impl.check_entry(trade, last_candle, db)

            elif trade["status"] == "ACTIVE":
                # Active Management
                msg = strat_impl.manage_active_trade(trade, df_hist, db)

            if msg:
                alerts.append(msg)

        if alerts and self.telegram:
            self.telegram.send("⚡ **Trade Updates**\n" + "\n".join(alerts))

    def _export_orders_to_yaml(self, budget: float) -> None:
        """Erstellt Order-File für den nächsten Tag."""
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            return

        market_data = self._fetch_market_data_batch(trades)
        orders_by_date = {}
        today_str = datetime.now().strftime("%Y-%m-%d")

        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in market_data and trade["status"] == "ACTIVE":
                continue  # Active Trades brauchen History

            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue

            # Historie übergeben (kann leer sein bei CREATED wenn nicht benötigt,
            # aber besser konsistent übergeben)
            df_hist = market_data.get(symbol, pd.DataFrame())

            # Guard Clause: Verhindert Abstürze durch leere Dataframes in den Strategien
            if df_hist.empty and trade["status"] == "ACTIVE":
                logger.warning(
                    f"[{symbol}] Keine Historie für aktiven Trade gefunden. Überspringe Order-Generierung."
                )
                continue

            try:
                order = strat_impl.generate_orders(trade, df_hist, budget, db)
                if order:
                    if today_str not in orders_by_date:
                        orders_by_date[today_str] = []
                    orders_by_date[today_str].append(self._dataclass_to_dict(order))
            except Exception as e:
                logger.error(f"Order Gen Error {symbol}: {e}")

        self._write_yaml_files(orders_by_date)

    def _get_strategy(self, trade: dict) -> BaseTradeStrategy | None:
        raw = str(trade.get("strategy", "")).lower().replace(" ", "")
        for key, impl in self.strategies.items():
            if key in raw:
                return impl
        return None

    def _fetch_market_data_batch(self, trades: list[dict]) -> dict[str, pd.DataFrame]:
        symbols = list({t["symbol"] for t in trades})
        if not symbols:
            return {}

        start_date = (datetime.now() - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        placeholders = ",".join("?" for _ in symbols)

        sql = f"""
            SELECT date, symbol, open, high, low, close
            FROM market_prices
            WHERE symbol IN ({placeholders}) AND date >= ? AND timeframe = '1D'
            ORDER BY date ASC
        """
        cache = {}
        try:
            with sqlite3.connect(self.stocks_db_path) as conn:
                df = pd.read_sql_query(sql, conn, params=symbols + [start_date])
                if not df.empty:
                    df["date"] = pd.to_datetime(df["date"])
                    for sym, group in df.groupby("symbol"):
                        cache[sym] = group.reset_index(drop=True)
        except Exception as e:
            logger.error(f"Market Data Error: {e}")

        return cache

    def _write_yaml_files(self, orders_map: dict):
        for date_key, orders in orders_map.items():
            path = self.orders_dir / f"orders_{date_key}.yaml"
            with open(path, "w", encoding="utf-8") as f:
                yaml.dump(orders, f, sort_keys=False)
            if self.telegram:
                self.telegram.send(
                    f"📁 **Orders Generated**: {len(orders)} Orders für {date_key}"
                )

    def _dataclass_to_dict(self, obj):
        if hasattr(obj, "__dataclass_fields__"):
            excluded = ["last_status", "last_update"]
            return {
                k: self._dataclass_to_dict(v)
                for k, v in obj.__dict__.items()
                if v is not None and k not in excluded
            }

        if isinstance(obj, list):
            return [self._dataclass_to_dict(i) for i in obj]

        # --- GLOBALE TYP-KONVERTIERUNG (NumPy Fix) ---
        # Fängt alle NumPy Integer ab (int32, int64 etc.)
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        # Fängt alle NumPy Floats ab (float32, float64 etc.)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return round(float(obj), 2)

        return obj
