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
from .strategies.split_target import SplitTargetStrategy

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
        self.strategies: dict[str, BaseTradeStrategy] = {
            "dipbuyer": DipBuyerStrategy(),
            "moonbag": MoonbagStrategy(),
            "moonshot": MoonbagStrategy(),  # Alias
            "crocsetup": MoonbagStrategy(),  # Alias
            # Mapping für die Split Strategie
            "split": SplitTargetStrategy(),
            "webhook": SplitTargetStrategy(),
            "tp1_tp3": SplitTargetStrategy(),
        }

    def run_daily_process(self, investment_per_trade: float = 2000.0) -> None:
        """Hauptprozess: Updates & Order Generierung (01:00 Uhr)."""
        try:
            self._cleanup_stale_trades()
            self._update_positions_status()
            self._export_orders_to_yaml(investment_per_trade)

        except Exception as error:
            logger.error(f"TradeManager Crash: {error}", exc_info=True)
            if self.telegram:
                self.telegram.send(
                    f"⚠️ **CRITICAL ERROR**: TradeManager ist abgestürzt!\n`{error}`"
                )

    def run_backfill(self, default_budget: float = 2000.0) -> dict:
        """
        Simuliert die Vergangenheit.
        1. Prüft 'CREATED' Trades auf Entry (Next Day Logic).
        2. Prüft 'ACTIVE' Trades auf verpasste Exits (Full Replay).
        """
        logger.info("Starte Smart-Backfill-Prozess...")

        stats = {
            "processed": 0,
            "filled": 0,
            "missed": 0,
            "closed_active": 0,  # Neu
            "errors": 0,
            "skipped_no_data": 0,
        }
        db = SignalDatabase(self.db_path)

        # 1. Trades laden: CREATED (alt) ODER ACTIVE (alle)
        with db._get_conn() as conn:
            sql = """
                SELECT * FROM active_trades
                WHERE (status = 'CREATED' AND entry_date < date('now'))
                   OR status = 'ACTIVE'
                ORDER BY entry_date ASC
            """
            trades = [dict(row) for row in conn.execute(sql).fetchall()]

        if not trades:
            logger.info("Keine Trades für Backfill gefunden.")
            return stats

        market_cache = {}

        for trade in trades:
            stats["processed"] += 1
            symbol = trade["symbol"]
            status = trade["status"]
            entry_date_str = trade["entry_date"]

            try:
                strat_impl = self._get_strategy(trade)
                if not strat_impl:
                    continue

                # Marktdaten laden
                if symbol not in market_cache:
                    with sqlite3.connect(self.stocks_db_path) as conn:
                        df = pd.read_sql_query(
                            "SELECT date, open, high, low, close FROM market_prices WHERE symbol = ? AND timeframe='1D' ORDER BY date ASC",
                            conn,
                            params=(symbol,),
                        )
                        df["date"] = pd.to_datetime(df["date"])
                        market_cache[symbol] = df

                df_hist = market_cache[symbol]
                if df_hist.empty:
                    stats["skipped_no_data"] += 1
                    continue

                entry_ts = pd.Timestamp(entry_date_str)
                future_candles = df_hist[df_hist["date"] > entry_ts].sort_values("date")

                if future_candles.empty:
                    continue

                # === FALL A: Trade ist noch CREATED (Entry Check) ===
                if status == "CREATED":
                    execution_candle = future_candles.iloc[0]  # Erster Tag nach Signal

                    result_msg = strat_impl.check_entry(trade, execution_candle, db)

                    if result_msg and "FILLED" in result_msg:
                        stats["filled"] += 1
                        # Quantity Fix (wie gehabt)
                        try:
                            entry_price = float(trade["entry_price"])
                            new_qty = 1
                            if isinstance(strat_impl, DipBuyerStrategy):
                                new_qty = int(default_budget / entry_price)
                            elif isinstance(strat_impl, SplitTargetStrategy):
                                # Split: Risk based on Signal Candle Low
                                signal_candle_row = df_hist[df_hist["date"] == entry_ts]
                                sl_price = entry_price * 0.99
                                if not signal_candle_row.empty:
                                    sl_price = float(signal_candle_row.iloc[0]["low"])

                                if sl_price >= entry_price:
                                    sl_price = entry_price * 0.99
                                risk = entry_price - sl_price
                                if risk > 0:
                                    new_qty = int(100.0 / risk)
                                else:
                                    new_qty = 2
                            else:
                                new_qty = int(default_budget / entry_price)

                            new_qty = max(1, new_qty)
                            db.update_trade_quantity(trade["id"], new_qty)
                            trade["quantity"] = new_qty
                            trade["status"] = (
                                "ACTIVE"  # Lokal updaten für sofortige Simulation unten
                            )

                        except Exception as e:
                            logger.error(f"Qty Calc Error {symbol}: {e}")
                    else:
                        if not result_msg:
                            db.update_trade_status(
                                trade["id"], "MISSED", "BACKFILL_NO_FILL"
                            )
                        stats["missed"] += 1
                        continue  # Weiter zum nächsten Trade

                # === FALL B: Trade ist ACTIVE (Simulation Management) ===
                # (Läuft auch durch, wenn er gerade eben oben erst auf ACTIVE gesetzt wurde)

                # Wir simulieren JEDEN Tag ab dem Entry (bzw. ab Execution)
                # Finden wo der Entry war (Execution Candle)
                # Falls wir schon mitten drin sind, starten wir beim ersten Tag nach Entry

                # KORREKTUR: Unused variable removed
                if not future_candles.empty:
                    # Wir suchen den Index der ersten Kerze nach Entry Date
                    # Das ist future_candles.iloc[0]
                    start_idx_in_hist = df_hist.index.get_loc(
                        future_candles.iloc[0].name
                    )
                    candles_to_simulate = df_hist.iloc[start_idx_in_hist:]
                else:
                    candles_to_simulate = pd.DataFrame()

                for _, current_candle in candles_to_simulate.iterrows():
                    # Historie wächst mit jedem Tag
                    simulated_history = df_hist[
                        df_hist["date"] <= current_candle["date"]
                    ]

                    # Management Logik aufrufen
                    # Hinweis: Da wir "ACTIVE" Trades aus der DB geladen haben, könnte 'exit_reason'
                    # schon "TP1_LOCKED" enthalten. Das ist gut! Die Strategie macht da weiter wo sie war.

                    # Wir müssen den aktuellen Trade-Status aus der DB neu lesen, falls er sich
                    # in der Loop geändert hat (z.B. TP1 Update)
                    # Performance-Optimierung: Wir updaten das lokale 'trade' dict manuell,
                    # wenn die Strategie Strings wie "TP1 HIT" zurückgibt, um SQL-Reads zu sparen?
                    # Besser: Wir vertrauen der Strategie, dass sie DB updates macht,
                    # aber für den Loop müssen wir wissen, ob CLOSED.

                    mgmt_msg = strat_impl.manage_active_trade(
                        trade, simulated_history, db
                    )

                    if mgmt_msg:
                        # Prüfen ob geschlossen
                        if (
                            "EXIT" in mgmt_msg
                            or "STOP" in mgmt_msg
                            or "WIN" in mgmt_msg
                        ):
                            logger.info(f"[{symbol}] BACKFILL CATCH-UP: {mgmt_msg}")
                            if (
                                trade["status"] == "ACTIVE"
                            ):  # Nur zählen wenn vorher aktiv
                                stats["closed_active"] += 1
                            break  # Trade vorbei

                        # Wenn TP1 Hit: Wir müssen das lokale 'trade' Objekt updaten,
                        # damit im nächsten Loop-Durchlauf (nächster Tag) das 'TP1_LOCKED' bekannt ist!
                        if "TP1" in mgmt_msg and "LOCKED" in mgmt_msg:
                            # Wir lesen exit_reason neu aus DB oder parsen die Message
                            # Einfacher: Kurz neu laden
                            with db._get_conn() as c:
                                row = c.execute(
                                    "SELECT exit_reason FROM active_trades WHERE id=?",
                                    (trade["id"],),
                                ).fetchone()
                                if row:
                                    trade["exit_reason"] = row[0]

            except Exception as e:
                logger.error(f"Fehler Backfill {symbol}: {e}", exc_info=True)
                stats["errors"] += 1

        logger.info(f"Backfill beendet: {stats}")
        return stats

    def _cleanup_stale_trades(self) -> None:
        """
        Bereinigt Trades, die hängen geblieben sind.
        Setzt alte 'CREATED' auf 'MISSED' und uralte 'ACTIVE' auf 'CLOSED'.
        """
        try:
            db = SignalDatabase(self.db_path)
            with db._get_conn() as conn:
                # CREATED älter als 3 Tage -> MISSED
                conn.execute("""
                    UPDATE active_trades
                    SET status = 'MISSED', exit_reason = 'STALE_CLEANUP', closed_at = CURRENT_TIMESTAMP
                    WHERE status = 'CREATED' AND entry_date < date('now', '-3 days')
                """)

                # ACTIVE älter als 20 Tage -> CLOSED (Sicherheitsnetz)
                conn.execute("""
                    UPDATE active_trades
                    SET status = 'CLOSED', exit_reason = 'STALE_CLEANUP', closed_at = CURRENT_TIMESTAMP
                    WHERE status = 'ACTIVE' AND entry_date < date('now', '-20 days')
                """)
                conn.commit()
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

            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue

            # Historie für Symbol
            df_hist = market_data[symbol]
            if df_hist.empty:
                continue

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
            if symbol not in market_data:
                continue

            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue

            # Historie übergeben
            df_hist = market_data.get(symbol, pd.DataFrame())

            if df_hist.empty and trade["status"] == "ACTIVE":
                continue

            try:
                order = strat_impl.generate_orders(trade, df_hist, budget, db)
                if order:
                    if today_str not in orders_by_date:
                        orders_by_date[today_str] = []
                    orders_by_date[today_str].append(self._dataclass_to_dict(order))
            except Exception as e:
                logger.error(f"Order Gen Error {symbol}: {e}", exc_info=True)

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

        # Etwas mehr Puffer für Indikatoren/TimeStop Checks
        start_date = (datetime.now() - pd.Timedelta(days=40)).strftime("%Y-%m-%d")
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

        # NumPy Fix
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return round(float(obj), 2)

        return obj
