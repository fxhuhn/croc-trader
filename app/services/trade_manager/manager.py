import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

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
    Stellt sicher, dass Backfill und Live-Betrieb denselben Code nutzen.
    """

    def __init__(self, db_path: Path, stocks_db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.stocks_db_path = stocks_db_path
        self.telegram = telegram_bot
        self.orders_dir = settings.get_folder("orders")

        # Datenbank-Schema prüfen und migrieren (Signal Date)
        self._ensure_db_schema()

        # Strategie-Register
        self.strategies: dict[str, BaseTradeStrategy] = {
            "dipbuyer": DipBuyerStrategy(),
            "moonbag": MoonbagStrategy(),
            "moonshot": MoonbagStrategy(),  # Alias
            "crocsetup": MoonbagStrategy(),  # Alias
            "split": SplitTargetStrategy(),
            "webhook": SplitTargetStrategy(),
            "tp1_tp3": SplitTargetStrategy(),
        }

    def _ensure_db_schema(self):
        """
        Prüft, ob die Spalte 'signal_date' existiert.
        Falls nicht, wird sie angelegt und initial befüllt.
        """
        try:
            db = SignalDatabase(self.db_path)
            with db._get_conn() as conn:
                # Prüfen ob Spalte existiert
                cursor = conn.execute("PRAGMA table_info(active_trades)")
                columns = [row["name"] for row in cursor.fetchall()]

                if "signal_date" not in columns:
                    logger.info(
                        "Migration: Füge Spalte 'signal_date' zu active_trades hinzu..."
                    )
                    conn.execute(
                        "ALTER TABLE active_trades ADD COLUMN signal_date TEXT"
                    )

                    # Migration: Bestehende Entry Dates als Signal Date setzen
                    conn.execute(
                        "UPDATE active_trades SET signal_date = entry_date WHERE signal_date IS NULL"
                    )
                    conn.commit()
                    logger.info("Migration 'signal_date' erfolgreich abgeschlossen.")
                else:
                    # Self-Healing: Falls Null-Werte existieren (z.B. durch Screener ohne Update)
                    conn.execute(
                        "UPDATE active_trades SET signal_date = entry_date WHERE signal_date IS NULL AND entry_date IS NOT NULL"
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"DB Schema Check Failed: {e}")

    def run_daily_process(self, investment_per_trade: float = 2000.0) -> None:
        """Hauptprozess: Cleanup, Updates & Order Generierung (01:00 Uhr)."""
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
        Simuliert die Vergangenheit mit exakt derselben Logik wie der Live-Betrieb.
        Nutzt Signal Date für Entry-Prüfung und aktualisiert Entry Date bei Fill.
        """
        logger.info("Starte Smart-Backfill-Prozess...")

        stats = {
            "processed": 0,
            "filled": 0,
            "missed": 0,
            "closed_active": 0,
            "errors": 0,
            "skipped_no_data": 0,
        }
        db = SignalDatabase(self.db_path)

        # 1. Trades laden (FIX: Datum-Filter entfernt, da unzuverlässig bei Timezones)
        with db._get_conn() as conn:
            # Debugging: Wie viele Trades gibt es überhaupt?
            total_count = conn.execute("SELECT count(*) FROM active_trades").fetchone()[
                0
            ]
            logger.info(f"DEBUG: Total Trades in DB: {total_count}")

            sql = """
                SELECT * FROM active_trades
                WHERE status IN ('CREATED', 'ACTIVE')
                ORDER BY signal_date ASC
            """
            trades = [dict(row) for row in conn.execute(sql).fetchall()]

        if not trades:
            logger.info("Keine Trades für Backfill gefunden (Liste leer).")
            return stats

        logger.info(f"Backfill: Verarbeite {len(trades)} Trades...")
        market_cache = {}

        for trade in trades:
            stats["processed"] += 1
            symbol = trade["symbol"]

            # WICHTIG: Wir starten beim SIGNAL DATE (Fallback Entry Date)
            start_date_str = trade.get("signal_date") or trade["entry_date"]

            if not start_date_str:
                logger.warning(f"Trade {trade['id']} hat kein Datum. Skipping.")
                continue

            try:
                strat_impl = self._get_strategy(trade)
                if not strat_impl:
                    continue

                # Marktdaten laden (mit Cache)
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
                    logger.warning(f"Keine Marktdaten für Symbol gefunden: {symbol}")
                    stats["skipped_no_data"] += 1
                    continue

                # Simulation ab Signal-Tag starten
                start_ts = pd.Timestamp(str(start_date_str).split(" ")[0])
                future_candles = df_hist[df_hist["date"] >= start_ts].sort_values(
                    "date"
                )

                if future_candles.empty:
                    continue

                # === SIMULATION LOOP ===
                # Wir gehen Tag für Tag durch die Historie
                for _, current_candle in future_candles.iterrows():
                    # 1. Aktueller Zustand des Trades (kann sich im Loop ändern!)
                    current_status = trade["status"]

                    # History Slice bis HEUTE (simuliert)
                    simulated_history = df_hist[
                        df_hist["date"] <= current_candle["date"]
                    ]

                    # FALL A: Warten auf Entry (CREATED)
                    if current_status == "CREATED":
                        # Aufruf der Strategie-Logik
                        # Die Strategie entscheidet, ob heute gefüllt wird (checkt T+1).
                        result_msg = strat_impl.check_entry(trade, current_candle, db)

                        if result_msg and "FILLED" in result_msg:
                            stats["filled"] += 1

                            # Trade Objekt lokal updaten für den nächsten Loop-Durchlauf
                            trade["status"] = "ACTIVE"
                            trade["entry_date"] = current_candle["date"].strftime(
                                "%Y-%m-%d"
                            )

                            # Quantity berechnen (nur einmalig beim Fill)
                            self._calculate_and_update_quantity(
                                trade, strat_impl, default_budget, db, df_hist, start_ts
                            )

                        elif not result_msg:
                            # Noch kein Entry Signal heute -> Warten auf morgen
                            pass
                        else:
                            # Explizites Missed Signal (z.B. TimeStop auf Entry)
                            trade["status"] = "MISSED"
                            stats["missed"] += 1
                            break  # Trade ist vorbei

                    # FALL B: Trade ist Aktiv (ACTIVE)
                    elif current_status == "ACTIVE":
                        # Strategie managen lassen
                        mgmt_msg = strat_impl.manage_active_trade(
                            trade, simulated_history, db
                        )

                        if mgmt_msg:
                            # Prüfen ob geschlossen
                            if any(
                                x in mgmt_msg
                                for x in ["EXIT", "STOP", "WIN", "TIME_STOP"]
                            ):
                                logger.info(f"[{symbol}] BACKFILL: {mgmt_msg}")
                                trade["status"] = "CLOSED"
                                stats["closed_active"] += 1
                                break  # Trade vorbei

                            # Status-Update (z.B. TP1 Locked) -> Trade lokal aktualisieren
                            if "TP1" in mgmt_msg and "LOCKED" in mgmt_msg:
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

    def _calculate_and_update_quantity(
        self, trade, strat, budget, db, df_hist, signal_ts
    ):
        """Hilfsmethode zur Quantity-Berechnung beim Backfill."""
        try:
            entry_price = float(trade["entry_price"])
            new_qty = 1

            if isinstance(strat, DipBuyerStrategy):
                new_qty = int(budget / entry_price)
            elif isinstance(strat, SplitTargetStrategy):
                # Risk based
                sl_price = entry_price * 0.99
                # Versuch, das Low der Signalkerze zu finden
                signal_candle = df_hist[df_hist["date"] == signal_ts]
                if not signal_candle.empty:
                    sl_price = float(signal_candle.iloc[0]["low"])

                risk = entry_price - sl_price
                if risk > 0:
                    new_qty = int(100.0 / risk)  # 100$ Risk
                else:
                    new_qty = int(budget / entry_price)
            else:
                new_qty = int(budget / entry_price)

            new_qty = max(1, new_qty)
            db.update_trade_quantity(trade["id"], new_qty)
            trade["quantity"] = new_qty
        except Exception as e:
            logger.warning(f"Qty Error {trade['symbol']}: {e}")

    def _cleanup_stale_trades(self) -> None:
        """
        Bereinigt Trades basierend auf ECHTEN HANDELSTAGEN (Candle Count).
        Verhindert das Löschen valider Trades über Wochenenden/Feiertage.
        """
        try:
            db = SignalDatabase(self.db_path)
            # 1. Kandidaten laden
            with db._get_conn() as conn:
                created_trades = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM active_trades WHERE status = 'CREATED'"
                    ).fetchall()
                ]
                active_trades = [
                    dict(r)
                    for r in conn.execute(
                        "SELECT * FROM active_trades WHERE status = 'ACTIVE'"
                    ).fetchall()
                ]

            if not created_trades and not active_trades:
                return

            # 2. Marktdaten-Cache aufbauen (Optimierung: Ein Batch-Query)
            all_symbols = list(
                set([t["symbol"] for t in created_trades + active_trades])
            )
            market_dates = {}

            with sqlite3.connect(self.stocks_db_path) as conn:
                # Wir laden genug Historie (60 Tage)
                start_date = (datetime.now() - pd.Timedelta(days=60)).strftime(
                    "%Y-%m-%d"
                )
                placeholders = ",".join("?" for _ in all_symbols)
                sql = f"SELECT symbol, date FROM market_prices WHERE symbol IN ({placeholders}) AND date >= ? AND timeframe='1D'"
                rows = conn.execute(sql, all_symbols + [start_date]).fetchall()

                for sym, date_str in rows:
                    if sym not in market_dates:
                        market_dates[sym] = []
                    market_dates[sym].append(pd.Timestamp(date_str))

            # Sortieren
            for sym in market_dates:
                market_dates[sym].sort()

            updates = []

            # 3. CREATED Trades prüfen (Max 5 Handelstage warten ab Signal)
            MAX_WAIT_TRADING_DAYS = 5
            for trade in created_trades:
                sym = trade["symbol"]
                # Referenz ist Signal Date (oder Fallback Entry Date)
                ref_date_str = trade.get("signal_date") or trade["entry_date"]
                ref_date = pd.Timestamp(str(ref_date_str).split(" ")[0])

                if sym not in market_dates:
                    continue

                # Zähle Kerzen NACH dem Signal
                candles_after = [d for d in market_dates[sym] if d > ref_date]

                if len(candles_after) > MAX_WAIT_TRADING_DAYS:
                    updates.append(("MISSED", "STALE_CLEANUP", trade["id"]))
                    logger.info(
                        f"Cleanup CREATED: {sym} (Signal: {ref_date.date()}) nach {len(candles_after)} Handelstagen entfernt."
                    )

            # 4. ACTIVE Trades prüfen (Sicherheitsnetz: 20 Handelstage ab Entry)
            MAX_HOLD_TRADING_DAYS = 20
            for trade in active_trades:
                sym = trade["symbol"]
                # Referenz ist Entry Date (Start des Trades)
                ref_date = pd.Timestamp(str(trade["entry_date"]).split(" ")[0])

                if sym not in market_dates:
                    continue

                candles_after = [d for d in market_dates[sym] if d > ref_date]

                if len(candles_after) > MAX_HOLD_TRADING_DAYS:
                    updates.append(("CLOSED", "STALE_CLEANUP", trade["id"]))
                    logger.info(
                        f"Cleanup ACTIVE: {sym} nach {len(candles_after)} Handelstagen zwangsgeschlossen."
                    )

            # 5. DB Updates
            if updates:
                with db._get_conn() as conn:
                    conn.executemany(
                        "UPDATE active_trades SET status = ?, exit_reason = ?, closed_at = CURRENT_TIMESTAMP WHERE id = ?",
                        updates,
                    )
                    conn.commit()

        except Exception as e:
            logger.error(f"Fehler bei _cleanup_stale_trades: {e}", exc_info=True)

    def _update_positions_status(self) -> None:
        """Prüft Fills (CREATED) und Exits (ACTIVE) basierend auf Live-Daten."""
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()
        if not trades:
            return

        market_data = self._fetch_market_data_batch(trades)
        alerts = []

        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in market_data:
                continue

            strat_impl = self._get_strategy(trade)
            if not strat_impl:
                continue

            df_hist = market_data[symbol]
            if df_hist.empty:
                continue

            msg = None
            if trade["status"] == "CREATED":
                # Letzte geschlossene Kerze prüfen
                last_candle = df_hist.iloc[-1]
                msg = strat_impl.check_entry(trade, last_candle, db)

            elif trade["status"] == "ACTIVE":
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

    def _get_strategy(self, trade: dict) -> Optional[BaseTradeStrategy]:
        raw = str(trade.get("strategy", "")).lower().replace(" ", "")
        for key, impl in self.strategies.items():
            if key in raw:
                return impl
        return None

    def _fetch_market_data_batch(self, trades: list[dict]) -> dict[str, pd.DataFrame]:
        symbols = list({t["symbol"] for t in trades})
        if not symbols:
            return {}

        start_date = (datetime.now() - pd.Timedelta(days=50)).strftime("%Y-%m-%d")
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
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return round(float(obj), 2)
        return obj
