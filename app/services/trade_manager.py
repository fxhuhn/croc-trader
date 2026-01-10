import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yaml
import yfinance as yf

from ..config import settings
from .database import SignalDatabase

logger = logging.getLogger(__name__)


class TradeManager:
    def __init__(self, db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.telegram = telegram_bot
        self.orders_dir = settings.get_folder("orders")

    def check_active_positions(self):
        """
        Hauptlogik:
        1. Prüft Einstiege (Fill?) und Exits (Stop?).
        2. Aktualisiert DB Status.
        3. Erstellt YAML Order-File für die Ausführung.
        """
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            logger.info("TradeManager: Keine Trades vorhanden.")
            return

        # --- VORBEREITUNG ---
        alerts = []
        watchlist = []  # Für Telegram

        # Liste für die YAML Orders
        yaml_orders = []

        # Datums-Handling
        today = datetime.now().date()
        valid_from = today + timedelta(days=1)

        # Bulk Download der Daten
        # [FIX] F841: Variable 'symbols' wird jetzt im Loop genutzt
        symbols = list(set(t["symbol"] for t in trades))
        logger.info(f"TradeManager: Prüfe {len(trades)} Trades...")

        # Cache für Historiendaten
        data_cache = {}

        # Daten einmal laden
        for symbol in symbols:
            try:
                df = yf.download(symbol, period="10d", progress=False, auto_adjust=True)
                if not df.empty:
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()
                    df["date"] = pd.to_datetime(df["date"])
                    data_cache[symbol] = df
            except Exception as e:
                logger.error(f"Fehler beim Laden von {symbol}: {e}")

        for trade in trades:
            trade_id = trade["id"]
            symbol = trade["symbol"]
            status = trade["status"]
            entry_price = trade["entry_price"]
            atr_entry = trade["atr_at_entry"]
            entry_date_str = trade["entry_date"]
            # Leerzeichen in ID vermeiden
            strategy = trade.get("strategy", "Unknown").replace(" ", "_")

            if symbol not in data_cache:
                continue

            df = data_cache[symbol]
            signal_date_obj = pd.to_datetime(entry_date_str).date()

            # ==============================================================================
            # PHASE 1: STATUS PRÜFUNG (Vergangenheit bis Heute)
            # ==============================================================================

            if status == "CREATED":
                # Wir suchen Tage NACH dem Signal
                potential_days = df[df["date"].dt.date > signal_date_obj]

                if not potential_days.empty:
                    # Check ersten Tag nach Signal
                    row = potential_days.iloc[0]
                    check_date = row["date"].strftime("%Y-%m-%d")

                    if row["low"] <= entry_price:
                        # FILL!
                        db.update_trade_status(trade_id, "ACTIVE")
                        status = "ACTIVE"
                        alerts.append(f"✅ **FILLED**: {symbol} am {check_date}")
                    else:
                        # MISSED!
                        db.update_trade_status(trade_id, "MISSED", "LIMIT_NOT_REACHED")
                        alerts.append(f"❌ **MISSED**: {symbol} am {check_date}")
                        continue  # Trade ist raus

            elif status == "ACTIVE":
                # Wir brauchen aktuelle Daten
                last_row = df.iloc[-1]
                current_close = last_row["close"]
                # [FIX] F841: current_high Zuweisung entfernt, da ungenutzt

                # PrevHigh Logic
                prev_high_check = 999999
                if len(df) >= 2:
                    prev_row = df.iloc[-2]
                    prev_high_check = prev_row["high"]

                days_since = (today - signal_date_obj).days

                # 1. TIME STOP CHECK (7 Tage)
                # [FIX] F841: time_stop_date Zuweisung entfernt, da ungenutzt
                if days_since >= 7:
                    db.close_trade(trade_id, reason="TIME_STOP")
                    alerts.append(f"⏰ **TIME STOP**: {symbol} wird geschlossen.")

                    # Exit Order für YAML
                    yaml_orders.append(
                        {
                            "id": f"{symbol}_{strategy}_EXIT_TIME",
                            "symbol": symbol,
                            "qty": trade["quantity"],
                            "mode": "SINGLE",
                            "action": "SELL",
                            "type": "MKT",  # Market on Close force
                            "tif": "DAY",
                            "comment": "Time Stop Triggered",
                        }
                    )
                    continue

                # 2. LOC EXIT CHECK (Close > PrevHigh)
                elif current_close > prev_high_check:
                    alerts.append(f"📈 **LOC SIGNAL**: {symbol} (Close > PrevHigh)")

                    # Exit Order für YAML
                    yaml_orders.append(
                        {
                            "id": f"{symbol}_{strategy}_EXIT_LOC",
                            "symbol": symbol,
                            "qty": trade["quantity"],
                            "mode": "SINGLE",
                            "action": "SELL",
                            "type": "LOC",
                            "price": 0.0,  # Market on Close logic (oder Limit=0 für MKT)
                            "tif": "DAY",
                            "comment": "LOC Triggered",
                        }
                    )

            # ==============================================================================
            # PHASE 2: ORDER ERSTELLUNG (Für Morgen)
            # ==============================================================================

            if status == "CREATED":
                # Target Berechnung
                target_price = entry_price + (0.8 * atr_entry)

                # Aufbau des YAML Eintrags gemäß Anforderung
                order_entry = {
                    "id": f"{symbol}_{strategy}",
                    "symbol": symbol,
                    "qty": trade["quantity"],
                    "mode": "BRACKET",  # Parent + Children
                    # 1. Der Einstieg (Parent)
                    "entry": {
                        "action": "BUY",
                        "type": "LMT",
                        "price": round(entry_price, 2),
                        "tif": "DAY",
                    },
                    # 2. Die Exits (Children)
                    "exits": [
                        {
                            "action": "SELL",
                            "type": "LOC",  # Wie im Beispiel gewünscht
                            "price": round(target_price, 2),
                            "tif": "DAY",
                        }
                    ],
                }

                yaml_orders.append(order_entry)
                watchlist.append(f"🆕 **BUY**: {symbol} @ {entry_price:.2f}")

            elif status == "ACTIVE":
                # Info für Telegram (Watchlist), aber keine neue Bracket Order
                loc_trigger_level = df.iloc[-1]["high"]
                stop_date = signal_date_obj + timedelta(days=7)
                days_left = (stop_date - today).days

                watchlist.append(
                    f"🔹 **{symbol}** (Tag {7 - days_left}/7)\n"
                    f"   LOC Trigger morgen: > {loc_trigger_level:.2f}"
                )

        # --- YAML EXPORT ---
        if yaml_orders:
            filename = f"orders_{valid_from.strftime('%Y-%m-%d')}.yaml"
            file_path = self.orders_dir / filename

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    # sort_keys=False behält die Reihenfolge (id, symbol, qty...)
                    yaml.dump(yaml_orders, f, sort_keys=False, allow_unicode=True)
                logger.info(f"Order-File erstellt: {file_path}")
            except Exception as e:
                logger.error(f"YAML Export Fehler: {e}")

        # --- TELEGRAM ---
        if alerts or watchlist:
            msg = []
            if alerts:
                msg.append("⚡ **STATUS UPDATES**\n" + "\n".join(alerts))
            if watchlist:
                msg.append("📋 **PORTFOLIO WATCH**\n" + "\n".join(watchlist))

            if yaml_orders:
                msg.append(
                    f"\n📁 Order-Datei für {valid_from} erstellt ({len(yaml_orders)} Orders)."
                )

            if self.telegram:
                self.telegram.send("\n\n".join(msg))
