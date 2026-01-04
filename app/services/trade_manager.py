import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from .database import SignalDatabase

logger = logging.getLogger(__name__)


class TradeManager:
    def __init__(self, db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.telegram = telegram_bot

    def check_active_positions(self):
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            logger.info("TradeManager: Keine Trades zu verwalten.")
            return

        alerts = []  # Dringende Aktionen (Entry, Exit)
        watchlist = []  # Statusbericht für laufende Trades

        # Mapping für Bulk-Download vorbereiten
        symbols = list(set(t["symbol"] for t in trades))
        logger.info(f"TradeManager: Prüfe {len(trades)} Trades...")

        for trade in trades:
            trade_id = trade["id"]
            symbol = trade["symbol"]
            status = trade["status"]
            entry_price = trade["entry_price"]
            entry_date_str = trade["entry_date"]
            atr_entry = trade["atr_at_entry"]

            # Daten laden
            try:
                df = yf.download(symbol, period="10d", progress=False, auto_adjust=True)
                if df.empty:
                    continue

                df = df.reset_index()
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df.columns = df.columns.str.lower()
                df["date"] = pd.to_datetime(df["date"])

            except Exception as e:
                logger.error(f"Datenfehler {symbol}: {e}")
                continue

            signal_date = pd.to_datetime(entry_date_str)
            trade_still_active = False

            # -----------------------------------------------------------
            # PHASE 1: ENTRY CHECK (CREATED)
            # -----------------------------------------------------------
            if status == "CREATED":
                potential_entry_days = df[df["date"] > signal_date].copy()

                if potential_entry_days.empty:
                    continue

                entry_day_row = potential_entry_days.iloc[0]
                entry_day_low = entry_day_row["low"]
                entry_day_date = entry_day_row["date"].strftime("%Y-%m-%d")

                if entry_day_low <= entry_price:
                    # ENTRY SUCCESS
                    db.update_trade_status(trade_id, "ACTIVE")
                    alerts.append(
                        f"✅ **ENTRY FILLED**: {symbol}\nLimit {entry_price} erreicht am {entry_day_date}."
                    )
                    status = "ACTIVE"  # Für den Watchlist-Block unten
                    trade_still_active = True
                else:
                    # ENTRY MISSED
                    db.update_trade_status(
                        trade_id, "MISSED", exit_reason="LIMIT_NOT_REACHED"
                    )
                    alerts.append(
                        f"❌ **ENTRY MISSED**: {symbol}\nLimit {entry_price} verpasst (Low: {entry_day_low:.2f})."
                    )
                    continue

            # -----------------------------------------------------------
            # PHASE 2: EXIT CHECK (ACTIVE)
            # -----------------------------------------------------------
            elif status == "ACTIVE":
                trade_still_active = True  # Gehen wir erstmal davon aus

                current_row = df.iloc[-1]
                current_price = current_row["close"]

                # Prev High für LOC Check HEUTE
                if len(df) < 2:
                    continue
                prev_row = df.iloc[-2]
                prev_high = prev_row["high"]

                days_since_signal = (datetime.now() - signal_date).days

                # A) TIME STOP CHECK
                if days_since_signal >= 7:
                    db.close_trade(trade_id, reason="TIME_STOP")
                    alerts.append(
                        f"⏰ **TIME STOP**: {symbol} (Tag {days_since_signal}). Trade geschlossen."
                    )
                    trade_still_active = False  # Ist jetzt zu

                # B) LOC EXIT CHECK (HEUTE)
                elif current_price > prev_high:
                    alerts.append(
                        f"📈 **LOC EXIT SIGNAL**: {symbol}\nClose {current_price:.2f} > PrevHigh {prev_high:.2f}.\nBitte manuell schließen!"
                    )
                    # Wir lassen ihn auf Active, bis du manuell schließt, oder wir schließen hier auch auto.
                    # trade_still_active = False

            # -----------------------------------------------------------
            # PHASE 3: WATCHLIST FÜR MORGEN (Nur wenn noch ACTIVE)
            # -----------------------------------------------------------
            if trade_still_active and status == "ACTIVE":
                # Wir berechnen die Levels für den NÄCHSTEN Tag

                # 1. Das "Prev High" von Morgen ist das "High" von Heute
                # (Wir nehmen die letzte Kerze im DF, das ist "Heute")
                current_row = df.iloc[-1]
                todays_high = current_row["high"]

                # 2. Target
                target = entry_price + (0.8 * atr_entry)

                # 3. Time Stop Datum
                time_stop_date = signal_date + timedelta(days=7)
                days_left = (time_stop_date - datetime.now()).days

                watchlist.append(
                    f"🔹 **{symbol}**\n"
                    f"   LOC Trigger (Morgen): > {todays_high:.2f}\n"
                    f"   Target: {target:.2f}\n"
                    f"   Time Stop: in {days_left} Tagen ({time_stop_date.strftime('%d.%m.')})"
                )

        # TELEGRAM ZUSAMMENBAUEN
        msgs = []
        if alerts:
            msgs.append("⚡ **ALERTS (Handlungsbedarf)** ⚡\n" + "\n".join(alerts))

        if watchlist:
            msgs.append("📋 **WATCHLIST (Für Morgen)**\n" + "\n".join(watchlist))

        if msgs and self.telegram:
            full_msg = "\n\n".join(msgs)
            self.telegram.send(full_msg)
            logger.info("Trade Report gesendet.")
