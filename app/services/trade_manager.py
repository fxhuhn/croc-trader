import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from ..config import settings  # NEU
from .database import SignalDatabase

logger = logging.getLogger(__name__)


class TradeManager:
    def __init__(self, db_path: Path, telegram_bot=None):
        self.db_path = db_path
        self.telegram = telegram_bot

        # NEU: Zentralisierter Ordner
        self.orders_dir = settings.get_folder("orders")

    def check_active_positions(self):
        """
        Hauptlogik:
        1. Prüft Einstiege (Fill?) und Exits (Stop?).
        2. Aktualisiert DB Status.
        3. Erstellt JSON Order-File für den nächsten Tag.
        """
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            logger.info("TradeManager: Keine Trades vorhanden.")
            return

        # --- VORBEREITUNG ---
        alerts = []
        watchlist = []  # Für Telegram

        # Listen für den JSON Export
        json_new_entries = []
        json_position_updates = []

        # Datums-Handling
        today = datetime.now().date()
        # Annahme: Nächster Tag ist morgen (für Sa/So müsste man Kalender prüfen, hier vereinfacht)
        valid_from = today + timedelta(days=1)
        valid_until = valid_from  # Day Order

        validity_str = {
            "valid_from": valid_from.strftime("%Y-%m-%d"),
            "valid_until": valid_until.strftime("%Y-%m-%d"),
        }

        # Bulk Download der Daten
        symbols = list(set(t["symbol"] for t in trades))
        logger.info(f"TradeManager: Prüfe {len(trades)} Trades...")

        for trade in trades:
            trade_id = trade["id"]
            symbol = trade["symbol"]
            status = trade["status"]
            entry_price = trade["entry_price"]  # Das Limit vom Screener
            atr_entry = trade["atr_at_entry"]
            entry_date_str = trade["entry_date"]

            # 1. Daten laden
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
                logger.error(f"Fehler {symbol}: {e}")
                continue

            signal_date_obj = pd.to_datetime(entry_date_str).date()
            trade_dirty = False  # Wurde der Status geändert?

            # ==============================================================================
            # PHASE 1: STATUS PRÜFUNG (Vergangenheit bis Heute)
            # ==============================================================================

            # Python 3.10+ match case für bessere Lesbarkeit
            match status:
                case "CREATED":
                    # Wir suchen Tage NACH dem Signal
                    potential_days = df[df["date"].dt.date > signal_date_obj]

                    if not potential_days.empty:
                        # Check ersten Tag nach Signal
                        row = potential_days.iloc[0]
                        check_date = row["date"].strftime("%Y-%m-%d")

                        if row["low"] <= entry_price:
                            # FILL!
                            db.update_trade_status(trade_id, "ACTIVE")
                            status = "ACTIVE"  # Für Phase 2 sofort nutzen
                            alerts.append(f"✅ **FILLED**: {symbol} am {check_date}")
                            trade_dirty = True
                        else:
                            # MISSED!
                            db.update_trade_status(
                                trade_id, "MISSED", "LIMIT_NOT_REACHED"
                            )
                            alerts.append(f"❌ **MISSED**: {symbol} am {check_date}")
                            continue  # Trade ist raus

                case "ACTIVE":
                    # Wir brauchen aktuelle Daten
                    last_row = df.iloc[-1]
                    current_close = last_row["close"]
                    current_high = last_row["high"]  # Das ist das PrevHigh für Morgen!

                    # PrevHigh für den heutigen Check (Gestern)
                    if len(df) >= 2:
                        prev_row = df.iloc[-2]
                        prev_high_check = prev_row["high"]
                    else:
                        prev_high_check = 999999

                    days_since = (today - signal_date_obj).days
                    time_stop_date = signal_date_obj + timedelta(days=7)

                    # 1. TIME STOP CHECK
                    if days_since >= 7:
                        db.close_trade(trade_id, reason="TIME_STOP")
                        alerts.append(f"⏰ **TIME STOP**: {symbol} wird geschlossen.")

                        json_position_updates.append(
                            {
                                "symbol": symbol,
                                "action": "SELL",
                                "type": "MARKET_ON_CLOSE",
                                "reason": "TIME_STOP_EXPIRED",
                                "validity": validity_str,
                            }
                        )
                        continue  # Trade ist zu

                    # 2. LOC EXIT CHECK (War Close heute > High gestern?)
                    elif current_close > prev_high_check:
                        alerts.append(
                            f"📈 **LOC SIGNAL**: {symbol} (Close {current_close:.2f} > {prev_high_check:.2f})"
                        )

                        json_position_updates.append(
                            {
                                "symbol": symbol,
                                "action": "SELL",
                                "type": "MARKET",
                                "reason": "LOC_TRIGGERED",
                                "comment": f"Close {current_close:.2f} > PrevHigh {prev_high_check:.2f}",
                                "validity": validity_str,
                            }
                        )

            # ==============================================================================
            # PHASE 2: ORDER ERSTELLUNG (Für Morgen)
            # ==============================================================================

            if status == "CREATED":
                target_price = entry_price + (0.8 * atr_entry)
                stop_date = signal_date_obj + timedelta(days=7)

                json_new_entries.append(
                    {
                        "symbol": symbol,
                        "action": "BUY",
                        "type": "LIMIT",
                        "limit_price": round(entry_price, 2),
                        "validity": validity_str,
                        "bracket_orders": {
                            "take_profit": {
                                "type": "LIMIT",
                                "price": round(target_price, 2),
                                "validity": "GTC",
                            },
                            "time_stop": {
                                "type": "MARKET_ON_CLOSE",
                                "trigger_date": stop_date.strftime("%Y-%m-%d"),
                                "comment": "Verkaufen am Ende von Tag 7",
                            },
                        },
                    }
                )
                watchlist.append(f"🆕 **BUY**: {symbol} @ {entry_price:.2f}")

            elif status == "ACTIVE":
                loc_trigger_level = df.iloc[-1]["high"]
                target_price = entry_price + (0.8 * atr_entry)
                stop_date = signal_date_obj + timedelta(days=7)
                days_left = (stop_date - today).days

                json_position_updates.append(
                    {
                        "symbol": symbol,
                        "status": "HOLD",
                        "current_close": round(df.iloc[-1]["close"], 2),
                        "validity": validity_str,
                        "active_orders": {
                            "loc_exit_watch": {
                                "trigger_condition": "CLOSE > PREV_HIGH",
                                "trigger_price_tomorrow": round(loc_trigger_level, 2),
                                "action_if_triggered": "SELL MARKET_ON_CLOSE",
                            },
                            "take_profit": {
                                "price": round(target_price, 2),
                                "status": "OPEN",
                            },
                            "time_stop": {
                                "days_remaining": days_left,
                                "expiry_date": stop_date.strftime("%Y-%m-%d"),
                            },
                        },
                    }
                )

                watchlist.append(
                    f"🔹 **{symbol}** (Tag {7 - days_left}/7)\n"
                    f"   LOC Trigger morgen: > {loc_trigger_level:.2f}\n"
                    f"   Target: {target_price:.2f}"
                )

        # --- JSON EXPORT ---
        filename = f"orders_{valid_from.strftime('%Y-%m-%d')}.json"
        file_path = self.orders_dir / filename

        output_data = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "valid_for": validity_str,
            "orders": {
                "new_entries": json_new_entries,
                "portfolio_management": json_position_updates,
            },
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=4, ensure_ascii=False)
            logger.info(f"Order-File erstellt: {file_path}")
        except Exception as e:
            logger.error(f"JSON Error: {e}")

        # --- TELEGRAM ---
        if alerts or watchlist:
            msg = []
            if alerts:
                msg.append("⚡ **STATUS UPDATES**\n" + "\n".join(alerts))
            if watchlist:
                msg.append("📋 **PORTFOLIO WATCH**\n" + "\n".join(watchlist))
            msg.append(f"\n📁 Order-Datei für {valid_from} erstellt.")

            if self.telegram:
                self.telegram.send("\n\n".join(msg))
