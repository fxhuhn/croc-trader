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

    def _get_yahoo_ticker(self, symbol: str) -> str:
        """
        Übersetzt TradingView/Broker-Symbole in Yahoo Finance Symbole.
        """
        mapping = {
            # --- FUTURES ---
            "ES1!": "ES=F",  # S&P 500 E-Mini
            "NQ1!": "NQ=F",  # Nasdaq 100 E-Mini
            "YM1!": "YM=F",  # Dow Jones E-Mini
            "RTY1!": "RTY=F",  # Russell 2000
            "FDAX1!": "DX=F",  # DAX Futures (Eurex)
            "GC1!": "GC=F",  # Gold
            "SI1!": "SI=F",  # Silber
            "CL1!": "CL=F",  # Rohöl
            "BTC1!": "BTC=F",  # Bitcoin Futures
            # --- FOREX ---
            "EURUSD": "EURUSD=X",
            "GBPUSD": "GBPUSD=X",
            # --- DEUTSCHE AKTIEN (Beispiele aus deinem Log) ---
            "JEN": "JEN.DE",  # Jenoptik
            "VH2": "VH2.DE",  # (Falls Valneva o.ä. auf Xetra gemeint ist)
        }

        # 1. Direkter Match im Mapping
        if symbol in mapping:
            return mapping[symbol]

        # 2. Heuristik für deutsche Aktien (wenn kein Suffix da ist und es kein Future ist)
        # Wenn das Symbol kurz ist (3-4 Zeichen) und keine Sonderzeichen hat, probieren wir .DE
        # (Das ist optional, aber hilft oft bei Xetra-Titeln)
        # if len(symbol) <= 4 and symbol.isalpha() and symbol not in ["AAPL", "TSLA", "MSFT"]:
        #    return f"{symbol}.DE"

        return symbol

    def check_active_positions(self):
        """
        Hauptlogik:
        1. Prüft Einstiege (Fill?) und Exits (Stop?).
        2. Aktualisiert DB Status.
        3. Erstellt YAML Order-Files (Gruppiert nach Datum).
        """
        db = SignalDatabase(self.db_path)
        trades = db.get_all_managed_trades()

        if not trades:
            logger.info("TradeManager: Keine Trades vorhanden.")
            return

        # --- VORBEREITUNG ---
        alerts = []
        watchlist = []  # Für Telegram
        orders_by_date = {}  # Sammeln der Orders pro Datum

        today = datetime.now().date()
        next_trading_day_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")

        symbols = list(set(t["symbol"] for t in trades))
        logger.info(
            f"TradeManager: Prüfe {len(trades)} Trades ({len(symbols)} Symbole)..."
        )

        # Cache für Historiendaten
        data_cache = {}

        # Daten laden (mit Übersetzung und Fehlerbehandlung)
        for symbol in symbols:
            yahoo_symbol = self._get_yahoo_ticker(symbol)

            try:
                # Wir laden Daten für das Yahoo-Symbol
                df = yf.download(
                    yahoo_symbol, period="10d", progress=False, auto_adjust=True
                )

                if not df.empty:
                    df = df.reset_index()
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.columns = df.columns.str.lower()

                    # Fix: Manchmal heißt die Spalte 'Date' oder 'Datetime'
                    if "date" in df.columns:
                        df["date"] = pd.to_datetime(df["date"])
                    elif "datetime" in df.columns:
                        df["date"] = pd.to_datetime(df["datetime"])

                    # WICHTIG: Wir speichern es unter dem ORIGINAL Symbol im Cache!
                    # Damit der Rest des Codes (der 'symbol' nutzt) funktioniert.
                    data_cache[symbol] = df
                else:
                    logger.warning(
                        f"Keine Daten für {symbol} (Yahoo: {yahoo_symbol}) gefunden."
                    )

            except Exception as e:
                # Fehler abfangen, damit der Loop nicht abbricht
                logger.error(f"Fehler beim Laden von {symbol} -> {yahoo_symbol}: {e}")

        # --- LOGIK DURCHLAUFEN ---
        for trade in trades:
            trade_id = trade["id"]
            symbol = trade["symbol"]
            status = trade["status"]
            entry_price = trade["entry_price"]
            atr_entry = trade["atr_at_entry"]
            entry_date_str = trade["entry_date"]

            strategy = trade.get("strategy", "Unknown").replace(" ", "_")

            if symbol not in data_cache:
                # Wenn Daten fehlen, überspringen wir diesen Trade leise
                continue

            df = data_cache[symbol]
            signal_date_obj = pd.to_datetime(entry_date_str).date()

            # ==============================================================================
            # PHASE 1: STATUS PRÜFUNG (Vergangenheit bis Heute)
            # ==============================================================================

            if status == "CREATED":
                potential_days = df[df["date"].dt.date > signal_date_obj]

                if not potential_days.empty:
                    row = potential_days.iloc[0]
                    check_date = row["date"].strftime("%Y-%m-%d")

                    if row["low"] <= entry_price:
                        db.update_trade_status(trade_id, "ACTIVE")
                        status = "ACTIVE"
                        alerts.append(f"✅ **FILLED**: {symbol} am {check_date}")
                    else:
                        db.update_trade_status(trade_id, "MISSED", "LIMIT_NOT_REACHED")
                        alerts.append(f"❌ **MISSED**: {symbol} am {check_date}")
                        continue

            elif status == "ACTIVE":
                last_row = df.iloc[-1]
                current_close = last_row["close"]

                prev_high_check = 999999
                if len(df) >= 2:
                    prev_row = df.iloc[-2]
                    prev_high_check = prev_row["high"]

                days_since = (today - signal_date_obj).days

                # 1. TIME STOP
                if days_since >= 7:
                    db.close_trade(trade_id, reason="TIME_STOP")
                    alerts.append(f"⏰ **TIME STOP**: {symbol} wird geschlossen.")

                    exit_order = {
                        "id": f"{symbol}_{strategy}_EXIT_TIME",
                        "symbol": symbol,
                        "qty": trade["quantity"],
                        "mode": "SINGLE",
                        "action": "SELL",
                        "type": "MKT",
                        "tif": "DAY",
                        "comment": "Time Stop Triggered",
                    }
                    if next_trading_day_str not in orders_by_date:
                        orders_by_date[next_trading_day_str] = []
                    orders_by_date[next_trading_day_str].append(exit_order)
                    continue

                # 2. LOC EXIT
                elif current_close > prev_high_check:
                    alerts.append(f"📈 **LOC SIGNAL**: {symbol} (Close > PrevHigh)")

                    exit_order = {
                        "id": f"{symbol}_{strategy}_EXIT_LOC",
                        "symbol": symbol,
                        "qty": trade["quantity"],
                        "mode": "SINGLE",
                        "action": "SELL",
                        "type": "LOC",
                        "price": 0.0,
                        "tif": "DAY",
                        "comment": "LOC Triggered",
                    }
                    if next_trading_day_str not in orders_by_date:
                        orders_by_date[next_trading_day_str] = []
                    orders_by_date[next_trading_day_str].append(exit_order)

            # ==============================================================================
            # PHASE 2: ORDER ERSTELLUNG (Entries)
            # ==============================================================================

            if status == "CREATED":
                target_price = entry_price + (0.8 * atr_entry)

                order_entry = {
                    "id": f"{symbol}_{strategy}",
                    "symbol": symbol,
                    "qty": trade["quantity"],
                    "mode": "BRACKET",
                    "entry": {
                        "action": "BUY",
                        "type": "LMT",
                        "price": round(entry_price, 2),
                        "tif": "DAY",
                    },
                    "exits": [
                        {
                            "action": "SELL",
                            "type": "LOC",
                            "price": round(target_price, 2),
                            "tif": "DAY",
                        }
                    ],
                }

                if entry_date_str not in orders_by_date:
                    orders_by_date[entry_date_str] = []

                orders_by_date[entry_date_str].append(order_entry)

                if entry_date_str >= datetime.now().strftime("%Y-%m-%d"):
                    watchlist.append(f"🆕 **BUY**: {symbol} @ {entry_price:.2f}")

            elif status == "ACTIVE":
                loc_trigger_level = df.iloc[-1]["high"]
                stop_date = signal_date_obj + timedelta(days=7)
                days_left = (stop_date - today).days

                watchlist.append(
                    f"🔹 **{symbol}** (Tag {7 - days_left}/7)\n"
                    f"   LOC Trigger morgen: > {loc_trigger_level:.2f}"
                )

        # --- YAML EXPORT ---
        generated_files = []

        for date_key, orders in orders_by_date.items():
            if not orders:
                continue

            filename = f"orders_{date_key}.yaml"
            file_path = self.orders_dir / filename

            try:
                with open(file_path, "w", encoding="utf-8") as f:
                    yaml.dump(orders, f, sort_keys=False, allow_unicode=True)
                logger.info(f"Order-File erstellt: {filename} ({len(orders)} Orders)")
                generated_files.append(filename)
            except Exception as e:
                logger.error(f"YAML Export Fehler für {filename}: {e}")

        # --- TELEGRAM ---
        if alerts or watchlist:
            msg = []
            if alerts:
                msg.append("⚡ **STATUS UPDATES**\n" + "\n".join(alerts))
            if watchlist:
                msg.append("📋 **PORTFOLIO WATCH (Aktuell)**\n" + "\n".join(watchlist))

            if generated_files:
                file_list = ", ".join(generated_files[-3:])
                if len(generated_files) > 3:
                    file_list += "..."
                msg.append(
                    f"\n📁 {len(generated_files)} Order-Dateien erstellt/aktualisiert."
                )

            if self.telegram:
                self.telegram.send("\n\n".join(msg))
