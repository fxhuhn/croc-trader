import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .database import SignalDatabase

logger = logging.getLogger(__name__)


class ScreenerEngine:
    def __init__(self, stocks_db_path: Path, signals_db_path: Path, telegram_bot=None):
        self.stocks_db_path = stocks_db_path
        self.signals_db = SignalDatabase(signals_db_path)
        self.telegram = telegram_bot

    def _load_market_data(self, days=400) -> pd.DataFrame:
        """Lädt OHLCV Daten und wandelt sie in Wide-Format um."""
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # Wir laden alles aus der stocks.db
        with sqlite3.connect(self.stocks_db_path) as conn:
            query = f"""
            SELECT date, symbol, open, high, low, close, volume
            FROM market_prices
            WHERE date >= '{start_date}'
                AND timeframe = '1D'
            ORDER BY date ASC
            """
            df = pd.read_sql_query(query, conn)

        if df.empty:
            logger.warning("Keine Daten für Screening gefunden.")
            return pd.DataFrame()

        # Datum konvertieren
        df["date"] = pd.to_datetime(df["date"])

        # Pivoting: Wir brauchen DataFrames wo Columns = Symbole sind
        # Das entspricht deiner Logik (closes.rolling...)
        closes = df.pivot(index="date", columns="symbol", values="close")
        opens = df.pivot(index="date", columns="symbol", values="open")
        highs = df.pivot(index="date", columns="symbol", values="high")
        lows = df.pivot(index="date", columns="symbol", values="low")
        volumes = df.pivot(index="date", columns="symbol", values="volume")

        return opens, highs, lows, closes, volumes

    def run_dip_buyer(self):
        logger.info("Starte Dip-Buyer Screening...")

        # 1. Daten laden
        try:
            opens, highs, lows, closes, volumes = self._load_market_data()
        except Exception as e:
            logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return

        if closes.empty:
            logger.warning("Keine Daten für Screening vorhanden.")
            return

        # -----------------------------------------------------------
        # 2. DEINE BERECHNUNGEN (1:1 übernommen & angepasst)
        # -----------------------------------------------------------

        # SMA 200 & Volumen SMA
        sma200 = closes.rolling(window=200, min_periods=150).mean()
        vol_sma20 = volumes.rolling(window=20).mean()

        # ATR (Average True Range)
        prev_close = closes.shift(1)
        # Wichtig: Alignment sicherstellen
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()

        # Maximum element-wise
        tr = pd.DataFrame(
            np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)),
            index=closes.index,
            columns=closes.columns,
        )

        # EMA-ATR (span=5)
        atr5 = tr.ewm(span=9, adjust=False).mean()

        # 3-Tages-Differenz
        diff_3day = closes - closes.shift(3)

        # atr_r3
        atr5_safe = atr5.replace(0, np.nan)
        atr_r3 = diff_3day / atr5_safe

        # SetupScore
        setup_score = atr_r3 * -1

        # IBS (Internal Bar Strength) für Bedingung 8
        # (Close - Low) / (High - Low)
        day_range = highs - lows
        day_range = day_range.replace(0, 0.01)
        ibs = (closes - lows) / day_range

        # -----------------------------------------------------------
        # 3. AKTUELLE WERTE (LETZTE ZEILE)
        # -----------------------------------------------------------
        # Wir nehmen die letzte Zeile (den aktuellen/letzten Handelstag)

        curr_date = closes.index[-1].strftime("%Y-%m-%d")

        # Slices für den letzten Tag
        c_open = opens.iloc[-1]
        c_high = highs.iloc[-1]  # Unused variable but kept for logic consistency
        c_low = lows.iloc[-1]  # Unused variable
        c_close = closes.iloc[-1]

        c_sma200 = sma200.iloc[-1]
        c_vol_sma20 = vol_sma20.iloc[-1]
        c_atr5 = atr5.iloc[-1]
        c_atr_r3 = atr_r3.iloc[-1]
        c_setup = setup_score.iloc[-1]
        c_ibs = ibs.iloc[-1]

        # Werte von Gestern (iloc[-2])
        p_open = opens.iloc[-2]
        p_close = closes.iloc[-2]

        # -----------------------------------------------------------
        # 4. FILTER LOGIK (Boolean Masks)
        # -----------------------------------------------------------

        cond1 = (
            c_vol_sma20 > 500_000
        )  # Liquidität (etwas entspannter als 1M zum Testen)
        cond2 = c_close > 5.0  # Kein Pennystock
        cond3 = c_close > c_sma200  # Trend long
        cond4 = c_atr_r3 < -1.0  # ATR Stretch
        cond5 = (c_atr5 / c_close) > 0.03  # Hohe Vola
        cond6 = c_close < c_open  # Heute rot
        cond7 = p_close < p_open  # Gestern rot
        cond8 = c_ibs < 0.25  # IBS (leicht entspannt auf 0.25)

        final_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8

        # Ergebnisse extrahieren
        # final_mask ist eine Series mit (Symbol -> True/False)
        hits = final_mask[final_mask].index.tolist()

        logger.info(
            f"Dip-Buyer Screening fertig. Datum: {curr_date}, Treffer: {len(hits)}"
        )

        # -----------------------------------------------------------
        # 5. BERECHNUNG ENTRY LIMIT & SPEICHERN
        # -----------------------------------------------------------

        # Vektorisierte Berechnung für ALLE Symbole (wir greifen später nur die Hits ab)
        # Entry Limit = Close - ATR5
        c_entry_limit = c_close - c_atr5

        results_to_save = []
        new_trades_count = 0

        for symbol in hits:
            # Werte extrahieren
            val_close = round(float(c_close[symbol]), 2)
            val_setup = round(float(c_setup[symbol]), 2)
            val_atr_r3 = round(float(c_atr_r3[symbol]), 2)
            val_ibs = round(float(c_ibs[symbol]), 2)
            val_atr5 = round(float(c_atr5[symbol]), 2)
            val_entry_limit = round(float(c_entry_limit[symbol]), 2)  # Unser Kaufpreis

            # A) Für Screener Historie speichern
            results_to_save.append(
                {
                    "strategy": "dip_buyer",
                    "symbol": symbol,
                    "date": curr_date,
                    "close": val_close,
                    "setup_score": val_setup,
                    "atr_r3": val_atr_r3,
                    "ibs": val_ibs,
                    "atr5": val_atr5,
                    "entry_limit": val_entry_limit,
                }
            )

            # B) AUTOMATISCH ALS TRADE ANLEGEN
            # Wir gehen davon aus, dass wir das Limit morgen fischen.
            # entry_date ist das heutige Screening-Datum (Startschuss).
            self.signals_db.add_trade(
                symbol=symbol,
                entry_date=curr_date,
                entry_price=val_entry_limit,  # Das Limit ist unser Preis
                atr_at_entry=val_atr5,  # ATR für Target-Berechnung
                quantity=1,
            )
            new_trades_count += 1

        self.signals_db.save_screener_results(results_to_save)

        if hits and self.telegram:
            msg = f"🔎 **Dip-Buyer Screener**\nDatum: {curr_date}\nTreffer: {new_trades_count}\n"
            for sym in hits[:5]:  # Max 5 auflisten
                msg += f"- {sym} (Score: {round(float(c_setup[sym]), 2)})\n"

            if len(hits) > 5:
                msg += "... und weitere."

            self.telegram.send(msg)

        if hits and self.telegram:
            # DataFrame für Anzeige bauen (nur wichtige Spalten)
            df_display = pd.DataFrame(results_to_save)[
                [
                    "symbol",
                    "entry_limit",
                    "setup_score",
                    "close",
                    "atr5",
                ]
            ]
            df_display.rename(
                columns={
                    "symbol": "Ticker",
                    "entry_limit": "Entry",
                    "setup_score": "Score",
                    "atr5": "ATR",
                },
                inplace=True,
            )

            self.telegram.send_dataframe(
                df_display, title=f"📉 Dip-Buyer Report ({curr_date})"
            )
        elif self.telegram:
            self.telegram.send_message(f"📉 Dip-Buyer ({curr_date}): Keine Treffer.")

        return len(hits)

    def run_historical_test(self, lookback_days=10):
        """
        Führt den Screener für die letzten X Tage aus und speichert Signale nachträglich.
        """
        logger.info(f"Starte historischen Test für die letzten {lookback_days} Tage...")

        # 1. Daten laden (wir brauchen genug Puffer für SMA200)
        try:
            # Wir laden 400 Tage + den Testzeitraum, damit der SMA200 auch am Anfang berechenbar ist
            opens, highs, lows, closes, volumes = self._load_market_data(
                days=400 + lookback_days
            )
        except Exception as e:
            logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return 0

        if closes.empty:
            logger.warning("Keine Daten für Backtest vorhanden.")
            return 0

        # -----------------------------------------------------------
        # 2. BERECHNUNGEN (Vektorisierung über den gesamten Zeitraum)
        # -----------------------------------------------------------

        # SMA 200 & Volumen
        sma200 = closes.rolling(window=200, min_periods=150).mean()
        vol_sma20 = volumes.rolling(window=20).mean()

        # ATR
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = pd.DataFrame(
            np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)),
            index=closes.index,
            columns=closes.columns,
        )
        atr5 = tr.ewm(span=5, adjust=False).mean()

        # Indikatoren
        diff_3day = closes - closes.shift(3)
        atr5_safe = atr5.replace(0, np.nan)
        atr_r3 = diff_3day / atr5_safe
        setup_score = atr_r3 * -1

        # IBS
        day_range = highs - lows
        day_range = day_range.replace(0, 0.01)
        ibs = (closes - lows) / day_range

        # Entry Limits (Close - ATR5) für den Kaufpreis
        entry_limits = closes - atr5

        # -----------------------------------------------------------
        # 3. SCHLEIFE ÜBER DIE VERGANGENHEIT
        # -----------------------------------------------------------
        total_signals = 0
        total_days = len(closes)

        # Start-Index berechnen: Wir wollen die letzten 'lookback_days' durchgehen
        start_index = max(200, total_days - lookback_days)

        logger.info(f"Prüfe Datenpunkte von Index {start_index} bis {total_days}")

        # LISTE FÜR SCREENER ERGEBNISSE
        results_to_save = []

        for i in range(start_index, total_days):
            loop_date = closes.index[i].strftime("%Y-%m-%d")

            # Slices (Zeilen) extrahieren
            c_open = opens.iloc[i]
            c_close = closes.iloc[i]
            c_sma200 = sma200.iloc[i]
            c_vol_sma20 = vol_sma20.iloc[i]
            c_atr5 = atr5.iloc[i]
            c_atr_r3 = atr_r3.iloc[i]
            c_setup = setup_score.iloc[i]  # Brauchen wir jetzt für die Stats
            c_ibs = ibs.iloc[i]
            c_entry = entry_limits.iloc[i]

            p_open = opens.iloc[i - 1]
            p_close = closes.iloc[i - 1]

            # Filter Logik
            cond1 = c_vol_sma20 > 500_000
            cond2 = c_close > 5.0
            cond3 = c_close > c_sma200
            cond4 = c_atr_r3 < -1.0
            cond5 = (c_atr5 / c_close) > 0.03
            cond6 = c_close < c_open
            cond7 = p_close < p_open
            cond8 = c_ibs < 0.25

            final_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
            hits = final_mask[final_mask].index.tolist()

            if not hits:
                continue

            for symbol in hits:
                # 1. Daten für 'screener_results' Tabelle sammeln
                results_to_save.append(
                    {
                        "strategy": "dip_buyer",
                        "symbol": symbol,
                        "date": loop_date,
                        "close": round(float(c_close[symbol]), 2),
                        "setup_score": round(float(c_setup[symbol]), 2),
                        "atr_r3": round(float(c_atr_r3[symbol]), 2),
                        "ibs": round(float(c_ibs[symbol]), 2),
                        "atr5": round(float(c_atr5[symbol]), 2),
                        "entry_limit": round(float(c_entry[symbol]), 2),
                    }
                )

                # 2. Trade anlegen (ignoriert Duplikate dank UNIQUE Constraint)
                self.signals_db.add_trade(
                    symbol=symbol,
                    entry_date=loop_date,
                    entry_price=round(float(c_entry[symbol]), 2),
                    atr_at_entry=round(float(c_atr5[symbol]), 2),
                    quantity=1,
                )
                total_signals += 1

        # AM ENDE: Screener Ergebnisse speichern!
        if results_to_save:
            self.signals_db.save_screener_results(results_to_save)
            logger.info(
                f"Backtest: {len(results_to_save)} historische Screener-Ergebnisse gespeichert."
            )

        if self.telegram:
            self.telegram.send(
                f"🧪 **Backtest ({lookback_days} Tage) fertig.**\nSignale: {total_signals}"
            )

        return total_signals
