import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import yaml

from ..mapping import mapper
from .database import SignalDatabase

logger = logging.getLogger(__name__)


class ScreenerEngine:
    def __init__(
        self,
        stocks_db_path: Path,
        signals_db_path: Path,
        config_path: Path = None,
        telegram_bot=None,
    ):
        self.stocks_db_path = stocks_db_path
        self.signals_db = SignalDatabase(signals_db_path)
        self.config_path = config_path
        self.telegram = telegram_bot

    def _load_strategies(self):
        if not self.config_path or not self.config_path.exists():
            return []
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Screener YAML Fehler: {e}")
            return []

    def _get_exchange(self, symbol):
        if mapper._mapping and symbol in mapper._mapping:
            return mapper._mapping[symbol]
        return "UNKNOWN"

    def _load_market_data(
        self, days=400
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        with sqlite3.connect(self.stocks_db_path) as conn:
            df = pd.read_sql_query(
                f"SELECT date, symbol, open, high, low, close, volume FROM market_prices WHERE date >= '{start_date}' AND timeframe = '1D' ORDER BY date ASC",
                conn,
            )

        if df.empty:
            return (
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
                pd.DataFrame(),
            )

        df["date"] = pd.to_datetime(df["date"])
        closes = df.pivot(index="date", columns="symbol", values="close")
        opens = df.pivot(index="date", columns="symbol", values="open")
        highs = df.pivot(index="date", columns="symbol", values="high")
        lows = df.pivot(index="date", columns="symbol", values="low")
        volumes = df.pivot(index="date", columns="symbol", values="volume")
        return opens, highs, lows, closes, volumes

    def _calculate_technical_indicators(self, opens, highs, lows, closes, volumes):
        sma200 = closes.rolling(window=200, min_periods=150).mean()
        vol_sma20 = volumes.rolling(window=20).mean()

        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()
        tr = pd.DataFrame(
            np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)),
            index=closes.index,
            columns=closes.columns,
        )
        atr5 = tr.ewm(span=9, adjust=False).mean()

        diff_3day = closes - closes.shift(3)
        atr5_safe = atr5.replace(0, np.nan)
        atr_r3 = diff_3day / atr5_safe
        setup_score = atr_r3 * -1
        day_range = highs - lows
        day_range = day_range.replace(0, 0.01)
        ibs = (closes - lows) / day_range
        entry_limits = closes - atr5

        return {
            "sma200": sma200,
            "vol_sma20": vol_sma20,
            "rsi": rsi,
            "atr5": atr5,
            "atr_r3": atr_r3,
            "setup_score": setup_score,
            "ibs": ibs,
            "entry_limits": entry_limits,
        }

    def run_dip_buyer(self):
        logger.info("Starte Dip-Buyer Screening...")
        opens, highs, lows, closes, volumes = self._load_market_data()
        if closes.empty:
            return 0
        ind = self._calculate_technical_indicators(opens, highs, lows, closes, volumes)

        curr_date = closes.index[-1].strftime("%Y-%m-%d")
        i_today = -1

        c_close = closes.iloc[i_today]
        # ... Filter Logik unverändert ...
        cond1 = ind["vol_sma20"].iloc[i_today] > 1_000_000
        cond2 = c_close > 5.0
        cond3 = c_close > ind["sma200"].iloc[i_today]
        cond4 = ind["atr_r3"].iloc[i_today] < -1.0
        cond5 = (ind["atr5"].iloc[i_today] / c_close) > 0.03
        cond6 = c_close < opens.iloc[i_today]
        cond7 = closes.iloc[i_today - 1] < opens.iloc[i_today - 1]
        cond8 = ind["ibs"].iloc[i_today] < 0.2
        final_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
        hits = final_mask[final_mask].index.tolist()

        logger.info(f"Dip-Buyer: {len(hits)} Treffer am {curr_date}.")
        results_to_save = []

        for symbol in hits:
            res = {
                "date": curr_date,
                "symbol": symbol,
                "exchange": self._get_exchange(symbol),
                "timeframe": "1D",
                "close": round(float(c_close[symbol]), 2),
                "high": round(
                    float(highs.iloc[i_today][symbol]), 2
                ),  # <--- NEU: High speichern
                "atr_r3": round(float(ind["atr_r3"].iloc[i_today][symbol]), 2),
                "setup_score": round(
                    float(ind["setup_score"].iloc[i_today][symbol]), 2
                ),
                "entry_limit": round(
                    float(ind["entry_limits"].iloc[i_today][symbol]), 2
                ),
                "atr5": round(float(ind["atr5"].iloc[i_today][symbol]), 2),
            }
            results_to_save.append(res)
            # Trade direkt anlegen (hier für Legacy support)
            self.signals_db.add_trade(
                symbol=symbol,
                entry_date=curr_date,
                entry_price=res["entry_limit"],
                atr_at_entry=res["atr5"],
            )

        if results_to_save:
            self.signals_db.save_screener_dip_buyer(results_to_save)
            self._send_telegram_report("📉 Dip-Buyer", curr_date, results_to_save)

        return len(hits)

    def process_croc_signals(self, lookback_days=5):
        """Batch-Import für Webhook Signale."""
        logger.info(f"Verarbeite Croc-Signale der letzten {lookback_days} Tage...")
        strategies = self._load_strategies()
        if not strategies:
            return 0
        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        with self.signals_db._get_conn() as conn:
            sql = f"SELECT *, date(timestamp) as date_str FROM signals WHERE date(timestamp) >= '{start_date}'"
            df = pd.read_sql_query(sql, conn)

        if df.empty:
            return 0

        numeric_cols = ["close", "high", "low", "rsi", "sma_200"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        total_matches = 0
        for strat in strategies:
            name = strat.get("name")
            logic = strat.get("logic")
            if not logic:
                continue
            try:
                matches = df.query(logic).copy()
                if not matches.empty:
                    results = []
                    for _, row in matches.iterrows():
                        res = {
                            "date": row["date_str"],
                            "symbol": row["symbol"],
                            "exchange": row.get("exchange")
                            or self._get_exchange(row["symbol"]),
                            "timeframe": row.get("timeframe", "1D"),
                            "strategy": name,
                            "signal": row["signal"],
                            "close": round(row["close"], 2),
                            "high": round(row["high"], 2),
                            "low": round(row["low"], 2),
                            "rsi": round(row["rsi"], 2),
                            "sma_200": round(row["sma_200"], 2),
                        }
                        results.append(res)
                    self.signals_db.save_screener_webhook(results)
                    total_matches += len(results)

                    # --- NEU: TELEGRAM NACHRICHT SENDEN ---
                    # Wir senden einen Report pro gefundener Strategie
                    self._send_telegram_report(
                        f"🚀 Signal: {name}", f"Letzte {lookback_days} Tage", results
                    )

            except Exception:
                pass

        return total_matches

    def run_historical_test(self, lookback_days=5):
        logger.info(f"Starte historischen DipBuyer Test für {lookback_days} Tage...")
        opens, highs, lows, closes, volumes = self._load_market_data(
            days=400 + lookback_days
        )
        if closes.empty:
            return 0
        ind = self._calculate_technical_indicators(opens, highs, lows, closes, volumes)

        total_signals = 0
        start_index = max(200, len(closes) - lookback_days)
        results_to_save = []

        for i in range(start_index, len(closes)):
            curr_date = closes.index[i].strftime("%Y-%m-%d")
            c_close = closes.iloc[i]
            # ... (Restliche Variablen c_open, p_close etc.) ...
            c_open = opens.iloc[i]
            p_close = closes.iloc[i - 1]
            p_open = opens.iloc[i - 1]

            # ... (Filter Conditions unverändert) ...
            cond1 = ind["vol_sma20"].iloc[i] > 1_000_000
            cond2 = c_close > 5.0
            cond3 = c_close > ind["sma200"].iloc[i]
            cond4 = ind["atr_r3"].iloc[i] < -1.0
            cond5 = (ind["atr5"].iloc[i] / c_close) > 0.03
            cond6 = c_close < c_open
            cond7 = p_close < p_open
            cond8 = ind["ibs"].iloc[i] < 0.2

            final_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8
            hits = final_mask[final_mask].index.tolist()

            for symbol in hits:
                res = {
                    "date": curr_date,
                    "symbol": symbol,
                    "exchange": self._get_exchange(symbol),
                    "timeframe": "1D",
                    "close": round(float(c_close[symbol]), 2),
                    "high": round(
                        float(highs.iloc[i][symbol]), 2
                    ),  # <--- NEU: High speichern
                    "atr_r3": round(float(ind["atr_r3"].iloc[i][symbol]), 2),
                    "setup_score": round(float(ind["setup_score"].iloc[i][symbol]), 2),
                    "entry_limit": round(float(ind["entry_limits"].iloc[i][symbol]), 2),
                    "atr5": round(float(ind["atr5"].iloc[i][symbol]), 2),
                }
                results_to_save.append(res)
                self.signals_db.add_trade(
                    symbol=symbol,
                    entry_date=curr_date,
                    entry_price=res["entry_limit"],
                    atr_at_entry=res["atr5"],
                    quantity=1,
                )
                total_signals += 1

        if results_to_save:
            self.signals_db.save_screener_dip_buyer(results_to_save)
            logger.info(
                f"DipBuyer Backtest: {len(results_to_save)} historische Ergebnisse gespeichert."
            )

        return total_signals

    def _send_telegram_report(self, title, date, results):
        if not self.telegram:
            return
        msg = f"🔎 **{title}**\nDatum: {date}\nTreffer: {len(results)}\n"
        for r in results[:5]:
            msg += f"- {r['symbol']}\n"
        self.telegram.send(msg)
