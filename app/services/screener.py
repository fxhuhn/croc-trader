import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..mapping import mapper
from .database import SignalDatabase

logger = logging.getLogger(__name__)


class ScreenerEngine:
    def __init__(
        self,
        stocks_db_path: Path,
        signals_db_path: Path,
        strategies: List[Dict] = None,
        telegram_bot=None,
    ):
        self.stocks_db_path = stocks_db_path
        self.signals_db = SignalDatabase(signals_db_path)
        self.strategies = strategies or []
        self.telegram = telegram_bot

    def _get_exchange(self, symbol: str) -> str:
        return mapper.get_exchange(symbol, default="UNKNOWN")

    def _load_market_data(self, days=400) -> Optional[Dict[str, pd.DataFrame]]:
        start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(self.stocks_db_path) as conn:
                df = pd.read_sql_query(
                    f"SELECT date, symbol, open, high, low, close, volume "
                    f"FROM market_prices WHERE date >= '{start_date}' "
                    f"AND timeframe = '1D' ORDER BY date ASC",
                    conn,
                )
        except Exception as e:
            logger.error(f"Fehler beim Laden der Marktdaten: {e}")
            return None

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])

        # Pivot für vektorisierte Berechnungen
        return {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

    def _calculate_technical_indicators(
        self, data: Dict[str, pd.DataFrame]
    ) -> Dict[str, pd.DataFrame]:
        closes = data["close"]
        highs = data["high"]
        lows = data["low"]
        volumes = data["volume"]

        # SMAs
        sma200 = closes.rolling(window=200, min_periods=150).mean()
        vol_sma20 = volumes.rolling(window=20).mean()

        # RSI
        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)

        # ATR Calculation
        prev_close = closes.shift(1)
        tr1 = highs - lows
        tr2 = (highs - prev_close).abs()
        tr3 = (lows - prev_close).abs()

        # Effizientes Maximum über 3 DataFrames
        tr_values = np.maximum.reduce(
            [tr1.values, tr2.fillna(0).values, tr3.fillna(0).values]
        )

        tr = pd.DataFrame(tr_values, index=closes.index, columns=closes.columns)
        atr5 = tr.ewm(span=9, adjust=False).mean()

        # Custom Metrics für Dip Buyer
        diff_3day = closes - closes.shift(3)
        atr5_safe = atr5.replace(0, np.nan)
        atr_r3 = diff_3day / atr5_safe

        day_range = (highs - lows).replace(0, 0.01)
        ibs = (closes - lows) / day_range
        entry_limits = closes - atr5

        return {
            "sma200": sma200,
            "vol_sma20": vol_sma20,
            "rsi": rsi,
            "atr5": atr5,
            "atr_r3": atr_r3,
            "setup_score": atr_r3 * -1,
            "ibs": ibs,
            "entry_limits": entry_limits,
            # Rohdaten für Filter durchreichen
            "close": closes,
            "open": data["open"],
            "high": highs,
            "low": lows,
        }

    def _apply_dip_buyer_logic(
        self, ind: Dict[str, pd.DataFrame], idx_pos: int
    ) -> List[Dict]:
        """Zentrale Logik für Dip Buyer Filterung an einem bestimmten Tag."""

        # Slice Data at specific index position
        # .iloc[idx] liefert eine Series mit Symbolen als Index
        i_vol_sma = ind["vol_sma20"].iloc[idx_pos]
        i_close = ind["close"].iloc[idx_pos]
        i_open = ind["open"].iloc[idx_pos]
        i_prev_close = ind["close"].iloc[idx_pos - 1]
        i_prev_open = ind["open"].iloc[idx_pos - 1]

        i_sma200 = ind["sma200"].iloc[idx_pos]
        i_atr_r3 = ind["atr_r3"].iloc[idx_pos]
        i_atr5 = ind["atr5"].iloc[idx_pos]
        i_ibs = ind["ibs"].iloc[idx_pos]

        # Filter Conditions
        cond1 = i_vol_sma > 1_000_000
        cond2 = i_close > 5.0
        cond3 = i_close > i_sma200
        cond4 = i_atr_r3 < -1.0
        cond5 = (i_atr5 / i_close) > 0.03
        cond6 = i_close < i_open
        cond7 = i_prev_close < i_prev_open
        cond8 = i_ibs < 0.2

        final_mask = cond1 & cond2 & cond3 & cond4 & cond5 & cond6 & cond7 & cond8

        hits = final_mask[final_mask].index.tolist()
        results = []
        curr_date = ind["close"].index[idx_pos].strftime("%Y-%m-%d")

        for symbol in hits:
            res = {
                "date": curr_date,
                "symbol": symbol,
                "exchange": self._get_exchange(symbol),
                "timeframe": "1D",
                "close": round(float(i_close[symbol]), 2),
                "high": round(float(ind["high"].iloc[idx_pos][symbol]), 2),
                "atr_r3": round(float(i_atr_r3[symbol]), 2),
                "setup_score": round(
                    float(ind["setup_score"].iloc[idx_pos][symbol]), 2
                ),
                "entry_limit": round(
                    float(ind["entry_limits"].iloc[idx_pos][symbol]), 2
                ),
                "atr5": round(float(i_atr5[symbol]), 2),
            }
            results.append(res)

        return results

    def run_dip_buyer(self) -> int:
        """Standard Run für Heute."""
        return self._run_analysis(backfill_days=0)

    def run_historical_test(self, lookback_days=5) -> int:
        """Historischer Test."""
        return self._run_analysis(backfill_days=lookback_days)

    def _run_analysis(self, backfill_days=0) -> int:
        """
        Kombinierte Logik für Daily und Backfill.
        """
        mode = "Backfill" if backfill_days > 0 else "Daily"
        logger.info(f"Starte Dip-Buyer ({mode})...")

        data = self._load_market_data(days=400 + backfill_days)
        if not data:
            return 0

        ind = self._calculate_technical_indicators(data)

        total_len = len(ind["close"])
        # Wenn Backfill, starte weiter hinten, sonst nur der letzte Tag
        start_idx = (
            max(200, total_len - backfill_days) if backfill_days > 0 else total_len - 1
        )

        all_results = []

        for i in range(start_idx, total_len):
            daily_hits = self._apply_dip_buyer_logic(ind, idx_pos=i)

            if daily_hits:
                all_results.extend(daily_hits)

                # Bei Backfill direkt Trades anlegen (Legacy Support)
                if backfill_days > 0:
                    for res in daily_hits:
                        self.signals_db.add_trade(
                            symbol=res["symbol"],
                            entry_date=res["date"],
                            entry_price=res["entry_limit"],
                            atr_at_entry=res["atr5"],
                            quantity=1,
                        )

        if all_results:
            self.signals_db.save_screener_dip_buyer(all_results)

            # Bei Daily Run auch Trade anlegen und Telegram Report senden
            if backfill_days == 0:
                curr_date = all_results[0]["date"]
                for res in all_results:
                    self.signals_db.add_trade(
                        symbol=res["symbol"],
                        entry_date=res["date"],
                        entry_price=res["entry_limit"],
                        atr_at_entry=res["atr5"],
                    )
                self._send_telegram_report("📉 Dip-Buyer", curr_date, all_results)

        count = len(all_results)
        logger.info(f"Dip-Buyer ({mode}): {count} Treffer.")
        return count

    def process_croc_signals(self, lookback_days=5):
        """Batch-Import für Webhook Signale unter Nutzung der zentralen Strategien."""
        logger.info(f"Verarbeite Croc-Signale der letzten {lookback_days} Tage...")

        if not self.strategies:
            logger.info("Keine Webhook-Strategien geladen.")
            return 0

        start_date = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%Y-%m-%d"
        )

        with self.signals_db._get_conn() as conn:
            # Optimierte Query
            sql = f"""
                SELECT symbol, signal, close, high, low, rsi, sma_200, exchange, timeframe,
                       date(timestamp) as date_str
                FROM signals
                WHERE date(timestamp) >= '{start_date}'
            """
            df = pd.read_sql_query(sql, conn)

        if df.empty:
            return 0

        # Numeric conversion safety
        cols = ["close", "high", "low", "rsi", "sma_200"]
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        total_matches = 0

        for strat in self.strategies:
            name = strat.get("name")
            logic = strat.get("logic")
            if not logic:
                continue

            try:
                # Pandas query Magic
                matches = df.query(logic).copy()

                if not matches.empty:
                    results = []
                    for _, row in matches.iterrows():
                        res = {
                            "date": row["date_str"],
                            "symbol": row["symbol"],
                            "exchange": row["exchange"]
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

                    self._send_telegram_report(
                        f"🚀 Signal: {name}", f"Letzte {lookback_days} Tage", results
                    )

            except Exception as e:
                logger.error(f"Fehler in Strategie '{name}': {e}")

        return total_matches

    def _send_telegram_report(self, title, date, results):
        if not self.telegram:
            return
        msg = f"🔎 **{title}**\nDatum: {date}\nTreffer: {len(results)}\n"
        for r in results[:5]:
            msg += f"- {r['symbol']}\n"
        if len(results) > 5:
            msg += f"... und {len(results) - 5} weitere."
        self.telegram.send(msg)
