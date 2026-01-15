import logging
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ....services.database import SignalDatabase
from ....services.telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class DipBuyerStrategy(BaseStrategy):
    name = "DipBuyer"

    # --- KONFIGURATION / KONSTANTEN ---
    # Filter-Schwellenwerte als Klassen-Konstanten
    MIN_VOLUME = 1_000_000  # Mindestvolumen (SMA 20)
    MIN_PRICE = 5.0  # Mindestpreis in Währung
    MAX_ATR_R3 = -1.0  # ATR R3 muss kleiner sein als dieser Wert (Überverkauft)
    MIN_VOLA_RATIO = 0.03  # Mindest-Volatilität (ATR5 / Close)
    MAX_IBS = 0.2  # IBS (Internal Bar Strength) muss kleiner sein als 0.2

    # Indikator-Einstellungen (Optional, aber empfohlen für Konsistenz)
    SMA_TREND_WINDOW = 200
    VOL_SMA_WINDOW = 20

    def __init__(
        self,
        stocks_db_path: Path,
        signals_db: SignalDatabase,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        super().__init__(signals_db, telegram_bot)
        self.stocks_db_path = stocks_db_path

    def run(self, days: int = 0) -> int:
        mode = "Backfill" if days > 0 else "Daily"
        logger.info(f"[{self.name}] Starte Analyse ({mode})...")

        if (market_data := self._load_market_data(days=400 + days)) is None:
            return 0

        indicators = self._calculate_indicators(market_data)

        total_len = len(indicators["close"])
        # Nutzung der Konstante für den SMA-Vorlauf
        start_idx = (
            max(self.SMA_TREND_WINDOW, total_len - days) if days > 0 else total_len - 1
        )

        all_results: list[dict[str, Any]] = []

        for i in range(start_idx, total_len):
            if daily_hits := self._apply_logic(indicators, idx_pos=i):
                all_results.extend(daily_hits)
                if days > 0:
                    self._create_trades(daily_hits)

        if all_results:
            self.signals_db.save_screener_dip_buyer(all_results)
            if days == 0:
                self._create_trades(all_results)
                latest_date = all_results[-1]["date"]
                self._send_telegram_report("📉 Dip-Buyer", latest_date, all_results)

        logger.info(f"[{self.name}] Fertig: {len(all_results)} Treffer.")
        return len(all_results)

    def _create_trades(self, results: list[dict[str, Any]]) -> None:
        for res in results:
            self.signals_db.add_trade(
                symbol=res["symbol"],
                entry_date=res["date"],
                entry_price=res["entry_limit"],
                atr_at_entry=res["atr5"],
                quantity=1,
                strategy=self.name,
            )

    def _load_market_data(self, days: int) -> dict[str, pd.DataFrame] | None:
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")

        try:
            with sqlite3.connect(self.stocks_db_path) as conn:
                query = (
                    f"SELECT date, symbol, open, high, low, close, volume "
                    f"FROM market_prices WHERE date >= '{start_date}' "
                    f"AND timeframe = '1D' ORDER BY date ASC"
                )
                df = pd.read_sql_query(query, conn)
        except Exception as e:
            logger.error(f"DipBuyer: Fehler beim Laden der Marktdaten: {e}")
            return None

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])

        return {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

    def _calculate_indicators(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        close, high, low, volume = (
            data["close"],
            data["high"],
            data["low"],
            data["volume"],
        )

        # Nutzung der Konstanten für Fenstergrößen
        sma200 = close.rolling(window=self.SMA_TREND_WINDOW, min_periods=150).mean()
        vol_sma20 = volume.rolling(window=self.VOL_SMA_WINDOW).mean()

        # RSI Calculation
        delta = close.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        # ATR Calculation
        prev_close = close.shift(1)
        tr_values = np.maximum.reduce(
            [
                (high - low).values,
                (high - prev_close).abs().fillna(0).values,
                (low - prev_close).abs().fillna(0).values,
            ]
        )
        atr5 = (
            pd.DataFrame(tr_values, index=close.index, columns=close.columns)
            .ewm(span=9, adjust=False)
            .mean()
        )

        atr_r3 = (close - close.shift(3)) / atr5.replace(0, np.nan)
        ibs = (close - low) / (high - low).replace(0, 0.01)

        return {
            "sma200": sma200,
            "vol_sma20": vol_sma20,
            "rsi": rsi.fillna(50),
            "atr5": atr5,
            "atr_r3": atr_r3,
            "setup_score": atr_r3 * -1,
            "ibs": ibs,
            "entry_limits": close - atr5,
            "close": close,
            "open": data["open"],
            "high": high,
        }

    def _apply_logic(self, ind: dict[str, Any], idx_pos: int) -> list[dict[str, Any]]:
        # Daten für den aktuellen Zeitpunkt extrahieren
        i_vol_sma = ind["vol_sma20"].iloc[idx_pos]
        i_close = ind["close"].iloc[idx_pos]
        i_open = ind["open"].iloc[idx_pos]
        i_prev_close = ind["close"].iloc[idx_pos - 1]
        i_prev_open = ind["open"].iloc[idx_pos - 1]
        i_sma200 = ind["sma200"].iloc[idx_pos]
        i_atr_r3 = ind["atr_r3"].iloc[idx_pos]
        i_atr5 = ind["atr5"].iloc[idx_pos]
        i_ibs = ind["ibs"].iloc[idx_pos]

        # Vektorisierte Maske mit KLASSEN-KONSTANTEN (self.NAME)
        mask = (
            (i_vol_sma > self.MIN_VOLUME)
            & (i_close > self.MIN_PRICE)
            & (i_close > i_sma200)
            & (i_atr_r3 < self.MAX_ATR_R3)
            & ((i_atr5 / i_close) > self.MIN_VOLA_RATIO)
            & (i_close < i_open)
            & (i_prev_close < i_prev_open)
            & (i_ibs < self.MAX_IBS)
        )

        hits = mask[mask].index.tolist()
        results = []
        curr_date = ind["close"].index[idx_pos].strftime("%Y-%m-%d")

        for symbol in hits:
            results.append(
                {
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
            )

        return results
