import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ....services.database import SignalDatabase
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class TurnoverTimingStrategy(BaseStrategy):
    name = "TurnoverTiming"

    def __init__(
        self,
        stocks_db_path: Path,
        signals_db: SignalDatabase,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        super().__init__(signals_db, telegram_bot)
        self.stocks_db_path = stocks_db_path
        self.universe = ExchangeSymbol()

    def run(self, days: int = 0) -> int:
        logger.info(f"[{self.name}] Starte Analyse (Strikte PDF-Logik)...")

        # Mehr Daten laden für den längeren SMA 150 (365 -> 400 Tage Puffer)
        if (data := self._load_market_data(days=400 + days)) is None:
            return 0

        indicators = self._calculate_indicators(data)

        target_ts, idx_pos = self._determine_analysis_date(indicators, days)
        if target_ts is None:
            return 0

        date_str = target_ts.strftime("%Y-%m-%d")
        all_hits: list[dict[str, Any]] = []

        # 1. ANPASSUNG: Russell 1000 statt DOW-30 gemäß PDF "Universum"
        universes = [
            ("RUSSELL-1000", self.universe.russell_1000),
            ("NASDAQ-100", self.universe.nasdaq_100),
            ("SP-500", self.universe.sp_500),
        ]

        processed_symbols: set[str] = set()

        for idx_name, symbol_list in universes:
            # 2. ANPASSUNG: Strikte Logik in _scan_universe (erst Top 20 Turnover, dann Trend)
            candidates = self._scan_universe(symbol_list, indicators, idx_pos, idx_name)

            # Ranking: Aus den verbleibenden Kandidaten die Top 4 nach Turnover wählen
            # (Die Liste kommt bereits sortiert aus _scan_universe, zur Sicherheit sortieren wir nochmal)
            candidates.sort(key=lambda x: x["turnover_sma20"], reverse=True)

            # Top 4 auswählen
            for c in candidates[:4]:
                if c["symbol"] in processed_symbols:
                    continue

                processed_symbols.add(c["symbol"])
                all_hits.append(
                    {
                        "date": date_str,
                        "symbol": c["symbol"],
                        "exchange": self._get_exchange(c["symbol"]),
                        "timeframe": "1D",
                        "source_index": c["source_index"],
                        "close": round(c["close"], 2),
                        "atr3": round(c["atr3"], 2),
                        "turnover_sma20": round(c["turnover_sma20"], 0),
                        "entry_1": round(c["close"] - (0.5 * c["atr3"]), 2),
                        "entry_2": round(c["close"] - (1.0 * c["atr3"]), 2),
                    }
                )

        if all_hits:
            self.signals_db.save_screener_turnover_timing(all_hits)
            if days == 0:
                self._send_telegram_report(
                    "🔄 Turnover-Timing (Strict)", date_str, all_hits
                )

        logger.info(f"[{self.name}] Fertig: {len(all_hits)} Top-Aktien identifiziert.")
        return len(all_hits)

    def _determine_analysis_date(
        self, indicators: dict[str, Any], days: int
    ) -> tuple[pd.Timestamp | None, int]:
        close_df = indicators["close"]

        if days == 0:
            target_date = self._get_target_analysis_date()
            target_ts = pd.Timestamp(target_date)

            past_dates = close_df.index[close_df.index <= target_ts]

            if past_dates.empty:
                logger.warning(
                    f"[{self.name}] Keine Daten bis zum Ziel-Datum {target_date} gefunden."
                )
                return None, -1

            current_date_idx = past_dates[-1]
            idx_pos = close_df.index.get_loc(current_date_idx)

            logger.info(
                f"[{self.name}] Analysiere Stichtag: {current_date_idx.date()} (Ziel war {target_date})"
            )
            return current_date_idx, idx_pos

        else:
            total_len = len(close_df)
            # Puffer erhöhen
            idx_pos = max(150, total_len - days)
            return close_df.index[idx_pos], idx_pos

    def _scan_universe(
        self, symbol_list: list[str], ind: dict[str, Any], idx_pos: int, idx_name: str
    ) -> list[dict[str, Any]]:
        """
        Implementiert die strikte Logik gemäß PDF:
        1. Turnover berechnen für alle.
        2. Die Top 20 nach Turnover isolieren.
        3. Nur diese 20 auf SMA150 prüfen.
        """
        raw_candidates = []

        # Effizientere Lookup-Struktur
        close_slice = ind["close"].iloc[idx_pos]
        sma150_slice = ind["sma150"].iloc[idx_pos]
        turn_slice = ind["turnover_sma20"].iloc[idx_pos]
        atr3_slice = ind["atr3"].iloc[idx_pos]

        # Schritt 1: Daten für alle Symbole sammeln
        for symbol in symbol_list:
            if symbol not in close_slice:
                continue

            try:
                close = close_slice[symbol]
                sma150 = sma150_slice[symbol]
                turnover = turn_slice[symbol]
                atr3 = atr3_slice[symbol]

                if pd.isna(close) or pd.isna(sma150) or pd.isna(turnover):
                    continue

                # Wir speichern vorerst alle validen Datensätze
                raw_candidates.append(
                    {
                        "symbol": symbol,
                        "close": close,
                        "sma150": sma150,
                        "atr3": atr3,
                        "turnover_sma20": turnover,
                        "source_index": idx_name,
                    }
                )
            except Exception as e:
                logger.debug(f"Fehler bei Symbol {symbol}: {e}")

        # Schritt 2: "Die 20 Aktien mit dem höchsten Turnover suchen"
        # Sortieren nach Turnover Absteigend
        raw_candidates.sort(key=lambda x: x["turnover_sma20"], reverse=True)

        # Die Top 20 isolieren
        top_20_turnover = raw_candidates[:20]

        # Schritt 3: "Trendfilter: Diese Aktien müssen über ihrem 150-Tage Durchschnitt notieren"
        final_candidates = []
        for cand in top_20_turnover:
            if cand["close"] > cand["sma150"]:
                final_candidates.append(cand)

        # Diese Liste geht zurück an run() und wird dort auf max 4 beschnitten
        return final_candidates

    def _get_target_analysis_date(self) -> datetime.date:
        today = datetime.now().date()
        # 0=Mon ... 4=Fri, 5=Sat, 6=Sun
        # Wenn Wochenende -> Freitag dieser Woche (eigentlich Freitag davor)
        # Wenn Wochentag -> Freitag der Vorwoche
        weekday = today.weekday()
        offset = weekday - 4 if weekday >= 5 else weekday + 3
        return today - timedelta(days=offset)

    def _load_market_data(self, days: int) -> dict[str, pd.DataFrame] | None:
        start_date = (pd.Timestamp.now() - pd.Timedelta(days=days)).strftime("%Y-%m-%d")
        try:
            with sqlite3.connect(self.stocks_db_path) as conn:
                df = pd.read_sql_query(
                    f"SELECT date, symbol, open, high, low, close, volume "
                    f"FROM market_prices WHERE date >= '{start_date}' "
                    f"AND timeframe = '1D' ORDER BY date ASC",
                    conn,
                )
        except Exception as e:
            logger.error(f"{self.name}: DB Fehler: {e}")
            return None

        if df.empty:
            return None

        df["date"] = pd.to_datetime(df["date"])
        return {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

    def _calculate_indicators(self, data: dict[str, pd.DataFrame]) -> dict[str, Any]:
        close = data["close"]

        # [cite_start]UPDATE: SMA 150 statt SMA 100 [cite: 13, 15]
        sma150 = close.rolling(window=150, min_periods=120).mean()

        # [cite_start]Turnover SMA 20 (Preis * Volumen) [cite: 15]
        turnover_sma20 = (close * data["volume"]).rolling(window=20).mean()

        # [cite_start]ATR 3 (TradingView Style RMA 3 ~= EMA span 5) [cite: 17, 29]
        prev_close = close.shift(1)
        tr_values = np.maximum.reduce(
            [
                (data["high"] - data["low"]).values,
                (data["high"] - prev_close).abs().fillna(0).values,
                (data["low"] - prev_close).abs().fillna(0).values,
            ]
        )
        atr3 = (
            pd.DataFrame(tr_values, index=close.index, columns=close.columns)
            .ewm(span=5, adjust=False)
            .mean()
        )

        return {
            "close": close,
            "sma150": sma150,
            "turnover_sma20": turnover_sma20,
            "atr3": atr3,
        }
