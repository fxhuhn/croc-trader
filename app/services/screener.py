import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from ..config import settings
from ..mapping import mapper
from ..tools.symbol_lists import ExchangeSymbol
from .database import SignalDatabase

logger = logging.getLogger(__name__)


# ==============================================================================
# 1. BASIS KLASSE (Interface)
# ==============================================================================


class BaseStrategy:
    name: str = "Base"

    def __init__(self, signals_db: SignalDatabase, telegram_bot=None):
        self.signals_db = signals_db
        self.telegram = telegram_bot

    def run(self, days: int = 0) -> int:
        raise NotImplementedError("Jede Strategie muss 'run' implementieren.")

    def _get_exchange(self, symbol: str) -> str:
        return mapper.get_exchange(symbol, default="UNKNOWN")

    def _send_telegram_report(self, title, date, results):
        if not self.telegram or not results:
            return
        # Kompaktere Darstellung für Telegram
        msg = f"🔎 **{title}** ({date})\n"
        for r in results[:10]:  # Top 10 des neuesten Tages
            rank_info = f"[#{r.get('rank', '-')}] "
            msg += f"• {rank_info}{r['symbol']} ({r.get('strategy', 'Unknown')}): {r['close']}\n"

        # Hinweis bei vielen Ergebnissen über mehrere Tage
        if len(results) > 10:
            unique_days = len(set(r["date"] for r in results))
            if unique_days > 1:
                msg += f"\n... und weitere Treffer aus insgesamt {unique_days} Tagen."
            else:
                msg += f"... und {len(results) - 10} weitere."

        self.telegram.send(msg)


# ==============================================================================
# 2. STRATEGIE: DIP BUYER
# ==============================================================================


class DipBuyerStrategy(BaseStrategy):
    name = "DipBuyer"

    def __init__(
        self, stocks_db_path: Path, signals_db: SignalDatabase, telegram_bot=None
    ):
        super().__init__(signals_db, telegram_bot)
        self.stocks_db_path = stocks_db_path

    def run(self, days: int = 0) -> int:
        mode = "Backfill" if days > 0 else "Daily"
        logger.info(f"[{self.name}] Starte Analyse ({mode})...")

        data = self._load_market_data(days=400 + days)
        if not data:
            return 0

        ind = self._calculate_technical_indicators(data)
        total_len = len(ind["close"])
        start_idx = max(200, total_len - days) if days > 0 else total_len - 1

        all_results = []
        for i in range(start_idx, total_len):
            daily_hits = self._apply_logic(ind, idx_pos=i)
            if daily_hits:
                all_results.extend(daily_hits)
                if days > 0:
                    self._create_trades(daily_hits)

        if all_results:
            self.signals_db.save_screener_dip_buyer(all_results)
            if days == 0:
                self._create_trades(all_results)
                self._send_telegram_report(
                    "📉 Dip-Buyer", all_results[0]["date"], all_results
                )

        logger.info(f"[{self.name}] Fertig: {len(all_results)} Treffer.")
        return len(all_results)

    def _create_trades(self, results):
        for res in results:
            self.signals_db.add_trade(
                symbol=res["symbol"],
                entry_date=res["date"],
                entry_price=res["entry_limit"],
                atr_at_entry=res["atr5"],
                quantity=1,
                strategy=self.name,
            )

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
            logger.error(f"DipBuyer: Fehler beim Laden der Marktdaten: {e}")
            return None

        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])

        return {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

    def _calculate_technical_indicators(self, data: Dict[str, pd.DataFrame]):
        closes = data["close"]
        highs = data["high"]
        lows = data["low"]
        volumes = data["volume"]

        sma200 = closes.rolling(window=200, min_periods=150).mean()
        vol_sma20 = volumes.rolling(window=20).mean()

        delta = closes.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        prev_close = closes.shift(1)
        tr_values = np.maximum.reduce(
            [
                (highs - lows).values,
                (highs - prev_close).abs().fillna(0).values,
                (lows - prev_close).abs().fillna(0).values,
            ]
        )
        tr = pd.DataFrame(tr_values, index=closes.index, columns=closes.columns)
        atr5 = tr.ewm(span=(2 * 5) - 1, adjust=False).mean()

        atr_r3 = (closes - closes.shift(3)) / atr5.replace(0, np.nan)
        ibs = (closes - lows) / (highs - lows).replace(0, 0.01)

        return {
            "sma200": sma200,
            "vol_sma20": vol_sma20,
            "rsi": rsi.fillna(50),
            "atr5": atr5,
            "atr_r3": atr_r3,
            "setup_score": atr_r3 * -1,
            "ibs": ibs,
            "entry_limits": closes - atr5,
            "close": closes,
            "open": data["open"],
            "high": highs,
        }

    def _apply_logic(self, ind, idx_pos: int) -> List[Dict]:
        i_vol_sma = ind["vol_sma20"].iloc[idx_pos]
        i_close = ind["close"].iloc[idx_pos]
        i_open = ind["open"].iloc[idx_pos]
        i_prev_close = ind["close"].iloc[idx_pos - 1]
        i_prev_open = ind["open"].iloc[idx_pos - 1]
        i_sma200 = ind["sma200"].iloc[idx_pos]
        i_atr_r3 = ind["atr_r3"].iloc[idx_pos]
        i_atr5 = ind["atr5"].iloc[idx_pos]
        i_ibs = ind["ibs"].iloc[idx_pos]

        mask = (
            (i_vol_sma > 1_000_000)
            & (i_close > 5.0)
            & (i_close > i_sma200)
            & (i_atr_r3 < -1.0)
            & ((i_atr5 / i_close) > 0.03)
            & (i_close < i_open)
            & (i_prev_close < i_prev_open)
            & (i_ibs < 0.2)
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


# ==============================================================================
# 3. STRATEGIE: TURNOVER TIMING (NEU)
# ==============================================================================


class TurnoverTimingStrategy(BaseStrategy):
    name = "TurnoverTiming"

    def __init__(
        self, stocks_db_path: Path, signals_db: SignalDatabase, telegram_bot=None
    ):
        super().__init__(signals_db, telegram_bot)
        self.stocks_db_path = stocks_db_path
        # Singleton Instanz für Symbol-Listen laden
        self.universe = ExchangeSymbol()

    def run(self, days: int = 0) -> int:
        """
        Führt das Screening durch.
        Logik:
        1. Laden aller Marktdaten.
        2. Indikatoren berechnen (SMA100, ATR3, SMA20(Turnover)).
        3. Datum bestimmen (Freitag der aktuellen oder letzten Woche).
        4. Pro Index (Dow, SP500, Nasdaq) filtern und Top 4 nach Turnover wählen.
        """
        logger.info(f"[{self.name}] Starte Analyse...")

        # 1. Daten laden (Ausreichend Historie für SMA100)
        data = self._load_market_data(days=365 + days)
        if not data:
            return 0

        # 2. Indikatoren berechnen
        ind = self._calculate_indicators(data)

        # 3. Bestimmung des Analyse-Zeitpunkts
        if days == 0:
            # AUTO-MODUS: Wir suchen den korrekten Freitag
            target_date = self._get_target_analysis_date()
            target_ts = pd.Timestamp(target_date)

            # Alle verfügbaren Handelsdaten
            available_dates = ind["close"].index

            # Filtern: Nur Daten VOR oder AM Zieltag (wir wollen nicht in die Zukunft schauen, falls DB neuer ist)
            # und wir wollen den letzten verfügbaren Tag DAVOR (z.B. Donnerstag, falls Freitag Feiertag war)
            past_dates = available_dates[available_dates <= target_ts]

            if past_dates.empty:
                logger.warning(
                    f"[{self.name}] Keine Daten bis zum Ziel-Datum {target_date} gefunden."
                )
                return 0

            current_date_idx = past_dates[-1]

            # Kurzer Check, ob die Daten nicht uralt sind (z.B. > 5 Tage Differenz zum Ziel)
            diff_days = (target_ts - current_date_idx).days
            if diff_days > 5:
                logger.warning(
                    f"[{self.name}] WARNUNG: Gefundenes Datum {current_date_idx.date()} ist {diff_days} Tage älter als Ziel {target_date}."
                )

            idx_pos = ind["close"].index.get_loc(current_date_idx)
            date_str = current_date_idx.strftime("%Y-%m-%d")
            logger.info(
                f"[{self.name}] Analysiere Stichtag: {date_str} (Ziel war {target_date})"
            )

        else:
            # BACKTEST / OFFSET MODUS: Einfach Offset vom Ende
            total_len = len(ind["close"])
            idx_pos = max(100, total_len - days) if days > 0 else total_len - 1
            current_date_idx = ind["close"].index[idx_pos]
            date_str = current_date_idx.strftime("%Y-%m-%d")

        all_hits = []

        # Definition der zu prüfenden Universen
        universes = [
            ("DOW-30", self.universe.dow_30),
            ("NASDAQ-100", self.universe.nasdaq_100),
            ("SP-500", self.universe.sp_500),
        ]

        processed_symbols = set()  # Um Duplikate zu vermeiden, falls gewünscht

        for idx_name, symbol_list in universes:
            candidates = []

            for symbol in symbol_list:
                # Prüfen ob Symbol in Daten vorhanden
                if symbol not in ind["close"].columns:
                    continue

                try:
                    # Werte holen
                    close = ind["close"].iloc[idx_pos].get(symbol)
                    sma100 = ind["sma100"].iloc[idx_pos].get(symbol)
                    turnover_sma20 = ind["turnover_sma20"].iloc[idx_pos].get(symbol)
                    atr3 = ind["atr3"].iloc[idx_pos].get(symbol)

                    if pd.isna(close) or pd.isna(sma100) or pd.isna(turnover_sma20):
                        continue

                    # REGEL 1: Aktie muss über dem SMA100 sein
                    if close > sma100:
                        candidates.append(
                            {
                                "symbol": symbol,
                                "close": close,
                                "atr3": atr3,
                                "turnover_sma20": turnover_sma20,
                                "source_index": idx_name,
                            }
                        )
                except Exception:
                    continue

            # REGEL 2: Sortiere nach sma20*(Close * Volumen) -> Turnover SMA
            candidates.sort(key=lambda x: x["turnover_sma20"], reverse=True)

            # REGEL 3: Top 4 auswählen
            top_4 = candidates[:4]

            for c in top_4:
                # Duplikat-Check (Ein Symbol kann im S&P500 und Nasdaq100 sein)
                if c["symbol"] in processed_symbols:
                    continue

                processed_symbols.add(c["symbol"])

                # Entry Berechnung
                entry_1 = c["close"] - (0.5 * c["atr3"])
                entry_2 = c["close"] - (1.0 * c["atr3"])

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
                        "entry_1": round(entry_1, 2),
                        "entry_2": round(entry_2, 2),
                    }
                )

        # Speichern
        if all_hits:
            self.signals_db.save_screener_turnover_timing(all_hits)
            if days == 0:
                self._send_telegram_report("🔄 Turnover-Timing", date_str, all_hits)

        logger.info(f"[{self.name}] Fertig: {len(all_hits)} Top-Aktien identifiziert.")
        return len(all_hits)

    def _get_target_analysis_date(self) -> datetime.date:
        """
        Ermittelt den Stichtag für die Analyse basierend auf dem aktuellen Wochentag.
        Regel:
        - Wochenende (Sa/So) -> Freitag dieser Woche
        - Wochentag (Mo-Fr)  -> Freitag der Vorwoche
        """
        today = datetime.now().date()
        weekday = today.weekday()  # 0=Mon, 4=Fri, 5=Sat, 6=Sun

        if weekday >= 5:  # Wochenende (Sa, So)
            offset = weekday - 4  # Sa(5)->1, So(6)->2
        else:  # Wochentag (Mo-Fr)
            offset = weekday + 3  # Mo(0)->3, Di(1)->4 ... Fr(4)->7

        target = today - timedelta(days=offset)
        return target

    def _load_market_data(self, days=250) -> Optional[Dict[str, pd.DataFrame]]:
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
            logger.error(f"{self.name}: DB Fehler: {e}")
            return None

        if df.empty:
            return None
        df["date"] = pd.to_datetime(df["date"])

        return {
            col: df.pivot(index="date", columns="symbol", values=col)
            for col in ["open", "high", "low", "close", "volume"]
        }

    def _calculate_indicators(self, data: Dict[str, pd.DataFrame]):
        closes = data["close"]
        highs = data["high"]
        lows = data["low"]
        volumes = data["volume"]

        # SMA 100
        sma100 = closes.rolling(window=100, min_periods=80).mean()

        # Turnover & Turnover SMA20
        turnover = closes * volumes
        turnover_sma20 = turnover.rolling(window=20).mean()

        # ATR 3 Berechnung
        prev_close = closes.shift(1)
        tr_values = np.maximum.reduce(
            [
                (highs - lows).values,
                (highs - prev_close).abs().fillna(0).values,
                (lows - prev_close).abs().fillna(0).values,
            ]
        )
        tr = pd.DataFrame(tr_values, index=closes.index, columns=closes.columns)
        atr3 = tr.ewm(span=(2 * 3) - 1, adjust=False).mean()

        return {
            "close": closes,
            "sma100": sma100,
            "turnover_sma20": turnover_sma20,
            "atr3": atr3,
        }


# ==============================================================================
# 4. STRATEGIE: WEBHOOK / YAML FILTER
# ==============================================================================


class WebhookFilterStrategy(BaseStrategy):
    name = "WebhookFilter"

    def __init__(self, signals_db: SignalDatabase, config: Dict, telegram_bot=None):
        super().__init__(signals_db, telegram_bot)

        self.config = config
        self.strategies = config.get("strategy_ranking", [])
        self.seasonal_roadmap = config.get("seasonal_roadmap", {})
        self.execution_rules = config.get("execution_rules", {})

        self.daily_limit = self.execution_rules.get("daily_limit", 10)
        self.management_style = self.execution_rules.get("management_style", "STANDARD")

    def run(self, days: int = 0) -> int:
        lookback = days if days > 0 else 1
        logger.info(f"[{self.name}] Starte Analyse (Lookback: {lookback} Tage)...")

        if not self.strategies:
            logger.warning(f"[{self.name}] Keine Strategien in YAML gefunden.")
            return 0

        # --- SCHRITT A: SAISONALITÄTS-CHECK ---
        current_month = datetime.now().strftime("%B")
        season_info = self.seasonal_roadmap.get(current_month)

        if season_info:
            multiplier = season_info.get("position_size_multiplier", 1.0)
            if multiplier <= 0.0:
                msg = f"⚠️ **Trading Pausiert** ({current_month})\nGrund: {season_info.get('description', 'Saisonalität')}"
                logger.warning(msg)
                if self.telegram:
                    self.telegram.send(msg)
                return 0

            logger.info(
                f"Saisonalität: {current_month} (Faktor {multiplier}) - {season_info.get('description')}"
            )

        # --- SCHRITT B: SIGNAL SCANNING ---
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")

        try:
            with self.signals_db._get_conn() as conn:
                sql = f"""
                    SELECT symbol, signal, close, high, low, rsi, sma_200, sma_20,
                           dist_sma_20, dist_sma_200, exchange, timeframe,
                           trend, welle, wolke, kerze, status, setter,
                           date(timestamp) as date_str
                    FROM signals
                    WHERE date(timestamp) >= '{start_date}'
                """
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"[{self.name}] DB Fehler: {e}")
            return 0

        if df.empty:
            logger.info(f"[{self.name}] Keine Signale im Zeitraum gefunden.")
            return 0

        cols = [
            "close",
            "high",
            "low",
            "rsi",
            "sma_200",
            "sma_20",
            "dist_sma_20",
            "dist_sma_200",
        ]
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        # --- SCHRITT C: STRATEGIE-MATCHING & FILTERING ---
        candidates = []

        for strat in self.strategies:
            strat_name = strat.get("signal_name")
            if not strat_name:
                continue

            # 1. Basis-Filter
            matches = df[df["signal"] == strat_name].copy()

            if matches.empty:
                logger.warning(
                    f"[{self.name}] Strategie '{strat_name}': Keine Einträge für Signal-Name '{strat_name}' gefunden."
                )
                continue

            # 2. Erweiterte Filter
            filters = strat.get("filters", {})
            filter_details_str = ", ".join([f"{k}: {v}" for k, v in filters.items()])

            valid_indices = []

            for idx, row in matches.iterrows():
                is_valid = True
                for key, condition in filters.items():
                    db_col = self._map_key(key)
                    if db_col not in row:
                        is_valid = False
                        break

                    market_value = row[db_col]
                    if not self._check_condition(market_value, condition):
                        is_valid = False
                        break

                if is_valid:
                    valid_indices.append(idx)

            if valid_indices:
                accepted = matches.loc[valid_indices].copy()
                accepted["rank"] = strat.get("rank", 999)
                accepted["filter_details"] = filter_details_str
                accepted["strategy_name"] = strat_name
                accepted["sort_metric"] = accepted["dist_sma_20"]
                candidates.append(accepted)
            else:
                logger.info(
                    f"[{self.name}] Strategie '{strat_name}': Signale vorhanden, aber Filter nicht erfüllt."
                )

        if not candidates:
            return 0

        final_df = pd.concat(candidates, ignore_index=True)

        # --- SCHRITT D: PRIORISIERUNG & RANKING (UPDATED) ---

        # 1. Sortieren: Erst Datum, dann Rank, dann Metrik
        #    Damit stellen wir sicher, dass bei der Groupierung die "Besten" oben stehen.
        final_df = final_df.sort_values(
            by=["date_str", "rank", "sort_metric"], ascending=[False, True, True]
        )

        # 2. Limitierung PRO TAG anwenden
        #    Wir gruppieren nach Datum und nehmen jeweils nur die Top X (daily_limit)
        before_count = len(final_df)
        final_df = final_df.groupby("date_str").head(self.daily_limit)

        if len(final_df) < before_count:
            logger.info(
                f"Limitierung: {before_count} Kandidaten auf {len(final_df)} gekürzt (Max {self.daily_limit} pro Tag)."
            )

        # --- SCHRITT E: OUTPUT-GENERIERUNG ---
        results = []
        for _, row in final_df.iterrows():
            entry = row["close"]
            risk_unit = entry * 0.01
            tp1 = entry + (1 * risk_unit)
            tp3 = entry + (3 * risk_unit)
            sl = entry - (1 * risk_unit)

            res = {
                "date": row["date_str"],
                "symbol": row["symbol"],
                "exchange": row["exchange"] or self._get_exchange(row["symbol"]),
                "timeframe": row.get("timeframe", "1D"),
                "strategy": row["strategy_name"],
                "signal": row["signal"],
                "close": round(row["close"], 2),
                "high": round(row["high"], 2),
                "low": round(row["low"], 2),
                "rsi": round(row["rsi"], 2),
                "sma_200": round(row["sma_200"], 2),
                "sma_20": round(row.get("sma_20", 0), 2),
                "rank": int(row.get("rank", 999)),
                "filter_details": row.get("filter_details", ""),
                "tp1": round(tp1, 2),
                "tp3": round(tp3, 2),
                "sl": round(sl, 2),
            }
            results.append(res)

        self.signals_db.save_screener_webhook(results)
        self._send_telegram_report(
            f"🚀 High-Priority ({current_month})", start_date, results
        )

        logger.info(f"[{self.name}] Fertig: {len(results)} Top-Kandidaten ausgewählt.")
        return len(results)

    def _map_key(self, yaml_key: str) -> str:
        key_map = {
            "dist_ema_20": "dist_sma_20",
            "dist_ema_200": "dist_sma_200",
            "RSI": "rsi",
            "trend": "trend",
            "welle": "welle",
            "wolke": "wolke",
            "kerze": "kerze",
            "status": "status",
            "setter": "setter",
        }
        return key_map.get(yaml_key, yaml_key)

    def _check_condition(self, market_value, condition_string: str) -> bool:
        try:
            if isinstance(market_value, str):
                return market_value == condition_string.strip()

            if not isinstance(condition_string, str):
                return market_value == condition_string

            condition_string = condition_string.strip()

            match_range = re.match(
                r"Bereich\s+(-?[\d\.]+)\s+bis\s+(-?[\d\.]+)",
                condition_string,
                re.IGNORECASE,
            )
            if match_range:
                min_val, max_val = map(float, match_range.groups())
                return min_val <= market_value < max_val

            if condition_string.startswith("<"):
                threshold = float(condition_string.replace("<", "").strip())
                return market_value < threshold

            if condition_string.startswith(">"):
                threshold = float(condition_string.replace(">", "").strip())
                return market_value > threshold

            return False
        except Exception:
            return False


# ==============================================================================
# 5. STRATEGIE: CROC SETUP (NEU)
# ==============================================================================


class CrocSetupStrategy(BaseStrategy):
    name = "CrocSetup"

    def __init__(self, signals_db: SignalDatabase, telegram_bot=None):
        super().__init__(signals_db, telegram_bot)
        # NEU: Nutzung der zentralen Config für saubere Pfad-Auflösung in data/
        self.config_path = settings.get_path("ranking_yaml")
        self.ranking_rules = self._load_config()

    def _load_config(self) -> List[Dict]:
        if not self.config_path.exists():
            logger.error(
                f"[{self.name}] Konfigurationsdatei nicht gefunden: {self.config_path}"
            )
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return data.get("ranking_2026", [])
        except Exception as e:
            logger.error(f"[{self.name}] Fehler beim Laden der YAML: {e}")
            return []

    def run(self, days: int = 0) -> int:
        # 1. Zeitraum definieren (Standard: 10 Tage Scan)
        scan_days = 10 if days <= 0 else days
        start_date = (datetime.now() - timedelta(days=scan_days)).strftime("%Y-%m-%d")

        logger.info(f"[{self.name}] Starte Scan ab {start_date}...")

        # 2. Signale laden
        try:
            with self.signals_db._get_conn() as conn:
                # Wir holen explizit Spalten, die für Filter relevant sein könnten
                # WICHTIG: 'deluxe' ist jetzt im SELECT enthalten
                # NEU: Filter auf timeframe = '1D'
                sql = f"""
                    SELECT symbol, signal, exchange, timeframe, close, high, low, rsi,
                           dist_sma_20, dist_sma_200,
                           kerze, welle, deluxe, status, trend,
                           date(timestamp) as date_str
                    FROM signals
                    WHERE date(timestamp) >= '{start_date}' AND timeframe = '1D'
                """
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"[{self.name}] DB Fehler beim Laden der Signale: {e}")
            return 0

        if df.empty:
            logger.info(f"[{self.name}] Keine Signale gefunden.")
            return 0

        # Vorverarbeitung
        numeric_cols = ["close", "high", "low", "rsi", "dist_sma_20", "dist_sma_200"]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

        results = []

        # 3. Matching Logik
        # Iteration über jedes Signal in der DB
        for _, row in df.iterrows():
            signal_name = row["signal"]

            # Suche passenden Eintrag in der YAML
            rule = next(
                (r for r in self.ranking_rules if r["signal"] == signal_name), None
            )
            if not rule:
                continue

            match_found = False
            active_r = rule.get("base_r_per_trade", 0.0)
            match_reason = ""

            filters = rule.get("filters", {})
            if not filters:
                continue

            # A) Check EMA
            if best_ema := filters.get("best_ema"):
                # Prüfe dist_sma_200
                if cond := best_ema.get("ema_200"):
                    if self._check_range_condition(row.get("dist_sma_200"), cond):
                        match_found = True
                        active_r = best_ema.get("r_per_trade", active_r)
                        match_reason = f"EMA 200 ({cond})"

                # Prüfe dist_sma_20 (falls noch kein Match oder als Alternative)
                if not match_found and (cond := best_ema.get("ema_20")):
                    if self._check_range_condition(row.get("dist_sma_20"), cond):
                        match_found = True
                        active_r = best_ema.get("r_per_trade", active_r)
                        match_reason = f"EMA 20 ({cond})"

            # B) Check RSI (ODER Verknüpfung)
            if not match_found and (best_rsi := filters.get("best_rsi")):
                if cond := best_rsi.get("rsi"):
                    if self._check_rsi_zone(row.get("rsi"), cond):
                        match_found = True
                        active_r = best_rsi.get("r_per_trade", active_r)
                        match_reason = f"RSI ({cond})"

            # C) Check Extra (ODER Verknüpfung)
            if not match_found and (best_extra := filters.get("best_extra")):
                # Extra Iteriert über keys wie 'kerze', 'welle', 'status' etc.
                for col_key, target_val in best_extra.items():
                    if col_key in ["r_per_trade", "active"]:
                        continue  # Metadaten überspringen

                    current_val = row.get(col_key)
                    if current_val == target_val:
                        match_found = True
                        active_r = best_extra.get("r_per_trade", active_r)
                        match_reason = f"{col_key.capitalize()} ({target_val})"
                        break

            if match_found:
                logger.info(
                    f"[{self.name}] Treffer: {row['symbol']} ({signal_name}) via {match_reason}"
                )

                # Dist EMA auswählen für Display (bevorzugt 200er wenn verfügbar, sonst 20er)
                display_ema = (
                    row.get("dist_sma_200")
                    if pd.notna(row.get("dist_sma_200"))
                    else row.get("dist_sma_20")
                )

                results.append(
                    {
                        "date": row["date_str"],
                        "symbol": row["symbol"],
                        "exchange": row["exchange"] or "UNKNOWN",
                        "timeframe": row["timeframe"],
                        "signal": signal_name,
                        "rank": rule.get("rank", 999),
                        "r_per_trade": float(active_r),
                        "recommended_strategy": rule.get(
                            "recommended_strategy", "Standard"
                        ),
                        "close": float(row["close"] or 0.0),
                        "high": float(row["high"] or 0.0),
                        "low": float(row["low"] or 0.0),
                        "rsi": float(row["rsi"] or 0.0),
                        "dist_ema": float(display_ema or 0.0),
                        "match_filter": match_reason,
                    }
                )

        # 4. Speichern
        if results:
            self.signals_db.save_screener_croc(results)
            self._send_telegram_report(
                f"🐊 Croc Setup ({len(results)} Hits)", start_date, results
            )

        logger.info(f"[{self.name}] Fertig: {len(results)} neue Setups identifiziert.")
        return len(results)

    def _check_range_condition(self, value: Optional[float], condition: str) -> bool:
        """Parsen von Bedingungen wie '-3 to 0%', '> 10%', '< -5%'."""
        if value is None or pd.isna(value):
            return False

        condition = condition.replace("%", "").strip()

        # Case: Range "X to Y"
        if " to " in condition:
            try:
                min_s, max_s = condition.split(" to ")
                return float(min_s) <= value <= float(max_s)
            except ValueError:
                return False

        # Case: Greater "> X"
        if condition.startswith(">"):
            try:
                return value > float(condition.replace(">", ""))
            except ValueError:
                return False

        # Case: Less "< X"
        if condition.startswith("<"):
            try:
                return value < float(condition.replace("<", ""))
            except ValueError:
                return False

        return False

    def _check_rsi_zone(self, rsi: Optional[float], zone_name: str) -> bool:
        """Mapping der Text-Zonen zu Zahlenwerten."""
        if rsi is None or pd.isna(rsi):
            return False

        zone_name = zone_name.lower()

        if "oversold" in zone_name or "<30" in zone_name:
            return rsi < 30
        elif "weak" in zone_name:  # 30-45
            return 30 <= rsi <= 45
        elif "neutral" in zone_name:  # 45-55
            return 45 <= rsi <= 55
        elif "strong" in zone_name:  # 55-70
            return 55 <= rsi <= 70
        elif "overbought" in zone_name or ">70" in zone_name:
            return rsi > 70
        else:
            return False


# ==============================================================================
# 6. SCREENER ENGINE (Manager)
# ==============================================================================


class ScreenerEngine:
    def __init__(
        self,
        stocks_db_path: Path,
        signals_db_path: Path,
        config: Dict = None,
        telegram_bot=None,
    ):
        self.signals_db = SignalDatabase(signals_db_path)
        self.telegram = telegram_bot
        self.active_strategies: List[BaseStrategy] = []

        if isinstance(config, list):
            config = {"strategy_ranking": config}
        elif config is None:
            config = {}

        self.register_strategy(
            DipBuyerStrategy(stocks_db_path, self.signals_db, self.telegram)
        )
        self.register_strategy(
            WebhookFilterStrategy(self.signals_db, config, self.telegram)
        )
        self.register_strategy(
            TurnoverTimingStrategy(stocks_db_path, self.signals_db, self.telegram)
        )
        # NEU: Registrierung CrocSetupStrategy
        self.register_strategy(CrocSetupStrategy(self.signals_db, self.telegram))

    def register_strategy(self, strategy: BaseStrategy):
        self.active_strategies.append(strategy)
        logger.debug(f"Strategie registriert: {strategy.name}")

    def run_all(self, days: int = 0) -> Dict[str, int]:
        results = {}
        for strat in self.active_strategies:
            try:
                hits = strat.run(days=days)
                results[strat.name] = hits
            except Exception as e:
                logger.error(
                    f"Fehler beim Ausführen von {strat.name}: {e}", exc_info=True
                )
                results[strat.name] = 0
        return results
