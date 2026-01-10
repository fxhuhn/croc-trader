import logging
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..mapping import mapper
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
        for r in results[:10]:  # Top 10
            # Wir zeigen nun auch den Rank im Telegram
            rank_info = f"[#{r.get('rank', '-')}] "
            msg += f"• {rank_info}{r['symbol']} ({r['strategy']}): {r['close']}\n"
        if len(results) > 10:
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
        atr5 = tr.ewm(span=9, adjust=False).mean()

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
# 3. STRATEGIE: WEBHOOK / YAML FILTER
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

            # --- NEU: Dokumentation der Filter für die DB ---
            # Erstellt String wie "dist_ema_20: < 20, RSI: < 70"
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

                # --- NEU: Daten für Ranking & Doku speichern ---
                accepted["rank"] = strat.get("rank", 999)
                accepted["filter_details"] = filter_details_str  # Die Doku-Spalte

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

        # --- SCHRITT D: PRIORISIERUNG & RANKING ---
        final_df = final_df.sort_values(
            by=["rank", "sort_metric"], ascending=[True, True]
        )

        if len(final_df) > self.daily_limit:
            logger.info(
                f"Limitierung: {len(final_df)} Kandidaten auf {self.daily_limit} gekürzt."
            )
            final_df = final_df.head(self.daily_limit)

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
                # --- NEU: Rank & Doku übergeben ---
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
# 4. SCREENER ENGINE (Manager)
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
