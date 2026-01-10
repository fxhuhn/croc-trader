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
        msg = f"🔎 **{title}**\nDatum: {date}\nTreffer: {len(results)}\n"
        for r in results[:5]:
            msg += f"- {r['symbol']}\n"
        if len(results) > 5:
            msg += f"... und {len(results) - 5} weitere."
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
                strategy=self.name,  # WICHTIG: Strategie-Name wird hier übergeben
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

    def __init__(
        self, signals_db: SignalDatabase, strategies: List[Dict], telegram_bot=None
    ):
        super().__init__(signals_db, telegram_bot)
        self.strategies = strategies

    def run(self, days: int = 0) -> int:
        lookback = days if days > 0 else 1
        logger.info(f"[{self.name}] Verarbeite Signale der letzten {lookback} Tage...")

        if not self.strategies:
            logger.info(f"[{self.name}] Keine YAML-Strategien geladen.")
            return 0

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

        total_matches = 0
        for strat in self.strategies:
            total_matches += self._process_single_strategy(strat, df, lookback)

        logger.info(f"[{self.name}] Fertig: {total_matches} Treffer.")
        return total_matches

    def _process_single_strategy(
        self, strat: Dict, df: pd.DataFrame, lookback_days: int
    ) -> int:
        name = strat.get("name", "Unbekannt")
        logic = self._parse_logic(strat)

        if not logic:
            return 0

        try:
            matches = df.query(logic).copy()

            if matches.empty:
                if name in df["signal"].values:
                    logger.info(
                        f"Strategie '{name}': Signale vorhanden, aber Filter-Kriterien nicht erfüllt."
                    )
                else:
                    logger.warning(
                        f"Strategie '{name}': Keine Einträge für Signal-Name '{name}' gefunden."
                    )
                return 0

            results = []
            for _, row in matches.iterrows():
                results.append(
                    {
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
                        "sma_20": round(row.get("sma_20", 0), 2),
                    }
                )

            self.signals_db.save_screener_webhook(results)
            self._send_telegram_report(
                f"🚀 Signal: {name}", f"Letzte {lookback_days} Tage", results
            )
            return len(results)

        except Exception as e:
            logger.error(f"Fehler in Strategie '{name}': {e} | Logik: {logic}")
            return 0

    def _parse_logic(self, strat: Dict) -> str:
        conditions = []
        name = strat.get("name")
        if name:
            safe_name = name.replace("'", "\\'")
            conditions.append(f"(signal == '{safe_name}')")

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

        for k, v in strat.items():
            db_col = key_map.get(k, k)
            if k in ["name", "win_rate", "trades", "source"]:
                continue

            if isinstance(v, str):
                v = v.strip()
                match_range = re.match(
                    r"Bereich\s+(-?[\d\.]+)\s+bis\s+(-?[\d\.]+)", v, re.IGNORECASE
                )
                if match_range:
                    min_v, max_v = match_range.groups()
                    conditions.append(f"({db_col} >= {min_v} and {db_col} <= {max_v})")
                    continue
                match_op = re.match(r"([<>])\s*(-?[\d\.]+)", v)
                if match_op:
                    op, val = match_op.groups()
                    conditions.append(f"({db_col} {op} {val})")
                    continue
                conditions.append(f"({db_col} == '{v}')")

        return " and ".join(conditions) if conditions else ""


# ==============================================================================
# 4. SCREENER ENGINE (Manager)
# ==============================================================================


class ScreenerEngine:
    def __init__(
        self,
        stocks_db_path: Path,
        signals_db_path: Path,
        strategies: List[Dict] = None,
        telegram_bot=None,
    ):
        self.signals_db = SignalDatabase(signals_db_path)
        self.telegram = telegram_bot
        self.active_strategies: List[BaseStrategy] = []

        # Strategien registrieren
        self.register_strategy(
            DipBuyerStrategy(stocks_db_path, self.signals_db, self.telegram)
        )
        self.register_strategy(
            WebhookFilterStrategy(self.signals_db, strategies or [], self.telegram)
        )

    def register_strategy(self, strategy: BaseStrategy):
        self.active_strategies.append(strategy)
        logger.debug(f"Strategie registriert: {strategy.name}")

    def run_all(self, days: int = 0) -> Dict[str, int]:
        """Führt ALLE registrierten Strategien aus."""
        results = {}
        for strat in self.active_strategies:
            try:
                hits = strat.run(days=days)
                results[strat.name] = hits
            except Exception as e:
                logger.error(f"Fehler beim Ausführen von {strat.name}: {e}")
                results[strat.name] = 0
        return results
