import logging
import re
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from ....services.database import SignalDatabase
from ....services.telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class WebhookFilterStrategy(BaseStrategy):
    name = "WebhookFilter"

    def __init__(
        self,
        signals_db: SignalDatabase,
        config: dict[str, Any],
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        super().__init__(signals_db, telegram_bot)
        self.config = config
        self.strategies = config.get("strategy_ranking", [])
        self.execution_rules = config.get("execution_rules", {})
        self.seasonal_roadmap = config.get("seasonal_roadmap", {})

        self.daily_limit = self.execution_rules.get("daily_limit", 10)

    def run(self, days: int = 0) -> int:
        lookback = max(days, 1)
        logger.info(f"[{self.name}] Starte Analyse (Lookback: {lookback} Tage)...")

        if not self.strategies:
            logger.warning(f"[{self.name}] Keine Strategien in YAML konfiguriert.")
            return 0

        # Saisonalitäts-Check
        if not self._check_seasonality():
            return 0

        # Signale laden
        df = self._load_signals(lookback)
        if df.empty:
            return 0

        # Strategien anwenden
        final_df = self._apply_strategies(df)
        if final_df.empty:
            return 0

        # Ergebnisse verarbeiten
        results = self._generate_results(final_df)

        self.signals_db.save_screener_webhook(results)

        current_month = datetime.now().strftime("%B")
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
        self._send_telegram_report(
            f"🚀 High-Priority ({current_month})", start_date, results
        )

        return len(results)

    def _check_seasonality(self) -> bool:
        current_month = datetime.now().strftime("%B")
        if season_info := self.seasonal_roadmap.get(current_month):
            multiplier = season_info.get("position_size_multiplier", 1.0)
            if multiplier <= 0.0:
                msg = f"⚠️ **Trading Pausiert** ({current_month})\nGrund: {season_info.get('description', 'Saisonalität')}"
                logger.warning(msg)
                if self.telegram:
                    self.telegram.send(msg)
                return False
        return True

    def _load_signals(self, lookback: int) -> pd.DataFrame:
        start_date = (datetime.now() - timedelta(days=lookback)).strftime("%Y-%m-%d")
        sql = f"""
            SELECT symbol, signal, close, high, low, rsi, sma_200, sma_20,
                   dist_sma_20, dist_sma_200, exchange, timeframe,
                   trend, welle, wolke, kerze, status, setter,
                   date(timestamp) as date_str
            FROM signals
            WHERE date(timestamp) >= '{start_date}'
        """
        try:
            with self.signals_db._get_conn() as conn:
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"[{self.name}] DB Fehler: {e}")
            return pd.DataFrame()

        # Numeric Conversions
        numeric_cols = [
            "close",
            "high",
            "low",
            "rsi",
            "sma_200",
            "sma_20",
            "dist_sma_20",
            "dist_sma_200",
        ]
        for c in numeric_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

        return df

    def _apply_strategies(self, df: pd.DataFrame) -> pd.DataFrame:
        candidates = []

        for strat in self.strategies:
            strat_name = strat.get("signal_name")
            if not strat_name:
                continue

            matches = df[df["signal"] == strat_name].copy()
            if matches.empty:
                continue

            filters = strat.get("filters", {})
            filter_details = ", ".join([f"{k}: {v}" for k, v in filters.items()])

            # Filtern
            valid_mask = matches.apply(
                lambda row: self._check_row_filters(row, filters), axis=1
            )
            accepted = matches[valid_mask].copy()

            if not accepted.empty:
                accepted["rank"] = strat.get("rank", 999)
                accepted["filter_details"] = filter_details
                accepted["strategy_name"] = strat_name
                # Sortierkriterium
                accepted["sort_metric"] = accepted["dist_sma_20"]
                candidates.append(accepted)

        if not candidates:
            return pd.DataFrame()

        final_df = pd.concat(candidates, ignore_index=True)

        # Sortieren: Date DESC, Rank ASC, Metric ASC
        final_df = final_df.sort_values(
            by=["date_str", "rank", "sort_metric"], ascending=[False, True, True]
        )

        # Daily Limit
        final_df = final_df.groupby("date_str").head(self.daily_limit)
        return final_df

    def _check_row_filters(self, row: pd.Series, filters: dict[str, Any]) -> bool:
        for key, condition in filters.items():
            db_col = self._map_key(key)
            if db_col not in row:
                return False
            if not self._check_condition(row[db_col], condition):
                return False
        return True

    def _generate_results(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        results = []
        for _, row in df.iterrows():
            entry = row["close"]
            risk = entry * 0.01

            results.append(
                {
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
                    "tp1": round(entry + risk, 2),
                    "tp3": round(entry + (3 * risk), 2),
                    "sl": round(entry - risk, 2),
                }
            )
        return results

    def _map_key(self, yaml_key: str) -> str:
        key_map = {
            "dist_ema_20": "dist_sma_20",
            "dist_ema_200": "dist_sma_200",
            "RSI": "rsi",
        }
        return key_map.get(yaml_key, yaml_key)

    def _check_condition(self, market_value: Any, condition: str | int | float) -> bool:
        if isinstance(market_value, str):
            return market_value == str(condition).strip()

        if isinstance(condition, (int, float)):
            return market_value == condition

        cond_str = str(condition).strip()

        # Regex Match für "Bereich x bis y"
        if match := re.match(
            r"Bereich\s+(-?[\d\.]+)\s+bis\s+(-?[\d\.]+)", cond_str, re.IGNORECASE
        ):
            min_val, max_val = map(float, match.groups())
            return min_val <= market_value < max_val

        # Simple Operator Matches
        match cond_str:
            case s if s.startswith("<"):
                return market_value < float(s.replace("<", ""))
            case s if s.startswith(">"):
                return market_value > float(s.replace(">", ""))
            case _:
                return False
