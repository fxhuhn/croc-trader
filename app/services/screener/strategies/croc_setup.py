import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ....config import settings
from ....services.database import SignalDatabase
from ....services.telegram import TelegramBot
from .base import BaseStrategy

logger = logging.getLogger(__name__)


class CrocSetupStrategy(BaseStrategy):
    name = "CrocSetup"

    def __init__(
        self, signals_db: SignalDatabase, telegram_bot: TelegramBot | None = None
    ) -> None:
        super().__init__(signals_db, telegram_bot)
        self.config_path: Path = settings.get_path("ranking_yaml")
        self.ranking_rules = self._load_config()

    def _load_config(self) -> list[dict[str, Any]]:
        if not self.config_path.exists():
            logger.error(f"[{self.name}] Konfigurationsdatei fehlt: {self.config_path}")
            return []

        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

            if isinstance(data, list):
                return data
            return data.get("ranking_2026", [])
        except Exception as e:
            logger.error(f"[{self.name}] YAML Fehler: {e}")
            return []

    def run(self, days: int = 0) -> int:
        scan_days = 10 if days <= 0 else days
        start_date = (datetime.now() - timedelta(days=scan_days)).strftime("%Y-%m-%d")

        logger.info(f"[{self.name}] Starte Scan ab {start_date}...")

        df = self._load_signals(start_date)
        if df.empty:
            return 0

        results = self._process_signals(df)

        if results:
            self.signals_db.save_screener_croc(results)

            # --- NEU: Automatische Trade-Erstellung ---
            if days == 0:
                self._create_trades(results)

            self._send_telegram_report(
                f"🐊 Croc Setup ({len(results)} Hits)", start_date, results
            )

        return len(results)

    def _create_trades(self, results: list[dict[str, Any]]) -> None:
        """Erstellt CREATED Trades in der DB für T+1 Entry."""
        for res in results:
            # Entry Logic: Stop Buy @ High der Signal-Kerze
            entry_price = res["high"]

            # Risiko-Proxy: High - Low (da ATR hier nicht direkt vorliegt)
            # Wird von Strategien (Moonbag/Split) später präzisiert
            risk_range = res["high"] - res["low"]

            # Strategie-Zuweisung aus Ranking (z.B. Moonbag, SplitTarget)
            strategy_name = res.get("recommended_strategy", "MANUAL")

            self.signals_db.add_trade(
                symbol=res["symbol"],
                signal_date=res["date"],  # WICHTIG: Signal Date
                entry_price=entry_price,
                atr_at_entry=risk_range,  # Proxy
                quantity=1,
                strategy=strategy_name,
            )

    def _load_signals(self, start_date: str) -> pd.DataFrame:
        sql = f"""
            SELECT symbol, signal, exchange, timeframe, close, high, low, rsi,
                   dist_sma_20, dist_sma_200,
                   kerze, welle, deluxe, status, trend, wolke, setter,
                   date(timestamp) as date_str
            FROM signals
            WHERE date(timestamp) >= '{start_date}' AND timeframe = '1D'
        """
        try:
            with self.signals_db._get_conn() as conn:
                df = pd.read_sql_query(sql, conn)
        except Exception as e:
            logger.error(f"[{self.name}] DB Fehler: {e}")
            return pd.DataFrame()

        # Numeric conversions
        for c in ["close", "high", "low", "rsi", "dist_sma_20", "dist_sma_200"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _process_signals(self, df: pd.DataFrame) -> list[dict[str, Any]]:
        results = []
        meta_keys = {"Signal", "Exit", "Score", "Status", "R_Hist", "Risk_95", "R_2026"}

        for _, row in df.iterrows():
            signal_name = row["signal"]

            candidates = [
                r for r in self.ranking_rules if r.get("Signal") == signal_name
            ]
            if not candidates:
                continue

            best_match = None
            best_score = -999.0
            match_reason = []

            for rule in candidates:
                passed, reasons = self._evaluate_rule(row, rule, meta_keys)

                if passed:
                    score = float(rule.get("Score", 0.0))
                    if best_match is None or score > best_score:
                        best_match = rule
                        best_score = score
                        match_reason = reasons

            if best_match:
                results.append(self._build_result_entry(row, best_match, match_reason))

        return results

    def _evaluate_rule(
        self, row: pd.Series, rule: dict[str, Any], meta_keys: set[str]
    ) -> tuple[bool, list[str]]:
        reasons = []
        for key, value in rule.items():
            if key in meta_keys:
                continue

            db_col = self._map_key(key)
            market_val = row.get(db_col)

            if not self._check_condition(market_val, value, key):
                return False, []

            reasons.append(f"{key}={value}")

        return True, reasons

    def _build_result_entry(
        self, row: pd.Series, match: dict[str, Any], reason: list[str]
    ) -> dict[str, Any]:
        display_ema = (
            row.get("dist_sma_200")
            if pd.notna(row.get("dist_sma_200"))
            else row.get("dist_sma_20")
        )
        r_val = match.get("R_Hist", match.get("R_2026", 0.0))
        reason_str = ", ".join(reason) if reason else "Basis-Signal (Match)"

        def safe_float(val):
            return float(val) if pd.notna(val) else 0.0

        return {
            "date": row["date_str"],
            "symbol": row["symbol"],
            "exchange": row["exchange"] or "UNKNOWN",
            "timeframe": row["timeframe"],
            "signal": row["signal"],
            "rank": match.get("Score", 0),
            "score": match.get("Score", 0),
            "r_per_trade": float(r_val),
            "recommended_strategy": match.get("Exit", "Standard"),
            "close": safe_float(row["close"]),
            "high": safe_float(row["high"]),
            "low": safe_float(row["low"]),
            "rsi": safe_float(row["rsi"]) if pd.notna(row["rsi"]) else None,
            "dist_ema": safe_float(display_ema) if pd.notna(display_ema) else None,
            "match_filter": reason_str,
        }

    def _map_key(self, yaml_key: str) -> str:
        mapping = {
            "ema_20": "dist_sma_20",
            "ema_200": "dist_sma_200",
        }
        return mapping.get(yaml_key.lower(), yaml_key.lower())

    def _check_condition(self, market_value: Any, condition: Any, key: str) -> bool:
        if pd.isna(market_value):
            return False

        key_lower = key.lower()
        val = (
            float(market_value)
            if isinstance(market_value, (int, float))
            else market_value
        )
        cond_str = str(condition).strip()

        if "ema" in key_lower:
            match cond_str:
                case "< -10%":
                    return val < -10.0
                case "-10 to -3%":
                    return -10.0 <= val <= -3.0
                case "-3 to 0%":
                    return -3.0 <= val <= 0.0
                case "0 to 3%":
                    return 0.0 <= val <= 3.0
                case "3 to 10%":
                    return 3.0 <= val <= 10.0
                case "> 10%":
                    return val > 10.0
                case _:
                    logger.error(f"Unbekannter EMA-Filter: '{cond_str}'")
                    return False

        elif key_lower == "rsi":
            match cond_str:
                case "Oversold (<30)":
                    return val < 30.0
                case "Weak (30-45)":
                    return 30.0 <= val < 45.0
                case "Neutral (45-55)":
                    return 45.0 <= val <= 55.0
                case "Strong (55-70)":
                    return 55.0 < val <= 70.0
                case "Overbought (>70)":
                    return val > 70.0
                case _:
                    logger.error(f"Unbekannter RSI-Filter: '{cond_str}'")
                    return False

        return str(market_value).lower() == str(condition).lower()
