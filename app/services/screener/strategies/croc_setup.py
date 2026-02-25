import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final

import pandas as pd
import yaml

from ....config import settings
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.signal import SignalRepository
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from .base import BaseStrategy
from ....const import Strategies

logger = logging.getLogger(__name__)

# Positive list of technical fields used for matching
MATCHING_CONDITION_KEYS: Final[set[str]] = {
    "status",
    "kerze",
    "wolke",
    "trend",
    "setter",
    "welle",
    "deluxe",
    "rsi_zone",
    "sma_20_cluster",
    "sma_200_cluster",
    "rsi",
    "dist_sma_20",
    "dist_sma_200",
}

# Optimized Condition Lookup
CONDITION_HANDLERS: dict[str, Callable[[float], bool]] = {
    "< -10%": lambda v: v < -10.0,
    "-10 to -3%": lambda v: -10.0 <= v <= -3.0,
    "-3 to 0%": lambda v: -3.0 <= v <= 0.0,
    "0 to 3%": lambda v: 0.0 <= v <= 3.0,
    "3 to 10%": lambda v: 3.0 <= v <= 10.0,
    "> 10%": lambda v: v > 10.0,
    "oversold": lambda v: v < 30.0,
    "weak": lambda v: 30.0 <= v < 45.0,
    "neutral": lambda v: 45.0 <= v <= 55.0,
    "strong": lambda v: 55.0 < v <= 70.0,
    "overbought": lambda v: v > 70.0,
    # Map explicit labels to same logic to ensure coverage
    "oversold (<30)": lambda v: v < 30.0,
    "weak (30-45)": lambda v: 30.0 <= v < 45.0,
    "neutral (45-55)": lambda v: 45.0 <= v <= 55.0,
    "strong (55-70)": lambda v: 55.0 < v <= 70.0,
    "overbought (>70)": lambda v: v > 70.0,
}


@dataclass(frozen=True)
class PriceData:
    high: float
    low: float
    close: float
    sma_20: float = 0.0
    sma_200: float = 0.0

    @property
    def risk_range(self) -> float:
        return self.high - self.low

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "PriceData | None":
        try:
            return cls(
                high=float(row.get("high") or 0.0),
                low=float(row.get("low") or 0.0),
                close=float(row.get("close") or 0.0),
                sma_20=float(row.get("sma_20") or 0.0),
                sma_200=float(row.get("sma_200") or 0.0),
            )
        except (ValueError, TypeError):
            return None


class CrocSetupStrategy(BaseStrategy):
    name: str = Strategies.CrocSetup

    def __init__(
        self,
        trade_repository: TradeRepository,
        data_provider: MarketDataProvider,
        signal_repository: SignalRepository,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        super().__init__(data_provider, telegram_bot)
        self.trade_repository = trade_repository
        self.signal_repository = signal_repository
        self.exchange_symbols = ExchangeSymbol()
        self.config_path: Path = settings.get_path("ranking_yaml")
        self.ranking_rules = self._load_config()
        logger.info(f"🐊 {self.name} initialized. Rules: {len(self.ranking_rules)}")

    def _load_config(self) -> list[dict[str, Any]]:
        if not self.config_path.exists():
            logger.error(f"Config missing: {self.config_path}")
            return []
        try:
            with open(self.config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f)
            # Ensure we return a list regardless of YAML structure
            rules = data if isinstance(data, list) else data.get("ranking_2026", [])
            logger.info(f"✅ Loaded {len(rules)} rules")
            return rules
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return []

    def run(
        self,
        days: int = 0,
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        if not analysis_date and days == 0:
            analysis_date = self.signal_repository.get_latest_signal_date()

        try:
            signals = self.signal_repository.get_signals_by_date(analysis_date, days)
        except Exception as e:
            logger.error(f"Error loading signals: {e}")
            return 0

        if not signals:
            return 0

        # 1. Find all matching signal candidates
        candidates = []
        for row in signals:
            item = self._find_candidate(dict(row))
            if item:
                candidates.append(item)

        # 2. Sort all candidates by SQN / MaxDD Ratio (High to Low)
        def _get_sort_score(c: dict[str, Any]) -> float:
            rule = c["match"]
            sqn = float(rule.get("SQN", rule.get("Score", 0.0)))
            max_dd = float(rule.get("MaxDD", 0.0))
            return (sqn / max_dd) if max_dd > 0 else 0.0

        sorted_candidates = sorted(
            candidates,
            key=_get_sort_score,
            reverse=True,
        )

        # 3. Limit to top 3 and create trades
        report_rows = []
        for c in sorted_candidates[:3]:
            trade = self._create_trade(c["normalized"], c["prices"], c["match"])
            if trade:
                report_rows.append(trade)

        logger.info(f"🐊 [{self.name}] Created {len(report_rows)} trades.")

        if self.telegram_bot and report_rows:
            # Sort report rows by date for cleaner log
            report_rows.sort(key=lambda x: x.get("Symbol", ""))
            self._send_report(report_rows, analysis_date or "LIVE")

        return len(report_rows)

    def _process_single_signal(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """
        Legacy helper for processing a single signal row.
        Used primarily by tests to skip the batching/limiting logic.
        """
        candidate = self._find_candidate(row)
        if not candidate:
            return None
        return self._create_trade(
            candidate["normalized"], candidate["prices"], candidate["match"]
        )

    def _find_candidate(self, row: dict[str, Any]) -> dict[str, Any] | None:
        """Finds a matching rule for a signal but does not create a trade yet."""
        # 1. Parse JSON data from DB
        try:
            signal_data = (
                json.loads(row["data"]) if isinstance(row.get("data"), str) else {}
            )
        except json.JSONDecodeError:
            signal_data = {}

        # 2. Normalize Keys
        full_data = {**row, **signal_data}
        normalized = {k.lower(): v for k, v in full_data.items()}

        # 3. Create Price Object
        prices = PriceData.from_row(normalized)
        if not prices:
            return None

        # 4. Enrich Data
        self._enrich_sma(normalized, prices)

        # 5. Match Rule
        match = self._find_best_match(normalized)
        if not match:
            return None

        return {"normalized": normalized, "prices": prices, "match": match}

    def _enrich_sma(self, row: dict[str, Any], prices: PriceData) -> None:
        if prices.sma_20 > 0:
            row["dist_sma_20"] = ((prices.close - prices.sma_20) / prices.sma_20) * 100
        if prices.sma_200 > 0:
            row["dist_sma_200"] = (
                (prices.close - prices.sma_200) / prices.sma_200
            ) * 100

    def _find_best_match(self, row: dict[str, Any]) -> dict[str, Any] | None:
        signal_name = str(row.get("signal", ""))
        
        candidates = []
        for rule in self.ranking_rules:
            rule_signal = str(rule.get("Signal", ""))
            # Extract base signal name if it contains a parenthetical suffix like (NEU)
            base_rule_signal = rule_signal.split(" (")[0].strip() if " (" in rule_signal else rule_signal.strip()
            
            if base_rule_signal == signal_name or base_rule_signal == signal_name.split(" (")[0].strip():
                candidates.append(rule)

        best_match = None
        best_score = -float("inf")
        best_rule_length = 0

        for rule in candidates:
            if self._is_rule_match(row, rule):
                # Prefer SQN over Score if available
                score = float(rule.get("SQN", rule.get("Score", 0.0)))
                rule_length = len([k for k in rule.keys() if k.lower() in MATCHING_CONDITION_KEYS])
                
                # If score is better, or if score is equal but rule is more specific (more conditions)
                if score > best_score or (score == best_score and rule_length > best_rule_length):
                    best_match = rule
                    best_score = score
                    best_rule_length = rule_length
                    
        return best_match

    def _is_rule_match(self, row: dict[str, Any], rule: dict[str, Any]) -> bool:
        for key, expected_val in rule.items():
            # Use Whitelist: skip any key that is not a technical condition
            db_key = key.lower()
            if db_key not in MATCHING_CONDITION_KEYS:
                continue

            # Map YAML key to DB key
            if "ema" in db_key or "sma" in db_key:
                db_key = "dist_sma_200" if "200" in db_key else "dist_sma_20"
            elif "rsi" in db_key:
                db_key = "rsi"

            if db_key not in row:
                return False

            # Check Value
            if not self._check_value(row[db_key], expected_val):
                return False
        return True

    def _check_value(self, market_val: Any, condition: Any) -> bool:
        if market_val is None:
            return False

        # Try numeric lookup
        try:
            val = float(market_val)
            cond_str = str(condition).lower().strip()
            
            # Strip numeric prefixes like "3. " or "6. "
            if cond_str and cond_str[0].isdigit() and ". " in cond_str:
                cond_str = cond_str.split(". ", 1)[1].strip()
                
            handler = CONDITION_HANDLERS.get(cond_str)
            if handler:
                return handler(val)
        except (ValueError, TypeError):
            pass

        # Fallback to string match
        return str(market_val).lower().replace(" ", "") == str(
            condition
        ).lower().replace(" ", "")

    def _create_trade(
        self, row: dict[str, Any], prices: PriceData, match: dict[str, Any]
    ) -> dict[str, Any] | None:
        symbol = row.get("symbol", "UNKNOWN")
        indices = self._get_indices_string(symbol)

        if indices == "-" or prices.risk_range <= 0:
            return None

        entry = prices.high
        stop = prices.high - prices.risk_range
        exit_name = str(match.get("Exit", "unknown")).lower().strip()

        # Strict Strategy Mapping
        strategy_enum: str
        if "split" in exit_name:
            strategy_enum = Strategies.SplitTarget
        elif "hold" in exit_name or "tp3" in exit_name:
            strategy_enum = Strategies.HoldTarget
        else:
            logger.error(
                f"[{self.name}] Unknown exit strategy '{exit_name}' for {symbol}. Skipping."
            )
            return None

        # Target Logic
        targets = self._calc_targets(entry, prices.risk_range, exit_name)

        context = {
            "source": "webhook",
            "date": row.get("date_str", row.get("timestamp")),
            "setup_score": float(match.get("SQN", match.get("Score", 0))),
            "match_rule": match,
            "tp1": targets["tp1"],
            "tp3": targets["tp3"],
            "indices": indices,
        }

        self.trade_repository.create_trade(
            symbol=symbol,
            strategy=strategy_enum,
            size=0,
            entry=entry,
            stop_loss=stop,
            target=targets["main"],
            context=context,
        )

        return {
            "Symbol": symbol,
            "Signal": str(row.get("signal", "-")),
            "Score": round(float(match.get("SQN", match.get("Score", 0))), 2),
            "Entry": round(entry, 2),
            "Stop": round(stop, 2),
            "TP": round(targets["main"], 2),
        }

    def _calc_targets(
        self, entry: float, risk: float, exit_name: str
    ) -> dict[str, float]:
        t = {"tp1": 0.0, "tp3": 0.0, "main": 0.0}
        
        if "split" in exit_name:
            t["tp1"] = round(entry + risk, 2)
            t["tp3"] = round(entry + (3 * risk), 2)
            t["main"] = t["tp3"]
        elif "tp3" in exit_name and "hold" in exit_name:
            t["tp3"] = round(entry + (3 * risk), 2)
            t["main"] = t["tp3"]
        elif "tp1" in exit_name and "hold" in exit_name:
            t["tp1"] = round(entry + risk, 2)
            t["main"] = t["tp1"]
        else:
            # Fallback
            t["tp1"] = round(entry + risk, 2)
            t["main"] = t["tp1"]
            
        return t

    def _get_indices_string(self, symbol: str) -> str:
        # Simplified for brevity
        indices = []
        if symbol in self.exchange_symbols.sp_500:
            indices.append("SPX")
        if symbol in self.exchange_symbols.nasdaq_100:
            indices.append("NDX")
        if symbol in self.exchange_symbols.dow_30:
            indices.append("DOW")
        if symbol in self.exchange_symbols.russell_1000:
            indices.append("RUS")
        return ",".join(indices) if indices else "-"

    def _send_report(self, rows: list[dict[str, Any]], date: str) -> None:
        if not self.telegram_bot:
            return
        df = pd.DataFrame(rows)
        # Select existing columns only
        cols = [
            c
            for c in ["Symbol", "Signal", "Score", "Entry", "Stop", "TP"]
            if c in df.columns
        ]
        self._send_telegram_report("Croc Signals", date, df[cols])
