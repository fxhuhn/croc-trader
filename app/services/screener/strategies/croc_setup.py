import json
import logging
import sqlite3
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


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    """Explicitly defines the capabilities of the technical indicator matching engine."""

    WHITELIST: frozenset[str] = frozenset(
        [
            "deluxe",
            "dist_sma_20",
            "dist_sma_200",
            "kerze",
            "long_blau_status",
            "rsi",
            "rsi_zone",
            "setter",
            "short_blau_status",
            "sma_20_cluster",
            "sma_200_cluster",
            "status",
            "trend",
            "welle",
            "wolke",
        ]
    )

    @staticmethod
    def get_handler(condition_str: str) -> Callable[[float], bool] | None:
        """Returns the appropriate boolean evaluation logic for a string condition."""
        handlers: dict[str, Callable[[float], bool]] = {
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
            "oversold (<30)": lambda v: v < 30.0,
            "weak (30-45)": lambda v: 30.0 <= v < 45.0,
            "neutral (45-55)": lambda v: 45.0 <= v <= 55.0,
            "strong (55-70)": lambda v: 55.0 < v <= 70.0,
            "overbought (>70)": lambda v: v > 70.0,
        }
        return handlers.get(condition_str)


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
        logger.info("🐊 %s initialized. Rules: %d", self.name, len(self.ranking_rules))

    _MAX_CONFIG_FILE_SIZE_BYTES: Final[int] = 1_048_576  # 1 MB guard (SEC-01)

    def _load_config(self) -> list[dict[str, Any]]:
        if not self.config_path.exists():
            logger.error("Ranking config missing: %s", self.config_path)
            return []
        if self.config_path.stat().st_size > self._MAX_CONFIG_FILE_SIZE_BYTES:
            raise RuntimeError(
                "Ranking config exceeds safe size limit — possible YAML anchor bomb: %s"
                % self.config_path
            )
        try:
            with open(self.config_path, encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file)
            rules = data if isinstance(data, list) else data.get("ranking_2026", [])
            logger.info("\u2705 Loaded %d rules from ranking config", len(rules))

            all_keys: set[str] = set()
            trigger_flags: set[str] = set()
            for rule in rules:
                for key in rule.keys():
                    all_keys.add(key)
                rule_signal = str(rule.get("Signal", ""))
                base_rule_signal = (
                    rule_signal.split(" (")[0].strip()
                    if " (" in rule_signal
                    else rule_signal.strip()
                )
                for signal_part in base_rule_signal.split("+"):
                    trigger_flags.add(signal_part.strip().lower())

            meta_keys = [
                k
                for k in all_keys
                if k.lower() not in TechnicalIndicatorConfig.WHITELIST
                and k.lower() not in trigger_flags
                and k.lower() != "signal"
            ]
            if meta_keys:
                logger.info(
                    "\u23ed\ufe0f Skipped/Meta Keys in YAML (ignored by matcher): %s",
                    sorted(meta_keys),
                )
            return rules
        except (yaml.YAMLError, OSError) as yaml_or_io_error:
            logger.error(
                "Failed to load ranking config from %s: %s",
                self.config_path,
                yaml_or_io_error,
            )
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
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "[%s] Database unavailable during signal load: %s"
                % (self.name, database_error)
            ) from database_error
        except (ValueError, KeyError) as data_error:
            logger.warning(
                "[%s] Data anomaly during signal load: %s", self.name, data_error
            )
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
        def _sort_by_sqn_over_max_drawdown(candidate_data: dict[str, Any]) -> float:
            """Pure sort key: higher SQN-to-MaxDD ratio = better risk-adjusted quality."""
            rule = candidate_data["match"]
            system_quality_number = float(rule.get("SQN", rule.get("Score", 0.0)))
            maximum_drawdown = float(rule.get("MaxDD", 0.0))
            return (
                (system_quality_number / maximum_drawdown)
                if maximum_drawdown > 0
                else 0.0
            )

        sorted_candidates = sorted(
            candidates,
            key=_sort_by_sqn_over_max_drawdown,
            reverse=True,
        )

        # 3. Limit to top 3 and create trades
        report_rows = []
        for candidate in sorted_candidates[:3]:
            trade = self._create_trade(
                candidate["normalized"], candidate["prices"], candidate["match"]
            )
            if trade:
                report_rows.append(trade)

        logger.info("🐊 [%s] Created %d trades.", self.name, len(report_rows))

        if self.telegram_bot and report_rows:
            report_rows.sort(key=lambda x: x.get("Symbol", ""))
            self._send_report(report_rows, analysis_date or "LIVE")

        return len(report_rows)

    def get_all_recommendations(
        self,
        days: int = 0,
        analysis_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Returns all recommended signals without creating trades in the database.
        """
        if not analysis_date and days == 0:
            analysis_date = self.signal_repository.get_latest_signal_date()

        try:
            signals = self.signal_repository.get_signals_by_date(analysis_date, days)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "[%s] Database unavailable during signal load: %s"
                % (self.name, database_error)
            ) from database_error
        except (ValueError, KeyError) as data_error:
            logger.warning(
                "[%s] Data anomaly loading signals for get_all_recommendations: %s",
                self.name,
                data_error,
            )
            return []

        if not signals:
            return []

        # 1. Find all matching signal candidates
        candidates = []
        for row in signals:
            item = self._find_candidate(dict(row))
            if item:
                candidates.append(item)

        # 2. Sort using the same SQN/MaxDD key used in run()
        def _sort_by_sqn_over_max_drawdown(candidate_data: dict[str, Any]) -> float:
            """Pure sort key: higher SQN-to-MaxDD ratio = better risk-adjusted quality."""
            rule = candidate_data["match"]
            system_quality_number = float(rule.get("SQN", rule.get("Score", 0.0)))
            maximum_drawdown = float(rule.get("MaxDD", 0.0))
            return (
                (system_quality_number / maximum_drawdown)
                if maximum_drawdown > 0
                else 0.0
            )

        sorted_candidates = sorted(
            candidates,
            key=_sort_by_sqn_over_max_drawdown,
            reverse=True,
        )

        results = []
        for c in sorted_candidates:
            recommendation = self._build_trade_recommendation(
                c["normalized"], c["prices"], c["match"]
            )
            if recommendation:
                recommendation.pop("_internal", None)
                results.append(recommendation)

        return results

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
        normalized = self._enrich_sma(normalized, prices)

        # 5. Match Rule
        match = self._find_best_match(normalized)
        if not match:
            return None

        return {"normalized": normalized, "prices": prices, "match": match}

    def _enrich_sma(self, row: dict[str, Any], prices: PriceData) -> dict[str, Any]:
        """Returns a new dict enriched with SMA distance metrics (no in-place mutation)."""
        enriched = dict(row)
        if prices.sma_20 > 0:
            enriched["dist_sma_20"] = (
                (prices.close - prices.sma_20) / prices.sma_20
            ) * 100
        if prices.sma_200 > 0:
            enriched["dist_sma_200"] = (
                (prices.close - prices.sma_200) / prices.sma_200
            ) * 100
        return enriched

    def _find_best_match(self, row: dict[str, Any]) -> dict[str, Any] | None:
        raw_signal = row.get("signal")
        signal_name = str(raw_signal).strip() if raw_signal is not None else ""
        if signal_name.lower() == "none":
            signal_name = ""

        candidates = []
        for rule in self.ranking_rules:
            rule_signal = str(rule.get("Signal", ""))
            # Extract base signal name if it contains a parenthetical suffix like (NEU)
            base_rule_signal = (
                rule_signal.split(" (")[0].strip()
                if " (" in rule_signal
                else rule_signal.strip()
            )

            is_string_match = bool(
                signal_name
                and (
                    base_rule_signal == signal_name
                    or base_rule_signal == signal_name.split(" (")[0].strip()
                )
            )

            # Handle combined signals safely, e.g. "bear_1 + bear_rot"
            base_signals = [s.strip().lower() for s in base_rule_signal.split("+")]

            is_json_flag_active = bool(base_signals)
            for sig in base_signals:
                sig_val = str(row.get(sig, "")).lower().strip()
                if sig_val not in ("true", "1", "yes", "on", "1.0"):
                    is_json_flag_active = False
                    break

            if is_string_match or is_json_flag_active:
                candidates.append(rule)

        best_match = None
        best_score = -float("inf")
        best_rule_length = 0

        for rule in candidates:
            if self._is_rule_match(row, rule):
                # Prefer SQN over Score if available
                score = float(rule.get("SQN", rule.get("Score", 0.0)))
                rule_length = len(
                    [
                        k
                        for k in rule.keys()
                        if k.lower() in TechnicalIndicatorConfig.WHITELIST
                    ]
                )

                # If score is better, or if score is equal but rule is more specific (more conditions)
                if score > best_score or (
                    score == best_score and rule_length > best_rule_length
                ):
                    best_match = rule
                    best_score = score
                    best_rule_length = rule_length

        return best_match

    def _is_rule_match(self, row: dict[str, Any], rule: dict[str, Any]) -> bool:
        for key, expected_val in rule.items():
            # Use Whitelist: skip any key that is not a technical condition
            db_key = key.lower()
            if db_key not in TechnicalIndicatorConfig.WHITELIST:
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

            handler = TechnicalIndicatorConfig.get_handler(cond_str)
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
        recommendation = self._build_trade_recommendation(row, prices, match)
        if not recommendation:
            return None

        internal_data = recommendation.pop("_internal")
        targets = internal_data["targets"]

        context = {
            "source": "webhook",
            "date": row.get("date_str", row.get("timestamp")),
            "setup_score": float(match.get("SQN", match.get("Score", 0))),
            "match_rule": match,
            "tp1": targets["tp1"],
            "tp3": targets["tp3"],
            "indices": internal_data["indices"],
            "direction": internal_data.get("direction", "long"),
        }

        self.trade_repository.create_trade(
            symbol=recommendation["Symbol"],
            strategy=internal_data["strategy_enum"],
            size=0,
            entry=recommendation["Entry"],
            stop_loss=recommendation["Stop"],
            target=targets["main"],
            context=context,
        )

        return recommendation

    def _build_trade_recommendation(
        self, row: dict[str, Any], prices: PriceData, match: dict[str, Any]
    ) -> dict[str, Any] | None:
        symbol = row.get("symbol", "UNKNOWN")
        indices = self._get_indices_string(symbol)

        if indices == "-" or prices.risk_range <= 0:
            return None

        direction = str(match.get("direction", "long")).lower().strip()
        if direction not in ("long", "short"):
            direction = "long"

        if direction == "short":
            entry = prices.low
            stop = prices.low + prices.risk_range
        else:
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
                "[%s] Unknown exit strategy '%s' for %s. Skipping.",
                self.name,
                exit_name,
                symbol,
            )
            return None

        # Target Logic
        targets = self._calc_targets(entry, prices.risk_range, exit_name, direction)

        raw_signal = row.get("signal")
        displayed_signal = (
            str(raw_signal)
            if raw_signal and str(raw_signal).lower() != "none"
            else str(match.get("Signal", "-"))
        )

        return {
            "Symbol": symbol,
            "Signal": displayed_signal,
            "Score": round(float(match.get("SQN", match.get("Score", 0))), 2),
            "Entry": round(entry, 2),
            "Stop": round(stop, 2),
            "TP": round(targets["main"], 2),
            "Date": row.get("date_str", row.get("timestamp")),
            "_internal": {
                "strategy_enum": strategy_enum,
                "targets": targets,
                "indices": indices,
                "direction": direction,
            },
        }

    def _calc_targets(
        self, entry: float, risk: float, exit_name: str, direction: str = "long"
    ) -> dict[str, float]:
        target_prices = {"tp1": 0.0, "tp3": 0.0, "main": 0.0}

        risk_multiplier = 1 if direction == "long" else -1

        if "split" in exit_name:
            target_prices["tp1"] = round(entry + (risk * risk_multiplier), 2)
            target_prices["tp3"] = round(entry + (3 * risk * risk_multiplier), 2)
            target_prices["main"] = target_prices["tp3"]
        elif "tp3" in exit_name and "hold" in exit_name:
            target_prices["tp3"] = round(entry + (3 * risk * risk_multiplier), 2)
            target_prices["main"] = target_prices["tp3"]
        elif "tp1" in exit_name and "hold" in exit_name:
            target_prices["tp1"] = round(entry + (risk * risk_multiplier), 2)
            target_prices["main"] = target_prices["tp1"]
        else:
            # Fallback
            target_prices["tp1"] = round(entry + (risk * risk_multiplier), 2)
            target_prices["main"] = target_prices["tp1"]

        return target_prices

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
