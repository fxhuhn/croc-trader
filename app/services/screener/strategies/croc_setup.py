import json
import logging
import re
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import yaml

from ....config import settings
from ....const import Strategies
from ....database.repositories.market_data_provider import MarketDataProvider
from ....database.repositories.signal import SignalRepository
from ....database.repositories.trade import TradeRepository
from ....services.telegram import TelegramBot
from ....tools.symbol_lists import ExchangeSymbol
from ..models import SignalReportItem
from .base import BaseStrategy

logger = logging.getLogger(__name__)

MAX_RANKING_CONFIG_SIZE_BYTES: Final[int] = 1_048_576  # 1 MB guard (SEC-01)
DEFAULT_TOP_CANDIDATES_LIMIT: Final[int] = 3


# Technical Indicator Threshold Constants
RSI_OVERSOLD_THRESHOLD: Final[float] = 30.0
RSI_WEAK_LOWER: Final[float] = 30.0
RSI_WEAK_UPPER: Final[float] = 45.0
RSI_NEUTRAL_LOWER: Final[float] = 45.0
RSI_NEUTRAL_UPPER: Final[float] = 55.0
RSI_STRONG_LOWER: Final[float] = 55.0
RSI_STRONG_UPPER: Final[float] = 70.0
RSI_OVERBOUGHT_THRESHOLD: Final[float] = 70.0

SMA_DIST_MINUS_10: Final[float] = -10.0
SMA_DIST_MINUS_3: Final[float] = -3.0
SMA_DIST_ZERO: Final[float] = 0.0
SMA_DIST_PLUS_3: Final[float] = 3.0
SMA_DIST_PLUS_10: Final[float] = 10.0


@dataclass(frozen=True)
class TechnicalIndicatorConfig:
    """Explicitly defines the capabilities of the technical indicator matching engine."""

    WHITELIST: frozenset[str] = frozenset(
        [
            "bear_1",
            "bear_2",
            "bear_3",
            "bear_4",
            "bear_5",
            "bear_6",
            "bear_7",
            "bear_8",
            "bear_9",
            "bear_10",
            "bear_11",
            "bear_12",
            "bear_13",
            "bear_14",
            "bear_15",
            "bear_bb",
            "bear_blau",
            "bear_grabber",
            "bear_grau",
            "bear_grau_klein",
            "bear_gruen",
            "bear_hell_gruen",
            "bear_hellgrün",
            "bear_line",
            "bear_orange",
            "bear_pink",
            "bear_plus",
            "bear_rot",
            "bear_schwarz",
            "bear_wolke",
            "bull_1",
            "bull_2",
            "bull_3",
            "bull_4",
            "bull_5",
            "bull_6",
            "bull_7",
            "bull_8",
            "bull_9",
            "bull_10",
            "bull_11",
            "bull_12",
            "bull_13",
            "bull_14",
            "bull_15",
            "bull_bb",
            "bull_blau",
            "bull_grabber",
            "bull_grau",
            "bull_grau_klein",
            "bull_gruen",
            "bull_hell_gruen",
            "bull_hellgrün",
            "bull_line",
            "bull_orange",
            "bull_pink",
            "bull_plus",
            "bull_rot",
            "bull_schwarz",
            "bull_wolke",
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
            "< -10%": lambda v: v < SMA_DIST_MINUS_10,
            "-10 to -3%": lambda v: SMA_DIST_MINUS_10 <= v <= SMA_DIST_MINUS_3,
            "-3 to 0%": lambda v: SMA_DIST_MINUS_3 <= v <= SMA_DIST_ZERO,
            "0 to 3%": lambda v: SMA_DIST_ZERO <= v <= SMA_DIST_PLUS_3,
            "3 to 10%": lambda v: SMA_DIST_PLUS_3 <= v <= SMA_DIST_PLUS_10,
            "> 10%": lambda v: v > SMA_DIST_PLUS_10,
            "oversold": lambda v: v < RSI_OVERSOLD_THRESHOLD,
            "weak": lambda v: RSI_WEAK_LOWER <= v < RSI_WEAK_UPPER,
            "neutral": lambda v: RSI_NEUTRAL_LOWER <= v <= RSI_NEUTRAL_UPPER,
            "strong": lambda v: RSI_STRONG_LOWER <= v <= RSI_STRONG_UPPER,
            "overbought": lambda v: v > RSI_OVERBOUGHT_THRESHOLD,
            "oversold (<30)": lambda v: v < RSI_OVERSOLD_THRESHOLD,
            "weak (30-45)": lambda v: RSI_WEAK_LOWER <= v < RSI_WEAK_UPPER,
            "neutral (45-55)": lambda v: RSI_NEUTRAL_LOWER <= v <= RSI_NEUTRAL_UPPER,
            "strong (55-70)": lambda v: RSI_STRONG_LOWER <= v <= RSI_STRONG_UPPER,
            "overbought (>70)": lambda v: v > RSI_OVERBOUGHT_THRESHOLD,
        }
        return handlers.get(condition_str)


def _extract_safe_price(row: Mapping[str, object], key: str) -> float:
    """Safely extracts a float price value or returns 0.0."""
    val = row.get(key)
    if val is None or val == "":
        return 0.0
    return float(str(val))


@dataclass(frozen=True)
class PriceData:
    """Validated price observations extracted from a webhook signal."""

    high: float
    low: float
    close: float
    sma_20: float = 0.0
    sma_200: float = 0.0

    @property
    def risk_range(self) -> float:
        """Returns non-negative risk range (high - low)."""
        return max(0.0, self.high - self.low)

    @classmethod
    def from_row(cls, row: Mapping[str, object]) -> "PriceData | None":
        """Factory creating PriceData from raw mapping with boundary validation."""
        try:
            close_val = _extract_safe_price(row, "close")
            high_val = _extract_safe_price(row, "high")
            if close_val <= 0.0 and high_val <= 0.0:
                return None

            resolved_high = close_val if high_val <= 0.0 else high_val
            low_val = _extract_safe_price(row, "low")
            resolved_low = close_val if low_val <= 0.0 else low_val

            return cls(
                high=resolved_high,
                low=resolved_low,
                close=close_val,
                sma_20=_extract_safe_price(row, "sma_20"),
                sma_200=_extract_safe_price(row, "sma_200"),
            )
        except (ValueError, TypeError):
            return None


type CrocPriceData = PriceData


@dataclass(frozen=True)
class CrocCandidate:
    """Immutable representation of a matched, priced CrocSetup candidate."""

    symbol: str
    signal_name: str
    score: float
    entry_price: float
    stop_loss: float
    target_profit: float
    target_level: int
    direction: str
    indices: str
    date_str: str
    rule_match: dict[str, object] = field(default_factory=dict)
    normalized_data: dict[str, object] = field(default_factory=dict)
    prices: PriceData | None = None

    def to_report_item(self) -> SignalReportItem:
        """Converts candidate into standardized SignalReportItem."""
        return SignalReportItem(
            symbol=self.symbol,
            action=f"BUY {'MKT' if self.direction == 'long' else 'SELL'}",
            entry_price=self.entry_price,
            stop_loss=self.stop_loss,
            target_profit=self.target_profit,
            details={
                "Signal": self.signal_name,
                "Score": f"{self.score:.2f}",
                "TP": f"{self.target_profit:.2f}",
            },
        )


# ═══════════════════════════════════════════════════════════════════════════════
# FUNCTIONAL CORE — Pure mathematical modeling and rule matching
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_indicator_condition(market_value: object, condition: object) -> bool:
    """Evaluates whether a single indicator value matches a rule condition."""
    if market_value is None:
        return False

    try:
        numeric_val = float(str(market_value))
        condition_string = str(condition).lower().strip()

        # Strip numeric prefixes like "3. " or "6. "
        if (
            condition_string
            and condition_string[0].isdigit()
            and ". " in condition_string
        ):
            condition_string = condition_string.split(". ", 1)[1].strip()

        handler = TechnicalIndicatorConfig.get_handler(condition_string)
        if handler is not None:
            return handler(numeric_val)
    except (ValueError, TypeError):
        pass

    # Fallback to string equivalence
    return str(market_value).lower().replace(" ", "") == str(condition).lower().replace(
        " ", ""
    )


def calculate_croc_candidate_score(score: float, max_drawdown: float) -> float:
    """Pure Function: Calculates SQN / MaxDD ratio score."""
    if max_drawdown <= 0.0:
        return 0.0
    return score / max_drawdown


def calculate_croc_targets(
    entry: float, risk: float, target_level: int, direction: str = "long"
) -> dict[str, float]:
    """Pure Function: Calculates price target boundaries for a given TP level."""
    risk_multiplier = 1.0 if direction == "long" else -1.0
    target_price = round(entry + (risk * target_level * risk_multiplier), 2)
    return {"main": target_price}


def is_rule_signal_match(
    row: Mapping[str, object], rule: Mapping[str, object], signal_name: str
) -> bool:
    """Pure Function: Checks if a rule's Signal field matches the active signal."""
    rule_signal = str(rule.get("Signal", ""))
    base_rule_signal = (
        rule_signal.split(" (", maxsplit=1)[0].strip()
        if " (" in rule_signal
        else rule_signal.strip()
    )

    is_string_match = bool(
        signal_name
        and (
            base_rule_signal == signal_name
            or base_rule_signal == signal_name.split(" (", maxsplit=1)[0].strip()
        )
    )

    # Handle combined signals safely, e.g. "bear_1 + bear_rot"
    base_signals = [
        signal_part.strip().lower() for signal_part in base_rule_signal.split("+")
    ]

    is_json_flag_active = bool(base_signals)
    for signal_name_part in base_signals:
        signal_value = str(row.get(signal_name_part, "")).lower().strip()
        if signal_value not in ("true", "1", "yes", "on", "1.0"):
            is_json_flag_active = False
            break

    return is_string_match or is_json_flag_active


def is_rule_match(row: Mapping[str, object], rule: Mapping[str, object]) -> bool:
    """Pure Function: Evaluates all technical condition requirements in a rule."""
    for key, expected_value in rule.items():
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

        if not evaluate_indicator_condition(row[db_key], expected_value):
            return False
    return True


def _normalize_signal_name(raw_signal: object) -> str:
    """Normalizes raw signal name to non-empty string or empty string."""
    if raw_signal is None:
        return ""
    signal_name = str(raw_signal).strip()
    return "" if signal_name.lower() == "none" else signal_name


def _rule_ranking_key(rule: Mapping[str, object]) -> tuple[float, int]:
    """Computes (score, rule_length) comparison key for rule ranking."""
    score = float(str(rule.get("SQN", rule.get("Score", 0.0))))
    rule_length = sum(
        1 for key in rule.keys() if key.lower() in TechnicalIndicatorConfig.WHITELIST
    )
    return score, rule_length


def find_best_rule_match(
    row: Mapping[str, object], rules: Sequence[Mapping[str, object]]
) -> dict[str, object] | None:
    """Pure Function: Finds the best matching YAML ranking rule for a signal row."""
    signal_name = _normalize_signal_name(row.get("signal"))
    valid_rules = [
        rule
        for rule in rules
        if is_rule_signal_match(row, rule, signal_name) and is_rule_match(row, rule)
    ]
    if not valid_rules:
        return None

    best_rule = max(valid_rules, key=_rule_ranking_key)
    return dict(best_rule)


def enrich_sma_distances(
    row: Mapping[str, object], prices: PriceData | None
) -> dict[str, object]:
    """Pure Function: Enriches dictionary with SMA distance metrics without mutation."""
    enriched = dict(row)
    if prices is None:
        return enriched
    if prices.sma_20 > 0:
        enriched["dist_sma_20"] = (
            (prices.close - prices.sma_20) / prices.sma_20
        ) * 100.0
    if prices.sma_200 > 0:
        enriched["dist_sma_200"] = (
            (prices.close - prices.sma_200) / prices.sma_200
        ) * 100.0
    return enriched


def _resolve_croc_direction(match: Mapping[str, object]) -> str:
    """Extracts normalized direction, defaulting to 'long'."""
    raw_direction = str(match.get("direction", "long")).lower().strip()
    return raw_direction if raw_direction in ("long", "short") else "long"


def _resolve_croc_entry_and_stop(
    prices: PriceData, direction: str
) -> tuple[float, float]:
    """Computes entry and stop loss based on direction and risk range."""
    if direction == "short":
        return prices.low, prices.low + prices.risk_range
    return prices.high, prices.high - prices.risk_range


def _extract_tp_level(exit_name: str) -> int:
    """Extracts TP numeric level (e.g. tp1 -> 1, tp2 -> 2) or defaults to 1."""
    match_tp = re.search(r"tp(\d+)", exit_name.lower().strip())
    return int(match_tp.group(1)) if match_tp else 1


def _resolve_display_signal(raw_signal: object, rule_signal: object) -> str:
    """Resolves human-readable signal name from raw signal or rule fallback."""
    if raw_signal and str(raw_signal).lower() != "none":
        return str(raw_signal)
    return str(rule_signal) if rule_signal is not None else "-"


def _resolve_rule_score(match: Mapping[str, object]) -> float:
    """Calculates weighted candidate score factoring in SQN/Score and MaxDD."""
    raw_score = float(str(match.get("SQN", match.get("Score", 0.0))))
    max_dd = float(str(match.get("MaxDD", 0.0)))
    return (
        calculate_croc_candidate_score(raw_score, max_dd) if max_dd > 0.0 else raw_score
    )


def build_croc_candidate(
    row: Mapping[str, object],
    prices: PriceData,
    match: Mapping[str, object],
    indices: str,
) -> CrocCandidate | None:
    """Pure Function: Constructs a validated CrocCandidate from match and price data."""
    if indices == "-" or prices.risk_range <= 0.0:
        return None

    symbol = str(row.get("symbol", "UNKNOWN"))
    direction = _resolve_croc_direction(match)
    entry, stop = _resolve_croc_entry_and_stop(prices, direction)
    tp_level = _extract_tp_level(str(match.get("Exit", "unknown")))
    targets = calculate_croc_targets(entry, prices.risk_range, tp_level, direction)

    displayed_signal = _resolve_display_signal(row.get("signal"), match.get("Signal"))
    score = _resolve_rule_score(match)
    date_val = str(row.get("date_str") or row.get("timestamp") or "")

    return CrocCandidate(
        symbol=symbol,
        signal_name=displayed_signal,
        score=score,
        entry_price=round(entry, 2),
        stop_loss=round(stop, 2),
        target_profit=round(targets["main"], 2),
        target_level=tp_level,
        direction=direction,
        indices=indices,
        date_str=date_val,
        rule_match=dict(match),
        normalized_data=dict(row),
        prices=prices,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# IMPERATIVE SHELL — I/O, database interaction, notification orchestration
# ═══════════════════════════════════════════════════════════════════════════════


class CrocSetupStrategy(BaseStrategy[int]):
    """CrocSetup screening strategy evaluating webhook signals against YAML ranking rules."""

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
        self.ranking_rules: list[dict[str, object]] = self._load_config()
        logger.info("🐊 %s initialized. Rules: %d", self.name, len(self.ranking_rules))

    def _load_config(self) -> list[dict[str, object]]:
        """Loads and parses the YAML ranking configuration safely."""
        if not self.config_path.exists():
            logger.error("Ranking config missing: %s", self.config_path)
            return []
        if self.config_path.stat().st_size > MAX_RANKING_CONFIG_SIZE_BYTES:
            raise RuntimeError(
                f"Ranking config exceeds safe size limit — possible YAML anchor bomb: {self.config_path}"
            )
        try:
            with open(self.config_path, encoding="utf-8") as config_file:
                data = yaml.safe_load(config_file)
            rules_raw = data if isinstance(data, list) else data.get("ranking_2026", [])
            rules = [dict(r) for r in rules_raw if isinstance(r, dict)]
            logger.info("✅ Loaded %d rules from ranking config", len(rules))
            return rules
        except (yaml.YAMLError, OSError) as yaml_or_io_error:
            logger.error(
                "Failed to load ranking config from %s: %s",
                self.config_path,
                yaml_or_io_error,
            )
            return []

    def _fetch_and_sort_candidates(
        self, analysis_date: str | None, days: int
    ) -> list[CrocCandidate]:
        """Fetches signals from database, evaluates rules, and returns sorted candidates."""
        try:
            target_date = analysis_date or ""
            signals = self.signal_repository.get_signals_by_date(target_date, days)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"[{self.name}] Database unavailable during signal load: {database_error}"
            ) from database_error
        except (ValueError, KeyError) as data_error:
            logger.warning(
                "[%s] Data anomaly during signal load: %s", self.name, data_error
            )
            return []

        if not signals:
            return []

        candidates: list[CrocCandidate] = []
        for row in signals:
            candidate = self._find_croc_candidate(dict(row))
            if candidate is not None:
                candidates.append(candidate)

        # Sort candidates descending by score (SQN / MaxDD)
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    def _find_croc_candidate(self, row: Mapping[str, object]) -> CrocCandidate | None:
        """Parses signal row, normalizes data, matches rules and builds candidate."""
        try:
            raw_data = row.get("data")
            signal_data: dict[str, object] = (
                json.loads(str(raw_data))
                if isinstance(raw_data, str) and raw_data
                else {}
            )
        except json.JSONDecodeError as error:
            logger.warning(
                "Failed to parse signal data JSON for symbol %s: %s",
                row.get("symbol"),
                error,
            )
            signal_data = {}

        full_data = {**dict(row), **signal_data}
        normalized = {str(k).lower(): v for k, v in full_data.items()}

        prices = PriceData.from_row(normalized)
        if prices is None:
            return None

        enriched = enrich_sma_distances(normalized, prices)
        match = find_best_rule_match(enriched, self.ranking_rules)
        if match is None:
            return None

        symbol = str(row.get("symbol", "UNKNOWN"))
        indices = self._get_indices_string(symbol)

        return build_croc_candidate(enriched, prices, match, indices)

    def _filter_by_specific_symbols(
        self,
        candidates: list[CrocCandidate],
        specific_symbols: list[str] | None,
    ) -> list[CrocCandidate]:
        """Filters candidates to match specified symbols if provided."""
        if not specific_symbols:
            return candidates
        allowed = {symbol.upper() for symbol in specific_symbols}
        return [c for c in candidates if c.symbol.upper() in allowed]

    def _persist_candidate_trade(self, candidate: CrocCandidate) -> None:
        """Persists a single candidate as a created trade in signals.db."""
        context = {
            "source": "webhook",
            "date": candidate.date_str,
            "setup_score": candidate.score,
            "match_rule": candidate.rule_match,
            "target_level": candidate.target_level,
            "indices": candidate.indices,
            "direction": candidate.direction,
        }
        self.trade_repository.create_trade(
            symbol=candidate.symbol,
            strategy=Strategies.HoldTarget,
            size=0,
            entry=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            target=candidate.target_profit,
            context=context,
        )

    def _notify_created_candidates(
        self,
        created_candidates: list[CrocCandidate],
        analysis_date: str | None,
    ) -> None:
        """Sends Telegram notification report for created candidates."""
        if not self.telegram_bot or not created_candidates:
            return
        report_items = [c.to_report_item() for c in created_candidates]
        report_items.sort(key=lambda x: x.symbol)
        self._send_telegram_report(
            display_name="Croc Signals",
            data=report_items,
            date=analysis_date or "LIVE",
        )

    def run(
        self,
        days: int = 0,
        analysis_date: str | None = None,
        specific_symbols: list[str] | None = None,
    ) -> int:
        """Runs the CrocSetup screening pipeline and persists top 3 created trades."""
        target_date = (
            self.signal_repository.get_latest_signal_date()
            if not analysis_date and days == 0
            else analysis_date
        )

        candidates = self._fetch_and_sort_candidates(target_date, days)
        filtered = self._filter_by_specific_symbols(candidates, specific_symbols)
        top_candidates = filtered[:DEFAULT_TOP_CANDIDATES_LIMIT]

        for candidate in top_candidates:
            self._persist_candidate_trade(candidate)

        logger.info("🐊 [%s] Created %d trades.", self.name, len(top_candidates))
        self._notify_created_candidates(top_candidates, target_date)
        return len(top_candidates)

    def get_all_recommendations(
        self,
        days: int = 0,
        analysis_date: str | None = None,
    ) -> list[dict[str, object]]:
        """Returns all candidate recommendations without database persistence."""
        if not analysis_date and days == 0:
            analysis_date = self.signal_repository.get_latest_signal_date()

        sorted_candidates = self._fetch_and_sort_candidates(analysis_date, days)
        return [
            {
                "Symbol": c.symbol,
                "Signal": c.signal_name,
                "Score": round(c.score, 2),
                "Entry": c.entry_price,
                "Stop": c.stop_loss,
                "TP": c.target_profit,
                "Date": c.date_str,
            }
            for c in sorted_candidates
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Backward Compatibility Adapter Methods for Existing Unit/Integration Tests
    # ─────────────────────────────────────────────────────────────────────────

    def _find_candidate(self, row: dict[str, object]) -> dict[str, object] | None:
        """Adapter for legacy tests expecting dictionary structure."""
        candidate = self._find_croc_candidate(row)
        if candidate is None:
            return None
        return {
            "normalized": candidate.normalized_data,
            "prices": candidate.prices,
            "match": candidate.rule_match,
        }

    def _create_trade(
        self,
        row: dict[str, object],
        prices: PriceData,
        match: dict[str, object],
    ) -> dict[str, object] | None:
        """Adapter for legacy tests calling _create_trade directly."""
        symbol = str(row.get("symbol", "UNKNOWN"))
        indices = self._get_indices_string(symbol)
        candidate = build_croc_candidate(row, prices, match, indices)
        if candidate is None:
            return None

        context = {
            "source": "webhook",
            "date": candidate.date_str,
            "setup_score": candidate.score,
            "match_rule": match,
            "target_level": candidate.target_level,
            "indices": indices,
            "direction": candidate.direction,
        }

        self.trade_repository.create_trade(
            symbol=candidate.symbol,
            strategy=Strategies.HoldTarget,
            size=0,
            entry=candidate.entry_price,
            stop_loss=candidate.stop_loss,
            target=candidate.target_profit,
            context=context,
        )

        return {
            "Symbol": candidate.symbol,
            "Signal": candidate.signal_name,
            "Score": round(candidate.score, 2),
            "Entry": candidate.entry_price,
            "Stop": candidate.stop_loss,
            "TP": candidate.target_profit,
            "Date": candidate.date_str,
        }

    def _build_trade_recommendation(
        self,
        row: dict[str, object],
        prices: PriceData,
        match: dict[str, object],
    ) -> dict[str, object] | None:
        """Adapter for legacy tests inspecting recommendation dictionaries."""
        symbol = str(row.get("symbol", "UNKNOWN"))
        indices = self._get_indices_string(symbol)
        candidate = build_croc_candidate(row, prices, match, indices)
        if candidate is None:
            return None

        return {
            "Symbol": candidate.symbol,
            "Signal": candidate.signal_name,
            "Score": round(candidate.score, 2),
            "Entry": candidate.entry_price,
            "Stop": candidate.stop_loss,
            "TP": candidate.target_profit,
            "Date": candidate.date_str,
            "_internal": {
                "strategy_enum": Strategies.HoldTarget,
                "targets": {"main": candidate.target_profit},
                "target_level": candidate.target_level,
                "indices": indices,
                "direction": candidate.direction,
            },
        }

    def _process_single_signal(
        self, row: dict[str, object]
    ) -> dict[str, object] | None:
        """Adapter for legacy test processing a single signal row."""
        candidate = self._find_croc_candidate(row)
        if candidate is None or candidate.prices is None:
            return None
        return self._create_trade(
            candidate.normalized_data, candidate.prices, candidate.rule_match
        )

    def _check_value(self, market_value: object, condition: object) -> bool:
        """Adapter for legacy tests calling _check_value."""
        return evaluate_indicator_condition(market_value, condition)

    def _find_best_match(self, row: dict[str, object]) -> dict[str, object] | None:
        """Adapter for legacy tests calling _find_best_match."""
        return find_best_rule_match(row, self.ranking_rules)

    def _enrich_sma(
        self, row: dict[str, object], prices: PriceData
    ) -> dict[str, object]:
        """Adapter for legacy tests calling _enrich_sma."""
        return enrich_sma_distances(row, prices)

    def _calc_targets(
        self, entry: float, risk: float, target_level: int, direction: str = "long"
    ) -> dict[str, float]:
        """Adapter for legacy tests calling _calc_targets."""
        return calculate_croc_targets(entry, risk, target_level, direction)
