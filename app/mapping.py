import json
import logging
from typing import ClassVar

import yfinance as yf

from .config import settings

logger = logging.getLogger(__name__)


class ExchangeMapper:
    """Singleton mapping equity symbols to their primary exchange."""

    _instance: ClassVar["ExchangeMapper | None"] = None
    _mapping: dict[str, str] = {}

    CANONICAL_EXCHANGES: ClassVar[dict[str, str]] = {
        "QQQ": "NASDAQ",
        "SPY": "AMEX",
        "DIA": "AMEX",
        "IWM": "AMEX",
        "MDY": "AMEX",
        "XLK": "AMEX",
        "XLF": "AMEX",
        "XLE": "AMEX",
        "XLV": "AMEX",
        "XLY": "AMEX",
        "XLP": "AMEX",
        "XLU": "AMEX",
        "XLI": "AMEX",
        "XLB": "AMEX",
        "XLRE": "AMEX",
        "XLC": "AMEX",
        "TLT": "NASDAQ",
        "SXRV.DE": "XETR",
        "^VIX": "CBOE",
        "VIX": "CBOE",
        "^NDX": "NASDAQ",
        "NDX": "NASDAQ",
        "^GSPC": "INDEX",
        "^DJI": "DJ",
        "^RUT": "RUSSELL",
    }
    # Backward-compatibility alias
    DEFAULT_ETF_EXCHANGES = CANONICAL_EXCHANGES

    def __new__(cls) -> "ExchangeMapper":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def load(self) -> None:
        """Explicitly loads the mapping. Called from create_app() when logging is ready."""
        if not self._mapping:
            self._load_mapping()

    def _load_mapping(self) -> None:
        """Loads the JSON mapping file into memory (one-time)."""
        json_path = settings.get_path("exchange_mapping")

        if not json_path.exists():
            logger.warning("Exchange mapping file not found: %s", json_path)
            return

        try:
            with open(json_path, encoding="utf-8") as file:
                raw_data = json.load(file)
                if isinstance(raw_data, dict):
                    self._mapping = {str(k): str(v) for k, v in raw_data.items()}
            logger.info("Exchange mapping loaded: %d symbols.", len(self._mapping))
        except (json.JSONDecodeError, OSError) as error:
            logger.error("Failed to load exchange JSON: %s", error)

    @staticmethod
    def normalize_exchange_code(raw_code: str) -> str | None:
        """Maps diverse exchange code identifiers to standard system names."""
        code = raw_code.strip().upper()
        if code in ("NYQ", "NYSE", "NEW YORK STOCK EXCHANGE"):
            return "NYSE"
        if code in ("NMS", "NGS", "NCM", "NASDAQ", "NIDS"):
            return "NASDAQ"
        if code in ("ASE", "AMEX", "NYSE AMERICAN", "BATS"):
            return "AMEX"
        if code in ("GER", "XETRA", "FRA", "XETR"):
            return "XETR"
        if code in ("CBOE", "CHICAGO BOARD OPTIONS EXCHANGE"):
            return "CBOE"
        return None

    def get_exchange(self, symbol: str, default: str | None = None) -> str | None:
        """Returns the primary exchange for a symbol using a 3-tier resolution:
        1. Canonical rule-based heuristics (.DE suffix, known indices, core ETFs)
        2. Persistent local mapping store (symbol_exchange.json)
        3. Configured default fallback
        """
        symbol_upper = symbol.upper().strip()

        # 1. Deterministic Heuristics
        if symbol_upper in self.CANONICAL_EXCHANGES:
            return self.CANONICAL_EXCHANGES[symbol_upper]

        if symbol_upper.endswith(".DE"):
            return "XETR"

        # 2. Local Store
        if not self._mapping:
            self._load_mapping()

        if symbol_upper in self._mapping:
            return self._mapping[symbol_upper]

        return default

    def register_exchange(
        self, symbol: str, exchange: str, *, persist: bool = True
    ) -> None:
        """Registers a symbol-to-exchange mapping in memory and optionally merges to disk."""
        symbol_upper = symbol.upper().strip()
        exchange_upper = exchange.upper().strip()
        self._mapping[symbol_upper] = exchange_upper

        if not persist:
            return

        json_path = settings.get_path("exchange_mapping")
        try:
            existing_data: dict[str, str] = {}
            if json_path.exists():
                with open(json_path, encoding="utf-8") as file:
                    loaded = json.load(file)
                    if isinstance(loaded, dict):
                        existing_data = {str(k): str(v) for k, v in loaded.items()}

            existing_data[symbol_upper] = exchange_upper
            json_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_path = json_path.with_suffix(".tmp")
            with open(temporary_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, indent=2, sort_keys=True)
            temporary_path.replace(json_path)
            logger.debug(
                "Persisted exchange mapping for %s -> %s", symbol_upper, exchange_upper
            )
        except OSError as error:
            logger.error(
                "Failed to persist exchange mapping for %s: %s", symbol_upper, error
            )

    def auto_discover_exchange(self, symbol: str) -> str | None:
        """Attempts to discover and cache a symbol's exchange via yfinance metadata."""
        symbol_upper = symbol.upper().strip()
        try:
            ticker = yf.Ticker(symbol_upper)
            fast_info = getattr(ticker, "fast_info", None)
            raw_exchange = getattr(fast_info, "exchange", None) if fast_info else None
            if not raw_exchange:
                info = getattr(ticker, "info", None) or {}
                raw_exchange = info.get("exchange")

            if raw_exchange:
                normalized = self.normalize_exchange_code(str(raw_exchange))
                if normalized:
                    self.register_exchange(symbol_upper, normalized, persist=True)
                    logger.info(
                        "Auto-discovered exchange for %s: %s (raw: %s)",
                        symbol_upper,
                        normalized,
                        raw_exchange,
                    )
                    return normalized
        except Exception as error:
            logger.debug("Auto-discovery failed for %s: %s", symbol_upper, error)

        return None


# Global instance (initially empty, populated via load())
mapper: ExchangeMapper = ExchangeMapper()
