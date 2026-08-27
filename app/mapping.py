import json
import logging
from typing import ClassVar

from .config import settings

logger = logging.getLogger(__name__)


class ExchangeMapper:
    """Singleton mapping equity symbols to their primary exchange."""

    _instance: ClassVar["ExchangeMapper | None"] = None
    _mapping: dict[str, str] = {}

    DEFAULT_ETF_EXCHANGES: ClassVar[dict[str, str]] = {
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
        "SXRV.DE": "XETR",
    }

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

    def get_exchange(self, symbol: str, default: str | None = None) -> str | None:
        """Returns the exchange for a symbol, or the default value."""
        # Fallback: auto-load if load() was not called yet
        if not self._mapping:
            self._load_mapping()

        symbol_upper = symbol.upper()
        if symbol_upper in self._mapping:
            return self._mapping[symbol_upper]
        if symbol_upper in self.DEFAULT_ETF_EXCHANGES:
            return self.DEFAULT_ETF_EXCHANGES[symbol_upper]

        return default


# Global instance (initially empty, populated via load())
mapper: ExchangeMapper = ExchangeMapper()
