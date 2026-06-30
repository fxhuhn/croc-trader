import json
import logging

from .config import settings

logger = logging.getLogger(__name__)


class ExchangeMapper:
    _instance = None
    _mapping: dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExchangeMapper, cls).__new__(cls)
        return cls._instance

    def load(self):
        """Explicitly loads the mapping. Called from create_app() when logging is ready."""
        if not self._mapping:
            self._load_mapping()

    def _load_mapping(self):
        """Loads the JSON mapping file into memory (one-time)."""
        json_path = settings.get_path("exchange_mapping")

        if not json_path.exists():
            logger.warning("Exchange mapping file not found: %s", json_path)
            return

        try:
            with open(json_path, encoding="utf-8") as f:
                self._mapping = json.load(f)
            logger.info("Exchange mapping loaded: %d symbols.", len(self._mapping))
        except Exception as e:
            logger.error("Failed to load exchange JSON: %s", e)

    def get_exchange(self, symbol: str, default: str | None = None) -> str:
        """Returns the exchange for a symbol, or the default value."""
        # Fallback: auto-load if load() was not called yet
        if not self._mapping:
            self._load_mapping()

        return self._mapping.get(symbol.upper(), default)


# Global instance (initially empty, populated via load())
mapper = ExchangeMapper()
