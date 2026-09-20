"""Backward-compatibility adapter for symbol exchange resolution.

Delegates all symbol resolution directly to canonical ExchangeMapper (app.mapping.mapper),
ensuring zero duplicate caches and preventing background thread cache overwrites.
"""

import logging

from app.mapping import ExchangeMapper, mapper

logger = logging.getLogger(__name__)


class SymbolExchange:
    """Singleton adapter delegating directly to canonical ExchangeMapper."""

    _instance: "SymbolExchange | None" = None

    def __new__(cls) -> "SymbolExchange":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, exchange_mapper: ExchangeMapper | None = None) -> None:
        self._mapper = exchange_mapper or mapper

    def get_exchange(self, symbol: str) -> str:
        """Returns exchange for symbol, defaulting to 'NASDAQ' if unresolvable."""
        return self._mapper.get_exchange(symbol, default="NASDAQ") or "NASDAQ"

    @property
    def mapping(self) -> dict[str, str]:
        """Returns a copy of the active mapping."""
        # Ensure mapping is loaded
        if not self._mapper._mapping:
            self._mapper._load_mapping()
        return self._mapper._mapping.copy()
