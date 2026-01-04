import json
import logging
from typing import Dict, Optional

from .config import settings

logger = logging.getLogger(__name__)


class ExchangeMapper:
    _instance = None
    _mapping: Dict[str, str] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ExchangeMapper, cls).__new__(cls)
            cls._instance._load_mapping()
        return cls._instance

    def _load_mapping(self):
        """Lädt die JSON Datei einmalig in den Speicher."""
        json_path = settings.db_root_path / "symbol_exchange.json"

        if not json_path.exists():
            logger.warning(f"Exchange Mapping Datei nicht gefunden: {json_path}")
            return

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                self._mapping = json.load(f)
            logger.info(f"Exchange Mapping geladen: {len(self._mapping)} Symbole.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Exchange JSON: {e}")

    def get_exchange(self, symbol: str, default: Optional[str] = None) -> str:
        """Gibt den Exchange für ein Symbol zurück oder den Default-Wert."""
        # Wir suchen case-insensitive, da Symbole oft uppercase sind
        return self._mapping.get(symbol.upper(), default)


# Globale Instanz für einfachen Zugriff
mapper = ExchangeMapper()
