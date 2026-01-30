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
            # WICHTIG: Nicht mehr sofort laden!
            # cls._instance._load_mapping()
        return cls._instance

    def load(self):
        """
        Lädt das Mapping explizit.
        Wird von create_app() aufgerufen, wenn das Logging bereit ist.
        """
        if not self._mapping:
            self._load_mapping()

    def _load_mapping(self):
        """Lädt die JSON Datei einmalig in den Speicher."""
        json_path = settings.get_path("exchange_mapping")

        if not json_path.exists():
            logger.warning(f"Exchange Mapping Datei nicht gefunden: {json_path}")
            return

        try:
            with open(json_path, encoding="utf-8") as f:
                self._mapping = json.load(f)
            logger.info(f"Exchange Mapping geladen: {len(self._mapping)} Symbole.")
        except Exception as e:
            logger.error(f"Fehler beim Laden der Exchange JSON: {e}")

    def get_exchange(self, symbol: str, default: str | None = None) -> str:
        """Gibt den Exchange für ein Symbol zurück oder den Default-Wert."""
        # Fallback: Falls load() vergessen wurde, hier versuchen (Silent Auto-Load)
        if not self._mapping:
            self._load_mapping()

        return self._mapping.get(symbol.upper(), default)


# Globale Instanz (jetzt initial leer)
mapper = ExchangeMapper()
