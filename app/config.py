import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "settings.yaml"


# Logger Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Config")

load_dotenv()


# ---------------------------------------------------------
# Teil 1: Die Unter-Konfigurationen (Nested Classes)
# ---------------------------------------------------------


@dataclass
class DatabaseConfig:
    """Verwaltet Datenbank-Einstellungen."""

    base_folder: str = "data"
    files: Dict[str, str] = field(
        default_factory=lambda: {
            "signals": "signals.db",
            # "trades": "trades.db",
            # "backtest": "backtest.db",
            "stocks": "stocks.db",
            # Optional: Man könnte hier auch den Dateinamen der Strategie hinterlegen
            # "strategy": "croc-strategie.yaml"
        }
    )


@dataclass
class WebserverConfig:
    """Verwaltet Webserver-Einstellungen."""

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass
class LoggingConfig:
    """Verwaltet Logging-Einstellungen."""

    base_folder: str = "logs"
    file_name: str = "croc-trader.log"
    level: str = "info"


@dataclass
class WebhookWorkerConfig:
    """Verwaltet Worker-Einstellungen."""

    size: int = 40
    timeout: int = 5


@dataclass
class SecurityConfig:
    mode: str = "warning"
    whitelist: List[str] = field(default_factory=list)


@dataclass
class TelegramConfig:
    token: str = ""
    chat_id: str = ""
    enabled: bool = False


# ---------------------------------------------------------
# Teil 2: Die Haupt-Applikations-Konfiguration
# ---------------------------------------------------------


@dataclass
class AppConfig:
    """
    Repräsentiert die gesamte config.yaml Struktur.
    """

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    webserver: WebserverConfig = field(default_factory=WebserverConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    worker: WebhookWorkerConfig = field(default_factory=WebhookWorkerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """
        Wandelt ein Dictionary (aus YAML) rekursiv in Dataclasses um.
        """
        # 1. Datenbank Config
        db_data = data.get("database", {})
        db_config = DatabaseConfig(**db_data) if db_data else DatabaseConfig()

        # 2. Webserver Config
        web_data = data.get("webserver", {})
        web_config = WebserverConfig(**web_data) if web_data else WebserverConfig()

        # 3. Logging Config
        logging_data = data.get("logging", {})
        logging_config = (
            LoggingConfig(**logging_data) if logging_data else LoggingConfig()
        )

        # 4. Webhook Worker Config
        worker_data = data.get("webhook_worker", {})
        worker_config = (
            WebhookWorkerConfig(**worker_data) if worker_data else WebhookWorkerConfig()
        )

        # 5. Security Config
        sec_data = data.get("security", {})
        sec_config = SecurityConfig(**sec_data) if sec_data else SecurityConfig()

        # 6. Telegram Config
        tele_data = data.get("telegram", {})
        tele_config = TelegramConfig(**tele_data) if tele_data else TelegramConfig()

        return cls(
            database=db_config,
            webserver=web_config,
            logging=logging_config,
            worker=worker_config,
            security=sec_config,
            telegram=tele_config,
        )


# ---------------------------------------------------------
# Teil 3: Environment Config (Secrets)
# ---------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    APP_ENV: str
    SECRET_KEY: str


# ---------------------------------------------------------
# Teil 4: Der Manager
# ---------------------------------------------------------


class ConfigManager:
    def __init__(self):
        self.env = self._load_env()
        self.app = self._load_or_create_yaml()

        # NEU: Hier überschreiben wir die YAML-Werte mit ENV-Werten, falls vorhanden
        self._apply_env_overrides()

        # Berechneter Pfad für Datenbanken (Absoluter Pfad)
        self.db_root_path = BASE_DIR / self.app.database.base_folder
        self.db_root_path.mkdir(parents=True, exist_ok=True)

        # Berechneter Pfad für Logging (Absoluter Pfad)
        self.logging_root_path = BASE_DIR / self.app.logging.base_folder
        self.logging_root_path.mkdir(parents=True, exist_ok=True)

    def _apply_env_overrides(self):
        """
        Überschreibt YAML-Werte mit Environment-Variablen (kurz & bündig).
        """
        # 1. Strings (Token & Chat ID)
        # Nimm ENV, sonst behalte den Wert aus der YAML
        self.app.telegram.token = os.getenv("TELEGRAM_TOKEN", self.app.telegram.token)
        self.app.telegram.chat_id = os.getenv(
            "TELEGRAM_CHAT_ID", self.app.telegram.chat_id
        )

    def _load_env(self) -> EnvConfig:
        return EnvConfig(
            APP_ENV=os.getenv("APP_ENV", "development"),
            SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-fallback-key"),
        )

    def _load_or_create_yaml(self) -> AppConfig:
        if not CONFIG_FILE.exists():
            logger.info("Erstelle Standard settings.yaml...")
            default_conf = AppConfig()

            # Helper um nested dataclasses in dicts zu wandeln
            def dataclass_to_dict(obj):
                if hasattr(obj, "__dataclass_fields__"):
                    return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
                return obj

            with open(CONFIG_FILE, "w") as f:
                yaml.dump(dataclass_to_dict(default_conf), f, sort_keys=False)
            return default_conf

        try:
            with open(CONFIG_FILE, "r") as f:
                data = yaml.safe_load(f) or {}
                return AppConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Config: {e}")
            sys.exit(1)

    def get_db_path(self, db_key: str) -> str:
        """Liefert den vollen Pfad zu einer spezifischen DB."""
        filename = self.app.database.files.get(db_key)
        if not filename:
            # Fallback oder Fehler, je nach Wunsch. Hier Fehler.
            raise KeyError(
                f"DB '{db_key}' nicht in config.yaml unter database.files gefunden."
            )
        return str(self.db_root_path / filename)

    def get_log_path(self) -> str:
        """Liefert den vollen Pfad zur Logdatei."""
        filename = self.app.logging.file_name
        return str(self.logging_root_path / filename)

    def get_strategy_path(self) -> Path:
        """
        NEU: Liefert den vollen Pfad zur Strategie-YAML Datei.
        Der Dateiname ist aktuell fest 'croc-strategie.yaml', liegt aber
        im konfigurierten Daten-Ordner (db_root_path).
        """
        return self.db_root_path / "croc-strategie.yaml"


# Singleton
settings = ConfigManager()
