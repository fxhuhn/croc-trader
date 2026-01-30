import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = (
    BASE_DIR / "data" / "settings.yaml"
)  # Pfad explizit auf data/settings.yaml prüfen, falls gewünscht, oder Standard lassen.
# HINWEIS: Im originalen Code war CONFIG_FILE = BASE_DIR / "settings.yaml".
# Wenn Sie settings.yaml im root haben, lassen wir es so.
# Wenn settings.yaml in data/ liegt, müsste man das anpassen.
# Standard aus Ihrem Code war:
CONFIG_FILE = BASE_DIR / "settings.yaml"

# Logger Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Config")

load_dotenv()


# ---------------------------------------------------------
# DEFAULTS (Konstanten für Robustheit)
# ---------------------------------------------------------

DEFAULT_DB_FILES = {
    "signals": "signals.db",
    "stocks": "stocks.db",
    "strategies": "strategies.db",
    "strategy_yaml": "croc-strategie.yaml",
    "exchange_mapping": "symbol_exchange.json",
    "stats_import": "croc_statistik.csv",
    "ranking_yaml": "ranking_2026.yaml",  # NEU
}

DEFAULT_DB_FOLDERS = {"orders": "orders"}


# ---------------------------------------------------------
# Teil 1: Die Unter-Konfigurationen (Nested Classes)
# ---------------------------------------------------------


@dataclass
class DatabaseConfig:
    """Verwaltet Pfade zu Datenbanken und Datendateien."""

    base_folder: str = "data"
    files: dict[str, str] = field(default_factory=lambda: DEFAULT_DB_FILES.copy())
    folders: dict[str, str] = field(default_factory=lambda: DEFAULT_DB_FOLDERS.copy())


@dataclass
class WebserverConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass
class LoggingConfig:
    base_folder: str = "logs"
    file_name: str = "croc-trader.log"
    level: str = "info"


@dataclass
class WebhookWorkerConfig:
    size: int = 40
    timeout: int = 5


@dataclass
class SecurityConfig:
    mode: str = "warning"
    whitelist: list[str] = field(default_factory=list)


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
    """Repräsentiert die gesamte config.yaml Struktur."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    webserver: WebserverConfig = field(default_factory=WebserverConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    worker: WebhookWorkerConfig = field(default_factory=WebhookWorkerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """
        Wandelt ein Dictionary (aus YAML) rekursiv in Dataclasses um.
        Führt Smart-Merging durch, um fehlende Keys in alten Config-Files zu ergänzen.
        """
        # 1. Datenbank Config mit Merge-Logik
        db_data = data.get("database", {})
        if db_data:
            # Files Merging: Defaults nehmen und mit geladenen Werten überschreiben
            # So bleiben existierende Pfade erhalten, aber neue Keys (wie ranking_yaml) kommen dazu.
            loaded_files = db_data.get("files", {})
            merged_files = DEFAULT_DB_FILES.copy()
            merged_files.update(loaded_files)
            db_data["files"] = merged_files

            # Folders Merging
            loaded_folders = db_data.get("folders", {})
            merged_folders = DEFAULT_DB_FOLDERS.copy()
            merged_folders.update(loaded_folders)
            db_data["folders"] = merged_folders

            db_config = DatabaseConfig(**db_data)
        else:
            db_config = DatabaseConfig()

        # 2. Restliche Configs
        web_data = data.get("webserver", {})
        web_config = WebserverConfig(**web_data) if web_data else WebserverConfig()

        logging_data = data.get("logging", {})
        logging_config = (
            LoggingConfig(**logging_data) if logging_data else LoggingConfig()
        )

        worker_data = data.get("webhook_worker", {})
        worker_config = (
            WebhookWorkerConfig(**worker_data) if worker_data else WebhookWorkerConfig()
        )

        sec_data = data.get("security", {})
        sec_config = SecurityConfig(**sec_data) if sec_data else SecurityConfig()

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
        # Basis-Verzeichnis verfügbar machen
        self.BASE_DIR = BASE_DIR

        self.env = self._load_env()
        self.app = self._load_or_create_yaml()

        self._apply_env_overrides()

        # Berechneter Pfad für Datenbanken (Absoluter Pfad)
        # HINWEIS: base_folder ist standardmäßig 'data'
        self.db_root_path = BASE_DIR / self.app.database.base_folder
        self.db_root_path.mkdir(parents=True, exist_ok=True)

        # Berechneter Pfad für Logging (Absoluter Pfad)
        self.logging_root_path = BASE_DIR / self.app.logging.base_folder
        self.logging_root_path.mkdir(parents=True, exist_ok=True)

    def _load_env(self) -> EnvConfig:
        return EnvConfig(
            APP_ENV=os.getenv("APP_ENV", "development"),
            SECRET_KEY=os.getenv("FLASK_SECRET_KEY", "dev-fallback-key"),
        )

    def _apply_env_overrides(self):
        self.app.telegram.token = os.getenv("TELEGRAM_TOKEN", self.app.telegram.token)
        self.app.telegram.chat_id = os.getenv(
            "TELEGRAM_CHAT_ID", self.app.telegram.chat_id
        )
        if env_enabled := os.getenv("TELEGRAM_ENABLED"):
            self.app.telegram.enabled = env_enabled.lower() == "true"

    def _load_or_create_yaml(self) -> AppConfig:
        # Prüfen ob settings.yaml in data/ liegt (Migrationsempfehlung) oder im Root
        # Hier behalten wir die Logik bei: CONFIG_FILE zeigt auf Root/settings.yaml laut Zeile 17

        if not CONFIG_FILE.exists():
            logger.info(f"Erstelle Standard Config: {CONFIG_FILE}")
            default_conf = AppConfig()

            def dataclass_to_dict(obj):
                if hasattr(obj, "__dataclass_fields__"):
                    return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
                return obj

            try:
                with open(CONFIG_FILE, "w") as f:
                    yaml.dump(dataclass_to_dict(default_conf), f, sort_keys=False)
            except Exception as e:
                logger.error(f"Konnte Config nicht schreiben: {e}")

            return default_conf

        try:
            with open(CONFIG_FILE) as f:
                data = yaml.safe_load(f) or {}
                # Hier greift nun das Smart-Merging in from_dict
                return AppConfig.from_dict(data)
        except Exception as e:
            logger.error(f"Fehler beim Laden der Config: {e}")
            sys.exit(1)

    def get_path(self, key: str) -> Path:
        """Universeller Abruf für konfigurierte Dateipfade."""
        filename = self.app.database.files.get(key)
        if not filename:
            # Fallback, falls Key wirklich fehlt -> Loggt Fehler im Caller
            return self.db_root_path / f"MISSING_CONFIG_{key}"
        return self.db_root_path / filename

    def get_folder(self, key: str) -> Path:
        foldername = self.app.database.folders.get(key, key)
        path = self.db_root_path / foldername
        path.mkdir(parents=True, exist_ok=True)
        return path

    # Helper Methoden
    def get_db_path(self, db_key: str) -> str:
        return str(self.get_path(db_key))

    def get_log_path(self) -> str:
        filename = self.app.logging.file_name
        return str(self.logging_root_path / filename)

    def get_strategy_path(self) -> Path:
        return self.get_path("strategy_yaml")


# Singleton
settings = ConfigManager()
