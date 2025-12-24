import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

_base_path = Path(__file__).parent.parent
# --- Configuration Schema (Dataclasses) ---


@dataclass
class AppConfig:
    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass
class DatabaseConfig:
    path: str = "local_database.db"


@dataclass
class Tradingview:
    base_url: str = "https://tradingview/"
    token: str = "change_me"


@dataclass
class LoggingConfig:
    level: str = "INFO"
    path: str = "app/logs/app.log"


@dataclass
class Config:
    app: AppConfig = field(default_factory=AppConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    tradingview: Tradingview = field(default_factory=Tradingview)


# --- Loading Logic ---


def _merge_dict(base_config: Config, data: dict):
    """Recursively update dataclass fields from a dictionary."""
    for key, value in data.items():
        if hasattr(base_config, key):
            attr = getattr(base_config, key)
            if (
                isinstance(attr, object)
                and hasattr(attr, "__dataclass_fields__")
                and isinstance(value, dict)
            ):
                # Recurse into nested dataclass
                _merge_dict(attr, value)
            else:
                # Set simple value (with basic type casting if needed)
                field_type = type(getattr(base_config, key))
                if field_type == bool and isinstance(value, str):
                    value = value.lower() in ("true", "1", "yes")
                elif field_type == int and isinstance(value, str):
                    value = int(value)
                elif field_type == float and isinstance(value, str):
                    value = float(value)
                elif key == "path" and isinstance(value, str):
                    value = Path(_base_path, value)
                setattr(base_config, key, value)
                logging.info(f"Config: {key} = {value}")


def load_config(config_yaml: str = "app/config.yaml") -> Config:
    """
    Load configuration from YAML and override with Environment Variables.
    """
    cfg = Config()

    # 1. Load from YAML
    path = Path(_base_path, config_yaml)
    if path.exists():
        logging.info(f"Config: load data from {path}")
        with open(path, "r") as f:
            yaml_data = yaml.safe_load(f) or {}
            _merge_dict(cfg, yaml_data)

    # 2. Override from Environment Variables (Secrets)
    load_dotenv()  # Load from .env file if present

    # App Section
    if os.getenv("APP_HOST"):
        cfg.app.host = os.getenv("APP_HOST")
        logging.info(f"Config: update app: host = {cfg.app.host}")

    if os.getenv("APP_PORT"):
        cfg.app.port = int(os.getenv("APP_PORT"))
        logging.info(f"Config: update app: port = {cfg.app.port}")

    # Database Section
    if os.getenv("DB_PATH"):
        cfg.database.path = Path(_base_path, os.getenv("DB_PATH"))
        logging.info(f"Config: update database: path = {cfg.database.path}")

    return cfg
