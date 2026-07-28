import logging
import os
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypedDict

import yaml
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_FILE = BASE_DIR / "settings.yaml"

# Logger Setup (No basicConfig side-effect)
logger = logging.getLogger("Config")

load_dotenv()


# ---------------------------------------------------------
# Defaults (Constants for robustness)
# ---------------------------------------------------------

DEFAULT_DB_FILES = {
    "signals": "signals.db",
    "stocks": "stocks.db",
    "strategies": "strategies.db",
    "strategy_yaml": "croc-strategie.yaml",
    "exchange_mapping": "symbol_exchange.json",
    "stats_import": "croc_statistik.csv",
    "ranking_yaml": "ranking_2026.yaml",
    "holidays_yaml": "holidays.yaml",
    "trading": "../tws/data/trading.db",
}

DEFAULT_DB_FOLDERS = {"orders": "orders"}


# ---------------------------------------------------------
# Part 1: Nested Configuration Dataclasses
# ---------------------------------------------------------


@dataclass
class DatabaseConfig:
    """Manages paths to databases and data files."""

    base_folder: str = "data"
    files: dict[str, str] = field(default_factory=DEFAULT_DB_FILES.copy)
    folders: dict[str, str] = field(default_factory=DEFAULT_DB_FOLDERS.copy)


@dataclass(frozen=True)
class WebserverConfig:
    """Web server binding configuration."""

    host: str = "127.0.0.1"
    port: int = 5000
    debug: bool = False


@dataclass(frozen=True)
class LoggingConfig:
    """Logging output configuration."""

    base_folder: str = "logs"
    file_name: str = "croc-trader.log"
    level: str = "info"


@dataclass(frozen=True)
class WebhookWorkerConfig:
    """Webhook worker pool configuration."""

    size: int = 40
    timeout: int = 5


@dataclass(frozen=True)
class SecurityConfig:
    """IP whitelist and access control configuration."""

    mode: str = "warning"
    whitelist: tuple[str, ...] = ()


@dataclass(frozen=True)
class TelegramConfig:
    """Telegram notification bot configuration."""

    token: str = ""
    chat_id: str = ""
    enabled: bool = False


@dataclass(frozen=True)
class PortfolioStrategyConfig:
    """Sizing parameters for a single strategy."""

    budget: float = 0.0
    risk_amount: float = 0.0


@dataclass(frozen=True)
class PortfolioConfig:
    """Central sizing configuration for all strategies."""

    dip_buyer: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=2500.0)
    )
    turnover_timing: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=2500.0)
    )
    two_percent: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=2000.0)
    )
    ndx_momentum: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=10000.0)
    )
    tgim: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=10000.0)
    )
    bridge_scout: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=10000.0)
    )
    bounce_bandit: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(budget=10000.0)
    )
    croc_setup: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(risk_amount=100.0)
    )
    hold_target: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(risk_amount=100.0)
    )
    split_target: PortfolioStrategyConfig = field(
        default_factory=lambda: PortfolioStrategyConfig(risk_amount=100.0)
    )

    def get_budget(self, strategy_key: str) -> float:
        """Returns the budget for a strategy. 0.0 if not budget-based."""
        normalized_key = strategy_key.replace(".", "_").lower()
        if normalized_key.startswith("turnover_timing"):
            normalized_key = "turnover_timing"
        from .const import STRATEGY_ALIASES

        canonical = STRATEGY_ALIASES.get(normalized_key)
        target_name = canonical.value if hasattr(canonical, "value") else normalized_key
        config = getattr(self, str(target_name), None)
        return config.budget if config else 0.0

    def get_risk_amount(self, strategy_key: str) -> float:
        """Returns the risk amount for a strategy. 0.0 if not risk-based."""
        normalized_key = strategy_key.replace(".", "_").lower()
        from .const import STRATEGY_ALIASES

        canonical = STRATEGY_ALIASES.get(normalized_key)
        target_name = canonical.value if hasattr(canonical, "value") else normalized_key
        config = getattr(self, str(target_name), None)
        return config.risk_amount if config else 0.0


class SymbolOverride(TypedDict, total=False):
    """Optional contract definition overrides for a specific symbol in order generation."""

    target_symbol: str
    sec_type: str
    exchange: str
    currency: str


# ---------------------------------------------------------
# Part 2: Main Application Configuration
# ---------------------------------------------------------


@dataclass
class AppConfig:
    """Top-level application configuration combining all subsections."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    webserver: WebserverConfig = field(default_factory=WebserverConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    worker: WebhookWorkerConfig = field(default_factory=WebhookWorkerConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    portfolio: PortfolioConfig = field(default_factory=PortfolioConfig)
    order_overrides: dict[str, SymbolOverride] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AppConfig":
        """Recursively converts a dictionary (from YAML) into nested dataclasses.

        Performs smart-merging to fill in missing keys from older config files.
        """
        # 1. Database config with merge logic
        db_data = data.get("database", {})
        if db_data:
            # Files merging: start from defaults, override with loaded values.
            # Existing paths are preserved while new keys are added automatically.
            loaded_files = db_data.get("files", {})
            merged_files = DEFAULT_DB_FILES.copy()
            merged_files.update(loaded_files)
            db_data["files"] = merged_files

            # Folders merging
            loaded_folders = db_data.get("folders", {})
            merged_folders = DEFAULT_DB_FOLDERS.copy()
            merged_folders.update(loaded_folders)
            db_data["folders"] = merged_folders

            db_config = DatabaseConfig(**db_data)
        else:
            db_config = DatabaseConfig()

        # 2. Remaining configs
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

        security_data = data.get("security", {})
        # Convert list whitelist to tuple for frozen SecurityConfig
        if security_data and "whitelist" in security_data:
            security_data["whitelist"] = tuple(security_data["whitelist"])
        security_config = (
            SecurityConfig(**security_data) if security_data else SecurityConfig()
        )

        telegram_data = data.get("telegram", {})
        telegram_config = (
            TelegramConfig(**telegram_data) if telegram_data else TelegramConfig()
        )

        portfolio_data = data.get("portfolio", {})
        if portfolio_data and "strategies" in portfolio_data:
            strategies_data = portfolio_data["strategies"]
            strat_configs = {}
            if isinstance(strategies_data, dict):
                for strat_name, strat_val in strategies_data.items():
                    if isinstance(strat_val, dict):
                        strat_configs[strat_name] = PortfolioStrategyConfig(
                            budget=float(strat_val.get("budget") or 0.0),
                            risk_amount=float(strat_val.get("risk_amount") or 0.0),
                        )
            elif isinstance(strategies_data, list):
                for entry in strategies_data:
                    if isinstance(entry, dict):
                        for strat_name, strat_val in entry.items():
                            budget_val = 0.0
                            risk_val = 0.0
                            if isinstance(strat_val, list):
                                for prop in strat_val:
                                    if isinstance(prop, dict):
                                        if "quantity" in prop:
                                            budget_val = float(prop["quantity"])
                                        elif "budget" in prop:
                                            budget_val = float(prop["budget"])
                                        elif "risk_amount" in prop:
                                            risk_val = float(prop["risk_amount"])
                            elif isinstance(strat_val, dict):
                                budget_val = float(
                                    strat_val.get("budget")
                                    or strat_val.get("quantity")
                                    or 0.0
                                )
                                risk_val = float(strat_val.get("risk_amount") or 0.0)
                            strat_configs[strat_name] = PortfolioStrategyConfig(
                                budget=budget_val,
                                risk_amount=risk_val,
                            )

            def get_strat_config(
                name: str, default_budget: float = 0.0, default_risk: float = 0.0
            ) -> PortfolioStrategyConfig:
                if name in strat_configs:
                    return strat_configs[name]
                return PortfolioStrategyConfig(
                    budget=default_budget, risk_amount=default_risk
                )

            portfolio_config = PortfolioConfig(
                dip_buyer=get_strat_config("dip_buyer", default_budget=2500.0),
                turnover_timing=get_strat_config(
                    "turnover_timing", default_budget=2500.0
                ),
                two_percent=get_strat_config("two_percent", default_budget=2000.0),
                ndx_momentum=get_strat_config("ndx_momentum", default_budget=10000.0),
                tgim=get_strat_config("tgim", default_budget=10000.0),
                bridge_scout=get_strat_config("bridge_scout", default_budget=10000.0),
                bounce_bandit=get_strat_config("bounce_bandit", default_budget=10000.0),
                croc_setup=get_strat_config("croc_setup", default_risk=100.0),
                hold_target=get_strat_config("hold_target", default_risk=100.0),
                split_target=get_strat_config("split_target", default_risk=100.0),
            )
        else:
            portfolio_config = PortfolioConfig()

        order_overrides_data = data.get("order_overrides", {})
        order_overrides: dict[str, SymbolOverride] = {}
        if isinstance(order_overrides_data, dict):
            for sym, override in order_overrides_data.items():
                if isinstance(override, dict):
                    override_typed: SymbolOverride = {}
                    if "target_symbol" in override:
                        override_typed["target_symbol"] = str(override["target_symbol"])
                    if "sec_type" in override:
                        override_typed["sec_type"] = str(override["sec_type"])
                    if "exchange" in override:
                        override_typed["exchange"] = str(override["exchange"])
                    if "currency" in override:
                        override_typed["currency"] = str(override["currency"])
                    order_overrides[str(sym)] = override_typed

        return cls(
            database=db_config,
            webserver=web_config,
            logging=logging_config,
            worker=worker_config,
            security=security_config,
            telegram=telegram_config,
            portfolio=portfolio_config,
            order_overrides=order_overrides,
        )


# ---------------------------------------------------------
# Part 3: Environment Config (Secrets)
# ---------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    APP_ENV: str
    SECRET_KEY: str


# ---------------------------------------------------------
# Part 4: Configuration Manager
# ---------------------------------------------------------


class ConfigManager:
    def __init__(self):
        # Expose the base directory for path resolution
        self.BASE_DIR = BASE_DIR

        self.env = self._load_env()
        self.app = self._load_or_create_yaml()

        self._apply_env_overrides()

        # Computed absolute path for databases (base_folder defaults to 'data')
        self.db_root_path = BASE_DIR / self.app.database.base_folder
        self.db_root_path.mkdir(parents=True, exist_ok=True)

        # Computed absolute path for log files
        self.logging_root_path = BASE_DIR / self.app.logging.base_folder
        self.logging_root_path.mkdir(parents=True, exist_ok=True)

    def _load_env(self) -> EnvConfig:
        app_env = os.getenv("APP_ENV", "development")
        secret_key = os.getenv("FLASK_SECRET_KEY")
        if not secret_key:
            if app_env == "production":
                raise RuntimeError(
                    "❌ SECURITY: FLASK_SECRET_KEY is not set in a production environment!"
                )
            secret_key = secrets.token_hex(32)

        return EnvConfig(
            APP_ENV=app_env,
            SECRET_KEY=secret_key,
        )

    def _apply_env_overrides(self) -> None:
        """Applies environment variable overrides to the loaded config.

        Creates a new frozen TelegramConfig with overridden values rather
        than mutating the existing instance.
        """
        telegram_token = os.getenv("TELEGRAM_TOKEN", self.app.telegram.token)
        telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID", self.app.telegram.chat_id)
        env_enabled = os.getenv("TELEGRAM_ENABLED")
        telegram_enabled = (
            env_enabled.lower() == "true" if env_enabled else self.app.telegram.enabled
        )

        # Replace the frozen TelegramConfig with a new instance
        overridden_telegram = TelegramConfig(
            token=telegram_token,
            chat_id=telegram_chat_id,
            enabled=telegram_enabled,
        )
        # AppConfig is not frozen, so we can reassign its telegram attribute
        self.app.telegram = overridden_telegram

    def _load_or_create_yaml(self) -> AppConfig:
        if not CONFIG_FILE.exists():
            logger.info("Creating default config: %s", CONFIG_FILE)
            default_config = AppConfig()

            def dataclass_to_dict(obj):
                if hasattr(obj, "__dataclass_fields__"):
                    return {k: dataclass_to_dict(v) for k, v in obj.__dict__.items()}
                if isinstance(obj, list | tuple):
                    return [dataclass_to_dict(v) for v in obj]
                if isinstance(obj, dict):
                    return {k: dataclass_to_dict(v) for k, v in obj.items()}
                return obj

            try:
                with open(CONFIG_FILE, "w") as f:
                    yaml.dump(dataclass_to_dict(default_config), f, sort_keys=False)
            except Exception as error:
                logger.error("Could not write config: %s", error)

            return default_config

        try:
            with open(CONFIG_FILE) as f:
                data = yaml.safe_load(f) or {}
                return AppConfig.from_dict(data)
        except Exception as e:
            logger.error("Failed to load config: %s", e)
            sys.exit(1)

    def get_path(self, key: str) -> Path:
        """Retrieves a configured file path with path-traversal protection."""
        filename = self.app.database.files.get(key)
        if not filename:
            return self.db_root_path / f"MISSING_CONFIG_{key}"

        target_path = (self.db_root_path / filename).resolve()

        # Security: Prevent Path Traversal (allow base data folder or workspace root)
        if not (
            str(target_path).startswith(str(self.db_root_path.resolve()))
            or str(target_path).startswith(str(BASE_DIR.resolve()))
        ):
            logger.error("❌ SECURITY: Path Traversal Attempt blocked: %s", filename)
            raise ValueError(f"Insecure path detected: {filename}")

        return target_path

    def get_folder(self, key: str) -> Path:
        """Retrieves a configured folder path with path-traversal protection."""
        foldername = self.app.database.folders.get(key, key)
        target_path = (self.db_root_path / foldername).resolve()

        # Security: Prevent Path Traversal (allow base data folder or workspace root)
        if not (
            str(target_path).startswith(str(self.db_root_path.resolve()))
            or str(target_path).startswith(str(BASE_DIR.resolve()))
        ):
            logger.error(
                "❌ SECURITY: Path Traversal Attempt blocked in folder: %s",
                foldername,
            )
            raise ValueError(f"Insecure folder path detected: {foldername}")

        target_path.mkdir(parents=True, exist_ok=True)
        return target_path

    # Helper methods
    def get_db_path(self, db_key: str) -> str:
        return str(self.get_path(db_key))

    def get_log_path(self) -> str:
        filename = self.app.logging.file_name
        return str(self.logging_root_path / filename)

    def get_strategy_path(self) -> Path:
        return self.get_path("strategy_yaml")


# Singleton
settings = ConfigManager()
