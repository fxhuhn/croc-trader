"""Unit tests for app/config.py configuration dataclasses and ConfigManager."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from app.config import (
    AppConfig,
    ConfigManager,
    DatabaseConfig,
    EnvConfig,
    LoggingConfig,
    PortfolioConfig,
    SecurityConfig,
    TelegramConfig,
    WebhookWorkerConfig,
    WebserverConfig,
    _parse_database_config,
    _parse_order_overrides,
    _parse_portfolio_config,
)


def test_default_dataclass_instantiation() -> None:
    """Tests default attribute values for configuration dataclasses."""
    db_cfg = DatabaseConfig()
    assert db_cfg.base_folder == "data"
    assert "signals" in db_cfg.files

    web_cfg = WebserverConfig()
    assert web_cfg.host == "127.0.0.1"
    assert web_cfg.port == 5000

    log_cfg = LoggingConfig()
    assert log_cfg.file_name == "croc-trader.log"

    worker_cfg = WebhookWorkerConfig()
    assert worker_cfg.size == 40

    sec_cfg = SecurityConfig()
    assert sec_cfg.mode == "warning"
    assert sec_cfg.whitelist == ()

    tg_cfg = TelegramConfig()
    assert not tg_cfg.enabled

    port_cfg = PortfolioConfig()
    assert port_cfg.dip_buyer.budget == 2500.0
    assert port_cfg.croc_setup.risk_amount == 100.0


def test_portfolio_config_get_budget_and_risk() -> None:
    """Tests get_budget and get_risk_amount with aliases and unknown strategies."""
    port_cfg = PortfolioConfig()

    # Normal & Aliased budgets
    assert port_cfg.get_budget("dip_buyer") == 2500.0
    assert port_cfg.get_budget("turnover_timing.1d") == 2500.0
    assert port_cfg.get_budget("unknown_strategy") == 0.0

    # Normal & Aliased risk amounts
    assert port_cfg.get_risk_amount("croc_setup") == 100.0
    assert port_cfg.get_risk_amount("hold_target") == 100.0
    assert port_cfg.get_risk_amount("unknown_strategy") == 0.0


def test_parse_database_config() -> None:
    """Tests merging logic in _parse_database_config."""
    assert isinstance(_parse_database_config({}), DatabaseConfig)

    custom_data = {
        "base_folder": "custom_data",
        "files": {"signals": "custom_signals.db"},
        "folders": {"orders": "custom_orders"},
    }
    parsed = _parse_database_config(custom_data)
    assert parsed.base_folder == "custom_data"
    assert parsed.files["signals"] == "custom_signals.db"
    assert parsed.files["stocks"] == "stocks.db"  # Merged default
    assert parsed.folders["orders"] == "custom_orders"


def test_parse_portfolio_config_dict_and_list() -> None:
    """Tests parsing raw dictionary and list structures for portfolio strategy parameters."""
    assert isinstance(_parse_portfolio_config({}), PortfolioConfig)

    # Parsing dict format
    dict_data = {
        "strategies": {
            "dip_buyer": {"budget": 3000.0, "risk_amount": 50.0},
            "croc_setup": {"risk_amount": 150.0},
        }
    }
    parsed_dict = _parse_portfolio_config(dict_data)
    assert parsed_dict.dip_buyer.budget == 3000.0
    assert parsed_dict.croc_setup.risk_amount == 150.0

    # Parsing list format
    list_data = {
        "strategies": [
            {"turnover_timing": [{"budget": 4000.0}]},
            {"two_percent": [{"quantity": 1500.0}]},
            {"croc_setup": [{"risk_amount": 200.0}]},
        ]
    }
    parsed_list = _parse_portfolio_config(list_data)
    assert parsed_list.turnover_timing.budget == 4000.0
    assert parsed_list.two_percent.budget == 1500.0
    assert parsed_list.croc_setup.risk_amount == 200.0

    # Parsing nested list with dict entry
    list_data_dict = {
        "strategies": [
            {"ndx_momentum": {"budget": 12000.0, "risk_amount": 0.0}},
        ]
    }
    parsed_list_dict = _parse_portfolio_config(list_data_dict)
    assert parsed_list_dict.ndx_momentum.budget == 12000.0


def test_parse_order_overrides() -> None:
    """Tests parsing symbol override options."""
    assert _parse_order_overrides({}) == {}
    assert _parse_order_overrides(None) == {}  # type: ignore[arg-type]

    raw_overrides = {
        "AAPL": {
            "target_symbol": "AAPL_US",
            "sec_type": "STK",
            "exchange": "SMART",
            "currency": "USD",
        }
    }
    parsed = _parse_order_overrides(raw_overrides)
    assert parsed["AAPL"]["target_symbol"] == "AAPL_US"
    assert parsed["AAPL"]["sec_type"] == "STK"


def test_app_config_from_dict() -> None:
    """Tests recursive AppConfig creation from raw dictionary."""
    data = {
        "webserver": {"host": "0.0.0.0", "port": 8080, "debug": True},
        "security": {"whitelist": ["127.0.0.1"]},
        "telegram": {"token": "123:abc", "enabled": True},
    }
    cfg = AppConfig.from_dict(data)
    assert cfg.webserver.host == "0.0.0.0"
    assert cfg.security.whitelist == ("127.0.0.1",)
    assert cfg.telegram.token == "123:abc"


def test_config_manager_load_env() -> None:
    """Tests environment loading, random key generation, and production security check."""
    with patch.dict(os.environ, {"APP_ENV": "development", "FLASK_SECRET_KEY": ""}):
        cm = ConfigManager()
        assert isinstance(cm.env, EnvConfig)
        assert len(cm.env.SECRET_KEY) > 0

    with patch.dict(
        os.environ, {"APP_ENV": "production", "FLASK_SECRET_KEY": ""}, clear=True
    ):
        with pytest.raises(
            RuntimeError,
            match="FLASK_SECRET_KEY is not set in a production environment",
        ):
            ConfigManager()


def test_config_manager_apply_env_overrides() -> None:
    """Tests overriding Telegram settings via environment variables."""
    env_vars = {
        "TELEGRAM_TOKEN": "token_override",
        "TELEGRAM_CHAT_ID": "chat_override",
        "TELEGRAM_ENABLED": "true",
    }
    with patch.dict(os.environ, env_vars):
        cm = ConfigManager()
        assert cm.app.telegram.token == "token_override"
        assert cm.app.telegram.chat_id == "chat_override"
        assert cm.app.telegram.enabled is True


def test_config_manager_load_or_create_yaml(tmp_path: Path) -> None:
    """Tests loading existing YAML config and writing default config if missing."""
    config_file = tmp_path / "settings.yaml"

    with (
        patch("app.config.CONFIG_FILE", config_file),
        patch("app.config.BASE_DIR", tmp_path),
    ):
        # 1. Missing file -> creates default
        cm = ConfigManager()
        assert config_file.exists()
        assert cm.app.webserver.port == 5000

        # 2. Existing file -> loads from disk
        with open(config_file, "w") as f:
            yaml.dump({"webserver": {"port": 9090}}, f)

        cm_loaded = ConfigManager()
        assert cm_loaded.app.webserver.port == 9090


def test_config_manager_path_resolution_and_security(tmp_path: Path) -> None:
    """Tests get_path, get_folder, and path traversal security guards."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    with patch("app.config.BASE_DIR", tmp_path):
        cm = ConfigManager()

        # Valid database path
        signals_path = cm.get_path("signals")
        assert signals_path.name == "signals.db"
        assert cm.get_db_path("signals") == str(signals_path)

        # Missing key handling
        missing_path = cm.get_path("unknown_db_key")
        assert "MISSING_CONFIG" in missing_path.name

        # Folder resolution
        orders_folder = cm.get_folder("orders")
        assert orders_folder.exists()

        # Path Traversal attack prevention
        cm.app.database.files["malicious"] = "../../../etc/passwd"
        with pytest.raises(ValueError, match="Insecure path detected"):
            cm.get_path("malicious")

        cm.app.database.folders["malicious_folder"] = "../../../etc/secret_dir"
        with pytest.raises(ValueError, match="Insecure folder path detected"):
            cm.get_folder("malicious_folder")


def test_config_manager_helper_paths(tmp_path: Path) -> None:
    """Tests helper methods get_log_path and get_strategy_path."""
    with patch("app.config.BASE_DIR", tmp_path):
        cm = ConfigManager()
        assert "croc-trader.log" in cm.get_log_path()
        assert cm.get_strategy_path().name == "croc-strategie.yaml"
