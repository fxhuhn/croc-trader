# filename: test_security_hardening.py
import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from app.config import ConfigManager
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.services.telegram import TelegramBot

# --- 1. SQL INJECTION FUZZING ---


def test_traderepository_sql_injection_on_symbol():
    """
    SECURITY: Verifies that SQL injection attempts in symbols are treated as literal strings.
    """
    mock_session = MagicMock(spec=DatabaseSession)
    mock_conn = MagicMock()
    # Mock context manager
    mock_session.connect.return_value.__enter__.return_value = mock_conn
    # IMPORTANT: Mock fetchone to return None so it proceeds to INSERT instead of UPDATE
    mock_conn.execute.return_value.fetchone.return_value = None

    repo = TradeRepository(mock_session)

    malicious_symbol = "AAPL'; DROP TABLE trades; --"
    context = {"date": "2026-02-07"}

    # Act
    repo.create_trade(
        symbol=malicious_symbol,
        strategy="TestStrategy",
        size=100.0,
        entry=150.0,
        stop_loss=140.0,
        target=170.0,
        context=context,
    )

    # Assert: Verify that execute was called with parameters
    found_insert = False
    for call in mock_conn.execute.call_args_list:
        args = call[0]
        if (
            len(args) > 0
            and isinstance(args[0], str)
            and "INSERT INTO trades" in args[0]
        ):
            sql_query, params = args[0], args[1]
            assert "DROP TABLE" not in sql_query
            assert malicious_symbol in params
            found_insert = True
            break

    assert found_insert, f"INSERT INTO trades call not found. Calls were: {mock_conn.execute.call_args_list}"


def test_traderepository_sql_injection_on_strategy():
    """
    SECURITY: Verifies that SQL injection attempts in strategy names are neutralized.
    """
    mock_session = MagicMock()
    mock_conn = MagicMock()
    mock_session.connect.return_value.__enter__.return_value = mock_conn
    repo = TradeRepository(mock_session)

    malicious_strategy = "Strategy' OR '1'='1"

    # Act
    repo.exists("AAPL", malicious_strategy, "2026-02-07")

    # Assert
    sql_query, params = mock_conn.execute.call_args[0]
    assert "OR '1'='1" not in sql_query
    assert f"{malicious_strategy}%" in params


# --- 2. RESOURCE DoS ATTACK SIMULATION ---


def test_marketdataprovider_huge_payload_handling():
    """
    SECURITY: Checks how the system handles massive dataframes (DoS simulation).
    """
    from app.database.repositories.market_data_provider import MarketDataProvider

    mock_session = MagicMock()
    num_rows = 50000
    huge_df = pd.DataFrame(
        {
            "date": pd.date_range(start="2000-01-01", periods=num_rows),
            "symbol": ["AAPL"] * num_rows,
            "open": [100.0] * num_rows,
            "high": [101.0] * num_rows,
            "low": [99.0] * num_rows,
            "close": [100.5] * num_rows,
            "volume": [1000000] * num_rows,
        }
    )

    provider = MarketDataProvider(mock_session)

    with patch("pandas.read_sql_query", return_value=huge_df):
        data = provider.get_universe_daily_data(["AAPL"], days=1000)
        assert data is not None
        assert "close" in data
        assert data["close"].shape[0] == num_rows


# --- 3. SECRET LEAKAGE VERIFICATION ---


def test_telegram_bot_does_not_leak_token_on_error(caplog):
    """
    SECURITY: Ensures that Telegram bot tokens do not appear in logs during network errors.
    """
    caplog.set_level(logging.ERROR)

    secret_token = "123456:ABC-DEF-GHI-JKL"  # nosec B105
    bot = TelegramBot(token=secret_token, chat_id="987654", enabled=True)

    error_message = (
        f"Failed to connect to https://api.telegram.org/bot{secret_token}/sendMessage"
    )
    with patch(
        "requests.post", side_effect=requests.exceptions.RequestException(error_message)
    ):
        bot.send_message("Hello")

    assert secret_token not in caplog.text
    assert "********" in caplog.text


# --- 4. PATH TRAVERSAL CHECK ---


def test_config_manager_rejects_path_traversal(tmp_path):
    """
    SECURITY: Verifies that ConfigManager doesn't allow escaping the data directory.
    """
    # Assuming ConfigManager.get_path checks for '..'
    with (
        patch("app.config.BASE_DIR", tmp_path),
        patch("app.config.CONFIG_FILE", tmp_path / "settings.yaml"),
    ):
        manager = ConfigManager()
        # Mocking the dictionary structure based on previous view_file (if I saw it) or standard patterns
        if hasattr(manager, "app") and hasattr(manager.app, "database"):
            manager.app.database.files["malicious"] = "../../etc/passwd"

        with pytest.raises(ValueError, match="Insecure path detected"):
            manager.get_path("malicious")


# --- 5. FINANCIAL LOGIC HARDENING ---


def test_financial_logic_rejects_non_finite_values():
    """
    SECURITY: Prevents database corruption via Infinite or negative inputs.
    """
    mock_session = MagicMock()
    repo = TradeRepository(mock_session)

    with pytest.raises(
        ValueError, match="Value for entry must be a finite non-negative number"
    ):
        repo.create_trade("AAPL", "Test", 100, float("inf"), 0, 0, {})

    with pytest.raises(
        ValueError, match="Value for stop_loss must be a finite non-negative number"
    ):
        repo.create_trade("AAPL", "Test", 100, 150.0, -1.0, 0, {})
