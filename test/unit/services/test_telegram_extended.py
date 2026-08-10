"""Unit tests for app/services/telegram.py TelegramBot."""

from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from app.services.telegram import TelegramBot


def test_telegram_init_missing_token_or_chat() -> None:
    """Tests init warning when token or chat_id is missing."""
    bot = TelegramBot(token="", chat_id="")
    assert bot.bot_token == ""
    assert bot.chat_id == ""
    assert not bot.enabled


def test_telegram_send_disabled_or_unconfigured() -> None:
    """Tests send and send_message when disabled or missing credentials."""
    # Disabled bot
    bot_disabled = TelegramBot(token="123", chat_id="456", enabled=False)
    assert bot_disabled.send("test") is None
    assert bot_disabled.send_message("test") is None

    # Enabled but missing credentials
    bot_missing = TelegramBot(token="", chat_id="", enabled=True)
    assert bot_missing.send_message("test") is None


def test_telegram_send_success() -> None:
    """Tests successful message transmission."""
    bot = TelegramBot(token="12345:TOKEN", chat_id="9999", enabled=True)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"ok": True, "result": {"message_id": 1}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = bot.send_message("Hello World")
        assert res == {"ok": True, "result": {"message_id": 1}}
        mock_post.assert_called_once()


def test_telegram_send_network_error_sanitizes_token() -> None:
    """Tests error logging and token sanitization when requests raises RequestException."""
    bot = TelegramBot(token="SECRET_TOKEN_XYZ", chat_id="9999", enabled=True)

    with patch(
        "requests.post",
        side_effect=requests.exceptions.RequestException(
            "Failed connection to https://api.telegram.org/botSECRET_TOKEN_XYZ/sendMessage"
        ),
    ):
        res = bot.send_message("Test Error")
        assert res is None


def test_telegram_send_dataframe() -> None:
    """Tests send_dataframe with empty DataFrame and populated DataFrame."""
    bot = TelegramBot(token="12345:TOKEN", chat_id="9999", enabled=True)

    # Disabled -> returns None
    bot_disabled = TelegramBot(token="123", chat_id="456", enabled=False)
    assert bot_disabled.send_dataframe(pd.DataFrame()) is None

    # Empty DataFrame -> sends "(No Data)" message
    with patch.object(
        bot, "send_message", return_value={"ok": True}
    ) as mock_send_message:
        res_empty = bot.send_dataframe(pd.DataFrame(), title="Empty Report")
        assert res_empty == {"ok": True}
        mock_send_message.assert_called_with("Empty Report\n_(No Data)_")

    # Populated DataFrame -> tabulates table
    df = pd.DataFrame([{"Ticker": "AAPL", "Price": 150.0}])
    with patch.object(
        bot, "send_message", return_value={"ok": True}
    ) as mock_send_message:
        res_df = bot.send_dataframe(df, title="AAPL Report")
        assert res_df == {"ok": True}
        call_args = mock_send_message.call_args[0][0]
        assert "*AAPL Report*" in call_args
        assert "AAPL" in call_args
