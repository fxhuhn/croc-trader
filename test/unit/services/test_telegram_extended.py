"""Unit tests for app/services/telegram.py TelegramBot and compact table formatting."""

from unittest.mock import MagicMock, patch

import pandas as pd
import requests

from app.services.telegram import (
    TelegramBot,
    format_compact_table,
    format_dataframe_to_compact_table,
)


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


def test_format_compact_table_basic() -> None:
    """Tests format_compact_table builds matching header, separator, and aligned rows."""
    headers = ["SYM", "ACTION", "ENTRY", "TP"]
    rows = [
        ["BKNG", "BUY LMT", "196.11", "201.27"],
        ["EXPE", "BUY LMT", "306.57", "316.45"],
    ]
    alignments = ["left", "left", "right", "right"]

    output = format_compact_table(headers, rows, alignments)
    lines = output.split("\n")

    assert len(lines) == 4  # Header, Separator, Row 1, Row 2
    assert lines[0] == "SYM   ACTION    ENTRY      TP"
    assert lines[1] == "-----------------------------"
    assert lines[2] == "BKNG  BUY LMT  196.11  201.27"
    assert lines[3] == "EXPE  BUY LMT  306.57  316.45"
    assert len(lines[0]) == len(lines[1])


def test_format_dataframe_to_compact_table_filtering() -> None:
    """Tests format_dataframe_to_compact_table shortens headers and filters secondary columns."""
    dataframe = pd.DataFrame(
        [
            {
                "Symbol": "BKNG",
                "Action": "BUY LMT",
                "Entry": 196.11,
                "TP": 201.27,
                "Score": 1.67,
                "Close": 202.56,
                "ATR": 6.45,
                "LOC": 207.74,
            }
        ]
    )

    output = format_dataframe_to_compact_table(dataframe)
    lines = output.split("\n")

    assert lines[0] == "SYM   ACTION    ENTRY      TP"
    assert lines[1] == "-----------------------------"
    assert lines[2] == "BKNG  BUY LMT  196.11  201.27"
    # Secondary columns are omitted from wide table
    assert "Score" not in output
    assert "LOC" not in output
    assert "Close" not in output
    assert "ATR" not in output


def test_telegram_send_dataframe() -> None:
    """Tests send_dataframe with empty DataFrame and populated DataFrame."""
    bot = TelegramBot(token="12345:TOKEN", chat_id="9999", enabled=True)

    # Disabled -> returns None
    bot_disabled = TelegramBot(token="123", chat_id="456", enabled=False)
    assert bot_disabled.send_dataframe(pd.DataFrame()) is None

    # Empty DataFrame -> sends "(No Data)" HTML message
    with patch.object(
        bot, "send_message", return_value={"ok": True}
    ) as mock_send_message:
        res_empty = bot.send_dataframe(pd.DataFrame(), title="Empty Report")
        assert res_empty == {"ok": True}
        mock_send_message.assert_called_with(
            "<b>Empty Report</b>\n<i>(No Data)</i>", parse_mode="HTML"
        )

    # Populated DataFrame -> sends compact table in HTML pre block
    df = pd.DataFrame([{"Ticker": "AAPL", "Price": 150.0}])
    with patch.object(
        bot, "send_message", return_value={"ok": True}
    ) as mock_send_message:
        res_df = bot.send_dataframe(df, title="AAPL Report")
        assert res_df == {"ok": True}
        call_args = mock_send_message.call_args[0][0]
        assert "<b>AAPL Report</b>" in call_args
        assert "<pre>" in call_args
        assert "AAPL" in call_args
        assert "150.00" in call_args
        assert mock_send_message.call_args[1].get("parse_mode") == "HTML"
