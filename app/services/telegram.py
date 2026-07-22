import logging

import pandas as pd
import requests
from tabulate import tabulate

logger = logging.getLogger(__name__)

# Default network timeout in seconds
DEFAULT_TELEGRAM_TIMEOUT_SECONDS: float = 10.0


class TelegramBot:
    """Dispatches operational alerts and daily market data tables to Telegram.

    Acts as an external integration gateway within the imperative shell layer,
    ensuring sensitive API tokens are sanitized prior to log recording.

    Attributes:
        bot_token: Secret Telegram API authentication token.
        chat_id: Target Telegram channel or user chat identifier.
        enabled: Controls whether outbound notifications are active.
    """

    bot_token: str
    chat_id: str
    enabled: bool

    def __init__(self, token: str, chat_id: str, enabled: bool = False) -> None:
        """Initializes the Telegram bot client context.

        Args:
            token: Telegram bot token.
            chat_id: Destination chat identifier.
            enabled: Flags if sending messages is permitted.
        """
        self.bot_token = token
        self.chat_id = chat_id
        self.enabled = enabled

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram bot not fully configured (token/chat_id missing).")

    def send(self, text: str, parse_mode: str = "Markdown") -> dict[str, object] | None:
        """Legacy compatibility wrapper for message dispatching.

        Args:
            text: Message body to transmit.
            parse_mode: Formatting engine designation.

        Returns:
            dict[str, object] | None: Parsed JSON response payload from Telegram.
        """
        logger.debug("Executing legacy Telegram send wrapper.")
        return self.send_message(text, parse_mode)

    def send_message(
        self, text: str, parse_mode: str = "Markdown"
    ) -> dict[str, object] | None:
        """Transmits a formatted text message to the configured Telegram chat.

        Args:
            text: Raw message content.
            parse_mode: Telegram markup mode ('Markdown' or 'HTML').

        Returns:
            dict[str, object] | None: API response payload or None on failure/skip.
        """
        if not self.enabled:
            logger.debug("Telegram Send skipped (disabled).")
            return None

        if not self.bot_token or not self.chat_id:
            logger.error("Telegram bot not configured.")
            return None

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload: dict[str, object] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(
                url, json=payload, timeout=DEFAULT_TELEGRAM_TIMEOUT_SECONDS
            )
            response.raise_for_status()
            json_payload: dict[str, object] = response.json()
            return json_payload
        except requests.exceptions.RequestException as network_error:
            error_message = str(network_error)
            if self.bot_token:
                error_message = error_message.replace(self.bot_token, "********")
            logger.error("Telegram network error: %s", error_message)
            return None

    def send_dataframe(
        self, dataframe: pd.DataFrame, title: str = ""
    ) -> dict[str, object] | None:
        """Formats a pandas DataFrame into a monospaced ASCII code block table.

        Args:
            dataframe: Tabular data to be displayed.
            title: Header title preceding the table.

        Returns:
            dict[str, object] | None: API response payload or None if skipped.
        """
        if not self.enabled:
            return None

        if dataframe.empty:
            return self.send_message(f"{title}\n_(No Data)_")

        table_string = tabulate(
            dataframe, tablefmt="simple", showindex=False, headers="keys"
        )
        message = f"*{title}*\n```\n{table_string}\n```"
        return self.send_message(message, parse_mode="Markdown")
