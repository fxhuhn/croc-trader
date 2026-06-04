import logging
from typing import Any

import pandas as pd
import requests
from tabulate import tabulate

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, token: str, chat_id: str, enabled: bool = False) -> None:
        self.bot_token = token
        self.chat_id = chat_id
        self.enabled = enabled

        if not self.bot_token or not self.chat_id:
            logger.warning(
                "Telegram Bot nicht vollständig konfiguriert (Token/ChatID fehlt)."
            )

    def send(self, text: str, parse_mode: str = "Markdown") -> dict[str, Any] | None:
        logger.debug("old Telegram pattern.")

        return self.send_message(text, parse_mode)

    def send_message(
        self, text: str, parse_mode: str = "Markdown"
    ) -> dict[str, Any] | None:
        """Sendet eine Textnachricht, falls der Bot aktiviert ist."""
        if not self.enabled:
            logger.debug("Telegram Send skipped (disabled).")
            return None

        if not self.bot_token or not self.chat_id:
            logger.error("Telegram Bot nicht konfiguriert.")
            return None

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }

        try:
            response = requests.post(url, json=payload, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            # Mask token in logs (Security Hardening)
            error_msg = str(e)
            if self.bot_token:
                error_msg = error_msg.replace(self.bot_token, "********")
            logger.error(f"Telegram Netzwerk-Fehler: {error_msg}")
            return None
        except Exception as e:
            error_msg = str(e)
            if self.bot_token:
                error_msg = error_msg.replace(self.bot_token, "********")
            logger.error(f"Telegram Unbekannter Fehler: {error_msg}")
            return None

    def send_dataframe(
        self, df: pd.DataFrame, title: str = ""
    ) -> dict[str, Any] | None:
        """Formatiert ein DataFrame als Code-Block Tabelle."""
        if not self.enabled:
            return None

        if df.empty:
            return self.send_message(f"{title}\n_(Keine Daten)_")

        # Tabulate für schöne ASCII-Tabellen
        table_str = tabulate(df, tablefmt="simple", showindex=False, headers="keys")

        message = f"*{title}*\n```\n{table_str}\n```"
        return self.send_message(message, parse_mode="Markdown")
