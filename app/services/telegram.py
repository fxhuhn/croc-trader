import logging
from typing import Optional

import pandas as pd
import requests
from tabulate import tabulate

logger = logging.getLogger(__name__)


class TelegramBot:
    def __init__(self, token: str, chat_id: str, enabled: bool = False):
        self.bot_token = token
        self.chat_id = chat_id
        self.enabled = enabled

        if not self.bot_token or not self.chat_id:
            logger.warning(
                "Telegram Bot nicht vollständig konfiguriert (Token oder ChatID fehlt)."
            )

    def send(self, text: str, parse_mode: str = "Markdown") -> Optional[dict]:
        """Sendet eine einfache Textnachricht."""
        if not self.bot_token or not self.chat_id:
            logger.error("Telegram Bot nicht configured (Token oder ChatID fehlt).")
            return None

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=10.0)
            r.raise_for_status()

            if r.status_code != 200:
                logger.error(f"Telegram Fehler {r.status_code}: {r.text}")

            return r.json()
        except Exception as e:
            logger.error(f"Telegram Sende-Fehler: {e}")
            return None

    def send_dataframe(self, df: pd.DataFrame, title: str = "") -> Optional[dict]:
        """Formatiert ein DataFrame als Code-Block Tabelle."""
        if df.empty:
            return self.send_message(f"{title}\n_(Keine Daten)_")

        # Tabelle erstellen
        table_str = tabulate(df, tablefmt="simple", showindex=False, headers="keys")

        # Nachricht zusammenbauen
        message = f"*{title}*\n```\n\n{table_str}\n```"
        return self.send(message, parse_mode="Markdown")
