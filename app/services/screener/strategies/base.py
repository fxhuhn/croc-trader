import logging
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, final

import pandas as pd

from ....database.repositories.market_data_provider import MarketDataProvider
from ....mapping import mapper
from ....services.telegram import TelegramBot

logger = logging.getLogger(__name__)

T = TypeVar("T")


class BaseStrategy(ABC, Generic[T]):
    name: str = "Base"

    def __init__(
        self,
        data_provider: MarketDataProvider,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        """
        Basis-Klasse für Strategien.
        Entkoppelt von der Datenbank - Persistenz muss in der Subklasse definiert werden via DI.
        """
        self.data_provider = data_provider
        self.telegram_bot = telegram_bot

    @abstractmethod
    def run(self, days: int = 0) -> int:
        raise NotImplementedError("Subclasses must implement this method")

    @final
    def _get_exchange(self, symbol: str) -> str:
        return mapper.get_exchange(symbol, default="UNKNOWN")

    @final
    def _send_telegram_report(self, title: str, date: str, data: pd.DataFrame) -> None:
        if not self.telegram_bot or data.empty:
            return

        full_title = f"🔎 {title} ({date})"
        self.telegram_bot.send_dataframe(data, title=full_title)
