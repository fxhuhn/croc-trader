import logging
from abc import ABC, abstractmethod

import pandas as pd

# NEU: Repository statt SignalDatabase
from ....database.repositories.trade import TradeRepository
from ....types import Order, TradeParams

logger = logging.getLogger(__name__)


class BaseTradeStrategy(ABC):
    """
    Interface für alle Handelsstrategien.
    Nutzt jetzt konsequent das TradeRepository.
    """

    @abstractmethod
    def check_entry(
        self,
        trade: dict,
        candle: pd.Series,
        df_history: pd.DataFrame,
        repo: TradeRepository,  # <--- Geändert
    ) -> str | None:
        """
        Prüft für CREATED Trades, ob der Entry gefüllt wurde.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def manage_active_trade(
        self, 
        trade: dict, 
        df_history: pd.DataFrame, 
        repo: TradeRepository # <--- Geändert
    ) -> str | None:
        """
        Verwaltet ACTIVE Trades (Exit-Prüfung).
        """
        raise NotImplementedError("Subclasses must implement this method")


    @abstractmethod
    def generate_orders(
        self, 
        trade: dict, 
        df_history: pd.DataFrame, 
        budget: float, 
        repo: TradeRepository # <--- Geändert
    ) -> Order | None:
        """
        Erstellt Orders für den nächsten Tag.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_current_params(
        self, 
        trade: dict, 
        df_history: pd.DataFrame, 
        repo: TradeRepository # <--- Geändert
    ) -> TradeParams | None:
        """
        Berechnet Parameter für Logging/Anzeige.
        """
        raise NotImplementedError("Subclasses must implement this method")
