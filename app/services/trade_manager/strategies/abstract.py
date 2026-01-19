import logging
from abc import ABC, abstractmethod
from typing import Optional

import pandas as pd

from ...database import SignalDatabase
from ..types import Order

logger = logging.getLogger(__name__)


class BaseTradeStrategy(ABC):
    """
    Interface für alle Handelsstrategien im TradeManager.
    """

    @abstractmethod
    def check_entry(
        self, trade: dict, candle: pd.Series, db: SignalDatabase
    ) -> Optional[str]:
        """
        Prüft für CREATED Trades, ob der Entry gefüllt wurde.
        Gibt Log-String zurück oder None.
        """
        pass

    @abstractmethod
    def manage_active_trade(
        self, trade: dict, df_history: pd.DataFrame, db: SignalDatabase
    ) -> Optional[str]:
        """
        Verwaltet ACTIVE Trades (Exit-Prüfung, TimeStop).
        Gibt Log-String zurück oder None.
        """
        pass

    @abstractmethod
    def generate_orders(
        self, trade: dict, df_history: pd.DataFrame, budget: float, db: SignalDatabase
    ) -> Optional[Order]:
        """
        Erstellt die YAML-Order Struktur für den nächsten Tag.
        Sowohl für Entry (CREATED) als auch Management (ACTIVE).
        """
        pass

    def _close_trade_in_db(
        self,
        db: SignalDatabase,
        trade_id: int,
        reason: str,
        price: float,
        exit_date: Optional[str] = None,
    ):
        """Hilfsmethode zum sauberen Schließen in der DB."""
        try:
            with db._get_conn() as conn:
                conn.execute(
                    """
                    UPDATE active_trades
                    SET status = 'CLOSED',
                        exit_reason = ?,
                        exit_price = ?,
                        exit_date = ?,
                        closed_at = CURRENT_TIMESTAMP
                    WHERE id = ?
                    """,
                    (reason, price, exit_date, trade_id),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"DB Error beim Schließen von Trade {trade_id}: {e}")
