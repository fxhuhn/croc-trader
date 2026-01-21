import logging
from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar, final

from ....mapping import mapper
from ....services.database import SignalDatabase
from ....services.telegram import TelegramBot

logger = logging.getLogger(__name__)

# T ist ein Platzhalter für den Ergebnistyp (z.B. DipBuyerResult oder dict)
T = TypeVar("T")


class BaseStrategy(ABC, Generic[T]):
    name: str = "Base"

    def __init__(
        self, signals_db: SignalDatabase, telegram_bot: TelegramBot | None = None
    ) -> None:
        self.signals_db = signals_db
        # Standardisierung des Namens auf telegram_bot
        self.telegram_bot = telegram_bot

    @abstractmethod
    def run(self, days: int = 0) -> int:
        """Muss von der Strategie implementiert werden."""
        raise NotImplementedError

    @final
    def _get_exchange(self, symbol: str) -> str:
        return mapper.get_exchange(symbol, default="UNKNOWN")

    @final
    def _send_telegram_report(
        self, title: str, date: str, results: list[dict[str, Any]]
    ) -> None:
        """
        Sendet einen Report. Akzeptiert vorerst dicts für Abwärtskompatibilität,
        sollte aber perspektivisch typisierte Result-Objekte nutzen.
        """
        if not self.telegram_bot or not results:
            return

        msg_lines = [f"🔎 **{title}** ({date})"]

        # Top 10 des neuesten Tages
        # Wir nutzen enumerate für saubere Indizierung
        for i, r in enumerate(results[:10]):
            score_info = r.get("score") or r.get("setup_score") or r.get("rank") or "-"
            signal_info = r.get("signal", "Signal")
            price = r.get("close", 0.0)
            symbol = r.get("symbol", "N/A")

            line = f"{i + 1}. {symbol} | {price} | Score: {score_info}"
            msg_lines.append(line)

        # Footer Logik
        if len(results) > 10:
            msg_lines.append(f"\n... und {len(results) - 10} weitere Treffer.")

        # Hier lag der Fehler: Methode heißt jetzt send_message
        self.telegram_bot.send_message("\n".join(msg_lines))
