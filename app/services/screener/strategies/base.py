import logging
from abc import ABC, abstractmethod
from typing import Any, final

from ....mapping import mapper
from ....services.database import SignalDatabase
from ....services.telegram import TelegramBot

logger = logging.getLogger(__name__)


class BaseStrategy(ABC):
    name: str = "Base"

    def __init__(
        self, signals_db: SignalDatabase, telegram_bot: TelegramBot | None = None
    ) -> None:
        self.signals_db = signals_db
        self.telegram = telegram_bot

    @abstractmethod
    def run(self, days: int = 0) -> int:
        raise NotImplementedError

    @final
    def _get_exchange(self, symbol: str) -> str:
        return mapper.get_exchange(symbol, default="UNKNOWN")

    @final
    def _send_telegram_report(
        self, title: str, date: str, results: list[dict[str, Any]]
    ) -> None:
        if not self.telegram or not results:
            return

        msg_lines = [f"🔎 **{title}** ({date})"]

        # Top 10 des neuesten Tages
        for r in results[:10]:
            score_info = r.get("score") or r.get("rank") or "-"
            signal_info = r.get("signal", "Unknown")
            price = r.get("close", 0.0)
            msg_lines.append(
                f"• [Score: {score_info}] {r['symbol']} ({signal_info}): {price}"
            )

        if len(results) > 10:
            unique_days = {r["date"] for r in results}
            if len(unique_days) > 1:
                msg_lines.append(
                    f"\n... und weitere Treffer aus insgesamt {len(unique_days)} Tagen."
                )
            else:
                msg_lines.append(f"... und {len(results) - 10} weitere.")

        self.telegram.send("\n".join(msg_lines))
