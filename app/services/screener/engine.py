import logging
from pathlib import Path
from typing import Any

from ...services.database import SignalDatabase
from ...services.telegram import TelegramBot
from .protocols import StrategyProtocol
from .strategies.croc_setup import CrocSetupStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.turnover import TurnoverTimingStrategy
from .strategies.webhook import WebhookFilterStrategy

logger = logging.getLogger(__name__)


class ScreenerEngine:
    def __init__(
        self,
        stocks_db_path: Path,
        signals_db_path: Path,
        config: dict[str, Any] | None = None,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        self.signals_db = SignalDatabase(signals_db_path)
        self.telegram = telegram_bot
        self.active_strategies: list[StrategyProtocol] = []

        # Config Normalization
        if config is None:
            config = {}
        elif isinstance(config, list):
            # Fallback für alte Listen-Configs
            config = {"strategy_ranking": config}

        # Strategien registrieren
        # Hinweis: DipBuyerStrategy Signatur muss mit deiner aktuellen Version übereinstimmen
        self.register_strategy(
            DipBuyerStrategy(stocks_db_path, self.signals_db, self.telegram)
        )
        self.register_strategy(
            WebhookFilterStrategy(self.signals_db, config, self.telegram)
        )
        self.register_strategy(
            TurnoverTimingStrategy(stocks_db_path, self.signals_db, self.telegram)
        )
        self.register_strategy(CrocSetupStrategy(self.signals_db, self.telegram))

    def register_strategy(self, strategy: StrategyProtocol) -> None:
        self.active_strategies.append(strategy)
        logger.debug(f"Strategie registriert: {strategy.name}")

    def run_all(
        self, days: int = 0, strategy_filter: str | None = None
    ) -> dict[str, int]:
        """
        Führt alle registrierten Strategien aus.

        :param days: Anzahl der Tage für Backfill (0 = Heute/Live)
        :param strategy_filter: Optionaler Name einer spezifischen Strategie, die ausgeführt werden soll.
        """
        results = {}
        for strat in self.active_strategies:
            # Wenn Filter aktiv ist und Name nicht übereinstimmt -> Skip
            if strategy_filter and strat.name != strategy_filter:
                continue

            try:
                hits = strat.run(days=days)
                results[strat.name] = hits
            except Exception as e:
                logger.error(f"Fehler bei {strat.name}: {e}", exc_info=True)
                results[strat.name] = 0

        return results
