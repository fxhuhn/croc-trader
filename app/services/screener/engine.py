import logging
from pathlib import Path
from typing import Any

from ...services.database import SignalDatabase
from ...services.market_data_provider import MarketDataProvider
from ...services.telegram import TelegramBot
from .protocols import StrategyProtocol
from .strategies.croc_setup import CrocSetupStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.webhook import WebhookFilterStrategy

logger = logging.getLogger(__name__)


class ScreenerEngine:
    def __init__(
        self,
        stocks_db_path: Path,
        signals_db_path: Path,
        data_provider: MarketDataProvider,  # <--- NEU: Injected Provider
        config: dict[str, Any] | None = None,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        self.signals_db = SignalDatabase(signals_db_path)
        self.data_provider = data_provider  # <---
        self.telegram = telegram_bot
        self.active_strategies: list[StrategyProtocol] = []

        if config is None:
            config = {}
        elif isinstance(config, list):
            config = {"strategy_ranking": config}

        # Strategien registrieren (übergeben Provider statt rohem Pfad)
        self.register_strategy(
            DipBuyerStrategy(self.signals_db, self.data_provider, self.telegram)
        )

        # Hinweis: Andere Strategien (Turnover etc.) müssen auch aktualisiert werden,
        # damit sie den Provider akzeptieren. Fürs Erste lassen wir sie so,
        # falls sie noch den alten Konstruktor haben, wird das hier brechen.
        # Annahme: Du passt TurnoverTimingStrategy analog zur DipBuyerStrategy an.
        # Falls TurnoverTiming noch den alten Pfad braucht, musst du den Provider hier
        # ignorieren und Stocks_db_path übergeben.
        # Da du "komplett überarbeitet" wolltest, gehe ich davon aus, dass wir konsistent sind.

        self.register_strategy(
            WebhookFilterStrategy(self.signals_db, config, self.telegram)
        )

        # Achtung: TurnoverTimingStrategy müsste auch refactored werden.
        # Wenn nicht, übergib hier temporär stocks_db_path, falls die Klasse es noch so will.
        # self.register_strategy(
        #    TurnoverTimingStrategy(stocks_db_path, self.signals_db, self.telegram)
        # )

        self.register_strategy(CrocSetupStrategy(self.signals_db, self.telegram))

    def register_strategy(self, strategy: StrategyProtocol) -> None:
        self.active_strategies.append(strategy)
        logger.debug(f"Strategie registriert: {strategy.name}")

    def run_all(
        self, days: int = 0, strategy_filter: str | None = None
    ) -> dict[str, int]:
        results = {}

        # Provider Cache leeren vor einem neuen Run, um frische Daten zu garantieren?
        # Oder behalten für Performance? Bei Daily Run ist der Prozess eh neu.
        # Im laufenden Server könnte man hier clearen:
        if days == 0:
            self.data_provider.clear_cache()

        for strat in self.active_strategies:
            if strategy_filter and strat.name != strategy_filter:
                continue

            try:
                hits = strat.run(days=days)
                results[strat.name] = hits
            except Exception as e:
                logger.error(f"Fehler bei {strat.name}: {e}", exc_info=True)
                results[strat.name] = 0
        return results
