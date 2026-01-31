import logging
from pathlib import Path
from typing import Any

from ...database.repositories.market_data_provider import MarketDataProvider
from ...database.repositories.trade import TradeRepository
from ...database.repositories.signal import SignalRepository
from ...services.telegram import TelegramBot
from .protocols import StrategyProtocol

# Strategien
from .strategies.croc_setup import CrocSetupStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.turnover_timing import TurnoverTimingStrategy

logger = logging.getLogger(__name__)

class ScreenerEngine:
    def __init__(
        self,
        # Pfade brauchen wir hier nicht mehr, da Repos schon fertig sind
        trade_repo: TradeRepository,     # <--- Dependency Injection
        signal_repo: SignalRepository,   # <--- Dependency Injection
        data_provider: MarketDataProvider,
        config: dict[str, Any] | None = None,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        self.trade_repo = trade_repo
        self.signal_repo = signal_repo
        self.data_provider = data_provider
        self.telegram = telegram_bot
        self.active_strategies: list[StrategyProtocol] = []

        if config is None:
            config = {}
        elif isinstance(config, list):
            config = {"strategy_ranking": config}

        # --- 1. DipBuyer Strategy ---
        self.register_strategy(
            DipBuyerStrategy(
                self.trade_repo, 
                self.data_provider, 
                self.telegram
            )
        )

        # --- 2. Turnover Timing Strategy ---
        self.register_strategy(
            TurnoverTimingStrategy(
                trade_repo=self.trade_repo,
                data_provider=self.data_provider,
                telegram_bot=self.telegram
            )
        )

        # --- 3. Croc Setup Strategy ---
        self.register_strategy(
            CrocSetupStrategy(
                trade_repo=self.trade_repo,
                data_provider=self.data_provider,
                signal_repo=self.signal_repo,
                telegram_bot=self.telegram
            )
        )

    def register_strategy(self, strategy: StrategyProtocol) -> None:
        self.active_strategies.append(strategy)
        logger.debug(f"Strategie registriert: {strategy.name}")

    def run_all(
        self, days: int = 0, strategy_filter: str | None = None
    ) -> dict[str, int]:
        results = {}

        # Cache leeren für frischen Run
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
                results[strat.name] = 0
        return results

    def get_strategy(self, name: str) -> StrategyProtocol | None:
        """Findet eine registrierte Strategie anhand des Namens."""
        for s in self.active_strategies:
            if s.name == name:
                return s
        return None