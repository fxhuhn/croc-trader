import logging
from typing import TypedDict

from ...database.repositories.market_data_provider import MarketDataProvider
from ...database.repositories.trade import TradeRepository
from ...database.repositories.signal import SignalRepository
from ...services.telegram import TelegramBot
from .protocols import StrategyProtocol

logger = logging.getLogger(__name__)


class ScreenerConfiguration(TypedDict, total=False):
    """Configuration for the screening process."""

    strategy_ranking: list[str]


class ScreenerEngine:
    """
    Orchestrates the execution of multiple trading strategies for screening.

    This engine follows the Open-Closed Principle by allowing strategies to be
    registered dynamically rather than being hardcoded in the constructor.
    """

    def __init__(
        self,
        trade_repository: TradeRepository,
        signal_repository: SignalRepository,
        data_provider: MarketDataProvider,
        strategies: list[StrategyProtocol] | None = None,
        configuration: ScreenerConfiguration | None = None,
        telegram_bot: TelegramBot | None = None,
    ) -> None:
        """
        Initializes the ScreenerEngine with required repositories and strategies.

        Args:
            trade_repository: Repository for trade-related database operations.
            signal_repository: Repository for signal-related database operations.
            data_provider: Provider for market historical data.
            strategies: Optional initial list of strategies to register.
            configuration: Optional configuration dictionary.
            telegram_bot: Optional bot for sending notifications.
        """
        self.trade_repository = trade_repository
        self.signal_repository = signal_repository
        self.data_provider = data_provider
        self.telegram = telegram_bot
        self.active_strategies: list[StrategyProtocol] = []
        self.configuration = configuration or {}

        if strategies:
            for strategy in strategies:
                self.register_strategy(strategy)

    def register_strategy(self, strategy: StrategyProtocol) -> None:
        """
        Registers a new strategy to the engine.

        Args:
            strategy: An object implementing the StrategyProtocol.
        """
        self.active_strategies.append(strategy)
        logger.debug("Strategy registered: %s", strategy.name)

    def run_all(
        self, days: int = 0, strategy_filter: str | None = None
    ) -> dict[str, int]:
        """
        Executes all registered strategies and returns the hit count per strategy.

        Args:
            days: Lookback period for screening (0 for latest data).
            strategy_filter: Optional name to run only a specific strategy.

        Returns:
            dict[str, int]: A mapping of strategy names to their respective signal hits.
        """
        results: dict[str, int] = {}
        global_analysis_date: str | None = None

        # Reset cache and sync date for fresh runs
        if days == 0:
            self.data_provider.clear_cache()
            global_analysis_date = self.data_provider.get_latest_date()

            if global_analysis_date:
                logger.info(
                    "[ScreenerEngine] Global Analysis Date detected: %s",
                    global_analysis_date,
                )
            else:
                logger.warning(
                    "[ScreenerEngine] Could not detect global date from market data."
                )

        for strategy in self.active_strategies:
            if strategy_filter and strategy.name != strategy_filter:
                continue

            try:
                hits = strategy.run(days=days, analysis_date=global_analysis_date)
                results[strategy.name] = hits
            except (ValueError, KeyError, RuntimeError) as error:
                logger.error("Error executing strategy %s: %s", strategy.name, error)
                results[strategy.name] = 0
            except Exception:
                logger.exception(
                    "Critical unexpected error in strategy %s", strategy.name
                )
                results[strategy.name] = 0

        return results

    def get_strategy(self, name: str) -> StrategyProtocol | None:
        """
        Finds a registered strategy by its name.

        Args:
            name: The name of the strategy to find.

        Returns:
            StrategyProtocol | None: The found strategy or None if not registered.
        """
        for strategy in self.active_strategies:
            if strategy.name == name:
                return strategy
        return None
