import logging
from typing import TypedDict

import pandas as pd

from ...const import Strategies
from ...database.repositories.market_data_provider import MarketDataProvider
from ...database.repositories.signal import SignalRepository
from ...database.repositories.trade import TradeRepository
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
        self.telegram_bot = telegram_bot
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

        latest_date_str = self.data_provider.get_latest_date()
        if latest_date_str:
            if days == 0:
                self.data_provider.clear_cache()
                global_analysis_date = latest_date_str
            else:
                start_search_date = (
                    pd.Timestamp(latest_date_str) - pd.Timedelta(days=days * 5)
                ).strftime("%Y-%m-%d")
                available_trading_dates = self.data_provider.get_available_dates(
                    start_date=start_search_date,
                    end_date=latest_date_str,
                )
                if len(available_trading_dates) > days:
                    global_analysis_date = available_trading_dates[-1 - days].strftime(
                        "%Y-%m-%d"
                    )
                else:
                    global_analysis_date = (
                        pd.Timestamp(latest_date_str) - pd.Timedelta(days=days)
                    ).strftime("%Y-%m-%d")

            logger.info(
                "[ScreenerEngine] Global Analysis Date resolved: %s (days=%d)",
                global_analysis_date,
                days,
            )
        else:
            logger.warning(
                "[ScreenerEngine] Could not detect global date from market data."
            )

        for strategy in self.active_strategies:
            strat_name = getattr(strategy, "name", "")
            strat_ident = getattr(strategy, "STRATEGY_IDENTIFIER", "")
            strat_key = (
                getattr(strat_name, "value", str(strat_name)).lower().replace("-", "_")
            )
            ident_key = (
                getattr(strat_ident, "value", str(strat_ident))
                .lower()
                .replace("-", "_")
            )

            if strategy_filter:
                filter_key = (
                    getattr(strategy_filter, "value", str(strategy_filter))
                    .lower()
                    .replace("-", "_")
                )
                if filter_key not in (strat_key, ident_key):
                    continue

            try:
                hits = strategy.run(days=days, analysis_date=global_analysis_date)
                results[str(strat_name)] = hits
            except (ValueError, KeyError, RuntimeError) as error:
                logger.error("Error executing strategy %s: %s", strat_name, error)
                results[str(strat_name)] = 0
            except Exception:
                logger.exception("Critical unexpected error in strategy %s", strat_name)
                results[str(strat_name)] = 0

        return results

    def get_strategy(self, name: str | Strategies) -> StrategyProtocol | None:
        """Finds a registered strategy by its name or canonical enum identifier.

        Args:
            name: The name or enum of the strategy to find.

        Returns:
            StrategyProtocol | None: The found strategy or None if not registered.
        """
        search_key = getattr(name, "value", str(name)).lower().replace("-", "_")
        for strategy in self.active_strategies:
            strat_name = getattr(strategy, "name", "")
            strat_ident = getattr(strategy, "STRATEGY_IDENTIFIER", "")
            strat_key = (
                getattr(strat_name, "value", str(strat_name)).lower().replace("-", "_")
            )
            ident_key = (
                getattr(strat_ident, "value", str(strat_ident))
                .lower()
                .replace("-", "_")
            )
            if search_key in (strat_key, ident_key):
                return strategy
        return None
