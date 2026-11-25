import logging
import sqlite3
from collections.abc import Sequence
from typing import TypedDict

import pandas as pd

from ...const import Strategies
from ...database.repositories.market_data_provider import (
    MarketDataLoadError,
    MarketDataProvider,
)
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
        strategies: Sequence[StrategyProtocol] | None = None,
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

    @staticmethod
    def _matches_strategy_filter(
        strategy: StrategyProtocol, strategy_filter: str | None
    ) -> bool:
        """Checks whether a strategy matches the optional filter key."""
        if not strategy_filter:
            return True
        strategy_name = getattr(strategy, "name", "")
        strategy_identifier = getattr(strategy, "STRATEGY_IDENTIFIER", "")
        name_key = (
            getattr(strategy_name, "value", str(strategy_name))
            .lower()
            .replace("-", "_")
        )
        identifier_key = (
            getattr(strategy_identifier, "value", str(strategy_identifier))
            .lower()
            .replace("-", "_")
        )
        filter_key = (
            getattr(strategy_filter, "value", str(strategy_filter))
            .lower()
            .replace("-", "_")
        )
        return filter_key in (name_key, identifier_key)

    def _resolve_analysis_date(
        self, days: int, expected_min_date: str | None
    ) -> str | None:
        """Resolves global analysis date and validates the optional market-data cutoff."""
        latest_date_str = self.data_provider.get_latest_date()
        if days == 0 and expected_min_date is not None:
            if not latest_date_str or latest_date_str < expected_min_date:
                raise MarketDataLoadError(
                    f"Market data cutoff not met: latest={latest_date_str}, "
                    f"expected>={expected_min_date}"
                )

        if not latest_date_str:
            logger.warning(
                "[ScreenerEngine] Could not detect global date from market data."
            )
            return None

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
        return global_analysis_date

    def run_all(
        self,
        days: int = 0,
        strategy_filter: str | None = None,
        *,
        expected_min_date: str | None = None,
    ) -> dict[str, int]:
        """Executes registered strategies and returns the hit count per strategy.

        Args:
            days: Lookback period for screening (0 for latest data).
            strategy_filter: Optional name to run only a specific strategy.
            expected_min_date: Optional minimum required market-data date (YYYY-MM-DD).

        Returns:
            dict[str, int]: A mapping of strategy names to their respective signal hits.
        """
        results: dict[str, int] = {}
        global_analysis_date = self._resolve_analysis_date(days, expected_min_date)

        for strategy in self.active_strategies:
            if not self._matches_strategy_filter(strategy, strategy_filter):
                continue

            strat_name = getattr(strategy, "name", "")
            try:
                hits = strategy.run(days=days, analysis_date=global_analysis_date)
                results[str(strat_name)] = hits
            except (MarketDataLoadError, sqlite3.Error, MemoryError):
                raise
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
