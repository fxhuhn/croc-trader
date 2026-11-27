import datetime
import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import cast

import pandas as pd

from ...config import settings
from ...const import STRATEGY_ALIASES, Strategies
from ...database.repositories.market import MarketRepository
from ...database.repositories.market_data_provider import MarketDataProvider
from ...database.repositories.trade import TradeRepository
from ...database.session import DatabaseSession
from ...models import Order
from ...services.market.updater import MarketDataUpdater
from ...services.screener.strategies.wounded_bull_sprint import (
    DEFAULT_HISTORY_FETCH_DAYS,
    DEFAULT_MAX_POSITIONS,
    load_wbs_qualifying_setups,
    normalize_trading_date,
    wbs_created_trade_sort_key,
)
from ...services.telegram import TelegramBot
from ...tools.market_holidays import MarketHolidayChecker
from ...tools.symbol_filter import SymbolFilter
from ...tools.symbol_lists import ExchangeSymbol
from ...tools.trading_calendar import get_last_completed_trading_day
from ...types import TradeData, TradeStatus
from .order_export import write_csv_orders_file
from .strategies.abstract import BaseTradeStrategy
from .strategies.bounce_bandit import BounceBanditTradeStrategy
from .strategies.bridge_scout import BridgeScoutTradeStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.hold_target import HoldTargetStrategy
from .strategies.ndx_momentum import NDXMomentumTradeStrategy
from .strategies.tgim import TGIMTradeStrategy
from .strategies.turnover_timing import TurnoverTimingStrategy
from .strategies.two_percent_strategy import TwoPercentStrategy
from .strategies.wounded_bull_sprint import WoundedBullSprintTradeStrategy
from .types import TradeTransition

logger = logging.getLogger(__name__)

_HARDCODED_HISTORY_FALLBACK_DATE = "2024-01-01"

# Strategies that enforce a single position per symbol.
# For these, a CREATED entry order is suppressed when an ACTIVE position
# already exists for the same symbol.
_SINGLE_POSITION_STRATEGIES: frozenset[Strategies] = frozenset(
    {
        Strategies.NDXMomentum,
        Strategies.TurnOverTiming,
        Strategies.TurnOverTiming_10,
        Strategies.TurnOverTiming_05,
        Strategies.TwoPercent,
        Strategies.HoldTarget,
        Strategies.SplitTarget,
        Strategies.DipBuyer,
        Strategies.TGIM,
        Strategies.BridgeScout,
        Strategies.BounceBandit,
        Strategies.WoundedBullSprint,
    }
)


class TradeManager:
    """
    Manages the lifecycle of trades, including entry, exit, and order generation.

    Resolves legacy strategy names to strict Enum constants and orchestrates
    the daily EOD batch processing cycle. Fails closed on database errors.
    """

    def __init__(
        self,
        db_path: Path,
        stocks_db_path: Path,
        telegram_bot: TelegramBot | None = None,
        ibkr_account_id: str | None = None,
    ) -> None:
        """Initializes repositories and strategy registry.

        Args:
            db_path: Path to the signals database.
            stocks_db_path: Path to the market data (stocks) database.
            telegram_bot: Optional Telegram notification client.
            ibkr_account_id: IBKR account identifier for CSV order export.
                Falls back to settings if not provided.
        """
        self.db_path = db_path
        self.stocks_db_path = stocks_db_path
        self.telegram = telegram_bot
        self._ibkr_account_id = ibkr_account_id or os.environ.get(
            "IBKR_ACCOUNT_ID", "YOUR_IBKR_ACCOUNT"
        )
        if self._ibkr_account_id == "YOUR_IBKR_ACCOUNT":
            logger.warning(
                "IBKR_ACCOUNT_ID is not configured in the environment. "
                "Using default placeholder: 'YOUR_IBKR_ACCOUNT'"
            )
        else:
            logger.info("TradeManager initialized: IBKR Account found.")

        self.stocks_session = DatabaseSession(str(stocks_db_path))
        self.market_repository = MarketRepository(self.stocks_session)
        self.market_data_provider = MarketDataProvider(self.stocks_session)
        self.exchange_symbols = ExchangeSymbol()
        self._wbs_run_cache: dict[datetime.date, set[str]] = {}

        self.signals_session = DatabaseSession(str(db_path))
        self.trade_repository = TradeRepository(self.signals_session)
        self.holiday_checker = MarketHolidayChecker()

        self.strategies: dict[Strategies, BaseTradeStrategy] = {
            Strategies.DipBuyer: DipBuyerStrategy(),
            Strategies.TurnOverTiming: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_10: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_05: TurnoverTimingStrategy(),
            Strategies.HoldTarget: HoldTargetStrategy(),
            Strategies.SplitTarget: HoldTargetStrategy(),  # Fallback for old trades
            Strategies.TwoPercent: TwoPercentStrategy(),
            Strategies.NDXMomentum: NDXMomentumTradeStrategy(),
            Strategies.TGIM: TGIMTradeStrategy(),
            Strategies.BridgeScout: BridgeScoutTradeStrategy(),
            Strategies.BounceBandit: BounceBanditTradeStrategy(),
            Strategies.WoundedBullSprint: WoundedBullSprintTradeStrategy(),
        }

        logger.info(
            "TradeManager initialized. Registered strategies: %s",
            list(self.strategies.keys()),
        )

    def _attempt_targeted_market_update(self, symbol: str) -> None:
        """Attempts a targeted market data update for a single active trade symbol."""
        try:
            updater = MarketDataUpdater(self.stocks_session, self.signals_session)
            updater.run_update(specific_symbols=[symbol], reconcile_eod=True)
        except Exception as error:
            logger.warning(
                "Failed targeted market data update for %s: %s", symbol, error
            )

    def _load_trade_history(
        self,
        symbol: str,
        trade: dict[str, object],
        lookback_days: int = 60,
    ) -> pd.DataFrame:
        """Loads price history for a trade with pre-entry lookback buffer."""
        actual_lookback = _resolve_trade_lookback_days(
            str(trade.get("strategy", "")), lookback_days=lookback_days
        )
        start_date = _resolve_history_start_date(trade, lookback_days=actual_lookback)
        return self.market_repository.get_symbol_history_raw(
            symbol, start_date=start_date
        )

    def _get_wbs_qualifying_setups(
        self,
        eval_date: datetime.date,
        run_cache: dict[datetime.date, set[str]],
    ) -> set[str]:
        """Computes and caches full-universe WBS qualifying setups for eval_date."""
        return load_wbs_qualifying_setups(
            eval_date,
            self.market_data_provider,
            self.exchange_symbols.sp_100,
            self.holiday_checker,
            run_cache,
        )

    def _sort_created_trades(
        self, created_trades: list[dict[str, object]]
    ) -> list[dict[str, object]]:
        """Orders WBS CREATED trades by setup_score descending while keeping others intact."""
        wbs_trades: list[dict[str, object]] = []
        other_trades: list[dict[str, object]] = []
        for trade in created_trades:
            if (
                self._resolve_strategy_name(str(trade.get("strategy", "")))
                == Strategies.WoundedBullSprint
            ):
                wbs_trades.append(trade)
            else:
                other_trades.append(trade)

        wbs_trades.sort(key=wbs_created_trade_sort_key)
        return other_trades + wbs_trades

    def _get_daily_updates_for_strategy(
        self,
        strategy: BaseTradeStrategy,
        trade: dict[str, object],
        history_dataframe: pd.DataFrame,
        wbs_setups_cache: dict[datetime.date, set[str]],
    ) -> dict[str, object]:
        """Dispatches daily updates with full-universe WBS setups only for WBS strategy."""
        if isinstance(strategy, WoundedBullSprintTradeStrategy):
            _, current_date = _resolve_wbs_eval_dates(history_dataframe)
            qualifying_setups = (
                self._get_wbs_qualifying_setups(current_date, wbs_setups_cache)
                if current_date is not None
                else set()
            )
            return strategy.get_daily_updates(
                cast(TradeData, trade),
                history_dataframe,
                qualifying_setups=qualifying_setups,
            )
        return strategy.get_daily_updates(cast(TradeData, trade), history_dataframe)

    def _get_ndx_momentum_leaders(self) -> set[str]:
        """Loads current leader symbols for NDXMomentum strategy."""
        try:
            ndx_trades = self.trade_repository.get_all_by_strategy(
                Strategies.NDXMomentum
            )
            if isinstance(ndx_trades, list):
                return NDXMomentumTradeStrategy.extract_latest_leaders(ndx_trades)
        except (
            sqlite3.OperationalError,
            sqlite3.DatabaseError,
            Exception,
        ) as database_error:
            logger.warning(
                "Database error loading NDX leaders. Defaulting to empty set: %s",
                database_error,
            )
        return set()

    def _resolve_strategy_name(self, name: str) -> Strategies | None:
        """Resolves a string (legacy or canonical) to a strict Strategies Enum member.

        Args:
            name: Raw strategy name from the database record.

        Returns:
            Strategies | None: Resolved enum member, or None if unresolvable.
        """
        if not name:
            return None

        try:
            return Strategies(name)
        except ValueError as value_error:
            logger.debug(
                "Strategy name '%s' is not direct member of Strategies enum: %s",
                name,
                value_error,
            )

        lower_name = name.lower()
        if lower_name in STRATEGY_ALIASES:
            return STRATEGY_ALIASES[lower_name]

        clean_name = re.sub(r"[^a-z0-9]", "", lower_name)
        keyword_mappings: tuple[tuple[str, Strategies], ...] = (
            ("turnover", Strategies.TurnOverTiming),
            ("dip", Strategies.DipBuyer),
            ("hold", Strategies.HoldTarget),
            ("tp3", Strategies.HoldTarget),
            ("split", Strategies.SplitTarget),
            ("percent", Strategies.TwoPercent),
        )
        for keyword, strategy_enum in keyword_mappings:
            if keyword in clean_name:
                return strategy_enum

        return None

    def _get_strategy(self, strategy_name: str) -> BaseTradeStrategy | None:
        """Resolves a strategy name to its registered handler instance.

        Args:
            strategy_name: Raw strategy name string from the trade record.

        Returns:
            BaseTradeStrategy | None: The strategy instance, or None if not registered.
        """
        enum_key = self._resolve_strategy_name(strategy_name)

        if enum_key and enum_key in self.strategies:
            return self.strategies[enum_key]

        logger.warning(
            "Unknown strategy '%s' (resolved: %s). No handler registered.",
            strategy_name,
            enum_key,
        )
        return None

    def run_daily_process(self, reference_date: str | None = None) -> None:
        """Orchestrates the full EOD batch: exit checks then entry checks.

        Fails closed on database errors — a locked or corrupt database raises
        RuntimeError to halt the pipeline rather than silently skipping trades.

        Args:
            reference_date: Optional reference date string (YYYY-MM-DD) for batch execution.
        """
        logger.info("TradeManager: Starting daily process.")

        # 1. Active trade exit checks
        try:
            active_trades = self.trade_repository.get_by_status(TradeStatus.ACTIVE)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "Database unavailable during active trade load. Halting daily process."
            ) from database_error

        # Get latest leaders for NDXMomentum strategy
        ndx_leaders = self._get_ndx_momentum_leaders()
        self._wbs_run_cache = {}
        try:
            logger.info("Checking %d active trades for exits.", len(active_trades))
            for trade in active_trades:
                self._process_active_trade(
                    trade,
                    latest_leaders=ndx_leaders,
                    reference_date=reference_date,
                    verify_recency=True,
                    wbs_setups_cache=self._wbs_run_cache,
                )

            # 2. Pending trade entry checks
            try:
                created_trades = self._sort_created_trades(
                    self.trade_repository.get_by_status(TradeStatus.CREATED)
                )
                # Re-fetch active trades because some might have closed in step 1
                current_active_trades = self.trade_repository.get_by_status(
                    TradeStatus.ACTIVE
                )
                # Group active trade symbols by their resolved strategy enum
                active_symbols_by_strategy: dict[Strategies, set[str]] = {}
                for active_trade in current_active_trades:
                    resolved_strategy = self._resolve_strategy_name(
                        str(active_trade.get("strategy", ""))
                    )
                    if resolved_strategy:
                        active_symbols_by_strategy.setdefault(
                            resolved_strategy, set()
                        ).add(str(active_trade["symbol"]))
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
                raise RuntimeError(
                    "Database unavailable during created trade load. Halting daily process."
                ) from database_error

            logger.info("Checking %d pending trades for entries.", len(created_trades))
            for trade in created_trades:
                resolved_strategy = self._resolve_strategy_name(
                    str(trade.get("strategy", ""))
                )
                strategy_active_symbols = (
                    active_symbols_by_strategy.get(resolved_strategy, set())
                    if resolved_strategy
                    else set()
                )
                new_status = self._process_created_trade(
                    trade,
                    active_symbols=strategy_active_symbols,
                    wbs_setups_cache=self._wbs_run_cache,
                )
                if new_status == TradeStatus.ACTIVE and resolved_strategy:
                    active_symbols_by_strategy.setdefault(resolved_strategy, set()).add(
                        str(trade["symbol"])
                    )
        finally:
            self._wbs_run_cache.clear()

        logger.info("TradeManager: Daily process complete.")

    def _verify_and_sync_recency(
        self,
        symbol: str,
        history_dataframe: pd.DataFrame,
        start_date: str,
        reference_date: str | pd.Timestamp | datetime.date | None,
    ) -> tuple[bool, pd.DataFrame]:
        """Checks if the history has the expected candle, attempting a targeted sync if missing."""
        if reference_date is not None:
            ref_date_obj = pd.Timestamp(reference_date).date()
        else:
            ref_date_obj = datetime.datetime.now().date()

        expected_last_trading_day = get_last_completed_trading_day(
            ref_date_obj, self.holiday_checker
        )
        latest_candle_date = pd.Timestamp(history_dataframe.iloc[-1]["date"]).date()

        if latest_candle_date < expected_last_trading_day:
            logger.info(
                "Active trade [%s] missing expected candle for %s (latest in DB: %s). Attempting targeted update...",
                symbol,
                expected_last_trading_day,
                latest_candle_date,
            )
            self._attempt_targeted_market_update(symbol)
            history_dataframe = self.market_repository.get_symbol_history_raw(
                symbol, start_date=start_date
            )
            if not history_dataframe.empty:
                latest_candle_date = pd.Timestamp(
                    history_dataframe.iloc[-1]["date"]
                ).date()

        if latest_candle_date < expected_last_trading_day:
            logger.warning(
                "Active trade [%s] still missing candle for %s after sync (latest in DB: %s). Deferring exit check.",
                symbol,
                expected_last_trading_day,
                latest_candle_date,
            )
            return False, history_dataframe

        return True, history_dataframe

    def _apply_active_trade_updates(
        self,
        trade: dict[str, object],
        symbol: str,
        history_dataframe: pd.DataFrame,
        strategy: BaseTradeStrategy,
        transition: TradeTransition | None,
    ) -> None:
        """Applies price, dynamic context, and transition updates to the active trade record."""
        current_close = history_dataframe.iloc[-1]["close"]
        updates: dict[str, object] = {"current_price": current_close}

        is_closing = _is_transition_status(transition, TradeStatus.CLOSED)
        if not is_closing:
            daily_updates = self._get_daily_updates_for_strategy(
                strategy, trade, history_dataframe, self._wbs_run_cache
            )
            if daily_updates:
                merged_context = _merge_signal_context(
                    trade.get("signal_context"), daily_updates, trade_id=trade.get("id")
                )
                if merged_context is not None:
                    updates["signal_context"] = merged_context

        trade_id = int(str(trade["id"]))
        if transition:
            updates.update(transition.updates)
            self.trade_repository.update_trade(
                trade_id, updates, reason=transition.reason
            )
            logger.info("Trade update [%s]: %s", symbol, transition.message)
        else:
            self.trade_repository.update_trade(trade_id, updates)

    def _resolve_active_trade_leaders(
        self,
        resolved_strategy: Strategies | None,
        history_dataframe: pd.DataFrame,
        ndx_leaders: set[str] | None,
        wbs_setups_cache: dict[datetime.date, set[str]],
    ) -> set[str] | None:
        """Resolves strategy-isolated leader or setup sets for active trade management."""
        if resolved_strategy == Strategies.NDXMomentum:
            return ndx_leaders
        if resolved_strategy == Strategies.WoundedBullSprint:
            prior_date, _ = _resolve_wbs_eval_dates(history_dataframe)
            if prior_date is None:
                return set()
            return self._get_wbs_qualifying_setups(prior_date, wbs_setups_cache)
        return None

    def _process_active_trade(
        self,
        trade: dict[str, object],
        latest_leaders: set[str] | None = None,
        reference_date: str | pd.Timestamp | datetime.date | None = None,
        verify_recency: bool = False,
        *,
        wbs_setups_cache: dict[datetime.date, set[str]] | None = None,
    ) -> None:
        """Processes a single active trade: loads history and evaluates exit conditions.

        Data-level errors (bad symbol, missing history) are logged as warnings.
        Database errors propagate upward as RuntimeError (fail-closed).
        """
        symbol = str(trade.get("symbol", ""))
        strategy_name = str(trade.get("strategy", ""))
        strategy = self._get_strategy(strategy_name)

        if not strategy:
            logger.warning(
                "No strategy handler for '%s' (%s). Skipping.",
                strategy_name,
                symbol,
            )
            return

        try:
            history_dataframe = self._load_trade_history(
                symbol, trade, lookback_days=60
            )
            if history_dataframe.empty:
                return

            if verify_recency:
                actual_lookback = _resolve_trade_lookback_days(
                    strategy_name, lookback_days=60
                )
                start_date = _resolve_history_start_date(
                    trade, lookback_days=actual_lookback
                )
                is_valid, history_dataframe = self._verify_and_sync_recency(
                    symbol, history_dataframe, start_date, reference_date
                )
                if not is_valid:
                    return

            cache = wbs_setups_cache if wbs_setups_cache is not None else {}
            resolved_strategy = self._resolve_strategy_name(strategy_name)
            strategy_leaders = self._resolve_active_trade_leaders(
                resolved_strategy, history_dataframe, latest_leaders, cache
            )

            transition = strategy.manage_active_trade(
                cast(TradeData, trade),
                history_dataframe,
                latest_leaders=strategy_leaders,
            )

            if not history_dataframe.empty:
                self._apply_active_trade_updates(
                    trade,
                    symbol,
                    history_dataframe,
                    strategy,
                    transition,
                )

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"Database error processing active trade [{symbol}]."
            ) from database_error
        except (ValueError, KeyError, TypeError) as data_error:
            logger.warning(
                "Data error during exit check for [%s]: %s", symbol, data_error
            )

    def _process_created_trade(
        self,
        trade: dict[str, object],
        active_symbols: set[str] | None = None,
        *,
        wbs_setups_cache: dict[datetime.date, set[str]] | None = None,
    ) -> TradeStatus | None:
        """Processes a single pending trade: loads history and evaluates entry conditions.

        Args:
            trade: The trade record dictionary from the database.
            active_symbols: Set of active symbols to check duplicate positions.
            wbs_setups_cache: Optional per-run cache for WBS qualifying setups.

        Returns:
            TradeStatus | None: New trade status if a transition occurred, else None.
        """
        symbol = str(trade.get("symbol", ""))
        strategy_name = str(trade.get("strategy", ""))
        strategy = self._get_strategy(strategy_name)

        if not strategy:
            return None

        try:
            history_dataframe = self._load_trade_history(
                symbol, trade, lookback_days=60
            )
            if history_dataframe.empty:
                return None

            candle = history_dataframe.iloc[-1]
            transition = strategy.check_entry(
                cast(TradeData, trade),
                candle,
                history_dataframe,
                active_symbols=active_symbols,
            )
            if not transition:
                return None

            cache = wbs_setups_cache if wbs_setups_cache is not None else {}
            self._enrich_activation_context(
                trade,
                strategy,
                history_dataframe,
                transition,
                wbs_setups_cache=cache,
            )

            trade_id = int(str(trade["id"]))
            self.trade_repository.update_trade(
                trade_id, transition.updates, reason=transition.reason
            )
            logger.info("Entry check [%s]: %s", symbol, transition.message)
            return _extract_trade_status(transition.updates)

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"Database error processing created trade [{symbol}]."
            ) from database_error
        except (ValueError, KeyError, TypeError) as data_error:
            logger.warning(
                "Data error during entry check for [%s]: %s", symbol, data_error
            )
            return None

    def _enrich_activation_context(
        self,
        trade: dict[str, object],
        strategy: BaseTradeStrategy,
        history_dataframe: pd.DataFrame,
        transition: TradeTransition,
        *,
        wbs_setups_cache: dict[datetime.date, set[str]] | None = None,
    ) -> None:
        """Enriches transition updates with initial daily updates if status transitions to ACTIVE."""
        if not _is_transition_status(transition, TradeStatus.ACTIVE):
            return

        activated_trade = {**trade, **transition.updates}
        cache = wbs_setups_cache if wbs_setups_cache is not None else {}
        daily_updates = self._get_daily_updates_for_strategy(
            strategy, activated_trade, history_dataframe, cache
        )
        if daily_updates:
            raw_context = transition.updates.get("signal_context") or trade.get(
                "signal_context"
            )
            merged_context = _merge_signal_context(
                raw_context, daily_updates, trade_id=trade.get("id")
            )
            if merged_context is not None:
                transition.updates["signal_context"] = merged_context

    def generate_daily_orders(self, reference_date: str | None = None) -> str | None:
        """Generates daily order file in CSV format for CREATED and ACTIVE trades.

        Processes only specific strategies (ndx_momentum, turnover_timing,
        two_percent) and writes the result to a CSV file in data/orders/.

        Args:
            reference_date: ISO date string (YYYY-MM-DD) for the output filename.
                Defaults to today's date when None. Injectable for testing.

        Returns:
            str | None: Path to the written CSV file, or None if no orders generated.
        """
        logger.info("Generating daily orders.")

        try:
            created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)
            active_trades = self.trade_repository.get_by_status(TradeStatus.ACTIVE)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "Database unavailable during order generation. Halting."
            ) from database_error

        created_symbols = {str(t["symbol"]) for t in created_trades}

        date_string = reference_date or datetime.datetime.now().strftime("%Y-%m-%d")
        wbs_setups_cache: dict[datetime.date, set[str]] = {}

        active_exit_orders, active_symbols_by_strategy = (
            self._collect_exit_orders_for_active_trades(
                active_trades,
                created_symbols,
                reference_date=date_string,
                wbs_setups_cache=wbs_setups_cache,
            )
        )

        exiting_wbs_symbols = {
            str(trade.get("symbol", "")).upper().strip()
            for trade, _ in active_exit_orders
            if self._resolve_strategy_name(str(trade.get("strategy", "")))
            == Strategies.WoundedBullSprint
        }

        entry_orders = self._collect_entry_orders_for_created_trades(
            created_trades,
            created_symbols,
            active_symbols_by_strategy,
            reference_date=date_string,
            exiting_wbs_symbols=exiting_wbs_symbols,
        )

        orders_data = entry_orders + active_exit_orders

        if not orders_data:
            logger.info("No orders to generate.")
            return None

        csv_file_path = write_csv_orders_file(
            orders_data,
            date_string,
            self._ibkr_account_id,
            self._resolve_strategy_name,
        )

        if csv_file_path is None:
            return None

        return str(csv_file_path)

    def _collect_exit_orders_for_active_trades(
        self,
        active_trades: list[dict[str, object]],
        created_symbols: set[str],
        reference_date: str | None = None,
        *,
        wbs_setups_cache: dict[datetime.date, set[str]] | None = None,
    ) -> tuple[list[tuple[dict[str, object], Order]], dict[Strategies, set[str]]]:
        """Processes active trades to generate exit orders and track blocked symbols.

        Args:
            active_trades: List of active trade records from the database.
            created_symbols: Set of symbols with currently pending trades.
            reference_date: Target date for generating orders.
            wbs_setups_cache: Optional per-run cache for WBS qualifying setups.

        Returns:
            Tuple of (exit_orders, blocked_symbols_by_strategy).
        """
        active_symbols_by_strategy: dict[Strategies, set[str]] = {}
        active_exit_orders: list[tuple[dict[str, object], Order]] = []
        cache = wbs_setups_cache if wbs_setups_cache is not None else {}

        # Get latest leaders for NDXMomentum strategy to avoid exiting active leaders on month switch
        ndx_leaders = self._get_ndx_momentum_leaders()

        for active_trade in active_trades:
            resolved_strategy = self._resolve_strategy_name(
                str(active_trade.get("strategy", ""))
            )
            if not resolved_strategy:
                continue

            symbol = str(active_trade.get("symbol", ""))

            if resolved_strategy in _SINGLE_POSITION_STRATEGIES:
                active_symbols_by_strategy.setdefault(resolved_strategy, set()).add(
                    symbol
                )

            try:
                trade_created_symbols = created_symbols
                if resolved_strategy == Strategies.NDXMomentum:
                    trade_created_symbols = ndx_leaders

                order = self._generate_order_for_trade(
                    active_trade,
                    created_symbols=trade_created_symbols,
                    reference_date=reference_date,
                    wbs_setups_cache=cache,
                )
                if order:
                    active_exit_orders.append((active_trade, order))
                    logger.info("Exit order generated for [%s].", symbol)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
                raise RuntimeError(
                    f"Database error generating exit order for [{symbol}]."
                ) from database_error
            except (ValueError, KeyError, TypeError) as data_error:
                logger.warning(
                    "Data error generating exit order for [%s]: %s",
                    symbol,
                    data_error,
                )

        return active_exit_orders, active_symbols_by_strategy

    def _collect_entry_orders_for_created_trades(
        self,
        created_trades: list[dict[str, object]],
        created_symbols: set[str],
        active_symbols_by_strategy: dict[Strategies, set[str]],
        reference_date: str | None = None,
        *,
        exiting_wbs_symbols: set[str] | None = None,
    ) -> list[tuple[dict[str, object], Order]]:
        """Processes created trades to generate entry orders.

        Skips trades whose symbol is already active in a single-position strategy
        and enforces WBS score-prioritized portfolio slot capacity.
        """
        orders_data: list[tuple[dict[str, object], Order]] = []
        sorted_trades = self._sort_created_trades(created_trades)
        wbs_active_set = {
            s.upper().strip()
            for s in active_symbols_by_strategy.get(Strategies.WoundedBullSprint, set())
        } - (exiting_wbs_symbols or set())

        for trade in sorted_trades:
            symbol = str(trade.get("symbol", ""))
            resolved_strategy = self._resolve_strategy_name(
                str(trade.get("strategy", ""))
            )

            if self._is_entry_order_blocked(
                symbol, resolved_strategy, active_symbols_by_strategy, wbs_active_set
            ):
                continue

            try:
                order = self._generate_order_for_trade(
                    trade,
                    created_symbols=created_symbols,
                    reference_date=reference_date,
                )
                if order:
                    orders_data.append((trade, order))
                    if resolved_strategy == Strategies.WoundedBullSprint:
                        wbs_active_set.add(symbol.upper().strip())
                    logger.info("Order generated for [%s].", symbol)
            except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
                raise RuntimeError(
                    f"Database error generating order for [{symbol}]."
                ) from database_error
            except (ValueError, KeyError, TypeError) as data_error:
                logger.warning(
                    "Data error generating order for [%s]: %s", symbol, data_error
                )

        return orders_data

    def _is_entry_order_blocked(
        self,
        symbol: str,
        resolved_strategy: Strategies | None,
        active_symbols_by_strategy: dict[Strategies, set[str]],
        effective_wbs_active: set[str],
    ) -> bool:
        """Checks whether a CREATED trade entry order is blocked by active position or capacity."""
        if not resolved_strategy:
            return False

        if resolved_strategy == Strategies.WoundedBullSprint:
            related_symbols = SymbolFilter().get_related_symbols(symbol)
            if related_symbols & effective_wbs_active:
                logger.info(
                    "Skipping entry order for [%s] (%s) - position or share class already active.",
                    symbol,
                    resolved_strategy,
                )
                return True
            if len(effective_wbs_active) >= DEFAULT_MAX_POSITIONS:
                logger.info(
                    "Skipping entry order for [%s] (%s) - max portfolio slots (%d) reached.",
                    symbol,
                    resolved_strategy,
                    DEFAULT_MAX_POSITIONS,
                )
                return True
            return False

        if symbol in active_symbols_by_strategy.get(resolved_strategy, set()):
            logger.info(
                "Skipping entry order for [%s] (%s) - position already active.",
                symbol,
                resolved_strategy,
            )
            return True
        return False

    def _generate_order_for_trade(
        self,
        trade: dict[str, object],
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
        *,
        wbs_setups_cache: dict[datetime.date, set[str]] | None = None,
    ) -> Order | None:
        """Generates a single order object for a given trade proposal."""
        symbol = str(trade.get("symbol", ""))
        strategy_name = str(trade.get("strategy", ""))
        strategy = self._get_strategy(strategy_name)

        if not strategy:
            logger.warning(
                "No strategy handler for trade [%s] (%s).",
                symbol,
                strategy_name,
            )
            return None

        history_dataframe = self._load_trade_history(symbol, trade, lookback_days=60)
        resolved_strategy = self._resolve_strategy_name(strategy_name)

        effective_created_symbols = created_symbols
        trade_status = trade.get("status", "CREATED")
        status_str = (
            trade_status.value if hasattr(trade_status, "value") else str(trade_status)
        )
        if (
            resolved_strategy == Strategies.WoundedBullSprint
            and status_str == TradeStatus.ACTIVE.value
            and not history_dataframe.empty
        ):
            _, current_date = _resolve_wbs_eval_dates(history_dataframe)
            cache = wbs_setups_cache if wbs_setups_cache is not None else {}
            effective_created_symbols = (
                self._get_wbs_qualifying_setups(current_date, cache)
                if current_date is not None
                else set()
            )

        config_budget = 0.0
        if resolved_strategy:
            config_budget = settings.app.portfolio.get_budget(resolved_strategy.value)

        budget_val = trade.get("budget")
        budget: float = (
            float(str(budget_val)) if budget_val is not None else config_budget
        )

        # Call strategy order generation
        return strategy.generate_orders(
            cast(TradeData, trade),
            history_dataframe,
            budget,
            created_symbols=effective_created_symbols,
            reference_date=reference_date,
        )


def _resolve_trade_lookback_days(strategy_name: str, lookback_days: int = 60) -> int:
    """Resolves strategy-specific historical lookback buffer in calendar days."""
    lowered = strategy_name.lower()
    if Strategies.WoundedBullSprint.lower() in lowered:
        return max(lookback_days, DEFAULT_HISTORY_FETCH_DAYS)
    if Strategies.BounceBandit.lower() in lowered:
        return max(lookback_days, 350)
    return lookback_days


def _resolve_wbs_eval_dates(
    history_dataframe: pd.DataFrame,
) -> tuple[datetime.date | None, datetime.date | None]:
    """Extracts (prior_candle_date, current_candle_date) from sorted price history."""
    if history_dataframe.empty or "date" not in history_dataframe.columns:
        return None, None
    current_date = normalize_trading_date(history_dataframe.iloc[-1]["date"])
    if current_date is None:
        return None, None
    history_dates = pd.to_datetime(history_dataframe["date"]).dt.date
    prior_rows = history_dataframe[history_dates < current_date]
    prior_date = (
        normalize_trading_date(prior_rows.iloc[-1]["date"])
        if not prior_rows.empty
        else None
    )
    return prior_date, current_date


def _extract_trade_status(updates: dict[str, object]) -> TradeStatus | None:
    """Extracts a TradeStatus enum member from transition updates if present."""
    status_val = updates.get("status")
    if isinstance(status_val, TradeStatus):
        return status_val
    if isinstance(status_val, str):
        try:
            return TradeStatus(status_val)
        except ValueError:
            return None
    return None


def _is_transition_status(
    transition: TradeTransition | None, target_status: TradeStatus
) -> bool:
    """Checks whether a TradeTransition sets the given target status."""
    if transition is None:
        return False
    status_val = transition.updates.get("status")
    status_str = status_val.value if hasattr(status_val, "value") else status_val
    return status_str == target_status.value


def _extract_anchor_date(trade: dict[str, object]) -> str | None:
    """Extracts the anchor date string from trade record or signal context."""
    entry_date = trade.get("entry_date")
    if entry_date:
        return str(entry_date).split(" ")[0]

    signal_context = trade.get("signal_context")
    if signal_context:
        try:
            context_dict = (
                json.loads(str(signal_context))
                if isinstance(signal_context, str)
                else signal_context
            )
            if isinstance(context_dict, dict) and "date" in context_dict:
                return str(context_dict["date"]).split(" ")[0]
        except (json.JSONDecodeError, TypeError) as parse_error:
            logger.warning("Failed to parse date from signal_context: %s", parse_error)

    for date_key in ("created_at", "signal_date"):
        date_value = trade.get(date_key)
        if date_value:
            return str(date_value).split(" ")[0]

    return None


def _resolve_history_start_date(
    trade: dict[str, object],
    lookback_days: int = 0,
) -> str:
    """Derives the history start date from the trade's signal or entry date.

    Optionally subtracts a lookback_days buffer (e.g. 60 calendar days)
    to ensure technical indicators like SMA and RSI have sufficient pre-entry history.

    Args:
        trade: The trade record dictionary.
        lookback_days: Number of calendar days before the anchor date to load. Defaults to 0.

    Returns:
        str: ISO date string (YYYY-MM-DD) to use as the history query start.
    """
    anchor_date_str = _extract_anchor_date(trade)
    if not anchor_date_str:
        return _HARDCODED_HISTORY_FALLBACK_DATE

    if lookback_days <= 0:
        return anchor_date_str

    try:
        anchor_dt = pd.Timestamp(anchor_date_str)
        start_dt = anchor_dt - pd.Timedelta(days=lookback_days)
        return str(start_dt.strftime("%Y-%m-%d"))
    except (ValueError, TypeError):
        return anchor_date_str


def _merge_signal_context(
    raw_context: object,
    daily_updates: dict[str, object],
    trade_id: object = None,
) -> str | None:
    """Merges daily updates into the trade's signal_context JSON string.

    Args:
        raw_context: Current raw signal context (JSON string, dict, or None).
        daily_updates: Dictionary of new key-value updates to merge.
        trade_id: Optional trade identifier for diagnostic logging.

    Returns:
        str | None: Serialized updated JSON string, or None if parsing failed.
    """
    if not daily_updates:
        return None

    try:
        raw_source = raw_context or "{}"
        context_dict = (
            json.loads(str(raw_source)) if isinstance(raw_source, str) else raw_source
        )
        if isinstance(context_dict, dict):
            context_dict.update(daily_updates)
            return json.dumps(context_dict, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError) as parse_error:
        logger.warning(
            "Failed to parse signal_context JSON for trade %s: %s",
            trade_id,
            parse_error,
        )

    return None
