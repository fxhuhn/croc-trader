import datetime
import json
import logging
import os
import re
import sqlite3
from pathlib import Path

import pandas as pd

from ...config import settings
from ...const import STRATEGY_ALIASES, Strategies
from ...database.repositories.market import MarketRepository
from ...database.repositories.trade import TradeRepository
from ...database.session import DatabaseSession
from ...models import Order
from ...services.market.updater import MarketDataUpdater
from ...services.telegram import TelegramBot
from ...tools.market_holidays import MarketHolidayChecker
from ...tools.trading_calendar import get_last_completed_trading_day
from ...types import TradeStatus
from .order_export import write_csv_orders_file
from .strategies.abstract import BaseTradeStrategy
from .strategies.bridge_scout import BridgeScoutTradeStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.hold_target import HoldTargetStrategy
from .strategies.ndx_momentum import NDXMomentumTradeStrategy
from .strategies.tgim import TGIMTradeStrategy
from .strategies.turnover_timing import TurnoverTimingStrategy
from .strategies.two_percent_strategy import TwoPercentStrategy

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
        }

        logger.info(
            "TradeManager initialized. Registered strategies: %s",
            list(self.strategies.keys()),
        )

    def _attempt_targeted_market_update(self, symbol: str) -> None:
        """Attempts a targeted market data update for a single active trade symbol."""
        try:
            updater = MarketDataUpdater(self.stocks_session, self.signals_session)
            updater.run_update(specific_symbols=[symbol])
        except Exception as error:
            logger.warning(
                "Failed targeted market data update for %s: %s", symbol, error
            )

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

        if "turnover" in clean_name:
            return Strategies.TurnOverTiming
        if "dip" in clean_name:
            return Strategies.DipBuyer
        if "hold" in clean_name or "tp3" in clean_name:
            return Strategies.HoldTarget
        if "split" in clean_name:
            return Strategies.SplitTarget
        if "percent" in clean_name:
            return Strategies.TwoPercent

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
        try:
            ndx_trades = self.trade_repository.get_all_by_strategy(
                Strategies.NDXMomentum
            )
            latest_leaders = NDXMomentumTradeStrategy.extract_latest_leaders(ndx_trades)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            logger.warning(
                "Database error loading NDX leaders. Defaulting to empty set: %s",
                database_error,
            )
            latest_leaders = set()

        logger.info("Checking %d active trades for exits.", len(active_trades))
        for trade in active_trades:
            self._process_active_trade(
                trade,
                latest_leaders=latest_leaders,
                reference_date=reference_date,
                verify_recency=True,
            )

        # 2. Pending trade entry checks
        try:
            created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)
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
                    active_symbols_by_strategy.setdefault(resolved_strategy, set()).add(
                        str(active_trade["symbol"])
                    )
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
            self._process_created_trade(trade, active_symbols=strategy_active_symbols)

        logger.info("TradeManager: Daily process complete.")

    def _process_active_trade(
        self,
        trade: dict[str, object],
        latest_leaders: set[str] | None = None,
        reference_date: str | pd.Timestamp | datetime.date | None = None,
        verify_recency: bool = False,
    ) -> None:
        """Processes a single active trade: loads history and evaluates exit conditions.

        Data-level errors (bad symbol, missing history) are logged as warnings.
        Database errors propagate upward as RuntimeError (fail-closed).

        Args:
            trade: The trade record dictionary from the database.
            latest_leaders: Optional set of active leaders.
            reference_date: Optional reference date for recency calculation.
            verify_recency: Whether to enforce recency verification against the expected trading day.
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
            start_date = _resolve_history_start_date(trade)
            history_dataframe = self.market_repository.get_symbol_history_raw(
                symbol, start_date=start_date
            )
            if history_dataframe.empty:
                return

            if verify_recency:
                if reference_date is not None:
                    ref_date_obj = pd.Timestamp(reference_date).date()
                else:
                    ref_date_obj = datetime.datetime.now().date()

                expected_last_trading_day = get_last_completed_trading_day(
                    ref_date_obj, self.holiday_checker
                )
                latest_candle_date = pd.Timestamp(
                    history_dataframe.iloc[-1]["date"]
                ).date()

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
                    return

            transition = strategy.manage_active_trade(
                trade, history_dataframe, latest_leaders=latest_leaders
            )

            if not history_dataframe.empty:
                current_close = history_dataframe.iloc[-1]["close"]
                updates = {"current_price": current_close}

                # Update dynamic trade state (e.g. thresholds, targets) via the strategy
                daily_updates = strategy.get_daily_updates(trade, history_dataframe)
                if daily_updates:
                    try:
                        signal_context_raw = trade.get("signal_context") or "{}"
                        context_dict = (
                            json.loads(signal_context_raw)
                            if isinstance(signal_context_raw, str)
                            else signal_context_raw
                        )
                        if isinstance(context_dict, dict):
                            context_dict.update(daily_updates)
                            updates["signal_context"] = json.dumps(
                                context_dict, ensure_ascii=False
                            )
                    except (json.JSONDecodeError, TypeError) as parse_error:
                        logger.warning(
                            "Failed to parse signal_context JSON for trade %s: %s",
                            trade.get("id"),
                            parse_error,
                        )

                if transition:
                    updates.update(transition.updates)
                    self.trade_repository.update_trade(
                        trade["id"], updates, reason=transition.reason
                    )
                    logger.info("Trade update [%s]: %s", symbol, transition.message)
                else:
                    self.trade_repository.update_trade(trade["id"], updates)

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
    ) -> None:
        """Processes a single pending trade: loads history and evaluates entry conditions.

        Args:
            trade: The trade record dictionary from the database.
            active_symbols: Set of active symbols to check duplicate positions.
        """
        symbol = str(trade.get("symbol", ""))
        strategy_name = str(trade.get("strategy", ""))
        strategy = self._get_strategy(strategy_name)

        if not strategy:
            return

        try:
            start_date = _resolve_history_start_date(trade)
            history_dataframe = self.market_repository.get_symbol_history_raw(
                symbol, start_date=start_date
            )
            if history_dataframe.empty:
                return

            candle = history_dataframe.iloc[-1]
            transition = strategy.check_entry(
                trade, candle, history_dataframe, active_symbols=active_symbols
            )

            if transition:
                status_val = transition.updates.get("status")
                status_str = (
                    status_val.value if hasattr(status_val, "value") else status_val
                )
                if status_str == "ACTIVE":
                    daily_updates = strategy.get_daily_updates(trade, history_dataframe)
                    if daily_updates:
                        try:
                            signal_context_raw = (
                                transition.updates.get("signal_context")
                                or trade.get("signal_context")
                                or "{}"
                            )
                            context_dict = (
                                json.loads(signal_context_raw)
                                if isinstance(signal_context_raw, str)
                                else signal_context_raw
                            )
                            if isinstance(context_dict, dict):
                                context_dict.update(daily_updates)
                                transition.updates["signal_context"] = json.dumps(
                                    context_dict, ensure_ascii=False
                                )
                        except (json.JSONDecodeError, TypeError) as parse_error:
                            logger.warning(
                                "Failed to parse signal_context JSON for trade %s: %s",
                                trade.get("id"),
                                parse_error,
                            )

                self.trade_repository.update_trade(
                    trade["id"], transition.updates, reason=transition.reason
                )
                logger.info("Entry check [%s]: %s", symbol, transition.message)

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"Database error processing created trade [{symbol}]."
            ) from database_error
        except (ValueError, KeyError, TypeError) as data_error:
            logger.warning(
                "Data error during entry check for [%s]: %s", symbol, data_error
            )

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

        active_exit_orders, active_symbols_by_strategy = (
            self._collect_exit_orders_for_active_trades(
                active_trades, created_symbols, reference_date=date_string
            )
        )

        entry_orders = self._collect_entry_orders_for_created_trades(
            created_trades,
            created_symbols,
            active_symbols_by_strategy,
            reference_date=date_string,
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
    ) -> tuple[list[tuple[dict[str, object], Order]], dict[Strategies, set[str]]]:
        """Processes active trades to generate exit orders and track blocked symbols.

        Args:
            active_trades: List of active trade records from the database.
            created_symbols: Set of symbols with currently pending trades.
            reference_date: Target date for generating orders.

        Returns:
            Tuple of (exit_orders, blocked_symbols_by_strategy).
        """
        active_symbols_by_strategy: dict[Strategies, set[str]] = {}
        active_exit_orders: list[tuple[dict[str, object], Order]] = []

        # Get latest leaders for NDXMomentum strategy to avoid exiting active leaders on month switch
        ndx_leaders: set[str] = set()
        try:
            ndx_trades = self.trade_repository.get_all_by_strategy(
                Strategies.NDXMomentum
            )
            if isinstance(ndx_trades, list):
                ndx_leaders = NDXMomentumTradeStrategy.extract_latest_leaders(
                    ndx_trades
                )
        except Exception as database_error:
            logger.warning(
                "Failed to load NDX leaders during exit collection: %s. Using empty set.",
                database_error,
            )

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
    ) -> list[tuple[dict[str, object], Order]]:
        """Processes created trades to generate entry orders.

        Skips trades whose symbol is already active in a single-position strategy.

        Args:
            created_trades: List of created trade records from the database.
            created_symbols: Set of symbols with currently pending trades.
            active_symbols_by_strategy: Symbols blocked by active single-position trades.
            reference_date: Target date for generating orders.

        Returns:
            List of (trade, order) tuples for entry orders.
        """
        orders_data: list[tuple[dict[str, object], Order]] = []

        for trade in created_trades:
            symbol = str(trade.get("symbol", ""))
            resolved_strategy = self._resolve_strategy_name(
                str(trade.get("strategy", ""))
            )

            if resolved_strategy and symbol in active_symbols_by_strategy.get(
                resolved_strategy, set()
            ):
                logger.info(
                    "Skipping entry order for [%s] (%s) - position already active.",
                    symbol,
                    resolved_strategy,
                )
                continue

            try:
                order = self._generate_order_for_trade(
                    trade,
                    created_symbols=created_symbols,
                    reference_date=reference_date,
                )
                if order:
                    orders_data.append((trade, order))
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

    def _generate_order_for_trade(
        self,
        trade: dict[str, object],
        created_symbols: set[str] | None = None,
        reference_date: str | None = None,
    ) -> Order | None:
        """Generates a single order object for a given trade proposal.

        Args:
            trade: The trade record dictionary from the database.
            created_symbols: Set of symbols with currently pending (CREATED) trades.
            reference_date: Target date for generating orders.

        Returns:
            Order | None: The generated order or None if no handler is found.
        """
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

        start_date = _resolve_history_start_date(trade)
        history_dataframe = self.market_repository.get_symbol_history_raw(
            symbol, start_date=start_date
        )

        resolved_strategy = self._resolve_strategy_name(strategy_name)
        config_budget = 0.0
        if resolved_strategy:
            config_budget = settings.app.portfolio.get_budget(resolved_strategy.value)

        budget: float = float(trade.get("budget") or config_budget)

        # Call strategy order generation
        return strategy.generate_orders(
            trade,
            history_dataframe,
            budget,
            created_symbols=created_symbols,
            reference_date=reference_date,
        )

    def _write_csv_orders_file(
        self,
        orders_data: list[tuple[dict[str, object], Order]],
        date_string: str,
    ) -> Path | None:
        """Backward-compatible helper calling the extracted write_csv_orders_file function."""
        return write_csv_orders_file(
            orders_data,
            date_string,
            self._ibkr_account_id,
            self._resolve_strategy_name,
        )


def _resolve_history_start_date(trade: dict[str, object]) -> str:
    """Derives the history start date from the trade's signal or entry date.

    Avoids the hardcoded '2024-01-01' fallback by using the trade's own
    temporal anchor. Falls back to a constant only when no date is available.

    Args:
        trade: The trade record dictionary.

    Returns:
        str: ISO date string (YYYY-MM-DD) to use as the history query start.
    """
    # 1. Prefer entry_date if available
    entry_date = trade.get("entry_date")
    if entry_date:
        return str(entry_date).split(" ")[0]

    # 2. Prefer date from signal_context if available (crucial for CREATED trades)
    signal_context = trade.get("signal_context")
    if signal_context:
        try:
            context_dict = (
                json.loads(signal_context)
                if isinstance(signal_context, str)
                else signal_context
            )

            if isinstance(context_dict, dict) and "date" in context_dict:
                return str(context_dict["date"]).split(" ")[0]
        except (json.JSONDecodeError, TypeError) as parse_error:
            logger.warning("Failed to parse date from signal_context: %s", parse_error)

    # 3. Fall back to other keys
    for date_key in ("created_at", "signal_date"):
        date_value = trade.get(date_key)
        if date_value:
            return str(date_value).split(" ")[0]

    return _HARDCODED_HISTORY_FALLBACK_DATE
