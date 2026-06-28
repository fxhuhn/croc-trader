import csv
import json
import logging
import os
import re
import sqlite3
from datetime import datetime
from pathlib import Path


from ...config import settings
from ...const import Strategies, STRATEGY_ALIASES
from ...types import TradeStatus
from ...database.repositories.trade import TradeRepository
from ...database.repositories.market import MarketRepository
from ...database.session import DatabaseSession
from ...services.telegram import TelegramBot
from ...models import Order

from .strategies.abstract import BaseTradeStrategy
from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.hold_target import HoldTargetStrategy
from .strategies.turnover_timing import TurnoverTimingStrategy
from .strategies.two_percent_strategy import TwoPercentStrategy
from .strategies.ndx_momentum import NDXMomentumTradeStrategy

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
    }
)

# Strategies whose orders are written to the daily CSV export.
# SplitTarget and HoldTarget are managed via the broker UI directly
# and are therefore intentionally excluded.
_CSV_SUPPORTED_STRATEGIES: frozenset[Strategies] = frozenset(
    {
        Strategies.NDXMomentum,
        Strategies.TurnOverTiming,
        Strategies.TurnOverTiming_10,
        Strategies.TurnOverTiming_05,
        Strategies.TwoPercent,
        Strategies.DipBuyer,
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

        self.strategies: dict[Strategies, BaseTradeStrategy] = {
            Strategies.DipBuyer: DipBuyerStrategy(),
            Strategies.TurnOverTiming: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_10: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_05: TurnoverTimingStrategy(),
            Strategies.HoldTarget: HoldTargetStrategy(),
            Strategies.SplitTarget: HoldTargetStrategy(),  # Fallback for old trades
            Strategies.TwoPercent: TwoPercentStrategy(),
            Strategies.NDXMomentum: NDXMomentumTradeStrategy(),
        }

        logger.info(
            "TradeManager initialized. Registered strategies: %s",
            list(self.strategies.keys()),
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

    def run_daily_process(self) -> None:
        """Orchestrates the full EOD batch: exit checks then entry checks.

        Fails closed on database errors — a locked or corrupt database raises
        RuntimeError to halt the pipeline rather than silently skipping trades.
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
            self._process_active_trade(trade, latest_leaders=latest_leaders)

        # 2. Pending trade entry checks
        try:
            created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)
            # Re-fetch active trades because some might have closed in step 1
            current_active_trades = self.trade_repository.get_by_status(
                TradeStatus.ACTIVE
            )
            active_symbols = {str(t["symbol"]) for t in current_active_trades}
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "Database unavailable during created trade load. Halting daily process."
            ) from database_error

        logger.info("Checking %d pending trades for entries.", len(created_trades))
        for trade in created_trades:
            self._process_created_trade(trade, active_symbols=active_symbols)

        logger.info("TradeManager: Daily process complete.")

    def _process_active_trade(
        self,
        trade: dict[str, object],
        latest_leaders: set[str] | None = None,
    ) -> None:
        """Processes a single active trade: loads history and evaluates exit conditions.

        Data-level errors (bad symbol, missing history) are logged as warnings.
        Database errors propagate upward as RuntimeError (fail-closed).

        Args:
            trade: The trade record dictionary from the database.
            latest_leaders: Optional set of active leaders.
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

        active_exit_orders, active_symbols_by_strategy = (
            self._collect_exit_orders_for_active_trades(active_trades, created_symbols)
        )

        entry_orders = self._collect_entry_orders_for_created_trades(
            created_trades, created_symbols, active_symbols_by_strategy
        )

        orders_data = entry_orders + active_exit_orders

        if not orders_data:
            logger.info("No orders to generate.")
            return None

        date_string = reference_date or datetime.now().strftime("%Y-%m-%d")
        csv_file_path = self._write_csv_orders_file(orders_data, date_string)

        if csv_file_path is None:
            return None

        return str(csv_file_path)

    def _collect_exit_orders_for_active_trades(
        self,
        active_trades: list[dict[str, object]],
        created_symbols: set[str],
    ) -> tuple[list[tuple[dict[str, object], Order]], dict[Strategies, set[str]]]:
        """Processes active trades to generate exit orders and track blocked symbols.

        Args:
            active_trades: List of active trade records from the database.
            created_symbols: Set of symbols with currently pending trades.

        Returns:
            Tuple of (exit_orders, blocked_symbols_by_strategy).
        """
        active_symbols_by_strategy: dict[Strategies, set[str]] = {}
        active_exit_orders: list[tuple[dict[str, object], Order]] = []

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
                order = self._generate_order_for_trade(
                    active_trade, created_symbols=created_symbols
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
    ) -> list[tuple[dict[str, object], Order]]:
        """Processes created trades to generate entry orders.

        Skips trades whose symbol is already active in a single-position strategy.

        Args:
            created_trades: List of created trade records from the database.
            created_symbols: Set of symbols with currently pending trades.
            active_symbols_by_strategy: Symbols blocked by active single-position trades.

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
                    trade, created_symbols=created_symbols
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
    ) -> Order | None:
        """Generates a single order object for a given trade proposal.

        Args:
            trade: The trade record dictionary from the database.
            created_symbols: Set of symbols with currently pending (CREATED) trades.

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
        )

    def _write_csv_orders_file(
        self,
        orders_data: list[tuple[dict[str, object], Order]],
        date_string: str,
    ) -> Path | None:
        """Transforms and saves generated orders to a CSV file in bracket layout."""
        filtered_orders_data: list[tuple[dict[str, object], Order, Strategies]] = []
        for trade, order in orders_data:
            resolved_strategy = self._resolve_strategy_name(
                str(trade.get("strategy", ""))
            )
            if resolved_strategy in _CSV_SUPPORTED_STRATEGIES:
                filtered_orders_data.append((trade, order, resolved_strategy))

        if not filtered_orders_data:
            logger.info("No orders found for CSV-supported strategies.")
            return None

        csv_rows = []

        for trade, order, resolved_strategy in filtered_orders_data:
            strategy_display_name = _get_strategy_display_name(resolved_strategy)
            trade_database_id = trade.get("id")
            symbol = str(trade.get("symbol", ""))
            trade_group_id = f"{trade_database_id}_{strategy_display_name}_{symbol}"

            rows = self._map_order_to_csv_rows(
                trade, order, trade_group_id, strategy_display_name
            )
            csv_rows.extend(rows)

        if not csv_rows:
            return None

        output_directory = Path("data/orders")
        output_directory.mkdir(parents=True, exist_ok=True)
        csv_filename = f"orders_{date_string.replace('-', '_')}.csv"
        csv_file_path = output_directory / csv_filename

        header = [
            "trade_group_id",
            "bracket_role",
            "symbol",
            "sec_type",
            "exchange",
            "account_id",
            "action",
            "quantity",
            "order_type",
            "target_price",
            "tif",
            "strategy_name",
        ]

        with open(csv_file_path, "w", newline="") as csv_file_handle:
            writer = csv.DictWriter(csv_file_handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(csv_rows)

        logger.info("CSV Orders saved to: %s", csv_file_path)
        return csv_file_path

    def _map_order_to_csv_rows(
        self,
        trade: dict[str, object],
        order: Order,
        trade_group_id: str,
        strategy_display_name: str,
    ) -> list[dict[str, object]]:
        """Maps an order model and its legs to structured CSV row dictionaries."""
        rows = []
        if order.entry:
            entry_leg = order.entry
            rows.append(
                {
                    "trade_group_id": trade_group_id,
                    "bracket_role": "ENTRY",
                    "symbol": order.symbol,
                    "sec_type": "STK",
                    "exchange": "SMART",
                    "account_id": self._ibkr_account_id,
                    "action": entry_leg.action,
                    "quantity": (
                        entry_leg.quantity
                        if entry_leg.quantity is not None
                        else order.quantity
                    ),
                    "order_type": entry_leg.type,
                    "target_price": f"{entry_leg.price:.2f}",
                    "tif": entry_leg.time_in_force,
                    "strategy_name": strategy_display_name,
                }
            )

        for exit_leg in order.exits:
            if order.entry is None:
                bracket_role = "EXIT"
            else:
                bracket_role = "SL" if exit_leg.type == "STP" else "TP"
            rows.append(
                {
                    "trade_group_id": trade_group_id,
                    "bracket_role": bracket_role,
                    "symbol": order.symbol,
                    "sec_type": "STK",
                    "exchange": "SMART",
                    "account_id": self._ibkr_account_id,
                    "action": exit_leg.action,
                    "quantity": (
                        exit_leg.quantity
                        if exit_leg.quantity is not None
                        else order.quantity
                    ),
                    "order_type": exit_leg.type,
                    "target_price": f"{exit_leg.price:.2f}",
                    "tif": exit_leg.time_in_force,
                    "strategy_name": strategy_display_name,
                }
            )

        return rows


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


_STRATEGY_DISPLAY_NAMES: dict[Strategies, str] = {
    Strategies.NDXMomentum: "NDXMomentum",
    Strategies.TurnOverTiming: "TurnoverTiming",
    Strategies.TurnOverTiming_10: "TurnoverTiming_1.0",
    Strategies.TurnOverTiming_05: "TurnoverTiming_0.5",
    Strategies.DipBuyer: "DipBuyer",
    Strategies.TwoPercent: "TwoPercent",
    Strategies.HoldTarget: "HoldTarget",
    Strategies.SplitTarget: "SplitTarget",
}


def _get_strategy_display_name(strategy_enum: Strategies) -> str:
    """Returns the standardized display name of a strategy for order reporting.

    Falls back to the raw enum value for any strategy not listed in the
    display-name table — new strategies are covered automatically.

    Args:
        strategy_enum: The resolved Strategies enum member.

    Returns:
        str: Human-readable display name used in CSV output.
    """
    return _STRATEGY_DISPLAY_NAMES.get(strategy_enum, str(strategy_enum.value))
