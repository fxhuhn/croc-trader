import logging
import re
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import yaml

from ...const import Strategies, STRATEGY_ALIASES
from ...types import TradeStatus
from ...database.repositories.trade import TradeRepository
from ...database.repositories.market import MarketRepository
from ...database.session import DatabaseSession
from ...services.telegram import TelegramBot

from .strategies.dip_buyer import DipBuyerStrategy
from .strategies.hold_target import HoldTargetStrategy
from .strategies.split_target import SplitTargetStrategy
from .strategies.turnover_timing import TurnoverTimingStrategy
from .strategies.two_percent_strategy import TwoPercentStrategy
from .strategies.ndx_momentum import NDXMomentumTradeStrategy

logger = logging.getLogger(__name__)

_HARDCODED_HISTORY_FALLBACK_DATE = "2024-01-01"


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
    ) -> None:
        """Initializes repositories and strategy registry.

        Args:
            db_path: Path to the signals database.
            stocks_db_path: Path to the market data (stocks) database.
            telegram_bot: Optional Telegram notification client.
        """
        self.db_path = db_path
        self.stocks_db_path = stocks_db_path
        self.telegram = telegram_bot

        self.stocks_session = DatabaseSession(str(stocks_db_path))
        self.market_repo = MarketRepository(self.stocks_session)

        self.signals_session = DatabaseSession(str(db_path))
        self.trade_repository = TradeRepository(self.signals_session)

        self.strategies = {
            Strategies.DipBuyer: DipBuyerStrategy(),
            Strategies.TurnOverTiming: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_10: TurnoverTimingStrategy(),
            Strategies.TurnOverTiming_05: TurnoverTimingStrategy(),
            Strategies.HoldTarget: HoldTargetStrategy(),
            Strategies.SplitTarget: SplitTargetStrategy(),
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
        except ValueError:
            pass

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

    def _get_strategy(self, strategy_name: str) -> object | None:
        """Resolves a strategy name to its registered handler instance.

        Args:
            strategy_name: Raw strategy name string from the trade record.

        Returns:
            The strategy instance, or None if not registered.
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

        logger.info("Checking %d active trades for exits.", len(active_trades))
        for trade in active_trades:
            self._process_active_trade(trade)

        # 2. Pending trade entry checks
        try:
            created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "Database unavailable during created trade load. Halting daily process."
            ) from database_error

        logger.info("Checking %d pending trades for entries.", len(created_trades))
        for trade in created_trades:
            self._process_created_trade(trade)

        logger.info("TradeManager: Daily process complete.")

    def _process_active_trade(self, trade: dict[str, object]) -> None:
        """Processes a single active trade: loads history and evaluates exit conditions.

        Data-level errors (bad symbol, missing history) are logged as warnings.
        Database errors propagate upward as RuntimeError (fail-closed).

        Args:
            trade: The trade record dictionary from the database.
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
            df_hist = self.market_repo.get_symbol_history_raw(
                symbol, start_date=start_date
            )
            if df_hist.empty:
                return

            result = strategy.manage_active_trade(trade, df_hist, self.trade_repository)

            if result:
                logger.info("Trade update [%s]: %s", symbol, result)

            if not df_hist.empty:
                current_close = df_hist.iloc[-1]["close"]
                self.trade_repository.update_trade(
                    trade["id"], {"current_price": current_close}
                )

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"Database error processing active trade [{symbol}]."
            ) from database_error
        except (ValueError, KeyError, TypeError) as data_error:
            logger.warning(
                "Data error during exit check for [%s]: %s", symbol, data_error
            )

    def _process_created_trade(self, trade: dict[str, object]) -> None:
        """Processes a single pending trade: loads history and evaluates entry conditions.

        Args:
            trade: The trade record dictionary from the database.
        """
        symbol = str(trade.get("symbol", ""))
        strategy_name = str(trade.get("strategy", ""))
        strategy = self._get_strategy(strategy_name)

        if not strategy:
            return

        try:
            start_date = _resolve_history_start_date(trade)
            df_hist = self.market_repo.get_symbol_history_raw(
                symbol, start_date=start_date
            )
            if df_hist.empty:
                return

            candle = df_hist.iloc[-1]
            result = strategy.check_entry(trade, candle, df_hist, self.trade_repository)

            if result:
                logger.info("Entry check [%s]: %s", symbol, result)

        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                f"Database error processing created trade [{symbol}]."
            ) from database_error
        except (ValueError, KeyError, TypeError) as data_error:
            logger.warning(
                "Data error during entry check for [%s]: %s", symbol, data_error
            )

    def generate_daily_orders(self) -> str | None:
        """Generates daily order file for all CREATED trades across all strategies.

        Serializes the result as a YAML file in data/orders/.

        Returns:
            str | None: Path to the written YAML file, or None if no orders generated.
        """
        logger.info("Generating daily orders.")

        try:
            created_trades = self.trade_repository.get_by_status(TradeStatus.CREATED)
        except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
            raise RuntimeError(
                "Database unavailable during order generation. Halting."
            ) from database_error

        orders = []

        for trade in created_trades:
            symbol = str(trade.get("symbol", ""))
            strategy_name = str(trade.get("strategy", ""))
            strategy = self._get_strategy(strategy_name)

            if not strategy:
                logger.warning(
                    "No strategy handler for trade [%s] (%s).",
                    symbol,
                    strategy_name,
                )
                continue

            try:
                start_date = _resolve_history_start_date(trade)
                df_hist = self.market_repo.get_symbol_history_raw(
                    symbol, start_date=start_date
                )

                budget = float(trade.get("budget") or 2000.0)

                order = strategy.generate_orders(
                    trade, df_hist, budget, self.trade_repository
                )

                if order:
                    order_dict = asdict(order)
                    cleaned_order = _remove_none_values(order_dict)  # type: ignore
                    orders.append(cleaned_order)
                    logger.info("Order generated for [%s].", symbol)

            except (sqlite3.OperationalError, sqlite3.DatabaseError) as database_error:
                raise RuntimeError(
                    f"Database error generating order for [{symbol}]."
                ) from database_error
            except (ValueError, KeyError, TypeError) as data_error:
                logger.warning(
                    "Data error generating order for [%s]: %s", symbol, data_error
                )

        if not orders:
            logger.info("No orders to generate.")
            return None

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = f"orders_{date_str}.yaml"
        output_dir = Path("data/orders")
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / filename

        with open(file_path, "w") as file_handle:
            yaml.dump(orders, file_handle, sort_keys=False, default_flow_style=False)

        logger.info("Orders saved to: %s", file_path)
        return str(file_path)


def _resolve_history_start_date(trade: dict[str, object]) -> str:
    """Derives the history start date from the trade's signal or entry date.

    Avoids the hardcoded '2024-01-01' fallback by using the trade's own
    temporal anchor. Falls back to a constant only when no date is available.

    Args:
        trade: The trade record dictionary.

    Returns:
        str: ISO date string (YYYY-MM-DD) to use as the history query start.
    """
    for date_key in ("entry_date", "created_at", "signal_date"):
        date_value = trade.get(date_key)
        if date_value:
            return str(date_value).split(" ")[0]
    return _HARDCODED_HISTORY_FALLBACK_DATE


def _remove_none_values(data: object) -> object:
    """Recursively removes all keys with None values from nested dicts or lists.

    Args:
        data: The object to clean (dict, list, or scalar).

    Returns:
        The cleaned object with all None-valued keys removed.
    """
    if isinstance(data, dict):
        return {
            key: _remove_none_values(value)
            for key, value in data.items()
            if value is not None
        }
    if isinstance(data, list):
        return [_remove_none_values(item) for item in data]
    return data
