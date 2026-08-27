import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, TypedDict, cast

import pandas as pd
import plotly.graph_objects as go

from ...const import (
    STRATEGY_ALIASES,
    ExitReason,
    IndexAliases,
    Strategies,
    TargetColumn,
    TradeStatus,
)
from ...database.repositories.broker import (
    ActivePositionRecord,
    BrokerRepository,
    SettlementRecord,
)
from ...database.repositories.market import MarketRepository
from ...database.repositories.trade import TradeRepository
from ...tools import metrics
from ...types import TradeData
from .strategies.bounce_bandit import BounceBanditTradeStrategy

logger = logging.getLogger(__name__)

CRITICAL_STOP_LOSS_THRESHOLD: float = 0.01
MIN_HISTORY_FOR_BOUNCE_BANDIT_DAILY_UPDATE: int = 8
MIN_SERIES_LENGTH_FOR_SQN: int = 2
MIN_ROWS_FOR_PREVIOUS_CANDLE: int = 2
MIN_TIMESTAMP_SPLIT_PARTS: int = 2


@dataclass(frozen=True)
class ViewBuildParams:
    """Encapsulates parameters for constructing a TradeViewData object."""

    trade: dict[str, object]
    context_dict: dict[str, object]
    display_entry: str
    display_exit: str
    days_held: int
    prices: dict[str, float]
    pnl_data: dict[str, object]


@dataclass
class StrategyMetricAccumulator:
    """Accumulator for computing broker strategy statistics."""

    pnl: float = 0.0
    fees: float = 0.0
    total_count: int = 0
    win_count: int = 0
    slippage_sum: float = 0.0


class SymbolTradeGroup(TypedDict):
    """Grouped active trades for a single symbol."""

    symbol: str
    total_pnl: float
    total_invested: float
    total_pnl_percentage: float
    variants: list["TradeViewData"]


class HistoryTradeGroup(TypedDict):
    """Grouped closed trade history records."""

    symbol: str
    entry_date: str
    max_exit: str
    display_index: str
    trades: list["TradeViewData"]


def _map_strategy_filter_name(raw_name: str) -> str:
    """Normalizes a raw strategy name into its UI filter representation."""
    raw_lower = raw_name.lower()
    if "dip" in raw_lower:
        return "DipBuyer"
    if "turnover" in raw_lower:
        return "TurnoverTiming"
    if "twopercent" in raw_lower or "percent" in raw_lower:
        return "TwoPercent"
    if "ndx" in raw_lower or "momentum" in raw_lower:
        return "NDXMomentum"
    return raw_name


class TradeViewData(TradeData, total=False):
    """
    Extended dictionary for trade view data, inheriting from TradeData.
    Adds display-specific fields.
    """

    # Inherited fields from TradeData: id, symbol, strategy, status, etc.

    # Calculated / Display Fields
    display_entry: str
    display_exit: str
    days_held: int
    unrealized_pnl: float
    pnl_percentage: float
    is_critical: bool
    progress: float
    display_size: float
    sparkline: str
    max_days: int | None
    version: str | None
    current_price: float | None
    stop_loss: float | None
    take_profit: float | None
    tws_status: str | None
    tws_orders: list[object]
    green_candle_count: int | None

    # Context (overrides str | None from TradeData for parsed dict)
    context: dict[str, object]


class TradeViewService:
    """Service to prepare trade data for UI views."""

    def __init__(
        self,
        trade_repository: TradeRepository,
        market_repository: MarketRepository,
        broker_repository: BrokerRepository | None = None,
    ) -> None:
        """Initializes the view service with repositories.

        Args:
            trade_repository: The repository for trade database access.
            market_repository: The repository for market history database access.
            broker_repository: The repository for TWS broker database access.
        """
        self.trade_repository = trade_repository
        self.market_repository = market_repository
        self.broker_repository = broker_repository

    def resolve_strategy(self, trade: dict[str, object]) -> str:
        """Resolves a trade's strategy string to its Enum value.

        Args:
            trade: The trade record dictionary.

        Returns:
            str: The resolved strategy name.
        """
        raw_strategy_name = str(trade.get("strategy", "")).lower()
        # Check exact match first
        try:
            # Check if it's already a valid Enum value
            Strategies(raw_strategy_name)
        except ValueError as value_error:
            # Not a valid enum member, proceed to alias lookup
            logger.debug(
                "Strategy '%s' is not direct member of Strategies enum: %s",
                raw_strategy_name,
                value_error,
            )

        resolved = STRATEGY_ALIASES.get(raw_strategy_name)
        return resolved if resolved else raw_strategy_name

    def is_strategy_match(
        self, trade: dict[str, object], target: str | Sequence[str]
    ) -> bool:
        """Checks if a trade belongs to a strategy or list of strategies.

        Args:
            trade: The trade record dictionary.
            target: Strategy name or list of strategy names.

        Returns:
            bool: True if the trade strategy matches target.
        """
        resolved_strategy = self.resolve_strategy(trade)

        if not isinstance(target, str):
            return resolved_strategy in target

        return resolved_strategy == target

    def _parse_context(self, trade: dict[str, object]) -> dict[str, object]:
        """Parses the JSON context string safely."""
        try:
            raw_context = trade.get("signal_context")
            if isinstance(raw_context, str) and raw_context:
                parsed = json.loads(raw_context)
                if isinstance(parsed, dict):
                    return {str(k): v for k, v in parsed.items()}
                return {}
            if isinstance(raw_context, dict):
                return {str(k): v for k, v in raw_context.items()}
            return {}
        except (json.JSONDecodeError, TypeError) as parse_error:
            logger.warning(
                "Failed to parse signal_context for trade %s: %s",
                trade.get("id"),
                parse_error,
            )
            return {}

    def _calculate_days_held(
        self,
        symbol: str,
        entry_date: object | None,
        exit_date: object | None,
    ) -> int:
        """Calculates holding period using trading days.

        Args:
            symbol: The ticker symbol.
            entry_date: Date of trade entry.
            exit_date: Date of trade exit.

        Returns:
            int: Number of trading days held.
        """
        if not entry_date:
            return 0

        start_date_string = str(entry_date).split(" ")[0]
        if exit_date:
            end_date_string = str(exit_date).split(" ")[0]
        else:
            end_date_string = date.today().strftime("%Y-%m-%d")

        return self.market_repository.get_trading_days_count(
            symbol, start_date_string, end_date_string
        )

    def _prepare_active_trade_view_fields(
        self,
        trade: dict[str, object],
        context_dict: dict[str, object],
        entry_price: float,
        current_price: float,
        initial_size: float,
    ) -> tuple[float, float, bool, float]:
        """Calculates PnL, progress, and criticality for an active trade.

        Args:
            trade: The trade dictionary record.
            context_dict: The parsed JSON context dictionary.
            entry_price: The trade entry price.
            current_price: The current price of the asset.
            initial_size: The initial share size.

        Returns:
            tuple[float, float, bool, float]: Tuple of (unrealized_pnl, pnl_percentage, is_critical, progress).
        """
        if entry_price <= 0:
            return 0.0, 0.0, False, 0.0

        direction = str(context_dict.get("direction", "long")).lower()
        if direction == "short":
            unrealized_pnl = (entry_price - current_price) * initial_size
            pnl_percentage = ((entry_price - current_price) / entry_price) * 100
        else:
            unrealized_pnl = (current_price - entry_price) * initial_size
            pnl_percentage = ((current_price - entry_price) / entry_price) * 100

        stop_loss_val = trade.get("current_stop_loss")
        stop_loss = float(str(stop_loss_val)) if stop_loss_val is not None else 0.0
        target_price = 0.0

        # Target Hierarchy using Enums
        for key in [
            TargetColumn.TARGET_PRICE,
            TargetColumn.TP3,
            TargetColumn.TAKE_PROFIT_3,
            TargetColumn.TP1,
            TargetColumn.TAKE_PROFIT_1,
        ]:
            if value := context_dict.get(key):
                target_price = float(value)  # type: ignore[arg-type]  # value is dynamically extracted and expected to be float-convertible
                break

        # Progress Calculation
        progress = 0.0
        if stop_loss > 0.0 and target_price > 0.0 and stop_loss != target_price:
            total_range = target_price - stop_loss
            current_distance = current_price - stop_loss
            percentage_value = (current_distance / total_range) * 100
            progress = max(0.0, min(100.0, percentage_value))

        # Critical SL
        is_critical = False
        if stop_loss > 0.0:
            distance = abs(current_price - stop_loss)
            if (
                current_price > 0
                and (distance / current_price) < CRITICAL_STOP_LOSS_THRESHOLD
            ):
                is_critical = True

        return unrealized_pnl, pnl_percentage, is_critical, progress

    def _prepare_closed_trade_view_fields(
        self,
        trade: dict[str, object],
        context_dict: dict[str, object],
        entry_price: float,
        exit_price: float,
        initial_size: float,
    ) -> tuple[float, float]:
        """Calculates PnL and percentage for a closed trade.

        Args:
            trade: The trade dictionary record.
            context_dict: The parsed JSON context dictionary.
            entry_price: The trade entry price.
            exit_price: The trade exit price.
            initial_size: The initial share size.

        Returns:
            tuple[float, float]: Tuple of (realized_pnl, pnl_percentage).
        """
        realized_pnl_val = trade.get("realized_pnl")
        realized_pnl = (
            float(str(realized_pnl_val)) if realized_pnl_val is not None else 0.0
        )
        pnl_percentage = 0.0

        if entry_price > 0:
            direction = str(context_dict.get("direction", "long")).lower()

            if realized_pnl == 0.0 and exit_price > 0:
                if direction == "short":
                    realized_pnl = (entry_price - exit_price) * initial_size
                else:
                    realized_pnl = (exit_price - entry_price) * initial_size

            if direction == "short":
                price_difference = entry_price - exit_price
            else:
                price_difference = exit_price - entry_price

            pnl_percentage = (price_difference / entry_price) * 100

        return realized_pnl, pnl_percentage

    def _enrich_bounce_bandit_context(
        self, trade: dict[str, object], context_dict: dict[str, object]
    ) -> None:
        """Enriches context for BounceBandit trades with dynamic indicators if missing."""
        if not self.is_strategy_match(trade, Strategies.BounceBandit):
            return
        if (
            context_dict.get("sma_8") is not None
            and context_dict.get("target") is not None
        ):
            return

        symbol = str(trade.get("symbol", ""))
        if not symbol:
            return

        start_date = (pd.Timestamp.now() - pd.Timedelta(days=60)).strftime("%Y-%m-%d")
        trade_entry = trade.get("entry_date") or trade.get("created_at")
        if trade_entry:
            try:
                start_date = (
                    pd.Timestamp(trade_entry) - pd.Timedelta(days=60)
                ).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                pass

        df_hist = self.market_repository.get_symbol_history_raw(
            symbol, start_date=start_date
        )
        if (
            not df_hist.empty
            and len(df_hist) >= MIN_HISTORY_FOR_BOUNCE_BANDIT_DAILY_UPDATE
        ):
            updates = BounceBanditTradeStrategy().get_daily_updates(
                cast(TradeData, trade), df_hist
            )
            if updates:
                context_dict.update(updates)

    def prepare_trade_view(self, trade: dict[str, object]) -> TradeViewData:
        """Transforms a raw trade dict into a strictly typed TradeViewData.

        Args:
            trade: Raw trade record from database.

        Returns:
            TradeViewData: Populated data object for UI representation.
        """
        context_dict = self._parse_context(trade)
        if "indices" not in context_dict and "bucket" in context_dict:
            context_dict["indices"] = context_dict["bucket"]

        self._enrich_bounce_bandit_context(trade, context_dict)

        display_entry, display_exit, days_held = self._extract_dates_and_holding(trade)

        entry_price_val = trade.get("entry_price")
        entry_price = (
            float(str(entry_price_val)) if entry_price_val is not None else 0.0
        )
        current_price_val = trade.get("current_price")
        raw_current_price = (
            float(str(current_price_val)) if current_price_val is not None else 0.0
        )
        current_price = self._resolve_current_price(
            str(trade.get("symbol", "")),
            trade.get("status"),
            raw_current_price,
        )
        initial_size_val = trade.get("initial_size")
        initial_size = (
            float(str(initial_size_val)) if initial_size_val is not None else 0.0
        )
        current_size_val = trade.get("current_size")
        current_size = (
            float(str(current_size_val)) if current_size_val is not None else 0.0
        )
        exit_price_val = trade.get("exit_price")
        exit_price = float(str(exit_price_val)) if exit_price_val is not None else 0.0

        unrealized_pnl = 0.0
        realized_pnl_val = trade.get("realized_pnl")
        realized_pnl = (
            float(str(realized_pnl_val)) if realized_pnl_val is not None else 0.0
        )
        pnl_percentage = 0.0
        is_critical = False
        progress = 0.0

        if trade.get("status") == TradeStatus.ACTIVE:
            (
                unrealized_pnl,
                pnl_percentage,
                is_critical,
                progress,
            ) = self._prepare_active_trade_view_fields(
                trade, context_dict, entry_price, current_price, initial_size
            )
        elif trade.get("status") == TradeStatus.CLOSED:
            (
                realized_pnl,
                pnl_percentage,
            ) = self._prepare_closed_trade_view_fields(
                trade, context_dict, entry_price, exit_price, initial_size
            )

        prices = {
            "entry": entry_price,
            "current": current_price,
            "initial_size": initial_size,
            "current_size": current_size,
            "exit": exit_price,
        }
        pnl_data: dict[str, object] = {
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "pnl_percentage": pnl_percentage,
            "is_critical": is_critical,
            "progress": progress,
        }

        return self._build_view_data(
            ViewBuildParams(
                trade=trade,
                context_dict=context_dict,
                display_entry=display_entry,
                display_exit=display_exit,
                days_held=days_held,
                prices=prices,
                pnl_data=pnl_data,
            )
        )

    def _extract_dates_and_holding(
        self, trade: dict[str, object]
    ) -> tuple[str, str, int]:
        """Extracts display entry/exit dates and trading days holding period."""
        entry_date = trade.get("entry_date")
        exit_date = trade.get("exit_date")
        display_entry = str(entry_date).split(" ")[0] if entry_date else "-"
        display_exit = str(exit_date).split(" ")[0] if exit_date else "-"
        days_held = self._calculate_days_held(
            str(trade.get("symbol", "")), entry_date, exit_date
        )
        return display_entry, display_exit, days_held

    def _resolve_current_price(
        self, symbol: str, status: object, current_price: float
    ) -> float:
        """Resolves the current price of an active trade if set to 0.0."""
        if status == TradeStatus.ACTIVE and current_price == 0.0:
            latest_price = self.market_repository.get_latest_price(symbol)
            if latest_price:
                return latest_price
        return current_price

    def _extract_strategy_version(self, strategy_string: str) -> str | None:
        """Extracts the version tag from the strategy name string."""
        if "0.5" in strategy_string:
            return "0.5"
        if "1.0" in strategy_string:
            return "1.0"
        return None

    def _build_view_data(
        self,
        params: ViewBuildParams,
    ) -> TradeViewData:
        """Assembles and returns a TradeViewData dictionary."""
        entry_date = params.trade.get("entry_date")
        exit_date = params.trade.get("exit_date")
        signal_context_val = params.trade.get("signal_context")
        exit_reason_val = params.trade.get("exit_reason")

        stop_loss_val = params.trade.get("current_stop_loss")
        stop_loss = float(str(stop_loss_val)) if stop_loss_val is not None else 0.0

        target_val = params.trade.get("current_target")
        target = float(str(target_val)) if target_val is not None else 0.0

        budget_val = params.trade.get("budget")
        budget = float(str(budget_val)) if budget_val is not None else 0.0

        return {
            "id": str(params.trade.get("id", "")),
            "symbol": str(params.trade.get("symbol", "")),
            "strategy": str(params.trade.get("strategy", "")),
            "version": self._extract_strategy_version(
                str(params.trade.get("strategy", ""))
            ),
            "status": str(params.trade.get("status", "")),
            "entry_date": str(entry_date) if entry_date else None,
            "exit_date": str(exit_date) if exit_date else None,
            "entry_price": params.prices["entry"],
            "exit_price": params.prices["exit"],
            "current_price": params.prices["current"],
            "initial_size": int(params.prices["initial_size"]),
            "current_size": int(params.prices["current_size"]),
            "current_stop_loss": stop_loss,
            "current_target": target,
            "budget": budget,
            "signal_context": (
                str(signal_context_val) if signal_context_val is not None else None
            ),
            "exit_reason": (
                str(exit_reason_val) if exit_reason_val is not None else None
            ),
            "stop_loss": stop_loss,
            "take_profit": 0.0,
            "display_entry": params.display_entry,
            "display_exit": params.display_exit,
            "days_held": params.days_held,
            "unrealized_pnl": float(str(params.pnl_data["unrealized_pnl"])),
            "realized_pnl": float(str(params.pnl_data["realized_pnl"])),
            "pnl_percentage": float(str(params.pnl_data["pnl_percentage"])),
            "is_critical": bool(params.pnl_data["is_critical"]),
            "progress": float(str(params.pnl_data["progress"])),
            "display_size": params.prices["initial_size"],
            "sparkline": "",
            "max_days": None,
            "context": params.context_dict,
            "tws_status": None,
            "tws_orders": [],
        }

    def generate_sparkline(
        self, dates: list[str], prices: list[float], is_positive: bool
    ) -> str:
        """Generates a minimalistic sparkline chart."""
        color = "#10b981" if is_positive else "#ef4444"
        fill_color = (
            "rgba(16, 185, 129, 0.1)" if is_positive else "rgba(239, 68, 68, 0.1)"
        )

        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=dates,
                y=prices,
                mode="lines",
                line={"color": color, "width": 2, "shape": "spline", "smoothing": 1.3},
                fill="tozeroy",
                fillcolor=fill_color,
                hoverinfo="skip",
            )
        )

        figure.update_layout(
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis={"visible": False},
            yaxis={"visible": False},
            showlegend=False,
            height=50,
            width=120,
        )
        return str(
            figure.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={"displayModeBar": False},
            )
        )

    def generate_donut_chart(
        self, labels: list[str], values: list[float], colors: list[str]
    ) -> str:
        """Generates a donut chart."""
        figure = go.Figure(
            data=[
                go.Pie(
                    labels=labels,
                    values=values,
                    hole=0.8,
                    textinfo="none",
                    hoverinfo="label+percent+value",
                    marker={"colors": colors},
                    sort=False,
                )
            ]
        )

        figure.update_layout(
            margin={"l": 0, "r": 0, "t": 10, "b": 10},
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            height=200,
        )
        return str(
            figure.to_html(
                full_html=False,
                include_plotlyjs="cdn",
                config={"displayModeBar": False},
            )
        )

    def attach_sparklines(
        self,
        trades: list[TradeViewData],
        reference_date: pd.Timestamp | None = None,
    ) -> None:
        """Batch-fetches price history and attaches sparklines to each trade.

        Args:
            trades: List of prepared trade view objects.
            reference_date: Override for the current date (injectable for tests).
                            Defaults to pd.Timestamp.now() when None.
        """
        if not trades:
            return

        today = reference_date if reference_date is not None else pd.Timestamp.now()
        start_date = (today - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
        symbols = list({trade["symbol"] for trade in trades})

        history_dataframe = self.market_repository.get_batch_history_raw(
            symbols, start_date, today.strftime("%Y-%m-%d")
        )

        for trade in trades:
            symbol = trade["symbol"]
            rows = history_dataframe[history_dataframe["symbol"] == symbol].sort_values(
                "date"
            )

            if not rows.empty:
                dates = rows["date"].tolist()
                prices = rows["close"].tolist()

                # Determine color based on PnL first, else price action
                is_positive = prices[-1] >= prices[0]
                if trade["unrealized_pnl"] != 0:
                    is_positive = trade["unrealized_pnl"] > 0

                trade["sparkline"] = self.generate_sparkline(dates, prices, is_positive)

    def get_trades(
        self,
        strategies: Sequence[str] | str | None = None,
        status: str = TradeStatus.ACTIVE,
        exclude_exit_reasons: Sequence[str] | None = None,
    ) -> list[TradeViewData]:
        """
        Fetches and prepares trades, optionally filtering by strategy and exclusion criteria.

        Args:
            strategies: Strategy name or list of names to include.
            status: Trade status to fetch (default: ACTIVE).
            exclude_exit_reasons: List of ExitReasons to exclude (e.g. ['EXPIRED']).
        """

        raw_trades = self.trade_repository.get_by_status([status])
        view_models = []

        for trade in raw_trades:
            # Strategy Filter
            if strategies:
                if not self.is_strategy_match(trade, strategies):
                    continue

            # Exit Reason Exclusion Filter
            if exclude_exit_reasons:
                trade_exit_reason = trade.get("exit_reason")
                if trade_exit_reason and trade_exit_reason in exclude_exit_reasons:
                    continue

            view_models.append(self.prepare_trade_view(trade))

        return view_models

    def _calculate_open_pnl_5d_change(
        self,
        active_trades: list[TradeViewData],
        reference_date: pd.Timestamp | None = None,
    ) -> float:
        """Calculates 5-day Open PnL change for active trades.

        Args:
            active_trades: List of active trade view objects.
            reference_date: Optional timestamp override for reference date.

        Returns:
            float: Aggregated 5-day Open PnL change in base currency.
        """
        if not active_trades:
            return 0.0

        today = reference_date if reference_date is not None else pd.Timestamp.now()
        start_date = (today - pd.Timedelta(days=20)).strftime("%Y-%m-%d")
        symbols = list({trade["symbol"] for trade in active_trades})

        history_dataframe = self.market_repository.get_batch_history_raw(
            symbols, start_date, today.strftime("%Y-%m-%d")
        )
        if history_dataframe.empty:
            return 0.0

        total_5d_change = 0.0

        for trade in active_trades:
            symbol = trade["symbol"]
            rows = history_dataframe[history_dataframe["symbol"] == symbol].sort_values(
                "date"
            )
            if rows.empty:
                total_5d_change += trade["unrealized_pnl"]
                continue

            past_row = (
                rows.iloc[-2]
                if len(rows) >= MIN_ROWS_FOR_PREVIOUS_CANDLE
                else rows.iloc[0]
            )
            past_date_str = str(past_row["date"]).split("T")[0].split(" ")[0]
            past_price = float(past_row["close"])

            entry_date_str = (
                str(trade.get("entry_date") or "").split("T")[0].split(" ")[0]
            )
            initial_size = float(trade.get("initial_size") or 0.0)
            current_price_val = trade.get("current_price")
            current_price = (
                float(str(current_price_val))
                if current_price_val is not None
                else past_price
            )
            direction = str(trade.get("context", {}).get("direction", "long")).lower()

            if entry_date_str and entry_date_str > past_date_str:
                total_5d_change += float(trade.get("unrealized_pnl") or 0.0)
            else:
                if direction == "short":
                    trade_5d_change = (past_price - current_price) * initial_size
                else:
                    trade_5d_change = (current_price - past_price) * initial_size
                total_5d_change += trade_5d_change

        return total_5d_change

    def get_latest_signal_date(self) -> str | None:
        """Fetches the latest updated_at timestamp from the trades table in signals.db."""
        if self.trade_repository:
            try:
                latest_ts = self.trade_repository.get_latest_updated_at()
                if latest_ts:
                    parts = latest_ts.replace("T", " ").split(" ")
                    if len(parts) >= MIN_TIMESTAMP_SPLIT_PARTS:
                        date_part = parts[0]
                        time_part = parts[1].split(".")[0][:5]
                        return f"{date_part} {time_part}"
                    return parts[0]
            except Exception as err:
                logger.debug("Failed to query trades updated_at: %s", err)

        if self.market_repository:
            try:
                latest_market_ts = self.market_repository.get_latest_updated_at()
                if latest_market_ts:
                    return latest_market_ts
            except Exception as err:
                logger.debug("Failed to query market_prices updated_at: %s", err)

        return None

    def get_portfolio_summary(
        self,
        active_trades: list[TradeViewData],
        reference_date: pd.Timestamp | None = None,
        closed_trades: list[TradeViewData] | None = None,
    ) -> dict[str, float | int | str | None]:
        """Calculates summary metrics for active trades."""
        total_invested = sum(
            float(trade.get("entry_price") or 0.0)
            * float(trade.get("initial_size") or 0.0)
            for trade in active_trades
        )
        total_open_pnl = sum(
            float(trade.get("unrealized_pnl") or 0.0) for trade in active_trades
        )
        open_pnl_5d_change = self._calculate_open_pnl_5d_change(
            active_trades, reference_date=reference_date
        )

        try:
            if closed_trades is None:
                closed_trades = self.get_trades(
                    status=TradeStatus.CLOSED,
                    exclude_exit_reasons=[ExitReason.EXPIRED, ExitReason.INVALIDATED],
                )
            pnl_series = pd.Series(
                [float(t.get("realized_pnl") or 0.0) for t in closed_trades]
            )
            win_rate = (
                metrics.calculate_win_rate(pnl_series) if not pnl_series.empty else 0.0
            )
            profit_factor = (
                metrics.calculate_profit_factor(pnl_series)
                if not pnl_series.empty
                else 0.0
            )

            r_list: list[float] = []
            for trade in closed_trades:
                pnl = float(trade.get("realized_pnl") or 0.0)
                entry = float(trade.get("entry_price") or 0.0)
                size = float(trade.get("initial_size") or 0.0)
                stop_val = (
                    trade.get("current_stop_loss") or trade.get("stop_loss") or 0.0
                )
                stop = float(str(stop_val))
                if entry > 0 and stop > 0 and entry != stop:
                    risk = abs(entry - stop) * size
                elif entry > 0 and size > 0:
                    risk = entry * size * 0.05
                else:
                    risk = 1.0
                r_list.append(pnl / risk if risk > 0 else 0.0)

            sqn = (
                metrics.calculate_sqn(pd.Series(r_list))
                if len(r_list) >= MIN_SERIES_LENGTH_FOR_SQN
                else 0.0
            )
        except Exception as err:
            logger.debug(
                "Failed to calculate closed trade metrics for portfolio summary: %s",
                err,
            )
            win_rate = 0.0
            profit_factor = 0.0
            sqn = 0.0

        return {
            "invested": total_invested,
            "open_pnl": total_open_pnl,
            "open_pnl_5d_change": open_pnl_5d_change,
            "equity": 100000.0 + total_open_pnl,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "sqn": sqn,
            "count": len(active_trades),
            "last_updated": self.get_latest_signal_date(),
        }

    def get_closed_summary(
        self, closed_trades: list[TradeViewData]
    ) -> dict[str, float | int]:
        """Calculates summary metrics for closed trades."""
        total_pnl = sum(
            float(trade.get("realized_pnl") or 0.0) for trade in closed_trades
        )
        count = len(closed_trades)
        avg_pnl = total_pnl / count if count > 0 else 0.0

        return {"total_pnl": total_pnl, "count": count, "average_pnl": avg_pnl}

    def group_trades_by_symbol(
        self, trades: list[TradeViewData]
    ) -> list[dict[str, object]]:
        """Groups active trades by symbol."""
        grouped: dict[str, SymbolTradeGroup] = {}
        for trade in trades:
            symbol = str(trade.get("symbol", ""))
            if symbol not in grouped:
                grouped[symbol] = {
                    "symbol": symbol,
                    "total_pnl": 0.0,
                    "total_invested": 0.0,
                    "total_pnl_percentage": 0.0,
                    "variants": [],
                }
            grouped[symbol]["variants"].append(trade)
            unrealized_pnl = float(trade.get("unrealized_pnl") or 0.0)
            entry_price = float(trade.get("entry_price") or 0.0)
            initial_size = float(trade.get("initial_size") or 0.0)
            grouped[symbol]["total_pnl"] += unrealized_pnl
            grouped[symbol]["total_invested"] += entry_price * initial_size

        for group in grouped.values():
            if group["total_invested"] > 0:
                group["total_pnl_percentage"] = (
                    group["total_pnl"] / group["total_invested"]
                ) * 100

        sorted_groups = sorted(grouped.values(), key=lambda x: str(x["symbol"]))
        return [cast(dict[str, object], g) for g in sorted_groups]

    def group_trades_history(
        self, trades: list[TradeViewData]
    ) -> list[dict[str, object]]:
        """Groups closed trades by Symbol + Entry Date."""
        grouped: dict[tuple[str, str], HistoryTradeGroup] = {}

        for trade in trades:
            entry_date_key = str(trade.get("display_entry", "-"))
            if entry_date_key == "-" and trade.get("context", {}).get("setup_date"):
                entry_date_key = str(trade["context"]["setup_date"]).split(" ")[0]

            symbol = str(trade.get("symbol", ""))
            key = (symbol, entry_date_key)
            display_index = str(trade.get("context", {}).get("indices", ""))
            exit_date_str = str(trade.get("exit_date") or "")

            if key not in grouped:
                grouped[key] = {
                    "symbol": symbol,
                    "entry_date": entry_date_key,
                    "max_exit": exit_date_str,
                    "display_index": display_index,
                    "trades": [],
                }

            grouped[key]["trades"].append(trade)
            grouped[key]["max_exit"] = max(grouped[key]["max_exit"], exit_date_str)

        sorted_history = sorted(
            grouped.values(), key=lambda x: str(x["max_exit"]), reverse=True
        )
        return [cast(dict[str, object], h) for h in sorted_history]

    def get_index_stats(
        self, trades: list[TradeViewData]
    ) -> dict[str, dict[str, object]]:
        """Aggregates PnL statistics by Index (SPX, NDX, etc.)."""
        statistics: dict[str, dict[str, object]] = {
            IndexAliases.SPX.value: {
                "name": IndexAliases.SPX.value,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.NDX.value: {
                "name": IndexAliases.NDX.value,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.DOW.value: {
                "name": IndexAliases.DOW.value,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.RUS.value: {
                "name": IndexAliases.RUS.value,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.NO_INDEX.value: {
                "name": IndexAliases.NO_INDEX.value,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
        }

        for trade in trades:
            pnl = float(trade.get("realized_pnl") or 0.0)
            raw_indices = str(trade.get("context", {}).get("indices", ""))
            matched = False

            if IndexAliases.SPX in raw_indices:
                self._update_statistics(statistics[IndexAliases.SPX.value], pnl)
                matched = True
            if IndexAliases.NDX in raw_indices:
                self._update_statistics(statistics[IndexAliases.NDX.value], pnl)
                matched = True
            if IndexAliases.DOW in raw_indices:
                self._update_statistics(statistics[IndexAliases.DOW.value], pnl)
                matched = True
            if IndexAliases.RUS in raw_indices:
                self._update_statistics(statistics[IndexAliases.RUS.value], pnl)
                matched = True

            if not matched:
                self._update_statistics(statistics[IndexAliases.NO_INDEX.value], pnl)

        for item in statistics.values():
            cnt = int(cast(int | float, item["count"]))
            total_pnl = float(cast(int | float, item["pnl"]))
            item["average_pnl"] = total_pnl / cnt if cnt > 0 else 0.0

        return statistics

    def get_weekday_stats(
        self, trades: list[TradeViewData]
    ) -> dict[int, dict[str, object]]:
        """Aggregates PnL statistics by entry weekday.

        Args:
            trades: List of TradeViewData dictionaries representing closed trades.

        Returns:
            dict[int, dict[str, object]]: Dictionary mapping weekday index (0-6)
                to aggregated statistics (name, count, win, loss, pnl, average_pnl).
        """
        weekdays = {
            0: "Monday",
            1: "Tuesday",
            2: "Wednesday",
            3: "Thursday",
            4: "Friday",
            5: "Saturday",
            6: "Sunday",
        }

        statistics: dict[int, dict[str, object]] = {
            i: {
                "name": weekdays[i],
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
                "average_pnl": 0.0,
            }
            for i in range(7)
        }

        for trade in trades:
            pnl = float(trade.get("realized_pnl") or 0.0)
            entry_date_str = trade.get("entry_date")
            if not entry_date_str:
                continue

            try:
                weekday_idx = pd.Timestamp(entry_date_str).weekday()
                if weekday_idx in statistics:
                    self._update_statistics(statistics[weekday_idx], pnl)
            except Exception as e:
                logger.warning(
                    "Could not parse entry date '%s' for weekday analysis: %s",
                    entry_date_str,
                    e,
                )

        for item in statistics.values():
            cnt = int(cast(int | float, item["count"]))
            total_pnl = float(cast(int | float, item["pnl"]))
            item["average_pnl"] = total_pnl / cnt if cnt > 0 else 0.0

        return statistics

    def _update_statistics(
        self, statistics_dict: dict[str, object], pnl: float
    ) -> None:
        """Helper to update a stats dictionary entry.

        Args:
            statistics_dict: The dictionary holding index stats.
            pnl: The realized profit and loss value.
        """
        count_val = int(cast(int | float, statistics_dict.get("count", 0))) + 1
        pnl_val = float(cast(int | float, statistics_dict.get("pnl", 0.0))) + pnl
        statistics_dict["count"] = count_val
        statistics_dict["pnl"] = pnl_val
        if pnl > 0:
            win_val = int(cast(int | float, statistics_dict.get("win", 0))) + 1
            statistics_dict["win"] = win_val
        else:
            loss_val = int(cast(int | float, statistics_dict.get("loss", 0))) + 1
            statistics_dict["loss"] = loss_val

    def get_broker_summary(self) -> dict[str, dict[str, Any]]:
        """Calculates performance metrics from trades_settlement grouped by strategy.

        Returns:
            dict[str, dict[str, Any]]: Metrics map for 'all' and individual strategies.
        """
        if self.broker_repository is None:
            return {}
        settlements = self.broker_repository.get_settlements()

        strategies = ["all", "DipBuyer", "TurnoverTiming", "TwoPercent", "NDXMomentum"]
        accumulators: dict[str, StrategyMetricAccumulator] = {
            strat: StrategyMetricAccumulator() for strat in strategies
        }

        for settlement in settlements:
            trade_group_identifier = settlement.get("trade_group_id") or ""
            parts = trade_group_identifier.split("_")
            raw_strat = parts[1] if len(parts) > 1 else ""
            mapped_strategy = _map_order_strategy_filter(raw_strat)

            mapped_keys = ["all"]
            if mapped_strategy in accumulators and mapped_strategy != "all":
                mapped_keys.append(mapped_strategy)

            net_pnl = float(settlement.get("net_pnl") or 0.0)
            slippage = float(settlement.get("price_diff_slippage") or 0.0)
            commission = float(settlement.get("total_commissions") or 0.0)

            for key in mapped_keys:
                acc = accumulators[key]
                acc.pnl += net_pnl
                acc.fees += commission
                acc.total_count += 1
                acc.slippage_sum += slippage
                if net_pnl > 0:
                    acc.win_count += 1

        result: dict[str, dict[str, Any]] = {}
        for strat in strategies:
            acc = accumulators[strat]
            pnl_text = f"+{acc.pnl:,.0f}" if acc.pnl > 0 else f"{acc.pnl:,.0f}"
            winrate = (
                f"{(acc.win_count / acc.total_count) * 100:.1f}%"
                if acc.total_count > 0
                else "0.0%"
            )
            avg_slippage = (
                acc.slippage_sum / acc.total_count if acc.total_count > 0 else 0.0
            )
            slippage_text = (
                f"+{avg_slippage:.2f}" if avg_slippage > 0 else f"{avg_slippage:.2f}"
            )
            result[strat] = {
                "pnl": acc.pnl,
                "pnlText": pnl_text,
                "winrate": winrate,
                "slippage": slippage_text,
                "fees": acc.fees,
                "win_count": acc.win_count,
                "total_count": acc.total_count,
                "slippage_sum": acc.slippage_sum,
            }

        return result

    def get_broker_settlements(self) -> list[SettlementRecord]:
        """Retrieves closed trade settlements with execution details attached.

        Returns:
            list[SettlementRecord]: List of settlements with attached execution lists.
        """
        if self.broker_repository is None:
            return []
        settlements = self.broker_repository.get_settlements()
        for settlement in settlements:
            trade_group_identifier = settlement.get("trade_group_id") or ""
            executions = self.broker_repository.get_executions_for_trade_group(
                trade_group_identifier
            )
            settlement["executions"] = executions

            parts = trade_group_identifier.split("_")
            settlement["local_trade_id"] = parts[0] if parts else ""
            settlement["symbol"] = parts[-1] if len(parts) > 1 else ""
            settlement["strategy_name"] = (
                "_".join(parts[1:-1])
                if len(parts) > MIN_ROWS_FOR_PREVIOUS_CANDLE
                else (parts[1] if len(parts) > 1 else "")
            )

            # Calculate days_held, quantity and entry_date from executions
            entry_date_str = ""
            days_held = 0
            if executions:
                timestamps = []
                for e in executions:
                    if e.get("executed_at"):
                        ts = pd.Timestamp(e["executed_at"])
                        if ts.tzinfo is not None:
                            ts = ts.tz_convert(None).tz_localize(None)
                        timestamps.append(ts)
                executed_dates = sorted(timestamps)
                if executed_dates:
                    entry_date_str = str(executed_dates[0].date())
                    delta = executed_dates[-1].date() - executed_dates[0].date()
                    days_held = max(0, delta.days)

            settlement["entry_date"] = entry_date_str
            settlement["days_held"] = days_held
            settlement["quantity"] = sum(
                e["qty"] for e in executions if e.get("action") == "BUY"
            )

            entry_price = float(settlement.get("avg_entry_price") or 0.0)
            exit_price = float(settlement.get("avg_exit_price") or 0.0)
            pnl_percentage = (
                ((exit_price - entry_price) / entry_price * 100)
                if entry_price > 0
                else 0.0
            )
            settlement["pnl_percentage"] = pnl_percentage
            settlement["strategy_filter"] = _map_order_strategy_filter(
                settlement["strategy_name"]
            )

        return settlements

    def _check_local_trade_discrepancy(
        self,
        symbol: str,
        local_trades: list[dict[str, object]],
        broker_positions: dict[str, float],
    ) -> dict[str, Any] | None:
        """Checks for discrepancy between local trade record and broker position."""
        symbol_trades = [t for t in local_trades if t.get("symbol") == symbol]
        active_trades = [t for t in symbol_trades if t.get("status") == "ACTIVE"]

        local_status = "CLOSED"
        local_position = 0.0
        if active_trades:
            local_status = "ACTIVE"
            size_val = (
                active_trades[0].get("current_size")
                or active_trades[0].get("initial_size")
                or 0.0
            )
            local_position = float(str(size_val))

        tws_position = float(broker_positions.get(symbol, 0.0))
        trade_obj = symbol_trades[0] if symbol_trades else local_trades[0]
        strategy = _map_order_strategy_filter(self.resolve_strategy(trade_obj))

        if local_status == "ACTIVE" and tws_position == 0.0:
            return {
                "symbol": symbol,
                "local_status": local_status,
                "broker_status": "Keine Position",
                "discrepancy_type": "MISSING_EXECUTION",
                "quantity_difference": local_position,
                "recommended_action": "Trade in Croc-Trader stornieren / bereinigen",
                "strategy": strategy,
            }
        if local_status == "CLOSED" and tws_position > 0.0:
            return {
                "symbol": symbol,
                "local_status": local_status,
                "broker_status": f"Offene Position ({int(tws_position)})",
                "discrepancy_type": "GHOST_POSITION",
                "quantity_difference": tws_position,
                "recommended_action": "Position direkt in TWS schließen",
                "strategy": strategy,
            }
        return None

    def _check_broker_orphan_discrepancy(
        self, symbol: str, tws_position: float
    ) -> dict[str, Any]:
        """Creates a discrepancy record for an orphan broker position not present locally."""
        raw_strat = "Unknown"
        if self.broker_repository:
            query = """
                SELECT o.strategy_name
                FROM executions e
                JOIN orders o ON e.order_id = o.order_id
                WHERE o.symbol = ? AND o.strategy_name NOT IN ('SplitTarget', 'HoldTarget')
                ORDER BY e.executed_at DESC LIMIT 1
            """
            rows = self.broker_repository.fetch_all(query, (symbol,))
            if rows:
                raw_strat = str(rows[0]["strategy_name"])

        strategy = _map_order_strategy_filter(raw_strat)
        return {
            "symbol": symbol,
            "local_status": "CLOSED",
            "broker_status": f"Offene Position ({int(tws_position)})",
            "discrepancy_type": "GHOST_POSITION",
            "quantity_difference": tws_position,
            "recommended_action": "Position direkt in TWS schließen",
            "strategy": strategy,
        }

    def get_reconciliation_discrepancies(self) -> list[dict[str, Any]]:
        """Compares local signals.db trades with TWS executions in trading.db.

        Ignores manual strategies 'SplitTarget' and 'HoldTarget'.

        Returns:
            list[dict[str, Any]]: List of discrepancy records.
        """
        if self.broker_repository is None:
            return []

        local_active = self.trade_repository.get_by_status("ACTIVE")
        local_closed = self.trade_repository.get_by_status("CLOSED")
        combined_local_trades = local_active + local_closed
        local_trades = [
            trade
            for trade in combined_local_trades
            if self.resolve_strategy(trade)
            not in [
                Strategies.HoldTarget,
                Strategies.SplitTarget,
            ]
        ]

        broker_positions = self.broker_repository.get_net_positions_by_symbol()
        discrepancies: list[dict[str, Any]] = []
        checked_symbols: set[str] = set()

        for trade in local_trades:
            symbol = str(trade.get("symbol") or "")
            if not symbol or symbol in checked_symbols:
                continue
            checked_symbols.add(symbol)
            disc = self._check_local_trade_discrepancy(
                symbol, local_trades, broker_positions
            )
            if disc is not None:
                discrepancies.append(disc)

        for symbol, tws_position in broker_positions.items():
            if symbol in checked_symbols:
                continue
            if tws_position > 0.0:
                discrepancies.append(
                    self._check_broker_orphan_discrepancy(symbol, tws_position)
                )

        return discrepancies

    def get_broker_active_trades(self) -> list[ActivePositionRecord]:
        """Loads active trades/positions directly from TWS trading.db executions.

        Returns:
            list[ActivePositionRecord]: Active positions list.
        """
        if self.broker_repository is None:
            return []

        positions = self.broker_repository.get_active_positions()

        for pos in positions:
            symbol = pos["symbol"]
            current_price = pos["current_price"]

            if self.market_repository is not None:
                latest_price = self.market_repository.get_latest_price(symbol)
                if latest_price:
                    current_price = latest_price
                    pos["current_price"] = current_price

            entry_price = pos["entry_price"]
            size = pos["current_size"]
            unrealized_pnl = (current_price - entry_price) * size
            pnl_percentage = (
                ((current_price - entry_price) / entry_price * 100)
                if entry_price > 0
                else 0.0
            )

            pos["unrealized_pnl"] = unrealized_pnl
            pos["pnl_percentage"] = pnl_percentage

            entry_date_str = pos["entry_date"]
            if entry_date_str and entry_date_str != "-":
                try:
                    delta = date.today() - pd.Timestamp(entry_date_str).date()
                    pos["days_held"] = max(0, delta.days)
                except Exception:
                    pos["days_held"] = 0

            pos["strategy_filter"] = _map_order_strategy_filter(pos["strategy"])

        return positions

    def get_broker_active_orders(self) -> list[dict[str, Any]]:
        """Fetches active submitted and presubmitted orders from TWS broker database.

        Returns:
            list[dict[str, Any]]: List of active order dictionaries with strategy_filter.
        """
        if self.broker_repository is None:
            return []

        submitted = [
            dict(r) for r in self.broker_repository.get_orders_by_status("Submitted")
        ]
        presubmitted = [
            dict(r) for r in self.broker_repository.get_orders_by_status("PreSubmitted")
        ]
        raw_orders = submitted + presubmitted

        for order in raw_orders:
            order["strategy_filter"] = _map_order_strategy_filter(
                str(order.get("strategy_name") or "")
            )

        return raw_orders

    def get_broker_error_orders(self) -> list[dict[str, Any]]:
        """Fetches orders with error status from TWS broker database.

        Returns:
            list[dict[str, Any]]: List of error order dictionaries with strategy_filter.
        """
        if self.broker_repository is None:
            return []

        error_orders = [
            dict(r) for r in self.broker_repository.get_orders_by_status("Error")
        ]
        for order in error_orders:
            order["strategy_filter"] = _map_order_strategy_filter(
                str(order.get("strategy_name") or "")
            )

        return error_orders


def _map_order_strategy_filter(strategy_name: str) -> str:
    """Maps a raw broker strategy name to a standardized UI filter key."""
    strategy_lower = strategy_name.lower()
    token_map = (
        ("dip", "DipBuyer"),
        ("turnover", "TurnoverTiming"),
        ("twopercent", "TwoPercent"),
        ("ndx", "NDXMomentum"),
        ("momentum", "NDXMomentum"),
        ("tgim", "TGIM"),
        ("bridge", "BridgeScout"),
        ("scout", "BridgeScout"),
        ("bounce", "BounceBandit"),
        ("bandit", "BounceBandit"),
    )
    for token, label in token_map:
        if token in strategy_lower:
            return label
    return strategy_name or "Unknown"
