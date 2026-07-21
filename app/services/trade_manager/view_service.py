import json
import logging
from datetime import date
from typing import Any

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
from ...database.repositories.broker import BrokerRepository
from ...database.repositories.market import MarketRepository
from ...database.repositories.trade import TradeRepository
from ...tools import metrics

# Import TradeData from types to avoid duplication definition (if possible), or keep TradeViewData as extended
# The user asked to avoid duplication with app/types.py.
# We can import TradeData and use it as a base or reference.
from ...types import TradeData

logger = logging.getLogger(__name__)


class TradeViewData(TradeData):
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
        self, trade: dict[str, object], target: str | list[str]
    ) -> bool:
        """Checks if a trade belongs to a strategy or list of strategies.

        Args:
            trade: The trade record dictionary.
            target: Strategy name or list of strategy names.

        Returns:
            bool: True if the trade strategy matches target.
        """
        resolved_strategy = self.resolve_strategy(trade)

        if isinstance(target, list):
            return resolved_strategy in target

        return resolved_strategy == target

    def _parse_context(self, trade: dict[str, object]) -> dict[str, object]:
        """Parses the JSON context string safely."""
        try:
            raw_context = trade.get("signal_context")
            if isinstance(raw_context, str) and raw_context:
                return json.loads(raw_context)
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

        stop_loss = float(trade.get("current_stop_loss") or 0.0)
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
            if current_price > 0 and (distance / current_price) < 0.01:
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
        realized_pnl = float(trade.get("realized_pnl") or 0.0)
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

        display_entry, display_exit, days_held = self._extract_dates_and_holding(trade)

        entry_price = float(trade.get("entry_price") or 0.0)
        current_price = self._resolve_current_price(
            str(trade.get("symbol", "")),
            trade.get("status"),
            float(trade.get("current_price") or 0.0),
        )
        initial_size = float(trade.get("initial_size") or 0.0)
        current_size = float(trade.get("current_size") or 0.0)
        exit_price = float(trade.get("exit_price") or 0.0)

        unrealized_pnl = 0.0
        realized_pnl = float(trade.get("realized_pnl") or 0.0)
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
        pnl_data = {
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "pnl_percentage": pnl_percentage,
            "is_critical": is_critical,
            "progress": progress,
        }

        return self._build_view_data(
            trade,
            context_dict,
            display_entry,
            display_exit,
            days_held,
            prices,
            pnl_data,
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
        trade: dict[str, object],
        context_dict: dict[str, object],
        display_entry: str,
        display_exit: str,
        days_held: int,
        prices: dict[str, float],
        pnl_data: dict[str, object],
    ) -> TradeViewData:
        """Assembles and returns a TradeViewData dictionary."""
        entry_date = trade.get("entry_date")
        exit_date = trade.get("exit_date")

        # Query TWS details (not used by strategy trade pages)
        tws_status = None
        tws_orders = []

        return {
            "id": trade.get("id", ""),
            "symbol": trade.get("symbol", ""),
            "strategy": trade.get("strategy", ""),
            "version": self._extract_strategy_version(str(trade.get("strategy", ""))),
            "status": trade.get("status", ""),
            "entry_date": str(entry_date) if entry_date else None,
            "exit_date": str(exit_date) if exit_date else None,
            "entry_price": prices["entry"],
            "exit_price": prices["exit"],
            "current_price": prices["current"],
            "initial_size": prices["initial_size"],
            "current_size": prices["current_size"],
            "current_stop_loss": float(trade.get("current_stop_loss") or 0.0),
            "current_target": float(trade.get("current_target") or 0.0),
            "budget": float(trade.get("budget") or 0.0),
            "signal_context": trade.get("signal_context"),
            "exit_reason": trade.get("exit_reason"),
            "stop_loss": float(trade.get("current_stop_loss") or 0.0),
            "take_profit": 0.0,
            "display_entry": display_entry,
            "display_exit": display_exit,
            "days_held": days_held,
            "unrealized_pnl": float(pnl_data["unrealized_pnl"]),
            "realized_pnl": float(pnl_data["realized_pnl"]),
            "pnl_percentage": float(pnl_data["pnl_percentage"]),
            "is_critical": bool(pnl_data["is_critical"]),
            "progress": float(pnl_data["progress"]),
            "display_size": prices["initial_size"],
            "sparkline": "",
            "max_days": None,
            "context": context_dict,
            "tws_status": tws_status,
            "tws_orders": tws_orders,
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
        return figure.to_html(
            full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
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
        return figure.to_html(
            full_html=False, include_plotlyjs="cdn", config={"displayModeBar": False}
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
        strategies: list[str] | str | None = None,
        status: str = TradeStatus.ACTIVE,
        exclude_exit_reasons: list[str] | None = None,
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

            past_row = rows.iloc[-6] if len(rows) >= 6 else rows.iloc[0]
            past_date_str = str(past_row["date"]).split("T")[0].split(" ")[0]
            past_price = float(past_row["close"])

            entry_date_str = (
                str(trade.get("entry_date") or "").split("T")[0].split(" ")[0]
            )
            initial_size = float(trade.get("initial_size") or 0.0)
            current_price = float(trade.get("current_price") or past_price)
            direction = str(trade.get("context", {}).get("direction", "long")).lower()

            if entry_date_str and entry_date_str > past_date_str:
                total_5d_change += trade["unrealized_pnl"]
            else:
                if direction == "short":
                    trade_5d_change = (past_price - current_price) * initial_size
                else:
                    trade_5d_change = (current_price - past_price) * initial_size
                total_5d_change += trade_5d_change

        return total_5d_change

    def get_latest_signal_date(self) -> str | None:
        """Fetches the latest updated_at timestamp from market_prices in stocks.db."""
        if self.market_repository:
            try:
                latest_market_ts = self.market_repository.get_latest_updated_at()
                if latest_market_ts:
                    return latest_market_ts
            except Exception as err:
                logger.debug("Failed to query market_prices updated_at: %s", err)

        try:
            row = self.trade_repository.fetch_one(
                "SELECT timestamp FROM croc WHERE timestamp IS NOT NULL AND timestamp != '' ORDER BY timestamp DESC LIMIT 1"
            )
            if row and row[0]:
                raw_ts = str(row[0]).strip()
                parts = raw_ts.replace("T", " ").split(" ")
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1].split(".")[0][:5]
                    return f"{date_part} {time_part}"
                return parts[0]
        except Exception as err:
            logger.debug("Failed to query croc timestamp: %s", err)

        try:
            row = self.trade_repository.fetch_one(
                "SELECT created_at FROM trades WHERE created_at IS NOT NULL ORDER BY created_at DESC LIMIT 1"
            )
            if row and row[0]:
                raw_ts = str(row[0]).strip()
                parts = raw_ts.replace("T", " ").split(" ")
                if len(parts) >= 2:
                    date_part = parts[0]
                    time_part = parts[1].split(".")[0][:5]
                    return f"{date_part} {time_part}"
                return parts[0]
        except Exception as err:
            logger.debug("Failed to query trades timestamp: %s", err)

        return None

    def get_portfolio_summary(
        self,
        active_trades: list[TradeViewData],
        reference_date: pd.Timestamp | None = None,
        closed_trades: list[TradeViewData] | None = None,
    ) -> dict[str, float | int | str | None]:
        """Calculates summary metrics for active trades."""
        total_invested = sum(
            trade["entry_price"] * trade["initial_size"] for trade in active_trades
        )
        total_open_pnl = sum(trade["unrealized_pnl"] for trade in active_trades)
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
                stop = float(
                    trade.get("current_stop_loss") or trade.get("stop_loss") or 0.0
                )
                if entry > 0 and stop > 0 and entry != stop:
                    risk = abs(entry - stop) * size
                elif entry > 0 and size > 0:
                    risk = entry * size * 0.05
                else:
                    risk = 1.0
                r_list.append(pnl / risk if risk > 0 else 0.0)

            sqn = metrics.calculate_sqn(pd.Series(r_list)) if len(r_list) >= 2 else 0.0
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
        total_pnl = sum(trade["realized_pnl"] for trade in closed_trades)
        count = len(closed_trades)
        avg_pnl = total_pnl / count if count > 0 else 0.0

        return {"total_pnl": total_pnl, "count": count, "average_pnl": avg_pnl}

    def group_trades_by_symbol(
        self, trades: list[TradeViewData]
    ) -> list[dict[str, object]]:
        """Groups active trades by symbol."""
        grouped: dict[str, dict[str, object]] = {}
        for trade in trades:
            symbol = trade["symbol"]
            if symbol not in grouped:
                grouped[symbol] = {
                    "symbol": symbol,
                    "total_pnl": 0.0,
                    "total_invested": 0.0,
                    "total_pnl_percentage": 0.0,
                    "variants": [],
                }
            grouped[symbol]["variants"].append(trade)
            grouped[symbol]["total_pnl"] += trade["unrealized_pnl"]
            grouped[symbol]["total_invested"] += (
                trade["entry_price"] * trade["initial_size"]
            )

        for group in grouped.values():
            if group["total_invested"] > 0:
                group["total_pnl_percentage"] = (
                    group["total_pnl"] / group["total_invested"]
                ) * 100

        return sorted(grouped.values(), key=lambda x: x["symbol"])

    def group_trades_history(
        self, trades: list[TradeViewData]
    ) -> list[dict[str, object]]:
        """Groups closed trades by Symbol + Entry Date."""
        grouped: dict[tuple[str, str], dict[str, object]] = {}

        for trade in trades:
            # Entry date key fallback
            entry_date_key = trade["display_entry"]
            if entry_date_key == "-" and trade["context"].get("setup_date"):
                entry_date_key = str(trade["context"]["setup_date"]).split(" ")[0]

            key = (trade["symbol"], entry_date_key)

            # Map Index
            display_index = str(trade["context"].get("indices", ""))

            if key not in grouped:
                grouped[key] = {
                    "symbol": trade["symbol"],
                    "entry_date": entry_date_key,
                    "max_exit": trade["exit_date"] or "",
                    "display_index": display_index,
                    "trades": [],
                }

            grouped[key]["trades"].append(trade)

            # Update max exit for sorting
            current_exit = trade["exit_date"] or ""
            grouped[key]["max_exit"] = max(grouped[key]["max_exit"], current_exit)

        # Return sorted list
        return sorted(grouped.values(), key=lambda x: str(x["max_exit"]), reverse=True)

    def get_index_stats(
        self, trades: list[TradeViewData]
    ) -> dict[str, dict[str, object]]:
        """Aggregates PnL statistics by Index (SPX, NDX, etc.)."""
        statistics = {
            IndexAliases.SPX: {
                "name": IndexAliases.SPX,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.NDX: {
                "name": IndexAliases.NDX,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.DOW: {
                "name": IndexAliases.DOW,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.RUS: {
                "name": IndexAliases.RUS,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
            IndexAliases.NO_INDEX: {
                "name": IndexAliases.NO_INDEX,
                "count": 0,
                "win": 0,
                "loss": 0,
                "pnl": 0.0,
            },
        }

        for trade in trades:
            pnl = trade["realized_pnl"]
            raw_indices = str(trade["context"].get("indices", ""))
            matched = False

            # Use IndexAliases Enum values for strict containment checks
            if IndexAliases.SPX in raw_indices:
                self._update_statistics(statistics[IndexAliases.SPX], pnl)
                matched = True

            if IndexAliases.NDX in raw_indices:
                self._update_statistics(statistics[IndexAliases.NDX], pnl)
                matched = True

            if IndexAliases.DOW in raw_indices:
                self._update_statistics(statistics[IndexAliases.DOW], pnl)
                matched = True

            if IndexAliases.RUS in raw_indices:
                self._update_statistics(statistics[IndexAliases.RUS], pnl)
                matched = True

            if not matched:
                self._update_statistics(statistics[IndexAliases.NO_INDEX], pnl)

        # Calc averages
        for item in statistics.values():
            item["average_pnl"] = (
                item["pnl"] / item["count"] if item["count"] > 0 else 0.0
            )

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

        statistics = {
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
            pnl = trade.get("realized_pnl", 0.0)
            entry_date_str = trade.get("entry_date")
            if not entry_date_str:
                continue

            try:
                # Use pandas.Timestamp to safely handle multiple formats
                weekday_idx = pd.Timestamp(entry_date_str).weekday()
                if weekday_idx in statistics:
                    self._update_statistics(statistics[weekday_idx], pnl)
            except Exception as e:
                logger.warning(
                    "Could not parse entry date '%s' for weekday analysis: %s",
                    entry_date_str,
                    e,
                )

        # Calculate average PnL for each weekday
        for item in statistics.values():
            item["average_pnl"] = (
                item["pnl"] / item["count"] if item["count"] > 0 else 0.0
            )

        return statistics

    def _update_statistics(
        self, statistics_dict: dict[str, object], pnl: float
    ) -> None:
        """Helper to update a stats dictionary entry.

        Args:
            statistics_dict: The dictionary holding index stats.
            pnl: The realized profit and loss value.
        """
        statistics_dict["count"] += 1
        statistics_dict["pnl"] += pnl
        if pnl > 0:
            statistics_dict["win"] += 1
        else:
            statistics_dict["loss"] += 1

    def get_broker_summary(self) -> dict[str, dict[str, Any]]:
        """Calculates performance metrics from trades_settlement grouped by strategy.

        Returns:
            dict[str, dict[str, Any]]: Metrics map for 'all' and individual strategies.
        """
        if self.broker_repository is None:
            return {}
        settlements = self.broker_repository.get_settlements()

        strategies = ["all", "DipBuyer", "TurnoverTiming", "TwoPercent", "NDXMomentum"]
        metrics = {
            strat: {
                "pnl": 0.0,
                "pnlText": "0.00",
                "winrate": "0.0%",
                "slippage": "0.00",
                "fees": 0.0,
                "win_count": 0,
                "total_count": 0,
                "slippage_sum": 0.0,
            }
            for strat in strategies
        }

        for settlement in settlements:
            trade_group_identifier = settlement.get("trade_group_id") or ""
            parts = trade_group_identifier.split("_")
            raw_strat = parts[1] if len(parts) > 1 else ""

            mapped_keys = ["all"]
            raw_strat_lower = raw_strat.lower()
            if "dip" in raw_strat_lower:
                mapped_keys.append("DipBuyer")
            elif "turnover" in raw_strat_lower:
                mapped_keys.append("TurnoverTiming")
            elif "twopercent" in raw_strat_lower:
                mapped_keys.append("TwoPercent")
            elif "ndx" in raw_strat_lower or "momentum" in raw_strat_lower:
                mapped_keys.append("NDXMomentum")

            net_pnl = float(settlement.get("net_pnl") or 0.0)
            slippage = float(settlement.get("price_diff_slippage") or 0.0)
            commission = float(settlement.get("total_commissions") or 0.0)

            for key in mapped_keys:
                metrics[key]["pnl"] += net_pnl
                metrics[key]["fees"] += commission
                metrics[key]["total_count"] += 1
                metrics[key]["slippage_sum"] += slippage
                if net_pnl > 0:
                    metrics[key]["win_count"] += 1

        for strat in strategies:
            strat_metrics = metrics[strat]
            total_count = strat_metrics["total_count"]

            pnl_val = strat_metrics["pnl"]
            if pnl_val > 0:
                strat_metrics["pnlText"] = f"+{pnl_val:,.2f}"
            else:
                strat_metrics["pnlText"] = f"{pnl_val:,.2f}"

            if total_count > 0:
                winrate_val = (strat_metrics["win_count"] / total_count) * 100
                strat_metrics["winrate"] = f"{winrate_val:.1f}%"

                avg_slippage = strat_metrics["slippage_sum"] / total_count
                if avg_slippage > 0:
                    strat_metrics["slippage"] = f"+{avg_slippage:.3f}"
                else:
                    strat_metrics["slippage"] = f"{avg_slippage:.3f}"
            else:
                strat_metrics["winrate"] = "0.0%"
                strat_metrics["slippage"] = "0.00"

        return metrics

    def get_broker_settlements(self) -> list[dict[str, Any]]:
        """Retrieves closed trade settlements with execution details attached.

        Returns:
            list[dict[str, Any]]: List of settlements with attached execution lists.
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
                if len(parts) > 2
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

            # Map raw strategy name for the template's data-strategy attribute
            raw_strat_lower = settlement["strategy_name"].lower()
            if "dip" in raw_strat_lower:
                settlement["strategy_filter"] = "DipBuyer"
            elif "turnover" in raw_strat_lower:
                settlement["strategy_filter"] = "TurnoverTiming"
            elif "twopercent" in raw_strat_lower:
                settlement["strategy_filter"] = "TwoPercent"
            elif "ndx" in raw_strat_lower or "momentum" in raw_strat_lower:
                settlement["strategy_filter"] = "NDXMomentum"
            else:
                settlement["strategy_filter"] = settlement["strategy_name"]

        return settlements

    def get_reconciliation_discrepancies(self) -> list[dict[str, Any]]:
        """Compares local signals.db trades with TWS executions in trading.db.

        Ignores manual strategies 'SplitTarget' and 'HoldTarget'.

        Returns:
            list[dict[str, Any]]: List of discrepancy records.
        """
        if self.broker_repository is None:
            return []
        # Fetch active and closed trades from local database
        local_active = self.trade_repository.get_by_status("ACTIVE")
        local_closed = self.trade_repository.get_by_status("CLOSED")

        # Combine and filter out manual strategies
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

        # Fetch TWS broker net position sizes
        broker_positions = self.broker_repository.get_net_positions_by_symbol()

        discrepancies = []
        checked_symbols = set()

        # Check local trades first
        for trade in local_trades:
            symbol = str(trade.get("symbol") or "")
            if symbol in checked_symbols:
                continue
            checked_symbols.add(symbol)

            # Get local status and position size
            symbol_trades = [t for t in local_trades if t.get("symbol") == symbol]
            active_trades = [t for t in symbol_trades if t.get("status") == "ACTIVE"]

            local_status = "CLOSED"
            local_position = 0.0
            if active_trades:
                local_status = "ACTIVE"
                local_position = float(
                    active_trades[0].get("current_size")
                    or active_trades[0].get("initial_size")
                    or 0.0
                )

            # Get TWS position size
            tws_position = float(broker_positions.get(symbol, 0.0))

            # Resolve strategy for local trade
            trade_obj = symbol_trades[0] if symbol_trades else trade
            raw_strat = self.resolve_strategy(trade_obj)
            raw_strat_lower = str(raw_strat).lower()
            if "dip" in raw_strat_lower:
                strategy = "DipBuyer"
            elif "turnover" in raw_strat_lower:
                strategy = "TurnoverTiming"
            elif "twopercent" in raw_strat_lower:
                strategy = "TwoPercent"
            elif "ndx" in raw_strat_lower or "momentum" in raw_strat_lower:
                strategy = "NDXMomentum"
            else:
                strategy = raw_strat

            # Mismatch detection
            if local_status == "ACTIVE" and tws_position == 0.0:
                discrepancies.append(
                    {
                        "symbol": symbol,
                        "local_status": local_status,
                        "broker_status": "Keine Position",
                        "discrepancy_type": "MISSING_EXECUTION",
                        "quantity_difference": local_position,
                        "recommended_action": "Trade in Croc-Trader stornieren / bereinigen",
                        "strategy": strategy,
                    }
                )
            elif local_status == "CLOSED" and tws_position > 0.0:
                discrepancies.append(
                    {
                        "symbol": symbol,
                        "local_status": local_status,
                        "broker_status": f"Offene Position ({int(tws_position)})",
                        "discrepancy_type": "GHOST_POSITION",
                        "quantity_difference": tws_position,
                        "recommended_action": "Position direkt in TWS schließen",
                        "strategy": strategy,
                    }
                )

        # Check TWS symbols that had no local trade checked
        for symbol, tws_position in broker_positions.items():
            if symbol in checked_symbols:
                continue
            if tws_position > 0.0:
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
                        raw_strat = rows[0]["strategy_name"]

                raw_strat_lower = str(raw_strat).lower()
                if "dip" in raw_strat_lower:
                    strategy = "DipBuyer"
                elif "turnover" in raw_strat_lower:
                    strategy = "TurnoverTiming"
                elif "twopercent" in raw_strat_lower:
                    strategy = "TwoPercent"
                elif "ndx" in raw_strat_lower or "momentum" in raw_strat_lower:
                    strategy = "NDXMomentum"
                else:
                    strategy = raw_strat

                discrepancies.append(
                    {
                        "symbol": symbol,
                        "local_status": "CLOSED",
                        "broker_status": f"Offene Position ({int(tws_position)})",
                        "discrepancy_type": "GHOST_POSITION",
                        "quantity_difference": tws_position,
                        "recommended_action": "Position direkt in TWS schließen",
                        "strategy": strategy,
                    }
                )

        return discrepancies

    def get_broker_active_trades(self) -> list[dict[str, Any]]:
        """Loads active trades/positions directly from TWS trading.db executions.

        Returns:
            list[dict[str, Any]]: Active positions list.
        """
        if self.broker_repository is None:
            return []

        positions = self.broker_repository.get_active_positions()

        for pos in positions:
            symbol = pos["symbol"]
            current_price = pos["current_price"]

            # Fetch current price from market database if available
            if self.market_repository is not None:
                latest_price = self.market_repository.get_latest_price(symbol)
                if latest_price:
                    current_price = latest_price
                    pos["current_price"] = current_price

            # Recalculate unrealized PnL: (current - entry) * size
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

            # Calculate days held
            entry_date_str = pos["entry_date"]
            if entry_date_str and entry_date_str != "-":
                try:
                    delta = date.today() - pd.Timestamp(entry_date_str).date()
                    pos["days_held"] = max(0, delta.days)
                except Exception:
                    pos["days_held"] = 0

            # Map raw strategy name for the template's strategy_filter attribute
            raw_strat_lower = pos["strategy"].lower()
            if "dip" in raw_strat_lower:
                pos["strategy_filter"] = "DipBuyer"
            elif "turnover" in raw_strat_lower:
                pos["strategy_filter"] = "TurnoverTiming"
            elif "twopercent" in raw_strat_lower:
                pos["strategy_filter"] = "TwoPercent"
            elif "ndx" in raw_strat_lower or "momentum" in raw_strat_lower:
                pos["strategy_filter"] = "NDXMomentum"
            else:
                pos["strategy_filter"] = pos["strategy"]

        return positions
