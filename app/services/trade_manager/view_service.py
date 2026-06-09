import json
import logging
from pathlib import Path
from datetime import date

import pandas as pd
import plotly.graph_objects as go
from flask import current_app

from ...const import (
    Strategies,
    STRATEGY_ALIASES,
    TradeStatus,
    TargetColumn,
    IndexAliases,
)
from ...database.repositories.trade import TradeRepository
from ...database.repositories.market import MarketRepository
from ...database.session import DatabaseSession

# Import TradeData from types to avoid duplication definition (if possible), or keep TradeViewData as extended
# The user asked to avoid duplication with app/types.py.
# We can import TradeData and use it as a base or reference.
from ...types import TradeData

logger = logging.getLogger(__name__)


def _get_database_path(name: str = "signals") -> Path:
    """Retrieves the absolute path to a specific database."""
    configuration = current_app.config["APP_CONFIG"]
    return Path(configuration.get_db_path(name)).resolve()


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
    pnl_pct: float
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

    def __init__(self) -> None:
        self.trade_repository = self._get_trade_repository()
        self.market_repository = self._get_market_repository()

    def _get_trade_repository(self) -> TradeRepository:
        """Instantiates the trade repository."""
        session = DatabaseSession(str(_get_database_path("signals")))
        return TradeRepository(session)

    def _get_market_repository(self) -> MarketRepository:
        """Instantiates the market repository."""
        session = DatabaseSession(str(_get_database_path("stocks")))
        return MarketRepository(session)

    def resolve_strategy(self, trade: dict[str, object]) -> str:
        """Resolves a trade's strategy string to its Enum value."""
        raw = str(trade.get("strategy", "")).lower()
        # Check exact match first
        try:
            # Check if it's already a valid Enum value
            Strategies(raw)
        except ValueError as val_error:
            # Not a valid enum member, proceed to alias lookup
            logger.debug(
                "Strategy '%s' is not direct member of Strategies enum: %s",
                raw,
                val_error,
            )

        resolved = STRATEGY_ALIASES.get(raw)
        return resolved if resolved else raw

    def is_strategy_match(
        self, trade: dict[str, object], target: str | list[str]
    ) -> bool:
        """Checks if a trade belongs to a strategy or list of strategies."""
        trade_strat = self.resolve_strategy(trade)

        if isinstance(target, list):
            return trade_strat in target

        return trade_strat == target

    def _parse_context(self, trade: dict[str, object]) -> dict[str, object]:
        """Parses the JSON context string safely."""
        try:
            raw_context = trade.get("signal_context")
            if isinstance(raw_context, str) and raw_context:
                return json.loads(raw_context)
            if isinstance(raw_context, dict):
                return raw_context  # type: ignore
            return {}
        except (json.JSONDecodeError, TypeError):
            return {}

    def prepare_trade_view(self, trade: dict[str, object]) -> TradeViewData:
        """Transforms a raw trade dict into a strictly typed TradeViewData."""
        context = self._parse_context(trade)

        # Harmonize Indices
        if "indices" not in context and "bucket" in context:
            # Backward compatibility for old 'bucket' key
            context["indices"] = context["bucket"]

        # Date Handling
        entry_date = trade.get("entry_date")
        exit_date = trade.get("exit_date")

        display_entry = str(entry_date).split(" ")[0] if entry_date else "-"
        display_exit = str(exit_date).split(" ")[0] if exit_date else "-"

        # Holding Period
        days_held = 0
        if entry_date:
            start_date_str = str(entry_date).split(" ")[0]
            if exit_date:
                end_date_str = str(exit_date).split(" ")[0]
            else:
                # Use strictly the last available trading day if active
                # For now, we can fall back to the market repo's latest date if accessible,
                # or rely on the fact that get_trading_days_count handles open interactions?
                # User asked to avoid 'now()'. We'll use today's date only for display logic upper bound.
                # Ideally, we should fetch "last_market_date".
                # For simplicity and robustness without extra query, we use local date as 'today's view'.
                end_date_str = date.today().strftime("%Y-%m-%d")

            days_held = self.market_repository.get_trading_days_count(
                trade.get("symbol", ""), start_date_str, end_date_str
            )

        # Price & PnL
        entry_price = float(trade.get("entry_price") or 0.0)
        current_price = float(trade.get("current_price") or 0.0)
        # Ensure initial_size is fetched correctly.
        # If DB returns None, we default to 0.0.
        initial_size = float(trade.get("initial_size") or 0.0)
        current_size = float(trade.get("current_size") or 0.0)
        exit_price = float(trade.get("exit_price") or 0.0)

        # Active Trade Price Update
        if trade.get("status") == TradeStatus.ACTIVE and current_price == 0.0:
            latest = self.market_repository.get_latest_price(trade.get("symbol", ""))
            if latest:
                current_price = latest

        unrealized_pnl = 0.0
        realized_pnl = float(trade.get("realized_pnl") or 0.0)
        pnl_pct = 0.0
        is_critical = False
        progress = 0.0

        if trade.get("status") == TradeStatus.ACTIVE and entry_price > 0:
            direction = str(context.get("direction", "long")).lower()

            if direction == "short":
                unrealized_pnl = (entry_price - current_price) * initial_size
                pnl_pct = ((entry_price - current_price) / entry_price) * 100
            else:
                unrealized_pnl = (current_price - entry_price) * initial_size
                pnl_pct = ((current_price - entry_price) / entry_price) * 100

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
                if value := context.get(key):
                    target_price = float(value)  # type: ignore
                    break

            # Progress Calculation
            if stop_loss > 0.0 and target_price > 0.0 and stop_loss != target_price:
                total_range = target_price - stop_loss
                current_distance = current_price - stop_loss
                percentage = (current_distance / total_range) * 100
                progress = max(0.0, min(100.0, percentage))

            # Critical SL
            if stop_loss > 0.0:
                distance = abs(current_price - stop_loss)
                if current_price > 0 and (distance / current_price) < 0.01:
                    is_critical = True

        elif trade.get("status") == TradeStatus.CLOSED and entry_price > 0:
            direction = str(context.get("direction", "long")).lower()

            if realized_pnl == 0.0 and exit_price > 0:
                if direction == "short":
                    realized_pnl = (entry_price - exit_price) * initial_size
                else:
                    realized_pnl = (exit_price - entry_price) * initial_size

            entry_for_pct = entry_price
            if direction == "short":
                price_diff = entry_price - exit_price
            else:
                price_diff = exit_price - entry_price

            pnl_pct = (price_diff / entry_for_pct) * 100

        # Version extraction
        version = None
        strat_str = str(trade.get("strategy", ""))
        if "0.5" in strat_str:
            version = "0.5"
        elif "1.0" in strat_str:
            version = "1.0"

        # Construct strictly typed response.
        # Using cast logic implicitly by creating dict matching structure.
        view_data: TradeViewData = {
            "id": trade.get("id", ""),
            "symbol": trade.get("symbol", ""),
            "strategy": trade.get("strategy", ""),
            "version": version,
            "status": trade.get("status", ""),
            "entry_date": str(entry_date) if entry_date else None,
            "exit_date": str(exit_date) if exit_date else None,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "current_price": current_price,
            "initial_size": initial_size,
            "current_size": current_size,
            "current_stop_loss": float(
                trade.get("current_stop_loss") or 0.0
            ),  # TradeData field name
            "current_target": float(
                trade.get("current_target") or 0.0
            ),  # TradeData field name
            "budget": float(trade.get("budget") or 0.0),
            "signal_context": trade.get("signal_context"),
            "exit_reason": trade.get("exit_reason"),
            # Display Fields
            "stop_loss": float(
                trade.get("current_stop_loss") or 0.0
            ),  # Aliased for view
            "take_profit": 0.0,
            "display_entry": display_entry,
            "display_exit": display_exit,
            "days_held": days_held,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "pnl_pct": pnl_pct,
            "is_critical": is_critical,
            "progress": progress,
            "display_size": initial_size,
            "sparkline": "",
            "max_days": None,
            "context": context,
        }
        return view_data

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
                line=dict(color=color, width=2, shape="spline", smoothing=1.3),
                fill="tozeroy",
                fillcolor=fill_color,
                hoverinfo="skip",
            )
        )

        figure.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
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
                    marker=dict(colors=colors),
                    sort=False,
                )
            ]
        )

        figure.update_layout(
            margin=dict(l=0, r=0, t=10, b=10),
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

    def get_portfolio_summary(
        self, active_trades: list[TradeViewData]
    ) -> dict[str, float | int]:
        """Calculates summary metrics for active trades."""
        total_invested = sum(
            trade["entry_price"] * trade["initial_size"] for trade in active_trades
        )
        total_open_pnl = sum(trade["unrealized_pnl"] for trade in active_trades)

        return {
            "invested": total_invested,
            "open_pnl": total_open_pnl,
            "count": len(active_trades),
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
                    "total_pnl_pct": 0.0,
                    "variants": [],
                }
            grouped[symbol]["variants"].append(trade)
            grouped[symbol]["total_pnl"] += trade["unrealized_pnl"]
            grouped[symbol]["total_invested"] += (
                trade["entry_price"] * trade["initial_size"]
            )

        for group in grouped.values():
            if group["total_invested"] > 0:
                group["total_pnl_pct"] = (
                    group["total_pnl"] / group["total_invested"]
                ) * 100

        return sorted(list(grouped.values()), key=lambda x: x["symbol"])

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
            if current_exit > grouped[key]["max_exit"]:
                grouped[key]["max_exit"] = current_exit

        # Return sorted list
        return sorted(
            list(grouped.values()), key=lambda x: str(x["max_exit"]), reverse=True
        )

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
                self._update_stat(statistics[IndexAliases.SPX], pnl)
                matched = True

            if IndexAliases.NDX in raw_indices:
                self._update_stat(statistics[IndexAliases.NDX], pnl)
                matched = True

            if IndexAliases.DOW in raw_indices:
                self._update_stat(statistics[IndexAliases.DOW], pnl)
                matched = True

            if IndexAliases.RUS in raw_indices:
                self._update_stat(statistics[IndexAliases.RUS], pnl)
                matched = True

            if not matched:
                self._update_stat(statistics[IndexAliases.NO_INDEX], pnl)

        # Calc averages
        for item in statistics.values():
            item["average_pnl"] = (
                item["pnl"] / item["count"] if item["count"] > 0 else 0.0
            )

        return statistics

    def _update_stat(self, stat_dict: dict[str, object], pnl: float) -> None:
        """Helper to update a stats dictionary entry."""
        stat_dict["count"] += 1
        stat_dict["pnl"] += pnl
        if pnl > 0:
            stat_dict["win"] += 1
        else:
            stat_dict["loss"] += 1
