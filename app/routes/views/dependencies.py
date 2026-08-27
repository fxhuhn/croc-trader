"""Shared dependencies, repositories, and charting helpers for views."""

from pathlib import Path

import plotly.graph_objects as go
from flask import current_app

from ...database.repositories.broker import BrokerRepository
from ...database.repositories.market import MarketRepository
from ...database.repositories.signal import SignalRepository
from ...database.repositories.trade import TradeRepository
from ...database.session import DatabaseSession
from ...extensions import cache
from ...services.screener.view_service import ScreenerViewService
from ...services.trade_manager.view_service import TradeViewService

__all__ = [
    "BrokerRepository",
    "MarketRepository",
    "ScreenerViewService",
    "SignalRepository",
    "TradeRepository",
    "TradeViewService",
    "_get_database_path",
    "_get_screener_view_service",
    "_get_signal_repository",
    "_get_trade_view_service",
    "cache",
    "generate_donut_chart",
    "generate_sparkline",
]


def _get_database_path(name: str = "signals") -> Path:
    """Retrieves the absolute path to a specific database.

    Args:
        name: The key of the database path within the app configuration.

    Returns:
        Path: Resolved absolute path to the database (or :memory: for in-memory DBs).
    """
    configuration = current_app.config["APP_CONFIG"]
    db_path = configuration.get_db_path(name)
    if db_path == ":memory:":
        return Path(":memory:")
    return Path(db_path).resolve()


def _get_signal_repository() -> SignalRepository:
    """Instantiates the signal repository database connection.

    Returns:
        SignalRepository: Instantiated signal repository instance.
    """
    session = DatabaseSession(str(_get_database_path("signals")))
    return SignalRepository(session)


def _get_screener_view_service() -> ScreenerViewService:
    """Instantiates the screener view service.

    Returns:
        ScreenerViewService: Instantiated screener view service.
    """
    return ScreenerViewService(_get_signal_repository())


def _get_trade_view_service() -> TradeViewService:
    """Instantiates the trade view service.

    Returns:
        TradeViewService: Instantiated trade view service.
    """
    signals_session = DatabaseSession(str(_get_database_path("signals")))
    stocks_session = DatabaseSession(str(_get_database_path("stocks")))
    trading_session = DatabaseSession(
        str(_get_database_path("trading")), read_only=True
    )

    return TradeViewService(
        trade_repository=TradeRepository(signals_session),
        market_repository=MarketRepository(stocks_session),
        broker_repository=BrokerRepository(trading_session),
    )


def generate_sparkline(
    dates: list[object], prices: list[float], is_positive: bool
) -> str:
    """Generates a minimalistic Plotly sparkline chart in HTML.

    Args:
        dates: Timestamps or generic date labels.
        prices: Historical price list matching the dates.
        is_positive: Visual indicator flag to use positive color palette.

    Returns:
        str: Raw Plotly HTML string representing the sparkline chart.
    """
    color = "#10b981" if is_positive else "#ef4444"
    fill_color = "rgba(16, 185, 129, 0.1)" if is_positive else "rgba(239, 68, 68, 0.1)"

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
    labels: list[str], values: list[float], colors: list[str]
) -> str:
    """Generates a professional donut chart in HTML for allocation mapping.

    Args:
        labels: Asset or strategy identifier categories.
        values: Corresponding investment allocations.
        colors: Color hex mapping for visual matching.

    Returns:
        str: Raw Plotly HTML string representing the donut chart.
    """
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
        margin={"l": 0, "r": 0, "t": 0, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        height=180,
    )
    return str(
        figure.to_html(
            full_html=False,
            include_plotlyjs="cdn",
            config={"displayModeBar": False},
        )
    )
