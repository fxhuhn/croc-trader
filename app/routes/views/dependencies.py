"""Shared dependencies, repositories, and charting helpers for views."""

from pathlib import Path
from flask import current_app
import plotly.graph_objects as go

from ...database.repositories.signal import SignalRepository
from ...database.session import DatabaseSession
from ...models import BacktestMetrics
from ...services.screener.view_service import ScreenerViewService
from ...services.trade_manager.view_service import TradeViewService

# Re-expose classes used by views sub-modules for clean importing
from ...extensions import cache  # noqa
from ...services.backtester.backtest_results import ResultsPersistence  # noqa
from ...services.backtester.analytics import BacktestAnalytics  # noqa


def _get_database_path(name: str = "signals") -> Path:
    """Retrieves the absolute path to a specific database.

    Args:
        name: The key of the database path within the app configuration.

    Returns:
        Path: Resolved absolute path to the database.
    """
    configuration = current_app.config["APP_CONFIG"]
    return Path(configuration.get_db_path(name)).resolve()


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
    return TradeViewService()


def _get_backtest_database_path() -> Path:
    """Retrieves the absolute path to the backtests database.

    Returns:
        Path: Absolute path pointing to the backtests SQLite database.
    """
    signals_database_path = _get_database_path("signals")
    return signals_database_path.parent / "backtest.db"


def _prepare_backtest_metrics(summary_data: dict[str, object]) -> BacktestMetrics:
    """Maps raw summary dictionary data to a typed BacktestMetrics object.

    Args:
        summary_data: Dictionary containing raw metrics from backtest results.

    Returns:
        BacktestMetrics: Highly-structured populated metrics object.
    """

    # Mapping helper helper for types
    def get_float(key: str) -> float:
        value = summary_data.get(key, 0.0)
        return float(value) if value is not None else 0.0

    def get_int(key: str) -> int:
        value = summary_data.get(key, 0)
        return int(value) if value is not None else 0

    return BacktestMetrics(
        total_trades=get_int("total_trades"),
        win_rate=get_float("win_rate"),
        profit_factor=get_float("profit_factor"),
        net_profit=get_float("net_profit"),
        maximum_drawdown=get_float("maximum_drawdown"),
        sharpe_ratio=get_float("sharpe_ratio"),
        expectancy=get_float("expectancy"),
        system_quality_number=get_float("sqn"),
        kelly_safe=get_float("kelly_safe"),
        strategy_return=get_float("strategy_return"),
        benchmark_return=get_float("benchmark_return"),
        kelly_criterion=get_float("kelly_safe"),
        average_win=0.0,
        average_loss=0.0,
        average_maximum_adverse_excursion=0.0,
        average_maximum_favorable_excursion=0.0,
        risk_of_ruin=0.0,
        kelly_mean=0.0,
        kelly_std=0.0,
        market_exposure_pct=get_float("market_exposure_pct"),
        risk_adjusted_benchmark=0.0,
        exposure_efficiency=0.0,
        return_over_maximum_drawdown=0.0,
        diversification_score=get_float("diversification_score"),
    )


def _prepare_strategy_metrics(
    strategy_list: list[dict[str, object]],
) -> dict[str, BacktestMetrics]:
    """Converts strategy metrics records into BacktestMetrics map.

    Args:
        strategy_list: List of raw strategy metrics records from database.

    Returns:
        dict[str, BacktestMetrics]: Named mapping of strategies to metrics.
    """
    return {
        str(strategy["strategy_name"]): _prepare_backtest_metrics(strategy)
        for strategy in strategy_list
    }


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
                marker=dict(colors=colors),
                sort=False,
            )
        ]
    )

    figure.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
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
