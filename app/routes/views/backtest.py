"""Routes and views for backtest visualization and charting dashboard."""

from flask import render_template, request
import pandas as pd

from .blueprint import views_bp
from .dependencies import (
    _get_database_path,
    _get_backtest_database_path,
    _prepare_backtest_metrics,
    _prepare_strategy_metrics,
    ResultsPersistence,
    BacktestAnalytics,
    BacktestMetrics,
)


@views_bp.route("/backtest", methods=["GET"])
def view_backtest_dashboard() -> str:
    """Displays the backtest dashboard by retrieving pre-calculated results.

    Returns:
        str: Rendered HTML dashboard template or error message.
    """
    # 1. Configuration & Paths
    backtest_database_path = _get_backtest_database_path()
    market_database_path = _get_database_path("stocks")

    # 2. Results Persistence
    persistence = ResultsPersistence(str(backtest_database_path))
    run_identifier = (
        request.args.get("run_id", type=int) or persistence.get_latest_run_id()
    )

    if not run_identifier:
        return (
            "No backtest results found. Please run the backtester first. "
            f"DB Path: {backtest_database_path}"
        )

    # 3. Data Retrieval
    run_data = persistence.get_run_results(run_identifier)
    if not run_data:
        return f"Results for Run ID {run_identifier} not found."

    summary_data = run_data["summary"]

    # 4. Metric Preparation (Object Mapping)
    main_metrics = _prepare_backtest_metrics(summary_data)
    strategy_metrics_map = _prepare_strategy_metrics(run_data["strategies"])
    portfolio_metrics = run_data["portfolio"]

    # 5. Chart Data Preparation
    equity_dataframe = pd.DataFrame(run_data.get("equity_curves", []))
    regime_dataframe = pd.DataFrame(run_data.get("regime_data", []))
    exposure_dataframe = pd.DataFrame(run_data.get("exposure_data", []))

    # Identify Dates for Benchmarks
    start_date_str = str(summary_data["start_date"])
    end_date_str = str(summary_data["end_date"])

    # 6. Chart Generation
    analytics = BacktestAnalytics(
        str(backtest_database_path), str(market_database_path)
    )
    dashboard_charts = _generate_dashboard_charts(
        analytics=analytics,
        main_metrics=main_metrics,
        equity_dataframe=equity_dataframe,
        regime_dataframe=regime_dataframe,
        exposure_dataframe=exposure_dataframe,
        start_date=start_date_str,
        end_date=end_date_str,
    )

    # 7. Rendering
    trade_lists = analytics.get_trade_lists()

    safety_impact = run_data.get("safety_impact", {})
    final_equity = float(safety_impact.get("final_equity", 100000.0))
    kelly_metrics_view = {
        "net_profit": final_equity - 100000.0,
        "total_return": (final_equity - 100000.0) / 100000.0,
        "max_drawdown": float(
            portfolio_metrics.get("leveraged_max_drawdown", 0.0)
            if portfolio_metrics
            else 0.0
        ),
    }

    return render_template(
        "backtest_dashboard.html",
        run_id=run_identifier,
        metrics=main_metrics,
        kelly_metrics=kelly_metrics_view,
        strategy_metrics=strategy_metrics_map,
        wfa_results=run_data.get("wfa", []),
        stress_results=run_data.get("stress", {}),
        funnel_data=run_data.get("funnel", []),
        quality_data=run_data.get("quality", []),
        start_date=start_date_str,
        end_date=end_date_str,
        **dashboard_charts,
        recent_trades=trade_lists["recent"],
        top_trades=trade_lists["top"],
        worst_trades=trade_lists["worst"],
    )


def _generate_dashboard_charts(
    analytics: BacktestAnalytics,
    main_metrics: BacktestMetrics,
    equity_dataframe: pd.DataFrame,
    regime_dataframe: pd.DataFrame,
    exposure_dataframe: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> dict[str, str]:
    """Orchestrates chart generation for the backtest dashboard.

    Args:
        analytics: Analytics engine for benchmark fetching.
        main_metrics: BacktestMetrics model.
        equity_dataframe: Time series of equity curves.
        regime_dataframe: Time series of regime/VIX data.
        exposure_dataframe: Time series of strategy utilization.
        start_date: Backtest start date.
        end_date: Backtest end date.

    Returns:
        dict[str, str]: Map of template variable names to HTML chart strings.
    """
    from ...services.backtester.charts import (
        generate_backtest_charts,
        generate_profit_factor_gauge,
        generate_win_rate_gauge,
        generate_sqn_gauge,
        generate_regime_overlay_chart,
        generate_price_of_safety_chart,
        generate_exposure_heatmap,
        generate_risk_reward_scatter,
    )

    # 1. Benchmarks
    initial_capital = 100000.0
    spy_dataframe = analytics.fetch_benchmark_data(
        "SPY", start_date, end_date, initial_capital=initial_capital
    )
    qqq_dataframe = analytics.fetch_benchmark_data(
        "QQQ", start_date, end_date, initial_capital=initial_capital
    )

    # 2. Split Equity Curves
    base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Base"]
    if base_equity.empty:
        base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Portfolio"]
    if base_equity.empty:
        base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Kelly"]
    if base_equity.empty:
        base_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Safety"]
    kelly_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Kelly"]
    safety_equity = equity_dataframe[equity_dataframe["strategy_name"] == "Safety"]

    # 3. Generate individual charts
    chart_equity_base, chart_drawdown_base = (
        "<div>No Data</div>",
        "<div>No Data</div>",
    )
    if not base_equity.empty:
        chart_equity_base, chart_drawdown_base = generate_backtest_charts(
            base_equity["date"],
            base_equity["equity"],
            base_equity["drawdown_pct"],
            benchmark_df=spy_dataframe,
            id_prefix="base",
        )

    chart_equity_kelly, chart_drawdown_kelly = (
        "<div>No Data</div>",
        "<div>No Data</div>",
    )
    if not kelly_equity.empty:
        chart_equity_kelly, chart_drawdown_kelly = generate_backtest_charts(
            kelly_equity["date"],
            kelly_equity["equity"],
            kelly_equity["drawdown_pct"],
            benchmark_df=spy_dataframe,
            id_prefix="kelly",
        )

    # Specialized Charts
    regime_input = regime_dataframe.rename(columns={"vix_close": "vix"})

    # Merge 'equity' from base_equity into regime_input for overlay chart
    if not base_equity.empty:
        regime_input = pd.merge(
            regime_input,
            base_equity[["date", "equity"]],
            on="date",
            how="left",
        )
    else:
        regime_input["equity"] = 0.0

    # Rename exposure columns for exposure heatmap compatibility
    exposure_pivot = exposure_dataframe.pivot(
        index="date", columns="strategy_name", values="exposure_value"
    ).reset_index()
    exposure_pivot.columns = [
        f"exposure_{column}" if column != "date" else column
        for column in exposure_pivot.columns
    ]

    # Map keys expected strictly by the frontend template engine
    return {
        "chart_equity": chart_equity_base,
        "chart_underwater": chart_drawdown_base,
        "chart_equity_kelly": chart_equity_kelly,
        "chart_underwater_kelly": chart_drawdown_kelly,
        "chart_regime": generate_regime_overlay_chart(regime_input),
        "chart_safety": generate_price_of_safety_chart(
            kelly_equity,
            base_equity,
            safety_equity,
            spy_dataframe=spy_dataframe,
            qqq_dataframe=qqq_dataframe,
        ),
        "chart_exposure": generate_exposure_heatmap(exposure_pivot),
        "chart_risk": generate_risk_reward_scatter(safety_equity),
        "chart_pf": generate_profit_factor_gauge(main_metrics.profit_factor),
        "chart_wr": generate_win_rate_gauge(main_metrics.win_rate * 100),
        "chart_sqn": generate_sqn_gauge(main_metrics.system_quality_number),
    }
