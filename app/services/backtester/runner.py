# python3 -m app.services.backtester.runner --start 2023-01-01 --end 2025-12-31

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from ...database.repositories.market_data_provider import MarketDataProvider
from ...database.repositories.trade import TradeRepository
from ...database.session import DatabaseSession
from .analytics import (
    BacktestAnalytics,
    BacktestMetrics,
    PortfolioMetrics,
    NoiseTester,
    WalkForwardAnalyzer,
    PerformancePeriods,
    TradeQualityAnalyzer,
    DiversificationAnalyzer,
)
from .engine import BacktestEngine
from .backtest_results import (
    ResultsPersistence,
    SafetyEvent,
    SimulationImpact,
)
from app.services.portfolio.simulation import CapacitySimulator

# Configure Logging to file, so console is clean for Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("backtest.log"),
    ],
)
logger = logging.getLogger(__name__)


def _display_performance_table(
    console: Console, performance_metrics: BacktestMetrics, benchmark_return: float
) -> None:
    """Displays the main strategy performance table.

    Args:
        console: Rich console object.
        performance_metrics: Metrics object from analysis.
        benchmark_return: Calculated benchmark return value.
    """
    table = Table(title="Strategy Performance")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row("Total Trades", str(performance_metrics.total_trades))
    table.add_row("Win Rate", f"{performance_metrics.win_rate * 100:.1f}%")
    table.add_row("Profit Factor", f"{performance_metrics.profit_factor:.2f}")
    table.add_row("Net Profit", f"${performance_metrics.net_profit:,.2f}")
    table.add_row("Expectancy", f"${performance_metrics.expectancy:.2f}")
    table.add_row("Kelly (Mean)", f"{performance_metrics.kelly_mean:.2f}")
    table.add_row(
        "Kelly (Safe - 25th)",
        f"[bold green]{performance_metrics.kelly_safe:.2f}[/bold green]",
    )
    table.add_row("SQN", f"{performance_metrics.system_quality_number:.2f}")
    table.add_row("Max Drawdown", f"{performance_metrics.maximum_drawdown * 100:.1f}%")

    console.print(table)

    # Benchmark
    console.print(
        Panel(
            f"Strategy: [green]{performance_metrics.strategy_return * 100:.1f}%[/green] "
            f"vs SPY: [yellow]{benchmark_return * 100:.1f}%[/yellow]",
            title="Benchmark Comparison",
        )
    )


def _display_strategy_breakdown(
    console: Console, strategy_performance_map: dict[str, BacktestMetrics]
) -> None:
    """Displays comparison table between different strategies.

    Args:
        console: Rich console object.
        strategy_performance_map: Dictionary mapping strategy names to metrics.
    """
    if not strategy_performance_map:
        return

    console.print("\n[bold blue]--- Strategy Breakdown ---[/bold blue]")

    comparison_table = Table(title="Strategy Comparison")
    comparison_table.add_column("Metric", style="cyan")

    for strategy_name in strategy_performance_map.keys():
        comparison_table.add_column(strategy_name, style="magenta")

    # Define metrics to show
    rows = [
        ("Total Trades", lambda metrics: str(metrics.total_trades)),
        ("Win Rate", lambda metrics: f"{metrics.win_rate * 100:.1f}%"),
        ("Profit Factor", lambda metrics: f"{metrics.profit_factor:.2f}"),
        ("Net Profit", lambda metrics: f"${metrics.net_profit:,.2f}"),
        ("Kelly (Safe)", lambda metrics: f"{metrics.kelly_safe:.2f}"),
        ("SQN", lambda metrics: f"{metrics.system_quality_number:.2f}"),
        ("Max Drawdown", lambda metrics: f"{metrics.maximum_drawdown * 100:.1f}%"),
        ("Exposure", lambda metrics: f"{metrics.market_exposure_pct * 100:.1f}%"),
        (
            "Risk-Adj Ret",
            lambda metrics: f"{metrics.risk_adjusted_benchmark * 100:.1f}% (Ben)",
        ),
        ("Eff (Exp)", lambda metrics: f"{metrics.exposure_efficiency:.2f}"),
        ("Ret/DD", lambda metrics: f"{metrics.return_over_maximum_drawdown:.2f}"),
    ]

    for label, accessor in rows:
        row_values = [label]
        for strategy_metrics in strategy_performance_map.values():
            row_values.append(accessor(strategy_metrics))
        comparison_table.add_row(*row_values)

    console.print(comparison_table)


def _display_safety_switch_steps(console: Console, events: list[SafetyEvent]) -> None:
    """Displays the Safety Switch Trigger Log.

    Args:
        console: Rich console object.
        events: List of safety switch event dictionaries.
    """
    if not events:
        return

    console.print("\n[bold blue]--- Safety Switch Trigger Log ---[/bold blue]")

    trigger_table = Table(title="Safety Switch Events")
    trigger_table.add_column("Start Date", style="dim")
    trigger_table.add_column("End Date", style="dim")
    trigger_table.add_column("Trigger Reason", style="bold yellow")
    trigger_table.add_column("Duration (Days)", style="cyan")
    trigger_table.add_column("Saved Profit", style="green")

    for event in events:
        trigger_table.add_row(
            pd.to_datetime(event["start_date"]).strftime("%Y-%m-%d"),
            (
                pd.to_datetime(event["end_date"]).strftime("%Y-%m-%d")
                if event.get("end_date")
                else "Active"
            ),
            event["reason"],
            str(event["days"]),
            f"${event['saved_profit']:,.2f}",
        )

    console.print(trigger_table)


def _display_impact_analysis(console: Console, simulation: SimulationImpact) -> None:
    """Displays the Safety Switch Impact Analysis Scorecard."""
    console.print("\n[bold blue]--- Safety Switch Impact Analysis ---[/bold blue]")

    table = Table(title="Impact Analysis (Saved Loss vs Opp. Cost)")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="magenta")

    table.add_row(
        "Saved Loss (Drawdown Avoided)",
        f"[bold green]${simulation['saved_loss']:,.2f}[/bold green]",
    )
    table.add_row(
        "Opportunity Cost (Missed Profit)",
        f"[bold red]${simulation['opportunity_cost']:,.2f}[/bold red]",
    )

    table.add_row(
        "Margin Interest Paid",
        f"[bold yellow]${simulation.get('margin_interest_paid', 0.0):,.2f}[/bold yellow]",
    )

    efficiency_style = "bold green" if simulation["net_efficiency"] > 0 else "bold red"
    table.add_row(
        "Net Efficiency",
        f"[{efficiency_style}]${simulation['net_efficiency']:,.2f}[/{efficiency_style}]",
    )

    console.print(table)


def _display_switch_tournament(
    console: Console, tournament_results: list[dict[str, object]]
) -> None:
    """Displays the Switch Optimization Tournament results.

    Args:
        console: Rich console object.
        tournament_results: List of tournament result dictionaries.
    """
    if not tournament_results:
        return

    console.print(
        "\n[bold blue]--- Switch Optimization Tournament (Winner Takes All) ---[/bold blue]"
    )

    table = Table(title="Safety Logic Comparison (Tournament)")
    table.add_column("Trigger Logic", style="cyan")
    table.add_column("Total Return", style="bold yellow")
    table.add_column("Max Drawdown", style="bold red")
    table.add_column("Net Efficiency*", style="magenta")
    table.add_column("Switch Grade", style="bold green")

    for result in tournament_results:
        efficiency_style = "bold green" if result["net_efficiency"] > 0 else "bold red"
        efficiency_text = f"${result['net_efficiency']:,.0f}"
        if result["net_efficiency"] > 0:
            efficiency_text += " (Gain)"
        elif result["net_efficiency"] < 0:
            efficiency_text += " (Cost)"

        table.add_row(
            str(result["logic"]),
            f"${result['total_return']:,.0f}",
            f"{result['max_drawdown'] * 100:.1f}%",
            f"[{efficiency_style}]{efficiency_text}[/{efficiency_style}]",
            str(result["grade"]),
        )

    console.print(table)
    console.print(
        "[dim]*Net Efficiency = (Profit with Switch) - (Profit Baseline)[/dim]"
    )


def _display_regime_comparison(
    console: Console, regime_comparison: dict[str, dict[str, object]]
) -> None:
    """Displays the Regime-Specific Strategy Comparison Matrix."""
    if not regime_comparison:
        return

    console.print(
        "\n[bold blue]--- Regime-Specific Performance Comparison ---[/bold blue]"
    )

    table = Table(title="Strategy Performance by Market Regime (Raw vs Safe)")
    table.add_column("Strategy", style="cyan")
    table.add_column("Regime", style="dim")
    table.add_column("Net Profit (Kelly-Safe Impact)", style="magenta")
    table.add_column("Sample Days", style="dim")

    for strategy_name, regimes in regime_comparison.items():
        first = True
        for regime, strategy_metrics in regimes.items():
            table.add_row(
                strategy_name if first else "",
                regime,
                f"${strategy_metrics['Return']:,.2f}",
                str(int(strategy_metrics["Sample_Count"])),
            )
            first = False
        table.add_section()

    console.print(table)


def _display_walk_forward_insights(
    console: Console, walk_forward_dataframe: pd.DataFrame
) -> None:
    """Displays Actionable WFA Insights and Recommendations."""
    if walk_forward_dataframe is None or walk_forward_dataframe.empty:
        return

    console.print(
        "\n[bold blue]--- WFA Actionable Insights & Recommendations ---[/bold blue]"
    )

    table = Table(title="WFA Stability & Allocation Recommendations")
    table.add_column("Window", style="dim")
    table.add_column("Degradation %", style="yellow")
    table.add_column("OOS PF", style="magenta")
    table.add_column("Recommendation", style="bold")

    insights = walk_forward_dataframe.to_dict("records")
    for row in insights:
        recommendation = row["Recommendation"]
        style = "white"
        if "STABLE" in recommendation:
            style = "bold green"
        elif "WARNING" in recommendation:
            style = "bold yellow"
        elif "CRITICAL" in recommendation:
            style = "bold red"

        table.add_row(
            row["Window"],
            f"{row['Degradation'] * 100:.1f}%",
            f"{row['OOS_PF']:.2f}",
            f"[{style}]{recommendation}[/{style}]",
        )

    console.print(table)


def _display_portfolio_kelly(
    console: Console, portfolio_metrics: PortfolioMetrics
) -> None:
    """Displays portfolio optimization results.

    Args:
        console: Rich console object.
        portfolio_metrics: Portfolio optimization metrics.
    """
    if (
        portfolio_metrics.safe_kelly_25 == 0
        and portfolio_metrics.combined_mean_kelly == 0
    ):
        return

    portfolio_kelly_table = Table(title="Portfolio Optimization (Simulations)")
    portfolio_kelly_table.add_column("Metric", style="cyan")
    portfolio_kelly_table.add_column("Constrained (100% Cap)", style="bold green")
    portfolio_kelly_table.add_column("Unconstrained", style="yellow")
    portfolio_kelly_table.add_column("Description", style="dim")

    portfolio_kelly_table.add_row(
        "Safe Kelly (25th)",
        f"{portfolio_metrics.safe_kelly_25:.2f}%",
        f"{portfolio_metrics.safe_kelly_25:.2f}%",
        "Conservative risk % per trade",
    )
    portfolio_kelly_table.add_row(
        "Max Concurrent Trades",
        f"{portfolio_metrics.max_concurrent_trades} Peak ({portfolio_metrics.max_concurrent_trades_days}d)",
        f"{portfolio_metrics.percentile_95_concurrent_trades:.1f} (95th%)",
        "Global overlapping trades",
    )

    for strategy_name, max_count in portfolio_metrics.max_trades_per_strategy.items():
        days_at_max = portfolio_metrics.max_trades_per_strategy_days.get(
            strategy_name, 0
        )
        p95_strat = portfolio_metrics.percentile_95_trades_per_strategy.get(
            strategy_name, 0.0
        )
        portfolio_kelly_table.add_row(
            f"  • {strategy_name}",
            f"{max_count} Peak ({days_at_max}d)",
            f"{p95_strat:.1f} (95th%)",
            "Strategy-specific concurrency",
        )

    portfolio_kelly_table.add_row(
        "Suggested Multiplier",
        f"x{portfolio_metrics.suggested_multiplier:.2f}",
        f"x{portfolio_metrics.uncapped_multiplier:.2f}",
        "Scaling Factor for positions",
    )

    portfolio_kelly_table.add_row(
        "Max Total Exposure",
        f"{portfolio_metrics.max_total_exposure:.1f}%",
        f"{portfolio_metrics.uncapped_max_total_exposure:.1f}%",
        "Peak Projected Gross Exposure",
    )

    portfolio_kelly_table.add_row(
        "Leveraged MaxDD",
        f"{portfolio_metrics.leveraged_max_drawdown * 100:.1f}%",
        f"{portfolio_metrics.uncapped_leveraged_max_drawdown * 100:.1f}%",
        "Projected Portfolio Drawdown",
    )

    console.print(portfolio_kelly_table)


def _display_period_analysis(console: Console, period_metrics: pd.DataFrame):
    if period_metrics is None or period_metrics.empty:
        return

    table = Table(title="Rolling Performance Windows")
    table.add_column("Period", style="cyan")
    table.add_column("Avg Sharpe", style="magenta")
    table.add_column("Avg DD", style="red")
    table.add_column("Win Rate Stability", style="green")

    # Aggregating for display
    windows = sorted(period_metrics["window_trades"].unique())
    for window in windows:
        subset = period_metrics[period_metrics["window_trades"] == window]
        if subset.empty:
            continue

        table.add_row(
            f"{window}-Trade Rolling",
            f"{subset['sharpe_proxy'].mean():.2f}",
            f"{subset['max_drawdown_pnl'].mean():.2f}",
            f"{subset['win_rate'].std() * 100:.1f}%",
        )

    console.print(table)


def _display_quality_distribution(console: Console, quality_scores: pd.DataFrame):
    if quality_scores is None or quality_scores.empty:
        return

    table = Table(title="Trade Quality Distribution")
    table.add_column("Grade", style="bold")
    table.add_column("Count", style="cyan")
    table.add_column("Avg Profit", style="green")
    table.add_column("Common Weakness", style="yellow")

    for grade in ["A+", "A", "B", "C", "D", "F"]:
        subset = quality_scores[quality_scores["grade"] == grade]
        if subset.empty:
            continue

        weakness_mode = subset["weakest_link"].mode()
        weakness_str = weakness_mode[0] if not weakness_mode.empty else "N/A"

        table.add_row(
            grade, str(len(subset)), f"${subset['profit'].mean():,.2f}", weakness_str
        )

    console.print(table)

    if "weakest_link" in quality_scores.columns:
        modes = quality_scores["weakest_link"].mode()
        if not modes.empty:
            console.print(
                Panel(
                    f"[yellow]Actionable Insight:[/yellow] "
                    f"{modes[0]} is your weakest area. "
                    f"Focus on improving this component to boost overall performance."
                )
            )


def _display_diversification(console: Console, score: float, matrices: dict):
    console.print("\n[bold blue]--- Diversification Analysis ---[/bold blue]")
    console.print(
        f"Diversification Score: [bold magenta]{score:.1f}/100[/bold magenta]"
    )
    # Could print matrix here if needed, but console space is limited.


def _display_portfolio_funnel(
    console: Console, funnel_data: list[dict[str, object]]
) -> None:
    """Displays the Advanced Portfolio Allocation (Funnel Logic).

    Args:
        console: Rich console object.
        funnel_data: List of funnel allocation dictionaries.
    """
    if not funnel_data:
        return

    console.print(
        "\n[bold blue]--- AUTOMATED PORTFOLIO ALLOCATION (KELLY OPTIMIZATION) ---[/bold blue]"
    )

    table = Table(title="Portfolio Allocation (Funnel Logic)")
    table.add_column("Strategy Name", style="cyan")
    table.add_column("Kelly", style="bold yellow")
    table.add_column("Raw Share*", style="magenta")
    table.add_column("Status", style="bold")
    table.add_column("FINAL ALLOCATION", style="bold green")

    total_kelly = 0.0
    total_raw_share = 0.0
    total_final_allocation = 0.0

    for item in funnel_data:
        status_style = "green" if item["status"] == "ACTIVE" else "red"
        status_text = str(item["status"])
        if item["reason"]:
            status_text += f" ({item['reason']})"

        table.add_row(
            str(item["name"]),
            f"{item['kelly']:.2f}",
            f"{item['raw_share']:.1f}%",
            f"[{status_style}]{status_text}[/{status_style}]",
            f"{item['final_allocation']:.1f}%",
        )
        total_kelly += float(item["kelly"])
        total_raw_share += float(item["raw_share"])
        total_final_allocation += float(item["final_allocation"])

    table.add_section()
    table.add_row(
        "TOTAL",
        f"{total_kelly:.2f}",
        f"{total_raw_share:.0f}%",
        "",
        f"{total_final_allocation:.0f}%",
    )

    console.print(table)
    console.print(
        "[dim]*Raw Share = Theoretical allocation without quality filters.[/dim]"
    )


def _display_capacity_ratio_analysis(
    console: Console, portfolio_metrics: PortfolioMetrics
) -> None:
    """Displays the Per-Strategy Capacity Ratio table."""
    if not portfolio_metrics.strategy_capacity_ratios:
        return

    console.print(
        "\n[bold blue]--- Per-Strategy Capacity Ratio Analysis ---[/bold blue]"
    )

    table = Table(title="Strategy Capacity Analysis")
    table.add_column("Strategy", style="cyan")
    table.add_column("Peak", style="magenta")
    table.add_column("95th", style="magenta")
    table.add_column("Ratio", style="bold yellow")
    table.add_column("Multiplier", style="bold green")

    for strat, ratio in portfolio_metrics.strategy_capacity_ratios.items():
        peak = portfolio_metrics.max_trades_per_strategy.get(strat, 0)
        p95 = portfolio_metrics.percentile_95_trades_per_strategy.get(strat, 0.0)
        multiplier = np.sqrt(ratio)

        table.add_row(
            strat, str(peak), f"{p95:.1f}", f"{ratio:.2f}", f"x{multiplier:.2f}"
        )

    console.print(table)


def _display_allocation_comparison(
    console: Console, funnel_data: list[dict[str, object]]
) -> None:
    """Displays the Comparison between Current and New (95th) allocation."""
    if not funnel_data:
        return

    console.print(
        "\n[bold blue]--- ALLOCATION COMPARISON (CURRENT vs NEW 95TH) ---[/bold blue]"
    )

    table = Table(title="Portfolio Allocation Comparison")
    table.add_column("Strategy", style="cyan")
    table.add_column("Current", style="magenta")
    table.add_column("New (95th)", style="bold green")
    table.add_column("Change", style="yellow")
    table.add_column("Reasoning", style="dim")

    total_current = 0.0
    total_new = 0.0

    for item in funnel_data:
        current = item["final_allocation"]
        new_p95 = item.get("final_allocation_p95", 0.0)
        delta = new_p95 - current

        reasoning = "N/A"
        if item["status"] == "ACTIVE":
            if delta > 2.0:
                reasoning = "High capacity waste"
            elif delta < -2.0:
                reasoning = "High efficiency"
            else:
                reasoning = "Balanced"
        else:
            reasoning = item.get("reason", "Rejected")

        table.add_row(
            str(item["name"]),
            f"{current:.1f}%",
            f"{new_p95:.1f}%",
            f"{delta:+.1f}pp",
            reasoning,
        )
        total_current += current
        total_new += new_p95

    table.add_section()
    table.add_row("TOTAL", f"{total_current:.0f}%", f"{total_new:.0f}%", "-", "-")

    console.print(table)


def _setup_resources(
    args: argparse.Namespace, console: Console
) -> tuple[MarketDataProvider, TradeRepository]:
    """Sets up market and trade repositories.

    Args:
        args: Parsed command line arguments.
        console: Rich console for output.

    Returns:
        tuple: (MarketDataProvider, TradeRepository)
    """
    root_dir = Path.cwd()
    stocks_db_path = root_dir / args.stocks_db
    backtest_database_path = root_dir / args.backtest_db

    if not stocks_db_path.exists():
        console.print(f"[red]Error: Stocks DB not found at {stocks_db_path}[/red]")
        sys.exit(1)

    console.print("[dim]Connecting to Market Data...[/dim]")
    stocks_session = DatabaseSession(str(stocks_db_path))
    market_provider = MarketDataProvider(stocks_session)

    console.print("[dim]Initializing Backtest Database...[/dim]")
    # Do NOT delete WAL/SHM files if preserving the DB. SQLite handles recovery automatically.
    # Deleting them can cause "database disk image is malformed" errors.

    backtest_session = DatabaseSession(str(backtest_database_path))
    trade_repository = TradeRepository(backtest_session)
    trade_repository.init_schema()

    # NEW: Clear current trades/logs so that each Run ID has its own clean set of trades
    # but the Run Summary history is preserved.
    trade_repository.clear_trades()

    return market_provider, trade_repository


def _run_extended_analytics(
    backtest_db_path: str, stocks_db_path: str, console: Console
) -> tuple[BacktestAnalytics, dict]:
    """Runs the full analytical pipeline after backtest execution.

    Args:
        backtest_db_path: Path to backtest results.
        stocks_db_path: Path to market data.
        console: Rich console object for debug printing.

    Returns:
        tuple: (analytics_engine, results_dictionary)
    """
    analytics = BacktestAnalytics(backtest_db_path, stocks_db_path)
    trades_dataframe = analytics.loader.fetch_closed_trades()

    # --- NUCLEAR DEDUPLICATION ---
    # We enforce strict consistency here to resolve discrepancies (e.g. 2197 vs 824).
    if not trades_dataframe.empty:
        initial_count = len(trades_dataframe)

        # 1. Ensure primary date columns are consistent datetime objects for analytics
        trades_dataframe["entry_date"] = pd.to_datetime(trades_dataframe["entry_date"])
        trades_dataframe["exit_date"] = pd.to_datetime(trades_dataframe["exit_date"])

        # 2. Use normalized strings ONLY for identifying duplicates
        # This handles mixed precision/formats in the underlying data.
        normalized_keys = pd.DataFrame(
            {
                "s": trades_dataframe["symbol"].str.strip(),
                "st": trades_dataframe["strategy"].str.strip(),
                "en": trades_dataframe["entry_date"].dt.strftime("%Y-%m-%d"),
                "ex": trades_dataframe["exit_date"].dt.strftime("%Y-%m-%d"),
            }
        )

        # Apply the mask
        is_duplicate = normalized_keys.duplicated(keep="first")
        trades_dataframe = trades_dataframe[~is_duplicate].copy()

        # 3. GHOST TRADE ELIMINATION
        # Remove trades that were never execution-active (0 PnL and non-profit exit reasons)
        # We use a small epsilon for PnL to handle float precision.
        active_mask = abs(trades_dataframe["realized_pnl"]) > 1e-6

        ghost_count = len(trades_dataframe) - active_mask.sum()
        if ghost_count > 0:
            console.print(
                f"[yellow]Ghost Trade Elimination: Removing {ghost_count} inactive/expired signals.[/yellow]"
            )
            trades_dataframe = trades_dataframe[active_mask].copy()

        final_count = len(trades_dataframe)
        if initial_count != final_count:
            console.print(
                f"[bold green]Data Alignment Complete:[/bold green] "
                f"Using {final_count} physically executed trades for all reports."
            )
        else:
            console.print(
                f"[green]Data Integrity Verified: {final_count} unique trades found.[/green]"
            )

    if trades_dataframe.empty:
        console.print(
            "[yellow]No closed trades found. Skipping extended analytics.[/yellow]"
        )
        return analytics, {}

    performance_metrics = analytics.run_analysis(trades_dataframe=trades_dataframe)
    strategy_performance_map = analytics.run_strategy_analysis(
        trades_dataframe=trades_dataframe
    )

    # Simulations
    portfolio_kelly_metrics = None
    try:
        portfolio_kelly_metrics = analytics.calculate_portfolio_kelly(iterations=1000)
    except Exception as portfolio_error:
        logger.warning("Portfolio simulation failed: %s", portfolio_error)

    walk_forward_results = None
    try:
        walk_forward_engine = WalkForwardAnalyzer(trades_dataframe)
        walk_forward_results = walk_forward_engine.run_analysis()
    except Exception as walk_forward_error:
        logger.warning("Walk Forward analysis failed: %s", walk_forward_error)

    stress_test_results = None
    try:
        noise_engine = NoiseTester(analytics, trades_dataframe)
        stress_test_results = noise_engine.run_stress_test(n_simulations=50)
    except Exception as stress_error:
        logger.warning("Stress Test failed: %s", stress_error)

    # --- New Analytics ---
    period_df = None
    try:
        periods = PerformancePeriods()
        period_df = periods.calculate_rolling_metrics(trades_dataframe)
    except Exception as e:
        logger.warning("Periods failed: %s", e)

    quality_df = None
    try:
        quality_analyzer = TradeQualityAnalyzer()
        if not trades_dataframe.empty:
            # Vectorized scoring (Refactoring Order 3)
            quality_df = quality_analyzer.score_dataframe(trades_dataframe)
            # Add profit column for display compatibility
            quality_df["profit"] = quality_df["realized_pnl"]
    except Exception as e:
        logger.warning("Quality Analysis failed: %s", e)

    div_score = 0.0
    try:
        div = DiversificationAnalyzer()
        if not trades_dataframe.empty:
            strat_map = {k: v for k, v in trades_dataframe.groupby("strategy")}
            corr = div.calculate_strategy_correlations(strat_map)
            div_score = div.calculate_diversification_score(corr)
    except Exception as e:
        logger.warning("Diversification failed: %s", e)

    return analytics, {
        "performance": performance_metrics,
        "strategies": strategy_performance_map,
        "portfolio": portfolio_kelly_metrics,
        "walk_forward": walk_forward_results,
        "stress": stress_test_results,
        "periods": period_df,
        "quality": quality_df,
        "diversification": div_score,
        "trades_dataframe": trades_dataframe,
    }


def _load_portfolio_configuration() -> dict | None:
    """Loads portfolio.yaml if available in the runner's directory."""
    import yaml

    config_path = Path(__file__).parent / "portfolio.yaml"

    if not config_path.exists():
        return None

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            logger.info("Loaded custom portfolio configuration from %s", config_path)
            return config
    except Exception as e:
        logger.error("Failed to load portfolio.yaml: %s", e)
        return None


def _export_portfolio_configuration(
    funnel_data: list[dict],
    portfolio_metrics: PortfolioMetrics,
    equity: float = 100000.0,
) -> None:
    """Generates portfolio.yaml with both Static and Dynamic recommendations.

    Args:
        funnel_data: Data from the portfolio funnel.
        portfolio_metrics: Global portfolio metrics.
        equity: Initial capital for calculations.
    """
    import yaml

    config_path = Path(__file__).parent / "portfolio.yaml"

    output = {"equity": int(equity), "strategies": []}

    if not portfolio_metrics:
        logger.warning(
            "No portfolio metrics found. Using default max_trades=1 for YAML export."
        )

    for item in funnel_data:
        if item["status"] != "ACTIVE":
            continue

        name = item["name"]

        # Static Parameters
        alloc_static = item["final_allocation"]
        quota_static = equity * (alloc_static / 100.0)

        # Dynamic (P95) Parameters
        alloc_p95 = item.get("final_allocation_p95", alloc_static)
        quota_p95 = equity * (alloc_p95 / 100.0)

        # Max Trades from backtest observed concurrency
        max_trades = (
            portfolio_metrics.max_trades_per_strategy.get(name, 1)
            if portfolio_metrics
            else 1
        )
        if max_trades < 1:
            max_trades = 1

        # 95th Percentile Max Trades
        max_trades_p95 = (
            portfolio_metrics.percentile_95_trades_per_strategy.get(name, 1.0)
            if portfolio_metrics
            else 1.0
        )
        if max_trades_p95 < 1.0:
            max_trades_p95 = 1.0

        # Budget per position
        qty_static = quota_static / max_trades
        qty_p95 = quota_p95 / max_trades_p95

        # Combined Entry (matches user's requested style)
        output["strategies"].append(
            {
                name: [
                    {"allocation": float(round(alloc_static, 1))},
                    {"allocation_95th": float(round(alloc_p95, 1))},
                    {"quota": int(round(quota_static, 0))},
                    {"quota_95th": int(round(quota_p95, 0))},
                    {"max_trades": int(max_trades)},
                    {"max_trades_95th": float(round(max_trades_p95, 1))},
                    {"quantity": int(round(qty_static, 0))},
                    {"quantity_95th": int(round(qty_p95, 0))},
                ]
            }
        )

    try:
        with open(config_path, "w") as f:
            yaml.safe_dump(output, f, sort_keys=False, default_flow_style=False)
        logger.info(
            "Recommended portfolio allocation (Enhanced) saved to %s", config_path
        )
    except Exception as e:
        logger.error("Failed to export portfolio.yaml: %s", e)


def _display_audit_reports(
    console: Console,
    analytics: BacktestAnalytics,
    strategy_performance_map: dict,
    safety_simulation: dict,
    regime_dataframe: pd.DataFrame,
) -> None:
    """Displays Regime and Safety Switch Audit tables.

    Args:
        console: Rich console object.
        analytics: Backtest analytics orchestrator.
        strategy_performance_map: Map of strategy metrics.
        safety_simulation: Kelly/Safety simulation results.
        regime_dataframe: Market regime data.
    """
    console.print("\n[bold blue]--- Regime-Specific Performance Audit ---[/bold blue]")

    baseline_simulation = analytics.run_constrained_kelly_simulation(
        regime_dataframe=None
    )

    if baseline_simulation and baseline_simulation.get("multipliers"):
        comparison_safety_table = Table(title="Kelly Scaling & Safety Switch Impact")
        comparison_safety_table.add_column("Strategy", style="cyan")
        comparison_safety_table.add_column("Multiplier", style="bold yellow")
        comparison_safety_table.add_column("Metric", style="dim")
        comparison_safety_table.add_column("Baseline (Raw)", style="white")
        comparison_safety_table.add_column("Safety Switch (Safe)", style="bold green")

        strategy_multipliers = safety_simulation["multipliers"]
        for strategy, multiplier in strategy_multipliers.items():
            comparison_safety_table.add_row(strategy, f"x{multiplier:.2f}", "", "", "")

        comparison_safety_table.add_section()
        comparison_safety_table.add_row(
            "OVERALL",
            "",
            "Final Equity",
            f"${baseline_simulation['theoretical_equity']:,.2f}",
            f"${safety_simulation['final_equity']:,.2f}",
        )
        comparison_safety_table.add_row(
            "",
            "",
            "Saved Loss",
            "-",
            f"[green]${safety_simulation['saved_loss']:,.2f}[/green]",
        )
        comparison_safety_table.add_row(
            "",
            "",
            "Opp. Cost",
            "-",
            f"[red]${safety_simulation['opportunity_cost']:,.2f}[/red]",
        )
        console.print(comparison_safety_table)

    _display_impact_analysis(console, safety_simulation)
    regime_comparison_matrix = analytics.run_regime_comparison(
        regime_dataframe, strategy_performance_map
    )
    _display_regime_comparison(console, regime_comparison_matrix)
    _display_safety_switch_steps(console, safety_simulation.get("events", []))

    with console.status("[bold green]Running Safety Switch Tournament...[/bold green]"):
        tournament_results = analytics.run_safety_tournament(
            regime_dataframe, strategy_performance_map
        )
        _display_switch_tournament(console, tournament_results)


def _display_capacity_simulation(
    console: Console,
    trades_dataframe: pd.DataFrame,
    base_kelly: float = 0.39,
    p95_concurrency: float = 10.0,
) -> None:
    """Runs and displays capacity simulation results."""
    console.print(
        "\n[bold cyan]--- Capacity Simulation (Dynamic Sizing) ---[/bold cyan]"
    )

    simulator = CapacitySimulator(initial_capital=100000.0, n_simulations=100)

    try:
        results_df = simulator.run(
            trades_dataframe,
            base_kelly=base_kelly / 100.0 if base_kelly > 1.0 else base_kelly,
            p95_concurrency=p95_concurrency,
        )

        if results_df.empty:
            console.print("[yellow]Simulation returned no results.[/yellow]")
            return

        table = Table(title="Projected Performance (Monte Carlo)")
        table.add_column("Scenario", style="white")
        table.add_column("Median Return", style="green")
        table.add_column("Max Drawdown (95%)", style="red")
        table.add_column("Avg Commission", style="yellow")
        table.add_column("Avg Margin Int.", style="yellow")
        table.add_column("Delta vs Static", style="cyan")

        # Find baseline (Static) for delta calc
        baseline_row = results_df[results_df["Scenario"] == "A_Static"]
        base_ret = (
            baseline_row.iloc[0]["Median Return"] if not baseline_row.empty else 0.0
        )

        for _, row in results_df.iterrows():
            ret_pct = row["Median Return"] * 100
            dd_pct = row["Max Drawdown (95th Worst)"] * 100
            comm = row["Avg Commission"]
            interest = row["Avg Margin Interest"]

            delta_ret = (row["Median Return"] - base_ret) * 100
            delta_str = (
                f"{delta_ret:+.2f}%" if row["Scenario"] != "A_Static" else "Baseline"
            )

            table.add_row(
                row["Scenario"],
                f"{ret_pct:.2f}%",
                f"{dd_pct:.2f}%",
                f"${comm:,.0f}",
                f"${interest:,.0f}",
                delta_str,
            )

        console.print(table)

    except Exception as e:
        logger.error(f"Simulation Failed: {e}", exc_info=True)
        console.print(f"[bold red]Simulation Failed: {e}[/bold red]")


def main() -> None:
    """Main execution entry point for the backtester runner."""
    parser = argparse.ArgumentParser(description="Multi-Strategy Backtest Runner")
    parser.add_argument(
        "--start", type=str, required=True, help="Start Date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYY-MM-DD)")
    parser.add_argument(
        "--stocks-db",
        type=str,
        default="data/stocks.db",
        help="Path to Real Market Data DB",
    )
    parser.add_argument(
        "--backtest-db",
        type=str,
        default="data/backtest.db",
        help="Path to Backtest Result DB",
    )
    args = parser.parse_args()

    console = Console()
    console.print(f"[bold]Starting Backtest[/bold] from {args.start} to {args.end}")

    market_provider, trade_repository = _setup_resources(args, console)

    # Load custom sizing if available
    portfolio_config = _load_portfolio_configuration()

    engine = BacktestEngine(
        start_date=args.start,
        end_date=args.end,
        market_provider=market_provider,
        trade_repository=trade_repository,
        console=console,
        portfolio_config=portfolio_config,
    )

    try:
        engine.run()
        console.print("[bold green]Backtest Execution Completed![/bold green]")

        # 4. Starting Analytics
        console.print("\n[bold blue]--- Starting Analytics ---[/bold blue]")
        analytics, results = _run_extended_analytics(
            args.backtest_db, args.stocks_db, console
        )

        # Persistence
        regime_dataframe = analytics.fetch_regime_data()
        safety_simulation = analytics.run_constrained_kelly_simulation(
            regime_dataframe=regime_dataframe, include_history=True
        )

        equity_df, regime_df, exposure_df = analytics.get_granular_persistence_data(
            safety_simulation
        )

        funnel_data = analytics.calculate_portfolio_funnel(
            results["strategies"], portfolio_metrics=results["portfolio"]
        )

        console.print(f"[bold red]DEBUG: Saving to DB: {args.backtest_db}[/bold red]")
        if not equity_df.empty:
            console.print(
                f"[bold red]DEBUG: Equity DF Columns: {equity_df.columns}[/bold red]"
            )
            console.print(
                f"[bold red]DEBUG: Equity Strategies: {equity_df['strategy_name'].unique()}[/bold red]"
            )
        else:
            console.print("[bold red]DEBUG: Equity DF is EMPTY![/bold red]")

        persistence = ResultsPersistence(db_path=args.backtest_db)
        run_id = persistence.save_run(
            start_date=args.start,
            end_date=args.end,
            metrics=results["performance"],
            strategy_metrics=results["strategies"],
            portfolio_kelly=results["portfolio"],
            walk_forward_df=results["walk_forward"],
            stress_test_results=results["stress"],
            daily_equity_curves=equity_df,
            regime_data=regime_df,
            strategy_exposures=exposure_df,
            safety_impact=safety_simulation,
            funnel_data=funnel_data,
            quality_df=results.get("quality"),
            diversification_score=results.get("diversification", 0.0),
        )

        # 6. Print Reports
        console.print(f"\n[bold green]Backtest Complete! Run ID: {run_id}[/bold green]")
        _display_performance_table(
            console, results["performance"], results["performance"].benchmark_return
        )
        _display_strategy_breakdown(console, results["strategies"])

        if results["performance"].total_trades > 0:
            robust_table = Table(title="Monte Carlo Robustness (Bootstrap Runs)")
            robust_table.add_column("Metric", style="cyan")
            robust_table.add_column("Value", style="bold yellow")
            robust_table.add_row(
                "Risk of Ruin", f"{results['performance'].risk_of_ruin * 100:.1f}%"
            )
            console.print(robust_table)

        if results["portfolio"]:
            _display_portfolio_kelly(console, results["portfolio"])
            _display_capacity_ratio_analysis(console, results["portfolio"])

        _display_portfolio_funnel(console, funnel_data)
        _display_allocation_comparison(console, funnel_data)

        _display_audit_reports(
            console,
            analytics,
            results["strategies"],
            safety_simulation,
            regime_dataframe,
        )

        if "trades_dataframe" in results and not results["trades_dataframe"].empty:
            # We use Kelly Mean to allow capacity to hit leverage limits
            kelly_target = results["performance"].kelly_mean
            p95_concurrency = (
                results["portfolio"].percentile_95_concurrent_trades
                if results["portfolio"]
                else 10.0
            )

            _display_capacity_simulation(
                console,
                results["trades_dataframe"],
                base_kelly=kelly_target,
                p95_concurrency=p95_concurrency,
            )

        if results.get("periods") is not None:
            _display_period_analysis(console, results["periods"])

        if results.get("quality") is not None:
            _display_quality_distribution(console, results["quality"])

        if results.get("diversification"):
            _display_diversification(console, results["diversification"], {})

        if results["walk_forward"] is not None:
            _display_walk_forward_insights(console, results["walk_forward"])

        # 7. Export Configuration
        _export_portfolio_configuration(
            funnel_data=funnel_data,
            portfolio_metrics=results["portfolio"],
            equity=100000.0,  # Default equity used in allocator
        )

    except Exception as fatal_error:
        console.print(f"[bold red]Backtest Failed:[/bold red] {fatal_error}")
        logger.exception("Backtest Fatal Error")
        sys.exit(1)


if __name__ == "__main__":
    main()
