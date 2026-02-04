import sys
import argparse
from pathlib import Path
import logging

from rich.console import Console

from ...database.session import DatabaseSession
from ...database.repositories.trade import TradeRepository
from ...database.repositories.market import MarketRepository
from ...database.repositories.market_data_provider import MarketDataProvider
from .engine import BacktestEngine

# Configure Logging to file, so console is clean for Rich
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("backtest.log"), 
        # logging.StreamHandler() # Disable stream handler to not mess up UI
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Run Backtest for DipBuyer Strategy")
    parser.add_argument("--start", type=str, required=True, help="Start Date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End Date (YYYY-MM-DD)")
    parser.add_argument("--stocks-db", type=str, default="data/stocks.db", help="Path to Real Market Data DB")
    parser.add_argument("--backtest-db", type=str, default="data/backtest.db", help="Path to Backtest Result DB")
    
    args = parser.parse_args()
    
    console = Console()
    console.print(f"[bold]Starting Backtest[/bold] from {args.start} to {args.end}")
    
    # 1. Paths
    root_dir = Path.cwd()
    stocks_db_path = root_dir / args.stocks_db
    backtest_db_path = root_dir / args.backtest_db
    
    if not stocks_db_path.exists():
        console.print(f"[red]Error: Stocks DB not found at {stocks_db_path}[/red]")
        sys.exit(1)
        
    # 2. Setup Resources
    # Real Market Data
    console.print("[dim]Connecting to Market Data...[/dim]")
    stocks_session = DatabaseSession(str(stocks_db_path))
    
    # MarketDataProvider expects the Session directly
    market_provider = MarketDataProvider(stocks_session)
    
    # We might need MarketRepository for other things if needed, but Engine uses Provider.
    # market_repo = MarketRepository(stocks_session)
    
    # Backtest DB (Execution Phase)
    console.print("[dim]Initializing Backtest Database...[/dim]")
    if backtest_db_path.exists():
        console.print("[yellow]Removing existing backtest DB...[/yellow]")
        backtest_db_path.unlink()
        
    backtest_session = DatabaseSession(str(backtest_db_path))
    trade_repo = TradeRepository(backtest_session)
    
    # Init Schema
    trade_repo.init_schema()
    
    # 3. Run Engine
    engine = BacktestEngine(
        start_date=args.start,
        end_date=args.end,
        market_provider=market_provider,
        trade_repo=trade_repo,
        console=console
    )
    
    try:
        engine.run()
        console.print("[bold green]Backtest Execution Completed![/bold green]")
        
        # 4. Trigger Analysis (Next Step)
        console.print("\n[bold blue]--- Starting Analytics ---[/bold blue]")
        from .analytics import BacktestAnalytics
        
        analytics = BacktestAnalytics(str(backtest_db_path), str(stocks_db_path))
        metrics = analytics.run_analysis()
        mc_results = analytics.run_monte_carlo()
        
        # 5. Print Report
        from rich.table import Table
        from rich.panel import Panel
        
        # Performance Table
        table = Table(title="Strategy Performance")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Total Trades", str(metrics.total_trades))
        table.add_row("Win Rate", f"{metrics.win_rate*100:.1f}%")
        table.add_row("Profit Factor", f"{metrics.profit_factor:.2f}")
        table.add_row("Net Profit", f"${metrics.net_profit:,.2f}")
        table.add_row("Expectancy", f"${metrics.expectancy:.2f}")
        table.add_row("Kelly (Mean)", f"{metrics.kelly_mean:.2f}")
        table.add_row("Kelly (Safe - 25th)", f"[bold green]{metrics.kelly_safe:.2f}[/bold green]")
        table.add_row("SQN", f"{metrics.sqn:.2f}")
        table.add_row("Max Drawdown", f"{metrics.max_drawdown*100:.1f}%")
        
        console.print(table)
        
        # Benchmark
        console.print(Panel(
            f"Strategy: [green]{metrics.strategy_return*100:.1f}%[/green] vs SPY: [yellow]{metrics.benchmark_return*100:.1f}%[/yellow]", 
            title="Benchmark Comparison"
        ))

        # 6. Strategy Breakdown
        strategy_metrics = analytics.run_strategy_analysis()
        if strategy_metrics:
            console.print("\n[bold blue]--- Strategy Breakdown ---[/bold blue]")
            
            # Create a comparison table
            comp_table = Table(title="Strategy Comparison")
            comp_table.add_column("Metric", style="cyan")
            
            for strat_name in strategy_metrics.keys():
                comp_table.add_column(strat_name, style="magenta")
                
            # Define metrics to show
            rows = [
                ("Total Trades", lambda m: str(m.total_trades)),
                ("Win Rate", lambda m: f"{m.win_rate*100:.1f}%"),
                ("Profit Factor", lambda m: f"{m.profit_factor:.2f}"),
                ("Net Profit", lambda m: f"${m.net_profit:,.2f}"),
                ("Kelly (Safe)", lambda m: f"{m.kelly_safe:.2f}"),
                ("SQN", lambda m: f"{m.sqn:.2f}"),
                ("Max Drawdown", lambda m: f"{m.max_drawdown*100:.1f}%"),
            ]
            
            for label, accessor in rows:
                row_values = [label]
                for m in strategy_metrics.values():
                    row_values.append(accessor(m))
                comp_table.add_row(*row_values)
                
            console.print(comp_table)
        
        # Robustness
        if mc_results:
             mc_table = Table(title="Monte Carlo Robustness (1000 Runs)")
             mc_table.add_column("Metric", style="cyan")
             mc_table.add_column("Value", style="bold yellow")
             mc_table.add_row("Risk of Ruin", f"{mc_results.get('risk_of_ruin', 0)*100:.1f}%")
             mc_table.add_row("Median MaxDD", f"{mc_results.get('median_dd', 0)*100:.1f}%")
             mc_table.add_row("Profit Probability", f"{mc_results.get('profit_prob', 0)*100:.1f}%")
             mc_table.add_row("Worst Case Profit", f"${mc_results.get('worst_case_profit', 0):,.2f}")
             console.print(mc_table)

    except Exception as e:
        console.print(f"[bold red]Backtest Failed:[/bold red] {e}")
        logger.exception("Backtest Fatal Error")
        sys.exit(1)

if __name__ == "__main__":
    main()
