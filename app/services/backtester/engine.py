import logging
import pandas as pd
from datetime import date, timedelta
from typing import Type

from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskID
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.table import Table

from ...database.repositories.trade import TradeRepository
from ...database.repositories.market_data_provider import MarketDataProvider
from ...const import Strategies, STRATEGY_ALIASES

# Strategies
from ...services.screener.strategies.dip_buyer import DipBuyerStrategy
from ...services.trade_manager.strategies.dip_buyer import DipBuyerStrategy as TradeManagerDipBuyer

from ...services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from ...services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy as TradeManagerTurnover

from ...services.screener.strategies.two_percent_strategy import TwoPercentStrategy
from ...services.trade_manager.strategies.two_percent_strategy import TwoPercentStrategy as TradeManagerTwoPercent

from ...types import TradeStatus

logger = logging.getLogger(__name__)

from ...services.portfolio.manager import PortfolioManager

class BacktestEngine:
    """
    Orchestrates the backtesting simulation.
    Moves 'Virtual Time' forward day by day.
    """
    
    def __init__(
        self,
        start_date: str,
        end_date: str,
        market_provider: MarketDataProvider,
        trade_repository: TradeRepository,
        console: Console | None = None,
        portfolio_config: dict | None = None
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.market = market_provider
        self.trade_repository = trade_repository
        self.console = console or Console()
        
        self.portfolio_manager = PortfolioManager(
            self.trade_repository, 
            portfolio_config=portfolio_config
        )
        
        # --- Strategy Registry ---
        self.screeners = [
            DipBuyerStrategy(trade_repository=self.trade_repository, data_provider=self.market),
            TurnoverTimingStrategy(trade_repository=self.trade_repository, data_provider=self.market),
            TwoPercentStrategy(trade_repository=self.trade_repository, data_provider=self.market)
        ]
        
        self.trade_managers = {
            Strategies.DipBuyer: TradeManagerDipBuyer(),
            Strategies.TurnOverTiming: TradeManagerTurnover(),
            Strategies.TurnOverTiming_10: TradeManagerTurnover(strategy_name=Strategies.TurnOverTiming_10),
            Strategies.TurnOverTiming_05: TradeManagerTurnover(strategy_name=Strategies.TurnOverTiming_05),
            Strategies.TwoPercent: TradeManagerTwoPercent()
        }
        
        self.market_dates: list[pd.Timestamp] = []
        
        # State
        self.current_date: pd.Timestamp | None = None
        self._symbol_cache: dict[str, pd.DataFrame] = {}
    
    def setup(self):
        """Prepares the backtest environment."""
        self.console.print("[bold blue]Initialising Backtest Environment...[/bold blue]")
        
        spy_hist = self.market.get_symbol_history("SPY", days=10000)
        
        if spy_hist.empty:
            self.console.print("[yellow]WARNING: No market data for 'SPY'. Attempting fallback...[/yellow]")
            fallback_dates = self.market.get_available_dates(str(self.start_date.date()), str(self.end_date.date()))
            if not fallback_dates:
                 self.console.print("[bold red]CRITICAL: No market data found.[/bold red]")
                 return
            self.market_dates = sorted(fallback_dates)
        else:
            if "date" in spy_hist.columns:
                spy_hist = spy_hist.set_index("date")
            mask = (spy_hist.index >= self.start_date) & (spy_hist.index <= self.end_date)
            self.market_dates = sorted(spy_hist[mask].index.to_list())
        
        self.console.print(f"Found [bold green]{len(self.market_dates)}[/bold green] trading days.")
        self.market.preload_all_data(days=2000)

    def _get_symbol_data_slice(self, symbol: str, current_date: pd.Timestamp) -> pd.DataFrame:
        """Retrieves a slice of historical data up to the current date from cache.
        
        Loads the full history into memory on the first request for a symbol.
        """
        if symbol not in self._symbol_cache:
            # Load full history once. Using a large enough day count to cover backtest + lookback
            df_full = self.market.get_symbol_history(symbol, days=2500)
            
            if df_full.empty:
                logger.warning("No data for %s - returning empty DataFrame", symbol)
                return pd.DataFrame()

            if "date" in df_full.columns:
                df_full = df_full.set_index("date")
            self._symbol_cache[symbol] = df_full
            
        dataframe = self._symbol_cache[symbol]
        if dataframe.empty:
            return dataframe
            
        return dataframe[dataframe.index <= current_date]

    def run(self):
        """Main Execution execution."""
        if not self.market_dates:
            self.setup()
            
        total_days = len(self.market_dates)
        
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TextColumn("• {task.fields[date]}")
        )
        task_id = progress.add_task("[cyan]Running Backtest...", total=total_days, date=self.start_date.date())
        
        with Live(progress, console=self.console, refresh_per_second=10) as live:
            for sim_date in self.market_dates:
                self.current_date = sim_date
                date_str = sim_date.strftime("%Y-%m-%d")
                
                progress.update(task_id, advance=1, date=date_str)
                
                # 0. Portfolio Allocation (Sizing)
                # Assigns size to CREATED trades before they are checked for entry.
                self.portfolio_manager.process_daily_signals()

                # 1. Check Entries FIRST (Morning Logic)
                # If a trade was created Yesterday, check if it fills Today at Open/Limit.
                self._run_trade_manager_entries(sim_date)
                
                # 2. Manage Active Trades (During Day Logic)
                # Check if stops or targets were hit TODAY.
                self._run_trade_manager_exits(sim_date)
                
                # 3. Run Screener (Evening Logic - prepares for Tomorrow)
                # Finds new setups based on today's closing data.
                self._run_screener(sim_date)

    def _run_screener(self, date_obj: pd.Timestamp):
        """Runs all registered screeners for the given date."""
        date_str = date_obj.strftime("%Y-%m-%d")
        for screener in self.screeners:
            try:
                screener.run(analysis_date=date_str)
            except Exception as error:
                logger.error(
                    "Screener %s Error %s: %s", 
                    screener.name, date_obj.date(), error
                )

    def _get_manager_for_strategy(self, strategy_name: str) -> object | None:
        """Dispatches to the correct manager based on strategy name."""
        # 1. Try exact match (Enum value)
        if strategy_name in self.trade_managers:
            return self.trade_managers[strategy_name]
            
        # 2. Try Alias Resolution
        resolved = STRATEGY_ALIASES.get(strategy_name.lower())
        if resolved and resolved in self.trade_managers:
            return self.trade_managers[resolved]

        return None

    def _run_trade_manager_entries(self, current_date: pd.Timestamp):
        """Checks if CREATED trades get filled."""
        created = self.trade_repository.get_by_status(TradeStatus.CREATED)
        for trade in created:
            strategy_name = trade['strategy']
            manager = self._get_manager_for_strategy(strategy_name)
            if not manager:
                logger.warning(f"No manager found for strategy: {strategy_name}")
                continue
            
            symbol = trade['symbol']
            df_slice = self._get_symbol_data_slice(symbol, current_date)
            
            if df_slice.empty or df_slice.index[-1] != current_date:
                continue
                
            candle = df_slice.iloc[-1].copy()
            candle['date'] = current_date
            
            manager.check_entry(
                trade, 
                candle, 
                df_slice.reset_index(), 
                self.trade_repository
            )

    def _run_trade_manager_exits(self, current_date: pd.Timestamp):
        """Checks exits for ACTIVE trades."""
        active = self.trade_repository.get_by_status(TradeStatus.ACTIVE)
        for trade in active:
            strategy_name = trade['strategy']
            manager = self._get_manager_for_strategy(strategy_name)
            if not manager:
                continue

            symbol = trade['symbol']
            df_slice = self._get_symbol_data_slice(symbol, current_date)
            
            if df_slice.empty or df_slice.index[-1] != current_date:
                continue
            
            manager.manage_active_trade(
                trade, 
                df_slice.reset_index(), 
                self.trade_repository
            )

