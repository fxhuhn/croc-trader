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

# Strategies
from ...services.screener.strategies.dip_buyer import DipBuyerStrategy
from ...services.trade_manager.strategies.dip_buyer import DipBuyerStrategy as TradeManagerDipBuyer

from ...services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from ...services.trade_manager.strategies.turnover_timing import TurnoverTimingStrategy as TradeManagerTurnover

from ...types import TradeStatus

logger = logging.getLogger(__name__)

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
        trade_repo: TradeRepository,
        console: Console | None = None
    ):
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.market = market_provider
        self.trade_repo = trade_repo
        self.console = console or Console()
        
        # --- Strategy Registry ---
        # 1. Screeners (Run every evening)
        self.screeners = [
            DipBuyerStrategy(trade_repo=self.trade_repo, data_provider=self.market),
            TurnoverTimingStrategy(trade_repo=self.trade_repo, data_provider=self.market)
        ]
        
        # 2. Trade Managers (Run every morning / intraday)
        # Mapping: partial strategy name -> manager instance
        # "DipBuyer" -> DipBuyerManager
        # "TurnoverTiming" -> TurnoverManager
        # "TurnoverTiming_0.5" -> TurnoverManager
        self.trade_managers = {
            "DipBuyer": TradeManagerDipBuyer(),
            "TurnoverTiming": TradeManagerTurnover()
        }
        
        self.market_dates: list[pd.Timestamp] = []
        
        # State
        self.current_date: pd.Timestamp | None = None
    
    def setup(self):
        """Prepares the backtest environment."""
        self.console.print("[bold blue]Initialising Backtest Environment...[/bold blue]")
        
        # 0. Identify valid trading days in the range (using SPY or similar as proxy for 'market open')
        # We use the existing market repo to find all unique dates
        # Or just fetch SPY history
        spy_hist = self.market.get_symbol_history("SPY", days=10000) # Fetch enough to cover range
        
        if spy_hist.empty:
            self.console.print("[yellow]WARNING: No market data for 'SPY'. Attempting fallback to find ANY trading days...[/yellow]")
            fallback_dates = self.market.get_available_dates(str(self.start_date.date()), str(self.end_date.date()))
            
            if not fallback_dates:
                 self.console.print("[bold red]CRITICAL: No market data found AT ALL in the requested period.[/bold red]")
                 return
                 
            self.market_dates = sorted(fallback_dates)
        else:
            if "date" in spy_hist.columns:
                spy_hist = spy_hist.set_index("date")
            
            # Filter for range
            mask = (spy_hist.index >= self.start_date) & (spy_hist.index <= self.end_date)
            self.market_dates = sorted(spy_hist[mask].index.to_list())
        
        self.console.print(f"Found [bold green]{len(self.market_dates)}[/bold green] trading days.")
        
        # 1. Reset Test DB (Schema is already init by caller locally, but we ensure clean slate if needed)
        # Assuming repo passed is already connected to CLEAN db.
        pass

    def run(self):
        """Main Execution execution."""
        if not self.market_dates:
            self.setup()
            
        total_days = len(self.market_dates)
        
        # UI Setup
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
                
                # Update UI
                progress.update(task_id, advance=1, date=date_str)
                
                # 1. Manage Active Trades (Morning/During Day Logic simulation)
                # In real life, TM runs periodically. In BT, we verify if exits happened TODAY.
                self._run_trade_manager_exits(sim_date)
                
                # 2. Check Entries for Created Trades
                # If a trade was created Yesterday, check if it fills Today.
                self._run_trade_manager_entries(sim_date)
                
                # 3. Run Screener (Evening Logic - prepares for Tomorrow)
                # Screener analyzes data UP TO today.
                self._run_screener(sim_date)

    def _run_screener(self, date_obj: pd.Timestamp):
        """Runs all registered screeners for the given date."""
        date_str = date_obj.strftime("%Y-%m-%d")
        
        for screener in self.screeners:
            try:
                # DipBuyer/Turnover strategies take analysis_date
                # They determine lookback internally or via config
                screener.run(analysis_date=date_str)
            except Exception as e:
                logger.error(f"Screener {screener.name} Error {date_obj.date()}: {e}")

    def _get_manager_for_strategy(self, strategy_name: str):
        """Dispatches to the correct manager based on strategy name prefix."""
        for key, manager in self.trade_managers.items():
            if strategy_name.startswith(key):
                return manager
        return None

    def _run_trade_manager_entries(self, current_date: pd.Timestamp):
        """Checks if CREATED trades get filled."""
        created = self.trade_repo.get_by_status(TradeStatus.CREATED)
        for trade in created:
            # Dispatch
            strategy_name = trade['strategy']
            manager = self._get_manager_for_strategy(strategy_name)
            if not manager:
                logger.warning(f"No manager found for strategy: {strategy_name}")
                continue
            
            symbol = trade['symbol']
            
            # Load history FOR THIS SYMBOL up to Today
            # We fetch history up to current_date
            
            # Optimization: Fetch simpler if we can, but get_symbol_history is fine
            df_hist_full = self.market.get_symbol_history(symbol, days=2000) 
            if df_hist_full.empty: continue
            
            if "date" in df_hist_full.columns:
                df_hist_full = df_hist_full.set_index("date")
            
            # Slice strictly <= current_date
            df_slice = df_hist_full[df_hist_full.index <= current_date]
            if df_slice.empty: continue
            
            # Check if today is actually in the slice
            if df_slice.index[-1] != current_date:
                continue
                
            # Prepare candle
            candle = df_slice.iloc[-1]
            candle_series = candle.copy()
            candle_series['date'] = current_date
            
            manager.check_entry(
                trade, 
                candle_series, 
                df_slice.reset_index(), 
                self.trade_repo
            )

    def _run_trade_manager_exits(self, current_date: pd.Timestamp):
        """Checks exits for ACTIVE trades."""
        active = self.trade_repo.get_by_status(TradeStatus.ACTIVE)
        for trade in active:
            # Dispatch
            strategy_name = trade['strategy']
            manager = self._get_manager_for_strategy(strategy_name)
            if not manager:
                continue

            symbol = trade['symbol']
            
            df_hist_full = self.market.get_symbol_history(symbol, days=2000)
            if df_hist_full.empty: continue
            
            if "date" in df_hist_full.columns:
                df_hist_full = df_hist_full.set_index("date")
            
            df_slice = df_hist_full[df_hist_full.index <= current_date]
            if df_slice.empty: continue
            
            if df_slice.index[-1] != current_date: continue
            
            manager.manage_active_trade(
                trade, 
                df_slice.reset_index(), 
                self.trade_repo
            )

