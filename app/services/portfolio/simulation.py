
import pandas as pd
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
from .dynamic_sizing import CapacityMonitor, DynamicPositionSizer

logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    scenario_name: str
    median_return: float
    max_drawdown: float
    avg_exposure: float = 0.0

class CapacitySimulator:
    """
    Runs Monte Carlo simulations to compare Static vs. Dynamic position sizing.
    Designed to be integrated into the Backtest Runner.
    """
    
    def __init__(self, initial_capital: float = 100000.0, n_simulations: int = 100):
        self.initial_capital = initial_capital
        self.n_simulations = n_simulations
        
    def run(self, trades_df: pd.DataFrame, base_kelly: float = 0.39, p95_concurrency: float = 10.0) -> pd.DataFrame:
        """
        Runs the simulation scenarios on the provided trades DataFrame.
        expected trades_df columns: entry_date, exit_date, strategy, return_pct
        """
        if trades_df.empty:
            logger.warning("No trades provided for capacity simulation.")
            return pd.DataFrame()

        # Ensure datetime
        if not np.issubdtype(trades_df['entry_date'].dtype, np.datetime64):
             trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
        if not np.issubdtype(trades_df['exit_date'].dtype, np.datetime64):
             trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
             
        # Calculate return_pct if missing
        if 'return_pct' not in trades_df.columns:
            # Handle possible zero entry price? 
            # In backtest logic, entry should be > 0.
            trades_df = trades_df.copy() # Avoid SettingWithCopy
            trades_df['return_pct'] = (trades_df['exit_price'] - trades_df['entry_price']) / trades_df['entry_price']

        # Calculate base unit size (normalized by expected concurrency)
        # This gives dynamic sizing "room" to fluctuate around the baseline.
        base_unit_size = base_kelly / max(p95_concurrency, 1.0)
        
        # Scenarios Definition
        scenarios = {
            'A_Static': {'dynamic': False, 'unit_size': base_unit_size},
            'B_Dynamic_95': {'dynamic': True, 'unit_size': base_unit_size, 'percentile': 95},
            'C_Aggressive_99': {'dynamic': True, 'unit_size': base_unit_size, 'percentile': 99},
        }
        
        results = []
        logger.info(f"Starting Capacity Simulation ({self.n_simulations} runs, Base Unit: {base_unit_size:.2%})...")

        for name, config in scenarios.items():
            scenario_res = self._run_scenario(trades_df, name, config)
            results.append(scenario_res)
            
        return pd.DataFrame(results)

    def _run_scenario(self, trades_df: pd.DataFrame, name: str, config: dict) -> dict:
        """Runs a single scenario simulation."""
        scenario_returns = []
        scenario_drawdowns = []
        scenario_interest = []
        scenario_commissions = []
        
        for i in range(self.n_simulations):
            # Bootstrap Resampling
            sim_trades = trades_df.sample(frac=1.0, replace=True).sort_values('entry_date')
            
            equity = self.initial_capital
            peak_equity = self.initial_capital
            max_dd = 0.0
            total_interest = 0.0
            total_commissions = 0.0
            
            monitor = CapacityMonitor()
            sizer = DynamicPositionSizer(
                base_kelly=config['unit_size'],
                target_percentile=config.get('percentile', 95)
            )
            
            active_trades = [] # List of {'exit_date': date, 'strategy': str, 'size_pct': float}
            
            # Unique dates in simulation
            all_dates = sorted(sim_trades['entry_date'].unique())
            last_sim_date = None
            
            for current_date in all_dates:
                # 1. Clean up expired trades
                active_trades = [t for t in active_trades if t['exit_date'] >= current_date]
                
                # 2. Get new trades for today
                todays_trades = sim_trades[sim_trades['entry_date'] == current_date]
                
                # 2b. Margin Interest Calculation (Gap-based for accuracy)
                # We deduct interest for the days passed since the last trade entry event.
                if last_sim_date is not None:
                    # Physical days gap
                    gap_days = (current_date - last_sim_date).days
                    if gap_days > 0 and active_trades:
                        # Current exposure is the sum of sizes of active trades
                        # (Allocated at their respective entry moments)
                        current_exposure_pct = sum(t['size_pct'] for t in active_trades)
                        
                        if current_exposure_pct > 1.0:
                            margin_amount = (current_exposure_pct - 1.0) * equity
                            # Deduct for all days in the gap
                            gap_interest = margin_amount * (0.06 / 360.0) * gap_days
                            equity -= gap_interest
                            total_interest += gap_interest

                # 3. Update Monitor for sizer
                active_map = {}
                for t in active_trades:
                    strat = t['strategy']
                    if strat not in active_map: active_map[strat] = []
                    active_map[strat].append(t)
                monitor.update(current_date, active_map)
                
                # 4. Process New Trades (Daily)
                for _, trade in todays_trades.iterrows():
                    strategy = trade['strategy']
                    
                    if config['dynamic']:
                        strat_count = len(active_map.get(strategy, []))
                        position_size = sizer.calculate_position_size(
                            equity, strategy, strat_count, monitor
                        )
                    else:
                        position_size = equity * config['unit_size']
                    
                    # 5% Hard Cap per single trade
                    position_size = min(position_size, equity * 0.05)
                    size_pct = position_size / equity if equity > 0 else 0.0
                    
                    # Transaction Cost Calculation
                    entry_price = trade['entry_price']
                    if entry_price > 0:
                        shares = position_size / entry_price
                        commission = max(2.0, shares * 0.01) * 2 # Round trip
                        equity -= commission
                        total_commissions += commission

                    pnl = position_size * trade['return_pct']
                    equity += pnl
                    
                    active_trades.append({
                        'exit_date': trade['exit_date'],
                        'strategy': strategy,
                        'size_pct': size_pct
                    })
                    
                    # Update local map for same-day concurrency scaling
                    if strategy not in active_map: active_map[strategy] = []
                    active_map[strategy].append({'dummy': True})

                if equity > peak_equity: peak_equity = equity
                dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
                max_dd = max(max_dd, dd)
                
                last_sim_date = current_date
            
            # Final check: any remaining interest until the last exit?
            if active_trades and last_sim_date:
                final_exit = max(t['exit_date'] for t in active_trades)
                remaining_days = (final_exit - last_sim_date).days
                if remaining_days > 0:
                    current_exposure_pct = sum(t['size_pct'] for t in active_trades)
                    if current_exposure_pct > 1.0:
                        margin_amount = (current_exposure_pct - 1.0) * equity
                        final_interest = margin_amount * (0.06 / 360.0) * remaining_days
                        equity -= final_interest
                        total_interest += final_interest

            scenario_returns.append((equity - self.initial_capital) / self.initial_capital)
            scenario_drawdowns.append(max_dd)
            scenario_interest.append(total_interest)
            scenario_commissions.append(total_commissions)
            
        return {
            'Scenario': name,
            'Median Return': np.median(scenario_returns),
            'Max Drawdown (95th Worst)': np.percentile(scenario_drawdowns, 95),
            'Avg Commission': np.mean(scenario_commissions),
            'Avg Margin Interest': np.mean(scenario_interest)
        }
