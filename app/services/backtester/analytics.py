import duckdb
import pandas as pd
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class BacktestMetrics:
    total_trades: int
    win_rate: float
    profit_factor: float
    net_profit: float
    max_drawdown: float
    sharpe_ratio: float
    kelly_criterion: float
    expectancy: float
    sqn: float
    avg_win: float
    avg_loss: float
    
    # Efficiency
    avg_mae: float
    avg_mfe: float
    
    # Robustness
    risk_of_ruin: float
    
    # Comparison
    benchmark_return: float
    strategy_return: float
    
    # Kelly
    kelly_mean: float
    kelly_std: float
    kelly_safe: float

class BacktestAnalytics:
    def __init__(self, backtest_db_path: str, market_db_path: str):
        self.backtest_db = backtest_db_path
        self.market_db = market_db_path
        
    def run_analysis(self, initial_capital: float = 100_000.0) -> BacktestMetrics:
        """
        Connects to the SQLite DBs using DuckDB and computes metrics.
        """
        # Connect strictly in-memory, but attach the SQLite files
        con = duckdb.connect(database=':memory:')
        
        # Load SQLite Extension - DuckDB includes it by default in modern versions,
        # but sometimes needs INSTALL sqlite; LOAD sqlite;
        try:
            con.sql("INSTALL sqlite; LOAD sqlite;")
        except Exception:
            pass # Might be pre-loaded
            
        # Attach Databases
        con.sql(f"ATTACH '{self.backtest_db}' AS bt (TYPE SQLITE);")
        con.sql(f"ATTACH '{self.market_db}' AS mkt (TYPE SQLITE);")
        
        # 1. Basic Trade Metrics
        # We limit to CLOSED trades for PnL
        trades_df = con.sql("""
            SELECT 
                realized_pnl, 
                entry_price, 
                exit_price,
                initial_size,
                (realized_pnl / (initial_size * entry_price)) as ret_pct
            FROM bt.trades 
            WHERE status = 'CLOSED'
        """).df()
        
        if trades_df.empty:
            return self._empty_metrics()
            
        total_trades = len(trades_df)
        wins = trades_df[trades_df['realized_pnl'] > 0]
        losses = trades_df[trades_df['realized_pnl'] <= 0]
        
        win_rate = len(wins) / total_trades if total_trades > 0 else 0
        gross_win = wins['realized_pnl'].sum()
        gross_loss = abs(losses['realized_pnl'].sum())
        
        profit_factor = gross_win / gross_loss if gross_loss > 0 else 999.0
        net_profit = trades_df['realized_pnl'].sum()
        
        avg_win = wins['realized_pnl'].mean() if not wins.empty else 0
        avg_loss = losses['realized_pnl'].mean() if not losses.empty else 0
        
        # Expectancy = (Win% * AvgWin) - (Loss% * AvgLoss) -> Simplified: Avg PnL
        expectancy = trades_df['realized_pnl'].mean()
        
        # SQN: sqrt(N) * (Expectancy / StdDev of R-multiples usually, or PnL)
        # Using simple PnL std dev for now
        std_dev = trades_df['realized_pnl'].std()
        sqn = (total_trades ** 0.5) * (expectancy / std_dev) if std_dev > 0 else 0
        
        # Kelly: W - ( (1-W) / R ) where W=WinRate, R=AvgWin/AvgLoss
        r_ratio = (avg_win / abs(avg_loss)) if avg_loss != 0 else 0
        kelly = win_rate - ((1 - win_rate) / r_ratio) if r_ratio > 0 else 0
        
        # 2. Risk / Drawdown
        # We need equity curve. Assuming accumulation of PnL on top of Initial Capital
        trades_df['equity'] = initial_capital + trades_df['realized_pnl'].cumsum()
        trades_df['peak'] = trades_df['equity'].cummax()
        trades_df['drawdown_pct'] = (trades_df['equity'] - trades_df['peak']) / trades_df['peak']
        
        max_drawdown = trades_df['drawdown_pct'].min() # Negative value
        
        # Sharpe (Simplified annualized)
        # Avg Return / StdDev Return * sqrt(252)
        # This requires daily returns of the portfolio, but we only have trade returns.
        # Approximation: Trade-based Sharpe
        
        # 3. Efficiency (MAE/MFE)
        # Requires joining with Market Data.
        # This demonstrates DuckDB's power: Join SQLite tables across files.
        try:
            efficiency_sql = """
                WITH trade_extremes AS (
                    SELECT 
                        t.symbol,
                        t.entry_price,
                        t.entry_date,
                        t.exit_date,
                        MIN(m.low) as min_price_during,
                        MAX(m.high) as max_price_during,
                        t.id,
                        t.initial_size
                    FROM bt.trades t
                    JOIN mkt.market_prices m ON t.symbol = m.symbol
                    WHERE t.status = 'CLOSED'
                      AND CAST(m.date AS VARCHAR) >= CAST(t.entry_date AS VARCHAR) 
                      AND CAST(m.date AS VARCHAR) <= CAST(t.exit_date AS VARCHAR)
                    GROUP BY t.symbol, t.entry_price, t.entry_date, t.exit_date, t.id, t.initial_size
                )
                SELECT
                    -- MAE: (MinPrice - Entry) / Entry. If Long, negative is bad.
                    AVG((min_price_during - entry_price) / entry_price) as avg_mae,
                    
                    -- MFE: (MaxPrice - Entry) / Entry. Positive is potential.
                    AVG((max_price_during - entry_price) / entry_price) as avg_mfe
                FROM trade_extremes
            """
            eff_res = con.sql(efficiency_sql).fetchone()
            avg_mae = eff_res[0] or 0.0
            avg_mfe = eff_res[1] or 0.0
        except Exception as e:
            logger.error(f"Efficiency calc failed: {e}")
            avg_mae, avg_mfe = 0.0, 0.0
        
        # 4. Safe Kelly (Bootstrapping)
        kelly_metrics = self._calculate_safe_kelly(trades_df, iterations=10_000)
        
        # Calculate Benchmark BEFORE closing connection
        benchmark_ret = self._calc_benchmark_return(con)
            
        con.close()
        
        return BacktestMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            net_profit=net_profit,
            max_drawdown=max_drawdown,
            sharpe_ratio=0.0, 
            kelly_criterion=kelly,
            expectancy=expectancy,
            sqn=sqn,
            avg_win=avg_win,
            avg_loss=avg_loss,
            avg_mae=avg_mae,
            avg_mfe=avg_mfe,
            risk_of_ruin=0.0, 
            benchmark_return=benchmark_ret, 
            strategy_return=net_profit / initial_capital,
            kelly_mean=kelly_metrics['mean'],
            kelly_std=kelly_metrics['std'],
            kelly_safe=kelly_metrics['safe']
        )

    def _calc_benchmark_return(self, con) -> float:
        """Calculates SPY return for the period found in trades."""
        # Find Date Range of trades
        try:
            range_df = con.sql("SELECT MIN(entry_date), MAX(exit_date) FROM bt.trades").fetchone()
            start_date, end_date = range_df[0], range_df[1]
            
            if not start_date or not end_date:
                return 0.0
                
            # Fetch SPY open/close
            # Using 'SPY' or 'QQQ' as default. Use SPY.
            # We need to ensure SPY is in market_prices.
            
            start_price = con.sql(f"SELECT open FROM mkt.market_prices WHERE symbol='SPY' AND date >= '{start_date}' ORDER BY date ASC LIMIT 1").fetchone()
            end_price = con.sql(f"SELECT close FROM mkt.market_prices WHERE symbol='SPY' AND date <= '{end_date}' ORDER BY date DESC LIMIT 1").fetchone()
            
            if start_price and end_price and start_price[0]:
                return (end_price[0] - start_price[0]) / start_price[0]
                
            return 0.0
        except Exception as e:
            logger.warning(f"Benchmark calc failed: {e}")
            return 0.0

    def get_equity_curve(self, initial_capital: float = 100_000.0) -> pd.DataFrame:
        """
        Returns a DataFrame with 'exit_date', 'realized_pnl', 'equity', 'drawdown_pct'.
        Useful for plotting.
        """
        con = duckdb.connect(database=':memory:')
        try:
            con.sql("INSTALL sqlite; LOAD sqlite;")
        except: pass
        
        con.sql(f"ATTACH '{self.backtest_db}' AS bt (TYPE SQLITE);")
        
        try:
            df = con.sql("""
                SELECT 
                    exit_date,
                    realized_pnl
                FROM bt.trades 
                WHERE status = 'CLOSED'
                ORDER BY exit_date ASC
            """).df()
            
            if df.empty:
                return pd.DataFrame()
            
            # Convert to datetime
            df['exit_date'] = pd.to_datetime(df['exit_date'])
            
            # Cumulative
            df['equity'] = initial_capital + df['realized_pnl'].cumsum()
            
            # Drawdown
            df['peak'] = df['equity'].cummax()
            df['drawdown_pct'] = (df['equity'] - df['peak']) / df['peak']
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to get equity curve: {e}")
            return pd.DataFrame()
        finally:
            con.close()

    def get_trade_lists(self) -> dict:
        """
        Returns dictionaries for 'recent', 'top', 'worst' trades.
        Each is a list of dicts.
        """
        con = duckdb.connect(database=':memory:')
        try:
            con.sql("INSTALL sqlite; LOAD sqlite;")
        except: pass
        con.sql(f"ATTACH '{self.backtest_db}' AS bt (TYPE SQLITE);")
        
        try:
            # Helper to fetch and convert
            def fetch(query):
                df = con.sql(query).df()
                if df.empty: return []
                # Round PnL for display
                df['realized_pnl'] = df['realized_pnl'].round(2)
                return df.to_dict(orient='records')
            
            # 1. Recent (Last 20)
            recent = fetch("""
                SELECT symbol, entry_date, exit_date, entry_price, exit_price, realized_pnl, exit_reason
                FROM bt.trades WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 20
            """)
            
            # 2. Top 10 Wins
            top = fetch("""
                SELECT symbol, entry_date, exit_date, entry_price, exit_price, realized_pnl, exit_reason
                FROM bt.trades WHERE status='CLOSED' ORDER BY realized_pnl DESC LIMIT 10
            """)
            
            # 3. Worst 10 Losses
            worst = fetch("""
                SELECT symbol, entry_date, exit_date, entry_price, exit_price, realized_pnl, exit_reason
                FROM bt.trades WHERE status='CLOSED' ORDER BY realized_pnl ASC LIMIT 10
            """)
            
            return {"recent": recent, "top": top, "worst": worst}
            
        except Exception as e:
            logger.error(f"Failed to get trade lists: {e}")
            return {"recent": [], "top": [], "worst": []}
        finally:
            con.close()


    def run_monte_carlo(self, iterations: int = 1000) -> dict:
        """
        Runs Monte Carlo simulation by shuffling trades.
        Returns: { 'median_dd': float, 'profit_prob': float, 'risk_of_ruin': float }
        """
        con = duckdb.connect(database=':memory:')
        con.sql("INSTALL sqlite; LOAD sqlite;")
        con.sql(f"ATTACH '{self.backtest_db}' AS bt (TYPE SQLITE);")
        
        # Get PnL series
        trades = con.sql("SELECT realized_pnl FROM bt.trades WHERE status='CLOSED'").df()
        pnls = trades['realized_pnl'].values
        
        if len(pnls) < 10:
            return {}
            
        import numpy as np
        
        drawdowns = []
        final_eqs = []
        ruin_count = 0
        initial_cap = 100_000
        
        for _ in range(iterations):
            # Shuffle
            shuffled = np.random.choice(pnls, size=len(pnls), replace=True)
            
            # Equity Curve
            equity = initial_cap + np.cumsum(shuffled)
            
            # Check Ruin (< 50% cap)
            if np.any(equity < (initial_cap * 0.5)):
                ruin_count += 1
                
            final_eqs.append(equity[-1])
            
            # DD
            peak = np.maximum.accumulate(equity)
            dd = (equity - peak) / peak
            drawdowns.append(np.min(dd))
            
        return {
            "median_dd": float(np.median(drawdowns)),
            "profit_prob": float(np.mean(np.array(final_eqs) > initial_cap)),
            "worst_case_profit": float(np.percentile(final_eqs, 1)) - initial_cap
        }

    def _calculate_safe_kelly(self, trades_df: pd.DataFrame, iterations: int = 10_000) -> dict:
        """
        Runs Monte Carlo Bootstrapping to find Safe Kelly fraction.
        """
        pnls = trades_df['realized_pnl'].values
        
        # Need at least a small sample
        if len(pnls) < 20:
            return {'mean': 0.0, 'std': 0.0, 'safe': 0.0}
            
        import numpy as np
        
        kelly_values = []
        
        # Pre-filter wins/losses just for shape logic, but inside loop we resample
        # Logic: 
        # 1. Sample N trades with replacement
        # 2. Calc Win Rate (p)
        # 3. Calc Win/Loss Ratio (b)
        # 4. f* = p - (q/b)
        
        n_trades = len(pnls)
        
        for _ in range(iterations):
            sample = np.random.choice(pnls, size=n_trades, replace=True)
            
            wins = sample[sample > 0]
            losses = sample[sample <= 0]
            
            if len(wins) == 0:
                kelly_values.append(0.0)
                continue
                
            p = len(wins) / n_trades
            q = 1.0 - p
            
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses)) if len(losses) > 0 else 0
            
            if avg_loss == 0:
                # Infinite profit ratio -> Kelly approaches 1.0 (approximated)
                kelly_values.append(0.99)
                continue
                
            b = avg_win / avg_loss
            
            f_star = p - (q / b)
            kelly_values.append(max(0.0, f_star))
            
        arr_kelly = np.array(kelly_values)
        
        return {
            'mean': float(np.mean(arr_kelly)),
            'std': float(np.std(arr_kelly)),
            'safe': float(np.percentile(arr_kelly, 25))
        }


    def _empty_metrics(self):
        return BacktestMetrics(
            0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
        )
