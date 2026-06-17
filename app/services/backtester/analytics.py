import duckdb
import pandas as pd
import numpy as np
import logging
import warnings
from dataclasses import dataclass
from typing import Any

from ...models import BacktestMetrics, PortfolioMetrics, SQNClassification
from ...tools.indicators import calculate_sma, calculate_rsi
from ...tools import metrics
from ...const import Strategies

logger = logging.getLogger(__name__)


def get_system_quality_classification(sqn_value: float) -> SQNClassification:
    """Classifies the System Quality Number (SQN) into descriptive sectors.

    Args:
        sqn_value: The calculated SQN value.

    Returns:
        SQNClassification: Object containing the label and color code.
    """
    if sqn_value < 1.0:
        return SQNClassification("Unterdurchschnittlich schlecht", "#ef4444")
    if sqn_value < 2.0:
        return SQNClassification("Noch Ok", "#f59e0b")
    if sqn_value < 3.0:
        return SQNClassification("Gut", "#10b981")
    if sqn_value < 5.0:
        return SQNClassification("Sehr gut", "#3b82f6")
    if sqn_value < 7.0:
        return SQNClassification("Ausgezeichnet", "#8b5cf6")
    return SQNClassification("Der heilige Gral", "#d4af37")


# Constants for maintainability
MIN_TRADES_FOR_SQN = 30
MIN_TRADES_FOR_KELLY = 10
DEFAULT_BOOTSTRAP_ITERATIONS = 10_000
MAX_LEVERAGE_MULTIPLIER = 15.0
EPSILON = 1e-6


def safe_divide(
    numerator: float | pd.Series | np.ndarray,
    denominator: float | pd.Series | np.ndarray,
    default: float = 0.0,
) -> float | pd.Series | np.ndarray:
    """Safely divides two numbers or arrays, handling division by zero.

    Args:
        numerator: Value to be divided.
        denominator: Value to divide by.
        default: Value to return if denominator is near zero.

    Returns:
        The result of the division or the default value.
    """
    if isinstance(denominator, (pd.Series, np.ndarray)):
        return np.where(np.abs(denominator) > EPSILON, numerator / denominator, default)

    return numerator / denominator if abs(denominator) > EPSILON else default


def safe_percentile(
    data_list: list[float] | np.ndarray, percentile: float, default: float = 0.0
) -> float:
    """Safely calculates percentile, returning default if data list is empty."""
    if len(data_list) == 0:
        return default
    return float(np.percentile(data_list, percentile))


class TransactionCostModel:
    """Calculates friction for trades (Commissions and Slippage)."""

    def __init__(self, fixed_commission: float = 0.0, slippage_bps: float = 0.0):
        """Initializes the cost model.

        Args:
            fixed_commission: Round-trip fixed cost.
            slippage_bps: Slippage in basis points per transaction side.
        """
        self.fixed_commission = fixed_commission
        self.slippage_bps = slippage_bps

    def calculate_cost(self, trade_dataframe: pd.DataFrame) -> pd.Series:
        """Calculates total friction cost per trade ($0.01/share, min $2.00)."""
        if trade_dataframe.empty:
            return pd.Series(dtype=float)

        # commissions: $0.01 per share, min $2.00 per order (entry and exit)
        commission_per_order = trade_dataframe["initial_size"] * 0.01
        commissions = commission_per_order.clip(lower=2.0) * 2

        notional_value = (
            trade_dataframe["entry_price"] * trade_dataframe["initial_size"]
        )
        slippage_costs = notional_value * (self.slippage_bps / 10000.0) * 2

        return commissions + slippage_costs


class BacktestDataLoader:
    """Handles all data extraction and performance-heavy SQL joins via DuckDB."""

    def __init__(self, backtest_db_path: str, market_db_path: str):
        self.backtest_db = backtest_db_path
        self.market_db = market_db_path

    def _get_connection(self) -> duckdb.DuckDBPyConnection:
        connection = duckdb.connect(database=":memory:")
        try:
            connection.sql("INSTALL sqlite; LOAD sqlite;")
        except Exception as e:
            logger.debug(
                "DuckDB SQLite extension loading issue (expected if pre-loaded): %s", e
            )

        # backtest.db is always SQLITE (Phase 1 persistence)
        connection.sql(f"ATTACH '{self.backtest_db}' AS backtest (TYPE SQLITE);")

        # market_db (stocks.db) is DuckDB native
        connection.sql(f"ATTACH '{self.market_db}' AS market;")
        return connection

    def fetch_closed_trades(self, strategy_filter: str | None = None) -> pd.DataFrame:
        """Fetches closed trades with an optional strategy filter."""
        connection = self._get_connection()
        try:
            query = """
                WITH unique_trades AS (
                    SELECT DISTINCT ON (symbol, strategy, CAST(entry_date AS DATE), CAST(exit_date AS DATE))
                        symbol, realized_pnl, entry_price, exit_price, initial_size,
                        entry_date, exit_date, strategy, id, exit_reason,
                        COALESCE(current_stop_loss, 0.0) as initial_stop
                    FROM backtest.trades 
                    WHERE status = 'CLOSED' 
                      AND exit_reason NOT IN ('EXPIRED', 'INVALIDATED')
                    ORDER BY symbol, strategy, CAST(entry_date AS DATE), CAST(exit_date AS DATE), id DESC
                )
                SELECT * FROM unique_trades
            """
            params = []
            if strategy_filter:
                query += " WHERE strategy LIKE ?"
                params.append(f"%{strategy_filter}%")

            query += " ORDER BY exit_date ASC, id ASC"

            return connection.execute(query, params).df()
        finally:
            connection.close()

    def calculate_exposure_and_benchmark(
        self, trades_df: pd.DataFrame
    ) -> dict[str, float]:
        """Calculates market exposure % and benchmark return."""
        if trades_df.empty:
            return {"exposure_pct": 0.0, "benchmark_return": 0.0}

        connection = self._get_connection()
        try:
            start_date = trades_df["entry_date"].min()
            end_date = trades_df["exit_date"].max()

            market_days_query = """
                SELECT COUNT(*) FROM market.market_prices 
                WHERE symbol='SPY' AND CAST(date AS TIMESTAMP) >= CAST(? AS TIMESTAMP) AND CAST(date AS TIMESTAMP) <= CAST(? AS TIMESTAMP)
            """
            total_market_days = connection.execute(
                market_days_query, [start_date, end_date]
            ).fetchone()[0]

            strat_unique = trades_df["strategy"].unique()

            active_days_query = """
                SELECT COUNT(DISTINCT m.date) 
                FROM (
                    SELECT unnest(generate_series(CAST(entry_date AS DATE), CAST(exit_date AS DATE), INTERVAL 1 DAY)) as d
                    FROM backtest.trades 
                    WHERE status='CLOSED'
                ) t
                JOIN market.market_prices m ON t.d = CAST(m.date AS DATE)
                WHERE m.symbol = 'SPY'
            """
            # Add strategy filter if unique
            if len(strat_unique) == 1:
                active_days_query = active_days_query.replace(
                    "WHERE status='CLOSED'", "WHERE status='CLOSED' AND strategy = ?"
                )
                active_days = connection.execute(
                    active_days_query, [strat_unique[0]]
                ).fetchone()[0]
            else:
                active_days = connection.execute(active_days_query).fetchone()[0]

            bench_query = """
                SELECT 
                    (SELECT close FROM market.market_prices WHERE symbol='SPY' AND CAST(date AS TIMESTAMP) <= CAST(? AS TIMESTAMP) ORDER BY date DESC LIMIT 1) /
                    (SELECT open FROM market.market_prices WHERE symbol='SPY' AND CAST(date AS TIMESTAMP) >= CAST(? AS TIMESTAMP) ORDER BY date ASC LIMIT 1) - 1
            """
            benchmark_return = (
                connection.execute(bench_query, [end_date, start_date]).fetchone()[0]
                or 0.0
            )

            return {
                "exposure_pct": (active_days / total_market_days)
                if total_market_days > 0
                else 0.0,
                "benchmark_return": benchmark_return,
            }
        finally:
            connection.close()

    def fetch_efficiency_metrics(self) -> dict[str, float]:
        """Fetches MAE and MFE averages."""
        connection = self._get_connection()
        try:
            efficiency_sql = """
                WITH unique_trades AS (
                    SELECT DISTINCT ON (symbol, strategy, CAST(entry_date AS DATE), CAST(exit_date AS DATE))
                        symbol, entry_price, entry_date, exit_date, id
                    FROM backtest.trades
                    WHERE status = 'CLOSED'
                    ORDER BY symbol, strategy, CAST(entry_date AS DATE), CAST(exit_date AS DATE), id DESC
                ),
                trade_extremes AS (
                    SELECT 
                        t.symbol, t.entry_price, t.entry_date, t.exit_date,
                        MIN(m.low) as min_during, MAX(m.high) as max_during
                    FROM unique_trades t
                    JOIN market.market_prices m ON t.symbol = m.symbol
                    WHERE CAST(m.date AS DATE) >= CAST(t.entry_date AS DATE) 
                      AND CAST(m.date AS DATE) <= CAST(t.exit_date AS DATE)
                    GROUP BY t.symbol, t.entry_price, t.entry_date, t.exit_date, t.id
                )
                SELECT
                    AVG((min_during - entry_price) / NULLIF(entry_price, 0)),
                    AVG((max_during - entry_price) / NULLIF(entry_price, 0))
                FROM trade_extremes
            """
            result = connection.sql(efficiency_sql).fetchone()
            return {"avg_mae": result[0] or 0.0, "avg_mfe": result[1] or 0.0}
        finally:
            connection.close()

    def fetch_market_regime(self) -> pd.DataFrame:
        """Fetches regime indicators and identifies safety switch triggers."""
        connection = self._get_connection()
        try:
            spy_df = connection.sql(
                "SELECT date, close FROM market.market_prices "
                "WHERE symbol='SPY' ORDER BY date"
            ).df()
            if spy_df.empty:
                return pd.DataFrame()

            spy_df["date"] = pd.to_datetime(spy_df["date"])
            spy_df.set_index("date", inplace=True)

            # 1. SPY Indicators
            smas = [50, 150, 200, 250, 300]
            for window in smas:
                spy_df[f"sma{window}"] = calculate_sma(spy_df["close"], window)

            spy_df["rsi2"] = calculate_rsi(spy_df["close"], 2)
            spy_df["rsi7"] = calculate_rsi(spy_df["close"], 7)

            # Count SMAs where price is above
            spy_df["spy_trend_score"] = sum(
                (spy_df["close"] > spy_df[f"sma{window}"]).astype(int)
                for window in smas
            )
            spy_df["spy_uptrend"] = spy_df["spy_trend_score"] >= 3

            vix_df = connection.sql(
                "SELECT date, close as vix_close FROM market.market_prices "
                "WHERE symbol='^VIX' ORDER BY date"
            ).df()

            if not vix_df.empty:
                vix_df["date"] = pd.to_datetime(vix_df["date"])
                vix_df.set_index("date", inplace=True)
                vix_df["vix_sma20"] = calculate_sma(vix_df["vix_close"], 20)
                regime_df = spy_df.join(vix_df, how="left")
            else:
                regime_df = spy_df
                regime_df["vix_close"] = 0.0
                regime_df["vix_sma20"] = 0.0

            # 2. Simplified Safety Triggers
            # As requested: keep RSI(2) < 10, remove SMA trend and VIX spikes.
            # We add a placeholder VIX < 12 trigger for the standard view.
            regime_df["vix_trigger"] = regime_df["vix_close"] < 12.0
            regime_df["rsi_trigger"] = regime_df["rsi2"] < 10.0

            regime_df["safety_active"] = (
                regime_df["vix_trigger"] | regime_df["rsi_trigger"]
            )

            # 3. Dynamic Trigger Reason
            def identify_reason(row: pd.Series) -> str:
                reasons = []
                if row["vix_trigger"]:
                    reasons.append(f"VIX Low Complacency ({row['vix_close']:.1f} < 12)")

                if row["rsi_trigger"]:
                    reasons.append(f"RSI(2) < 10 ({row['rsi2']:.1f})")

                return " & ".join(reasons) if reasons else ""

            regime_df["trigger_reason"] = regime_df.apply(identify_reason, axis=1)

            return regime_df.reset_index()
        finally:
            connection.close()

    def fetch_synchronized_matrix(
        self, initial_capital: float = 100000.0
    ) -> pd.DataFrame:
        """Fetches daily return impact per strategy."""
        connection = self._get_connection()
        try:
            query = """
                WITH trade_market_days AS (
                    SELECT 
                        t.id,
                        COUNT(m.date) as market_day_count
                    FROM backtest.trades t
                    JOIN market.market_prices m ON CAST(m.date AS DATE) >= CAST(t.entry_date AS DATE) 
                                               AND CAST(m.date AS DATE) <= CAST(t.exit_date AS DATE)
                    WHERE t.status = 'CLOSED' AND m.symbol = 'SPY'
                    GROUP BY t.id
                ),
                daily_trade_returns AS (
                    SELECT 
                        m.date,
                        t.strategy,
                        t.realized_pnl / NULLIF(tm.market_day_count, 0) as daily_pnl,
                        1 as active_trade
                    FROM backtest.trades t
                    JOIN trade_market_days tm ON t.id = tm.id
                    JOIN market.market_prices m ON CAST(m.date AS DATE) >= CAST(t.entry_date AS DATE) 
                                               AND CAST(m.date AS DATE) <= CAST(t.exit_date AS DATE)
                    WHERE m.symbol = 'SPY' AND t.status = 'CLOSED'
                )
                SELECT 
                    date,
                    strategy,
                    SUM(daily_pnl) / self.INITIAL_CAPITAL as daily_impact,
                    SUM(active_trade) as active_trades
                FROM daily_trade_returns
                GROUP BY date, strategy
            """
            # Replace placeholder with actual value
            query_ready = query.replace(
                "self.INITIAL_CAPITAL", f"{initial_capital:.1f}"
            )
            df = connection.sql(query_ready).df()
            if df.empty:
                return pd.DataFrame()

            # Pivot impact
            impact_df = df.pivot(
                index="date", columns="strategy", values="daily_impact"
            ).fillna(0.0)
            impact_df.columns = [f"impact_{c}" for c in impact_df.columns]

            # Pivot counts
            count_df = df.pivot(
                index="date", columns="strategy", values="active_trades"
            ).fillna(0)
            count_df.columns = [f"count_{c}" for c in count_df.columns]

            return impact_df.join(count_df).reset_index()
        finally:
            connection.close()

    def fetch_benchmark_data(
        self, symbol: str, start_date: str, end_date: str, initial_capital: float
    ) -> pd.DataFrame:
        """Calculates normalized benchmark equity curve."""
        connection = self._get_connection()
        try:
            query = """
                SELECT date, close 
                FROM market.market_prices 
                WHERE symbol = ? AND CAST(date AS TIMESTAMP) >= CAST(? AS TIMESTAMP) AND CAST(date AS TIMESTAMP) <= CAST(? AS TIMESTAMP)
                ORDER BY date ASC
            """
            df = connection.execute(query, [symbol, start_date, end_date]).df()
            if df.empty:
                return pd.DataFrame()

            start_price = df.iloc[0]["close"]
            if start_price < EPSILON:
                return pd.DataFrame()

            df["equity"] = initial_capital * (df["close"] / start_price)
            return df[["date", "equity"]]
        finally:
            connection.close()


class MetricsCalculator:
    """Core analytical engine for trade data calculations."""

    def __init__(self, cost_model: TransactionCostModel | None = None):
        self.cost_model = cost_model or TransactionCostModel()

    def calculate_trade_metrics(
        self,
        trades_dataframe: pd.DataFrame,
        initial_capital: float,
        exposure_percentage: float = 0.0,
        benchmark_return: float = 0.0,
        average_maximum_adverse_excursion: float = 0.0,
        average_maximum_favorable_excursion: float = 0.0,
        simulator: object | None = None,
    ) -> BacktestMetrics:
        """Computes all standard backtest metrics.

        This orchestrator utilizes pure mathematical functions from the
        metrics tool to calculate technical analysis indicators.
        """
        if trades_dataframe.empty:
            return self._empty_metrics(benchmark_return)

        working_df = trades_dataframe.copy()
        costs = self.cost_model.calculate_cost(working_df)
        working_df["net_pnl"] = working_df["realized_pnl"] - costs

        if working_df.empty:
            return self._empty_metrics(benchmark_return)

        total_trades = len(working_df)
        win_rate = metrics.calculate_win_rate(working_df["net_pnl"])
        profit_factor = metrics.calculate_profit_factor(working_df["net_pnl"])
        net_profit = working_df["net_pnl"].sum()
        expectancy = metrics.calculate_expectancy(working_df["net_pnl"])

        winning_trades = working_df[working_df["net_pnl"] > EPSILON]
        losing_trades = working_df[working_df["net_pnl"] < -EPSILON]
        average_win = (
            winning_trades["net_pnl"].mean() if not winning_trades.empty else 0.0
        )
        average_loss = (
            losing_trades["net_pnl"].mean() if not losing_trades.empty else 0.0
        )

        # Sizing and Risk Metrics
        risk_raw = (
            working_df["entry_price"] - working_df["initial_stop"]
        ).abs() * working_df["initial_size"]
        fallback_risk = 0.01 * initial_capital
        working_df["risk_unit"] = np.where(risk_raw > EPSILON, risk_raw, fallback_risk)
        working_df["r_multiple"] = safe_divide(
            working_df["net_pnl"], working_df["risk_unit"]
        )

        sqn = 0.0
        if total_trades >= MIN_TRADES_FOR_SQN:
            sqn = metrics.calculate_sqn(working_df["r_multiple"])

        # ROI calculations for Kelly
        entry_prices = pd.to_numeric(working_df["entry_price"], errors="coerce").fillna(
            0.0
        )
        initial_sizes = pd.to_numeric(
            working_df["initial_size"], errors="coerce"
        ).fillna(0.0)
        invested_capital = entry_prices * initial_sizes

        working_df["roi"] = np.where(
            invested_capital > EPSILON, working_df["net_pnl"] / invested_capital, 0.0
        )

        roi_win_rate = metrics.calculate_win_rate(working_df["roi"])
        roi_risk_reward_ratio = metrics.calculate_risk_reward_ratio(working_df["roi"])
        kelly = metrics.calculate_kelly_criterion(roi_win_rate, roi_risk_reward_ratio)

        # Kelly Defaults (Heuristics)
        kelly_mean = kelly
        kelly_std = 0.0
        kelly_safe = max(0.0, kelly * 0.25)
        risk_of_ruin = 0.0

        if simulator:
            bootstrap = simulator.run_safe_kelly_bootstrap(working_df)
            kelly_mean = bootstrap["mean"]
            kelly_std = bootstrap["std"]
            kelly_safe = bootstrap["safe"]
            risk_of_ruin = simulator.run_ruin_probability(working_df, initial_capital)

        # Equity and Drawdown
        working_df["equity"] = initial_capital + working_df["net_pnl"].cumsum()
        maximum_drawdown = metrics.calculate_max_drawdown(
            working_df["equity"], initial_capital
        )

        # Performance Ratios
        sharpe_annualization_factor = 252.0
        if "exit_date" in working_df.columns and len(working_df) >= 2:
            exit_dates = pd.to_datetime(working_df["exit_date"])
            first_exit = exit_dates.min()
            last_exit = exit_dates.max()
            days_range = (last_exit - first_exit).days
            days_span = max(1.0, float(days_range))
            years_span = days_span / 365.25
            trades_per_year = len(working_df) / years_span
            sharpe_annualization_factor = min(252.0, max(1.0, float(trades_per_year)))

        sharpe_ratio = metrics.calculate_sharpe_ratio(
            working_df["net_pnl"],
            initial_capital,
            annualization_factor=sharpe_annualization_factor,
        )
        strategy_return = safe_divide(net_profit, initial_capital)
        exposure_efficiency = safe_divide(strategy_return, exposure_percentage)
        return_over_maximum_drawdown = safe_divide(
            strategy_return, abs(maximum_drawdown)
        )

        return BacktestMetrics(
            total_trades=total_trades,
            win_rate=win_rate,
            profit_factor=profit_factor,
            net_profit=net_profit,
            maximum_drawdown=maximum_drawdown,
            sharpe_ratio=sharpe_ratio,
            kelly_criterion=kelly,
            expectancy=expectancy,
            system_quality_number=sqn,
            average_win=average_win,
            average_loss=average_loss,
            average_maximum_adverse_excursion=average_maximum_adverse_excursion,
            average_maximum_favorable_excursion=average_maximum_favorable_excursion,
            risk_of_ruin=risk_of_ruin,
            benchmark_return=benchmark_return,
            strategy_return=strategy_return,
            kelly_mean=kelly_mean,
            kelly_std=kelly_std,
            kelly_safe=kelly_safe,
            market_exposure_pct=exposure_percentage,
            risk_adjusted_benchmark=benchmark_return * exposure_percentage,
            exposure_efficiency=exposure_efficiency,
            return_over_maximum_drawdown=return_over_maximum_drawdown,
            diversification_score=0.0,
        )

    def _empty_metrics(self, benchmark_return: float = 0.0) -> BacktestMetrics:
        """Returns a zeroed-out BacktestMetrics object."""
        return BacktestMetrics(
            total_trades=0,
            win_rate=0.0,
            profit_factor=0.0,
            net_profit=0.0,
            maximum_drawdown=0.0,
            sharpe_ratio=0.0,
            kelly_criterion=0.0,
            expectancy=0.0,
            system_quality_number=0.0,
            average_win=0.0,
            average_loss=0.0,
            average_maximum_adverse_excursion=0.0,
            average_maximum_favorable_excursion=0.0,
            risk_of_ruin=0.0,
            benchmark_return=benchmark_return,
            strategy_return=0.0,
            kelly_mean=0.0,
            kelly_std=0.0,
            kelly_safe=0.0,
            market_exposure_pct=0.0,
            risk_adjusted_benchmark=0.0,
            exposure_efficiency=0.0,
            return_over_maximum_drawdown=0.0,
            diversification_score=0.0,
        )


class MonteCarloSimulator:
    """Handles stochastic simulations."""

    def __init__(self, iterations: int = DEFAULT_BOOTSTRAP_ITERATIONS):
        self.iterations = iterations

    def run_safe_kelly_bootstrap(
        self, trades_dataframe: pd.DataFrame
    ) -> dict[str, float]:
        """Calculates Safe Kelly via Uniform Decay Bootstrapping."""
        if len(trades_dataframe) < MIN_TRADES_FOR_KELLY:
            return {"mean": 0.0, "std": 0.0, "safe": 0.0}

        entry_prices = (
            pd.to_numeric(trades_dataframe["entry_price"], errors="coerce")
            .fillna(0.0)
            .values
        )
        initial_sizes = (
            pd.to_numeric(trades_dataframe["initial_size"], errors="coerce")
            .fillna(0.0)
            .values
        )
        invested_capital = entry_prices * initial_sizes

        net_pnls = trades_dataframe["net_pnl"].values
        roi_values = np.where(
            invested_capital > EPSILON, net_pnls / invested_capital, 0.0
        )

        number_of_trades = len(roi_values)

        kelly_values = []
        for _ in range(self.iterations):
            # Uniform sampling (replace=True)
            sample = np.random.choice(roi_values, size=number_of_trades, replace=True)
            wins = sample[sample > EPSILON]
            losses = sample[sample < -EPSILON]

            if len(wins) == 0:
                kelly_values.append(0.0)
                continue

            if len(losses) == 0:
                kelly_values.append(0.99)
                continue

            win_probability = len(wins) / number_of_trades
            risk_reward_ratio = np.mean(wins) / abs(np.mean(losses))

            f_star = metrics.calculate_kelly_criterion(
                win_probability, risk_reward_ratio
            )
            kelly_values.append(f_star)

        return {
            "mean": float(np.mean(kelly_values)),
            "std": float(np.std(kelly_values, ddof=1)),
            "safe": safe_percentile(kelly_values, 25),
        }

    def run_ruin_probability(
        self, trades_df: pd.DataFrame, initial_capital: float
    ) -> float:
        """Calculates probability of 50% drawdown."""
        pnl_values = trades_df["net_pnl"].values
        if len(pnl_values) < 10:
            return 0.0
        ruin_count = 0
        for _ in range(1000):
            shuffled = np.random.permutation(pnl_values)
            equity = initial_capital + np.cumsum(shuffled)
            peak = np.maximum.accumulate(equity)
            drawdown = (equity - peak) / peak
            if any(drawdown < -0.5):
                ruin_count += 1
        return ruin_count / 1000.0

    def run_portfolio_simulation(
        self, matrix_dataframe: pd.DataFrame, iterations: int, initial_equity: float
    ) -> PortfolioMetrics:
        """Runs portfolio-wide bootstrap simulation across all strategies."""
        impact_columns = [
            col for col in matrix_dataframe.columns if col.startswith("impact_")
        ]
        count_columns = [
            col for col in matrix_dataframe.columns if col.startswith("count_")
        ]

        if not impact_columns:
            return PortfolioMetrics(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                100.0,
                0.0,
                0.0,
                0.0,
                {},
                0,
                {},
                0.0,
                {},
                1.0,
                {},
            )

        returns_matrix = matrix_dataframe[impact_columns].values
        number_of_rows = len(returns_matrix)
        if number_of_rows < 5:
            return PortfolioMetrics(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                100.0,
                0.0,
                0.0,
                0.0,
                {},
                0,
                {},
                0.0,
                {},
                1.0,
                {},
            )

        # Weighting: favor more recent performance slightly (Phase 6 tweak)
        weights = np.ones(number_of_rows)
        cutoff_index = int(number_of_rows * 0.5)
        weights[cutoff_index:] = 2.0
        probabilities = weights / np.sum(weights)

        kelly_values = []
        for _ in range(iterations):
            indices = np.random.choice(
                np.arange(number_of_rows),
                size=number_of_rows,
                replace=True,
                p=probabilities,
            )
            synthetic_impacts = returns_matrix[indices]
            portfolio_daily_returns = np.sum(synthetic_impacts, axis=1)

            winning_returns = portfolio_daily_returns[portfolio_daily_returns > EPSILON]
            losing_returns = portfolio_daily_returns[portfolio_daily_returns < -EPSILON]

            if len(winning_returns) == 0:
                kelly_values.append(0.0)
                continue

            win_probability = len(winning_returns) / number_of_rows
            reward_to_risk = (
                np.mean(winning_returns) / abs(np.mean(losing_returns))
                if len(losing_returns) > 0
                else 999.0
            )

            f_star = metrics.calculate_kelly_criterion(win_probability, reward_to_risk)
            kelly_values.append(f_star)

        mean_kelly = float(np.mean(kelly_values))
        safe_kelly = safe_percentile(kelly_values, 25)

        # Correlation Analysis
        fail_together_count, multi_strategy_days_count = 0, 0
        for row in returns_matrix:
            active_impacts = row[row != 0]
            if len(active_impacts) > 1:
                multi_strategy_days_count += 1
                if np.all(active_impacts < 0):
                    fail_together_count += 1

        # Concurrency Metrics
        total_concurrent_series = matrix_dataframe[count_columns].sum(axis=1)
        max_concurrent_trades = (
            int(total_concurrent_series.max())
            if not total_concurrent_series.empty
            else 0
        )
        max_concurrent_trades_days = (
            int((total_concurrent_series == max_concurrent_trades).sum())
            if max_concurrent_trades > 0
            else 0
        )
        percentile_95_concurrent_trades = (
            safe_percentile(total_concurrent_series.to_list(), 95)
            if not total_concurrent_series.empty
            else 0.0
        )

        # Per Strategy Allocation Limits
        max_trades_per_strategy = {}
        max_trades_per_strategy_days = {}
        percentile_95_trades_per_strategy = {}

        for col in count_columns:
            strategy_name = col.replace("count_", "")
            max_val = int(matrix_dataframe[col].max())
            max_trades_per_strategy[strategy_name] = max_val
            max_trades_per_strategy_days[strategy_name] = (
                int((matrix_dataframe[col] == max_val).sum()) if max_val > 0 else 0
            )
            percentile_95_trades_per_strategy[strategy_name] = safe_percentile(
                matrix_dataframe[col].to_list(), 95
            )

        # Leverage and Exposure Caps
        # Kelly of 1% (0.01) is benchmark relative distance.
        suggested_multiplier = (
            min(safe_kelly / 0.01, 15.0, 100.0 / max_concurrent_trades)
            if max_concurrent_trades > 0
            else 1.0
        )
        uncapped_multiplier = min(safe_kelly / 0.01, 15.0)

        # Capacity Ratios
        global_capacity_ratio = (
            max_concurrent_trades / percentile_95_concurrent_trades
            if percentile_95_concurrent_trades > 0
            else 1.0
        )

        return PortfolioMetrics(
            combined_mean_kelly=mean_kelly,
            safe_kelly_25=safe_kelly,
            correlation_fail_rate=(fail_together_count / multi_strategy_days_count)
            if multi_strategy_days_count > 0
            else 0.0,
            suggested_multiplier=round(suggested_multiplier, 2),
            leveraged_max_drawdown=0.0,
            max_concurrent_trades=max_concurrent_trades,
            max_total_exposure=round(suggested_multiplier * max_concurrent_trades, 1),
            uncapped_multiplier=round(uncapped_multiplier, 2),
            uncapped_max_total_exposure=round(
                uncapped_multiplier * max_concurrent_trades, 1
            ),
            uncapped_leveraged_max_drawdown=0.0,
            max_trades_per_strategy=max_trades_per_strategy,
            max_concurrent_trades_days=max_concurrent_trades_days,
            max_trades_per_strategy_days=max_trades_per_strategy_days,
            percentile_95_concurrent_trades=percentile_95_concurrent_trades,
            percentile_95_trades_per_strategy=percentile_95_trades_per_strategy,
            global_capacity_ratio=global_capacity_ratio,
            strategy_capacity_ratios={
                s: (
                    max_trades_per_strategy[s] / percentile_95_trades_per_strategy[s]
                    if percentile_95_trades_per_strategy.get(s, 0) > 0
                    else 1.0
                )
                for s in max_trades_per_strategy
            },
        )


class BacktestAnalytics:
    """Orchestrator for backtest analysis."""

    INITIAL_CAPITAL = 100_000.0

    def __init__(self, backtest_db_path: str, market_db_path: str):
        warnings.warn(
            "The Backtester module is deprecated. TradeManager is now the sole source of truth for OrderTypes, Limits, and Sizing.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.backtest_db = backtest_db_path
        self.market_db = market_db_path
        self.loader = BacktestDataLoader(backtest_db_path, market_db_path)
        self.calculator = MetricsCalculator()
        self.simulator = MonteCarloSimulator()

    def fetch_regime_data(self) -> pd.DataFrame:
        """Dashboard proxy for fetch_market_regime."""
        return self.loader.fetch_market_regime()

    def run_analysis(
        self,
        initial_capital: float | None = None,
        trades_dataframe: pd.DataFrame | None = None,
    ) -> BacktestMetrics:
        """Dashboard entry point for default full-backtest analysis."""
        capital = initial_capital or self.INITIAL_CAPITAL
        if trades_dataframe is None:
            trades_dataframe = self.loader.fetch_closed_trades()

        if trades_dataframe.empty:
            return self.calculator._empty_metrics()

        market_context = self.loader.calculate_exposure_and_benchmark(trades_dataframe)
        efficiency = self.loader.fetch_efficiency_metrics()

        metrics_result = self.calculator.calculate_trade_metrics(
            trades_dataframe=trades_dataframe,
            initial_capital=capital,
            exposure_percentage=market_context["exposure_pct"],
            benchmark_return=market_context["benchmark_return"],
            average_maximum_adverse_excursion=efficiency["avg_mae"],
            average_maximum_favorable_excursion=efficiency["avg_mfe"],
            simulator=self.simulator,
        )
        return metrics_result

    def run_strategy_analysis(
        self,
        initial_capital: float | None = None,
        trades_dataframe: pd.DataFrame | None = None,
    ) -> dict[str, BacktestMetrics]:
        """Analyzes performance broken down by individual strategy."""
        capital = initial_capital or self.INITIAL_CAPITAL

        if trades_dataframe is None:
            trades_dataframe = self.loader.fetch_closed_trades()

        if trades_dataframe.empty:
            return {}

        strategies = set(trades_dataframe["strategy"].unique())
        # Ensure common strategies are present for dashboard consistency
        strategies.update(
            [
                Strategies.DipBuyer.value,
                Strategies.TurnOverTiming_10.value,
                Strategies.TurnOverTiming_05.value,
                Strategies.TwoPercent.value,
            ]
        )

        results = {}
        for strategy in strategies:
            strategy_trades = trades_dataframe[trades_dataframe["strategy"] == strategy]
            if strategy_trades.empty:
                results[strategy] = self.calculator._empty_metrics()
                continue

            market_context = self.loader.calculate_exposure_and_benchmark(
                strategy_trades
            )

            # Use columns if present, else default to zero for legacy compatibility
            average_mae = (
                strategy_trades["mae"].mean()
                if "mae" in strategy_trades.columns
                else 0.0
            )
            average_mfe = (
                strategy_trades["mfe"].mean()
                if "mfe" in strategy_trades.columns
                else 0.0
            )

            metrics_result = self.calculator.calculate_trade_metrics(
                trades_dataframe=strategy_trades,
                initial_capital=capital,
                exposure_percentage=market_context["exposure_pct"],
                benchmark_return=market_context["benchmark_return"],
                average_maximum_adverse_excursion=average_mae,
                average_maximum_favorable_excursion=average_mfe,
                simulator=self.simulator,
            )
            results[strategy] = metrics_result
        return results

    def calculate_portfolio_kelly(
        self, iterations: int = 10_000, initial_equity: float | None = None
    ) -> PortfolioMetrics:
        equity = initial_equity or self.INITIAL_CAPITAL
        matrix_df = self.loader.fetch_synchronized_matrix(initial_capital=equity)
        if matrix_df.empty:
            return PortfolioMetrics(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0,
                0.0,
                0.0,
                0.0,
                0.0,
                {},
                0,
                {},
                0.0,
                {},
                1.0,
                {},
            )
        return self.simulator.run_portfolio_simulation(matrix_df, iterations, equity)

    def run_constrained_kelly_simulation(
        self,
        regime_dataframe: pd.DataFrame | None = None,
        initial_capital: float | None = None,
        include_history: bool = False,
        strategy_results: dict[str, BacktestMetrics] | None = None,
    ) -> dict[str, object]:
        """Runs paths with Kelly and Safety filters, adding Impact Analysis (Vectorized)."""
        capital = initial_capital or self.INITIAL_CAPITAL
        bootstrap_results = strategy_results or self.run_strategy_analysis()

        matrix = self.loader.fetch_synchronized_matrix(initial_capital=capital)
        if matrix.empty:
            return {}

        matrix = matrix.sort_values("date")

        # Prepare strategy multipliers (Kelly scores)
        strategy_names = [
            s.replace("impact_", "") for s in matrix.columns if s.startswith("impact_")
        ]
        portfolio_metrics = self.calculate_portfolio_kelly()

        multipliers_list = []
        for name in strategy_names:
            metrics_obj = bootstrap_results.get(name)
            if metrics_obj:
                # Normalize by strategy-specific 95th percentile if available, else global
                percentile_95 = portfolio_metrics.percentile_95_trades_per_strategy.get(
                    name
                )
                if not percentile_95 or percentile_95 < 1.0:
                    percentile_95 = portfolio_metrics.percentile_95_concurrent_trades

                # Sizing per trade = Strategy Kelly / Expected Concurrency
                unit_size = metrics_obj.kelly_safe / max(percentile_95, 1.0)
                multipliers_list.append(unit_size)
            else:
                multipliers_list.append(0.0)

        multipliers_array = np.array(multipliers_list)

        # 1. Calculate Theoretical Impacts (without safety)
        impact_columns = [f"impact_{s}" for s in strategy_names]
        count_columns = [f"count_{s}" for s in strategy_names]

        daily_pnl_theoretical = (matrix[impact_columns].values * multipliers_array).sum(
            axis=1
        ) * capital

        # 2. Safety Switch Logic
        is_safety_active = np.zeros(len(matrix), dtype=bool)
        vix_values = np.zeros(len(matrix))
        trigger_reasons = [""] * len(matrix)

        if regime_dataframe is not None:
            # Map regime data to matrix dates
            regime_map = regime_dataframe.set_index("date")
            matrix_dates = matrix["date"].tolist()
            for i, date_val in enumerate(matrix_dates):
                if date_val in regime_map.index:
                    is_safety_active[i] = regime_map.loc[date_val, "safety_active"]
                    vix_values[i] = regime_map.loc[date_val, "vix_close"]
                    trigger_reasons[i] = regime_map.loc[date_val, "trigger_reason"]

        # 3. Margin Interest
        # Total Gross Exposure = Sum(active_trades * multiplier)
        total_exposure = (matrix[count_columns].values * multipliers_array).sum(axis=1)
        margin_interest = np.where(
            total_exposure > 1.0, (total_exposure - 1.0) * capital * (0.06 / 360.0), 0.0
        )

        # Apply Margin Interest to Theoretical
        daily_pnl_theoretical -= margin_interest

        # Calculate Active PnL (Safety Switch blocks trading)
        daily_pnl_active = np.where(
            is_safety_active, -margin_interest, daily_pnl_theoretical
        )

        # 4. Cumulative Equity
        theoretical_equity = capital + daily_pnl_theoretical.cumsum()
        active_equity = capital + daily_pnl_active.cumsum()

        # Final Metrics
        final_equity = active_equity[-1]
        final_theoretical = theoretical_equity[-1]

        # Drawdown Calculation (incorporating initial capital for correct peak reference)
        active_equity_extended = pd.concat(
            [pd.Series([capital]), pd.Series(active_equity)], ignore_index=True
        )
        peak = active_equity_extended.cummax()
        drawdowns_extended = (active_equity_extended / peak - 1.0).fillna(0.0)
        drawdowns = drawdowns_extended.iloc[1:].reset_index(drop=True)
        min_drawdown = drawdowns.min()

        # 5. Safety Switch Event Tracking (Vectorized blocks)
        events = []
        saved_profit_series = daily_pnl_theoretical - daily_pnl_active

        # Identify changes in safety state
        safety_changes = np.diff(is_safety_active.astype(int), prepend=0)
        start_indices = np.where(safety_changes == 1)[0]
        end_indices = np.where(safety_changes == -1)[0]

        # Handle open end
        if len(start_indices) > len(end_indices):
            end_indices = np.append(end_indices, len(matrix) - 1)

        for start_idx, end_idx in zip(start_indices, end_indices):
            event_slice = saved_profit_series[start_idx : end_idx + 1]
            events.append(
                {
                    "start_date": matrix.iloc[start_idx]["date"],
                    "end_date": matrix.iloc[end_idx]["date"],
                    "reason": trigger_reasons[start_idx],
                    "days": int(end_idx - start_idx + 1),
                    "saved_profit": float(event_slice.sum()),
                }
            )

        # History for charts
        history_df = pd.DataFrame()
        if include_history:
            history_data = {
                "date": matrix["date"],
                "equity": active_equity,
                "drawdown_pct": drawdowns,
                "theoretical_equity": theoretical_equity,
                "safety_active": is_safety_active,
                "vix": vix_values,
                "margin_interest": margin_interest,
            }
            # Add strategy exposures
            for i, s in enumerate(strategy_names):
                history_data[f"exposure_{s}"] = np.where(
                    is_safety_active, 0.0, matrix[f"count_{s}"] * multipliers_array[i]
                )

            history_df = pd.DataFrame(history_data)

        # Impact Scores
        saved_loss = sum(
            abs(event["saved_profit"]) for event in events if event["saved_profit"] < 0
        )
        opportunity_cost = sum(
            event["saved_profit"] for event in events if event["saved_profit"] > 0
        )

        # Build final multipliers dict for compatibility
        multipliers_dictionary = {
            strategy: multipliers_array[index]
            for index, strategy in enumerate(strategy_names)
        }

        return {
            "capital": capital,
            "final_equity": float(final_equity),
            "theoretical_equity": float(final_theoretical),
            "max_drawdown": float(min_drawdown),
            "saved_loss": float(saved_loss),
            "opportunity_cost": float(opportunity_cost),
            "net_efficiency": float(saved_loss - opportunity_cost),
            "margin_interest_paid": float(margin_interest.sum()),
            "multipliers": multipliers_dictionary,
            "events": events,
            "daily_history": history_df,
        }

    def run_regime_comparison(
        self,
        regime_dataframe: pd.DataFrame,
        bootstrap_results: dict[str, BacktestMetrics],
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Compares Raw vs Safe strategy performance across market regimes.

        Returns:
            dict: {strategy_name: {regime: {metric: value}}}
        """
        matrix = self.loader.fetch_synchronized_matrix().sort_values("date")
        if matrix.empty:
            return {}

        known_strategies = [
            Strategies.DipBuyer.value,
            Strategies.TurnOverTiming_10.value,
            Strategies.TurnOverTiming_05.value,
            Strategies.TwoPercent.value,
        ]
        strategy_names = set(bootstrap_results.keys())
        strategy_names.update(known_strategies)

        results = {}
        for strategy_name in strategy_names:
            multiplier = (
                bootstrap_results.get(strategy_name).kelly_safe
                if strategy_name in bootstrap_results
                else 0.0
            )
            results[strategy_name] = {
                "Bull": {"Return": 0.0, "Sample_Count": 0.0},
                "Bear": {"Return": 0.0, "Sample_Count": 0.0},
                "High Vol": {"Return": 0.0, "Sample_Count": 0.0},
            }

            impact_col = f"impact_{strategy_name}"
            if impact_col not in matrix.columns:
                continue

            for _, row in matrix.iterrows():
                day_reg = regime_dataframe[regime_dataframe["date"] == row["date"]]
                if day_reg.empty:
                    continue

                reg = day_reg.iloc[0]
                impact = row[impact_col] * multiplier

                # Assign to regimes
                if reg["spy_uptrend"] and reg["vix_close"] < 25:
                    results[strategy_name]["Bull"]["Return"] += impact
                    results[strategy_name]["Bull"]["Sample_Count"] += 1
                elif not reg["spy_uptrend"]:
                    results[strategy_name]["Bear"]["Return"] += impact
                    results[strategy_name]["Bear"]["Sample_Count"] += 1
                elif reg["vix_close"] >= 25:
                    results[strategy_name]["High Vol"]["Return"] += impact
                    results[strategy_name]["High Vol"]["Sample_Count"] += 1

        return results

    def run_safety_tournament(
        self,
        regime_dataframe: pd.DataFrame,
        strategy_performance_map: dict[str, BacktestMetrics],
    ) -> list[dict[str, object]]:
        """Simulates and compares different safety switch configurations.

        Returns:
            list: List of dictionaries containing tournament results for each logic.
        """
        configurations = [
            {"name": "Baseline (None)", "trigger": lambda row: False},
            {"name": "RSI(2) < 10", "trigger": lambda row: row["rsi2"] < 10.0},
            {"name": "RSI(2) < 8", "trigger": lambda row: row["rsi2"] < 8.0},
            {"name": "RSI(2) < 6", "trigger": lambda row: row["rsi2"] < 6.0},
            {"name": "VIX < 10", "trigger": lambda row: row["vix_close"] < 10.0},
            {"name": "VIX < 12", "trigger": lambda row: row["vix_close"] < 12.0},
            {"name": "VIX < 14", "trigger": lambda row: row["vix_close"] < 14.0},
            {"name": "VIX < 16", "trigger": lambda row: row["vix_close"] < 16.0},
        ]

        tournament_results = []

        # We need a baseline equity to compare efficiency
        baseline_sim = self.run_constrained_kelly_simulation(
            regime_dataframe=regime_dataframe.assign(safety_active=False),
            strategy_results=strategy_performance_map,
        )
        baseline_return = baseline_sim["final_equity"]

        for config in configurations:
            # Create temporary regime DF with custom trigger
            temp_regime = regime_dataframe.copy()
            temp_regime["safety_active"] = temp_regime.apply(config["trigger"], axis=1)

            sim = self.run_constrained_kelly_simulation(
                regime_dataframe=temp_regime, strategy_results=strategy_performance_map
            )

            net_efficiency = sim["final_equity"] - baseline_return

            # Grade based on efficiency
            grade = "C (Neutral)"
            if net_efficiency > 1000:
                grade = "A (Winner)"
            elif net_efficiency > 0:
                grade = "B (Helper)"
            elif net_efficiency < -2000:
                grade = "F (Useless)"
            elif net_efficiency < 0:
                grade = "D (Too Late)"

            tournament_results.append(
                {
                    "logic": config["name"],
                    "total_return": sim["final_equity"],
                    "max_drawdown": sim["max_drawdown"],
                    "net_efficiency": net_efficiency,
                    "grade": grade,
                }
            )

        return tournament_results

    def calculate_portfolio_funnel(
        self,
        strategy_results: dict[str, BacktestMetrics],
        portfolio_metrics: PortfolioMetrics | None = None,
    ) -> list[dict[str, object]]:
        """Calculates funnel allocation via quality filters and normalization.

        Args:
            strategy_results: Map of strategy names to their performance metrics.
            portfolio_metrics: Global portfolio metrics (for Phase 7 ratios).

        Returns:
            list: List of dictionaries for the portfolio funnel table.
        """
        if not strategy_results:
            return []

        # 1. Calculate Sum of Raw Kelly (Safe)
        sum_kelly_total = sum(m.kelly_safe for m in strategy_results.values())
        if sum_kelly_total < EPSILON:
            sum_kelly_total = 1.0  # Avoid division by zero

        funnel_data = []
        active_kelly_sum = 0.0

        # 2. Apply Quality Filters (Profit Factor >= 1.2 AND Total Trades >= 50)
        for strategy_name, metrics_obj in strategy_results.items():
            is_active = (
                metrics_obj.profit_factor >= 1.2 and metrics_obj.total_trades >= 50
            )
            status = "ACTIVE" if is_active else "REJECTED"

            reason = ""
            if not is_active:
                reasons_list = []
                if metrics_obj.profit_factor < 1.2:
                    reasons_list.append(f"Low PF ({metrics_obj.profit_factor:.2f})")
                if metrics_obj.total_trades < 50:
                    reasons_list.append(f"Low trades ({metrics_obj.total_trades})")
                reason = " & ".join(reasons_list)

            raw_share = (metrics_obj.kelly_safe / sum_kelly_total) * 100

            funnel_data.append(
                {
                    "name": strategy_name,
                    "kelly": metrics_obj.kelly_safe,
                    "raw_share": raw_share,
                    "status": status,
                    "reason": reason,
                    "final_allocation": 0.0,  # Placeholder for normalization
                }
            )

            if is_active:
                active_kelly_sum += metrics_obj.kelly_safe

        # 3. Re-normalization for Active Strategies
        if active_kelly_sum > 0:
            for item in funnel_data:
                if item["status"] == "ACTIVE":
                    item["final_allocation"] = (item["kelly"] / active_kelly_sum) * 100

        # --- Phase 7: Dynamic Portfolio (95th Percentile) Allocation ---
        if portfolio_metrics:
            p95_weighted_sum = 0.0
            ratios = portfolio_metrics.strategy_capacity_ratios

            for item in funnel_data:
                item["final_allocation_p95"] = 0.0  # Default
                if item["status"] == "ACTIVE":
                    # Formula: Weight * sqrt(Ratio)
                    ratio = ratios.get(item["name"], 1.0)
                    multiplier = np.sqrt(ratio)
                    item["p95_weight"] = item["final_allocation"] * multiplier
                    p95_weighted_sum += item["p95_weight"]
                else:
                    item["p95_weight"] = 0.0

            # Normalize P95Weights to 100%
            if p95_weighted_sum > 0:
                for item in funnel_data:
                    if item["status"] == "ACTIVE":
                        item["final_allocation_p95"] = (
                            item["p95_weight"] / p95_weighted_sum
                        ) * 100

                    # Clean up temp field
                    item.pop("p95_weight", None)

        # Sort by final allocation descending
        funnel_data.sort(key=lambda x: x["final_allocation"], reverse=True)

        return funnel_data

    def fetch_benchmark_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        initial_capital: float | None = None,
    ) -> pd.DataFrame:
        capital = initial_capital or self.INITIAL_CAPITAL
        return self.loader.fetch_benchmark_data(symbol, start_date, end_date, capital)

    def get_equity_curve(self, initial_capital: float | None = None) -> pd.DataFrame:
        """Calculates a rolling equity curve for the dashboard."""
        capital = initial_capital or self.INITIAL_CAPITAL
        trades_dataframe = self.loader.fetch_closed_trades()
        if trades_dataframe.empty:
            return pd.DataFrame()

        working_df = trades_dataframe.copy()
        costs = self.calculator.cost_model.calculate_cost(working_df)
        working_df["net_pnl"] = working_df["realized_pnl"] - costs
        working_df["exit_date"] = pd.to_datetime(working_df["exit_date"])
        working_df["equity"] = capital + working_df["net_pnl"].cumsum()
        equity_extended = pd.concat(
            [pd.Series([capital]), working_df["equity"]], ignore_index=True
        )
        peak_extended = equity_extended.cummax()
        drawdown_extended = (equity_extended - peak_extended) / peak_extended
        working_df["peak"] = peak_extended.iloc[1:].reset_index(drop=True).values
        working_df["drawdown_percentage"] = (
            drawdown_extended.iloc[1:].reset_index(drop=True).values
        )
        return working_df[["exit_date", "net_pnl", "equity", "drawdown_percentage"]]

    def get_trade_lists(self) -> dict[str, list[dict[str, object]]]:
        connection = self.loader._get_connection()
        try:

            def fetch(query: str) -> list[dict[str, object]]:
                dataframe = connection.sql(query).df()
                if dataframe.empty:
                    return []
                dataframe["realized_pnl"] = dataframe["realized_pnl"].round(2)
                return dataframe.to_dict(orient="records")

            return {
                "recent": fetch(
                    "SELECT symbol, strategy, entry_date, exit_date, entry_price, "
                    "exit_price, realized_pnl, exit_reason FROM backtest.trades "
                    "WHERE status='CLOSED' ORDER BY exit_date DESC LIMIT 20"
                ),
                "top": fetch(
                    "SELECT symbol, strategy, entry_date, exit_date, entry_price, "
                    "exit_price, realized_pnl, exit_reason FROM backtest.trades "
                    "WHERE status='CLOSED' AND realized_pnl > 0 "
                    "ORDER BY realized_pnl DESC LIMIT 10"
                ),
                "worst": fetch(
                    "SELECT symbol, strategy, entry_date, exit_date, entry_price, "
                    "exit_price, realized_pnl, exit_reason FROM backtest.trades "
                    "WHERE status='CLOSED' AND realized_pnl < 0 "
                    "ORDER BY realized_pnl ASC LIMIT 10"
                ),
            }
        finally:
            connection.close()

    def get_granular_persistence_data(
        self, constrained_simulation: dict[str, object]
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Extracts and formats data for database persistence.

        Args:
            constrained_simulation: Results from run_constrained_kelly_simulation.

        Returns:
            tuple: (equity_curves_df, regime_df, exposure_df)
        """
        history_df = constrained_simulation.get("daily_history")
        if history_df is None or history_df.empty:
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        # 1. Equity Curves (Split into Kelly/Theoretical and Safety/Actual)
        # A. Safety Curve (Actual Result)
        equity_safe = history_df[["date", "equity", "drawdown_pct"]].copy()
        equity_safe["is_benchmark"] = 0
        equity_safe["strategy_name"] = "Safety"

        # B. Kelly Curve (Theoretical - Unconstrained)
        equity_kelly = history_df[["date", "theoretical_equity"]].copy()
        equity_kelly.rename(columns={"theoretical_equity": "equity"}, inplace=True)
        equity_kelly["is_benchmark"] = 0
        equity_kelly["strategy_name"] = "Kelly"

        # Calculate Theoretical Drawdown (incorporating initial capital for correct peak reference)
        capital = constrained_simulation.get("capital", self.INITIAL_CAPITAL)
        theoretical_equity_extended = pd.concat(
            [pd.Series([capital]), equity_kelly["equity"]], ignore_index=True
        )
        theoretical_peak = theoretical_equity_extended.cummax()
        theoretical_drawdowns_extended = (
            theoretical_equity_extended / theoretical_peak - 1.0
        ).fillna(0.0)
        equity_kelly["drawdown_pct"] = theoretical_drawdowns_extended.iloc[
            1:
        ].reset_index(drop=True)

        equity_df = pd.concat([equity_safe, equity_kelly], ignore_index=True)

        # 2. Regime Data
        regime_df = history_df[["date", "vix", "safety_active"]].copy()
        regime_df.rename(columns={"vix": "vix_close"}, inplace=True)

        # 3. Exposure Heatmap
        exposure_cols = [c for c in history_df.columns if c.startswith("exposure_")]

        # Vectorized reshaping of exposure data
        exposure_df = history_df[["date"] + exposure_cols].melt(
            id_vars=["date"], var_name="strategy_name", value_name="exposure_value"
        )
        exposure_df["strategy_name"] = exposure_df["strategy_name"].str.replace(
            "exposure_", "", regex=False
        )

        return equity_df, regime_df, exposure_df

    # Compatibility
    def _fetch_sorted_trades(
        self, connection: Any, strategy: str | None = None
    ) -> pd.DataFrame:
        return self.loader.fetch_closed_trades(strategy_filter=strategy)

    def get_all_closed_trades(self) -> pd.DataFrame:
        """Legacy compatibility proxy."""
        return self.loader.fetch_closed_trades()


@dataclass(frozen=True)
class WFAMetrics:
    win_rate: float
    profit_factor: float
    kelly_criterion: float
    expectancy: float
    trade_count: int


class WalkForwardAnalyzer:
    def __init__(self, trades_dataframe: pd.DataFrame):
        warnings.warn(
            "The Backtester module is deprecated. TradeManager is now the sole source of truth for OrderTypes, Limits, and Sizing.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.trades = trades_dataframe.sort_values("exit_date").reset_index(drop=True)

    def _calculate_metrics(self, dataframe: pd.DataFrame) -> WFAMetrics:
        """Calculates performance metrics for a specific walk-forward window."""
        if dataframe.empty:
            return WFAMetrics(0.0, 0.0, 0.0, 0.0, 0)

        winning_trades = dataframe[dataframe["realized_pnl"] > EPSILON]
        losing_trades = dataframe[dataframe["realized_pnl"] <= -EPSILON]

        win_rate = metrics.calculate_win_rate(dataframe["realized_pnl"])
        average_win = (
            winning_trades["realized_pnl"].mean() if not winning_trades.empty else 0.0
        )
        average_loss = (
            abs(losing_trades["realized_pnl"].mean())
            if not losing_trades.empty
            else 0.0
        )

        profit_factor = metrics.calculate_profit_factor(dataframe["realized_pnl"])
        risk_reward_ratio = (
            average_win / average_loss if average_loss > EPSILON else 0.0
        )
        kelly = metrics.calculate_kelly_criterion(win_rate, risk_reward_ratio)
        expectancy = (win_rate * average_win) - ((1.0 - win_rate) * average_loss)

        return WFAMetrics(win_rate, profit_factor, kelly, expectancy, len(dataframe))

    def run_analysis(
        self,
        train_window_pct: float = 0.6,
        test_window_pct: float = 0.1,
        step_pct: float = 0.1,
        regime_dataframe: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        total_trades_count = len(self.trades)
        if total_trades_count < 50:
            return pd.DataFrame()

        training_window_size = int(total_trades_count * train_window_pct)
        testing_window_size = int(total_trades_count * test_window_pct)
        step_window_size = int(total_trades_count * step_pct)

        results = []
        iteration_start = 0
        while (
            iteration_start + training_window_size + testing_window_size
        ) <= total_trades_count:
            train_end = iteration_start + training_window_size
            test_end = iteration_start + training_window_size + testing_window_size

            training_dataframe = self.trades.iloc[iteration_start:train_end]
            testing_dataframe = self.trades.iloc[train_end:test_end]

            in_sample = self._calculate_metrics(training_dataframe)
            out_of_sample = self._calculate_metrics(testing_dataframe)

            # Degradation calculation
            degradation = 0.0
            if abs(in_sample.kelly_criterion) > EPSILON:
                degradation = (
                    in_sample.kelly_criterion - out_of_sample.kelly_criterion
                ) / in_sample.kelly_criterion

            recommendation = "NEUTRAL"
            if degradation < 0.20 and out_of_sample.profit_factor > 1.5:
                recommendation = "STABLE - 100% Allocation"
            elif degradation > 0.20 or out_of_sample.profit_factor < 1.2:
                recommendation = "WARNING - 50% Allocation"

            if out_of_sample.expectancy < 0:
                recommendation = "CRITICAL - DO NOT TRADE"

            start_date_val = training_dataframe["exit_date"].min()
            end_date_val = testing_dataframe["exit_date"].max()
            row = {
                "Window": f"{start_date_val.date()} -> {end_date_val.date()}",
                "IS_Kelly": in_sample.kelly_criterion,
                "OOS_Kelly": out_of_sample.kelly_criterion,
                "OOS_PF": out_of_sample.profit_factor,
                "Degradation": degradation,
                "Recommendation": recommendation,
            }

            if regime_dataframe is not None and not regime_dataframe.empty:
                # Optimized regime lookup
                test_start = pd.to_datetime(
                    testing_dataframe["exit_date"].min()
                ).tz_localize(None)
                test_end_dt = pd.to_datetime(
                    testing_dataframe["exit_date"].max()
                ).tz_localize(None)

                # Internal copy for performance if needed
                reg_d = regime_dataframe.copy()
                reg_d["date"] = pd.to_datetime(reg_d["date"]).dt.tz_localize(None)

                mask = (reg_d["date"] >= test_start) & (reg_d["date"] <= test_end_dt)
                win_reg = reg_d.loc[mask]

                if not win_reg.empty:
                    row["Avg_VIX"] = win_reg["vix_close"].mean()
                    row["Uptrend_Pct"] = win_reg["spy_uptrend"].mean()

            results.append(row)
            iteration_start += step_window_size

        return pd.DataFrame(results)


class NoiseTester:
    def __init__(self, analytics: "BacktestAnalytics", trades_dataframe: pd.DataFrame):
        self.analytics = analytics
        self.trades_dataframe = trades_dataframe.copy()

    def run_stress_test(self, n_simulations: int = 50) -> dict[str, float]:
        stress_results = []
        winners = self.trades_dataframe[self.trades_dataframe["realized_pnl"] > 0]
        losers = self.trades_dataframe[self.trades_dataframe["realized_pnl"] <= 0]
        for _ in range(n_simulations):
            keep_winners = (
                winners.sample(frac=0.9, replace=False)
                if not winners.empty
                else winners
            )
            new_losers = (
                pd.concat([losers, losers.sample(frac=0.1, replace=True)])
                if not losers.empty
                else losers
            )
            stress_dataframe = (
                pd.concat([keep_winners, new_losers])
                .sort_values("exit_date")
                .reset_index(drop=True)
            )
            # Minimal proxy for stress run
            metrics_obj = self.analytics.run_analysis(trades_dataframe=stress_dataframe)
            stress_results.append(
                {
                    "max_drawdown": metrics_obj.maximum_drawdown,
                    "final_equity": 100000 + metrics_obj.net_profit,
                }
            )

        drawdowns = [r["max_drawdown"] for r in stress_results]
        equities = [r["final_equity"] for r in stress_results]
        return {
            "simulations": float(n_simulations),
            "avg_max_drawdown": float(np.mean(drawdowns)),
            "worst_max_drawdown": float(np.min(drawdowns)),
            "failure_rate": float(np.mean(np.array(drawdowns) < -0.30)),
            "avg_equity": float(np.mean(equities)),
        }


# --- Advanced Analytics Classes ---


class PerformancePeriods:
    """Analyze performance across different time horizons."""

    def calculate_rolling_metrics(
        self, trades_dataframe: pd.DataFrame, windows: list[int] | None = None
    ) -> pd.DataFrame:
        """Calculates rolling Sharpe Ratio, Drawdown, and Win Rate.

        Args:
            trades_dataframe: DataFrame containing closed trades.
            windows: List of trade counts for rolling windows (e.g. [30, 90]).

        Returns:
            DataFrame with rolling metrics.
        """
        if windows is None:
            windows = [30, 90, 180, 365]

        if trades_dataframe.empty:
            return pd.DataFrame()

        working_df = trades_dataframe.copy()
        if "exit_date" in working_df.columns:
            working_df["exit_date"] = pd.to_datetime(working_df["exit_date"])
            working_df = working_df.sort_values("exit_date")
        else:
            return pd.DataFrame()

        rolling_results = []
        for window_size in windows:
            if len(working_df) < window_size:
                continue

            # Rolling window of N trades
            for i in range(len(working_df) - window_size + 1):
                window_trades = working_df.iloc[i : i + window_size]

                # Use metrics tool for pure calculations
                win_rate = metrics.calculate_win_rate(window_trades["realized_pnl"])

                # Trade-based Sharpe Approximation (using nominal values as return proxy)
                pnl_series = window_trades["realized_pnl"]
                mean_pnl = pnl_series.mean()
                standard_deviation_pnl = pnl_series.std(ddof=1)
                sharpe_proxy = safe_divide(mean_pnl, standard_deviation_pnl)

                # Local Peak-to-Trough Drawdown (PnL based, starting at 0.0)
                cumulative_pnl_extended = pd.concat(
                    [pd.Series([0.0]), pnl_series.cumsum()], ignore_index=True
                )
                peak_pnl_extended = cumulative_pnl_extended.cummax()
                drawdown_pnl_extended = cumulative_pnl_extended - peak_pnl_extended
                maximum_drawdown_pnl = drawdown_pnl_extended.iloc[1:].min()

                rolling_results.append(
                    {
                        "window_trades": window_size,
                        "end_date": window_trades["exit_date"].max(),
                        "sharpe_proxy": sharpe_proxy,
                        "max_drawdown_pnl": maximum_drawdown_pnl,
                        "win_rate": win_rate,
                    }
                )

        return pd.DataFrame(rolling_results)

    def get_calendar_returns(self, trades_df: pd.DataFrame) -> pd.DataFrame:
        """Monthly/Quarterly/Yearly breakdowns."""
        if trades_df.empty:
            return pd.DataFrame()

        df = trades_df.copy()
        df["date"] = pd.to_datetime(df["exit_date"])
        df["year"] = df["date"].dt.year
        df["quarter"] = df["date"].dt.quarter
        df["month"] = df["date"].dt.month

        return df.groupby(["year", "quarter", "month"]).agg(
            {
                "realized_pnl": ["sum", "mean", "count"],
                "symbol": "nunique",
            }
        )


class TradeQualityAnalyzer:
    def score_dataframe(self, trades_dataframe: pd.DataFrame) -> pd.DataFrame:
        """Vectorized scoring of trades."""
        if trades_dataframe.empty:
            return pd.DataFrame()

        df = trades_dataframe.copy()

        # 1. Entry Quality (Vectorized)
        df["score_entry"] = np.where(df["realized_pnl"] > 0, 25.0, 10.0)

        # 2. Risk Management (Vectorized)
        risk = (
            df["entry_price"] - df.get("initial_stop", df["entry_price"])
        ).abs() * df.get("initial_size", 1.0)
        r_multiple = np.where(risk > EPSILON, df["realized_pnl"] / risk, 0.0)

        df["score_risk"] = np.select(
            [r_multiple > 2.0, r_multiple > 1.0, r_multiple > 0.0, r_multiple > -1.0],
            [20.0, 18.0, 15.0, 10.0],
            default=0.0,
        )

        # 3. Exit Efficiency (Vectorized)
        maximum_favorable_excursion = df.get("mfe", 0.0)
        max_profit_potential = (
            df["entry_price"]
            * df.get("initial_size", 1.0)
            * maximum_favorable_excursion
        )
        capture_ratio = np.where(
            max_profit_potential > EPSILON,
            df["realized_pnl"] / max_profit_potential,
            0.0,
        )

        score_mfe = np.select(
            [capture_ratio > 0.8, capture_ratio > 0.6, capture_ratio > 0.4],
            [30.0, 25.0, 20.0],
            default=10.0,
        )

        # Fallback to reason if MFE is missing/zero
        reason_score = np.where(
            df["exit_reason"].str.upper().str.contains("TARGET|PROFIT|LIMIT", na=False),
            30.0,
            np.where(
                df["exit_reason"].str.upper().str.contains("STOP", na=False), 10.0, 15.0
            ),
        )

        df["score_exit"] = np.where(
            maximum_favorable_excursion > EPSILON, score_mfe, reason_score
        )

        # 4. Market Alignment (Neutral fallback for now as it's typically complex)
        df["score_context"] = 15.0

        df["total_score"] = (
            df["score_entry"]
            + df["score_risk"]
            + df["score_exit"]
            + df["score_context"]
        )

        # Grading
        df["grade"] = pd.cut(
            df["total_score"],
            bins=[0, 60, 70, 80, 83, 87, 90, 93, 97, 100],
            labels=["F", "D", "C", "B-", "B", "B+", "A-", "A", "A+"],
            include_lowest=True,
        ).astype(str)

        # Weakest Link
        components = ["score_entry", "score_risk", "score_exit", "score_context"]
        df["weakest_link"] = df[components].idxmin(axis=1)

        return df

    def score_trade(self, trade: dict, market_context: dict | None = None) -> dict:
        """Assigns quality score 0-100 (Legacy/Single)."""
        score_components = {
            "entry_quality": self._score_entry(trade),  # 0-30 points
            "risk_management": self._score_risk(trade),  # 0-20 points
            "exit_efficiency": self._score_exit(trade),  # 0-30 points
            "market_alignment": self._score_context(
                trade, market_context or {}
            ),  # 0-20 points
        }

        total_score = sum(score_components.values())
        grade = self._get_grade(total_score)

        return {
            "trade_id": trade.get("id"),
            "total_score": total_score,
            "grade": grade,
            "components": score_components,
            "weakest_link": min(score_components, key=score_components.get),
        }

    def _score_entry(self, trade: dict) -> float:
        # If PnL > 0, entry was effectively good enough
        pnl = trade.get("realized_pnl", 0.0)
        return 25.0 if pnl > 0 else 10.0

    def _score_risk(self, trade: dict) -> float:
        # Check R-Multiple
        risk = (
            trade.get("entry_price", 0.0) - trade.get("initial_stop", 0.0)
        ) * trade.get("initial_size", 0.0)
        risk = abs(risk)
        if risk < EPSILON:
            # Fallback if no risk defined: Use 1% of plausible capital or just nominal
            return 10.0

        pnl = trade.get("realized_pnl", 0.0)
        r_multiple = pnl / risk

        if r_multiple > 2.0:
            return 20.0
        if r_multiple > 1.0:
            return 18.0
        if r_multiple > 0.0:
            return 15.0
        if r_multiple > -1.0:
            return 10.0  # Normal loss
        return 0.0  # Busted risk

    def _score_context(self, trade: dict, context: dict) -> float:
        # If context missing, assume neutral
        if not context:
            # Boost if winning trade to avoid punishing winners
            return 15.0 if trade.get("realized_pnl", 0.0) > 0 else 10.0

        # Placeholder for real context logic (e.g. VIX < 20 -> +5)
        return 15.0

    def _get_grade(self, score: float) -> str:
        if score >= 97:
            return "A+"
        if score >= 93:
            return "A"
        if score >= 90:
            return "A-"
        if score >= 87:
            return "B+"
        if score >= 83:
            return "B"
        if score >= 80:
            return "B-"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"

    def _score_exit(self, trade: dict) -> float:
        """Score exit based on MFE capture or Exit Reason."""
        mfe = trade.get("mfe", 0.0)

        # 1. MFE Capture Logic (if available)
        if mfe > EPSILON:
            actual_profit = trade.get("realized_pnl", 0.0)
            max_potential = (
                trade.get("entry_price", 0.0) * trade.get("initial_size", 0.0) * mfe
            )
            capture_ratio = safe_divide(actual_profit, max_potential)

            if capture_ratio > 0.8:
                return 30.0
            elif capture_ratio > 0.6:
                return 25.0
            elif capture_ratio > 0.4:
                return 20.0
            else:
                return 10.0

        # 2. Fallback: Exit Reason
        reason = trade.get("exit_reason", "").upper()
        if "TARGET" in reason or "PROFIT" in reason or "LIMIT" in reason:
            return 30.0  # Perfect execution
        if "STOP" in reason and "TRAIL" in reason:
            return 20.0  # Trailing stop is okay
        if "STOP" in reason:
            return 10.0  # Stopped out
        if "TIME" in reason or "EXPIRED" in reason:
            return 15.0  # Timed out

        return 15.0  # Unknown/Neutral


class DiversificationAnalyzer:
    def calculate_strategy_correlations(
        self, trades_by_strategy: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """Calculate pairwise correlations between strategies."""
        daily_returns = {}
        for strategy_name, trades_df in trades_by_strategy.items():
            if trades_df.empty:
                continue
            df = trades_df.copy()
            df["exit_date"] = pd.to_datetime(df["exit_date"])
            df = df.sort_values("exit_date")
            df.set_index("exit_date", inplace=True)
            daily_returns[strategy_name] = df["realized_pnl"].resample("D").sum()

        if not daily_returns:
            return pd.DataFrame()

        returns_df = pd.DataFrame(daily_returns).fillna(0)
        return returns_df.corr()

    def calculate_diversification_score(self, correlations: pd.DataFrame) -> float:
        """Score 0-100 where 100 = perfectly uncorrelated strategies."""
        if correlations.empty or len(correlations) < 2:
            return 0.0

        upper_triangle = np.triu(correlations.values, k=1)
        elements = upper_triangle[np.triu_indices(len(correlations), k=1)]
        avg_correlation = np.mean(elements) if len(elements) > 0 else 0.0

        score = (1.0 - avg_correlation) * 50.0 + 50.0
        return max(0.0, min(100.0, score))


class RegimeDetector:
    """Proactive regime detection using multiple indicators."""

    def detect_current_regime(self, market_data: pd.DataFrame) -> str:
        """Returns: BULL_QUIET, BULL_VOLATILE, BEAR_QUIET, BEAR_VOLATILE, SIDEWAYS."""
        if market_data.empty:
            return "UNKNOWN"

        current = market_data.iloc[-1]

        close = current.get("close", 0.0)
        sma200 = current.get("sma200", 0.0)
        trend = "BULL" if close > sma200 else "BEAR"

        vol_val = current.get("vix_close", 0.0)
        volatility = "QUIET"
        if vol_val > 20.0:
            volatility = "VOLATILE"

        rsi = current.get("rsi7", 50.0)
        if 45 <= rsi <= 55:
            return "SIDEWAYS"

        return f"{trend}_{volatility}"

    def get_strategy_allocations(self, regime: str) -> dict[str, float]:
        """Returns optimal % allocation per strategy for current regime."""
        allocations = {
            "BULL_QUIET": {
                Strategies.DipBuyer: 0.40,
                Strategies.TurnOverTiming_10: 0.35,
                Strategies.TwoPercent: 0.25,
            },
            "BULL_VOLATILE": {
                Strategies.DipBuyer: 0.50,
                Strategies.TurnOverTiming_10: 0.20,
                Strategies.TwoPercent: 0.30,
            },
            "BEAR_QUIET": {
                Strategies.DipBuyer: 0.10,
                Strategies.TurnOverTiming_10: 0.20,
                Strategies.TwoPercent: 0.10,
            },
        }
        return allocations.get(regime, {"default": 1.0})


class EquityDiagnostics:
    """Advanced equity curve analysis."""

    def detect_regime_shifts(self, equity_series: pd.Series) -> list[dict]:
        """Detect periods where strategy behavior changed."""
        if len(equity_series) < 60:
            return []

        returns = equity_series.pct_change().fillna(0)
        rolling_mean = returns.rolling(60).mean()
        rolling_std = returns.rolling(60).std()

        rolling_sharpe = safe_divide(rolling_mean, rolling_std) * np.sqrt(252)
        sharpe_std = rolling_sharpe.std(ddof=1)

        shifts = []
        for i in range(len(rolling_sharpe) - 1):
            curr = rolling_sharpe.iloc[i]
            nxt = rolling_sharpe.iloc[i + 1]
            if pd.isna(curr) or pd.isna(nxt):
                continue

            if abs(nxt - curr) > sharpe_std:
                shifts.append(
                    {
                        "date": rolling_sharpe.index[i + 1],
                        "direction": "IMPROVEMENT" if nxt > curr else "DETERIORATION",
                        "magnitude": abs(nxt - curr),
                    }
                )
        return shifts

    def calculate_ulcer_index(self, equity_series: pd.Series) -> float:
        """Calculates the Ulcer Index - measures depth and duration of drawdowns."""
        return metrics.calculate_ulcer_index(equity_series)
