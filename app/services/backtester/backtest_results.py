import sqlite3
import logging
from typing import Any, TypedDict
import pandas as pd
from ...models import BacktestMetrics, PortfolioMetrics

logger = logging.getLogger(__name__)


class SafetyEvent(TypedDict):
    start_date: str
    end_date: str | None
    reason: str
    days: int
    saved_profit: float


class SimulationImpact(TypedDict):
    final_equity: float
    theoretical_equity: float
    saved_loss: float
    opportunity_cost: float
    net_efficiency: float
    margin_interest_paid: float
    events: list[SafetyEvent]


class FunnelData(TypedDict):
    name: str
    kelly: float
    raw_share: float
    status: str
    reason: str
    final_allocation: float


class ResultsPersistence:
    """Handles persistence of backtest results into the existing backtest.db."""

    def __init__(self, db_path: str) -> None:
        """Initializes the persistence layer.

        Args:
            db_path: Path to the existing backtest.db file.
        """
        self.db_path = db_path
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        """Creates a standard sqlite3 connection."""
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_schema(self) -> None:
        """Creates necessary tables if they do not exist."""
        with self._get_connection() as connection:
            cursor = connection.cursor()

            # 1. Backtest Runs Summary
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_date TEXT DEFAULT (datetime('now')),
                    start_date TEXT,
                    end_date TEXT,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    net_profit REAL,
                    expectancy REAL,
                    maximum_drawdown REAL,
                    sharpe_ratio REAL,
                    sqn REAL,
                    kelly_safe REAL,
                    strategy_return REAL,
                    benchmark_return REAL,
                    risk_of_ruin REAL,
                    average_win REAL,
                    average_loss REAL,
                    average_mae REAL,
                    average_mfe REAL,
                    kelly_mean REAL,
                    kelly_std REAL,
                    market_exposure_pct REAL,
                    risk_adjusted_benchmark REAL,
                    exposure_efficiency REAL,
                    return_over_max_drawdown REAL,
                    diversification_score REAL
                )
            """)

            # 2. Strategy Metrics
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS strategy_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    strategy_name TEXT,
                    total_trades INTEGER,
                    win_rate REAL,
                    profit_factor REAL,
                    net_profit REAL,
                    maximum_drawdown REAL,
                    sqn REAL,
                    kelly_safe REAL,
                    risk_of_ruin REAL,
                    average_win REAL,
                    average_loss REAL,
                    market_exposure_pct REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 3. Portfolio Simulations (Kelly)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_simulations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    combined_mean_kelly REAL,
                    safe_kelly_25 REAL,
                    suggested_multiplier REAL,
                    leveraged_max_drawdown REAL,
                    max_total_exposure REAL,
                    correlation_fail_rate REAL,
                    max_concurrent_trades INTEGER,
                    uncapped_multiplier REAL,
                    uncapped_max_total_exposure REAL,
                    uncapped_leveraged_max_drawdown REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 4. Walk-Forward Windows
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS walk_forward_windows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    window_label TEXT,
                    is_kelly REAL,
                    oos_kelly REAL,
                    oos_pf REAL,
                    avg_vix REAL,
                    uptrend_pct REAL,
                    degradation REAL,
                    recommendation TEXT,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 5. Stress Tests (Monte Carlo)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS stress_tests (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    avg_max_drawdown REAL,
                    worst_max_drawdown REAL,
                    failure_rate REAL,
                    avg_final_equity REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 6. Equity Curves (Time Series)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS equity_curves (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    date TEXT,
                    equity REAL,
                    drawdown_pct REAL,
                    is_benchmark INTEGER DEFAULT 0,
                    strategy_name TEXT,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 7. Regime Data (VIX / Safety)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS regime_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    date TEXT,
                    vix_close REAL,
                    safety_active INTEGER,
                    trigger_reason TEXT,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 8. Strategy Exposure (Heatmap Data)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS exposure_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    date TEXT,
                    strategy_name TEXT,
                    exposure_value REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 9. Safety Switch Impact Analysis
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS safety_switch_impact (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    final_equity REAL,
                    theoretical_equity REAL,
                    saved_loss REAL,
                    opportunity_cost REAL,
                    net_efficiency REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 10. Safety Switch Events Log
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS safety_switch_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    start_date TEXT,
                    end_date TEXT,
                    trigger_reason TEXT,
                    duration_days INTEGER,
                    saved_profit REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 11. Portfolio Funnel (Allocation)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS portfolio_funnel (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    strategy_name TEXT,
                    kelly REAL,
                    raw_share REAL,
                    status TEXT,
                    reason TEXT,
                    final_allocation REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 12. Trade Quality Scores
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trade_quality_scores (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id INTEGER,
                    trade_id TEXT,
                    total_score REAL,
                    grade TEXT,
                    weakest_link TEXT,
                    profit REAL,
                    FOREIGN KEY(run_id) REFERENCES backtest_runs(id)
                )
            """)

            # 13. View for Rankings (Performance Optimization)
            cursor.execute("DROP VIEW IF EXISTS view_trades_ranking")
            cursor.execute("""
                CREATE VIEW view_trades_ranking AS
                SELECT 
                    id, symbol, strategy, entry_date, exit_date, 
                    entry_price, exit_price, realized_pnl, exit_reason
                FROM trades
                WHERE status = 'CLOSED'
            """)

            # --- Schema Migration (Auto-Healing) ---
            # If we failed to delete the DB (locked), we might be working with an old schema.
            # We explicitly check for new columns and add them if missing.

            # 1. backtest_runs: comprehensive check
            cursor.execute("PRAGMA table_info(backtest_runs)")
            existing_columns = {info[1] for info in cursor.fetchall()}

            new_columns = {
                "risk_of_ruin": "REAL",
                "average_win": "REAL",
                "average_loss": "REAL",
                "average_mae": "REAL",
                "average_mfe": "REAL",
                "kelly_mean": "REAL",
                "kelly_std": "REAL",
                "market_exposure_pct": "REAL",
                "risk_adjusted_benchmark": "REAL",
                "exposure_efficiency": "REAL",
                "return_over_max_drawdown": "REAL",
                "diversification_score": "REAL",
            }

            for col, dtype in new_columns.items():
                if col not in existing_columns:
                    cursor.execute(
                        f"ALTER TABLE backtest_runs ADD COLUMN {col} {dtype}"
                    )

            # 2. strategy_metrics: risk_of_ruin and others if needed
            cursor.execute("PRAGMA table_info(strategy_metrics)")
            existing_strat_columns = {info[1] for info in cursor.fetchall()}

            new_strat_columns = {
                "risk_of_ruin": "REAL",
                "average_win": "REAL",
                "average_loss": "REAL",
                "market_exposure_pct": "REAL",
            }

            for col, dtype in new_strat_columns.items():
                if col not in existing_strat_columns:
                    cursor.execute(
                        f"ALTER TABLE strategy_metrics ADD COLUMN {col} {dtype}"
                    )

            connection.commit()

    def save_run(
        self,
        start_date: str,
        end_date: str,
        metrics: BacktestMetrics,
        strategy_metrics: dict[str, BacktestMetrics] | None = None,
        portfolio_kelly: PortfolioMetrics | None = None,
        walk_forward_df: pd.DataFrame | None = None,
        stress_test_results: dict[str, float] | None = None,
        daily_equity_curves: pd.DataFrame | None = None,
        regime_data: pd.DataFrame | None = None,
        strategy_exposures: pd.DataFrame | None = None,
        safety_impact: SimulationImpact | None = None,
        funnel_data: list[FunnelData] | None = None,
        quality_df: pd.DataFrame | None = None,
        diversification_score: float = 0.0,
    ) -> int:
        """Saves a complete backtest run and its associated metrics.

        Returns:
            int: The unique run_id.
        """
        with self._get_connection() as connection:
            cursor = connection.cursor()

            # 1. Main Run Summary
            cursor.execute(
                """
                INSERT INTO backtest_runs (
                    start_date, end_date, total_trades, win_rate, 
                    profit_factor, net_profit, expectancy, maximum_drawdown,
                    sharpe_ratio, sqn, kelly_safe, strategy_return, benchmark_return,
                    risk_of_ruin, average_win, average_loss, average_mae, average_mfe,
                    kelly_mean, kelly_std, market_exposure_pct, 
                    risk_adjusted_benchmark, exposure_efficiency, 
                    return_over_max_drawdown, diversification_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    start_date,
                    end_date,
                    metrics.total_trades,
                    metrics.win_rate,
                    metrics.profit_factor,
                    metrics.net_profit,
                    metrics.expectancy,
                    metrics.maximum_drawdown,
                    metrics.sharpe_ratio,
                    metrics.system_quality_number,
                    metrics.kelly_safe,
                    metrics.strategy_return,
                    metrics.benchmark_return,
                    metrics.risk_of_ruin,
                    metrics.average_win,
                    metrics.average_loss,
                    metrics.average_maximum_adverse_excursion,
                    metrics.average_maximum_favorable_excursion,
                    metrics.kelly_mean,
                    metrics.kelly_std,
                    metrics.market_exposure_pct,
                    metrics.risk_adjusted_benchmark,
                    metrics.exposure_efficiency,
                    metrics.return_over_maximum_drawdown,
                    diversification_score,
                ),
            )
            run_id = cursor.lastrowid
            if run_id is None:
                raise ValueError("Failed to retrieve last inserted run_id.")

            # 2. Strategy Metrics
            if strategy_metrics:
                cursor.executemany(
                    """
                    INSERT INTO strategy_metrics (
                        run_id, strategy_name, total_trades, win_rate, 
                        profit_factor, net_profit, maximum_drawdown, sqn, 
                        kelly_safe, risk_of_ruin, average_win, average_loss,
                        market_exposure_pct
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            name,
                            met.total_trades,
                            met.win_rate,
                            met.profit_factor,
                            met.net_profit,
                            met.maximum_drawdown,
                            met.system_quality_number,
                            met.kelly_safe,
                            met.risk_of_ruin,
                            met.average_win,
                            met.average_loss,
                            met.market_exposure_pct,
                        )
                        for name, met in strategy_metrics.items()
                    ],
                )

            # 3. Portfolio Kelly
            if portfolio_kelly:
                cursor.execute(
                    """
                    INSERT INTO portfolio_simulations (
                        run_id, combined_mean_kelly, safe_kelly_25, 
                        suggested_multiplier, leveraged_max_drawdown, max_total_exposure,
                        correlation_fail_rate, max_concurrent_trades, 
                        uncapped_multiplier, uncapped_max_total_exposure, 
                        uncapped_leveraged_max_drawdown
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        portfolio_kelly.combined_mean_kelly,
                        portfolio_kelly.safe_kelly_25,
                        portfolio_kelly.suggested_multiplier,
                        portfolio_kelly.leveraged_max_drawdown,
                        portfolio_kelly.max_total_exposure,
                        portfolio_kelly.correlation_fail_rate,
                        portfolio_kelly.max_concurrent_trades,
                        portfolio_kelly.uncapped_multiplier,
                        portfolio_kelly.uncapped_max_total_exposure,
                        portfolio_kelly.uncapped_leveraged_max_drawdown,
                    ),
                )

            # 4. Walk Forward
            if walk_forward_df is not None and not walk_forward_df.empty:
                cursor.executemany(
                    """
                    INSERT INTO walk_forward_windows (
                        run_id, window_label, is_kelly, oos_kelly, oos_pf, 
                        avg_vix, uptrend_pct, degradation, recommendation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            row.get("Window"),
                            row.get("IS_Kelly"),
                            row.get("OOS_Kelly"),
                            row.get("OOS_PF"),
                            row.get("Avg_VIX"),
                            row.get("Uptrend_Pct"),
                            row.get("Degradation"),
                            row.get("Recommendation"),
                        )
                        for row in walk_forward_df.to_dict("records")
                    ],
                )

            # 5. Stress Test
            if stress_test_results:
                cursor.execute(
                    """
                    INSERT INTO stress_tests (
                        run_id, avg_max_drawdown, worst_max_drawdown, 
                        failure_rate, avg_final_equity
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        stress_test_results.get("avg_max_drawdown"),
                        stress_test_results.get("worst_max_drawdown"),
                        stress_test_results.get("failure_rate"),
                        stress_test_results.get("avg_equity"),
                    ),
                )

            # 6. Granular Equity Curves
            if daily_equity_curves is not None and not daily_equity_curves.empty:
                cursor.executemany(
                    """
                    INSERT INTO equity_curves (
                        run_id, date, equity, drawdown_pct, is_benchmark, strategy_name
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            str(row["date"]),
                            row["equity"],
                            row["drawdown_pct"],
                            int(row.get("is_benchmark", 0)),
                            row.get("strategy_name"),
                        )
                        for row in daily_equity_curves.to_dict("records")
                    ],
                )

            # 7. Regime Data
            if regime_data is not None and not regime_data.empty:
                cursor.executemany(
                    """
                    INSERT INTO regime_data (
                        run_id, date, vix_close, safety_active, trigger_reason
                    ) VALUES (?, ?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            str(row["date"]),
                            row["vix_close"],
                            int(row["safety_active"]),
                            row.get("trigger_reason", ""),
                        )
                        for row in regime_data.to_dict("records")
                    ],
                )

            # 8. Exposure Data
            if strategy_exposures is not None and not strategy_exposures.empty:
                cursor.executemany(
                    """
                    INSERT INTO exposure_data (
                        run_id, date, strategy_name, exposure_value
                    ) VALUES (?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            str(row["date"]),
                            row["strategy_name"],
                            row["exposure_value"],
                        )
                        for row in strategy_exposures.to_dict("records")
                    ],
                )

            # 9. Safety Impact & Events
            if safety_impact:
                cursor.execute(
                    """
                    INSERT INTO safety_switch_impact (
                        run_id, final_equity, theoretical_equity, saved_loss, 
                        opportunity_cost, net_efficiency
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    (
                        run_id,
                        safety_impact.get("final_equity"),
                        safety_impact.get("theoretical_equity"),
                        safety_impact.get("saved_loss"),
                        safety_impact.get("opportunity_cost"),
                        safety_impact.get("net_efficiency"),
                    ),
                )

                events = safety_impact.get("events", [])
                if events:
                    cursor.executemany(
                        """
                        INSERT INTO safety_switch_events (
                            run_id, start_date, end_date, trigger_reason, 
                            duration_days, saved_profit
                        ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                        [
                            (
                                run_id,
                                str(ev["start_date"]),
                                str(ev.get("end_date", "")),
                                ev["reason"],
                                ev["days"],
                                ev["saved_profit"],
                            )
                            for ev in events
                        ],
                    )

            # 10. Portfolio Funnel
            if funnel_data:
                cursor.executemany(
                    """
                    INSERT INTO portfolio_funnel (
                        run_id, strategy_name, kelly, raw_share, status, 
                        reason, final_allocation
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            item["name"],
                            item["kelly"],
                            item["raw_share"],
                            item["status"],
                            item["reason"],
                            item["final_allocation"],
                        )
                        for item in funnel_data
                    ],
                )

            # 11. Trade Quality
            if quality_df is not None and not quality_df.empty:
                cursor.executemany(
                    """
                    INSERT INTO trade_quality_scores (
                        run_id, trade_id, total_score, grade, weakest_link, profit
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                    [
                        (
                            run_id,
                            str(row.get("trade_id")),
                            row.get("total_score"),
                            row.get("grade"),
                            row.get("weakest_link"),
                            row.get("profit"),
                        )
                        for row in quality_df.to_dict("records")
                    ],
                )

            connection.commit()
            return run_id

    def get_latest_run_id(self) -> int | None:
        """Retrieves the ID of the most recent backtest run."""
        with self._get_connection() as connection:
            cursor = connection.cursor()
            cursor.execute("SELECT id FROM backtest_runs ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return row[0] if row else None

    def get_run_results(self, run_id: int) -> dict[str, Any]:
        """Retrieves all stored metrics for a specific run."""
        with self._get_connection() as connection:
            cursor = connection.cursor()

            # 1. Summary
            cursor.execute("SELECT * FROM backtest_runs WHERE id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return {}
            summary = dict(row)

            # 2. Strategies
            cursor.execute("SELECT * FROM strategy_metrics WHERE run_id = ?", (run_id,))
            strategies = [dict(row) for row in cursor.fetchall()]

            # 3. Kelly / Portfolio
            cursor.execute(
                "SELECT * FROM portfolio_simulations WHERE run_id = ?", (run_id,)
            )
            row = cursor.fetchone()
            portfolio = dict(row) if row else {}

            # 4. WFA
            cursor.execute(
                "SELECT * FROM walk_forward_windows WHERE run_id = ?", (run_id,)
            )
            wfa = [dict(row) for row in cursor.fetchall()]

            # 5. Stress
            cursor.execute("SELECT * FROM stress_tests WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            stress = dict(row) if row else {}

            # 6. Granular Time Series (for Charts)
            cursor.execute(
                "SELECT date, equity, drawdown_pct, is_benchmark, strategy_name "
                "FROM equity_curves WHERE run_id = ? ORDER BY date ASC",
                (run_id,),
            )
            equity_curves = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT date, vix_close, safety_active, trigger_reason "
                "FROM regime_data WHERE run_id = ? ORDER BY date ASC",
                (run_id,),
            )
            regime_data = [dict(row) for row in cursor.fetchall()]

            cursor.execute(
                "SELECT date, strategy_name, exposure_value "
                "FROM exposure_data WHERE run_id = ? ORDER BY date ASC",
                (run_id,),
            )
            exposure_data = [dict(row) for row in cursor.fetchall()]

            # 7. Safety Impact
            cursor.execute(
                "SELECT * FROM safety_switch_impact WHERE run_id = ?", (run_id,)
            )
            row = cursor.fetchone()
            safety_impact = dict(row) if row else {}

            cursor.execute(
                "SELECT * FROM safety_switch_events WHERE run_id = ?", (run_id,)
            )
            events = [dict(row) for row in cursor.fetchall()]
            if safety_impact:
                safety_impact["events"] = events

            # 8. Funnel
            cursor.execute("SELECT * FROM portfolio_funnel WHERE run_id = ?", (run_id,))
            funnel = [dict(row) for row in cursor.fetchall()]

            # 9. Quality
            cursor.execute(
                "SELECT * FROM trade_quality_scores WHERE run_id = ?", (run_id,)
            )
            quality = [dict(row) for row in cursor.fetchall()]

            return {
                "summary": summary,
                "strategies": strategies,
                "portfolio": portfolio,
                "wfa": wfa,
                "stress": stress,
                "equity_curves": equity_curves,
                "regime_data": regime_data,
                "exposure_data": exposure_data,
                "safety_impact": safety_impact,
                "funnel": funnel,
                "quality": quality,
            }
