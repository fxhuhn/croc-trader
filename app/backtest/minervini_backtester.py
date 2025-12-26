import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


# --- Import from your existing files ---
from app.config import settings
from app.core.database import OHLCVRepository
from app.database.sqlite_repo import SQLiteRepository


# --- Configuration for Strategy Parameters ---
@dataclass
class StrategyConfig:
    strategy_name: str = "Minervini Trend Template (Weekly)"  # <--- NEW
    start_date: str = "2022-01-01"
    initial_capital: float = 100_000.0
    max_positions: int = 10

    # Minervini Parameters
    rs_window: int = 252
    sma_rise_window: int = 20
    min_rs: float = 70.0

    out_dir: str = "backtest_results"


# --- Persistence Layer ---
@dataclass
class BacktestTrade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    return_pct: float
    hold_days: int


class BacktestRepository(SQLiteRepository):
    def init_tables(self, clear_existing: bool = True):
        with self._get_connection() as conn:
            if clear_existing:
                conn.execute("DROP TABLE IF EXISTS backtest_trades")
                conn.execute("DROP TABLE IF EXISTS backtest_equity")

            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    entry_date TEXT NOT NULL,
                    exit_date TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL NOT NULL,
                    shares INTEGER NOT NULL,
                    pnl REAL NOT NULL,
                    return_pct REAL NOT NULL,
                    hold_days INTEGER NOT NULL,
                    strategy_name TEXT DEFAULT 'minervini_weekly'
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS backtest_equity (
                    date TEXT PRIMARY KEY,
                    total_equity REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    drawdown_pct REAL NOT NULL
                )
            """)
            conn.commit()

    def log_trade(self, t: BacktestTrade):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO backtest_trades
                (symbol, entry_date, exit_date, entry_price, exit_price,
                 shares, pnl, return_pct, hold_days)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    t.symbol,
                    t.entry_date,
                    t.exit_date,
                    t.entry_price,
                    t.exit_price,
                    t.shares,
                    t.pnl,
                    t.return_pct,
                    t.hold_days,
                ),
            )
            conn.commit()

    def log_equity(
        self, date_str: str, equity: float, cash: float, pos_value: float, dd: float
    ):
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO backtest_equity
                (date, total_equity, cash, positions_value, drawdown_pct)
                VALUES (?, ?, ?, ?, ?)
            """,
                (date_str, equity, cash, pos_value, dd),
            )
            conn.commit()

    def get_equity_curve(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM backtest_equity ORDER BY date", conn)
        df["date"] = pd.to_datetime(df["date"])
        return df.set_index("date")

    def get_trades(self) -> pd.DataFrame:
        with self._get_connection() as conn:
            df = pd.read_sql_query("SELECT * FROM backtest_trades", conn)
        return df


# --- Main Backtester Engine ---
class MinerviniBacktester:
    def __init__(self, config: StrategyConfig, tickers: list[str]):
        self.cfg = config
        self.tickers = tickers

        self.market_db_path = str(settings.database.market_data_path)
        self.ohlcv_repo = OHLCVRepository(self.market_db_path)

        self.backtest_db_path = str(settings.database.backtest_path)
        self.bt_repo = BacktestRepository(self.backtest_db_path)
        self.bt_repo.init_tables(clear_existing=True)

        self.cash = config.initial_capital
        self.positions: dict[str, dict] = {}
        self.peak_equity = config.initial_capital

    def _load_data(self) -> pd.DataFrame:
        lookback_start = pd.to_datetime(self.cfg.start_date) - timedelta(days=400)
        print(
            f"Loading data from {self.market_db_path} starting {lookback_start.date()}..."
        )
        return self.ohlcv_repo.get_data_after_date(
            self.tickers, str(lookback_start.date()), inclusive=True
        )

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Computing Minervini indicators...")
        df = df.sort_index()
        working = df.reset_index()

        def compute_group_features(g):
            sma50 = g["close"].rolling(50).mean()
            sma150 = g["close"].rolling(150).mean()
            sma200 = g["close"].rolling(200).mean()
            high52 = g["close"].rolling(252).max()
            low52 = g["close"].rolling(252).min()
            sma200_trending = sma200 > sma200.shift(20)
            rs_raw = g["close"].pct_change(252)

            return pd.DataFrame(
                {
                    "sma50": sma50,
                    "sma150": sma150,
                    "sma200": sma200,
                    "high52": high52,
                    "low52": low52,
                    "sma200_trending": sma200_trending,
                    "rs_raw": rs_raw,
                },
                index=g.index,
            )

        features = working.groupby("symbol", group_keys=False).apply(
            compute_group_features, include_groups=False
        )
        working = pd.concat([working, features], axis=1)

        working["rs_rank"] = working.groupby("date")["rs_raw"].rank(pct=True) * 100

        c1 = (
            (working["close"] > working["sma50"])
            & (working["close"] > working["sma150"])
            & (working["close"] > working["sma200"])
        )
        c2 = (working["sma50"] > working["sma150"]) & (
            working["sma150"] > working["sma200"]
        )
        c3 = working["sma200_trending"]
        c4 = working["close"] >= (1.3 * working["low52"])
        c5 = working["close"] >= (0.75 * working["high52"])
        c6 = working["rs_rank"] >= self.cfg.min_rs

        working["is_valid"] = c1 & c2 & c3 & c4 & c5 & c6
        working["dist_to_high"] = working["close"] / working["high52"]

        return working.set_index(["date", "symbol"]).sort_index()

    def get_weekly_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        dates = df.index.levels[0].unique().sort_values()
        dates = dates[dates >= pd.to_datetime(self.cfg.start_date)]
        cal = pd.DataFrame({"date": dates})
        cal["week"] = cal["date"].dt.to_period("W-SUN")
        grouped = (
            cal.groupby("week")["date"]
            .agg(["min", "max"])
            .rename(columns={"min": "monday", "max": "friday_dummy"})
        )

        schedule = []
        weeks = grouped.index.sort_values()
        for i in range(1, len(weeks)):
            schedule.append(
                {
                    "signal_date": grouped.loc[weeks[i - 1], "friday_dummy"],
                    "trade_date": grouped.loc[weeks[i], "monday"],
                }
            )
        return pd.DataFrame(schedule)

    def run(self):
        raw_df = self._load_data()
        if raw_df.empty:
            print("❌ No data loaded.")
            return

        data = self.prepare_features(raw_df)
        schedule = self.get_weekly_schedule(data)

        print(f"Starting backtest on {self.backtest_db_path}...")
        print(f"Total weeks to simulate: {len(schedule)}")

        for _, row in schedule.iterrows():
            sig_date = row["signal_date"]
            trade_date = row["trade_date"]

            try:
                daily_slice = data.loc[sig_date]
            except KeyError:
                continue

            candidates = daily_slice[daily_slice["is_valid"]].copy()
            candidates = candidates.sort_values(
                by=["rs_rank", "dist_to_high"], ascending=[False, False]
            )
            top_10 = candidates.head(self.cfg.max_positions).index.tolist()
            top_10_set = set(top_10)

            try:
                trade_slice = data.loc[trade_date]
            except KeyError:
                continue

            for sym in list(self.positions.keys()):
                if sym not in top_10_set:
                    self._close_position(sym, trade_slice, trade_date)

            slots_free = self.cfg.max_positions - len(self.positions)
            if slots_free > 0:
                buy_candidates = [s for s in top_10 if s not in self.positions]
                to_buy = buy_candidates[:slots_free]
                if len(to_buy) > 0 and self.cash > 1000:
                    alloc = self.cash / len(to_buy)
                    for sym in to_buy:
                        self._open_position(sym, alloc, trade_slice, trade_date)

            self._update_equity(trade_slice, trade_date)

        last_date = schedule.iloc[-1]["trade_date"]
        self._force_close_all(data, last_date)

        # --- NEW: Output Results ---
        self.generate_report()
        print(f"✅ Backtest finished. Results in {self.backtest_db_path}")

    def _open_position(self, sym, alloc, prices, date):
        if sym not in prices.index:
            return
        price = prices.loc[sym, "open"]
        if pd.isna(price) or price <= 0:
            return
        shares = int(alloc // price)
        if shares == 0:
            return
        cost = shares * price
        self.cash -= cost
        self.positions[sym] = {
            "shares": shares,
            "entry_price": price,
            "entry_date": date,
            "cost": cost,
        }

    def _close_position(self, sym, prices, date):
        if sym not in prices.index:
            return
        price = prices.loc[sym, "open"]
        pos = self.positions.pop(sym)
        proceeds = pos["shares"] * price
        self.cash += proceeds
        pnl = proceeds - pos["cost"]
        ret = (pnl / pos["cost"]) * 100
        hold_days = (date - pos["entry_date"]).days
        t = BacktestTrade(
            symbol=sym,
            entry_date=str(pos["entry_date"].date()),
            exit_date=str(date.date()),
            entry_price=pos["entry_price"],
            exit_price=price,
            shares=pos["shares"],
            pnl=pnl,
            return_pct=ret,
            hold_days=hold_days,
        )
        self.bt_repo.log_trade(t)

    def _update_equity(self, prices, date):
        pos_val = sum(
            pos["shares"] * prices.loc[sym, "close"]
            if sym in prices.index
            else pos["cost"]
            for sym, pos in self.positions.items()
        )
        total = self.cash + pos_val
        self.peak_equity = max(self.peak_equity, total)
        dd = (
            ((self.peak_equity - total) / self.peak_equity) * 100
            if self.peak_equity > 0
            else 0
        )
        self.bt_repo.log_equity(str(date.date()), total, self.cash, pos_val, dd)

    def _force_close_all(self, data, last_date):
        try:
            prices = data.loc[last_date]
            for sym in list(self.positions.keys()):
                self._close_position(sym, prices, last_date)
        except KeyError:
            pass

    def generate_report(self):
        """Calculate and save KPI metrics."""
        eq = self.bt_repo.get_equity_curve()
        trades = self.bt_repo.get_trades()

        if eq.empty:
            print("⚠️ No equity data to analyze.")
            return

        # --- 1. Equity Metrics ---
        start_val = eq.iloc[0]["total_equity"]
        end_val = eq.iloc[-1]["total_equity"]

        # Total Return
        total_return = (end_val / start_val) - 1

        # CAGR
        days = (eq.index[-1] - eq.index[0]).days
        years = (
            days / 365.25
        )  # <--- Defined here, available for both CAGR and Metrics dict

        cagr = ((end_val / start_val) ** (1 / years)) - 1 if years > 0 else 0

        # Max Drawdown
        max_dd = eq["drawdown_pct"].max()

        # Sharpe Ratio
        weekly_eq = eq["total_equity"].resample("W").last()
        weekly_ret = weekly_eq.pct_change().dropna()
        avg_weekly_ret = weekly_ret.mean()
        std_weekly_ret = weekly_ret.std()

        risk_free_rate = 0.02
        sharpe = 0.0
        if std_weekly_ret > 0:
            sharpe = (avg_weekly_ret * 52 - risk_free_rate) / (
                std_weekly_ret * np.sqrt(52)
            )

        # --- 2. Trade Metrics ---
        n_trades = len(trades)
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0

        if n_trades > 0:
            wins = trades[trades["pnl"] > 0]
            losses = trades[trades["pnl"] <= 0]

            win_rate = (len(wins) / n_trades) * 100

            avg_win = wins["return_pct"].mean() if not wins.empty else 0.0
            avg_loss = losses["return_pct"].mean() if not losses.empty else 0.0

            gross_win = wins["pnl"].sum()
            gross_loss = abs(losses["pnl"].sum())
            if gross_loss > 0:
                profit_factor = gross_win / gross_loss
            else:
                profit_factor = float("inf") if gross_win > 0 else 0.0

        # --- 3. Output ---
        metrics = {
            "strategy": self.cfg.strategy_name,
            "period": {
                "start": str(eq.index[0].date()),
                "end": str(eq.index[-1].date()),
                "duration_years": round(years, 2),  # <--- Now 'years' is valid
            },
            "performance": {
                "total_return_pct": round(total_return * 100, 2),
                "cagr_pct": round(cagr * 100, 2),
                "max_drawdown_pct": round(max_dd, 2),
                "sharpe_ratio": round(sharpe, 2),
                "end_equity": round(end_val, 2),
            },
            "trades": {
                "count": int(n_trades),
                "win_rate_pct": round(win_rate, 2),
                "profit_factor": round(profit_factor, 2),
                "avg_win_pct": round(avg_win, 2),
                "avg_loss_pct": round(avg_loss, 2),
            },
        }

        # Console Output
        print("\n" + "=" * 40)
        print(f"📊 REPORT: {metrics['strategy'].upper()}")
        print("=" * 40)
        print(f"Start Date:    {metrics['period']['start']}")
        print(f"End Date:      {metrics['period']['end']}")
        print(f"Duration:      {metrics['period']['duration_years']} years")
        print("-" * 40)
        print(f"Final Equity:  ${metrics['performance']['end_equity']:,.2f}")
        print(f"Total Return:  {metrics['performance']['total_return_pct']}%")
        print(f"CAGR:          {metrics['performance']['cagr_pct']}%")
        print(f"Max Drawdown:  {metrics['performance']['max_drawdown_pct']}%")
        print(f"Sharpe Ratio:  {metrics['performance']['sharpe_ratio']}")
        print("-" * 40)
        print(f"Total Trades:  {metrics['trades']['count']}")
        print(f"Win Rate:      {metrics['trades']['win_rate_pct']}%")
        print(f"Profit Factor: {metrics['trades']['profit_factor']}")
        print(f"Avg Win:       {metrics['trades']['avg_win_pct']}%")
        print(f"Avg Loss:      {metrics['trades']['avg_loss_pct']}%")
        print("=" * 40 + "\n")

        # Helper to clean numpy types for YAML
        def _to_python_type(obj):
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: _to_python_type(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_to_python_type(i) for i in obj]
            return obj

        metrics_clean = _to_python_type(metrics)

        # YAML Output
        out_path = Path(self.cfg.out_dir)
        out_path.mkdir(exist_ok=True)
        file_path = out_path / "backtest_metrics.yaml"

        with open(file_path, "w") as f:
            yaml.safe_dump(metrics_clean, f, sort_keys=False, default_flow_style=False)
        print(f"📄 Metrics saved to: {file_path}")
