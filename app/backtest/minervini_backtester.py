import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import pandas as pd

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


from app.backtest.backtest_core import BacktestRepository, BacktestTrade
from app.backtest.reporting import BacktestReporter

# --- Import from your existing files ---
from app.config import settings
from app.core.database import OHLCVRepository


# --- Configuration ---
@dataclass
class StrategyConfig:
    strategy_name: str = "Minervini Trend (Weekly)"
    start_date: str = "2022-01-01"
    initial_capital: float = 100_000.0
    max_positions: int = 10
    rs_window: int = 252
    sma_rise_window: int = 20
    min_rs: float = 70.0
    out_dir: str = "backtest_results"  # Unified


# --- Main Engine ---
class MinerviniBacktester:
    def __init__(self, config: StrategyConfig, tickers: list[str]):
        self.cfg = config
        self.tickers = tickers

        self.market_repo = OHLCVRepository(str(settings.database.market_data_path))
        self.bt_repo = BacktestRepository(str(settings.database.backtest_path))
        self.bt_repo.init_tables(clear_existing=True)

        self.cash = config.initial_capital
        self.positions: dict[str, dict] = {}
        self.peak_equity = config.initial_capital

        # Initialize Reusable Reporter
        self.reporter = BacktestReporter(self.bt_repo, self.market_repo, self.cfg)

    def _load_data(self) -> pd.DataFrame:
        lookback_start = pd.to_datetime(self.cfg.start_date) - timedelta(days=400)
        print(f"Loading data from {lookback_start.date()}...")
        return self.market_repo.get_data_after_date(
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

        # Ensure correct index naming
        if working.index.names != ["date", "symbol"]:
            try:
                working = working.set_index(["date", "symbol"])
            except KeyError:
                pass  # Already set or columns missing

        return working.sort_index()

    def get_weekly_schedule(self, df: pd.DataFrame) -> pd.DataFrame:
        # Robust date handling
        try:
            date_level = df.index.get_level_values("date").unique()
        except KeyError:
            # Fallback if index isn't named "date" but is level 0
            date_level = df.index.levels[0].unique()

        dates = pd.to_datetime(date_level).sort_values()
        start_dt = pd.to_datetime(self.cfg.start_date)
        dates = dates[dates >= start_dt]

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

        print(f"Starting backtest on {self.bt_repo.db_path}...")
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

        # --- REPORTING ---
        self.reporter.generate()
        print(f"✅ Backtest finished. Results in {self.cfg.out_dir}")

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

        # FIX: Pass self.cfg.strategy_name as the second argument
        self.bt_repo.log_trade(t, self.cfg.strategy_name)

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
