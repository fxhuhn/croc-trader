"""Generic backtesting engine."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.backtest.backtest_core import BacktestRepository, BacktestTrade
from app.backtest.domain import BacktestPortfolio, BacktestRunConfig
from app.backtest.protocols import StrategyProtocol
from app.backtest.reporting import BacktestReporter
from app.core.database import OHLCVRepository

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RebalanceEvent:
    signal_date: pd.Timestamp
    trade_date: pd.Timestamp


class BacktestEngine:
    """Runs a schedule of rebalance events and delegates signal generation to a strategy."""

    def __init__(
        self,
        *,
        strategy: StrategyProtocol,
        universe: list[str],
        config: BacktestRunConfig,
        market_repo: OHLCVRepository,
        backtest_repo: BacktestRepository,
    ) -> None:
        self.strategy = strategy
        self.universe = universe
        self.config = config
        self.market_repo = market_repo
        self.backtest_repo = backtest_repo

        self.portfolio = BacktestPortfolio(cash=config.initial_capital)
        self.reporter = BacktestReporter(
            self.backtest_repo, self.market_repo, self.config
        )

    def run(self) -> None:
        logger.info("Starting backtest: %s", self.strategy.name)
        logger.debug("Universe size: %d", len(self.universe))

        self.backtest_repo.init_tables()
        self.backtest_repo.cleanup_strategy(self.config.strategy_name)

        raw = self._load_data()
        if raw.empty:
            logger.warning("No data loaded; aborting.")
            return

        features = self.strategy.prepare_features(raw)
        schedule = self._build_schedule(features)

        logger.info("Rebalance periods: %d", len(schedule))
        for event in schedule:
            self._process_event(features, event)

        if schedule:
            self._force_close_all(features, schedule[-1].trade_date)

        self.reporter.generate()
        logger.info("Backtest finished. Results in %s", self.config.out_dir)

    def _load_data(self) -> pd.DataFrame:
        start_dt = pd.to_datetime(self.config.start_date) - timedelta(
            days=self.strategy.lookback_days
        )
        logger.info("Loading data from %s", start_dt.date())
        return self.market_repo.get_data_after_date(
            self.universe, str(start_dt.date()), inclusive=True
        )

    def _build_schedule(self, features: pd.DataFrame) -> list[RebalanceEvent]:
        sched = self.strategy.get_rebalance_schedule(features)
        if sched.empty:
            logger.warning("Schedule is empty.")
            return []

        events: list[RebalanceEvent] = []
        for _, row in sched.iterrows():
            events.append(
                RebalanceEvent(
                    signal_date=pd.to_datetime(row["signal_date"]),
                    trade_date=pd.to_datetime(row["trade_date"]),
                )
            )
        return events

    def _process_event(self, features: pd.DataFrame, event: RebalanceEvent) -> None:
        targets = self.strategy.generate_signals(features, event.signal_date)
        target_set = set(targets)

        logger.debug(
            "Event: signal_date=%s trade_date=%s targets=%d cash=%.2f positions=%d",
            event.signal_date.date(),
            event.trade_date.date(),
            len(targets),
            self.portfolio.cash,
            len(self.portfolio.positions),
        )

        try:
            trade_slice = features.loc[event.trade_date]
        except KeyError:
            logger.debug(
                "No prices for trade_date=%s (skipping)", event.trade_date.date()
            )
            return

        for sym in list(self.portfolio.positions.keys()):
            if sym not in target_set:
                self._close_position(sym, trade_slice, event.trade_date)

        slots = self.portfolio.free_slots(self.config.max_positions)
        if slots > 0 and targets and self.portfolio.cash > 1000:
            to_buy = [s for s in targets if not self.portfolio.has_position(s)][:slots]
            if to_buy:
                allocation = self.portfolio.cash / len(to_buy)
                for sym in to_buy:
                    self._open_position(sym, allocation, trade_slice, event.trade_date)

        self._mark_to_market(trade_slice, event.trade_date)

    def _open_position(
        self,
        symbol: str,
        allocation: float,
        prices: pd.DataFrame,
        date: datetime | pd.Timestamp,
    ) -> None:
        if symbol not in prices.index:
            return

        price = float(prices.loc[symbol, "close"])
        if pd.isna(price) or price <= 0:
            return

        opened = self.portfolio.open_position(symbol, allocation, price, date)
        if opened:
            logger.debug("OPEN %s price=%.2f alloc=%.2f", symbol, price, allocation)

    def _close_position(
        self,
        symbol: str,
        prices: pd.DataFrame,
        date: datetime | pd.Timestamp,
    ) -> None:
        if symbol not in prices.index:
            return

        exit_price = float(prices.loc[symbol, "close"])
        pos = self.portfolio.close_position(symbol, exit_price)
        if pos is None:
            return

        hold_days = (pd.to_datetime(date) - pd.to_datetime(pos.entry_date)).days
        trade = BacktestTrade(
            symbol=symbol,
            entry_date=str(pd.to_datetime(pos.entry_date).date()),
            exit_date=str(pd.to_datetime(date).date()),
            entry_price=pos.entry_price,
            exit_price=exit_price,
            shares=pos.shares,
            pnl=pos.pnl(exit_price),
            return_pct=pos.return_pct(exit_price),
            hold_days=int(hold_days),
        )
        self.backtest_repo.log_trade(trade, self.config.strategy_name)
        logger.debug("CLOSE %s exit=%.2f pnl=%.2f", symbol, exit_price, trade.pnl)

    def _mark_to_market(
        self, prices: pd.DataFrame, date: datetime | pd.Timestamp
    ) -> None:
        close_prices = prices["close"]
        total = float(self.portfolio.equity(close_prices))
        dd = float(self.portfolio.drawdown_pct(total))
        pos_val = float(total - self.portfolio.cash)

        self.backtest_repo.log_equity(
            str(pd.to_datetime(date).date()),
            total,
            float(self.portfolio.cash),
            pos_val,
            dd,
            self.config.strategy_name,
        )

    def _force_close_all(self, features: pd.DataFrame, date: pd.Timestamp) -> None:
        try:
            prices = features.loc[date]
        except KeyError:
            logger.debug("No prices for final date=%s", date.date())
            return

        for sym in list(self.portfolio.positions.keys()):
            self._close_position(sym, prices, date)
