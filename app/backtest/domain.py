"""Backtest domain dataclasses (positions, portfolio, engine config)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import pandas as pd

type Symbol = str
type Price = float


@dataclass(slots=True)
class BacktestPosition:
    symbol: Symbol
    shares: int
    entry_price: Price
    entry_date: datetime | pd.Timestamp
    cost: float

    @classmethod
    def open(
        cls,
        symbol: Symbol,
        allocation: float,
        price: Price,
        date: datetime | pd.Timestamp,
    ) -> BacktestPosition | None:
        if pd.isna(price) or price <= 0 or allocation < price:
            return None

        shares = int(allocation // price)
        if shares <= 0:
            return None

        cost = shares * price
        return cls(
            symbol=symbol,
            shares=shares,
            entry_price=price,
            entry_date=date,
            cost=cost,
        )

    def market_value(self, price: Price) -> float:
        return self.shares * price

    def pnl(self, exit_price: Price) -> float:
        return self.shares * exit_price - self.cost

    def return_pct(self, exit_price: Price) -> float:
        return (self.pnl(exit_price) / self.cost) * 100 if self.cost > 0 else 0.0


@dataclass(slots=True)
class BacktestPortfolio:
    cash: float
    positions: dict[Symbol, BacktestPosition] = field(default_factory=dict)
    peak_equity: float = field(init=False)

    def __post_init__(self) -> None:
        self.peak_equity = self.cash

    def has_position(self, symbol: Symbol) -> bool:
        return symbol in self.positions

    def free_slots(self, max_positions: int) -> int:
        return max(0, max_positions - len(self.positions))

    def equity(self, close_prices: pd.Series) -> float:
        positions_value = sum(
            pos.market_value(float(close_prices.get(sym, pos.entry_price)))
            for sym, pos in self.positions.items()
        )
        return self.cash + positions_value

    def drawdown_pct(self, total_equity: float) -> float:
        self.peak_equity = max(self.peak_equity, total_equity)
        if self.peak_equity <= 0:
            return 0.0
        return ((self.peak_equity - total_equity) / self.peak_equity) * 100

    def open_position(
        self,
        symbol: Symbol,
        allocation: float,
        price: Price,
        date: datetime | pd.Timestamp,
    ) -> bool:
        if self.has_position(symbol) or allocation > self.cash:
            return False

        pos = BacktestPosition.open(symbol, allocation, price, date)
        if pos is None:
            return False

        self.cash -= pos.cost
        self.positions[symbol] = pos
        return True

    def close_position(
        self, symbol: Symbol, exit_price: Price
    ) -> BacktestPosition | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        self.cash += pos.shares * exit_price
        return pos


@dataclass(frozen=True, slots=True)
class BacktestRunConfig:
    """
    Engine config.

    BacktestReporter expects:
    - strategy_name
    - out_dir
    - initial_capital
    [file:3]
    """

    strategy_name: str
    start_date: str
    initial_capital: float
    max_positions: int
    out_dir: Path
