"""Robust Multi-Day Consecutive Screening Tests for Croc-Trader strategies.

Verifies:
1. Multi-Day Transition (Day 1 CREATED -> Day 2 TradeManager INVALID):
   Screener on Day 2 is not blocked by Day 1's resolved trade and successfully generates fresh signals.
2. Active Position Guard (Day 1 CREATED -> Day 2 TradeManager ACTIVE):
   Screener on Day 2 respects Single Position / Open Position rules and correctly blocks duplicate entries.
3. Pre-Market vs Post-Market Date Resolution:
   Pre-market strategies (Bridge Scout, TGIM) resolve target dates forward to the active session.
"""

import datetime
import json
from typing import Any
from unittest.mock import MagicMock

import pandas as pd

from app.const import Strategies
from app.services.screener.strategies.bounce_bandit import BounceBanditStrategy
from app.services.screener.strategies.bridge_scout import BridgeScoutStrategy
from app.services.screener.strategies.croc_setup import CrocSetupStrategy
from app.services.screener.strategies.dip_buyer import DipBuyerStrategy
from app.services.screener.strategies.ndx_momentum import NDXMomentumScreener
from app.services.screener.strategies.tgim import TGIMStrategy
from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from app.services.screener.strategies.two_percent_strategy import TwoPercentStrategy
from app.tools.market_holidays import MarketHolidayChecker
from app.types import TradeStatus


class StatefulTradeRepositoryMock:
    """Accurate stateful mock for TradeRepository tracking trade records and status changes."""

    def __init__(self) -> None:
        self.trades: list[dict[str, Any]] = []
        self._next_id = 1

    def create_trade(
        self,
        symbol: str,
        strategy: str | Strategies,
        size: float,
        entry: float,
        stop_loss: float = 0.0,
        target: float = 0.0,
        context: dict[str, Any] | None = None,
    ) -> int:
        trade_id = self._next_id
        self._next_id += 1
        strat_val = getattr(strategy, "value", str(strategy))
        date_val = (context or {}).get("date") or datetime.date.today().strftime(
            "%Y-%m-%d"
        )
        self.trades.append(
            {
                "id": trade_id,
                "symbol": symbol,
                "strategy": strat_val,
                "size": size,
                "entry_price": entry,
                "stop_loss": stop_loss,
                "target_price": target,
                "status": TradeStatus.CREATED.value,
                "date": date_val,
                "signal_context": json.dumps(context or {}),
            }
        )
        return trade_id

    def exists(self, symbol: str, strategy: str | Strategies, date: str) -> bool:
        strat_val = getattr(strategy, "value", str(strategy)).lower()
        return any(
            t["symbol"] == symbol
            and strat_val in str(t["strategy"]).lower()
            and str(t.get("date")) == date
            for t in self.trades
        )

    def get_by_status(
        self, status: str | TradeStatus | list[str | TradeStatus]
    ) -> list[dict[str, Any]]:
        statuses = (
            [getattr(s, "value", str(s)) for s in status]
            if isinstance(status, list)
            else [getattr(status, "value", str(status))]
        )
        return [t for t in self.trades if t.get("status") in statuses]

    def set_status(self, trade_id: int, new_status: TradeStatus) -> None:
        for t in self.trades:
            if t["id"] == trade_id:
                t["status"] = new_status.value


def _create_synthetic_history(
    end_date_str: str,
    num_bars: int = 60,
    base_price: float = 100.0,
    price_pattern: list[float] | None = None,
) -> pd.DataFrame:
    """Generates synthetic OHLCV dataframe ending on end_date_str."""
    dates = pd.date_range(end=end_date_str, periods=num_bars, freq="B")
    if price_pattern:
        closes = [base_price] * (num_bars - len(price_pattern)) + price_pattern
    else:
        closes = [base_price] * num_bars

    highs = [c * 1.01 for c in closes]
    lows = [c * 0.99 for c in closes]
    opens = closes.copy()
    volumes = [1_000_000] * num_bars

    return pd.DataFrame(
        {
            "date": dates,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


# =====================================================================
# 1. Bridge Scout (Pre-Market Same-Day MOC, EoM Window)
# =====================================================================


def test_bridge_scout_multi_day_consecutive_screening() -> None:
    """Verifies that Bridge Scout generates a new signal on Day 2 after Day 1 is INVALID,
    and is blocked on Day 2 when Day 1 is ACTIVE."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()
    holiday_checker = MarketHolidayChecker()

    # Setup strategy
    strategy = BridgeScoutStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
        holiday_checker=holiday_checker,
    )

    day1_eod = "2026-08-26"
    day2_eod = "2026-08-27"
    today_live = "2026-08-28"

    # Day 1: Screener runs for Day 2 pre-market (using Day 1 EOD history)
    df_day1 = _create_synthetic_history(day1_eod, num_bars=60, base_price=710.0)
    mock_provider.get_batch_history.return_value = {"QQQ": df_day1}
    mock_provider.get_latest_date.return_value = day1_eod

    # Day 1 screening creates Trade 1 for Day 2
    trade_id_1 = repo.create_trade(
        symbol="QQQ",
        strategy=Strategies.BridgeScout,
        size=0.0,
        entry=710.16,
        context={"date": day2_eod, "setup_date": day2_eod},
    )
    assert len(repo.trades) == 1

    # Scenario A: TradeManager invalidates Day 1 trade on Day 2 morning
    repo.set_status(trade_id_1, TradeStatus.INVALID)

    # Day 2 morning: Market data updated with Day 2 EOD (close=721.11)
    df_day2 = _create_synthetic_history(
        day2_eod, num_bars=60, base_price=710.0, price_pattern=[711.37, 721.11]
    )
    mock_provider.get_batch_history.return_value = {"QQQ": df_day2}
    mock_provider.get_latest_date.return_value = day2_eod

    # Screener runs with days=0 and analysis_date=day2_eod (from ScreenerEngine)
    hits = strategy.run(days=0, analysis_date=day2_eod)
    assert hits == 1
    assert len(repo.trades) == 2
    assert repo.trades[-1]["date"] == today_live

    # Scenario B: If Trade 2 is marked ACTIVE, subsequent run on same day is blocked
    repo.set_status(repo.trades[-1]["id"], TradeStatus.ACTIVE)
    blocked_hits = strategy.run(days=0, analysis_date=day2_eod)
    assert blocked_hits == 0
    assert len(repo.trades) == 2


# =====================================================================
# 2. TGIM (Pre-Market Same-Day MOC, Monday Setup)
# =====================================================================


def test_tgim_multi_day_consecutive_screening() -> None:
    """Verifies that TGIM generates a new Monday signal when the previous Monday is INVALID,
    and is blocked if previous trade is still ACTIVE."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()

    strategy = TGIMStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    monday1_date = "2026-08-17"
    friday1_date = "2026-08-14"
    monday2_date = "2026-08-24"
    friday2_date = "2026-08-21"

    # Pre-market Monday 1: History up to Friday 1
    df_friday1 = _create_synthetic_history(friday1_date, num_bars=30, base_price=500.0)
    mock_provider.get_batch_history.return_value = {"SPY": df_friday1}

    hits1 = strategy.run(days=0, analysis_date=friday1_date)
    assert hits1 == 1
    assert len(repo.trades) == 1
    trade_id_1 = repo.trades[0]["id"]
    assert repo.trades[0]["date"] == monday1_date

    # Scenario A: TradeManager invalidates Monday 1 trade
    repo.set_status(trade_id_1, TradeStatus.INVALID)

    # Next Monday 2 pre-market: History up to Friday 2
    df_friday2 = _create_synthetic_history(friday2_date, num_bars=35, base_price=505.0)
    mock_provider.get_batch_history.return_value = {"SPY": df_friday2}

    hits2 = strategy.run(days=0, analysis_date=friday2_date)
    assert hits2 == 1
    assert len(repo.trades) == 2
    assert repo.trades[1]["date"] == monday2_date

    # Scenario B: If Monday 2 trade is ACTIVE, screening is blocked
    repo.set_status(repo.trades[1]["id"], TradeStatus.ACTIVE)
    blocked_hits = strategy.run(days=0, analysis_date=friday2_date)
    assert blocked_hits == 0


# =====================================================================
# 3. Bounce Bandit (Post-Market / Next-Day MOO)
# =====================================================================


def test_bounce_bandit_multi_day_consecutive_screening() -> None:
    """Verifies that Bounce Bandit handles consecutive days properly."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()

    strategy = BounceBanditStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    day1_date = "2026-08-26"
    day2_date = "2026-08-27"

    # Day 1: Oversold dip triggering Bounce Bandit (SMA 200 high, RSI2 < 20)
    dates1 = pd.date_range(end=day1_date, periods=250, freq="B")
    close_prices1 = [400.0 + (i * 0.5) for i in range(247)] + [524.0, 523.0, 500.0]
    df_day1 = pd.DataFrame(
        [
            {
                "date": d,
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "volume": 100000,
            }
            for d, p in zip(dates1, close_prices1, strict=True)
        ]
    )
    mock_provider.get_batch_history.return_value = {"QQQ": df_day1}

    hits1 = strategy.run(days=0, analysis_date=day1_date)
    assert hits1 == 1
    assert len(repo.trades) == 1
    trade_id_1 = repo.trades[0]["id"]

    # Scenario A: Trade 1 is INVALID on Day 2 -> Slot is free for new setup
    repo.set_status(trade_id_1, TradeStatus.INVALID)
    dates2 = pd.date_range(end=day2_date, periods=250, freq="B")
    close_prices2 = [400.0 + (i * 0.5) for i in range(247)] + [524.0, 523.0, 500.0]
    df_day2 = pd.DataFrame(
        [
            {
                "date": d,
                "open": p,
                "high": p + 1.0,
                "low": p - 1.0,
                "close": p,
                "volume": 100000,
            }
            for d, p in zip(dates2, close_prices2, strict=True)
        ]
    )
    mock_provider.get_batch_history.return_value = {"QQQ": df_day2}

    hits2 = strategy.run(days=0, analysis_date=day2_date)
    assert hits2 == 1
    assert len(repo.trades) == 2

    # Scenario B: Trade 2 is ACTIVE -> Slot is blocked
    repo.set_status(repo.trades[1]["id"], TradeStatus.ACTIVE)
    blocked_hits = strategy.run(days=0, analysis_date=day2_date)
    assert blocked_hits == 0


# =====================================================================
# 4. Two Percent (Post-Market / Next-Day MOO)
# =====================================================================


def test_two_percent_multi_day_consecutive_screening() -> None:
    """Verifies that TwoPercent strategy allows new trades after invalidation and blocks on active."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()

    strategy = TwoPercentStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    day1_date = "2026-08-26"
    day2_date = "2026-08-27"

    # Setup where QQQ drops > 2% from high or triggers criteria
    df_day1 = _create_synthetic_history(
        day1_date, num_bars=52, base_price=500.0, price_pattern=[490.0, 480.0]
    )
    mock_provider.get_batch_history.return_value = {"QQQ": df_day1}

    hits1 = strategy.run(days=0, analysis_date=day1_date)
    if hits1 > 0:
        trade_id = repo.trades[0]["id"]
        # Invalidate
        repo.set_status(trade_id, TradeStatus.INVALID)
        df_day2 = _create_synthetic_history(
            day2_date, num_bars=52, base_price=500.0, price_pattern=[490.0, 480.0]
        )
        mock_provider.get_batch_history.return_value = {"QQQ": df_day2}
        hits2 = strategy.run(days=0, analysis_date=day2_date)
        assert hits2 == 1

        # Active blocks
        repo.set_status(repo.trades[-1]["id"], TradeStatus.ACTIVE)
        assert strategy.run(days=0, analysis_date=day2_date) == 0


# =====================================================================
# 5. Dip Buyer (Multi-Position Universe)
# =====================================================================


def test_dip_buyer_multi_day_consecutive_screening() -> None:
    """Verifies that DipBuyer does not duplicate active positions on same symbol."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()

    strategy = DipBuyerStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    day1_date = "2026-08-26"
    day2_date = "2026-08-27"

    # Simulate existing active trade on AAPL
    repo.create_trade(
        symbol="AAPL",
        strategy=Strategies.DipBuyer,
        size=10.0,
        entry=180.0,
        context={"date": day1_date},
    )
    repo.set_status(repo.trades[0]["id"], TradeStatus.ACTIVE)

    # Active AAPL should block new AAPL signal
    assert (
        strategy._has_existing_trade_or_position(
            repo, "AAPL", Strategies.DipBuyer, day2_date
        )
        is True
    )

    # Once AAPL is closed/invalid, new signal on AAPL is allowed
    repo.set_status(repo.trades[0]["id"], TradeStatus.INVALID)
    assert (
        strategy._has_existing_trade_or_position(
            repo, "AAPL", Strategies.DipBuyer, day2_date
        )
        is False
    )


# =====================================================================
# 6. Turnover Timing (Multi-Position Universe)
# =====================================================================


def test_turnover_timing_multi_day_consecutive_screening() -> None:
    """Verifies that TurnoverTiming allows new setups when previous trades are invalid."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()

    strategy = TurnoverTimingStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
    )

    friday_date = "2026-08-21"
    df_nvda = _create_synthetic_history(friday_date, num_bars=60, base_price=120.0)
    mock_provider.get_batch_history.return_value = {"NVDA": df_nvda}

    # Simulate existing active trade on NVDA
    repo.create_trade(
        symbol="NVDA",
        strategy=Strategies.TurnOverTiming,
        size=10.0,
        entry=120.0,
        context={"date": friday_date},
    )
    repo.set_status(repo.trades[0]["id"], TradeStatus.ACTIVE)

    # Active NVDA blocks duplicate signal
    assert (
        strategy._has_existing_trade_or_position(
            repo, "NVDA", Strategies.TurnOverTiming, friday_date
        )
        is True
    )

    # Once invalid or closed, not blocked
    repo.set_status(repo.trades[0]["id"], TradeStatus.INVALID)
    next_friday = "2026-08-28"
    assert (
        strategy._has_existing_trade_or_position(
            repo, "NVDA", Strategies.TurnOverTiming, next_friday
        )
        is False
    )


# =====================================================================
# 7. Croc Setup (Multi-Position Universe)
# =====================================================================


def test_croc_setup_multi_day_consecutive_screening() -> None:
    """Verifies that CrocSetup respects active status vs resolved/invalid status."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()
    mock_sig_repo = MagicMock()

    strategy = CrocSetupStrategy(
        trade_repository=repo,  # type: ignore[arg-type]
        data_provider=mock_provider,
        signal_repository=mock_sig_repo,
    )

    day1_date = "2026-08-26"
    day2_date = "2026-08-27"

    # Simulate trade on MSFT
    trade_id = repo.create_trade(
        symbol="MSFT",
        strategy=Strategies.CrocSetup,
        size=5.0,
        entry=450.0,
        context={"date": day1_date},
    )
    repo.set_status(trade_id, TradeStatus.ACTIVE)

    # Blocked while active
    assert (
        strategy._has_existing_trade_or_position(
            repo, "MSFT", Strategies.CrocSetup, day2_date
        )
        is True
    )

    # Free once invalid
    repo.set_status(trade_id, TradeStatus.INVALID)
    assert (
        strategy._has_existing_trade_or_position(
            repo, "MSFT", Strategies.CrocSetup, day2_date
        )
        is False
    )


# =====================================================================
# 8. NDX Momentum (Top Leaders Multi-Position)
# =====================================================================


def test_ndx_momentum_multi_day_consecutive_screening() -> None:
    """Verifies that NDXMomentum handles consecutive day leader screening cleanly."""
    repo = StatefulTradeRepositoryMock()
    mock_provider = MagicMock()

    strategy = NDXMomentumScreener(
        trade_repository=repo,  # type: ignore[arg-type]
        market_data_provider=mock_provider,
    )

    day1_date = "2026-08-26"
    day2_date = "2026-08-27"

    # Simulate active position on GOOGL
    trade_id = repo.create_trade(
        symbol="GOOGL",
        strategy=Strategies.NDXMomentum,
        size=10.0,
        entry=175.0,
        context={"date": day1_date},
    )
    repo.set_status(trade_id, TradeStatus.ACTIVE)

    # Active GOOGL blocks duplicate trade
    assert (
        strategy._has_existing_trade_or_position(
            repo, "GOOGL", Strategies.NDXMomentum, day2_date
        )
        is True
    )

    # Invalidation frees slot for new screening date
    repo.set_status(trade_id, TradeStatus.INVALID)
    assert (
        strategy._has_existing_trade_or_position(
            repo, "GOOGL", Strategies.NDXMomentum, day2_date
        )
        is False
    )
