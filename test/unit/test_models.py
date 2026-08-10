"""Unit tests for app/models.py to achieve 100% test coverage."""

from datetime import datetime
from decimal import Decimal

import pytest

from app.models import (
    BacktestMetrics,
    CrocContext,
    CrocSignal,
    MarketPrice,
    Order,
    OrderLeg,
    PortfolioMetrics,
    SignalStat,
    SQNClassification,
    TradeParams,
)


def test_trade_params() -> None:
    params = TradeParams(stop_loss=10.0, take_profit_1=15.0, extras={"key": "val"})
    assert params.stop_loss == 10.0
    assert params.take_profit_1 == 15.0
    assert params.take_profit_2 is None
    assert params.extras == {"key": "val"}


def test_order_and_order_leg() -> None:
    leg = OrderLeg(action="BUY", type="LMT", price=Decimal("100.50"), quantity=10)
    order = Order(id="order_1", symbol="AAPL", quantity=10, mode="LIVE", entry=leg)

    assert leg.action == "BUY"
    assert leg.type == "LMT"
    assert leg.price == Decimal("100.50")
    assert order.id == "order_1"
    assert order.entry == leg
    assert order.last_status == "PendingSubmit"


def test_croc_context_and_sqn() -> None:
    ctx = CrocContext(high=150.0, low=140.0)
    assert ctx.high == 150.0
    assert ctx.low == 140.0

    sqn = SQNClassification(label="Good", color="green")
    assert sqn.label == "Good"
    assert sqn.color == "green"


def test_backtest_and_portfolio_metrics() -> None:
    bm = BacktestMetrics(
        total_trades=10,
        win_rate=0.6,
        profit_factor=1.5,
        net_profit=500.0,
        maximum_drawdown=0.1,
        sharpe_ratio=1.2,
        kelly_criterion=0.15,
        expectancy=50.0,
        system_quality_number=2.5,
        average_win=100.0,
        average_loss=-50.0,
        average_maximum_adverse_excursion=-20.0,
        average_maximum_favorable_excursion=80.0,
        risk_of_ruin=0.01,
        benchmark_return=0.08,
        strategy_return=0.15,
        kelly_mean=0.15,
        kelly_std=0.02,
        kelly_safe=0.1,
        market_exposure_pct=0.5,
        risk_adjusted_benchmark=0.07,
        exposure_efficiency=1.8,
        return_over_maximum_drawdown=1.5,
        diversification_score=0.8,
    )
    assert bm.total_trades == 10

    pm = PortfolioMetrics(
        combined_mean_kelly=0.2,
        safe_kelly_25=0.15,
        correlation_fail_rate=0.05,
        suggested_multiplier=1.0,
        leveraged_max_drawdown=0.12,
        max_concurrent_trades=3,
        max_total_exposure=10000.0,
        uncapped_multiplier=1.2,
        uncapped_max_total_exposure=12000.0,
        uncapped_leveraged_max_drawdown=0.15,
    )
    assert pm.max_concurrent_trades == 3


def test_croc_signal_post_init_sanitization_and_fallbacks() -> None:
    # Test valid iso string timestamp and exchange mapping
    sig = CrocSignal(
        symbol=" AAPL ",
        signal="BUY",
        timeframe="1D",
        close=150.0,
        high=152.0,
        low=148.0,
        wuk=1.0,
        status="ACTIVE",
        kerze="BULL",
        trend="UP",
        setter="SET1",
        welle="W1",
        timestamp="2026-08-01T10:00:00+00:00",
    )
    assert sig.symbol == "AAPL"
    assert sig.timestamp == datetime.fromisoformat("2026-08-01T10:00:00+00:00")
    assert sig.reference is not None
    assert sig.reference.startswith("AAPL_20260801100000")

    # Test invalid string timestamp fallback
    sig_invalid_ts = CrocSignal(
        symbol="MSFT",
        signal="BUY",
        timeframe="1D",
        close=200.0,
        high=205.0,
        low=198.0,
        wuk=1.0,
        status="ACTIVE",
        kerze="BULL",
        trend="UP",
        setter="SET1",
        welle="W1",
        timestamp="INVALID_DATE_STRING",
    )
    assert isinstance(sig_invalid_ts.timestamp, datetime)

    # Test BATS exchange fallback to UNKNOWN
    sig_bats = CrocSignal(
        symbol="UNKNOWN_SYMBOL",
        signal="BUY",
        timeframe="1D",
        close=10.0,
        high=11.0,
        low=9.0,
        wuk=1.0,
        status="ACTIVE",
        kerze="BULL",
        trend="UP",
        setter="SET1",
        welle="W1",
        exchange="BATS",
    )
    assert sig_bats.exchange == "UNKNOWN"

    row = sig.to_db_row()
    assert row["symbol"] == "AAPL"
    assert isinstance(row["timestamp"], str)


def test_signal_stat() -> None:
    stat = SignalStat(
        signal="BUY",
        symbol="SPY",
        timeframe="1D",
        level="L1",
        total="10",
        win="6",
        loss="4",
        rejected="0",
        win_rate="0.6",
        loss_rate="0.4",
    )
    assert stat.total == 10.0
    row = stat.to_db_row()
    assert "win_rate" not in row
    assert "updated_at" in row

    # Test conversion error handling
    invalid_stat = SignalStat(
        signal="BUY",
        symbol="SPY",
        timeframe="1D",
        level="L1",
        total="INVALID_NUM",
        win="0",
        loss="0",
        rejected="0",
        win_rate="0",
        loss_rate="0",
    )
    assert invalid_stat.total == "INVALID_NUM"


def test_market_price_factories() -> None:
    # Yahoo factory valid
    mp_y = MarketPrice.from_yahoo(
        "AAPL",
        {
            "date": "2026-08-10",
            "open": 150.0,
            "high": 155.0,
            "low": 149.0,
            "close": 153.0,
            "volume": 50000,
        },
    )
    assert mp_y.symbol == "AAPL"
    assert mp_y.provider == "yahoo"

    # Yahoo date strftime
    now = datetime.now()
    mp_y_date = MarketPrice.from_yahoo("AAPL", {"close": 100.0, "date": now})
    assert mp_y_date.date == now.strftime("%Y-%m-%d")

    # Yahoo negative close error
    with pytest.raises(ValueError, match="Negative close price"):
        MarketPrice.from_yahoo("AAPL", {"close": -5.0})

    # TradingView factory valid
    mp_tv = MarketPrice.from_tradingview(
        "MSFT",
        {
            "datetime": "2026-08-10T12:00:00",
            "open": 200.0,
            "high": 205.0,
            "low": 198.0,
            "close": 202.0,
            "volume": 30000,
        },
    )
    assert mp_tv.symbol == "MSFT"
    assert mp_tv.provider == "tradingview"

    # TradingView negative close error
    with pytest.raises(ValueError, match="Negative close price"):
        MarketPrice.from_tradingview("MSFT", {"close": -1.0})

    # to_db_row tuple
    tup = mp_y.to_db_row()
    assert tup[0] == "AAPL"
    assert tup[5] == 153.0
