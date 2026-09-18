"""Unit tests for app/models.py to achieve 100% test coverage."""

from datetime import datetime
from decimal import Decimal

import pytest

from app.models import (
    MarketPrice,
    Order,
    OrderLeg,
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
