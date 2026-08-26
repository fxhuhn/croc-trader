from unittest.mock import MagicMock

import pandas as pd  # type: ignore[import-untyped]

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.trade import TradeRepository
from app.services.screener.models import SignalReportItem
from app.services.screener.strategies.base import BaseStrategy
from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy


class ConcreteTestStrategy(BaseStrategy[object]):
    """Minimal concrete subclass for BaseStrategy testing."""

    def run(self, days: int = 0, analysis_date: str | None = None) -> int:
        return 0


def test_signal_report_item_to_row_dict() -> None:
    """Tests SignalReportItem converts to clean row dictionary with details."""
    item = SignalReportItem(
        symbol="QQQ",
        action="BUY MKT",
        entry_price=480.50,
        stop_loss=460.00,
        target_profit=500.00,
        details={"RSI(2)": 15.2, "ATR%": 2.1, "Note": "High Conf"},
    )
    row = item.to_row_dict()
    assert row["Symbol"] == "QQQ"
    assert row["Action"] == "BUY MKT"
    assert row["Entry"] == "480.50"
    assert row["Stop"] == "460.00"
    assert row["TP"] == "500.00"
    assert row["RSI(2)"] == "15.20"
    assert row["ATR%"] == "2.10"
    assert row["Note"] == "High Conf"


def test_base_strategy_indices_string() -> None:
    """Tests _get_indices_string formats comma-separated index tags."""
    data_provider = MagicMock(spec=MarketDataProvider)
    strategy = ConcreteTestStrategy(data_provider=data_provider)
    mock_exchange_symbols = MagicMock()
    mock_exchange_symbols.dow_30 = ["AAPL"]
    mock_exchange_symbols.sp_500 = ["AAPL"]
    mock_exchange_symbols.nasdaq_100 = ["AAPL"]
    mock_exchange_symbols.russell_1000 = []
    strategy.exchange_symbols = mock_exchange_symbols

    res = strategy._get_indices_string("AAPL")
    assert "DOW" in res
    assert "SPX" in res
    assert "NDX" in res

    res_empty = strategy._get_indices_string("UNKNOWN")
    assert res_empty == "-"


def test_base_strategy_has_existing_position() -> None:
    """Tests _has_existing_trade_or_position handles open positions and exists query."""
    data_provider = MagicMock(spec=MarketDataProvider)
    strategy = ConcreteTestStrategy(data_provider=data_provider)
    mock_trade_repo = MagicMock(spec=TradeRepository)

    # 1. Existing ACTIVE trade
    mock_trade_repo.get_by_status.return_value = [
        {"symbol": "QQQ", "strategy": "bounce_bandit", "status": "ACTIVE"}
    ]
    assert strategy._has_existing_trade_or_position(
        mock_trade_repo, "QQQ", "bounce_bandit", "2026-02-20"
    )

    # 2. No open trades, but exists on date
    mock_trade_repo.get_by_status.return_value = []
    mock_trade_repo.exists.return_value = True
    assert strategy._has_existing_trade_or_position(
        mock_trade_repo, "QQQ", "bounce_bandit", "2026-02-20"
    )

    # 3. Clean state -> False
    mock_trade_repo.exists.return_value = False
    assert not strategy._has_existing_trade_or_position(
        mock_trade_repo, "QQQ", "bounce_bandit", "2026-02-20"
    )


def test_base_strategy_format_report_row_aliases() -> None:
    """Tests _format_report_row correctly resolves standard and alias column names."""
    data_provider = MagicMock(spec=MarketDataProvider)
    strategy = ConcreteTestStrategy(data_provider=data_provider)

    # 1. Standard Entry
    row_standard = pd.Series({"Symbol": "AAPL", "Action": "BUY LMT", "Entry": 150.25})
    res_standard = strategy._format_report_row(row_standard)
    assert res_standard is not None
    assert res_standard["Symbol"] == "AAPL"
    assert res_standard["Action"] == "BUY LMT"
    assert res_standard["Entry"] == "150.25"

    # 2. Limit Entry alias
    row_alias = pd.Series({"Symbol": "SXRV.DE", "Limit Entry": 594.10})
    res_alias = strategy._format_report_row(row_alias)
    assert res_alias is not None
    assert res_alias["Symbol"] == "SXRV.DE"
    assert res_alias["Action"] == "BUY"
    assert res_alias["Entry"] == "594.10"

    # 3. entry_price alias
    row_price = pd.Series({"symbol": "msft", "entry_price": 400.0, "Signal": "BUY"})
    res_price = strategy._format_report_row(row_price)
    assert res_price is not None
    assert res_price["Symbol"] == "MSFT"
    assert res_price["Action"] == "BUY"
    assert res_price["Entry"] == "400.00"


def test_turnover_timing_report_signals_to_telegram() -> None:
    """Tests _report_signals_to_telegram generates distinct entries for each factor."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_telegram = MagicMock()

    strategy = TurnoverTimingStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        telegram_bot=mock_telegram,
    )

    candidates = [
        {
            "symbol": "NVDA",
            "close": 120.0,
            "sma_price": 100.0,
            "sma_turnover": 50000000.0,
            "atr": 4.0,
            "indices": "NDX, SPX",
        }
    ]

    setup_date = pd.Timestamp("2026-02-20")
    strategy._report_signals_to_telegram(candidates, setup_date)

    assert mock_telegram.send_dataframe.called
    df = mock_telegram.send_dataframe.call_args[0][0]
    title = mock_telegram.send_dataframe.call_args[1].get("title", "")

    assert "Turnover Signals" in title
    assert len(df) == 2  # 0.5 ATR and 1.0 ATR entries

    # Row 0: 0.5 ATR -> 120 - (4 * 0.5) = 118.00
    assert df.iloc[0]["Symbol"] == "NVDA"
    assert df.iloc[0]["Action"] == "BUY LMT (0.5 ATR)"
    assert df.iloc[0]["Entry"] == "118.00"

    # Row 1: 1.0 ATR -> 120 - (4 * 1.0) = 116.00
    assert df.iloc[1]["Symbol"] == "NVDA"
    assert df.iloc[1]["Action"] == "BUY LMT (1.0 ATR)"
    assert df.iloc[1]["Entry"] == "116.00"


def test_tgim_report_signals_to_telegram() -> None:
    """Tests TGIMStrategy sends BUY MOC in telegram report."""
    from app.services.screener.strategies.tgim import TGIMStrategy

    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_trade_repo.exists.return_value = False
    mock_trade_repo.create_trade.return_value = 1

    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_data_provider.get_batch_history.return_value = {
        "SPY": pd.DataFrame(
            [
                {"date": "2026-02-19", "close": 500.0},
                {"date": "2026-02-20", "close": 498.0},
                {"date": "2026-02-23", "close": 490.0},
            ]
        )
    }
    mock_telegram = MagicMock()

    strategy = TGIMStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        telegram_bot=mock_telegram,
    )

    signals = strategy.run(analysis_date="2026-02-23")
    assert signals == 1

    assert mock_telegram.send_dataframe.called
    df = mock_telegram.send_dataframe.call_args[0][0]
    title = mock_telegram.send_dataframe.call_args[1].get("title", "")

    assert "TGIM" in title
    assert df.iloc[0]["Symbol"] == "SPY"
    assert df.iloc[0]["Action"] == "BUY MOC"
    assert df.iloc[0]["Entry"] == "498.00"


def test_dip_buyer_report_signals_to_telegram() -> None:
    """Tests DipBuyerStrategy sends report with clean strategy name."""
    from app.services.screener.strategies.dip_buyer import DipBuyerStrategy

    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_telegram = MagicMock()

    strategy = DipBuyerStrategy(
        trade_repository=mock_trade_repo,
        data_provider=mock_data_provider,
        telegram_bot=mock_telegram,
    )

    signals = pd.DataFrame(
        [
            {
                "close": 150.0,
                "high": 155.0,
                "volume": 2000000.0,
                "atr": 3.5,
                "sma200": 130.0,
                "atr_ratio_3day": 1.2,
                "ibs": 0.15,
                "setup_score": 85.0,
            }
        ],
        index=["AAPL"],
    )

    saved_count = strategy._process_signals(signals, pd.Timestamp("2026-08-25"))
    assert saved_count == 1
    assert mock_telegram.send_dataframe.called

    df = mock_telegram.send_dataframe.call_args[0][0]
    title = mock_telegram.send_dataframe.call_args[1].get("title", "")

    assert "dip_buyer" in title
    assert df.iloc[0]["Symbol"] == "AAPL"
    assert df.iloc[0]["Action"] == "BUY LMT"
    assert df.iloc[0]["Entry"] == "146.50"
    assert df.iloc[0]["LOC"] == "155.01"
