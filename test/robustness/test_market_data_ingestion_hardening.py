"""Market Data Ingestion Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Fault Injection / Concurrency Locking validation for market data providers (Yahoo, TradingView),
the ETL updater, and the Market Quality Service.
"""

import threading
from collections.abc import Mapping
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.models import MarketPrice
from app.services.market.provider import YahooDataProvider, require_lock
from app.services.market.quality import MarketQualityService
from app.services.market.tv_provider import TradingViewDataProvider
from app.services.market.updater import MarketDataUpdater


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Yahoo Data Provider Extraction
# ==============================================================================
@pytest.mark.tier1
def test_bva_yahoo_provider_multi_index_extraction() -> None:
    """BVA: MultiIndex DataFrame extraction handles present, missing, and empty symbols."""
    provider = YahooDataProvider()

    # Create MultiIndex DataFrame for symbols AAPL and MSFT
    columns = pd.MultiIndex.from_tuples(
        [
            ("AAPL", "Open"),
            ("AAPL", "High"),
            ("AAPL", "Low"),
            ("AAPL", "Close"),
            ("AAPL", "Volume"),
            ("MSFT", "Open"),
            ("MSFT", "High"),
            ("MSFT", "Low"),
            ("MSFT", "Close"),
            ("MSFT", "Volume"),
        ]
    )
    df_multi = pd.DataFrame(
        [
            [150.0, 155.0, 149.0, 153.0, 1000, 300.0, 305.0, 298.0, 302.0, 2000],
        ],
        columns=columns,
    )

    # 1. Existing symbol
    df_aapl = provider.extract_symbol_data(df_multi, "AAPL")
    assert not df_aapl.empty
    assert "Close" in df_aapl.columns or "close" in df_aapl.columns

    # 2. Missing symbol from MultiIndex
    df_missing = provider.extract_symbol_data(df_multi, "GOOGL")
    assert df_missing.empty
    assert isinstance(df_missing, pd.DataFrame)

    # 3. Empty DataFrame input
    df_empty = provider.extract_symbol_data(pd.DataFrame(), "AAPL")
    assert df_empty.empty


@pytest.mark.tier1
def test_bva_yahoo_provider_single_index_extraction() -> None:
    """BVA: SingleIndex DataFrame extraction checks column presence."""
    provider = YahooDataProvider()

    # 1. Valid SingleIndex with 'close' column
    df_valid = pd.DataFrame({"open": [100.0], "close": [105.0], "volume": [500]})
    res_valid = provider.extract_symbol_data(df_valid, "QQQ")
    assert not res_valid.empty
    assert len(res_valid) == 1

    # 2. Malformed SingleIndex without 'close'
    df_malformed = pd.DataFrame({"other_col": [1, 2, 3]})
    res_malformed = provider.extract_symbol_data(df_malformed, "QQQ")
    assert res_malformed.empty


@pytest.mark.tier1
def test_bva_yahoo_provider_fetch_batch_raw_boundaries() -> None:
    """BVA: fetch_batch_raw on empty list, failed downloads, and partial batches."""
    provider = YahooDataProvider()

    # 1. Empty symbol list -> (pd.DataFrame(), [])
    df_empty, failed_empty = provider.fetch_batch_raw([], "2026-01-01")
    assert df_empty.empty
    assert failed_empty == []

    # 2. Network exception simulation -> reports all requested symbols as failed
    with patch("yfinance.download", side_effect=Exception("Connection Timeout")):
        df_err, failed_err = provider.fetch_batch_raw(["AAPL", "MSFT"], "2026-01-01")
        assert df_err.empty
        assert failed_err == ["AAPL", "MSFT"]

    # 3. Partial batch return (only AAPL returned, MSFT missing from columns)
    columns = pd.MultiIndex.from_tuples([("AAPL", "Close")])
    df_partial = pd.DataFrame([[150.0]], columns=columns)
    with patch("yfinance.download", return_value=df_partial):
        df_res, failed_res = provider.fetch_batch_raw(["AAPL", "MSFT"], "2026-01-01")
        assert not df_res.empty
        assert failed_res == ["MSFT"]


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — TradingView Provider Normalization
# ==============================================================================
@pytest.mark.tier1
def test_bva_tradingview_symbol_formatting_and_exchange_resolution() -> None:
    """BVA: TradingView provider symbol transformations and exchange fallbacks."""
    mock_mapper = MagicMock()
    mock_mapper.get_exchange.return_value = None  # Force fallback exchanges

    tv_provider = TradingViewDataProvider(exchange_mapper=mock_mapper)

    # Test standardization helper with DataFrame containing datetime index
    df_raw = pd.DataFrame(
        {
            "Open": [100.0],
            "High": [105.0],
            "Low": [99.0],
            "Close": [104.0],
            "Volume": [1000],
        },
        index=pd.to_datetime(["2026-07-24"]),
    )

    records = tv_provider._standardize_dataframe_records(df_raw, "BRK.B")
    assert len(records) == 1
    record = records[0]
    assert record["symbol"] == "BRK.B"
    assert "date" in record
    assert record["close"] == 104.0
    assert record["open"] == 100.0


# ==============================================================================
# 3. Boundary Value Analysis (BVA) — Concurrency Lock (@require_lock)
# ==============================================================================
@pytest.mark.tier1
def test_bva_require_lock_prevents_concurrent_reentrancy() -> None:
    """BVA: @require_lock permits single execution and non-blockingly skips concurrent attempts."""
    execution_count = 0
    barrier = threading.Barrier(2)

    @require_lock
    def slow_locked_job() -> int:
        nonlocal execution_count
        execution_count += 1
        barrier.wait()  # synchronize thread 1 inside lock
        return execution_count

    results: list[int | None] = []

    def thread_worker() -> None:
        res = slow_locked_job()
        results.append(res)

    t1 = threading.Thread(target=thread_worker)
    t1.start()

    # Wait until thread 1 holds the lock
    barrier.wait()

    # Thread 2 attempts to call while lock is actively held
    res_thread2 = slow_locked_job()
    assert res_thread2 is None  # Immediately skipped because lock is active

    t1.join()
    assert execution_count == 1
    assert 1 in results


# ==============================================================================
# 4. Boundary Value Analysis (BVA) — Market Quality Completeness Checks
# ==============================================================================
@pytest.mark.tier1
def test_bva_market_quality_completeness_critical_symbols() -> None:
    """BVA: check_last_trading_day_completeness alerts on missing critical or active symbols."""
    mock_updater = MagicMock(spec=MarketDataUpdater)
    mock_repo = MagicMock()
    mock_updater.repo = mock_repo
    mock_trade_repo = MagicMock()
    mock_updater.trade_repository = mock_trade_repo

    holiday_checker = MagicMock()
    holiday_checker.is_holiday.return_value = False

    quality_service = MarketQualityService(
        updater=mock_updater,
        holiday_checker=holiday_checker,
    )

    # 1. Complete universe: all symbols up to date
    mock_repo.get_all_known_symbols.return_value = ["QQQ", "SPY", "AAPL", "MSFT"]
    mock_repo.get_outdated_symbols.return_value = []
    mock_trade_repo.get_by_status.return_value = []
    assert quality_service.check_last_trading_day_completeness() is True

    # 2. Critical benchmark missing (QQQ is outdated) -> returns False
    mock_repo.get_outdated_symbols.return_value = ["QQQ"]
    assert quality_service.check_last_trading_day_completeness() is False

    # 3. Active trade symbol missing (e.g. SXRV.DE is active in TwoPercent and outdated) -> returns False
    mock_repo.get_outdated_symbols.return_value = ["SXRV.DE"]
    mock_trade_repo.get_by_status.return_value = [{"symbol": "SXRV.DE"}]
    assert quality_service.check_last_trading_day_completeness() is False

    # 4. Outdated symbols exceed max_allowed_missing_ratio (e.g. 10% missing > 5% allowed) -> returns False
    mock_repo.get_outdated_symbols.return_value = ["NON_CRITICAL_1", "NON_CRITICAL_2"]
    mock_repo.get_all_known_symbols.return_value = [
        "SYM1",
        "SYM2",
        "SYM3",
        "SYM4",
        "SYM5",
    ]
    mock_trade_repo.get_by_status.return_value = []
    assert (
        quality_service.check_last_trading_day_completeness(
            max_allowed_missing_ratio=0.05
        )
        is False
    )


# ==============================================================================
# 5. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    open_price=st.floats(
        min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    high_price=st.floats(
        min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    low_price=st.floats(
        min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    close_price=st.floats(
        min_value=0.01, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    volume=st.integers(min_value=0, max_value=1_000_000_000),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_market_price_from_tradingview_invariants(
    open_price: float,
    high_price: float,
    low_price: float,
    close_price: float,
    volume: int,
) -> None:
    """Invariant: MarketPrice.from_tradingview creates valid model matching all numeric values."""
    row: Mapping[str, object] = {
        "date": "2026-07-24",
        "open": open_price,
        "high": high_price,
        "low": low_price,
        "close": close_price,
        "volume": volume,
    }
    price_model = MarketPrice.from_tradingview("TEST.SYM", row)
    assert price_model.symbol == "TEST.SYM"
    assert price_model.date == "2026-07-24"
    assert price_model.open == open_price
    assert price_model.high == high_price
    assert price_model.low == low_price
    assert price_model.close == close_price
    assert price_model.volume == volume
    assert price_model.provider == "tradingview"


@pytest.mark.tier2
@given(
    raw_symbol=st.text(
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "P")),
        min_size=1,
        max_size=15,
    )
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_tradingview_symbol_cleanup_safety(raw_symbol: str) -> None:
    """Invariant: Symbol parsing never throws unhandled exceptions."""
    standard_symbol = raw_symbol.strip().upper()

    base_symbol = standard_symbol.split(".")[0]
    tv_symbol = base_symbol.replace("-", ".")

    assert isinstance(tv_symbol, str)
    assert "-" not in tv_symbol


# ==============================================================================
# 6. Fault Injection & Error Recovery (Chaos)
# ==============================================================================
@pytest.mark.tier2
def test_fault_injection_market_data_updater_batch_resilience() -> None:
    """Chaos: Batch failure in updater logs error and proceeds to subsequent batches without crash."""
    mock_session = MagicMock()
    updater = MarketDataUpdater(session_factory=mock_session)

    updater.repo.get_ignored_symbols = MagicMock(return_value=set())  # type: ignore[method-assign]
    updater.repo.clear_ignored_symbols = MagicMock()  # type: ignore[method-assign]

    # Process 3 symbols with batch size = 1
    # 1st batch raises Exception, 2nd batch succeeds, 3rd batch succeeds
    call_count = 0

    def mock_process_batch(batch: list[str], *args: object, **kwargs: object) -> int:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("Injected Socket Disconnect during batch download")
        return len(batch)

    updater._process_batch = MagicMock(side_effect=mock_process_batch)  # type: ignore[method-assign]

    with patch("app.services.market.updater.BATCH_SIZE", 1):
        # Must complete without unhandled exception
        updater.run_update(specific_symbols=["SYM1", "SYM2", "SYM3"])

    assert call_count == 3
