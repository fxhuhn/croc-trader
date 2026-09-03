# filename: test_croc_setup_pinning.py
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.signal import SignalRepository
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.croc_setup import (
    CrocCandidate,
    CrocSetupStrategy,
    PriceData,
    build_croc_candidate,
    calculate_croc_candidate_score,
    enrich_sma_distances,
    find_best_rule_match,
)
from app.services.telegram import TelegramBot


def test_price_data_boundary_conditions() -> None:
    """Verifies PriceData edge cases including zero bounds and fallbacks."""
    # Zero close and zero high -> None
    assert PriceData.from_row({"close": 0.0, "high": 0.0}) is None
    assert PriceData.from_row({"close": -5.0, "high": 0.0}) is None

    # High <= 0 falls back to close
    p1 = PriceData.from_row({"close": 100.0, "high": 0.0, "low": 90.0})
    assert p1 is not None
    assert p1.high == 100.0
    assert p1.close == 100.0

    # Low <= 0 falls back to close
    p2 = PriceData.from_row({"close": 100.0, "high": 110.0, "low": 0.0})
    assert p2 is not None
    assert p2.low == 100.0

    # Risk range never negative
    p3 = PriceData(high=90.0, low=100.0, close=95.0)
    assert p3.risk_range == 0.0


def test_find_best_rule_match_pinning() -> None:
    """Verifies rule matching, score extraction, and tie-breaking by rule length."""
    rules = [
        {"Signal": "CrocBuy", "Score": 10.0, "rsi": "strong"},
        # Same score, but more indicator conditions (rsi + sma) -> should win tie
        {"Signal": "CrocBuy", "Score": 10.0, "rsi": "strong", "dist_sma_20": "0 to 3%"},
        {"Signal": "CrocSell", "Score": 15.0},
    ]

    row_buy = {"signal": "CrocBuy", "rsi": 60.0, "dist_sma_20": 2.5}
    best = find_best_rule_match(row_buy, rules)
    assert best is not None
    assert "dist_sma_20" in best

    # Rule matching via boolean flag when signal is 'none'
    row_none = {"signal": "none", "bull_1": "1"}
    rule_flag = [{"Signal": "bull_1", "Score": 5.0}]
    assert find_best_rule_match(row_none, rule_flag) is not None

    # No matching rule returns None
    row_nomatch = {"signal": "UnknownSignal"}
    assert find_best_rule_match(row_nomatch, rules) is None


def test_build_croc_candidate_pinning() -> None:
    """Verifies build_croc_candidate construction and calculations."""
    row = {
        "symbol": "AAPL",
        "signal": "CrocLong",
        "date_str": "2026-05-01",
    }
    prices = PriceData(high=150.0, low=140.0, close=148.0)
    match_rule = {
        "direction": "long",
        "Exit": "tp2",
        "SQN": 3.0,
        "MaxDD": 10.0,
        "Signal": "CrocLong",
    }

    # Invalid index returns None
    assert build_croc_candidate(row, prices, match_rule, "-") is None

    # Zero risk range returns None
    flat_prices = PriceData(high=150.0, low=150.0, close=150.0)
    assert build_croc_candidate(row, flat_prices, match_rule, "NDX") is None

    # Long candidate calculation
    cand_long = build_croc_candidate(row, prices, match_rule, "NDX")
    assert cand_long is not None
    assert cand_long.symbol == "AAPL"
    assert cand_long.direction == "long"
    assert cand_long.entry_price == 150.0
    assert cand_long.stop_loss == 140.0  # 150 - 10
    assert cand_long.target_level == 2
    assert cand_long.date_str == "2026-05-01"

    # Short candidate calculation
    match_short = {
        "direction": "short",
        "Exit": "tp1",
        "Score": 2.5,
    }
    cand_short = build_croc_candidate(row, prices, match_short, "SPX")
    assert cand_short is not None
    assert cand_short.direction == "short"
    assert cand_short.entry_price == 140.0
    assert cand_short.stop_loss == 150.0  # 140 + 10
    assert cand_short.target_level == 1


def test_enrich_sma_distances_pinning() -> None:
    """Verifies percentage distance computation from SMAs."""
    row: dict[str, object] = {"symbol": "MSFT"}

    # Zero prices or missing SMAs
    p_zero = PriceData(high=100.0, low=90.0, close=95.0, sma_20=0.0, sma_200=0.0)
    assert enrich_sma_distances(row, p_zero) == row
    assert enrich_sma_distances(row, None) == row

    # Valid SMAs
    p_valid = PriceData(high=110.0, low=90.0, close=100.0, sma_20=90.0, sma_200=80.0)
    enriched = enrich_sma_distances(row, p_valid)
    assert pytest.approx(enriched["dist_sma_20"]) == (100.0 - 90.0) / 90.0 * 100.0
    assert pytest.approx(enriched["dist_sma_200"]) == (100.0 - 80.0) / 80.0 * 100.0


def test_croc_setup_strategy_run_workflow() -> None:
    """Verifies end-to-end run method including limits, filters, and persistence."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_signal_repo = MagicMock(spec=SignalRepository)
    mock_telegram_bot = MagicMock(spec=TelegramBot)

    mock_stat = MagicMock()
    mock_stat.st_size = 100

    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_ranking.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data="ranking_2026: []")):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        mock_ex.return_value.nasdaq_100 = [
                            "AAPL",
                            "MSFT",
                            "NVDA",
                            "AMZN",
                        ]
                        mock_ex.return_value.sp_500 = []
                        mock_ex.return_value.dow_30 = []
                        mock_ex.return_value.russell_1000 = []

                        strategy = CrocSetupStrategy(
                            trade_repository=mock_trade_repo,
                            data_provider=mock_data_provider,
                            signal_repository=mock_signal_repo,
                            telegram_bot=mock_telegram_bot,
                        )

    # Provide 4 real candidates
    c1 = CrocCandidate(
        symbol="AAPL",
        signal_name="CrocLong",
        score=10.0,
        date_str="2026-05-01",
        entry_price=100.0,
        stop_loss=90.0,
        target_profit=120.0,
        target_level=1,
        direction="long",
        indices="NDX",
    )
    c2 = CrocCandidate(
        symbol="MSFT",
        signal_name="CrocLong",
        score=9.0,
        date_str="2026-05-01",
        entry_price=200.0,
        stop_loss=190.0,
        target_profit=220.0,
        target_level=1,
        direction="long",
        indices="NDX",
    )
    c3 = CrocCandidate(
        symbol="NVDA",
        signal_name="CrocLong",
        score=8.0,
        date_str="2026-05-01",
        entry_price=300.0,
        stop_loss=280.0,
        target_profit=340.0,
        target_level=1,
        direction="long",
        indices="NDX",
    )
    c4 = CrocCandidate(
        symbol="AMZN",
        signal_name="CrocLong",
        score=7.0,
        date_str="2026-05-01",
        entry_price=400.0,
        stop_loss=380.0,
        target_profit=440.0,
        target_level=1,
        direction="long",
        indices="NDX",
    )

    with patch.object(
        strategy, "_fetch_and_sort_candidates", return_value=[c1, c2, c3, c4]
    ):
        # Run top 3 cap
        count = strategy.run(days=1, analysis_date="2026-05-01")
        assert count == 3
        assert mock_trade_repo.create_trade.call_count == 3

        # Run with specific_symbols filter
        mock_trade_repo.create_trade.reset_mock()
        count_filtered = strategy.run(
            days=1, analysis_date="2026-05-01", specific_symbols=["MSFT"]
        )
        assert count_filtered == 1
        assert mock_trade_repo.create_trade.call_count == 1
        call_kwargs = mock_trade_repo.create_trade.call_args[1]
        assert call_kwargs["symbol"] == "MSFT"

        # Default analysis date when not provided
        mock_signal_repo.get_latest_signal_date.return_value = "2026-05-01"
        strategy.run(days=0, analysis_date=None)
        assert mock_signal_repo.get_latest_signal_date.called

        # Test get_all_recommendations
        recs = strategy.get_all_recommendations(days=1, analysis_date="2026-05-01")
        assert len(recs) == 4
        assert recs[0]["Symbol"] == "AAPL"
        assert recs[0]["Score"] == 10.0


def test_croc_setup_legacy_adapters_and_candidates() -> None:
    """Verifies legacy test adapters and candidate creation helper methods."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_signal_repo = MagicMock(spec=SignalRepository)
    mock_telegram_bot = MagicMock(spec=TelegramBot)

    mock_stat = MagicMock()
    mock_stat.st_size = 100

    rules_yaml = """
ranking_2026:
  - Signal: CrocBuy
    Score: 12.5
    rsi: strong
    Exit: tp1
    direction: long
"""
    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_ranking.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data=rules_yaml)):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        mock_ex.return_value.nasdaq_100 = ["AAPL"]
                        mock_ex.return_value.sp_500 = []
                        mock_ex.return_value.dow_30 = []
                        mock_ex.return_value.russell_1000 = []

                        strategy = CrocSetupStrategy(
                            trade_repository=mock_trade_repo,
                            data_provider=mock_data_provider,
                            signal_repository=mock_signal_repo,
                            telegram_bot=mock_telegram_bot,
                        )

    # Signal row with JSON data payload
    signal_row: dict[str, object] = {
        "symbol": "AAPL",
        "signal": "CrocBuy",
        "date_str": "2026-05-01",
        "data": json.dumps({"high": 150.0, "low": 140.0, "close": 145.0, "rsi": 60.0}),
    }

    # _find_croc_candidate
    cand = strategy._find_croc_candidate(signal_row)
    assert cand is not None
    assert cand.symbol == "AAPL"
    assert cand.score == 12.5
    assert cand.indices == "NDX"

    # _find_candidate adapter
    res_dict = strategy._find_candidate(signal_row)
    assert res_dict is not None
    assert "prices" in res_dict
    assert "match" in res_dict

    # _create_trade adapter
    prices = PriceData(high=150.0, low=140.0, close=145.0)
    match_dict = {
        "Signal": "CrocBuy",
        "Score": 12.5,
        "direction": "long",
        "Exit": "tp1",
    }
    trade_res = strategy._create_trade(signal_row, prices, match_dict)
    assert trade_res is not None
    assert trade_res["Symbol"] == "AAPL"
    assert mock_trade_repo.create_trade.called

    # _build_trade_recommendation adapter
    rec_res = strategy._build_trade_recommendation(signal_row, prices, match_dict)
    assert rec_res is not None
    assert rec_res["Symbol"] == "AAPL"
    assert "_internal" in rec_res

    # _process_single_signal adapter
    single_res = strategy._process_single_signal(signal_row)
    assert single_res is not None
    assert single_res["Symbol"] == "AAPL"

    # Helper adapters
    assert strategy._check_value(60.0, "strong") is True
    assert strategy._find_best_match({"signal": "CrocBuy", "rsi": 60.0}) is not None
    assert strategy._enrich_sma({"symbol": "AAPL"}, prices) is not None
    assert strategy._calc_targets(100.0, 10.0, 1)["main"] == 110.0


def test_additional_croc_coverage_branches() -> None:
    """Verifies edge branches: negative SQN, invalid direction, fetch sorting."""
    # Negative SQN
    assert calculate_croc_candidate_score(-5.0, 10.0) < 0.0

    # Direction fallback to long
    row = {"symbol": "TEST"}
    prices = PriceData(high=100.0, low=90.0, close=95.0)
    match_invalid_dir = {"direction": "sideways", "Score": 5.0}
    c = build_croc_candidate(row, prices, match_invalid_dir, "SPX")
    assert c is not None
    assert c.direction == "long"

    # _find_croc_candidate returning None on missing prices or unmatching rules
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_signal_repo = MagicMock(spec=SignalRepository)
    mock_telegram_bot = MagicMock(spec=TelegramBot)

    mock_stat = MagicMock()
    mock_stat.st_size = 100

    rules_yaml = "ranking_2026: [{'Signal': 'CrocBuy', 'Score': 10.0}]"
    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_ranking.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data=rules_yaml)):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        mock_ex.return_value.nasdaq_100 = ["AAPL"]
                        mock_ex.return_value.sp_500 = []
                        mock_ex.return_value.dow_30 = []
                        mock_ex.return_value.russell_1000 = []

                        strategy = CrocSetupStrategy(
                            trade_repository=mock_trade_repo,
                            data_provider=mock_data_provider,
                            signal_repository=mock_signal_repo,
                            telegram_bot=mock_telegram_bot,
                        )

    # Missing prices -> None
    assert strategy._find_croc_candidate({"symbol": "AAPL", "close": 0.0}) is None

    # Unmatching rule -> None
    assert (
        strategy._find_croc_candidate(
            {
                "symbol": "AAPL",
                "high": 100.0,
                "low": 90.0,
                "close": 95.0,
                "signal": "Unknown",
            }
        )
        is None
    )

    # Malformed JSON in data field logs warning and continues
    assert (
        strategy._find_croc_candidate({"symbol": "AAPL", "data": "invalid json"})
        is None
    )

    # _fetch_and_sort_candidates returns sorted list
    mock_signal_repo.get_signals_by_date.return_value = [
        {
            "symbol": "AAPL",
            "high": 100.0,
            "low": 90.0,
            "close": 95.0,
            "signal": "CrocBuy",
        },
    ]
    candidates = strategy._fetch_and_sort_candidates("2026-05-01", 1)
    assert len(candidates) == 1
    assert candidates[0].symbol == "AAPL"
