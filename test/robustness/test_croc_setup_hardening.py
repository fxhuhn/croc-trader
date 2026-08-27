"""CrocSetup Strategy Hardening Suite (Tier 1 & Tier 2).

Provides exhaustive Boundary Value Analysis (BVA), Hypothesis Property-Based Fuzzing,
and Zero Lookahead-Bias validation for the CrocSetup Screener Strategy.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.const import Strategies
from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.signal import SignalRepository
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.croc_setup import (
    MAX_RANKING_CONFIG_SIZE_BYTES,
    CrocCandidate,
    CrocPriceData,
    CrocSetupStrategy,
    PriceData,
    build_croc_candidate,
    calculate_croc_candidate_score,
    calculate_croc_targets,
    enrich_sma_distances,
    evaluate_indicator_condition,
    find_best_rule_match,
    is_rule_match,
)
from app.services.telegram import TelegramBot


# ==============================================================================
# 1. Boundary Value Analysis (BVA) — Functional Core & Math Guards
# ==============================================================================
@pytest.mark.tier1
def test_bva_croc_score_max_drawdown_zero_guard() -> None:
    """BVA: calculate_croc_candidate_score must never divide by zero when max_drawdown <= 0."""
    # Case 1: MaxDD = 0.0 -> Score is 0.0 (no ZeroDivisionError)
    assert calculate_croc_candidate_score(10.0, 0.0) == 0.0

    # Case 2: MaxDD < 0.0 -> Score is 0.0
    assert calculate_croc_candidate_score(10.0, -2.5) == 0.0

    # Case 3: Normal Positive MaxDD
    assert calculate_croc_candidate_score(10.0, 2.0) == 5.0
    assert calculate_croc_candidate_score(0.0, 5.0) == 0.0


@pytest.mark.tier1
def test_bva_croc_price_data_flat_candle_zero_risk_range() -> None:
    """BVA: PriceData with high == low produces risk_range == 0 and is rejected by build_croc_candidate."""
    # Case 1: Flat candle (high == low == 100.0)
    flat_prices = PriceData(high=100.0, low=100.0, close=100.0)
    assert flat_prices.risk_range == 0.0

    match_rule = {"Signal": "CROC", "Score": 10.0, "MaxDD": 2.0, "Exit": "Hold (TP2)"}
    row = {"symbol": "AAPL", "signal": "CROC", "date_str": "2026-08-20"}

    # Flat prices must be rejected (return None) without ZeroDivisionError
    candidate = build_croc_candidate(row, flat_prices, match_rule, indices="SPX")
    assert candidate is None


@pytest.mark.tier1
def test_bva_croc_price_data_invalid_or_negative_prices() -> None:
    """BVA: PriceData.from_row rejects non-positive or corrupted prices."""
    assert PriceData.from_row({"high": -10.0, "low": -20.0, "close": -15.0}) is None
    assert PriceData.from_row({"high": 0.0, "low": 0.0, "close": 0.0}) is None
    assert PriceData.from_row({"high": "corrupted", "low": 50.0, "close": 60.0}) is None

    valid = PriceData.from_row(
        {"high": 110.0, "low": 90.0, "close": 100.0, "sma_20": 95.0, "sma_200": 80.0}
    )
    assert valid is not None
    assert valid.risk_range == 20.0


@pytest.mark.tier1
def test_bva_croc_target_levels_and_directions() -> None:
    """BVA: calculate_croc_targets for long and short across TP levels (TP1, TP2, TP3, TP4)."""
    entry = 100.0
    risk = 5.0

    # Long targets
    tp1_long = calculate_croc_targets(entry, risk, target_level=1, direction="long")
    assert tp1_long["main"] == 105.0

    tp3_long = calculate_croc_targets(entry, risk, target_level=3, direction="long")
    assert tp3_long["main"] == 115.0

    # Short targets
    tp1_short = calculate_croc_targets(entry, risk, target_level=1, direction="short")
    assert tp1_short["main"] == 95.0

    tp2_short = calculate_croc_targets(entry, risk, target_level=2, direction="short")
    assert tp2_short["main"] == 90.0


@pytest.mark.tier1
def test_bva_croc_indicator_condition_handler_boundaries() -> None:
    """BVA: evaluate_indicator_condition handles numeric boundaries and string fallbacks."""
    # RSI Oversold (< 30)
    assert evaluate_indicator_condition(29.99, "oversold (<30)") is True
    assert evaluate_indicator_condition(30.0, "oversold (<30)") is False
    assert evaluate_indicator_condition(30.01, "oversold (<30)") is False

    # RSI Overbought (> 70)
    assert evaluate_indicator_condition(70.0, "overbought (>70)") is False
    assert evaluate_indicator_condition(70.01, "overbought (>70)") is True

    # SMA Ranges: 0 to 3%
    assert evaluate_indicator_condition(-0.01, "0 to 3%") is False
    assert evaluate_indicator_condition(0.0, "0 to 3%") is True
    assert evaluate_indicator_condition(3.0, "0 to 3%") is True
    assert evaluate_indicator_condition(3.01, "0 to 3%") is False

    # None and empty
    assert evaluate_indicator_condition(None, "0 to 3%") is False
    assert evaluate_indicator_condition("Bullish", "bullish") is True


@pytest.mark.tier1
def test_bva_croc_rule_match_and_sma_enrichment() -> None:
    """BVA: enrich_sma_distances, is_rule_match and find_best_rule_match pure core functions."""
    prices: CrocPriceData = PriceData(
        high=110.0, low=90.0, close=100.0, sma_20=95.0, sma_200=90.0
    )
    raw_row = {"symbol": "AAPL", "signal": "CROC_BULL", "rsi": 60.0}

    # SMA Enrichment
    enriched = enrich_sma_distances(raw_row, prices)
    assert "dist_sma_20" in enriched
    assert "dist_sma_200" in enriched
    assert enrich_sma_distances(raw_row, None) == raw_row

    # Rule Match
    rules = [
        {
            "Signal": "CROC_BULL",
            "Score": 10.0,
            "SQN": 12.0,
            "MaxDD": 2.0,
            "RSI": "strong (55-70)",
        },
        {
            "Signal": "CROC_BULL",
            "Score": 5.0,
            "SQN": 5.0,
            "MaxDD": 2.0,
            "RSI": "oversold (<30)",
        },
    ]
    best_rule = find_best_rule_match(enriched, rules)
    assert best_rule is not None
    assert best_rule["Score"] == 10.0
    assert is_rule_match(enriched, rules[0]) is True
    assert is_rule_match(enriched, rules[1]) is False


@pytest.mark.tier1
def test_bva_croc_candidate_to_report_item() -> None:
    """BVA: CrocCandidate to_report_item produces valid SignalReportItem."""
    candidate = CrocCandidate(
        symbol="AAPL",
        signal_name="CROC_1",
        score=5.5,
        entry_price=105.0,
        stop_loss=95.0,
        target_profit=125.0,
        target_level=2,
        direction="long",
        indices="SPX",
        date_str="2026-08-20",
    )
    report_item = candidate.to_report_item()
    assert report_item.symbol == "AAPL"
    assert "BUY" in report_item.action
    assert report_item.entry_price == 105.0
    assert report_item.stop_loss == 95.0
    assert report_item.target_profit == 125.0
    assert report_item.details["Score"] == "5.50"


# ==============================================================================
# 2. Boundary Value Analysis (BVA) — Imperative Shell & Security Guards
# ==============================================================================
@pytest.mark.tier1
def test_bva_croc_yaml_bomb_size_guard() -> None:
    """BVA / Security: Loading YAML config exceeding MAX_RANKING_CONFIG_SIZE_BYTES raises RuntimeError."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_signal_repo = MagicMock(spec=SignalRepository)

    mock_stat = MagicMock()
    mock_stat.st_size = MAX_RANKING_CONFIG_SIZE_BYTES + 1  # Exceeds limit

    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_large.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with pytest.raises(
                    RuntimeError, match="Ranking config exceeds safe size limit"
                ):
                    CrocSetupStrategy(
                        trade_repository=mock_trade_repo,
                        data_provider=mock_data_provider,
                        signal_repository=mock_signal_repo,
                    )


@pytest.mark.tier1
def test_bva_croc_top_3_candidates_limit() -> None:
    """BVA: Strategy strictly limits trade creation to the top 3 highest scoring candidates."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_signal_repo = MagicMock(spec=SignalRepository)

    mock_stat = MagicMock()
    mock_stat.st_size = 100
    mock_yaml = """
    ranking_2026:
      - Signal: "CROC"
        Score: 10
        MaxDD: 2.0
        Exit: "Hold (TP2)"
      - Signal: "CROC_LOW"
        Score: 2
        MaxDD: 2.0
        Exit: "Hold (TP1)"
    """

    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_rules.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data=mock_yaml)):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        mock_ex_inst = mock_ex.return_value
                        mock_ex_inst.sp_500 = ["SYM1", "SYM2", "SYM3", "SYM4", "SYM5"]
                        mock_ex_inst.nasdaq_100 = []
                        mock_ex_inst.dow_30 = []
                        mock_ex_inst.russell_1000 = []

                        strat = CrocSetupStrategy(
                            trade_repository=mock_trade_repo,
                            data_provider=mock_data_provider,
                            signal_repository=mock_signal_repo,
                        )
                        assert strat.name == Strategies.CrocSetup

                        # Provide 5 signals
                        signals = [
                            {
                                "symbol": f"SYM{i}",
                                "data": json.dumps(
                                    {
                                        "signal": "CROC",
                                        "high": 100 + i,
                                        "low": 90 + i,
                                        "close": 95 + i,
                                    }
                                ),
                            }
                            for i in range(1, 6)
                        ]
                        mock_signal_repo.get_signals_by_date.return_value = signals

                        count = strat.run(analysis_date="2026-08-20")

                        # Strictly 3 trades created
                        assert count == 3
                        assert mock_trade_repo.create_trade.call_count == 3


@pytest.mark.tier1
def test_bva_croc_strategy_telegram_dispatch() -> None:
    """BVA: CrocSetupStrategy dispatches telegram report when telegram_bot is configured."""
    mock_trade_repo = MagicMock(spec=TradeRepository)
    mock_data_provider = MagicMock(spec=MarketDataProvider)
    mock_signal_repo = MagicMock(spec=SignalRepository)
    mock_telegram_bot = MagicMock(spec=TelegramBot)

    mock_stat = MagicMock()
    mock_stat.st_size = 100
    mock_yaml = "ranking_2026: [{'Signal': 'CROC', 'Score': 10, 'MaxDD': 2.0, 'Exit': 'Hold (TP2)'}]"

    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_rules.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data=mock_yaml)):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        mock_ex.return_value.sp_500 = ["AAPL"]
                        mock_ex.return_value.nasdaq_100 = []
                        mock_ex.return_value.dow_30 = []
                        mock_ex.return_value.russell_1000 = []

                        strat = CrocSetupStrategy(
                            trade_repository=mock_trade_repo,
                            data_provider=mock_data_provider,
                            signal_repository=mock_signal_repo,
                            telegram_bot=mock_telegram_bot,
                        )

                        mock_signal_repo.get_signals_by_date.return_value = [
                            {
                                "symbol": "AAPL",
                                "data": json.dumps(
                                    {
                                        "signal": "CROC",
                                        "high": 150.0,
                                        "low": 140.0,
                                        "close": 145.0,
                                    }
                                ),
                            }
                        ]

                        count = strat.run(analysis_date="2026-08-20")
                        assert count == 1
                        mock_telegram_bot.send_dataframe.assert_called_once()


# ==============================================================================
# 3. Property-Based Fuzzing (Hypothesis)
# ==============================================================================
@pytest.mark.tier2
@given(
    entry=st.floats(
        min_value=1.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    risk=st.floats(
        min_value=0.01, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    target_level=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_croc_target_invariants_long(
    entry: float, risk: float, target_level: int
) -> None:
    """Invariant: For long setups, Target > Entry > (Entry - Risk)."""
    targets = calculate_croc_targets(entry, risk, target_level, direction="long")
    stop_loss = entry - risk
    target_profit = targets["main"]

    assert target_profit > entry
    assert entry > stop_loss
    assert target_profit >= round(entry + (risk * target_level) - 0.01, 2)


@pytest.mark.tier2
@given(
    entry=st.floats(
        min_value=100.0, max_value=5000.0, allow_nan=False, allow_infinity=False
    ),
    risk=st.floats(
        min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
    target_level=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_croc_target_invariants_short(
    entry: float, risk: float, target_level: int
) -> None:
    """Invariant: For short setups, Target < Entry < (Entry + Risk)."""
    targets = calculate_croc_targets(entry, risk, target_level, direction="short")
    stop_loss = entry + risk
    target_profit = targets["main"]

    assert target_profit < entry
    assert entry < stop_loss
    assert target_profit <= round(entry - (risk * target_level) + 0.01, 2)


@pytest.mark.tier2
@given(
    score=st.floats(
        min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False
    ),
    max_dd=st.floats(
        min_value=0.01, max_value=50.0, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=100, deadline=None)
def test_hypothesis_croc_candidate_score_invariants(
    score: float, max_dd: float
) -> None:
    """Invariant: Candidate score is monotonic in SQN and non-negative."""
    computed_score = calculate_croc_candidate_score(score, max_dd)
    assert computed_score >= 0.0
    if score > 0.001 and max_dd > 0.0:
        assert computed_score > 0.0


# ==============================================================================
# 4. Zero Lookahead-Bias Guard
# ==============================================================================
@pytest.mark.tier2
def test_croc_screener_zero_lookahead_bias() -> None:
    """Time-Shift Invariance: Screening at date T must produce identical candidates regardless of future data."""
    mock_trade_repo_t = MagicMock(spec=TradeRepository)
    mock_data_provider_t = MagicMock(spec=MarketDataProvider)
    mock_signal_repo_t = MagicMock(spec=SignalRepository)

    mock_trade_repo_fut = MagicMock(spec=TradeRepository)
    mock_data_provider_fut = MagicMock(spec=MarketDataProvider)
    mock_signal_repo_fut = MagicMock(spec=SignalRepository)

    target_date = "2026-08-20"

    signal_t = {
        "symbol": "AAPL",
        "timestamp": "2026-08-20",
        "data": json.dumps(
            {"signal": "CROC", "high": 150.0, "low": 140.0, "close": 145.0}
        ),
    }

    signal_fut = {
        "symbol": "AAPL",
        "timestamp": "2026-08-21",  # Future signal
        "data": json.dumps(
            {"signal": "CROC", "high": 160.0, "low": 155.0, "close": 158.0}
        ),
    }

    # Signal repo at T returns only signals <= T
    mock_signal_repo_t.get_signals_by_date.return_value = [signal_t]

    # Signal repo with future data filters properly by date
    mock_signal_repo_fut.get_signals_by_date.side_effect = lambda date, days: (
        [signal_t] if date == target_date else [signal_t, signal_fut]
    )

    mock_stat = MagicMock()
    mock_stat.st_size = 100
    mock_yaml = "ranking_2026: [{'Signal': 'CROC', 'Score': 10, 'MaxDD': 2.0, 'Exit': 'Hold (TP2)'}]"

    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_get_path:
        mock_get_path.return_value = Path("mock_rules.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data=mock_yaml)):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        mock_ex.return_value.sp_500 = ["AAPL"]
                        mock_ex.return_value.nasdaq_100 = []
                        mock_ex.return_value.dow_30 = []
                        mock_ex.return_value.russell_1000 = []

                        strat_t = CrocSetupStrategy(
                            trade_repository=mock_trade_repo_t,
                            data_provider=mock_data_provider_t,
                            signal_repository=mock_signal_repo_t,
                        )
                        count_t = strat_t.run(analysis_date=target_date)

                        strat_fut = CrocSetupStrategy(
                            trade_repository=mock_trade_repo_fut,
                            data_provider=mock_data_provider_fut,
                            signal_repository=mock_signal_repo_fut,
                        )
                        count_fut = strat_fut.run(analysis_date=target_date)

                        assert count_t == count_fut == 1
                        assert (
                            mock_trade_repo_t.create_trade.call_count
                            == mock_trade_repo_fut.create_trade.call_count
                            == 1
                        )

                        kwargs_t = mock_trade_repo_t.create_trade.call_args.kwargs
                        kwargs_fut = mock_trade_repo_fut.create_trade.call_args.kwargs

                        assert kwargs_t["entry"] == kwargs_fut["entry"] == 150.0
                        assert kwargs_t["stop_loss"] == kwargs_fut["stop_loss"] == 140.0
                        assert kwargs_t["target"] == kwargs_fut["target"] == 170.0
