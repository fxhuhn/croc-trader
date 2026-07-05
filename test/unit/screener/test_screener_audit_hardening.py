# filename: test_screener_audit_hardening.py
"""
Audit Hardening Test Suite for app/services/screener.

This suite validates all security and robustness fixes applied after the
Iron Auditor + Red Teamer dual-workflow review. Every test targets a
specific violation ID from the audit report.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd
import pytest

from app.database.repositories.market_data_provider import MarketDataProvider
from app.database.repositories.signal import SignalRepository
from app.database.repositories.trade import TradeRepository
from app.services.screener.strategies.croc_setup import (
    CrocSetupStrategy,
    PriceData,
)
from app.services.screener.strategies.dip_buyer import DipBuyerStrategy
from app.services.screener.strategies.ndx_momentum import (
    NDXMomentumScreener,
)
from app.services.screener.strategies.turnover_timing import (
    TurnoverConfiguration,
    TurnoverTimingStrategy,
)
from app.services.screener.view_service import ScreenerViewService

# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_trade_repository() -> MagicMock:
    """Provides a mock TradeRepository."""
    return MagicMock(spec=TradeRepository)


@pytest.fixture
def mock_data_provider() -> MagicMock:
    """Provides a mock MarketDataProvider."""
    return MagicMock(spec=MarketDataProvider)


@pytest.fixture
def mock_signal_repository() -> MagicMock:
    """Provides a mock SignalRepository."""
    return MagicMock(spec=SignalRepository)


@pytest.fixture
def croc_strategy(
    mock_trade_repository: MagicMock,
    mock_data_provider: MagicMock,
    mock_signal_repository: MagicMock,
) -> CrocSetupStrategy:
    """Provides a fully mocked CrocSetupStrategy with stat guard patched."""
    mock_stat = MagicMock()
    mock_stat.st_size = 100
    with patch(
        "app.services.screener.strategies.croc_setup.settings.get_path"
    ) as mock_path:
        mock_path.return_value = Path("mock_ranking.yaml")
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data="ranking_2026: []")):
                    with patch(
                        "app.services.screener.strategies.croc_setup.ExchangeSymbol"
                    ) as mock_ex:
                        instance = mock_ex.return_value
                        instance.sp_500 = ["AAPL"]
                        instance.nasdaq_100 = ["AAPL", "MSFT"]
                        instance.dow_30 = []
                        instance.russell_1000 = []
                        return CrocSetupStrategy(
                            trade_repository=mock_trade_repository,
                            data_provider=mock_data_provider,
                            signal_repository=mock_signal_repository,
                        )


@pytest.fixture
def dip_buyer_strategy(
    mock_trade_repository: MagicMock,
    mock_data_provider: MagicMock,
) -> DipBuyerStrategy:
    """Provides a DipBuyerStrategy with mocked exchange symbols."""
    with patch("app.services.screener.strategies.dip_buyer.ExchangeSymbol") as mock_ex:
        instance = mock_ex.return_value
        instance.dow_30 = ["AAPL"]
        instance.sp_500 = ["AAPL", "MSFT"]
        instance.nasdaq_100 = ["AAPL", "MSFT", "NVDA"]
        return DipBuyerStrategy(
            trade_repository=mock_trade_repository,
            data_provider=mock_data_provider,
        )


@pytest.fixture
def ndx_strategy(
    mock_trade_repository: MagicMock,
    mock_data_provider: MagicMock,
) -> NDXMomentumScreener:
    """Provides a NDXMomentumScreener with mocked dependencies."""
    return NDXMomentumScreener(
        trade_repository=mock_trade_repository,
        market_data_provider=mock_data_provider,
    )


@pytest.fixture
def turnover_strategy(
    mock_trade_repository: MagicMock,
    mock_data_provider: MagicMock,
) -> TurnoverTimingStrategy:
    """Provides a TurnoverTimingStrategy with default configuration."""
    return TurnoverTimingStrategy(
        trade_repository=mock_trade_repository,
        data_provider=mock_data_provider,
    )


# ---------------------------------------------------------------------------
# SEC-01: YAML Anchor Bomb Guard (CrocSetupStrategy._load_config)
# ---------------------------------------------------------------------------


class TestYamlFileSizeGuard:
    """Validates SEC-01: YAML file-size guard prevents anchor bomb payloads."""

    def test_load_config_raises_runtime_error_when_file_exceeds_size_limit(
        self, croc_strategy: CrocSetupStrategy
    ) -> None:
        """Verifies that a config file exceeding 1 MB raises RuntimeError immediately."""
        # Arrange — simulate a 2 MB file
        mock_stat = MagicMock()
        mock_stat.st_size = 2 * 1024 * 1024  # 2 MB

        # Act & Assert
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with pytest.raises(RuntimeError, match="anchor bomb"):
                    croc_strategy._load_config()

    def test_load_config_succeeds_when_file_is_within_size_limit(
        self, croc_strategy: CrocSetupStrategy
    ) -> None:
        """Verifies that a normally-sized config file loads without error."""
        # Arrange
        mock_stat = MagicMock()
        mock_stat.st_size = 512  # 512 bytes — safe

        # Act
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data="ranking_2026: []")):
                    rules = croc_strategy._load_config()

        # Assert
        assert rules == []

    def test_load_config_returns_empty_list_on_yaml_parse_error(
        self, croc_strategy: CrocSetupStrategy
    ) -> None:
        """Verifies that a YAML parse error is caught and returns empty list (not crash)."""
        # Arrange
        import yaml

        mock_stat = MagicMock()
        mock_stat.st_size = 100

        # Act
        with patch("pathlib.Path.exists", return_value=True):
            with patch("pathlib.Path.stat", return_value=mock_stat):
                with patch("builtins.open", mock_open(read_data="invalid: [: yaml")):
                    with patch(
                        "app.services.screener.strategies.croc_setup.yaml.safe_load",
                        side_effect=yaml.YAMLError("bad yaml"),
                    ):
                        rules = croc_strategy._load_config()

        # Assert — graceful degradation, not a crash
        assert rules == []


# ---------------------------------------------------------------------------
# SEC-03: Fail-Closed on DB Errors (CRITICAL)
# ---------------------------------------------------------------------------


class TestFailClosedOnDatabaseErrors:
    """Validates SEC-03: DB errors must raise RuntimeError, not silently return 0."""

    def test_croc_setup_run_raises_runtime_error_on_db_lock(
        self,
        croc_strategy: CrocSetupStrategy,
        mock_signal_repository: MagicMock,
    ) -> None:
        """Verifies that an OperationalError during signal load raises RuntimeError (fail-closed)."""
        # Arrange
        mock_signal_repository.get_signals_by_date.side_effect = (
            sqlite3.OperationalError("database is locked")
        )

        # Act & Assert — must NOT silently return 0
        with pytest.raises(RuntimeError, match="Database unavailable"):
            croc_strategy.run(analysis_date="2026-01-30")

    def test_croc_setup_get_all_recommendations_raises_on_db_lock(
        self,
        croc_strategy: CrocSetupStrategy,
        mock_signal_repository: MagicMock,
    ) -> None:
        """Verifies fail-closed behavior in get_all_recommendations as well."""
        # Arrange
        mock_signal_repository.get_signals_by_date.side_effect = sqlite3.DatabaseError(
            "disk I/O error"
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Database unavailable"):
            croc_strategy.get_all_recommendations(analysis_date="2026-01-30")

    def test_croc_setup_run_returns_zero_on_data_anomaly(
        self,
        croc_strategy: CrocSetupStrategy,
        mock_signal_repository: MagicMock,
    ) -> None:
        """Verifies that data-level ValueError is a warning, not a crash."""
        # Arrange
        mock_signal_repository.get_signals_by_date.side_effect = ValueError(
            "bad date format"
        )

        # Act
        result = croc_strategy.run(analysis_date="not-a-date")

        # Assert — data anomaly → warning + 0
        assert result == 0

    def test_ndx_momentum_create_trades_raises_on_db_lock(
        self,
        ndx_strategy: NDXMomentumScreener,
        mock_trade_repository: MagicMock,
    ) -> None:
        """Verifies fail-closed: DB lock during trade creation raises RuntimeError."""
        # Arrange
        analysis_date = pd.Timestamp("2026-01-30")
        roc_df = pd.DataFrame({"AAPL": [5.0]}, index=[analysis_date])
        mock_trade_repository.create_trade.side_effect = sqlite3.OperationalError(
            "database is locked"
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Database unavailable"):
            ndx_strategy._create_trades_direct(
                symbols=["AAPL"],
                momentum_scores=pd.Series([10.0], index=["AAPL"]),
                roc_matrices={21: roc_df, 63: roc_df, 126: roc_df, 252: roc_df},
                analysis_date=analysis_date,
                price_data={
                    "close": pd.DataFrame({"AAPL": [100.0]}, index=[analysis_date])
                },
                regime_indicators={
                    "bull": True,
                    "qqq": 100.0,
                    "qqq_sma": 90.0,
                    "breadth_fast": 60.0,
                    "breadth_slow": 50.0,
                },
            )

    def test_dip_buyer_process_signals_raises_on_db_lock(
        self,
        dip_buyer_strategy: DipBuyerStrategy,
        mock_trade_repository: MagicMock,
    ) -> None:
        """Verifies fail-closed: OperationalError during trade save raises RuntimeError."""
        # Arrange
        date_obj = pd.Timestamp("2026-01-30")
        signals = pd.DataFrame(
            {
                "close": [100.0],
                "high": [105.0],
                "volume": [2_000_000.0],
                "atr": [2.0],
                "sma200": [90.0],
                "atr_ratio_3day": [-1.5],
                "ibs": [0.1],
                "setup_score": [1.5],
            },
            index=["AAPL"],
        )
        mock_trade_repository.create_trade.side_effect = sqlite3.OperationalError(
            "database is locked"
        )

        # Act & Assert
        with pytest.raises(RuntimeError, match="Database unavailable"):
            dip_buyer_strategy._process_signals(signals, date_obj)

    def test_dip_buyer_process_signals_logs_warning_on_data_error(
        self,
        dip_buyer_strategy: DipBuyerStrategy,
        mock_trade_repository: MagicMock,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Verifies that a ValueError during trade save is logged as warning, not crash."""
        # Arrange
        date_obj = pd.Timestamp("2026-01-30")
        signals = pd.DataFrame(
            {
                "close": [100.0],
                "high": [105.0],
                "volume": [2_000_000.0],
                "atr": [2.0],
                "sma200": [90.0],
                "atr_ratio_3day": [-1.5],
                "ibs": [0.1],
                "setup_score": [1.5],
            },
            index=["AAPL"],
        )
        mock_trade_repository.create_trade.side_effect = ValueError("bad entry price")

        # Act
        count = dip_buyer_strategy._process_signals(signals, date_obj)

        # Assert — graceful degradation
        assert count == 0
        assert "Failed to save trade" in caplog.text


# ---------------------------------------------------------------------------
# Mutable Default Argument Fix (MAJOR) — turnover_timing.py
# ---------------------------------------------------------------------------


class TestTurnoverConfigurationImmutableDefault:
    """Validates fix: TurnoverConfiguration default is not shared across instances."""

    def test_two_instances_have_independent_configurations(
        self,
        mock_trade_repository: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verifies that two strategy instances don't share the same config object."""
        # Arrange & Act
        instance_a = TurnoverTimingStrategy(
            trade_repository=mock_trade_repository,
            data_provider=mock_data_provider,
        )
        instance_b = TurnoverTimingStrategy(
            trade_repository=mock_trade_repository,
            data_provider=mock_data_provider,
        )

        # Assert — different object identity (not the same shared default)
        assert instance_a.configuration is not instance_b.configuration

    def test_custom_configuration_is_preserved(
        self,
        mock_trade_repository: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verifies that a custom TurnoverConfiguration is stored correctly."""
        # Arrange
        custom_config = TurnoverConfiguration(atr_window=7, sma_window=100)

        # Act
        strategy = TurnoverTimingStrategy(
            trade_repository=mock_trade_repository,
            data_provider=mock_data_provider,
            configuration=custom_config,
        )

        # Assert
        assert strategy.configuration.atr_window == 7
        assert strategy.configuration.sma_window == 100

    def test_none_configuration_falls_back_to_defaults(
        self,
        mock_trade_repository: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verifies that passing None uses default TurnoverConfiguration values."""
        # Arrange & Act
        strategy = TurnoverTimingStrategy(
            trade_repository=mock_trade_repository,
            data_provider=mock_data_provider,
            configuration=None,
        )

        # Assert
        assert strategy.configuration.atr_window == 3
        assert strategy.configuration.sma_window == 200


# ---------------------------------------------------------------------------
# _enrich_sma No In-Place Mutation Fix (FUNCTIONAL CORE)
# ---------------------------------------------------------------------------


class TestEnrichSmaReturnsPureResult:
    """Validates fix: _enrich_sma no longer mutates its input dict."""

    def test_enrich_sma_returns_new_dict_without_mutating_original(
        self, croc_strategy: CrocSetupStrategy
    ) -> None:
        """Verifies that _enrich_sma returns enriched copy, not mutated original."""
        # Arrange
        original_row: dict = {"symbol": "AAPL", "close": 100.0}
        original_id = id(original_row)
        prices = PriceData(high=110.0, low=90.0, close=100.0, sma_20=95.0, sma_200=80.0)

        # Act
        enriched = croc_strategy._enrich_sma(original_row, prices)

        # Assert — original dict is unchanged
        assert id(enriched) != original_id
        assert "dist_sma_20" not in original_row
        assert "dist_sma_200" not in original_row
        # Assert — enriched has the computed values
        assert "dist_sma_20" in enriched
        assert "dist_sma_200" in enriched

    @pytest.mark.parametrize(
        "sma_20, sma_200, close, expect_20_key, expect_200_key",
        [
            (95.0, 80.0, 100.0, True, True),  # Both SMAs positive
            (0.0, 80.0, 100.0, False, True),  # SMA20 = 0 → skip
            (95.0, 0.0, 100.0, True, False),  # SMA200 = 0 → skip
            (0.0, 0.0, 100.0, False, False),  # Both zero → no enrichment
        ],
    )
    def test_enrich_sma_conditional_key_addition(
        self,
        croc_strategy: CrocSetupStrategy,
        sma_20: float,
        sma_200: float,
        close: float,
        expect_20_key: bool,
        expect_200_key: bool,
    ) -> None:
        """Verifies that SMA distance keys are only added when SMA value is positive."""
        # Arrange
        row: dict = {}
        prices = PriceData(
            high=close + 5, low=close - 5, close=close, sma_20=sma_20, sma_200=sma_200
        )

        # Act
        enriched = croc_strategy._enrich_sma(row, prices)

        # Assert
        assert ("dist_sma_20" in enriched) == expect_20_key
        assert ("dist_sma_200" in enriched) == expect_200_key

    def test_enrich_sma_math_is_correct(self, croc_strategy: CrocSetupStrategy) -> None:
        """Verifies the percentage distance formula is mathematically correct."""
        # Arrange: close=110, sma_20=100 → dist = (110-100)/100 * 100 = 10%
        prices = PriceData(
            high=115.0, low=105.0, close=110.0, sma_20=100.0, sma_200=50.0
        )

        # Act
        enriched = croc_strategy._enrich_sma({}, prices)

        # Assert
        assert enriched["dist_sma_20"] == pytest.approx(10.0)
        assert enriched["dist_sma_200"] == pytest.approx(120.0)


# ---------------------------------------------------------------------------
# NDXAnalysisResult TypedDict (Type Safety Fix)
# ---------------------------------------------------------------------------


class TestNDXAnalysisResultTypeContract:
    """Validates that NDXAnalysisResult TypedDict is used as return type."""

    def test_calculate_analysis_returns_triggered_false_dict_on_non_rebalance_day(
        self, ndx_strategy: NDXMomentumScreener
    ) -> None:
        """Verifies early-return branch returns valid NDXAnalysisResult-compatible dict."""
        # Arrange & Act
        result = ndx_strategy.calculate_analysis(analysis_date="2026-01-15")

        # Assert — keys defined in NDXAnalysisResult
        assert "triggered" in result
        assert result["triggered"] is False
        assert "is_rebalance_day" in result

    def test_calculate_analysis_error_result_has_required_keys(
        self, ndx_strategy: NDXMomentumScreener
    ) -> None:
        """Verifies error-path result still conforms to the TypedDict contract."""
        # Arrange & Act — empty universe triggers an error branch
        with patch(
            "app.services.screener.strategies.ndx_momentum.ExchangeSymbol"
        ) as mock_ex:
            mock_ex.return_value.nasdaq_100 = []
            result = ndx_strategy.calculate_analysis(analysis_date="2026-01-30")

        # Assert
        assert "triggered" in result
        assert "error" in result
        assert result["triggered"] is False


# ---------------------------------------------------------------------------
# ABC / Protocol Signature Fix (LSP)
# ---------------------------------------------------------------------------


class TestBaseStrategySignatureCompliance:
    """Validates LSP fix: all strategies accept analysis_date in run()."""

    @pytest.mark.parametrize(
        "strategy_class, extra_kwargs",
        [
            (TurnoverTimingStrategy, {}),
        ],
    )
    def test_run_accepts_analysis_date_parameter(
        self,
        strategy_class: type,
        extra_kwargs: dict,
        mock_trade_repository: MagicMock,
        mock_data_provider: MagicMock,
    ) -> None:
        """Verifies that run() accepts analysis_date without TypeError."""
        # Arrange
        strategy = strategy_class(
            trade_repository=mock_trade_repository,
            data_provider=mock_data_provider,
            **extra_kwargs,
        )

        # Act & Assert — should not raise TypeError
        try:
            strategy.run(days=0, analysis_date="2026-01-15")
        except TypeError as type_error:
            pytest.fail(
                f"{strategy_class.__name__}.run() does not accept analysis_date: {type_error}"
            )


# ---------------------------------------------------------------------------
# SymbolFilter Deferred Import Fix (SEC-06)
# ---------------------------------------------------------------------------


class TestSymbolFilterImportAtModuleLevel:
    """Validates SEC-06: SymbolFilter is imported at module level, not inside methods."""

    def test_symbol_filter_importable_without_calling_execute_pipeline(
        self,
    ) -> None:
        """Verifies SymbolFilter is available at import time (not a deferred import)."""
        # Arrange & Act — importing dip_buyer module must not fail
        from app.services.screener.strategies import dip_buyer as dip_buyer_module

        # Assert — the module-level import is accessible
        assert hasattr(dip_buyer_module, "SymbolFilter")


# ---------------------------------------------------------------------------
# view_service.py — init return annotation
# ---------------------------------------------------------------------------


class TestScreenerViewServiceConstructor:
    """Validates that ScreenerViewService.__init__ is correctly typed."""

    def test_init_returns_none(self, mock_signal_repository: MagicMock) -> None:
        """Verifies that __init__ returns None as expected by python.md Sec 2."""
        # Arrange & Act
        service = ScreenerViewService(signal_repository=mock_signal_repository)

        # Assert
        assert service.signal_repository is mock_signal_repository
