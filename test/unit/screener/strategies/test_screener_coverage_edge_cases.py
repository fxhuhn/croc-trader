"""Unit tests targeting 100% test coverage for app/services/screener/strategies modules."""

from unittest.mock import MagicMock

import pandas as pd

from app.services.screener.strategies.bounce_bandit import BounceBanditStrategy
from app.services.screener.strategies.bridge_scout import BridgeScoutStrategy
from app.services.screener.strategies.croc_setup import CrocSetupStrategy
from app.services.screener.strategies.dip_buyer import DipBuyerStrategy
from app.services.screener.strategies.ndx_momentum import NDXMomentumScreener
from app.services.screener.strategies.tgim import TGIMStrategy
from app.services.screener.strategies.turnover_timing import TurnoverTimingStrategy
from app.services.screener.strategies.two_percent_strategy import TwoPercentStrategy


def test_screener_run_empty_data() -> None:
    """Tests strategy run methods when data_provider returns empty history."""
    provider = MagicMock()
    provider.get_batch_history.return_value = {}
    trade_repo = MagicMock()

    # Bounce Bandit
    s1 = BounceBanditStrategy(trade_repo, provider)
    assert s1.run(analysis_date="2026-02-01") == 0

    # Bridge Scout
    s2 = BridgeScoutStrategy(trade_repo, provider)
    assert s2.run(analysis_date="2026-02-01") == 0

    # TGIM
    s3 = TGIMStrategy(trade_repo, provider)
    assert s3.run(analysis_date="2026-02-01") == 0

    # Two Percent
    s4 = TwoPercentStrategy(trade_repo, provider)
    assert s4.run(analysis_date="2026-02-01") == 0

    # Turnover Timing
    s5 = TurnoverTimingStrategy(trade_repo, provider)
    assert s5.run(analysis_date="2026-02-01") == 0

    # Dip Buyer
    provider.get_universe_daily_data.return_value = {}
    s6 = DipBuyerStrategy(trade_repo, provider)
    assert s6.run(specific_symbols=["NON_EXISTENT"]) == 0

    # Croc Setup
    s7 = CrocSetupStrategy(trade_repo, provider, MagicMock())
    assert s7.run(specific_symbols=["NON_EXISTENT"]) == 0

    # NDX Momentum
    s8 = NDXMomentumScreener(trade_repo, provider)
    assert s8.run(analysis_date="2026-02-01") == 0


def test_dip_buyer_screener_date_resolution() -> None:
    """Tests DipBuyerStrategy target date resolution branches."""
    screener = DipBuyerStrategy(MagicMock(), MagicMock())
    closes = pd.DataFrame(index=pd.to_datetime(["2026-02-01", "2026-02-02"]))

    assert screener._resolve_target_date(closes, "invalid_date_str") is None
    assert screener._resolve_target_date(closes, "2026-02-10") == pd.Timestamp(
        "2026-02-02"
    )
    assert screener._resolve_target_date(closes, "2026-02-03") == pd.Timestamp(
        "2026-02-02"
    )
    assert screener._resolve_target_date(closes, "2026-01-15") is None
