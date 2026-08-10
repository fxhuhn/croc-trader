"""Extended unit tests for app/services/portfolio/allocator.py."""

from app.config import PortfolioConfig, PortfolioStrategyConfig
from app.services.portfolio.allocator import AllocationResult, PortfolioAllocator


def test_allocator_prefix_turnover_matching() -> None:
    """Tests strategy resolution using prefix matching for Turnover variants."""
    allocator = PortfolioAllocator()
    trade = {
        "symbol": "AAPL",
        "strategy": "turnover_timing_custom_variant_v1",
        "entry_price": 50.0,
    }
    res = allocator.allocate(trade)
    assert res.size == 50  # 2500 / 50
    assert "Turnover Budget" in res.reason


def test_allocator_context_decoding_and_dict() -> None:
    """Tests signal_context decoding error handling and raw dict context."""
    allocator = PortfolioAllocator()

    # Invalid JSON string in signal_context
    trade_bad_json = {
        "symbol": "AAPL",
        "strategy": "hold_target",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
        "signal_context": "{invalid_json: ",
    }
    res_bad = allocator.allocate(trade_bad_json)
    assert res_bad.size == 10  # Risk: 100 / (100 - 90) = 10 shares

    # Dict signal_context for short position
    trade_short = {
        "symbol": "AAPL",
        "strategy": "hold_target",
        "entry_price": 100.0,
        "current_stop_loss": 110.0,
        "signal_context": {"direction": "short"},
    }
    res_short = allocator.allocate(trade_short)
    assert res_short.size == 10  # Risk: 100 / (110 - 100) = 10 shares

    # Invalid SL for short position (SL <= entry)
    trade_short_bad_sl = {
        "symbol": "AAPL",
        "strategy": "hold_target",
        "entry_price": 100.0,
        "current_stop_loss": 95.0,
        "signal_context": {"direction": "short"},
    }
    res_short_bad = allocator.allocate(trade_short_bad_sl)
    assert res_short_bad.reason == "Invalid Stop Loss"


def test_allocator_croc_setup_and_zero_risk_amount() -> None:
    """Tests risk amount fallback when risk_amount <= 0 or croc_setup/split_target strategy_key."""
    cfg = PortfolioConfig(hold_target=PortfolioStrategyConfig(risk_amount=0.0))
    allocator = PortfolioAllocator(portfolio_config=cfg)

    trade_croc = {
        "symbol": "AAPL",
        "strategy": "croc_setup",
        "entry_price": 100.0,
        "current_stop_loss": 90.0,
    }
    res = allocator.allocate(trade_croc)
    assert res.risk_amount == 100.0  # Fallback to 100.0
    assert res.size == 10


def test_allocator_budget_exceeded_for_all_fixed_budget_strategies() -> None:
    """Tests price > budget branch (size < 1) across all fixed budget strategies."""
    cfg = PortfolioConfig(
        two_percent=PortfolioStrategyConfig(budget=10.0),
        ndx_momentum=PortfolioStrategyConfig(budget=10.0),
        tgim=PortfolioStrategyConfig(budget=10.0),
        bridge_scout=PortfolioStrategyConfig(budget=10.0),
        bounce_bandit=PortfolioStrategyConfig(budget=10.0),
    )
    allocator = PortfolioAllocator(portfolio_config=cfg)

    for strat_name in [
        "two_percent",
        "ndx_momentum",
        "tgim",
        "bridge_scout",
        "bounce_bandit",
    ]:
        trade = {
            "symbol": "HIGH_PRICE",
            "strategy": strat_name,
            "entry_price": 100.0,  # Price 100 > Budget 10
        }
        res = allocator.allocate(trade)
        assert res.size == 0
        assert res.reason == "Price > Budget"


def test_allocator_valid_allocations_for_all_strategies() -> None:
    """Tests successful allocation for TwoPercent, NDXMomentum, TGIM, BridgeScout, BounceBandit."""
    allocator = PortfolioAllocator()

    for strat_name in [
        "two_percent",
        "ndx_momentum",
        "tgim",
        "bridge_scout",
        "bounce_bandit",
    ]:
        trade = {
            "symbol": "AAPL",
            "strategy": strat_name,
            "entry_price": 100.0,
        }
        res = allocator.allocate(trade)
        assert res.size > 0
        assert isinstance(res, AllocationResult)
