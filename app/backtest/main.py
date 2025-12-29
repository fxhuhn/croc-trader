from __future__ import annotations

"""Backtest execution entry point."""


import logging
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))
"""Backtest entry point (wires repos + strategy + engine)."""


from app.backtest.backtest_core import BacktestRepository
from app.backtest.domain import BacktestRunConfig
from app.backtest.engine import BacktestEngine
from app.backtest.logging_config import configure_logging
from app.backtest.strategies import (
    BreadthMomentumStrategy,
    MinerviniConfig,
    MinerviniStrategy,
    MomentumConfig,
    NasdaqMomentumStrategy,
)
from app.config import settings
from app.core.database import OHLCVRepository
from app.core.symbol_lists import dow30_symbols, nasdaq100_symbols, sp500_symbols

logger = logging.getLogger(__name__)


def run_strategy(
    strategy: MinerviniStrategy | NasdaqMomentumStrategy | BreadthMomentumStrategy,
    max_positions: int,
    market_repo: OHLCVRepository,
    bt_repo: BacktestRepository,
) -> None:
    """
    Run a single strategy backtest.

    For strategies with universe_for_date(), engine will use dynamic universe.
    For static strategies (Minervini), uses strategy.universe.
    """
    run_cfg = BacktestRunConfig(
        strategy_name=strategy.name,
        start_date=strategy.config.start_date,
        initial_capital=100_000.0,
        max_positions=max_positions,
        out_dir=settings.backtest.report_path,
    )

    # Get universe (for static strategies or as fallback)
    universe = getattr(strategy, "universe", [])

    engine = BacktestEngine(
        strategy=strategy,
        universe=universe,
        config=run_cfg,
        market_repo=market_repo,
        backtest_repo=bt_repo,
    )
    engine.run()


def main() -> None:
    configure_logging(level=logging.INFO)

    # Base universe for Minervini (static)
    base_universe = list(set(sp500_symbols() + nasdaq100_symbols() + dow30_symbols()))
    logger.info("Base universe: %d symbols", len(base_universe))

    market_repo = OHLCVRepository(str(settings.database.market_data_path))
    bt_repo = BacktestRepository(str(settings.database.backtest_path))

    # Strategy 1: Minervini Trend Template (static universe)
    minervini_cfg = MinerviniConfig(start_date="2020-01-01", min_rs_rank=70.0)
    minervini = MinerviniStrategy(minervini_cfg)
    minervini.universe = base_universe  # Set static universe
    logger.info("Running Minervini with static universe")
    run_strategy(minervini, 10, market_repo, bt_repo)

    # Strategy 2: NASDAQ Momentum (QQQ Regime, dynamic monthly NDX universe)
    momentum_cfg = MomentumConfig(
        start_date="2022-01-01",
        top_n=5,
        regime_symbol="QQQ",
    )
    qqq_momentum = NasdaqMomentumStrategy(momentum_cfg)
    logger.info("Running NASDAQ Momentum (QQQ) with dynamic monthly universe")
    run_strategy(qqq_momentum, 5, market_repo, bt_repo)

    # Strategy 3: NASDAQ Momentum (Breadth Regime, dynamic monthly NDX universe)
    breadth_momentum = BreadthMomentumStrategy(momentum_cfg)
    logger.info("Running NASDAQ Momentum (Breadth) with dynamic monthly universe")
    run_strategy(breadth_momentum, 5, market_repo, bt_repo)


if __name__ == "__main__":
    main()
