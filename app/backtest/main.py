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
    """Run a single strategy backtest using strategy.universe."""
    run_cfg = BacktestRunConfig(
        strategy_name=strategy.name,
        start_date=strategy.config.start_date,
        initial_capital=100_000.0,
        max_positions=max_positions,
        out_dir=settings.backtest.report_path,
    )

    engine = BacktestEngine(
        strategy=strategy,
        universe=strategy.universe,
        config=run_cfg,
        market_repo=market_repo,
        backtest_repo=bt_repo,
    )
    engine.run()


def main() -> None:
    configure_logging(level=logging.INFO)

    universe = list(set(sp500_symbols() + nasdaq100_symbols() + dow30_symbols()))
    logger.info("Base universe: %d symbols", len(universe))

    market_repo = OHLCVRepository(str(settings.database.market_data_path))
    bt_repo = BacktestRepository(str(settings.database.backtest_path))

    # Strategy 1: Minervini Trend Template
    minervini_cfg = MinerviniConfig(start_date="2020-01-01", min_rs_rank=70.0)
    minervini = MinerviniStrategy(minervini_cfg)
    minervini.universe = universe  # Set universe explicitly
    run_strategy(minervini, 10, market_repo, bt_repo)

    # Strategy 2: NASDAQ Momentum (QQQ Regime)
    momentum_cfg = MomentumConfig(start_date="2022-01-01", top_n=5, regime_symbol="QQQ")
    momentum = NasdaqMomentumStrategy(momentum_cfg, universe)
    # momentum.universe already contains QQQ (added in __init__)
    logger.info(
        "QQQ Momentum universe: %d symbols (includes QQQ: %s)",
        len(momentum.universe),
        "QQQ" in momentum.universe,
    )
    run_strategy(momentum, 5, market_repo, bt_repo)

    # Strategy 3: NASDAQ Momentum (Breadth Regime)
    breadth_cfg = MomentumConfig(start_date="2022-01-01", top_n=5)
    breadth = BreadthMomentumStrategy(breadth_cfg, universe)
    run_strategy(breadth, 5, market_repo, bt_repo)


if __name__ == "__main__":
    configure_logging(level=logging.DEBUG)
    main()
