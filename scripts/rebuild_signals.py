"""Signals Database Rebuilder and Backfiller.

Wipes all tables in signals.db and performs a historical daily simulation starting
on '2025-12-29' up to today. Incorporates a 'TimeTravelMarketRepository' that forces the TradeManager
to only see historical market prices prior to/during each simulated day (avoiding lookahead bias).
Runs screeners and daily order processing day-by-day.

Usage:
    python script/rebuild_signals.py

Side Effects:
    DESTRUCTIVE. Truncates all existing records in the trades and trade_logs tables of data/signals.db
    and regenerates them step-by-step.
"""

import logging
import sqlite3
import sys
from datetime import timedelta
from pathlib import Path
from typing import override

import pandas as pd

# Setup Django-like path if run as script
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from app.database.repositories.market import MarketRepository  # noqa: E402
from app.database.repositories.market_data_provider import (
    MarketDataProvider,  # noqa: E402
)
from app.database.repositories.signal import SignalRepository  # noqa: E402
from app.database.repositories.trade import TradeRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402
from app.services.screener.engine import ScreenerEngine  # noqa: E402

# Strategies
from app.services.screener.strategies.croc_setup import CrocSetupStrategy  # noqa: E402
from app.services.screener.strategies.dip_buyer import DipBuyerStrategy  # noqa: E402
from app.services.screener.strategies.ndx_momentum import (  # noqa: E402
    NDXMomentumScreener,
)
from app.services.screener.strategies.turnover_timing import (  # noqa: E402
    TurnoverTimingStrategy,
)
from app.services.screener.strategies.two_percent_strategy import (  # noqa: E402
    TwoPercentStrategy,
)
from app.services.trade_manager.manager import TradeManager  # noqa: E402

# Setup Logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("RebuildSignals")

# Constants
DB_PATH = project_root / "data" / "signals.db"
STOCKS_DB_PATH = project_root / "data" / "stocks.db"
START_DATE = "2025-12-29"


class TimeTravelMarketRepository(MarketRepository):
    """
    Wraps the MarketRepository to filter data by a simulated 'current' date.
    This prevents the TradeManager from seeing future data during backfill.
    """

    def __init__(self, session: DatabaseSession):
        super().__init__(session)
        self.simulated_date: pd.Timestamp | None = None

    def set_date(self, date_str: str):
        self.simulated_date = pd.Timestamp(date_str)

    @override
    def get_symbol_history_raw(self, symbol: str, start_date: str) -> pd.DataFrame:
        """
        Fetches history but cuts off anything after self.simulated_date.
        """
        # Call original
        df = super().get_symbol_history_raw(symbol, start_date)

        if df.empty or self.simulated_date is None:
            return df

        # Filter
        # Ensure df['date'] is datetime (it should be from super)
        df_filtered = df[df["date"] <= self.simulated_date].copy()

        return df_filtered


def clean_database():
    """Deletes all trades, logs, and signals to start fresh."""
    logger.warning(f"cleaning database {DB_PATH}...")
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM trade_logs")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='trades'")
        conn.execute("DELETE FROM sqlite_sequence WHERE name='trade_logs'")
        conn.commit()
    logger.info("Database cleaned.")


def main():
    logger.info(f"Starting Rebuild Process from {START_DATE}...")

    # 1. Clean Data
    clean_database()

    # 2. Init Sessions
    signals_session = DatabaseSession(str(DB_PATH))
    stocks_session = DatabaseSession(str(STOCKS_DB_PATH))

    # 3. Init Repos
    trade_repository = TradeRepository(signals_session)
    signal_repository = SignalRepository(signals_session)
    # MarketDataProvider for Screener (Efficiency: loading big chunks is fine, strategies must respect analysis_date)
    market_provider = MarketDataProvider(stocks_session)

    # 4. Init Services
    # Screener
    active_strategies = [
        DipBuyerStrategy(
            trade_repository=trade_repository,
            data_provider=market_provider,
            telegram_bot=None,
        ),
        TurnoverTimingStrategy(
            trade_repository=trade_repository,
            data_provider=market_provider,
            telegram_bot=None,
        ),
        CrocSetupStrategy(
            trade_repository=trade_repository,
            data_provider=market_provider,
            signal_repository=signal_repository,
            telegram_bot=None,
        ),
        TwoPercentStrategy(
            trade_repository=trade_repository,
            data_provider=market_provider,
            telegram_bot=None,
        ),
        NDXMomentumScreener(
            trade_repository=trade_repository,
            market_data_provider=market_provider,
            telegram_bot=None,
        ),
    ]

    screener_engine = ScreenerEngine(
        trade_repository=trade_repository,
        signal_repository=signal_repository,
        data_provider=market_provider,
        strategies=active_strategies,
        configuration=None,
        telegram_bot=None,
    )

    # TradeManager
    trade_manager = TradeManager(
        db_path=DB_PATH, stocks_db_path=STOCKS_DB_PATH, telegram_bot=None
    )

    # 5. Inject TimeTravel Repo into TradeManager
    # TradeManager creates its own repo in __init__, so we overwrite it.
    time_travel_repo = TimeTravelMarketRepository(stocks_session)
    trade_manager.market_repository = time_travel_repo
    logger.info("Injected TimeTravelMarketRepository into TradeManager.")

    # 6. Time Loop
    current_date = pd.Timestamp(START_DATE)
    end_date = pd.Timestamp.now()

    # Preload Market Data for Speed (Optional, but good for Screener)
    # market_provider.preload_all_data(days=600)

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")

        # Check if it's a weekday (Mon=0, Sun=6) - Optional: Screener strategies might handle it,
        # but skipping weekends saves time.
        # But wait, holidays?
        # Better: Check if there is market data for this date in DB to decide if we run.
        # However, Screener usually runs on 'today' even if no data to check if signals *generated* from yesterday data should be processed?
        # No, Screener signals are usually EOD.
        # TradeManager needs to manage trades every day? Or just trading days?
        # Safe bet: Run every weekday. Strategies and Repo handle missing data gracefully.

        if current_date.dayofweek >= 5:  # Sat/Sun
            current_date += timedelta(days=1)
            continue

        logger.info(f"--- Processing {date_str} ---")

        # A) Update Time Travel
        time_travel_repo.set_date(date_str)

        # B) Run TradeManager (Manages Active/Pending Trades based on 'current' view)
        # Needs to run BEFORE Screener?
        # Standard:
        # 1. Manage existing trades (Exit/Entries from previous signals)
        # 2. Find new signals (Screener)

        try:
            trade_manager.run_daily_process()
        except Exception:
            logger.exception(f"Error in TradeManager on {date_str}")

        # C) Run Screener
        # We iterate strategies manually to enforce analysis_date
        for strategy in screener_engine.active_strategies:
            try:
                # Run strategy for specific date
                strategy.run(days=0, analysis_date=date_str)
            except Exception:
                logger.exception(f"Error in Strategies {strategy.name} on {date_str}")

        current_date += timedelta(days=1)

    logger.info("Rebuild Complete.")


if __name__ == "__main__":
    main()
