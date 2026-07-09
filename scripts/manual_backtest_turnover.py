"""Manual Backtest Runner for the TurnoverTiming Strategy.

Simulates the EOD screening loop and subsequent execution (checking entries and exits)
for TurnoverTiming signals over a historical range of dates. Directly reads market data from
the stocks database and writes/manages trade results in the active signals database.

Usage:
    python script/manual_backtest_turnover.py --start YYYY-MM-DD [--end YYYY-MM-DD] [--capital 2000.0]

Side Effects:
    Modifies trades and trade_logs in the active signals database (data/signals.db).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Setup Paths
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.database.repositories.market_data_provider import (
    MarketDataProvider,  # noqa: E402
)
from app.database.repositories.trade import TradeRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402

# Strategy
from app.services.screener.strategies.turnover_timing import (  # noqa: E402
    TurnoverConfig,
    TurnoverTimingStrategy,
)
from app.services.trade_manager.strategies.turnover_timing import (  # noqa: E402
    TurnoverTimingStrategy as TradeManagerTurnover,
)
from app.types import TradeStatus  # noqa: E402

# Logging (Nur wichtige Infos)
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("TurnoverBacktest")
logger.setLevel(logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2025-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--capital", type=float, default=2000.0)
    args = parser.parse_args()

    # DB Setup
    stocks_session = DatabaseSession(str(settings.get_path("stocks")))
    trade_session = DatabaseSession(str(settings.get_path("signals")))
    data_provider = MarketDataProvider(stocks_session)
    trade_repository = TradeRepository(trade_session)
    data_provider.clear_cache()

    # Strategies
    config = TurnoverConfig()
    screener = TurnoverTimingStrategy(trade_repository, data_provider, config=config)
    manager = TradeManagerTurnover()

    # Timeframe
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    logger.info(f"🚀 Turnover Timing Backtest: {start_date.date()} - {end_date.date()}")

    stats = {"signals": 0, "filled": 0, "closed": 0}

    for current_ts in tqdm(date_range, desc="Simulation"):
        current_date_str = current_ts.strftime("%Y-%m-%d")

        # 1. Screening (Freitags)
        if current_ts.weekday() == 4:
            hits = screener.run(analysis_date=current_date_str)
            stats["signals"] += hits

        # 2. Trading
        all_trades = trade_repository.get_by_status(
            [
                TradeStatus.CREATED,
                TradeStatus.ACTIVE,
            ]
        )
        turnover_trades = [t for t in all_trades if "TurnoverTiming" in t["strategy"]]

        if not turnover_trades:
            continue

        symbols = list({t["symbol"] for t in turnover_trades})
        market_data = data_provider.get_batch_history(
            symbols, days=10, end_date=current_date_str
        )

        for trade in turnover_trades:
            sym = trade["symbol"]
            if sym not in market_data:
                continue

            df_hist = market_data[sym]
            df_day = df_hist[df_hist["date"] <= current_ts].copy()
            if df_day.empty:
                continue

            candle = df_day.iloc[-1]
            if pd.Timestamp(candle["date"]).normalize() != current_ts.normalize():
                continue

            res = None
            if trade["status"] == TradeStatus.CREATED.value:
                res = manager.check_entry(trade, candle, df_day, trade_repository)
                if res and "FILLED" in res:
                    stats["filled"] += 1

            elif trade["status"] == TradeStatus.ACTIVE.value:
                res = manager.manage_active_trade(trade, df_day, trade_repository)
                if res and "EXIT" in res:
                    stats["closed"] += 1

    # --- CLEAN REPORTING ---
    closed = [
        t
        for t in trade_repository.get_by_status(TradeStatus.CLOSED)
        if "TurnoverTiming" in t["strategy"]
    ]

    total_pnl = sum(float(t.get("realized_pnl", 0)) for t in closed)
    win_count = sum(1 for t in closed if float(t.get("realized_pnl", 0)) > 0)
    win_rate = (win_count / len(closed) * 100) if closed else 0.0

    print("\n" + "=" * 60)
    print("🏁 TURNOVER TIMING ERGEBNIS")
    print("=" * 60)
    print(f"   Signale generiert:  {stats['signals']}")
    print(
        f"   Trades ausgeführt:  {len(closed) + len([t for t in trade_repository.get_by_status(TradeStatus.ACTIVE) if 'TurnoverTiming' in t['strategy']])} (davon {stats['filled']} neu)"
    )
    print(f"   Trades geschlossen: {len(closed)}")
    print("-" * 60)
    print(f"   💰 GESAMT PnL:      ${total_pnl:,.2f}")
    print(f"   📈 Win Rate:        {win_rate:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
