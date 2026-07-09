"""Manual Backtest Runner for the Croc Strategy.

Simulates the EOD screening loop and subsequent execution (checking entries and exits)
for Croc and Split signals over a historical range of dates using the HoldTarget strategy.
This script directly manipulates/records signals and trades in the active database.

Usage:
    python script/manual_backtest_croc.py --start YYYY-MM-DD [--end YYYY-MM-DD] [--risk 100.0]

Side Effects:
    Modifies trades and trade_logs in the active signals database (data/signals.db).
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.database.repositories.market_data_provider import (
    MarketDataProvider,  # noqa: E402
)
from app.database.repositories.signal import SignalRepository  # noqa: E402
from app.database.repositories.trade import TradeRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402
from app.services.screener.strategies.croc_setup import CrocSetupStrategy  # noqa: E402

# Wir importieren NUR HoldTarget, da dies die gewünschte Strategie ist
from app.services.trade_manager.strategies.hold_target import (  # noqa: E402
    HoldTargetStrategy,
)
from app.types import TradeStatus  # noqa: E402

# Logging konfigurieren
logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("CrocBacktest")
logger.setLevel(logging.INFO)

# Stummschalten anderer Module
logging.getLogger("app.services.screener.strategies.croc_setup").setLevel(
    logging.WARNING
)
logging.getLogger("app.database.repositories.trade").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=str, default="2026-01-01")
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--risk", type=float, default=100.0)
    args = parser.parse_args()

    # DB & Services
    stocks_session = DatabaseSession(str(settings.get_path("stocks")))
    signals_session = DatabaseSession(str(settings.get_path("signals")))
    data_provider = MarketDataProvider(stocks_session)
    trade_repository = TradeRepository(signals_session)
    signal_repository = SignalRepository(signals_session)

    screener = CrocSetupStrategy(
        trade_repository, data_provider, signal_repository, None
    )

    # ZWANG: Wir nutzen NUR die HoldTargetStrategy
    strategy_engine = HoldTargetStrategy()

    data_provider.clear_cache()
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    logger.info(
        f"🚀 Croc Backtest (HoldTarget Only): {start_date.date()} bis {end_date.date()}"
    )
    stats = {"signals": 0, "filled": 0, "closed": 0}

    # Debug-Helper
    debug_fill_count = 0

    for current_ts in tqdm(date_range, desc="Simulation"):
        current_date_str = current_ts.strftime("%Y-%m-%d")
        sim_date = current_ts.date()

        # 1. Screening
        hits = screener.run(days=0, analysis_date=current_date_str)
        stats["signals"] += hits

        # 2. Trading
        # Filter: Wir holen ALLE relevanten Trades
        keywords = ["croc", "split", "hold"]
        all_trades = trade_repository.get_by_status(
            [
                TradeStatus.CREATED,
                TradeStatus.ACTIVE,
            ]
        )
        active_trades = [
            t
            for t in all_trades
            if any(k in str(t.get("strategy", "")).lower() for k in keywords)
        ]

        if active_trades:
            syms = list({t["symbol"] for t in active_trades})
            market_data = data_provider.get_batch_history(
                syms, days=60, end_date=current_date_str
            )

            for trade in active_trades:
                sym = trade["symbol"]
                if sym not in market_data:
                    continue

                df_hist = market_data[sym]
                df_sim = df_hist[df_hist["date"].dt.date <= sim_date].copy()
                if df_sim.empty:
                    continue

                candle = df_sim.iloc[-1]
                if candle["date"].date() != sim_date:
                    continue

                # --- KORREKTUR: Datums-Check über SIGNAL CONTEXT ---
                # created_at ist unzuverlässig im Backtest (Systemzeit vs. Sim-Zeit).
                # Wir müssen das 'date' aus dem signal_context parsen.
                can_trade = True

                try:
                    ctx = json.loads(trade.get("signal_context") or "{}")
                    # Versuche verschiedene Keys für das Datum
                    sig_date_str = (
                        ctx.get("date")
                        or ctx.get("setup_date")
                        or ctx.get("analysis_date")
                    )

                    if sig_date_str:
                        sig_date = pd.Timestamp(sig_date_str).date()
                        # Trade darf erst am Tag NACH dem Signal ausgeführt werden
                        if sim_date <= sig_date:
                            can_trade = False
                except (json.JSONDecodeError, ValueError, TypeError) as error:
                    # Fallback: Ohne Datum im Context ist es riskant, aber wir lassen es zu
                    logging.warning(
                        "Failed to determine trade date from context: %s. Proceeding with fallback.",
                        error,
                    )

                if not can_trade:
                    continue

                # Strategie ausführen
                trader = strategy_engine

                if trade["status"] == TradeStatus.CREATED.value:
                    # WICHTIG: Risiko aus den Argumenten an den Trade übergeben!
                    trade["risk_amount"] = args.risk
                    res = trader.check_entry(trade, candle, df_sim, trade_repository)

                    if res and "FILLED" in str(res):
                        stats["filled"] += 1
                        if debug_fill_count < 5:
                            trigger = float(trade["entry_price"])
                            high = float(candle["high"])
                            logger.info(
                                f"✅ {sim_date} | {sym} FILLED. Trigger: {trigger:.2f} <= High: {high:.2f}"
                            )
                            debug_fill_count += 1

                elif trade["status"] == TradeStatus.ACTIVE.value:
                    res = trader.manage_active_trade(trade, df_sim, trade_repository)
                    if res and (
                        "EXIT" in str(res) or "STOP" in str(res) or "TARGET" in str(res)
                    ):
                        stats["closed"] += 1
                        logger.info(f"💰 {sim_date} | {sym} CLOSED: {res}")

    # Report
    closed = [
        t
        for t in trade_repository.get_by_status(TradeStatus.CLOSED)
        if any(k in str(t.get("strategy", "")).lower() for k in keywords)
    ]
    active = [
        t
        for t in trade_repository.get_by_status(TradeStatus.ACTIVE)
        if any(k in str(t.get("strategy", "")).lower() for k in keywords)
    ]

    total_pnl = sum(float(t.get("realized_pnl", 0) or 0) for t in closed)
    win_cnt = sum(1 for t in closed if float(t.get("realized_pnl", 0) or 0) > 0)
    wr = (win_cnt / len(closed) * 100) if closed else 0.0

    print("\n" + "=" * 60)
    print(f"🏁 CROC (HOLD TARGET) ERGEBNIS ({start_date.date()} - {end_date.date()})")
    print("=" * 60)
    print(f"   Signale generiert: {stats['signals']}")
    print(f"   Trades ausgeführt: {stats['filled']}")
    print(f"   Trades offen:      {len(active)}")
    print(f"   Trades geschlossen:{len(closed)}")
    print("-" * 60)
    print(f"   💰 GESAMT PnL:     ${total_pnl:,.2f}")
    print(f"   📈 Win Rate:       {wr:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
