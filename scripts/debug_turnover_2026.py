"""Diagnostics Viewer for TurnoverTiming Strategy signals.

Scans the signals database for any TurnoverTiming signals generated during the year 2026,
parses their JSON context, and prints details including setup date, close price, ATR,
and limit entry levels. Useful for verifying that screener executions created expected entries
in the database.

Usage:
    python script/debug_turnover_2026.py

Side Effects:
    None (performs read-only queries on signals and stocks databases).
"""

import json
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
from app.services.screener.strategies.turnover_timing import (  # noqa: E402
    TurnoverTimingStrategy,
)

# Logging stilllegen
logging.basicConfig(level=logging.CRITICAL)


def main():
    # 1. Setup
    stocks_session = DatabaseSession(str(settings.get_path("stocks")))
    trade_session = DatabaseSession(str(settings.get_path("signals")))

    data_provider = MarketDataProvider(stocks_session)
    trade_repository = TradeRepository(trade_session)
    data_provider.clear_cache()

    screener = TurnoverTimingStrategy(trade_repository, data_provider)

    print("\n" + "=" * 100)
    print(
        f"{'DATUM':<12} | {'SYMBOL':<6} | {'CLOSE':<8} | {'ATR(3)':<8} | {'ENTRY (Limit)':<14} | {'STRATEGIE'}"
    )
    print("=" * 100)

    # 2. Zeitraum definieren (2026)
    start_date = pd.Timestamp("2026-01-01")
    end_date = pd.Timestamp("2026-12-31")

    end_date = min(end_date, pd.Timestamp.now())

    date_range = pd.date_range(start=start_date, end=end_date, freq="B")

    signals_found = 0
    seen_signals = set()

    for current_ts in tqdm(date_range, desc="Scanne DB", unit="day"):
        # Wir prüfen nur Freitage (oder Backtest-Tage)
        if current_ts.weekday() == 4:
            current_date_str = current_ts.strftime("%Y-%m-%d")

            # 1. Trigger Screener (nur um sicherzugehen, aber eigentlich schauen wir in die DB)
            # Wir lassen den Screener laufen, ignorieren aber den Return-Wert 'hits',
            # weil er 0 ist, wenn die Trades schon da sind.
            days_delta = (pd.Timestamp.now().normalize() - current_ts.normalize()).days
            try:
                screener.run(days=days_delta)
            except Exception as error:
                # Fehler protokollieren, aber fortfahren, da wir die bestehenden Daten lesen wollen
                logging.debug(
                    "Screener execution failed for days=%d: %s", days_delta, error
                )

            # 2. DB Abfragen (ALLES lesen was da ist)
            # Wir suchen nach Signalen, die an diesem 'setup_date' erstellt wurden
            sql = """
                SELECT symbol, strategy, entry_price, signal_context
                FROM trades
                WHERE strategy LIKE 'TurnoverTiming%'
                AND status IN ('CREATED', 'ACTIVE', 'CLOSED')
                AND json_extract(signal_context, '$.setup_date') = ?
            """
            rows = trade_repository.fetch_all(sql, (current_date_str,))

            for r in rows:
                try:
                    ctx = json.loads(r["signal_context"])
                    symbol = r["symbol"]
                    strat = r["strategy"]
                    entry = r["entry_price"]

                    # Key für Duplikat-Filterung in der Anzeige
                    unique_key = f"{current_date_str}_{symbol}_{strat}"
                    if unique_key in seen_signals:
                        continue
                    seen_signals.add(unique_key)

                    setup_close = float(ctx.get("setup_close", 0))
                    setup_atr = float(ctx.get("setup_atr", 0))

                    print(
                        f"{current_date_str:<12} | {symbol:<6} | {setup_close:<8.2f} | {setup_atr:<8.2f} | {entry:<14.2f} | {strat}"
                    )
                    signals_found += 1

                except Exception as error:
                    logging.warning("Failed to process database row: %s", error)
                    continue

    print("=" * 100)
    print(f"GESAMT: {signals_found} Signale in der Datenbank gefunden.")
    print("=" * 100)


if __name__ == "__main__":
    main()
