"""Manual Croc Screener Strategy Tester.

Runs the CrocSetup screener strategy over a historical date range to verify that
trade signals are generated correctly and printed for development diagnostics.

Usage:
    python script/test_croc_screener.py --start YYYY-MM-DD [--end YYYY-MM-DD]

Side Effects:
    Creates/saves CREATED trades in the signals database (data/signals.db).
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Projekt-Root zum Pfad hinzufügen
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
from app.types import TradeStatus  # noqa: E402

# Logging Setup
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("CrocTest")


def main():
    parser = argparse.ArgumentParser(
        description="Testet den Croc Screener über einen Zeitraum"
    )
    parser.add_argument(
        "--start", type=str, default="2026-01-01", help="Startdatum YYYY-MM-DD"
    )
    parser.add_argument(
        "--end", type=str, default=None, help="Enddatum YYYY-MM-DD (Default: Heute)"
    )
    args = parser.parse_args()

    # 1. Setup & Init
    logger.info("🛠️ Initialisiere System...")

    # Datenbank-Pfade
    stocks_db_path = settings.get_path("stocks")
    signals_db_path = settings.get_path("signals")

    # Sessions
    stocks_session = DatabaseSession(str(stocks_db_path))
    signals_session = DatabaseSession(str(signals_db_path))

    # Repositories
    data_provider = MarketDataProvider(stocks_session)  # Wird vom BaseScreener erwartet
    trade_repository = TradeRepository(signals_session)
    signal_repository = SignalRepository(signals_session)

    # Schemas sicherstellen (erstellt Tabellen / führt Migrationen durch)
    logger.info("Prüfe Datenbank-Schema...")
    trade_repository.init_schema()
    signal_repository.init_schema()

    # Strategie initialisieren
    # WICHTIG: Wir übergeben hier das signal_repository für die neue Logik!
    screener = CrocSetupStrategy(
        trade_repository=trade_repository,
        data_provider=data_provider,
        signal_repository=signal_repository,
        telegram_bot=None,
    )

    # 2. Screening Loop
    start_date = pd.Timestamp(args.start)
    end_date = pd.Timestamp(args.end) if args.end else pd.Timestamp.now()
    date_range = pd.date_range(
        start=start_date, end=end_date, freq="B"
    )  # Business Days

    logger.info(f"🚀 Starte Croc-Scan von {start_date.date()} bis {end_date.date()}")

    total_hits = 0

    for current_ts in tqdm(date_range, desc="Processing Days"):
        current_date_str = current_ts.strftime("%Y-%m-%d")

        # Führe den Screener für diesen Tag aus
        # Der Screener liest aus der 'croc' Tabelle und schreibt in die 'trades' Tabelle
        hits = screener.run(analysis_date=current_date_str)
        total_hits += hits

    # 3. Reporting / Ergebnisse auslesen
    print("\n" + "=" * 100)
    print("🐊 CROC ERGEBNISSE (Status: CREATED)")
    print("=" * 100)

    # Wir holen alle Trades, die CREATED sind
    all_created = trade_repository.get_by_status(TradeStatus.CREATED)

    # Filtern: Nur Strategien, die zur Croc Logik gehören
    croc_trades = [
        t
        for t in all_created
        if "croc" in t.get("strategy", "").lower()
        or "split" in t.get("strategy", "").lower()
        or "hold" in t.get("strategy", "").lower()
    ]

    # Sortieren nach ID (neueste zuerst)
    croc_trades.sort(key=lambda x: x["id"], reverse=True)

    if croc_trades:
        print(
            f"{'ID':<4} {'DATE':<12} {'SYMBOL':<8} {'INDEX':<15} {'STRATEGY':<20} {'ENTRY':<10} {'STOP':<10} {'TARGET':<10}"
        )
        print("-" * 100)

        for t in croc_trades:
            # Robustes Parsen des Contexts für die Anzeige
            ctx_date = "N/A"
            idx_str = "-"
            raw_ctx = t.get("signal_context")

            if raw_ctx:
                try:
                    # Versuch 1: Ist es schon ein Dict?
                    if isinstance(raw_ctx, dict):
                        ctx = raw_ctx
                    else:
                        import json

                        # Versuch 2: JSON String parsen
                        ctx = json.loads(str(raw_ctx))
                        # Versuch 3: Double Encoded JSON String parsen
                        if isinstance(ctx, str):
                            ctx = json.loads(ctx)

                    if isinstance(ctx, dict):
                        ctx_date = ctx.get("date", "N/A")
                        idx_str = ctx.get("indices", "-")
                except (json.JSONDecodeError, TypeError, ValueError) as error:
                    logging.warning("Could not parse signal_context JSON: %s", error)

            # Formatiere Preise sicher als Float
            try:
                entry = float(t["entry_price"] or 0)
                sl = float(t["current_stop_loss"] or 0)
                tp = float(t["current_target"] or 0)
            except Exception:
                entry = sl = tp = 0.0

            print(
                f"{t['id']:<4} {ctx_date:<12} {t['symbol']:<8} {idx_str[:15]:<15} {t['strategy']:<20} {entry:<10.2f} {sl:<10.2f} {tp:<10.2f}"
            )

        print("-" * 100)
        print(f"Gesamt: {len(croc_trades)} Trades gefunden.")
    else:
        print("Keine Croc-Trades in der Datenbank gefunden.")

    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()
