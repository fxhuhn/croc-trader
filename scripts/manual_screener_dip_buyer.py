"""Manual Screener Runner for the DipBuyer Strategy.

Runs the DipBuyer screening algorithms over the market data for a given date (or today).
Discovers trading candidates that match the strategy filter requirements (e.g. minimum volume,
minimum price, price decline relative to moving averages) and stores the resulting proposals
in the signals database.

Usage:
    python script/manual_screener_dip_buyer.py [--date YYYY-MM-DD]

Side Effects:
    Writes new CREATED trades to the active signals database (data/signals.db)
    and sends Telegram notifications if active.
"""

import argparse  # NEU
import logging
import sys
from pathlib import Path

# 1. Pfad-Setup
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.database.repositories.market_data_provider import (
    MarketDataProvider,  # noqa: E402
)
from app.database.repositories.trade import TradeRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402
from app.services.screener.strategies.dip_buyer import (  # noqa: E402
    DipBuyerConfig,
    DipBuyerStrategy,
)
from app.services.telegram import TelegramBot  # noqa: E402
from app.tools.symbol_lists import ExchangeSymbol  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ScriptDipBuyer")


def main():
    # ---------------------------------------------------------
    # 0. CLI Argumente parsen
    # ---------------------------------------------------------
    parser = argparse.ArgumentParser(description="DipBuyer Screener")
    parser.add_argument(
        "--date",
        type=str,
        help="Optional: Datum für Backtest im Format YYYY-MM-DD",
        default=None,
    )
    args = parser.parse_args()

    mode_str = f"BACKTEST: {args.date}" if args.date else "LIVE (Heute)"
    logger.info(f"--- Starte DipBuyer Screener [{mode_str}] ---")

    try:
        # 1. Infrastruktur
        stocks_db_path = settings.get_path("stocks")
        croc_db_path = settings.get_path("signals")

        # Sessions
        stocks_session = DatabaseSession(str(stocks_db_path))
        data_provider = MarketDataProvider(stocks_session)

        trade_session = DatabaseSession(str(croc_db_path))
        trade_repository = TradeRepository(trade_session)
        trade_repository.init_schema()

        # Telegram
        telegram = None
        if settings.app.telegram.enabled:
            try:
                telegram = TelegramBot(
                    settings.app.telegram.token, settings.app.telegram.chat_id
                )
            except Exception as error:
                logging.warning("Telegram initialization failed: %s", error)

        # 2. Strategie
        config = DipBuyerConfig(
            MIN_PRICE=5.0,
            MIN_VOLUME=500_000,
        )

        strategy = DipBuyerStrategy(
            trade_repository=trade_repository,
            data_provider=data_provider,
            telegram_bot=telegram,
            config=config,
        )

        # 3. Ausführung
        # Vorher Cache leeren
        data_provider.clear_cache()

        # Index-Check (sicherstellen, dass Listen da sind)
        logger.info("Lade Index-Listen...")
        ExchangeSymbol()

        logger.info("Starte Analyse...")

        # Hier übergeben wir das Datum aus den Argumenten
        hits = strategy.run(days=0, analysis_date=args.date)

        logger.info("------------------------------------------------")
        logger.info(f"✅ Fertig. {hits} neue Trade-Kandidaten erstellt.")
        logger.info("------------------------------------------------")

    except Exception as e:
        logger.critical(f"Abbruch: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
