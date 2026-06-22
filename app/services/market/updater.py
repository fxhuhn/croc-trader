import logging
import time
from datetime import datetime, timedelta

from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.models import MarketPrice
from app.services.market.provider import YahooDataProvider, require_lock
from app.tools.symbol_lists import ExchangeSymbol

logger = logging.getLogger(__name__)

BATCH_SIZE = 500


class MarketDataUpdater:
    """
    Orchestrates the market data update process (ETL).
    Extract: Fetch from Yahoo (via Provider).
    Transform: Convert to MarketPrice domain models.
    Load: Save to Database (via Repository).
    """

    def __init__(
        self,
        session_factory: DatabaseSession,
        signals_session: DatabaseSession | None = None,
    ):
        self.session = session_factory
        self.repo = MarketRepository(self.session)

        # Use provided signals session or Fallback (graceful degradation if not provided, though ideally required)
        # If signals_session is None, TradeRepository might fail if used.
        # But we handle this via dependency injection now.
        if signals_session:
            self.trade_repository = TradeRepository(signals_session)
        else:
            # Fallback: Try to use the same session (legacy behavior, but mostly wrong for dual-db setup)
            # Or better: initializing it with None and checking before use?
            # For now, let's assume if it's not provided, we might not be able to fetch traded symbols.
            # But to keep 'self.trade_repository' valid type-wise:
            self.trade_repository = TradeRepository(self.session)

        self.provider = YahooDataProvider()

        # Ensure schema exists
        self.repo.init_schema()

    @require_lock
    def run_update(
        self, full_reload: bool = False, specific_symbols: list[str] | None = None
    ) -> None:
        """
        Main entry point for updating market data.
        """
        start_time = datetime.now()

        # 1. Determine Symbols
        symbols = self._get_symbols_to_process(specific_symbols)
        if not symbols:
            logger.warning("Keine Symbole zu verarbeiten.")
            return

        logger.info(f"Starte Update für {len(symbols)} Symbole (Full={full_reload})...")

        # 2. Determine Date Range
        # Full Reload: Since 2022
        # Incremental: Last 10 days (safety buffer)
        start_date = (
            "2021-01-01"
            if full_reload
            else (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
        )

        total_records = 0

        # 3. Batch Processing
        for i in range(0, len(symbols), BATCH_SIZE):
            batch_symbols = symbols[i : i + BATCH_SIZE]
            try:
                processed_count = self._process_batch(
                    batch_symbols, start_date, full_reload
                )
                total_records += processed_count

                # Rate Limiting / Politeness
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"Critical Error in Batch {i}: {e}", exc_info=True)

        duration = datetime.now() - start_time
        logger.info(f"Update fertig: {total_records} Records in {duration}.")

    def _get_symbols_to_process(self, specific: list[str] | None) -> list[str]:
        ignored = self.repo.get_ignored_symbols()

        if specific:
            candidates = set(specific)
        else:
            # Combine known DB symbols + Exchange Lists + Traded Symbols
            candidates = (
                set(ExchangeSymbol().all)
                .union(set(self.repo.get_all_known_symbols()))
                .union(set(self.trade_repository.get_all_traded_symbols()))
            )

        # Filter Ignored
        final_list = list(candidates - ignored)
        return final_list

    def _process_batch(
        self, batch: list[str], start_date: str, full_reload: bool
    ) -> int:
        """
        Fetches and saves a single batch. Returns count of saved records.
        """
        # Fetch Raw Data
        df_batch, raw_failures = self.provider.fetch_batch_raw(batch, start_date)

        failures = list(raw_failures)

        if df_batch.empty:
            # Handle Failures
            if failures and full_reload:
                for f in failures:
                    self.repo.ignore_symbol(f, "No Data (Full Reload)")
                    logger.warning(f"Ignoring symbol {f} (No Data)")
            return 0

        # Transform & Collect
        bulk_data: list[MarketPrice] = []

        for symbol in batch:
            if symbol in raw_failures:
                continue

            # Extract single symbol DataFrame
            df_sym = self.provider.extract_symbol_data(df_batch, symbol)

            if df_sym.empty:
                failures.append(symbol)
                continue

            # Standardize column names and clean data
            df_sym.columns = df_sym.columns.str.lower()
            df_sym = df_sym.dropna(subset=["close"])

            if df_sym.empty:
                failures.append(symbol)
                continue

            # Vectorized: reset index to make date a column
            df_sym = df_sym.reset_index().rename(columns={"index": "date"})

            # Batch-construct MarketPrice objects from dict records
            for row_dict in df_sym.to_dict("records"):
                try:
                    price_model = MarketPrice.from_yahoo(symbol, row_dict)
                    bulk_data.append(price_model)
                except ValueError as value_error:
                    logger.debug("Skipping row for %s: %s", symbol, value_error)
                    continue

        # Handle Failures
        if failures and full_reload:
            for f in failures:
                self.repo.ignore_symbol(f, "No Data (Full Reload)")
                logger.warning(f"Ignoring symbol {f} (No Data)")

        # Persist
        if bulk_data:
            self.repo.save_bulk_prices(bulk_data)

        return len(bulk_data)
