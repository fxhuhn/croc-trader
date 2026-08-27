import logging
import time
from datetime import datetime, timedelta

import pandas as pd

from app.database.repositories.market import MarketRepository
from app.database.repositories.trade import TradeRepository
from app.database.session import DatabaseSession
from app.models import MarketPrice
from app.services.market.provider import YahooDataProvider, require_lock
from app.services.market.tv_provider import TradingViewDataProvider
from app.tools.market_holidays import MarketHolidayChecker
from app.tools.symbol_lists import ExchangeSymbol
from app.tools.trading_calendar import get_last_completed_trading_day

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
        self.tv_provider = TradingViewDataProvider()

        # Ensure schema exists
        self.repo.init_schema()

    @require_lock
    def run_update(
        self,
        full_reload: bool = False,
        specific_symbols: list[str] | None = None,
        provider_mode: str = "auto",
        ignore_today: bool = False,
    ) -> None:
        """
        Main entry point for updating market data.

        :param full_reload: If True, fetches full historical bars (since 2021).
        :param specific_symbols: Optional subset list of symbols to process.
        :param provider_mode: Data provider strategy ('auto', 'tradingview', 'yahoo').
        :param ignore_today: If True, filters out bars matching today's date.
        """
        start_time = datetime.now()

        if full_reload and not specific_symbols:
            logger.info("Full reload requested: clearing ignored_symbols blacklist.")
            self.repo.clear_ignored_symbols()

        # 1. Determine Symbols
        symbols = self._get_symbols_to_process(specific_symbols)
        if not symbols:
            logger.warning("No symbols to process.")
            return

        logger.info(
            "Starting update for %d symbols (Full=%s, Provider=%s, IgnoreToday=%s)...",
            len(symbols),
            full_reload,
            provider_mode,
            ignore_today,
        )

        # 2. Determine Date Range
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
                    batch_symbols,
                    start_date,
                    full_reload,
                    provider_mode=provider_mode,
                    ignore_today=ignore_today,
                )
                total_records += processed_count

                # Rate Limiting / Politeness
                time.sleep(0.5)

            except Exception as e:
                logger.error("Critical Error in Batch %d: %s", i, e, exc_info=True)

        duration = datetime.now() - start_time
        logger.info("Update finished: %d records in %s.", total_records, duration)

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

    def _extract_symbol_market_prices(
        self,
        df_sym: pd.DataFrame,
        symbol: str,
        ignore_today: bool,
        today_str: str,
    ) -> tuple[list[MarketPrice], str]:
        """Extracts and cleans market prices for a single symbol from DataFrame."""
        df_sym.columns = df_sym.columns.str.lower()
        df_sym = df_sym.dropna(subset=["close"])
        if df_sym.empty:
            return [], ""

        df_sym = df_sym.reset_index().rename(columns={"index": "date"})
        symbol_prices: list[MarketPrice] = []
        symbol_max_date = ""

        for row_dict in df_sym.to_dict("records"):
            try:
                price_model = MarketPrice.from_yahoo(symbol, row_dict)
                if ignore_today and price_model.date == today_str:
                    logger.debug(
                        "Skipping current day record for %s due to ignore_today flag",
                        symbol,
                    )
                    continue
                symbol_prices.append(price_model)
                symbol_max_date = max(symbol_max_date, price_model.date)
            except ValueError as value_error:
                logger.debug("Skipping row for %s: %s", symbol, value_error)
                continue

        return symbol_prices, symbol_max_date

    def _handle_batch_failures(
        self,
        failures: list[str],
        provider_mode: str,
        full_reload: bool,
        ignore_today: bool,
    ) -> int:
        """Handles symbol failures according to the active provider mode."""
        if not failures:
            return 0
        if provider_mode == "auto":
            return self._fallback_to_tradingview(
                failures, full_reload, ignore_today=ignore_today
            )
        if provider_mode == "yahoo" and full_reload:
            for symbol in failures:
                self.repo.ignore_symbol(symbol, "No Data (Full Reload)")
        return 0

    def _process_batch(
        self,
        batch: list[str],
        start_date: str,
        full_reload: bool,
        provider_mode: str = "auto",
        ignore_today: bool = False,
    ) -> int:
        """Fetches and saves a single batch. Returns count of saved records."""
        today_str = datetime.now().strftime("%Y-%m-%d")

        if provider_mode == "tradingview":
            return self._fallback_to_tradingview(
                batch, full_reload, ignore_today=ignore_today
            )

        df_batch, raw_failures = self.provider.fetch_batch_raw(batch, start_date)
        failures = list(raw_failures)

        if df_batch.empty:
            return self._handle_batch_failures(
                batch if provider_mode != "yahoo" else failures,
                provider_mode,
                full_reload,
                ignore_today,
            )

        target_trading_day = get_last_completed_trading_day(
            datetime.now().date(), MarketHolidayChecker()
        ).strftime("%Y-%m-%d")

        bulk_data: list[MarketPrice] = []

        for symbol in batch:
            if symbol in raw_failures:
                continue

            df_sym = self.provider.extract_symbol_data(df_batch, symbol)
            if df_sym.empty:
                failures.append(symbol)
                continue

            symbol_prices, symbol_max_date = self._extract_symbol_market_prices(
                df_sym, symbol, ignore_today, today_str
            )
            if not symbol_prices:
                failures.append(symbol)
                continue

            bulk_data.extend(symbol_prices)

            if (
                provider_mode == "auto"
                and symbol_max_date
                and symbol_max_date < target_trading_day
                and symbol not in failures
            ):
                logger.debug(
                    "Yahoo recency gap for %s (latest=%s < target=%s), queuing for TradingView fallback.",
                    symbol,
                    symbol_max_date,
                    target_trading_day,
                )
                failures.append(symbol)

        if bulk_data:
            self.repo.save_bulk_prices(bulk_data)

        fallback_count = self._handle_batch_failures(
            failures, provider_mode, full_reload, ignore_today
        )
        return len(bulk_data) + fallback_count

    def _process_single_tradingview_symbol(
        self,
        symbol: str,
        n_bars: int,
        *,
        full_reload: bool,
        ignore_today: bool,
        today_str: str,
    ) -> int:
        """Fetches and saves historical data for a single symbol via TradingView."""
        records = self.tv_provider.fetch_symbol_history(symbol, number_of_bars=n_bars)
        if not records:
            logger.warning(
                "TradingView fallback yielded no records for symbol %s", symbol
            )
            if full_reload:
                self.repo.ignore_symbol(symbol, "No Data (Yahoo & TradingView)")
                logger.warning(
                    "Ignoring symbol %s (No Data from Yahoo or TradingView)", symbol
                )
            return 0

        tv_prices: list[MarketPrice] = []
        for row in records:
            try:
                price_model = MarketPrice.from_tradingview(symbol, row)
                if ignore_today and price_model.date == today_str:
                    logger.debug(
                        "Skipping current day TradingView record for %s due to ignore_today flag",
                        symbol,
                    )
                    continue
                tv_prices.append(price_model)
            except ValueError as err:
                logger.debug("Skipping TradingView row for %s: %s", symbol, err)
                continue

        if tv_prices:
            self.repo.save_bulk_prices(tv_prices)
            logger.debug(
                "TradingView saved %d records for %s",
                len(tv_prices),
                symbol,
            )
            return len(tv_prices)

        if full_reload:
            self.repo.ignore_symbol(symbol, "No Data (Yahoo & TradingView)")
        return 0

    def _fallback_to_tradingview(
        self,
        failed_symbols: list[str],
        full_reload: bool,
        ignore_today: bool = False,
    ) -> int:
        """Fallback method to fetch failed symbols from TradingView."""
        if not failed_symbols:
            return 0

        today_str = datetime.now().strftime("%Y-%m-%d")
        logger.info(
            "Triggering TradingView fetch/fallback for %d symbols...",
            len(failed_symbols),
        )
        saved_count = 0
        n_bars = 1200 if full_reload else 15
        total_symbols = len(failed_symbols)

        for idx, symbol in enumerate(failed_symbols, start=1):
            if idx % 25 == 1 or idx == total_symbols:
                logger.info(
                    "TradingView progress: %d/%d symbols processed...",
                    idx,
                    total_symbols,
                )
            saved_count += self._process_single_tradingview_symbol(
                symbol,
                n_bars,
                full_reload=full_reload,
                ignore_today=ignore_today,
                today_str=today_str,
            )

        return saved_count
