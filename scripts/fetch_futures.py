"""Futures Market Data Fetcher.

Downloads 30-minute futures bars from TradingView and aggregates them into
cash-session daily bars for comparison with equity ETFs (QQQ, SPY).

Usage:
    .venv/bin/python scripts/fetch_futures.py [--symbol MNQ MES] [--bars 500] [--contract MNQU2026]

Options:
    --symbol     Internal symbols to fetch (default: MNQ MES)
    --bars       Number of 30-minute bars to download (default: 500)
    --contract   Explicit contract override (skips rollover heuristic)

Side Effects:
    Writes 30-minute bars to futures_prices and cash-session daily bars
    to futures_daily in data/futures.db.
"""

import argparse
import logging
import sys
from datetime import date, datetime
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from app.config import settings  # noqa: E402
from app.database.repositories.futures import FuturesRepository  # noqa: E402
from app.database.session import DatabaseSession  # noqa: E402
from app.models import FuturesPrice  # noqa: E402
from app.services.market.futures_aggregation import (  # noqa: E402
    aggregate_cash_session_daily_bars,
)
from app.services.market.futures_provider import (  # noqa: E402
    FUTURES_REGISTRY,
    FuturesContract,
    FuturesDataProvider,
    resolve_front_month_contract,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("FuturesFetch")

DEFAULT_BAR_COUNT = 500


def main() -> None:
    """Entry point for the futures data fetcher."""
    arguments = _parse_arguments()

    futures_database_path = settings.get_path("futures")
    session = DatabaseSession(str(futures_database_path))
    repository = FuturesRepository(session)
    repository.init_schema()

    provider = FuturesDataProvider()
    reference_date = date.today()

    contracts = _resolve_contracts(arguments, reference_date)
    if not contracts:
        logger.error("No contracts to fetch.")
        return

    total_intraday_records = 0
    total_daily_records = 0

    for contract in contracts:
        intraday_count, daily_count = _fetch_and_store_contract(
            contract=contract,
            provider=provider,
            repository=repository,
            number_of_bars=arguments.bars,
        )
        total_intraday_records += intraday_count
        total_daily_records += daily_count

    logger.info(
        "Finished: %d intraday bars + %d daily bars saved to %s",
        total_intraday_records,
        total_daily_records,
        futures_database_path,
    )


def _parse_arguments() -> argparse.Namespace:
    """Parses CLI arguments for the futures fetcher."""
    parser = argparse.ArgumentParser(
        description="Fetch 30-minute futures data from TradingView"
    )
    parser.add_argument(
        "--symbol",
        nargs="+",
        default=list(FUTURES_REGISTRY.keys()),
        help=f"Internal symbols (default: {', '.join(FUTURES_REGISTRY.keys())})",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=DEFAULT_BAR_COUNT,
        help=f"Number of 30-min bars to fetch (default: {DEFAULT_BAR_COUNT})",
    )
    parser.add_argument(
        "--contract",
        type=str,
        default=None,
        help="Explicit contract override (e.g. MNQU2026). Applies to first --symbol only.",
    )
    return parser.parse_args()


def _resolve_contracts(
    arguments: argparse.Namespace,
    reference_date: date,
) -> list[FuturesContract]:
    """Resolves contracts from CLI arguments or front-month heuristic."""
    contracts: list[FuturesContract] = []

    for symbol in arguments.symbol:
        symbol_upper = symbol.strip().upper()
        if symbol_upper not in FUTURES_REGISTRY:
            logger.warning("Unknown symbol %s, skipping.", symbol_upper)
            continue

        if arguments.contract and symbol_upper == arguments.symbol[0].strip().upper():
            contract = _build_explicit_contract(symbol_upper, arguments.contract)
        else:
            contract = resolve_front_month_contract(symbol_upper, reference_date)

        logger.info(
            "Resolved %s → %s (exchange: %s)",
            symbol_upper,
            contract.tv_symbol,
            contract.exchange,
        )
        contracts.append(contract)

    return contracts


def _build_explicit_contract(
    base_symbol: str,
    contract_string: str,
) -> FuturesContract:
    """Builds a FuturesContract from an explicit contract string override."""
    spec = FUTURES_REGISTRY[base_symbol]
    contract_upper = contract_string.strip().upper()
    # Extract month code and year from the contract string
    # e.g. "MNQU2026" -> prefix="MNQ", month_code="U", year=2026
    prefix_length = len(spec.tv_prefix)
    month_code = contract_upper[prefix_length]
    expiry_year = int(contract_upper[prefix_length + 1 :])

    # Reverse lookup for month code to month number
    month_code_to_month = {"H": 3, "M": 6, "U": 9, "Z": 12}
    expiry_month = month_code_to_month.get(month_code, 0)

    return FuturesContract(
        base_symbol=base_symbol,
        tv_symbol=contract_upper,
        exchange=spec.exchange,
        month_code=month_code,
        expiry_month=expiry_month,
        expiry_year=expiry_year,
    )


def _fetch_and_store_contract(
    contract: FuturesContract,
    provider: FuturesDataProvider,
    repository: FuturesRepository,
    number_of_bars: int,
) -> tuple[int, int]:
    """Fetches, transforms, and persists data for a single contract.

    Returns:
        Tuple of (intraday_record_count, daily_record_count).
    """
    start_time = datetime.now()

    # 1. Fetch 30-min bars from TradingView
    raw_records = provider.fetch_history(contract, number_of_bars=number_of_bars)
    if not raw_records:
        logger.warning("No data received for %s", contract.tv_symbol)
        return 0, 0

    # 2. Transform to domain models
    futures_prices: list[FuturesPrice] = []
    for record in raw_records:
        try:
            price = FuturesPrice.from_tradingview(
                symbol=contract.base_symbol,
                contract=contract.tv_symbol,
                row=record,
            )
            futures_prices.append(price)
        except ValueError as error:
            logger.debug("Skipping bar for %s: %s", contract.tv_symbol, error)

    if not futures_prices:
        logger.warning("No valid bars after parsing for %s", contract.tv_symbol)
        return 0, 0

    # 3. Persist intraday bars
    repository.save_bulk_futures_prices(futures_prices)
    intraday_count = len(futures_prices)

    # 4. Aggregate and persist cash-session daily bars
    daily_bars = aggregate_cash_session_daily_bars(futures_prices)
    repository.save_bulk_daily_bars(daily_bars)
    daily_count = len(daily_bars)

    duration = datetime.now() - start_time
    logger.info(
        "%s: saved %d intraday + %d daily bars in %s",
        contract.tv_symbol,
        intraday_count,
        daily_count,
        duration,
    )

    return intraday_count, daily_count


if __name__ == "__main__":
    main()
