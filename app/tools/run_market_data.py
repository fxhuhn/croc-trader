import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------------------
# 1. SETUP & PATHS
# --------------------------------------------------------------------------

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.config import settings  # Centralized Config
from app.core.database import OHLCVRepository
from app.core.symbol_lists import dow30_symbols, nasdaq100_symbols, sp500_symbols
from services.market_data.base import MarketDataLoader
from services.market_data.yahoo_loader import YahooLoader

# Optional: Import symbol lists if you have them in core
# from core.symbol_lists import sp500_symbols, nasdaq100_symbols

# Default fallback list if "all" is used but no dynamic lists are available
DEFAULT_SYMBOLS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "BRK-B"]
SYMBOLS = list(set(sp500_symbols() + nasdaq100_symbols() + dow30_symbols()))

# --------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# --------------------------------------------------------------------------


def load_exchange_map(
    json_path: str = "config/us_symbol_exchange.json",
) -> Dict[str, str]:
    """Load symbol-to-exchange mapping from JSON file."""
    path = project_root / json_path
    if path.exists():
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not read exchange map '{json_path}': {e}")
    return {}


def run_loader(
    loader_name: str, loader: MarketDataLoader, symbols: List[str], db_path: str
):
    """
    Execute a specific data loader and upsert results into the database.
    """
    print(f"\n🚀 Starting Loader: {loader_name.upper()} for {len(symbols)} symbols...")

    try:
        # 1. Fetch Data
        df = loader.fetch_daily_history(symbols, lookback_years=4)

        if df is None or df.empty:
            print(f"⚠️ No data returned by {loader_name}.")
            return

        # 2. Save to DB
        repo = OHLCVRepository(db_path)
        # The 'upsert_dataframe' method automatically tags the source
        count = repo.upsert_dataframe(df, source=loader_name)

        print(f"💾 Success: Saved {count} rows to '{db_path}' (source='{loader_name}')")

    except Exception as e:
        print(f"❌ Error running loader '{loader_name}': {e}")


# --------------------------------------------------------------------------
# 3. MAIN EXECUTION
# --------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Market Data Loader")

    parser.add_argument(
        "--source",
        choices=["yahoo", "ibkr", "all"],
        default="yahoo",
        help="The data source to load from.",
    )

    parser.add_argument(
        "--db",
        default=settings.database.market_data_path,  # Uses centralized config
        help="Path to the market data SQLite database.",
    )

    parser.add_argument(
        "--symbols",
        default="all",
        help="Comma-separated list of symbols or 'all' for default list.",
    )

    args = parser.parse_args()

    # --- 1. Resolve Symbols ---
    if args.symbols.lower() == "all":
        # In production, you might combine lists: sp500_symbols() + nasdaq100_symbols()
        symbols_to_load = SYMBOLS
        print(f"ℹ️  Loading default symbol list ({len(symbols_to_load)} symbols)")
    else:
        symbols_to_load = [
            s.strip().upper() for s in args.symbols.split(",") if s.strip()
        ]
        print(f"ℹ️  Loading {len(symbols_to_load)} specific symbols from CLI argument")

    # --- 2. Load Configuration ---
    exchange_map = load_exchange_map()

    # --- 3. Initialize Loaders ---
    # Add new loaders (e.g., IBKR) to this dictionary as you implement them
    loaders: Dict[str, MarketDataLoader] = {
        "yahoo": YahooLoader(exchange_map),
        # "ibkr": IBKRLoader(exchange_map),  # Uncomment when implemented
    }

    # --- 4. Determine Sources to Run ---
    if args.source == "all":
        sources_to_run = list(loaders.keys())
    else:
        sources_to_run = [args.source]

    # --- 5. Execute ---
    for source_name in sources_to_run:
        loader_instance = loaders.get(source_name)
        if loader_instance:
            run_loader(source_name, loader_instance, symbols_to_load, args.db)
        else:
            print(f"❌ Loader '{source_name}' is not registered/implemented.")
