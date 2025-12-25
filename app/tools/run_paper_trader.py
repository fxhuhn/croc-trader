import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

from app.config import settings
from app.core.database import OHLCVRepository
from app.database.sqlite_repo import SQLiteRepository
from app.services.paper_trader import PaperTrader


def main():
    # 1. Init Repos
    signal_repo = SQLiteRepository(settings.database.signal_path)
    market_repo = OHLCVRepository(settings.database.market_data_path)

    # 2. Init Engine
    trader = PaperTrader(signal_repo, market_repo)

    # 3. Run
    print("🚀 Starting Paper Trader Update...")
    trader.run()
    print("✅ Done.")


if __name__ == "__main__":
    main()
