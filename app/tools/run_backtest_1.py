import sys
from pathlib import Path

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.backtest.minervini_backtester import MinerviniBacktester, StrategyConfig
from app.core.symbol_lists import dow30_symbols, nasdaq100_symbols, sp500_symbols

# Define your universe here (or load from file/DB)
nasdaq_100 = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "COST",
    "PEP",
]
SYMBOLS = list(set(sp500_symbols() + nasdaq100_symbols() + dow30_symbols()))

if __name__ == "__main__":
    # Create strategy-specific config (parameters only)
    # DB paths are automatically loaded from app/config.yaml via config.settings
    config = StrategyConfig(
        start_date="2020-01-01", initial_capital=100_000, max_positions=10
    )

    # Initialize and run
    # It will read 'market_data.db' and write to 'backtest.db'
    runner = MinerviniBacktester(config, SYMBOLS)
    runner.run()
