import sys
from pathlib import Path

# Add project root to sys.path so we can import 'app', 'core', 'services'
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

from app.backtest.nasdaq_momentum import MomentumConfig, NasdaqMomentumBacktester
from app.core.symbol_lists import nasdaq100_symbols

config = MomentumConfig(start_date="2022-01-01", initial_capital=100_000, top_n=5)

runner = NasdaqMomentumBacktester(config, nasdaq100_symbols())
runner.run()
