# Croc-Trader Developer & Maintenance Scripts

This directory contains manual runners, test generators, database sync tools, and historical simulation utilities. They are primarily used for development diagnostics, backtesting strategy logic, and performing manual database maintenance.

---

## 🛠️ Active Utilities

### 1. Database & Schema Utilities

#### `reset_trade_schema.py`
- **Purpose**: Resets the SQLite trade/signal schema by dropping the `trades` and `trade_logs` tables and recreating them.
- **Side Effects**: **DESTRUCTIVE**. Wipes all trade histories and log details in `data/signals.db`.
- **Usage**:
  ```bash
  python script/reset_trade_schema.py
  ```

#### `create_debug_view.py`
- **Purpose**: Creates or replaces the SQL view `view_screener_debug` in the signals database. This view flattens JSON data extracted from `signal_context` for `DipBuyer` signals (e.g., extracting `close`, `volume`, `setup_score`, `atr5`, and `atr_r3`). It also outputs sample data for debugging.
- **Side Effects**: Read-only query execution, plus creating/replacing a database View.
- **Usage**:
  ```bash
  python script/create_debug_view.py
  ```

#### `restore_ignored_symbols.py`
- **Purpose**: Checks the list of excluded/ignored symbols in `stocks.db` and queries Yahoo Finance. If valid market data is retrieved (resolving previous download failures), it automatically restores the symbols by removing them from the exclusion list.
- **Side Effects**: Modifies the `ignored_symbols` table in `data/stocks.db`.
- **Usage**:
  ```bash
  python script/restore_ignored_symbols.py
  ```

---

### 2. Market Data Integration

#### `manual_market_data_load.py`
- **Purpose**: Triggers a full historical reload of market prices from `2023-01-01` to today for all symbols tracked in major indices (S&P, NDX, DOW, RUS). Runs a gap check afterward.
- **Side Effects**: Heavy network usage (Yahoo Finance API) and bulk database writes to `data/stocks.db`.
- **Usage**:
  ```bash
  python script/manual_market_data_load.py
  ```

#### `manual_market_data_sync.py`
- **Purpose**: Triggers an incremental market data update (fetching only missing recent prices) followed by data validation and a gap check. Identical to the daily background task but run manually in the foreground.
- **Side Effects**: Incremental database writes to `data/stocks.db`.
- **Usage**:
  ```bash
  python script/manual_market_data_sync.py
  ```

#### `update_holidays.py`
- **Purpose**: Queries NYSE/NASDAQ trading calendars via the `exchange_calendars` package to extract all market holidays from 2025 to 2028. Writes the structured calendar metadata directly to `data/holidays.yaml`.
- **Side Effects**: Overwrites `data/holidays.yaml`.
- **Usage**:
  ```bash
  python script/update_holidays.py --start 2025 --end 2028
  ```

---

### 3. Screeners & Signal Creation

#### `manual_screener_dip_buyer.py`
- **Purpose**: Runs a one-shot live scan of the `DipBuyer` screener strategy for the current day (or a custom date). Generates trade proposals and logs candidates in the database.
- **Side Effects**: Writes new `CREATED` trades to `data/signals.db` and sends alerts to Telegram if configured.
- **Usage**:
  ```bash
  python script/manual_screener_dip_buyer.py [--date YYYY-MM-DD]
  ```

#### `manual_croc_screener_test.py`
- **Purpose**: Runs the `CrocSetup` screener strategy across a range of dates to test signal generation (created trades) and print results.
- **Side Effects**: Writes new `CREATED` trades to `data/signals.db`.
- **Usage**:
  ```bash
  python script/manual_croc_screener_test.py --start 2026-01-01 [--end YYYY-MM-DD]
  ```

---

### 4. Backtests & Historical Simulations

> [!NOTE]
> Standard backtest runners manipulate the active `signals.db` or use default parameters. Ensure you have backed up your databases before running extensive historical tests.

#### `manual_backtest_croc.py`
- **Purpose**: Simulates historical screening and execution for `Croc` setups using the pure `HoldTargetStrategy` logic.
- **Side Effects**: Runs screening and updates trade states (`CREATED`, `ACTIVE`, `CLOSED`, `realized_pnl`) in `data/signals.db`.
- **Usage**:
  ```bash
  python script/manual_backtest_croc.py --start 2026-01-01 [--risk 100.0]
  ```

#### `manual_backtest_dip_buyer.py`
- **Purpose**: Simulates historical screening and execution for `DipBuyer` setups.
- **Side Effects**: Writes and updates trade records in `data/signals.db`.
- **Usage**:
  ```bash
  python script/manual_backtest_dip_buyer.py --start 2026-01-01 [--budget 2000.0]
  ```

#### `manual_backtest_turnover.py`
- **Purpose**: Simulates historical screening and execution for `TurnoverTiming` setups.
- **Side Effects**: Writes and updates trade records in `data/signals.db`.
- **Usage**:
  ```bash
  python script/manual_backtest_turnover.py --start 2025-01-01 [--capital 2000.0]
  ```

#### `rebuild_signals.py`
- **Purpose**: Rebuilds the database signals history from scratch starting on `2025-12-29`. It uses a custom `TimeTravelMarketRepository` to prevent "lookahead bias" (ensuring the TradeManager only sees prices prior to the simulated day), runs daily trade updates, and executes all strategy screeners day-by-day.
- **Side Effects**: **DESTRUCTIVE**. Cleans out all tables in `signals.db` and rebuilds the trade logs step-by-step.
- **Usage**:
  ```bash
  python script/rebuild_signals.py
  ```

---

### 5. Diagnostics & Verifications

#### `debug_croc_fills.py`
- **Purpose**: Performs post-mortem logic checks on the Croc setup. Evaluates historical prices on the day after signal generation to print statistics on whether a trade would execute under breakout (High >= Entry) or limit (Low <= Entry) triggers.
- **Side Effects**: None (Read-only database checks).
- **Usage**:
  ```bash
  python script/debug_croc_fills.py
  ```

#### `debug_turnover_2026.py`
- **Purpose**: Scans the database for any historical `TurnoverTiming` strategy trades recorded in 2026, displaying their setup close, setup ATR, limit entry price, and current status.
- **Side Effects**: None (Read-only database checks).
- **Usage**:
  ```bash
  python script/debug_turnover_2026.py
  ```

#### `dry_run_orders.py`
- **Purpose**: Seeds a temporary database with dummy signals, mock-calculates allocations/exits, and runs the TradeManager's order generation system. Validates the resulting IBKR order CSV format and column structures.
- **Side Effects**: None (Uses a temporary database `data/signals_test.db` which is cleaned up automatically).
- **Usage**:
  ```bash
  python script/dry_run_orders.py
  ```

#### `manual_test_run.py`
- **Purpose**: Runs a dual diagnostics run:
  1. Live order generation (read-only) printout of today's CSV orders.
  2. Safe simulated daily process run (exits + entry evaluations) by copying `signals.db` to a temporary simulation database and running the cycle there to display prospective status changes before committing.
- **Side Effects**: Read-only for live databases; writes are limited to temporary simulation files.
- **Usage**:
  ```bash
  python script/manual_test_run.py
  ```

#### `verify_week_turnover.py`
- **Purpose**: Verifies the step-by-step logic of `TurnoverTiming` for the week of Feb 6–10, 2026. It checks the Friday screening, Monday entry checks (using Monday high/low candle data), and Tuesday exits (based on Monday/Friday candles).
- **Side Effects**: Deletes existing TurnoverTiming signals for `2026-02-06` to allow a clean run of the scenario.
- **Usage**:
  ```bash
  python script/verify_week_turnover.py
  ```

---

## 🗄️ Archived & Reference Scripts
Legacy, ad-hoc metrics analysis, and one-off migration scripts are moved to `script/archive/` to keep the working folder clean:
- `script/archive/check_db.py`: Quick test checking an obsolete SQLite DB path.
- `script/archive/fix_db.py`: Basic manual trigger for daily trade processing.
- `script/archive/legacy_drawdown.py`: Ad-hoc max drawdown metrics computer.
- `script/archive/legacy_keys.py`: Simple developer printout of trade dictionary keys.
- `script/archive/legacy_roi.py`: Ad-hoc win/loss and mean ROI metrics computer.
- `script/archive/filter_symbols.py`: Static volume analysis to build symbol mapping (superseded by dynamic background filter).
- `script/archive/verify_filter.py`: Ad-hoc assert verification for `SymbolFilter` logic.
- `script/archive/web_monitor.py`: Generic website load time measurer.
- `script/archive/update_umlauts.py`: One-time JSON character encoding repair migration.
- `script/archive/run_backtest.py`: Old engine-based backtest executor (deprecated).
