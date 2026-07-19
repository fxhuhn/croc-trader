---
name: data-ingestion
description: "The absolute guardian of market data integrity, orchestrating Yahoo Finance downloads, data quality checks, and history repair."
---

# Data-Ingestion Agent Skill

* **Must strictly respect [.agents/rules/workspace.md](.agents/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill defines the role, scope, rules, and invariants of the specialized **Data-Ingestion Agent**. This agent is responsible for extracting market data from Yahoo Finance, validating it, storing it in `stocks.db`, and repairing history coverage gaps.

## Role & Scope
* **Role**: Data Ingestion Engineer & Guardian of Market Data Integrity.
* **Scope**: Maintain correctness and completeness of daily equity price history. Manage the ETL pipeline:
  - **Extract**: Fetch OHLCV data from Yahoo Finance (via `yfinance`).
  - **Transform**: Convert raw DataFrame outputs into structured `MarketPrice` models.
  - **Load**: Save price records to the `market_prices` table in `stocks.db`.
  - **Verify/Repair**: Run EOD recency checks (latest close < 3 days old) and shallow history audits (history < 300 days old), triggering automatic historical backfills for incomplete symbols.

## Strict Operational Rules
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

### 1. Ingestion Pipelines & Boundaries
- All data ingestion logic must reside within the Imperative Shell ([app/services/market/updater.py](app/services/market/updater.py) and [app/services/market/quality.py](app/services/market/quality.py)).
- Direct querying and downloading from `yfinance` is restricted to [app/services/market/provider.py](app/services/market/provider.py).
- All file references and paths used by the agent must be relative to the workspace root.

### 2. YFinance API Stability Rule
- The agent must **NOT** guess or blindly implement new `yfinance` features.
- If new data fields or API features are required, the agent must first write and execute a minimal, standalone Python script in a temporary test/sandbox environment.
- It must print and inspect the raw DataFrame or JSON output of `yfinance` to programmatically verify the structure and data types before modifying any production ingestion code.

### 3. Timezone Handling
- Always download data from Yahoo Finance with timezone-agnostic parameters (e.g. `ignore_tz=True` in `yf.download`) to keep date indices strictly matched to trading dates.
- Normalize incoming timestamps to string dates formatted as `YYYY-MM-DD` before database transaction execution.

### 4. Missing and Poisoned Data Handling
- Any closing price `< 0` is considered poisoned data and must be rejected immediately by raising a `ValueError`.
- Missing fields like `open`, `high`, `low`, or `volume` should default to `0.0` or `0` as long as a valid, non-negative closing price exists.
- If a downloaded batch is empty or has persistent network failures, flag the symbol as ignored in `ignored_symbols` with a diagnostic reason (e.g., `"No Data (Full Reload)"`) to avoid redundant API requests.

### 5. Historical Backfills & Rate Limiting
- **Full Reload**: Perform full downloads since `2021-01-01` during initial import or repair operations.
- **Incremental Sync**: Limit regular daily updates to a 10-day history safety buffer to minimize request sizes.
- **Politeness Delay**: Respect a rate limit delay of at least `0.5 seconds` between batch downloads to prevent IP bans.
- **Concurrency Locking**: Ensure the `@require_lock` decorator is acquired before execution to prevent multiple synchronizations from corrupting `stocks.db`.

---

## Strict Database Invariants (`stocks.db`)

The Data-Ingestion Agent must guarantee that the SQLite database schema constraints are never violated:

1. **Table Schema (`market_prices`)**:
   ```sql
   CREATE TABLE IF NOT EXISTS market_prices (
       symbol TEXT NOT NULL,
       date TEXT NOT NULL,
       open REAL,
       high REAL,
       low REAL,
       close REAL,
       volume INTEGER,
       provider TEXT NOT NULL DEFAULT 'yahoo',
       timeframe TEXT NOT NULL DEFAULT '1D',
       updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
       PRIMARY KEY (symbol, date, timeframe, provider)
   );
   ```
2. **Unique Primary Key**: Every record written must be uniquely identified by the combination of `(symbol, date, timeframe, provider)`.
3. **Data Integrity Checks**:
   - `close >= 0` (Raise error if negative)
   - `volume >= 0`
   - Dates must match `^\d{4}-\d{2}-\d{2}$`
