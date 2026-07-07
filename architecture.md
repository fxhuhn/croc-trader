# Croc-Trader System Architecture

This document describes the design, directory structure, data flows, and strict implementation invariants of the Croc-Trader system.

---

## 1. System Overview & Component Responsibilities

Croc-Trader is a python-based End-of-Day (EOD) portfolio management and trading analysis platform built with a Flask web application, an SQLite database layer, and a scheduled background processing loop via APScheduler.

```
                  ┌──────────────────────────────┐
                  │       Flask Web Server       │
                  │  (app/routes/ & app/views/)  │
                  └──────────────┬───────────────┘
                                 │ HTTP / JSON API
                                 ▼
                     ┌───────────────────────┐
                     │   Scheduler (Jobs)    │
                     │  (app/services/setup) │
                     └───────────┬───────────┘
                                 │
     ┌───────────────────────────┼───────────────────────────┐
     ▼                           ▼                           ▼
┌──────────────┐          ┌──────────────┐            ┌──────────────┐
│   Screener   │          │Trade Manager │            │  Backtester  │
│ (app/services│          │ (app/services│            │ (DEPRECATED) │
│  /screener/) │          │/trade_manager│            │/backtester/  │
└──────┬───────┘          └──────┬───────┘            └──────┬───────┘
       │ Writes                  │ Writes                    │ Reads/Writes
       ▼ Signals                 ▼ Positions/Orders          ▼
┌────────────────────────────────────────────────────────────────────┐
│                         Database Layer                             │
│       SQLite: stocks.db (market data) / signals.db (ledger)        │
│                (app/database/repositories/)                        │
└────────────────────────────────────────────────────────────────────┘
```

### Component Details
- **Web Frontend & API Engine** ([app/routes/](app/routes/)): Exposes view dashboards ([app/routes/views/](app/routes/views/)) for analytics, trades, screeners, and backtester runs. Uses vanilla JavaScript on the frontend with a Tailwind CSS design system rendered by Jinja2 templates.
- **Scheduled Orchestration** ([app/services/setup.py](app/services/setup.py)): Schedules background tasks using APScheduler, coordinating market data synchronization, daily stock screening, position management, database backups, and cache pre-warming.
- **Screener Engine** ([app/services/screener/](app/services/screener/)): Selects candidate equities daily. Dispatches to specific strategies ([app/services/screener/strategies/](app/services/screener/strategies/)), runs indicator formulas, and records daily trade signals to `signals.db`.
- **Trade Manager** ([app/services/trade_manager/](app/services/trade_manager/)): Evaluates active positions, handles entry/exit execution rules, applies portfolio-level risk limits, and exports bracket order instructions as CSV.
- **Database Repositories** ([app/database/repositories/](app/database/repositories/)): Isolates DB access behind a repository interface:
  - [app/database/repositories/market_data_provider.py](app/database/repositories/market_data_provider.py): Interface for reading price history and computing indicators.
  - [app/database/repositories/trade.py](app/database/repositories/trade.py): Ledger mapping of orders, portfolio targets, and execution history.
  - [app/database/repositories/signal.py](app/database/repositories/signal.py): Registry of generated screening signals.
- **Backtesting System** ([app/services/backtester/](app/services/backtester/)) **[DEPRECATED / UNMAINTAINED]**: Simulates custom strategies on historical market databases. Active trade tracking and metrics execution has been moved to the Trade Manager and general analytics.

---

## 2. Dataflow Boundaries & API Integrations

### Market Data Synchronization
1. **Trigger**: APScheduler runs the updater daily.
2. **Fetch**: [app/services/market/updater.py](app/services/market/updater.py) calls `yfinance` to fetch stock splits, dividends, and closing prices.
3. **Persist**: Writes to `data/stocks.db` via standard parameterized SQL.

### Screener & Signal Generation
1. **Trigger**: Scheduler triggers `screener_engine` daily before market hours.
2. **Compute**: [app/services/screener/engine.py](app/services/screener/engine.py) reads stock prices from `stocks.db`, calculates ATR, SMA, and RSI using pure functions in [app/tools/indicators.py](app/tools/indicators.py).
3. **Record**: Evaluates candidate tickers against configured rules (e.g. `DipBuyer`, `TurnoverTiming`, `CrocSetup`) and writes active buy/sell alerts to the `signals.db` database.

### Portfolio Rebalancing & Order Export
1. **Trigger**: Scheduler triggers the Trade Manager daily.
2. **Process**: [app/services/trade_manager/manager.py](app/services/trade_manager/manager.py) analyzes active positions in the database, filters newly generated signals, and runs position-sizing algorithms.
3. **Export**: Orders are generated and written to `signals.db`. If a strategy is CSV-supported (defined in [app/services/trade_manager/order_export.py](app/services/trade_manager/order_export.py)), a bracket-formatted order row is written to `data/orders/orders_YYYY_MM_DD.csv`.

### Order CSV Interface Contract

This contract defines the schema, rules, and lifecycle for exchanging order instructions between the core trading engine and executing brokers or custom execution agents.

#### 1. File Location & File Lifecycle
- **Generation**: The trading engine writes daily order instructions to [data/orders/](data/orders/) in the format `orders_YYYY_MM_DD.csv`.
- **Reference Example**: A schema definition template is available in [data/orders/orders.csv.example](data/orders/orders.csv.example).
- **Consumption Flagging**: External consumers (such as custom execution bots) read the file. Once successfully executed or submitted, the consumer should move the file to a `data/orders/processed/` archive directory or append a `.processed` suffix to the file name to prevent double-execution.

#### 2. Technical Formatting Rules
- **Encoding**: UTF-8.
- **Delimiter**: Comma (`,`).
- **Newline character**: LF (`\n`).
- **Decimal Rounding**: All prices must be rounded to exactly 2 to 4 decimal places depending on asset requirements (using standard banking rounding via `decimal.Decimal` to avoid float precision drift).

#### 3. Column-by-Column Data Dictionary

| Column | Data Type | Format / Allowed Values | Description |
| :--- | :--- | :--- | :--- |
| `trade_group_id` | String | Alphanumeric identifier (e.g., `123_DipBuyer_SPY`) | Groups the entry and exit bracket order legs belonging to a single trade. |
| `bracket_role` | Enum | `ENTRY`, `TP`, `SL`, `EXIT` | The role of the order leg. `TP` = Take Profit, `SL` = Stop Loss. |
| `symbol` | String | Uppercase alphanumeric (e.g., `AAPL`, `SPY`) | The exchange symbol identifier of the equity or asset. |
| `sec_type` | String | `STK` | Security type (e.g., Stock). |
| `exchange` | String | Destination exchange (e.g., `SMART`) | The execution routing destination. |
| `account_id` | String | IBKR Account ID (e.g., `DU123456`) | The broker account ID to execute the order under. |
| `action` | Enum | `BUY`, `SELL` | The transaction direction. |
| `quantity` | Integer | Positive integer `> 0` | The number of shares/units to trade. |
| `order_type` | Enum | `LMT`, `MKT`, `STP` | Order type (Limit, Market, or Stop). |
| `target_price` | Decimal | Numeric string with 2 decimal places (e.g., `185.50`) | The target price for the order leg. |
| `tif` | Enum | `GTD`, `GTC`, `DAY` | Time-In-Force instructions. |
| `strategy_name` | String | Strategy display name (e.g., `DipBuyer`) | The name of the generating trading strategy. |
| `currency` | String | Currency code (e.g., `USD`, `EUR`) | The asset trading currency override. |

---

## 3. Strict Implementation Invariants

Every module added to this repository must respect these hard constraints:

### 1. The Quality Pyramid
- **Correctness First**: Never compromise logic or accuracy for speed or brevity.
- **Readability**: Intention-revealing naming. No abbreviations (except `df`, `db`, `avg`, `qty`, `pnl`). Max cognitive complexity is 15, and max cyclomatic complexity is 10.
- **Maintainability**: Fully strict Python 3.12+ type hints. Google-style docstrings are mandatory for all public classes and functions.
- **Changeability**: Modules must be orthogonal and loosely coupled.

### 2. Functional Core vs. Imperative Shell
- **Functional Core**: All mathematical calculations (such as indicators, rebalancing calculations, and sizing allocations) must be pure, deterministic functions with **zero side effects** (no I/O, database access, logging, or datetime calls).
- **Imperative Shell**: Handles DB access, file export, network calls, and logs warnings/errors. It validates all inputs before forwarding them to the Functional Core.

### 3. Financial Precision & Security
- **No Float Ledgers**: Float data types are strictly prohibited for financial accounting or currency values. Use integers (cents) or `decimal.Decimal` objects to prevent precision loss.
- **SQL Safety**: All query commands must be parameterized. Raw SQL concatenation or f-string parameterization is strictly forbidden to prevent SQL injection.
- **Fail-Closed Operations**: If database connection or external services fail, the system must abort execution immediately (`sys.exit(1)`) instead of falling back to unsafe default states.

---

## 4. System Ports & Custom Agent Interfaces

External automated agents interact with Croc-Trader through these designated integration boundaries (ports):

### IBKR Integration Interface (`ibkr-agent`)
- **API Wrapper**: Interacts using the `ib_async` library (see [tools/ibkr_check.py](tools/ibkr_check.py)).
- **Connection Port**: Connects via socket ports (`7496` for live TWS, `7497` for paper trading).
- **Data Contracts**:
  - Writes system state snapshots to [data/positions.json](data/positions.json) and [data/orders.json](data/orders.json).
  - Merges completed trade details with [data/orders_history.json](data/orders_history.json).
  - Can read generated orders from `data/orders/` CSV logs and submit them directly via `ib_async` trades.

### REST API Gateway (`api-service-agent`)
- **Routing**: Interacts with the Flask JSON API endpoints defined in [app/routes/api.py](app/routes/api.py).
- **Functionality**:
  - Exposes trade logs and live position summaries.
  - Allows agents to post override instructions or manual rebalance requests.
  - Ensures whitelisting and rate-limiting constraints defined in `settings.yaml` are strictly enforced.
- **Primary Endpoints**:
  - `POST /screener/run`: Manually triggers a scan of active strategies (e.g. `CrocSetup`, `DipBuyer`, `TurnoverTiming`, etc.) over lookback periods.
  - `POST /orders/generate`: Triggers the daily order generation process and updates the orders CSV files (saved in `data/orders/`).
  - `POST /webhook`: Ingests incoming third-party signal alerts to automate execution setups.
  - `POST /trades/backfill`: Manually runs daily order and position reconciliation processes.
  - `POST /market/sync`: Orchestrates background yfinance market data downloader jobs.
  - `POST /market/reload`: Performs a full history rebuild of pricing databases.

---

## 5. Appendix: Public API Index

The following index documents all public classes and functions in the repository codebase to fulfill architecture synchronization requirements:

- AllocationResult
- AppConfig
- BacktestMetrics
- BaseRepository
- BaseStrategy
- BaseTradeStrategy
- BrokerRepository
- CapacityMonitor
- CapacitySimulator
- ConfigManager
- CrocContext
- CrocSetupStrategy
- CrocSignal
- DatabaseConfig
- DatabaseSession
- DipBuyerConfig
- DipBuyerMarketState
- DipBuyerStrategy
- DynamicPositionSizer
- EntryReason
- EnvConfig
- ExchangeMapper
- ExchangeSymbol
- ExitReason
- HoldTargetStrategy
- HolidayConfig
- IndexAliases
- LoggingConfig
- MarketDataProvider
- MarketDataUpdater
- MarketHolidayChecker
- MarketPrice
- MarketQualityService
- MarketRepository
- MetricsOverview
- MissingRouteRateLimiter
- NDXAnalysisResult
- NDXMomentumConfiguration
- NDXMomentumScreener
- NDXMomentumTradeStrategy
- OrderLeg
- Order
- OverflowProtection
- PortfolioAllocator
- PortfolioConfig
- PortfolioManager
- PortfolioMetrics
- PortfolioStrategyConfig
- PriceData
- RankingVerificationResult
- SQNClassification
- ScreenerConfiguration
- ScreenerEngine
- ScreenerViewService
- SecurityConfig
- SignalRepository
- SignalStat
- SimulationResult
- Strategies
- StrategyOverview
- StrategyProtocol
- SymbolAnalysisResult
- SymbolExchange
- SymbolFilter
- SymbolOverride
- TargetColumn
- TechnicalIndicatorConfig
- TelegramBot
- TelegramConfig
- TradeData
- TradeEventType
- TradeManager
- TradeParams
- TradeRepository
- TradeStatus
- TradeTransition
- TradeViewData
- TradeViewService
- TurnoverCandidate
- TurnoverConfiguration
- TurnoverContext
- TurnoverSignalContext
- TurnoverTimingStrategy
- TwoPercentStrategy
- TwoPercentStrategyContext
- WebhookPayload
- WebhookWorkerConfig
- WebserverConfig
- YahooDataProvider
- YahooRow
- allocate
- analyze_croc
- analyze_dip_buyer
- analyze_ndx_momentum
- analyze_single_symbol
- analyze_turnover
- apple_touch_icon
- apple_touch_icon_precomposed
- apply_limits
- attach_sparklines
- calculate_analysis
- calculate_atr
- calculate_ema
- calculate_expectancy
- calculate_ibs
- calculate_kelly_criterion
- calculate_max_drawdown
- calculate_multiplier
- calculate_position_size
- calculate_profit_factor
- calculate_risk_reward_ratio
- calculate_roc
- calculate_rsi
- calculate_sharpe_ratio
- calculate_sma
- calculate_sqn
- calculate_true_range
- calculate_ulcer_index
- calculate_volume_sma
- calculate_win_rate
- check_entry
- check_ranking_attributes
- clear_cache
- clear_trades
- configure_scheduler
- create_app
- create_trade
- dataclass_to_dict
- dow_30
- execute
- exists
- extract_latest_leaders
- extract_symbol_data
- favicon
- fetch_all
- fetch_batch_raw
- fetch_one
- fetch_value
- filter_symbols
- from_dict
- from_row
- from_yahoo
- generate_daily_orders
- generate_donut_chart
- generate_orders
- generate_sparkline
- get_active_positions
- get_active_trades
- get_all_by_strategy
- get_all_daily_data
- get_all_orders
- get_all_known_symbols
- get_all_recommendations
- get_all_traded_symbols
- get_available_dates
- get_batch_history
- get_batch_history_raw
- get_broker_active_trades
- get_broker_settlements
- get_broker_summary
- get_budget
- get_by_status
- get_by_timestamp
- get_candidates
- get_closed_summary
- get_current_parameters
- get_current_utilization
- get_daily_updates
- get_data_for_lookback
- get_db_path
- get_exchange
- get_executions_for_order
- get_executions_for_trade_group
- get_float
- get_folder
- get_handler
- get_holiday_name
- get_ignored_symbols
- get_index_stats
- get_int
- get_latest_date
- get_latest_price
- get_latest_signal_date
- get_log_path
- get_net_positions_by_symbol
- get_ohlcv
- get_orders_by_local_trade_id
- get_orders_by_status
- get_outdated_symbols
- get_path
- get_percentile
- get_portfolio_summary
- get_reconciliation_discrepancies
- get_risk_amount
- get_settlements
- get_signal_by_id
- get_signals_by_date
- get_strat_config
- get_strategy
- get_strategy_display_name
- get_strategy_path
- get_symbol_history
- get_symbol_history_raw
- get_symbols_with_missing_history
- get_trade
- get_trade_candidates
- get_trades
- get_trading_days_count
- get_turnover_candidates
- get_unique_signal_attributes
- get_universe_daily_data
- get_unprocessed_signals
- get_weekday_stats
- group_trades_by_symbol
- group_trades_history
- harmonize_indices
- health_check
- ignore_symbol
- ingest_webhook
- init_schema
- internal_server_error
- is_holiday
- is_strategy_match
- load
- manage_active_trade
- map_order_to_csv_rows
- nasdaq_100
- normalize_timestamp
- page_not_found
- perform_gap_check
- preload_all_data
- prepare_trade_view
- process_daily_signals
- register_services
- register_strategy
- reload_market_data
- remove_ignored_symbol
- require_ip_whitelist
- require_lock
- resolve_strategy
- risk_range
- robots_txt
- root_check
- route_honeypot_admin
- route_honeypot_login
- run_all
- run_cache_prewarm
- run_daily_process
- run_daily_strategy_check
- run_db_backup
- run_db_maintenance
- run_market_data_update
- run_order_generation
- run_update
- russell_1000
- russell_1000_exclusive
- save_bulk_prices
- save_signal
- send
- send_dataframe
- send_message
- should_block
- sp_500
- special_symbols
- sync_market_data
- to_db_row
- trigger_orders
- trigger_screener
- trigger_trades_backfill
- update_trade
- verify_ranking_system
- view_analytics_dashboard
- view_backtest_dashboard
- view_broker_dashboard
- view_screener_croc
- view_screener_dip_buyer
- view_screener_ndx_momentum
- view_screener_overview
- view_screener_turnover
- view_screener_twopercent
- view_trades_croc
- view_trades_dip_buyer
- view_trades_ndx_momentum
- view_trades_overview
- view_trades_turnover
- view_trades_twopercent
- wrapper
- write_csv_orders_file
