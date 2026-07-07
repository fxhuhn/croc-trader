# Croc-Trader High-Level System Architecture

High-density technical overview of the Croc-Trader EOD portfolio management and trading analysis platform.

---

## 1. System Component Interactions

```mermaid
graph TD
    subgraph Web_Application_Layer [Web Application Layer]
        FlaskServer["Flask Web Server (app/routes/)"]
        DashboardUI["UI Dashboard (Jinja2/Tailwind)"]
    end

    subgraph Orchestration_Layer [Orchestration & Processing Layer]
        Scheduler["APScheduler (app/services/setup.py)"]
        ScreenerEngine["Screener Engine (app/services/screener/)"]
        TradeManager["Trade Manager (app/services/trade_manager/)"]
        DataSync["Market Data Synchronizer (app/services/market/)"]
    end

    subgraph Data_Storage_Layer [Data Storage Layer]
        MarketDB[("SQLite: stocks.db (WAL Mode)")]
        LedgerDB[("SQLite: signals.db (WAL Mode)")]
        BrokerDB[("SQLite: trading.db (WAL Mode) (ReadOnly Repository)")]
    end

    subgraph External_Interfaces [External Integration Ports]
        CSVExport["CSV Order Files (data/orders/)"]
        Telegram["Telegram Bot API Notifier"]
        IBKRAgent["TWS / IBKR API (via ib_async)"]
    end

    %% Node Connections
    DashboardUI -->|"Render views / API calls"| FlaskServer
    FlaskServer -->|"Accesses / Modifies"| LedgerDB
    FlaskServer -->|"Reads executions"| BrokerDB
    Scheduler -->|"Triggers background jobs"| DataSync
    Scheduler -->|"Triggers screening"| ScreenerEngine
    Scheduler -->|"Triggers rebalancing"| TradeManager
    DataSync -->|"Downloads historical prices"| MarketDB
    ScreenerEngine -->|"Queries price history"| MarketDB
    ScreenerEngine -->|"Writes signals & setup alerts"| LedgerDB
    TradeManager -->|"Queries active positions"| LedgerDB
    TradeManager -->|"Queries real execution sizes"| BrokerDB
    TradeManager -->|"Generates bracket orders"| LedgerDB
    TradeManager -->|"Exports instructions"| CSVExport
    TradeManager -->|"Sends alerts"| Telegram
    IBKRAgent -->|"Persists orders/fills"| BrokerDB
```

---

## 2. High-Level Dataflow Topologies

### 2.1 Market Data Synchronization
1. **Trigger**: APScheduler runs the updater daily.
2. **Fetch**: `app/services/market/updater.py` pulls historical equity quotes and distributions from yfinance.
3. **Persist**: Writes to `data/stocks.db` utilizing standard parameterized SQL queries.

### 2.2 Screener Scan & Signal Generation
1. **Trigger**: APScheduler fires the Screener Engine.
2. **Compute**: `app/services/screener/engine.py` reads historical prices from `stocks.db` and computes ATR, SMA, and RSI indicator states.
3. **Record**: Evaluates candidates against rule logic and stores active setups in `signals.db`.

### 2.3 Portfolio Rebalancing & Bracket Order Export
1. **Trigger**: APScheduler triggers the Trade Manager.
2. **Process**: `app/services/trade_manager/manager.py` queries active positions, validates new entries, runs position sizing, and logs orders in `signals.db`.
3. **Export**: Outputs bracket orders as CSV into `data/orders/orders_YYYY_MM_DD.csv`.

---

## 3. Global Invariants

- **Execution Environment**: Strictly target **Python 3.12+**.
- **Financial Accounting Precision**: Float data types are strictly prohibited for ledgers, cash values, or price calculations. Use `decimal.Decimal` objects (using banker's rounding) to ensure exact precision.
- **Database Concurrency (WAL Mode)**: All SQLite databases (`stocks.db`, `signals.db`, `trading.db`) MUST operate in Write-Ahead Logging (WAL) mode (`PRAGMA journal_mode=WAL;`) to allow concurrently active read queries from the Flask UI without blocking execution writes.
- **Stateless Execution Layers**: Core logic components (Screener strategies and Trade Manager rebalancers) must be stateless. They reconstruct context on-the-fly from the underlying database state on every run.
- **Functional Core / Imperative Shell**:
  - **Functional Core**: All mathematical modeling, indicator computations, and sizing algorithms must be pure, side-effect-free functions.
  - **Imperative Shell**: Handles network calls, file system exports, SQLite transactions, and log dispatching, validating boundary contracts before invoking the core.
- **SQL Parametrization**: String concatenation or format interpolation for raw SQL query composition is strictly prohibited to prevent SQL injection vulnerability risk.
