# Strategy Playbook: TurnoverTiming

- **Core Concept:** Exploits short-term institutional momentum shifts using price/turnover volume thresholds and weekly time stops.
- **Screener Ingestion:** Reads daily bars from `stocks.db` for the constituents of Nasdaq-100 (NDX), S&P 500 (SPX), and Russell 1000 (RUS):
  - `open`, `high`, `low`, `close`: Daily prices.
  - `volume`: Daily volume.
- **Screener Rules & Filters:**
  - **Screener Timing**: Runs at the last trading close of the week (normally Friday close, or Thursday if Friday is a holiday).
  - **Indicators Calculated**:
    - `turnover` = Close * Volume
    - `turnover_sma` = 20-day Simple Moving Average of turnover.
    - `sma_price` = 150-day Simple Moving Average of closing price.
    - `atr` = 3-day Average True Range (ATR).
  - **Ranking & Selection Cascade**:
    1. Rank all constituent symbols in each index by `turnover_sma` descending (highest turnover first).
    2. Exclude secondary share classes via `SymbolFilter`.
    3. Take the top 20 highest-turnover candidates in each index.
    4. Apply filters: `close > sma_price` (uptrend filter) and `atr > 0`.
    5. Select the top 4 candidates meeting these criteria per index.
    6. Deduplicate candidates across indices (merging index names if a symbol is in multiple indices).
- **Signal Triggers:**
  - **Entry Triggers**: The screener creates signals for two separate strategy variants:
    - **TurnOverTiming_0.5**: Limit Buy price is set to `Setup Close - (0.5 * ATR)` (using 3-day ATR).
    - **TurnOverTiming_1.0**: Limit Buy price is set to `Setup Close - (1.0 * ATR)` (using 3-day ATR).
  - **Entry Validation**: The entry limit order is strictly valid only on the first trading day following the setup day (normally Monday). If Monday is a public holiday, the entry window is extended to Tuesday. Otherwise, the setup is invalidated.
  - **Stop-Loss**: None (`0.0`).
  - **Take-Profit**: None (`0.0`).
- **Order Generation Rules:**
  - **Entry order type**: Limit (LMT) order. Valid strictly on the first trading day following the signal (normally Monday, or Tuesday if Monday is a holiday).
  - **Exit logic**: 
    - Early exit: Triggered when two consecutive trading days record green candles (`Close > Open`), checking the day's close and executing at the next day's open.
    - State tracking: Uses `green_candle_count`. Upon entry fill, if the setup day was green, it counts as 1. If the entry day itself is also green, `green_candle_count` is initialized to 2 (triggering the green sequence exit at the next day's open). If the current day is red or neutral, the count resets to 0.
    - Time Stop: Closed automatically at market close on Friday (MOC order).

---

## Strategy Decision Flowchart

```mermaid
%%{init: {'flowchart': {'rankSpacing': 100, 'nodeSpacing': 60, 'curve': 'basis'}}}%%
graph TD
    %% Start
    START([Start Strategy Iteration]) --> DB_READ[Read Trades from Database]
    
    DB_READ --> STATUS_CHECK{Trade Status?}

    %% Path A: New Trades (CREATED)
    STATUS_CHECK -- "CREATED" --> D1_CHECK["Evaluation Day 1 <br/>after signal?"]
    
    D1_CHECK -- "No (Wait)" --> SKIP[Skip until tomorrow]
    
    D1_CHECK -- "Yes" --> LIMIT_CHECK["Daily Low <= Entry Price?"]
    
    LIMIT_CHECK -- "No" --> EXPIRE[Mark as REJECTED / EXPIRED]
    EXPIRE --> DB_WRITE_EXP[Save to Database]
    
    LIMIT_CHECK -- "Yes" --> ACTIVATE[Mark as ACTIVE <br/>Set Fill @ min Open, Entry]
    ACTIVATE --> INIT_STATE[Initialize green_candle_count <br/>based on candle color]
    INIT_STATE --> DB_WRITE_ACT[Save to Database]

    %% Path B: Active Trades (ACTIVE)
    STATUS_CHECK -- "ACTIVE" --> IDEM_CHECK{Already processed <br/>today?}
    
    IDEM_CHECK -- "Yes" --> SKIP
    
    IDEM_CHECK -- "No" --> SEQ_CHECK["green_candle_count >= 2?"]
    
    %% Exit Option 1: Green Sequence
    SEQ_CHECK -- "Yes" --> EXIT_G[Close Trade @ OPEN <br/>Reason: GREEN_SEQUENCE]
    EXIT_G --> DB_WRITE_EXIT[Save to Database]
    
    %% Management Path
    SEQ_CHECK -- "No" --> UPDATE_COUNT[Update green_candle_count <br/>based on current candle]
    UPDATE_COUNT --> TIME_CHECK{Is End of Week <br/>Friday/Holiday?}
    
    %% Exit Option 2: Time Stop
    TIME_CHECK -- "Yes" --> EXIT_T[Close Trade @ CLOSE <br/>Reason: TIME_STOP]
    EXIT_T --> DB_WRITE_EXIT
    
    %% No Exit
    TIME_CHECK -- "No" --> DB_WRITE_COUNT[Save updated context to DB]

    %% Styling
    style START fill:#f9f,stroke:#333
    style DB_READ fill:#ddd,stroke:#333
    style STATUS_CHECK fill:#fff,stroke:#333
    style ACTIVATE fill:#99f,stroke:#333
    style EXPIRE fill:#f96,stroke:#333
    style EXIT_G fill:#9f9,stroke:#333
    style EXIT_T fill:#9f9,stroke:#333
    style DB_WRITE_EXP fill:#ddd,stroke:#333
    style DB_WRITE_ACT fill:#ddd,stroke:#333
    style DB_WRITE_EXIT fill:#ddd,stroke:#333
    style DB_WRITE_COUNT fill:#ddd,stroke:#333
```

