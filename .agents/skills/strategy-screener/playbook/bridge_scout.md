# Strategy Playbook: Bridge Scout (BRIDGE_SCOUT)

- **Core Concept:** End-of-Month Mean-reversion setup specifically traded on `QQQ` to scout and capture short-term dips in the final trading days of the calendar month, exiting on the first trading day of the new month ("bridging" the month turn).
- **Screener Ingestion:** Reads daily OHLCV prices from `stocks.db` for the target symbol `QQQ`:
  - `close`, `high`, `low`: Daily prices.
  - `date`: Trading day string.
- **Screener Rules & Filters:**
  - **Month-End Window Filter (`EndOfMonthWindow`)**: Active starting 4 trading days before the final trading day of the calendar month through the last trading day of the month ($D_{N-4}$ to $D_N$).
    - **Trading Day Calendar & Holiday Calculation**: Uses `MarketHolidayChecker` (`app/tools/market_holidays.py`) to skip weekend days (`weekday() >= 5`) and official US market holidays.
  - **Constituent Selection**: Traded exclusively on `QQQ` (fixed asset).
  - **Oversold Setup Condition**: `RSI(2) < 40.0`.
  - **Volatility Filter Condition**: `(ATR(10) / Close) * 100 < 3.5%`.
  - **Position Limit Check (Strict Single Entry)**: Maximum 1 open position (`MaxPositions = 1`). No multiple entries permitted; if an active trade or setup for `QQQ` exists in `signals.db`, new entry signals are blocked.
  - **Duplicate Check**: Prevents signal duplication by checking trade repository state.
- **Signal Triggers & Order Types:**
  - **Entry Order Type**: Market On Close (MOC) on the setup day where entry criteria are met.
  - **Stop-Loss / Take-Profit**: None (`0.0`).
  - **Exit Rule (`ExitRule`)**: Market On Close (MOC / `ThisClose`) triggered on the first trading day of the new calendar month (`DateYear(BarDate) != DateYear(EntryDate)` or `DateMonth(BarDate) != DateMonth(EntryDate)`).
- **Position Sizing & Risk Management:**
  - Standard **Budget-Based Fallback**: Calculated using standard portfolio allocation without a stop loss ($\text{size} = \lfloor \text{budget} / P_{\text{fill}} \rfloor$), configured via `portfolio.strategies.bridge_scout.budget`.
- **Configurable Parameters:**
  - `entry_days_before_month_end`: Lookback trading days before month-end (default: `4`).
  - `rsi_entry_threshold`: RSI(2) max threshold (default: `40.0`).
  - `atr_length`: ATR period (default: `10`).
  - `max_atr_pct`: Max ATR % relative to Close (default: `3.5`).
  - `target_symbol`: Ticker symbol (fixed: `"QQQ"`).

---

## Strategy Decision Flowchart

```mermaid
graph TD
    Start((Start)) --> Status{Status?}

    subgraph Setup ["Einstiegs-Phase (CREATED)"]
        Status -- CREATED --> WindowCheck{Im Monatsend-Fenster?<br/>(inkl. Holiday Check)}
        WindowCheck -- Nein --> Inval[Setup entfällt]
        WindowCheck -- Ja --> RSIFilter["Is RSI(2) < 40?"]
        RSIFilter -- Nein --> Inval
        RSIFilter -- Ja --> ATRFilter["Is ATR(10)/Close < 3.5%?"]
        ATRFilter -- Nein --> Inval
        ATRFilter -- Ja --> PosCheck{Bereits Position/Setup aktiv?<br/>(MaxPositions = 1)}
        PosCheck -- Ja --> Inval
        PosCheck -- Nein --> Buy[Kauf QQQ per MOC am Setup-Tag]
    end

    subgraph Management ["Management-Phase (ACTIVE)"]
        Status -- ACTIVE --> MonthCheck{Neuer Kalendermonat?<br/>(ExitRule)}
        MonthCheck -- Ja --> ExitMOC[Exit per MOC am 1. Handelstag]
        MonthCheck -- Nein --> Hold[Position halten]
    end

    %% Logische Verbindungen
    Buy --> Management
    Hold --> Finish((Ende Session))
    ExitMOC --> Finish
    Inval --> Finish

    %% Rules Notes
    Buy -.-> Rule1["MOC Einstieg am Setup-Tag (End of Month Window)"]
    ExitMOC -.-> Rule2["Exit am 1. Handelstag des neuen Monats MOC"]

    %% Styling
    style Start fill:#f9f,stroke:#333
    style Finish fill:#f9f,stroke:#333
    style Buy fill:#dcfce7,stroke:#166534
    style ExitMOC fill:#fee2e2,stroke:#991b1b
    style Inval fill:#f1f5f9,stroke:#475569
    style Setup fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
    style Management fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
```
