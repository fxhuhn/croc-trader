# Strategy Playbook: NDXMomentum

- **Core Concept:** Multi-asset relative strength strategy holding the top 5 momentum leaders in the NASDAQ-100 index, rebalanced monthly.
- **Screener Ingestion:** Reads daily closing prices from `stocks.db` for NASDAQ-100 constituent stocks plus the index tracking ETF `QQQ`:
  - `close`: Closing prices.
- **Screener Rules & Filters:**
  - **Screener Timing**: Executes strictly on the last trading day of each month (the last non-holiday weekday).
  - **QQQ Trend Filter**: The QQQ closing price must be above its 200-day Simple Moving Average (SMA_200) to trigger a rebalance. If QQQ is below SMA_200, no new positions are entered.
  - **Momentum Score Calculation**:
    - Calculate Rate of Change (ROC) percentages for each constituent over four lookback windows: 21, 63, 126, and 252 trading days.
    - Sum the four values: $\text{Momentum Score} = \text{ROC}_{21} + \text{ROC}_{63} + \text{ROC}_{126} + \text{ROC}_{252}$.
    - Rank all constituents by their score (highest first).
  - **Leaders Selection**: Select the top 5 highest-ranking constituent symbols.
- **Signal Triggers:**
  - **Entry**: Proposed at market open on the first trading day of the new month for any selected leader not currently held in the portfolio.
  - **Stop-Loss**: None (`0.0`).
  - **Take-Profit**: None (`0.0`).
- **Order Generation Rules:**
  - **Rebalancing**: 
    - Exit order (MKT) generated for symbols falling out of the top leaders list.
    - Entry order (MKT) generated for new leaders entering the top 5, sizing the position according to the allocated strategy budget.
  - **Lifespan**: Execution occurs at market open (DAY / MKT).

---

## Strategy Decision Flowchart

```mermaid
graph TD
    Start((Start)) --> Status{Status?}

    subgraph Setup ["Einstiegs-Filter (CREATED)"]
    Status -- CREATED --> RegimeCheck{QQQ BULL?}
    RegimeCheck -- Nein --> Reject[Setup ablehnen]
    RegimeCheck -- Ja --> DupCheck{Bereits aktiv?}
    DupCheck -- Ja --> Reject
    DupCheck -- Nein --> TimeWait{Tag >= 1?}
    TimeWait -- Nein --> Wait[Warten]
    TimeWait -- Ja --> Buy[Kauf / Market Open]
    end

    subgraph Rebalance ["Monatliches Rebalancing (ACTIVE)"]
    Status -- ACTIVE --> MonthCheck{Monatswechsel?}
    MonthCheck -- Nein --> Hold[Position halten]

    MonthCheck -- Ja --> LeaderCheck{Noch in Leaders?}
    LeaderCheck -- Ja --> Hold
    LeaderCheck -- Nein --> Exit[Verkauf / Rebalance Exit]
    end

    %% Logische Verbindungen
    Buy --> Hold

    %% Zusammenführung der Endzustände
    Hold --> Finish((Ende Zyklus))
    Exit --> Finish
    Reject --> Finish
    Wait --> Finish

    %% Styling
    style Start fill:#f9f,stroke:#333
    style Finish fill:#f9f,stroke:#333
    style Buy fill:#dcfce7,stroke:#166534
    style Exit fill:#fee2e2,stroke:#991b1b
    style Reject fill:#f1f5f9,stroke:#475569
    style Setup fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
    style Rebalance fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
```
