# Strategy Playbook: DipBuyer

- **Core Concept:** Mean-reversion setup that identifies oversold stocks in long-term uptrends, anticipating a short-term bounce.
- **Screener Ingestion:** Reads these fields for active symbols from `stocks.db`:
  - `close`: Closing price.
  - `open`: Opening price.
  - `high`: Daily high price.
  - `low`: Daily low price.
  - `volume`: Traded daily volume.
  - `date`: Trading day string.
- **Screener Rules & Filters:**
  - **Uptrend Trend Filter**: The closing price must be above its 200-day Simple Moving Average (SMA_200).
  - **Liquidity Filters**: 20-day Volume SMA must be $> 1,000,000$ and closing price must be $> \$5.0$.
  - **Consecutive Red Candles**: Both today's session and yesterday's session must be red candles (`Close < Open`).
  - **Dip Condition**: The price drop over 3 days (measured as $Price_{3} - Price_{0}$) divided by the 5-day ATR must be $< -1.0$.
  - **Volatility Buffer**: The 5-day ATR divided by the closing price must be $> 3\%$ (`volatility_ratio > 0.03`).
  - **IBS Filter**: Internal Bar Strength (IBS) must be $< 0.2$ (meaning: $\frac{\text{close} - \text{low}}{\text{high} - \text{low}} < 0.2$).
- **Signal Triggers:**
  - **Entry Price Limit**: `Close - ATR` (calculated at setup close).
  - **Stop-Loss**: None (`0.0`).
  - **Take-Profit**: `Entry + (0.8 * ATR)`.
- **Order Generation Rules:**
  - **Entry order type**: Limit (LMT) order placed at entry price.
  - **Lifespan**: Good-Till-Date (GTD) expiring at the end of the next trading day. Setup is invalidated if not filled on Day 1.
  - **Exit logic**: 
    - Target take profit order (Take Profit is active from Day 1 post-entry).
    - Limit On Close (LOC) order triggered if close price rises above the previous day's high.
    - Time Stop: Closed on the 8th trading day since entry.

---

## Strategy Decision Flowchart

```mermaid
graph TD
    Start((Start)) --> Status{Status?}

    subgraph Setup ["Einstiegs-Phase (CREATED)"]
    Status -- CREATED --> DayCheck{Genau Tag 1?}
    DayCheck -- Ja --> LimitCheck["Low <= Limit?"]
    LimitCheck -- Ja --> Buy[Kauf / Aktivierung]
    DayCheck -- Nein --> Inval[Setup verfällt]
    LimitCheck -- Nein --> Inval
    end

    subgraph Management ["Management-Phase (ACTIVE)"]
    Status -- ACTIVE --> ExitCheck{Exit Bedingung?}
    ExitCheck -- "Keine" --> Hold[Halten / Warten]
    ExitCheck -- "High >= Target" --> ExitTarget[Target Hit]
    ExitCheck -- "Close > Prev High" --> ExitLOC[LOC Hit]
    ExitCheck -- ">= 8 Tage" --> ExitTime[Time Stop]
    end

    %% Logische Verbindungen
    Buy --> Hold

    %% Zusammenführung der Endzustände zur Vermeidung von Überkreuzungen
    Hold --> Finish((Ende Session))
    ExitTarget --> Finish
    ExitLOC --> Finish
    ExitTime --> Finish
    Inval --> Finish

    %% Regeln als Tooltips/Anmerkungen seitlich
    ExitTarget -.-> Rule1["Regel: Nicht am Einstiegstag"]
    ExitLOC -.-> Rule2["Regel: Am Einstiegstag möglich"]

    %% Styling
    style Start fill:#f9f,stroke:#333
    style Finish fill:#f9f,stroke:#333
    style Buy fill:#dcfce7,stroke:#166534
    style ExitLOC fill:#fee2e2,stroke:#991b1b
    style ExitTarget fill:#fee2e2,stroke:#991b1b
    style ExitTime fill:#fee2e2,stroke:#991b1b
    style Inval fill:#f1f5f9,stroke:#475569
    style Setup fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
    style Management fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
```
