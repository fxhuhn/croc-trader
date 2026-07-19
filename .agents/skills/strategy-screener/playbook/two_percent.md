# Strategy Playbook: TwoPercent

- **Core Concept:** Mean-reversion setup specifically traded on `SXRV.DE` to capture weekend reversals or monday gap openings.
- **Screener Ingestion:** Reads daily closing prices from `stocks.db` for the target symbol `SXRV.DE`:
  - `close`: Closing prices.
- **Screener Rules & Filters:**
  - **Screener Timing**: Runs at the last trading close of the week (normally Friday close, or Thursday if Friday is a holiday).
  - **Constituent Selection**: Traded exclusively on `SXRV.DE` (no other assets screened).
  - **Duplicate Check**: Prevents signal duplication by checking if a trade proposal for the target symbol and signal date already exists in the trade repository.
- **Signal Triggers:**
  - **Entry Limit**: Limit buy placed at `Setup Close * 0.99` (1% discount). The entry trigger is strictly valid only on Monday. If Monday is a public holiday, the entry window is extended to Tuesday.
  - **Gap Opening Special Entry**: If Monday (or Tuesday, if Monday is a holiday) opens below the Limit Buy level, the entry fill price is adjusted to that day's opening price.
  - **Stop-Loss**: None (`0.0`).
  - **Take-Profit**: `Entry Price * 1.02` (+2%). Target only becomes active on Tuesday (Day + 1 post-setup, or Wednesday if Monday was a holiday).
- **Order Generation Rules:**
  - **Entry order type**: Limit (LMT) order. Valid strictly on Monday (or Tuesday if Monday is a holiday).
  - **Exit order type**: 
    - Limit (LMT) target exit at `Entry * 1.02` (Take Profit).
    - Market On Close (MOC) order triggered on Friday close if the target was not reached during the week (Time Stop).

---

## Strategy Decision Flowchart

```mermaid
graph TD
    Start((Start)) --> Status{Status?}

    subgraph Setup ["Einstiegs-Phase (CREATED)"]
    Status -- CREATED --> DayCheck{Welcher Tag?}

    DayCheck -- "Tag 1 (Mo)" --> HolCheck{Feiertag?}
    HolCheck -- Ja --> DayCheck
    HolCheck -- Nein --> FillCheck{Low <= Limit?}
    DayCheck -- "Tag 2 (Di)" --> PreHol{War Mo Feiertag?}
        PreHol -- Ja --> FillCheck
        PreHol -- Nein --> Inval[Setup verfällt]

        FillCheck -- Ja --> Buy[Kauf / Aktivierung]
        FillCheck -- Nein --> Inval
        end

        subgraph Management ["Management-Phase (ACTIVE)"]
        Status -- ACTIVE --> AgeCheck{Tag >= Entry + 1?}

        AgeCheck -- Ja --> TPCheck{High >= Target?}
        TPCheck -- Ja --> ExitTP[Verkauf / Take Profit]
        TPCheck -- Nein --> TimeCheck

        AgeCheck -- Nein --> TimeCheck{Wochenende erreicht?}

        TimeCheck -- "Fr (oder Do-Holiday)" --> ExitTime[Verkauf / Time Stop]
        TimeCheck -- Nein --> Hold[Position halten]
        end

        %% Logische Verbindungen
        Buy --> AgeCheck

        %% Zusammenführung
        Hold --> Finish((Ende Session))
        ExitTP --> Finish
        ExitTime --> Finish
        Inval --> Finish

        %% Regeln
        Buy -.-> Rule1[Bei Gap Down: Kauf zum Open]
        ExitTP -.-> Rule2[Benefit bei Gap Up möglich]

        %% Styling
        style Start fill:#f9f,stroke:#333
        style Finish fill:#f9f,stroke:#333
        style Buy fill:#dcfce7,stroke:#166534
        style ExitTP fill:#fee2e2,stroke:#991b1b
        style ExitTime fill:#fee2e2,stroke:#991b1b
        style Inval fill:#f1f5f9,stroke:#475569
        style Setup fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
        style Management fill:#f8fafc,stroke:#cbd5e1,stroke-dasharray: 5 5
```
