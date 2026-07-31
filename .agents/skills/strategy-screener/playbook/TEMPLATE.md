# Strategy Playbook: <Canonical Strategy Name>

> Specification status: Complete | Incomplete | Conflicting
> Last verified against code: <date or Not verified>
> Last verified against configuration: <date or Not verified>
> Last verified against tests: <date or Not verified>

## 1. Identity and Status

- Strategy identifier:
- Display name:
- Strategy family:
- Variants:
- Lifecycle status:
- Playbook version:
- Screener implementation:
- Trade Manager implementation:
- Configuration section:

## 2. Objective and Hypothesis

### Core concept

### Economic or behavioral hypothesis

### Expected market regime

### Hypothesis invalidation

### Known verified limitations

## 3. Responsibility Boundary

| Responsibility | Screener | Trade Manager | Portfolio / Order Layer |
|---|:---:|:---:|:---:|
| <task> | | | |

Document only strategy-specific responsibility or deviation from `overview.md`.

## 4. Universe, Data and Provider Requirements

### Universe

### Required market data

| Field | Frequency | Adjustment policy | Minimum history | Required |
|---|---|---|---:|---|

### Provider behavior

- Primary provider: `yfinance`
- Fallback: TradingView through `tvdatafeed.get_hist()`
- `TvDatafeedLive`: prohibited
- Existing provider provenance: preserved

### Data freshness and cutoff

## 5. Configuration Contract

| Business parameter | Configuration key | Type | Unit | Required | Default | Variant-specific |
|---|---|---|---|---|---|---|

Do not invent defaults.

## 6. Trading Calendar and Timeline

| Event | Time or phase | Trading-day rule | Holiday behavior |
|---|---|---|---|
| Screener run | | | |
| Signal creation | | | |
| Earliest entry | | | |
| Entry expiration | | | |
| Earliest exit | | | |
| Time-based exit | | | |

Also define:

- exchange timezone,
- effective trading date,
- same-session or next-session entry,
- data cutoff,
- look-ahead prevention,
- MOC or MOO cutoff assumptions where applicable.

## 7. Phase 1 — Screener

### Data inputs

### Setup indicators

For each indicator:

#### `<Indicator Name>`

- Purpose:
- Exact definition:
- Inputs:
- Lookback:
- Warm-up:
- Missing-value behavior:
- Configuration key:
- Implementation symbol:

### Setup conditions

| ID | Exact condition | Formula | Required | Evaluation order | Missing-data behavior |
|---|---|---|---|---:|---|

Define the final Boolean relationship.

### Ranking and candidate selection

### Duplicate and existing-position checks

### Signal generation

### Telegram or notification behavior

## 8. Phase 1 Output Contract

### Signal fields

| Field | Type | Required | Value or meaning |
|---|---|---|---|

### Entry reference semantics

- Stored field:
- Price or threshold:
- Executable:
- Used by Trade Manager:
- Rounding:
- Tick-size behavior:

### Initial `signal_context`

| Field | Type | Value or formula | Unit | Decision relevant | Diagnostic only |
|---|---|---|---|---|---|

## 9. Phase 2 — Trade Manager Entry

### Entry preconditions

### Entry timing

### Order and fill contract

| Priority | Condition | Order type | Fill rule | Status or reason |
|---:|---|---|---|---|

### Entry expiration and invalidation

### Position sizing

- Sizing owner:
- Budget or risk based:
- Configuration key:
- Zero-size behavior:
- Decimal boundary:

## 10. Phase 2 — Active Trade Management

### Daily inputs

### Runtime indicators

### Mutable runtime state

| Field | Initial value | Update rule | Update timing | Duplicate-processing guard | Decision relevant |
|---|---|---|---|---|---|

### Daily update sequence

## 11. Phase 2 — Exit Contract

| Priority | Exit condition | Earliest eligible time | Order type | Fill rule | Final status | Reason code |
|---:|---|---|---|---|---|---|

Also define:

- simultaneous-condition behavior,
- exit evaluation order,
- next-open versus same-close semantics,
- non-trigger behavior,
- terminal state.

## 12. Portfolio Reconciliation

> Use `Not applicable` for strategies without portfolio-level reconciliation.

Define when applicable:

- target portfolio,
- current portfolio,
- positions to keep,
- positions to close,
- positions to open,
- already-active handling,
- regime-off behavior,
- operation order,
- partial execution behavior.

## 13. Persistence and Data Lifecycle

| Artifact or field | Producer | Creation time | Storage target | Mutable | Updater | Terminal behavior |
|---|---|---|---|---|---|---|

Also define:

- uniqueness key,
- idempotency behavior,
- duplicate-run behavior,
- transaction boundary,
- partial-run behavior,
- cleanup behavior,
- persistence failure behavior.

## 14. Numerical and Execution Semantics

- Analytical numeric types:
- Monetary boundary type:
- Decimal conversion point:
- Tick-size source:
- Rounding mode:
- Reference price versus fill price:
- Budget versus risk sizing:
- Broker responsibility:
- CSV responsibility:

## 15. Visual Process Model

### Required strategy flow

Include one renderable Mermaid diagram that shows:

- Screener stage,
- signal creation,
- persisted initial data,
- CREATED transition,
- Trade Manager entry decision,
- ACTIVE or INVALIDATED transition,
- runtime updates where relevant,
- exit decision,
- CLOSED transition,
- order or CSV generation where relevant.

Do not repeat every shared architecture detail from `overview.md`.

### Additional diagrams

Add only when required:

- mutable runtime state machine,
- portfolio reconciliation flow,
- non-standard data lifecycle.

## 16. Implementation, Tests and Evidence

### Implementation mapping

| Contract area | Module | Symbol | Verification status |
|---|---|---|---|

### Test contract

#### Positive behavior

#### No-signal behavior

#### Threshold and boundary cases

#### Timing and holiday cases

#### Missing-data cases

#### Provider fallback

#### Duplicate and idempotency cases

#### Entry expiration

#### Exit priority

#### Look-ahead prevention

#### Runtime-state update

#### Strategy-specific special cases

### References

| Source | Classification | Purpose |
|---|---|---|
