# Strategy: <Canonical Strategy Name>

## 1. Strategy Identity

- Strategy identifier:
- Display name:
- Strategy family:
- Implementation module:
- Configuration section:
- Current lifecycle status:
- Playbook version:
- Last verified date:

## 2. Objective and Economic Hypothesis

Describe:

- intended market behavior,
- economic or behavioral hypothesis,
- expected source of return,
- expected market regime,
- hypothesis invalidation conditions.

Do not present backtest performance as the economic hypothesis.

## 3. Scope and Responsibility

Define:

- screener responsibilities,
- candidate definition,
- persisted output,
- trade-manager responsibilities,
- portfolio-allocation responsibilities,
- broker and order-handling responsibilities.

The screener must not calculate final order quantities or submit orders unless
the authoritative architecture explicitly assigns that responsibility.

## 4. Instrument Universe

Document:

- eligible asset classes,
- universe source,
- exchange restrictions,
- liquidity requirements,
- minimum data history,
- exclusion rules,
- update frequency.

Do not duplicate configured symbol lists as hard-coded playbook values.

## 5. Required Market Data

| Field | Frequency | Adjustment policy | Required history |
|---|---|---|---:|
| Open | Daily | <verified policy> | <bars> |
| High | Daily | <verified policy> | <bars> |
| Low | Daily | <verified policy> | <bars> |
| Close | Daily | <verified policy> | <bars> |
| Volume | Daily | <verified policy> | <bars> |

Central provider contract:

- Primary provider: `yfinance`.
- Fallback: TradingView through `tvdatafeed.get_hist()` only for missing
  symbols or missing required data.
- `TvDatafeedLive` is not used.
- Existing provider provenance stored per database record remains unchanged.
- The strategy playbook must not redefine the ingestion database design.

## 6. Trading-Date Semantics

Define:

- effective trading date,
- exchange timezone,
- data cutoff,
- last permissible candle,
- calculation date,
- earliest possible execution date,
- weekend and holiday behavior,
- explicit look-ahead-bias prevention.

Do not use a generic example as the strategy rule. Record the verified contract.

## 7. Configuration Mapping

| Playbook parameter | Configuration key | Type | Unit | Required | Default allowed |
|---|---|---|---|---|---|
| <parameter> | <exact key> | <type> | <unit> | Yes/No | Yes/No |

Rules:

- no invented defaults,
- no unexplained duplicated literals,
- identify whether a parameter changes strategy identity or tuning,
- missing required configuration fails explicitly.

## 8. Indicator Definitions

For each indicator:

### `<Indicator Name>`

- Purpose:
- Exact mathematical definition:
- Input series:
- Lookback:
- Warm-up period:
- Missing-value behavior:
- Library implementation:
- Precision requirements:
- Configuration key:

The mathematical intent must remain understandable independently from a
library function name.

## 9. Entry Conditions

| Identifier | Exact condition | Required | Evaluation order |
|---|---|---|---:|
| E1 | <condition> | Yes/No | 1 |

Define the complete Boolean relationship.

For crossovers define:

- previous-bar condition,
- current-bar condition,
- equality behavior,
- missing-value behavior.

Do not use non-measurable descriptions such as “strong trend”.

## 10. Exit Conditions

Separate:

- strategy exit,
- stop-related input,
- time-based exit,
- invalidation exit,
- portfolio-manager exit,
- trade-manager exit.

If the screener does not generate exits, state this explicitly.

## 11. Signal Contract

| Field | Type | Required | Meaning |
|---|---|---|---|
| strategy | <type> | Yes | Canonical strategy identifier |
| symbol | <type> | Yes | Canonical instrument symbol |
| trading_date | date | Yes | Effective strategy date |
| direction | <type> | Yes | Direction semantics |
| reference_price | <type> | Yes | Non-execution reference |
| signal_context | mapping | Yes | Strategy evidence |

Define:

- uniqueness key,
- duplicate behavior,
- idempotency,
- creation status,
- invalid-signal behavior.

Do not include final order quantity unless architecture assigns it to the
screener.

## 12. Signal Context Schema

| Key | Type | Required | Unit | Meaning |
|---|---|---|---|---|
| <key> | <type> | Yes/No | <unit> | <description> |

Rules:

- no undocumented keys,
- no convenience abbreviations,
- decision values must equal values used by the calculation,
- diagnostic-only values must be identified.

## 13. Reference Price Semantics

Define:

- price field,
- adjusted or unadjusted,
- informational or downstream use,
- execution limitations,
- tick-size or rounding behavior.

A reference price must not be described as an executable price.

## 14. Position and Risk Inputs

Document only strategy-produced or strategy-required risk inputs:

- stop reference,
- volatility measure,
- risk distance,
- ranking score,
- quality score.

Do not define account budgets, broker limits, portfolio allocation, or final
quantities unless assigned by architecture.

## 15. Ranking and Candidate Selection

Define:

- ranking metric,
- sort direction,
- deterministic tie-breaking,
- maximum candidates,
- missing-value behavior.

If absent, state:

`No cross-sectional ranking is performed.`

## 16. Missing and Invalid Data Behavior

| Condition | Required behavior |
|---|---|
| Symbol or required data missing from yfinance | Use established TradingView fallback |
| Required data missing from both providers | Reject evaluation |
| Insufficient warm-up history | Reject evaluation |
| Non-finite input | Reject evaluation |
| Stale final candle | Reject evaluation |
| Missing required configuration | Fail explicitly |
| Invalid OHLC relation | Reject affected data |
| Duplicate run | Apply documented idempotency contract |

Do not silently replace missing prices with zero.

## 17. Synchronous EOD Execution Sequence

1. Resolve effective trading date.
2. Load strategy configuration.
3. Load eligible universe.
4. Load verified daily market data.
5. Validate history and data cutoff.
6. Calculate indicators.
7. Evaluate conditions.
8. Create deterministic candidates.
9. Validate signal contract.
10. Persist idempotently when production application code invokes the strategy.
11. Return an execution summary.

The skill itself does not execute production signal generation.

## 18. Idempotency and Duplicate Protection

Define:

- run identity,
- uniqueness key,
- rerun behavior,
- partial-run behavior,
- skip or update behavior,
- transaction boundary.

The same inputs and trading date must not create duplicate signals.

## 19. Numerical Precision and Rounding

Define:

- analytical numeric types,
- monetary boundary types,
- tick-size source,
- rounding mode,
- Decimal conversion boundary,
- floating-point comparison tolerance where applicable.

Do not use generic “two decimals” rules unless valid for every permitted
instrument.

## 20. Logging and Observability

Define required structured events:

- evaluation started,
- provider fallback used,
- symbol rejected with reason,
- candidate generated,
- duplicate skipped,
- persistence completed,
- evaluation completed,
- evaluation failed.

Do not log credentials or unnecessary sensitive data.

## 21. Implementation Mapping

| Playbook section | Module | Symbol | Status |
|---|---|---|---|
| Indicator | <module> | <symbol> | Verified/Not verified |
| Entry | <module> | <symbol> | Verified/Not verified |
| Persistence | <module> | <symbol> | Verified/Not verified |

Every mapping must reference an inspected existing symbol.

Code is the reviewed artifact, not automatically the normative contract.

## 22. Test Contract

### Positive tests

- verified signal case,
- verified no-signal case,
- exact threshold case,
- crossover case where applicable.

### Boundary tests

- insufficient warm-up,
- first valid calculation date,
- threshold equality,
- missing final candle,
- provider fallback,
- duplicate run.

### Failure tests

- missing configuration,
- invalid OHLC,
- non-finite values,
- persistence failure where relevant.

### Bias tests

- no future candle access,
- no premature use of execution-day values,
- correct data cutoff.

## 23. Acceptance Criteria

- [ ] Strategy identity is canonical.
- [ ] Parameters map to existing configuration keys.
- [ ] Indicators have exact definitions.
- [ ] Conditions are unambiguous.
- [ ] Trading-date semantics prevent look-ahead bias.
- [ ] Missing-data behavior is explicit.
- [ ] Provider fallback follows the central ingestion contract.
- [ ] Signal schema and uniqueness are documented.
- [ ] Implementation mappings reference existing symbols.
- [ ] Required tests exist or missing tests are reported.
- [ ] Screener responsibilities do not conflict with architecture.

## 24. Known Limitations

Document only verified limitations.

Do not add speculative weaknesses or unsupported claims.

## 25. References

Classify each source as:

- normative,
- supporting,
- implementation evidence,
- empirical evidence.
