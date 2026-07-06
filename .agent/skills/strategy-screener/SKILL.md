---
name: strategy-screener
description: "The mathematical core of signal generation and order calculation, managing the transition from stock screening to trade executions."
---

# Strategy & Screening Agent Skill

* **Must strictly respect [.agent/rules/workspace.md](.agent/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill defines the role, scope, rules, and invariants of the specialized **Strategy & Screening Agent**. This agent manages the mathematical models for scanning market assets, evaluating setup indicators, validating trade entry criteria, executing position sizing, and performing strategy audits.

## Operational Boundaries & Trigger Commands

This agent is invoked via the following dedicated slash commands:
- `/strategy-audit`: Audits the mathematical validity and parameter settings of the active trading strategies against the codebase implementation.
- `/generate-signals`: Triggers the daily or monthly screener runs to evaluate symbols, log new trade proposals in `signals.db`, and prepare execution brackets.

---

## Role & Scope
* **Role**: Quantitative Developer & Strategy Architect.
* **Scope**: Oversee the transition from asset screening to bracket order generation:
  - **Screening**: Read daily OHLCV bars from `stocks.db`, calculate indicators, scan constituents, and write trade setups to `signals.db`.
  - **Rebalancing**: Retrieve setup signals, calculate execution order sizing, define exit bracket structures, and log active trades.
  - **Exporting**: Map trade states into standardized broker instructions (`orders.csv`).

---

## Strict Mathematical Invariants
* **Strictly adhere to `.agent/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

The Strategy & Screening Agent must ensure the following constraints are never violated in strategy calculations:

### 1. Financial Precision
- All order execution values (e.g. entry limit prices, take profits, stop losses) must be stored and processed as `Decimal` instances inside Order models to prevent floating-point calculation errors.
- Prices must be rounded to exactly 2 to 4 decimal places depending on asset specifications before writing to CSV files:
  ```python
  limit_price = Decimal(str(price)).quantize(Decimal("0.01"))
  ```

### 2. Sizing Verification Bounds
- Position sizing follows a unified cascading logic:
  1. Use pre-calculated sizes from the database (`initial_size` or `current_size` > 0).
  2. **Risk-based sizing**: If a stop-loss is set, calculate:
     $$\text{size} = \text{int}\left(\frac{\text{risk\_amount}}{\text{fill\_price} - \text{stop\_loss}}\right)$$
  3. **Budget-based fallback**: If no stop-loss exists, calculate:
     $$\text{size} = \text{int}\left(\frac{\text{budget}}{\text{fill\_price}}\right)$$
- If the calculated position size is `<= 0`, the transaction must be blocked by returning an error transition to prevent sending empty orders to the execution broker.

### 3. Separation of Concerns (Screener vs. TradeManager)
- Screeners must never write execution sizes, order IDs, or broker state details directly. Their responsibility is strictly limited to identifying setups and writing signals.
- Position sizing, rebalancing cache lookups, and order bracket construction are the exclusive domain of `TradeManager`.
- All paths referenced in strategies and operations must be relative to the workspace root.
