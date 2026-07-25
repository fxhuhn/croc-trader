# Strategy Playbook: Core Financial Math & Architecture Invariants

This document outlines the shared financial mathematics, precision constraints, and object-oriented architecture patterns governing all trading strategies in the Croc-Trader system.

---

## 1. Object-Oriented Design & DRY Principles

To prevent logic duplication and ensure structural consistency:
- **Central Abstract Base Classes**:
  - All screener strategies must inherit from [app/services/screener/strategies/base.py](app/services/screener/strategies/base.py) (`BaseStrategy`).
  - All execution strategies must inherit from [app/services/trade_manager/strategies/abstract.py](app/services/trade_manager/strategies/abstract.py) (`BaseTradeStrategy`).
- **Encapsulated Global Context**:
  - Global checks (like holiday validation via `MarketHolidayChecker`, index constituent mapping via `ExchangeSymbol`, and rate limits) must reside in base classes or dedicated singletons.
  - Individual strategy files are strictly limited to defining strategy-specific mathematical setups and state transitions.

---

## 2. Position Sizing & Risk Management

Position sizing is mathematically calculated on a trade-by-trade basis using a centralized calculation cascade:

### Risk-Based Sizing Formula
When a strategy defines a stop-loss price ($P_{\text{stop}}$) and is triggered at an execution price ($P_{\text{fill}}$), position size ($S$) is calculated using the target risk per trade ($R$, default $R = 50.0$ USD or as configured in `settings.yaml` under `portfolio.strategies.<strategy_name>.risk_amount`):

$$S = \text{int}\left( \frac{R}{|P_{\text{fill}} - P_{\text{stop}}|} \right)$$

### Budget-Based Fallback
If the strategy does not use a stop-loss, or if $P_{\text{fill}} = P_{\text{stop}}$, size calculations fallback to budget allocation ($B$, configured in `settings.yaml` under `portfolio.strategies.<strategy_name>.budget`):

$$S = \text{int}\left( \frac{B}{P_{\text{fill}}} \right)$$

- **Zero-Size Guard**: If $S \le 0$, the order generation process must abort immediately to prevent sending empty orders to the execution broker.

---

## 3. Decimal Precision Handling

To prevent rounding issues, pricing, and volume inconsistencies:
- **No Float Ledgers**: Float data types must never be used for final order generation, ledger entries, or account totals.
- **Strict Decimal Wrapping**: Wrap numeric strings into `decimal.Decimal` instances immediately before generating `Order` models:
  ```python
  from decimal import Decimal
  limit_price = Decimal(str(raw_price)).quantize(Decimal("0.01"))
  ```
- **Rounding Constraints**:
  - Prices: Rounded to exactly 2 decimal places (`Decimal("0.01")`) or 4 decimal places for micro-cap assets.
  - Size quantities: Cast to integer quantities.

---

## 4. Strategy Index

- [Bounce Bandit Playbook (BOUNCE_BANDIT)](bounce_bandit.md)
- [Bridge Scout Playbook (BRIDGE_SCOUT)](bridge_scout.md)
- [Dip Buyer Playbook](dip_buyer.md)
- [NDX Momentum Playbook](ndx_momentum.md)
- [TGIM Playbook](tgim.md)
- [Turnover Timing Playbook](turnover_timing.md)
- [Two Percent Playbook](two_percent.md)


