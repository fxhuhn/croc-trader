---
name: strategy-screener
description: "The mathematical core of signal generation and order calculation, managing the transition from stock screening to trade executions."
---

# Strategy & Screening Agent Skill

* **Must strictly respect [.agents/rules/workspace.md](.agents/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill defines the role, scope, rules, and invariants of the specialized **Strategy & Screening Agent**. This agent manages the mathematical models for scanning market assets, evaluating setup indicators, validating trade entry criteria, executing position sizing, and performing strategy audits.

## Operational Boundaries & Trigger Commands

This agent is invoked via the following dedicated slash commands:
- `/strategy-audit`: Audits the mathematical validity and parameter settings of the active trading strategies against the codebase implementation.
- `/generate-signals`: Triggers the daily or monthly screener runs to evaluate symbols, log new trade proposals in `signals.db`, and prepare execution brackets.

---

## Screener Scope (Signal-Erkennung)

The **Screener** (`app/services/screener/`) is responsible for identifying tradeable setups. Its output is a `trades` record with Status `CREATED`.

| Responsibility | Description |
|:---|:---|
| **OHLCV-Daten lesen** | Read daily bars from `stocks.db` via `MarketDataProvider` |
| **Setup-Indikatoren berechnen** | Calculate indicators once per signal (SMA, RSI, ATR, ROC) |
| **Setup-Bedingungen evaluieren** | Apply strategy-specific filter rules to identify setups |
| **Signal schreiben** | Write trade record to `signals.db` with Status `CREATED` |
| **`signal_context` befüllen** | Store setup metadata (date, indicators, reference price) |
| **`entry_price` setzen** | Calculate strategy-specific reference price (see Entry-Price Semantics) |
| **Duplikat-Check** | Prevent duplicate signals for same symbol/date/strategy |
| **Telegram: Signal erkannt** | Send notification with setup details (see Telegram section) |

**The Screener must NEVER:**
- Write execution sizes, order IDs, or broker state details
- Activate or close trades (status transitions)
- Calculate position sizing
- Generate order objects or CSV exports

---

## Trade Manager Scope (Trade-Ausführung & Management)

The **Trade Manager** (`app/services/trade_manager/`) consumes `CREATED` signals and manages the full trade lifecycle.

| Responsibility | Description |
|:---|:---|
| **Entry-Bedingung prüfen** | Evaluate if a `CREATED` trade should be activated (`CREATED → ACTIVE`) |
| **Fill-Preis bestimmen** | Determine actual entry price (Open, Limit fill, Close) |
| **Position Sizing** | Calculate trade size via risk-based or budget-based cascade |
| **Exit-Indikatoren berechnen** | Recalculate indicators daily for active trade management |
| **Exit-Bedingungen evaluieren** | Check stop-loss, take-profit, time-stop, or indicator exits |
| **`signal_context` aktualisieren** | Update daily state via `get_daily_updates()` |
| **Trade Status-Übergänge** | Manage `CREATED → ACTIVE → CLOSED` lifecycle |
| **Order-Generierung** | Create Order objects for entry and exit |
| **CSV-Export** | Export bracket orders to `data/orders/` |
| **Telegram: Trade aktiviert/geschlossen** | Notification on status changes (planned) |

---

## Entry-Price Semantics

The `entry_price` field is set by the Screener as a **strategy-specific reference price**. The Trade Manager interprets it per strategy:

| Strategy | Screener sets `entry_price` to | Trade Manager interprets as | Actual Fill |
|:---|:---|:---|:---|
| TwoPercent | `close * 0.99` | Limit price | `min(Open, Limit)` |
| Turnover | `close - (factor * ATR)` | Limit price | `min(Open, Limit)` |
| BounceBandit | `current_close` | Ignored → MOO | `Open[t+1]` |
| TGIM | `min(Fri, Thu)` | Threshold | `Monday Close` |
| NDXMomentum | `closing_price` | Ignored → MOO | `Open[t+1]` |
| BridgeScout | `close` | Reference from context | Limit from context |
| DipBuyer | `Close - ATR` | Limit price with SL guard | `min(Open, Limit)` |

---

## Telegram-Benachrichtigungen

### Screener-Telegram: „Signal erkannt" 🔎

Sent after successful signal generation via `_send_telegram_report()`:

| Strategy | Title | Fields |
|:---|:---|:---|
| TwoPercent | `🔎 TwoPercent Entries ({date})` | Symbol, Setup Close, Limit Entry |
| Turnover | `🔎 Turnover Signals ({date})` | Symbol, Entry 0.5 ATR, Entry 1.0 ATR, Close, ATR |
| DipBuyer | `🔎 DipBuyer (LIVE) ({date})` | Candidate details DataFrame |
| CrocSetup | `🔎 Croc Signals ({date})` | Symbol, Signal, Score, Entry, Stop, TP |
| BounceBandit | *(no custom report)* | — |
| TGIM | *(no custom report)* | — |
| NDXMomentum | *(no custom report)* | — |
| BridgeScout | *(no custom report)* | — |

### Trade Manager-Telegram (planned)

| Event | Content |
|:---|:---|
| Trade activated (CREATED → ACTIVE) | Symbol, Fill-Price, Position Size, Entry Reason |
| Trade closed (ACTIVE → CLOSED) | Symbol, Exit-Price, Exit Reason, PnL |
| Trade rejected / expired | Symbol, Rejection Reason |

---

## Sonderfall: NDX Momentum — Monatliches Rebalancing

NDX Momentum follows a fundamentally different model than all other strategies:

1. **Screener** identifies monthly the Top 5 Momentum Leaders and writes them as `CREATED`
2. **Trade Manager** checks on month switch for each `ACTIVE` trade if the symbol is **still in the current leaders list**
3. **If YES** → Position remains without rebalancing (no sell + rebuy)
4. **If NO** → Position is closed via `REBALANCE_EXIT`
5. **New leaders** not yet in the portfolio are opened via `REBALANCE_ENTRY`

This anti-rebalancing principle prevents unnecessary turnover for positions that remain in the top leaders across months.

---

## Strict Mathematical Invariants
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

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
