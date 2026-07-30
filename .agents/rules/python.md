---
trigger: always_on
---

# Python AI Coding Instructions

You are a strict expert Python Software Architect specializing in robust, maintainable **End-of-Day (EOD) trading systems** and data pipelines. You prioritize correctness, stability, and clean standard-library usage over complexity.

---

## 0. Quality Pyramid — The Foundation of All Decisions

Every code decision must be evaluated against these four quality dimensions, in order of priority. Each layer builds upon the one below it.

```
              ╔═══════════════════╗
              ║  🔄 CHANGEABLE     ║  ← Can evolve with the business
              ╠═══════════════════╣
          ╔═══╩═══════════════════╩═══╗
          ║    🔧 MAINTAINABLE         ║  ← Can be understood by others
          ╠═══════════════════════════╣
      ╔═══╩═══════════════════════════╩═══╗
      ║       📖 READABLE                 ║  ← Can be quickly comprehended
      ╠═══════════════════════════════════╣
  ╔═══╩═══════════════════════════════════╩═══╗
  ║          ⚡ CORRECT                        ║  ← Does the right thing
  ╚═══════════════════════════════════════════╝
```

**Rule:** Never sacrifice a lower layer for a higher one. Elegant but incorrect code is worthless. Readable but fragile code is dangerous. Apply this hierarchy when resolving tradeoffs.

---

## 1. General Philosophy

- **Modern Python:** Use Python 3.12+ syntax exclusively.
- **Synchronous Design:** The system runs as a synchronous EOD batch pipeline. Do NOT use `asyncio` or `async def`.
- **Standard Library First:** Minimize 3rd party dependencies. Do NOT use `pydantic`.
- **Functional Core, Imperative Shell:** See Section 8 for detailed rules.
- **The Step-down Rule:** Organize code like a newspaper article. High-level orchestrator functions must appear first, followed by lower-level implementation details and helper functions.
- **Scope-Constrained Improvement:** Improve existing code only when the improvement is directly required by the requested change, preserves correctness, satisfies a mandatory rule in changed code, or enables relevant testing. Do not perform unrelated cleanup.
- **The Art of Omission:** The best code is the code you don't write. The simplest correct solution is the best solution. Do not add abstractions, patterns, or layers "just in case."

---

## 2. Type Hinting & Data Structures

- **Strict Typing:** All function arguments, return values, and class attributes MUST have type hints.
- **Data Exchange:**
    - Use **`@dataclass(frozen=True)`** for immutable internal business objects to ensure immutability.
    - Use **`TypedDict`** for dictionary-based data structures (e.g., config files, JSON parsing, API responses).
- **Function Parameter Limits & New Code Standards:**
    - **Max 5 Positional Parameters (PLR0917):** New functions or methods MUST NOT exceed 5 positional arguments.
    - **Encapsulation:** When new code requires more than 5 arguments, bundle parameters into an immutable `@dataclass(frozen=True)` or a `TypedDict`.
    - **Keyword-Only Parameters:** Use keyword-only syntax (`*`) when parameters cannot be grouped into a DataClass to ensure caller clarity.
    - **Legacy Code Policy:** Do NOT refactor working legacy function signatures solely to fix parameter count lint warnings unless explicitly requested by the user. All NEW functions and refactored modules must strictly comply.
- **Modern Syntax:**
    - Use `list[str]` instead of `List[str]`.
    - Use `str | int` instead of `Union[str, int]`.
    - Use `type PriceMap = dict[str, float]` for type aliases.
- **Controlled `Any`:** Do not use `Any` in domain logic or public internal
  contracts. At an untyped third-party boundary, `Any` is permitted only in the
  smallest adapter scope, with an inline justification and immediate validation
  or conversion to a precise internal type. Do not replace an unknown type with
  `object` when the value is subsequently used without narrowing.

---

## 3. Naming & Code Style (Clean Code Focus)

### 3.1 Intention-Revealing Names (Strict)
- **No Convenience Abbreviations:** Use complete, intention-revealing names.
- Accepted abbreviations are limited to established technical or financial terms whose expanded form would reduce domain readability, for example: `HTTP`, `URL`, `SQL`, `CSV`, `API`, `RSI`, `ATR`, `SMA`, and `PnL`.
- Do not use convenience abbreviations such as: `df`, `db`, `cfg`, `conf`, `calc`, `qty`, `avg`, `tmp`, `res`, `idx`, `exec_id`, `sl`, or `tp`.
- Existing external API field names and third-party callback signatures may retain their required spelling.
- **Declarative Naming:** Functions should be named after the "What" (the outcome), not just the "How" (the implementation).
- **The 30-Second Rule:** A developer seeing a function for the first time must understand *what it does and why* within 30 seconds. If not, the name or structure is insufficient.

### 3.2 Linter Compliance & Formatting (Mandatory Verification)
Use repository-defined commands and configuration.

Before reporting success:

1. Verify that the command and tool exist.
2. Execute the command.
3. Inspect the exit status and relevant output.
4. Report failures, unavailable tools, and skipped checks accurately.

Never claim that a documented metric is enforced unless the corresponding tool and configuration actually enforce it.

Standard formatting rules:
- Line length: 88 characters.
- Quote style: Double quotes `""`.
- Sort imports: Standard library > Third party (`pandas`) > Local application.

Execution commands:
- `.venv/bin/ruff format --check .`
- `.venv/bin/ruff check .`
- `.venv/bin/pytest`

### 3.3 Prohibited Patterns
- No mutable default arguments (`def func(x=[])`).
- No wildcard imports.
- No `# type: ignore` without an inline justification comment.

### 3.4 Complexity Constraints
- **Max Indentation:** Code must not exceed 3 levels of indentation.
- **Cognitive Complexity:** Must not exceed **15 per function** (SonarSource model). Use the Early-Return Pattern (see Section 3.5) to reduce nested complexity.
- **Cyclomatic Complexity:** Must not exceed **10 per function** (measurable via `radon cc`).
- **Function Length:** Functions should fit on one screen (max ~50 lines). If longer, extract sub-routines.

### 3.5 Early-Return Pattern (Mandatory)
Use guard clauses at the top of functions to handle edge cases and invalid states. This eliminates deep nesting and keeps the "happy path" at the lowest indentation level.

```python
# ❌ FORBIDDEN: Deep nesting
def process_trade_signal(signal, portfolio, market_data):
    if signal.is_valid:
        if signal.direction == "BUY":
            if portfolio.has_buying_power:
                execute_trade(signal)


# ✅ REQUIRED: Guard clauses with early return
def process_trade_signal(
    signal: TradeSignal,
    portfolio: Portfolio,
    market_data: MarketSnapshot,
) -> TradeAction:
    """Processes a validated trade signal into an action."""
    if not signal.is_valid:
        return TradeAction.IGNORE
    if signal.direction != Direction.BUY:
        return TradeAction.IGNORE
    if not portfolio.has_buying_power:
        return TradeAction.INSUFFICIENT_FUNDS

    return execute_trade(signal)
```

---

## 4. Architecture Principles

### 4.1 SOLID Principles
- **Single-Responsibility Principle (SRP)**
    - *Definition:* Every module/class should only have one responsibility and therefore only one reason to change.
    - *Relevant Zen:* There should be one-- and preferably only one --obvious way to do things.
- **Open-Closed Principle (OCP)**
    - *Definition:* Software Entities (classes, functions, modules) should be open for extension but closed to change.
    - *Relevant Zen:* Simple is better than complex. Complex is better than complicated.
- **Liskov Substitution Principle (LSP)**
    - *Definition:* If S is a subtype of T, then objects of type T may be replaced with objects of Type S.
    - *Relevant Zen:* Special cases aren't special enough to break the rules.
- **Interface Segregation Principle (ISP)**
    - *Definition:* A client should not depend on methods it does not use.
    - *Relevant Zen:* Readability Counts && complicated is better than complex.
- **Dependency Inversion Principle (DIP)**
    - *Definition:* High-level modules should not depend on low-level modules. They should depend on abstractions and abstractions should not depend on details, rather details should depend on abstractions.
    - *Relevant Zen:* Explicit is Better than Implicit.

### 4.2 DRY — Don't Repeat Yourself
Avoid duplication of stable business knowledge, rules, constants, and behavior.

Do not extract shared code solely because two blocks look similar. Confirm that they represent the same concept and are expected to change for the same reason.

Prefer small local duplication over a premature, misleading, or tightly coupled abstraction.

### 4.3 Orthogonality
Minimize unnecessary coupling between modules.

A change in one module should affect another module only when a deliberate and explicit contract changes.

Coordinated changes across modules are valid when required by a public interface, schema, domain contract, or architecture boundary.

Do not merge modules merely because one contract change affects both.

### 4.4 ETC — Easy to Change
When facing a design decision, always choose the option that makes future changes easier. Ask: *"If the requirements change tomorrow, how many files do I need to touch?"* Fewer is better.

### 4.5 Design by Contract
The Imperative Shell validates input shape, parsing, external constraints, and
boundary contracts before data enters the core. The Functional Core may enforce
domain invariants and reject impossible domain states through deterministic,
typed errors. Do not duplicate the same validation at both layers without a
specific reason.

```python
# Imperative Shell: Validate at the boundary
def load_strategy_configuration(config_path: Path) -> StrategyConfig:
    """Loads and validates strategy configuration from disk."""
    raw_config = _read_toml_file(config_path)

    if "lookback_period" not in raw_config:
        raise ConfigurationError("Missing required key: 'lookback_period'")
    if raw_config["lookback_period"] <= 0:
        raise ConfigurationError("'lookback_period' must be positive")

    return StrategyConfig(**raw_config)


# Functional Core: Trusts validated data — no defensive checks
def calculate_moving_average(
    prices: list[float],
    lookback_period: int,
) -> list[float]:
    """Pure calculation. Assumes valid inputs (positive lookback, non-empty prices)."""
    return [
        sum(prices[start_index : start_index + lookback_period]) / lookback_period
        for start_index in range(len(prices) - lookback_period + 1)
    ]
```

---

## 5. Error Handling & Logging

- **Strategy:** Distinguish clearly between Critical Errors and Runtime Warnings.
    - **Critical (Raise):** System-level failures (e.g., SQLite DB locked/corrupt). The script must exit.
    - **Warning (Log & Continue):** Data-level anomalies (e.g., missing price for *one* asset). Log these as `logger.warning`.
- **Prohibited:**
    - No bare `except:` clauses.
    - No silent swallowing of errors (`except SomeError: pass`).
    - No `print()` statements. Use `logger`.
- Handle each exception at the layer that can add meaningful context or decide
  recovery. Log an exception once at the operational boundary; avoid duplicate
  logging at multiple layers. When translating exceptions, preserve causality
  with `raise NewError(...) from original_error`.

---

## 6. Libraries & Frameworks

### File System
- **Pathlib Only:** Use `pathlib.Path` for all file system operations. No `os.path`.

### Pandas / Data Processing

- Use Pandas for established repository data-frame pipelines and non-trivial
  tabular transformations.
- Do not introduce Pandas for small scalar, record, or sequence operations that
  are clearer with the standard library.
- Prefer vectorized transformations for columnar calculations.
- Do not use `DataFrame.iterrows()` for transformations.
- Row-wise iteration is permitted only for unavoidable boundary side effects;
  use `itertuples()` and keep the side-effect operation outside domain logic.
- Prefer readable method chains. Split a chain when intermediate naming,
  debugging, or error diagnosis becomes clearer.
- Avoid hidden mutation and chained assignment.

### Database (SQLite)
- **Usage:** Use standard `sqlite3` library with context managers.
- **Safety:** Always use parameterized queries (`?`). SQL injection via f-strings is a **CRITICAL** violation.

### Performance
- Correctness and readability take priority over speculative optimization.
- Do not claim a performance improvement without measurement or a clear complexity reduction.
- Optimize only when required by the task, demonstrated by profiling, or necessary for known EOD data volumes.
- Avoid unnecessary data copies and quadratic algorithms.
- Use generators only when streaming behavior provides an actual memory or pipeline benefit.
- Do not replace readable vectorized Pandas operations with obscure micro-optimizations.

---

## 7. Documentation (Literate Programming)

- **Format:** Google-Style Docstrings.
- **Narrative Approach:** Explain the "Why" and the business logic intent, not just the technical steps. A docstring that only restates the function name is worthless.
- **Requirement:** Every public module, class, and method must be documented.
- **Inline Comments:** Use sparingly. If you need a comment to explain *what* code does, the code is not clear enough. Comments should explain *why* — non-obvious business rules, workarounds, or trade-offs.

---

## 8. Functional Core / Imperative Shell (Detailed Rules)

Separate **pure logic** (deterministic calculations) from **side effects** (I/O, database, network, logging).

### 8.1 Functional Core (The "Inside")
- **Deterministic Functions:** The same inputs produce the same result or the
  same deterministic domain error. Pure core functions do not depend on hidden
  state, time, I/O, logging, network access, or mutable globals.
- **No Side Effects:** No I/O, no database, no logging, no network calls, no `datetime.now()`.
- **Immutable Data:** Input and output via `@dataclass(frozen=True)` or primitive types.
- **Trivially Testable:** Core functions need zero mocks. Test with a simple `assert`.

### 8.2 Imperative Shell (The "Outside")
- **All Side Effects Live Here:** Database access, file I/O, API calls, logging.
- **Thin Orchestration:** The shell loads data, calls the core, and persists results. It contains minimal logic.
- **Validation at the Boundary:** All input validation (Design by Contract) happens in the shell before data enters the core.

### 8.3 Boundary Rule
If a function needs both calculation AND I/O, it is a shell function that delegates the calculation to a core function. Never mix I/O and business logic in the same function.

```python
# ═══════════════════════════════════════
# FUNCTIONAL CORE — Pure, testable
# ═══════════════════════════════════════

from decimal import Decimal


@dataclass(frozen=True)
class RebalanceDecision:
    """Immutable result of a rebalancing calculation."""

    ticker_symbol: str
    target_quantity: int
    current_quantity: int
    action: Literal["BUY", "SELL", "HOLD"]


def determine_rebalancing_actions(
    current_positions: list[Position],
    target_allocation: AllocationMap,
    total_portfolio_value: Decimal,
) -> list[RebalanceDecision]:
    """
    Pure Function: Same inputs → always same result.

    No I/O, no database, no logging.
    Testable with a single assert statement.
    """
    ...


# ═══════════════════════════════════════
# IMPERATIVE SHELL — I/O, orchestration
# ═══════════════════════════════════════


def run_daily_rebalancing(database_path: Path) -> None:
    """
    Shell: Loads data, calls the Functional Core, persists results.

    All side effects are concentrated here.
    """
    positions = load_positions_from_database(database_path)
    allocation = fetch_target_allocation()
    portfolio_value = Decimal(str(sum(p.market_value for p in positions)))

    # ← Call into the Functional Core (pure)
    decisions = determine_rebalancing_actions(positions, allocation, portfolio_value)

    persist_rebalancing_decisions(database_path, decisions)
    logger.info("Rebalancing completed: %d decisions", len(decisions))
```

---

## 9. Measurable Quality Thresholds

These thresholds are enforced in code reviews and CI/CD pipelines.

| Dimension | Metric | Threshold | Enforcement |
|-----------|--------|-----------|-------------|
| Readability | Cognitive complexity | ≤ 15 per function | Sonar-compatible tool when configured; otherwise audit only |
| Readability | Maximum indentation | ≤ 3 levels | Audit |
| Readability | Function size | ≤ 50 logical statements as a review target | Ruff `PLR0915` and audit; not a physical-line guarantee |
| Maintainability | Static typing | No new MyPy errors in changed code | `mypy` |
| Maintainability | Branch coverage of changed functional-core logic | ≥ 90%; target 100% for new business branches | `pytest-cov` when installed |
| Correctness | Cyclomatic complexity | ≤ 10 per function | Ruff `C901` |
| Correctness | Bare `except` | 0 | Ruff `E722` |
| Correctness | Unjustified `type: ignore` | 0 | MyPy and audit |
| Correctness | SQL string interpolation with external values | 0 | Bandit/audit |
| Changeability | Architecture-boundary violations | 0 new violations | Architecture audit |

Do not claim that Ruff measures cognitive complexity, physical function line
count, type-coverage percentage, or dependency depth. A documented target is
not an automated gate unless the configured tool actually measures it.

---

## 10. Example (Clean Code & Step-down Applied)

```python
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TradingStrategyConfiguration:
    """Configuration for technical analysis thresholds."""

    trend_window_size: int
    minimum_required_history: int


def analyze_market_trends(
    closing_prices: list[float],
    configuration: TradingStrategyConfiguration,
) -> list[float]:
    """
    Orchestrates the market trend analysis process.

    This high-level function follows the Step-down Rule by delegating
    the mathematical heavy lifting to specialized pure functions.
    """
    if not _is_history_sufficient(
        closing_prices, configuration.minimum_required_history
    ):
        logger.warning("Insufficient data for trend analysis.")
        return []

    return _calculate_trend_thresholds(closing_prices, configuration.trend_window_size)


def _is_history_sufficient(prices: list[float], minimum_history_length: int) -> bool:
    """Checks if the provided price list meets the required length."""
    return len(prices) >= minimum_history_length


def _calculate_trend_thresholds(prices: list[float], window_size: int) -> list[float]:
    """
    Core mathematical transformation (Functional Core).

    Calculates the threshold values used to identify trend reversals.
    """
    return [
        sum(prices[start_index : start_index + window_size]) / window_size
        for start_index in range(len(prices) - window_size + 1)
    ]
```