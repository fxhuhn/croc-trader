---
name: flask-ui
description: Use for Flask views, Jinja templates, HTML, CSS, UI behavior, accessibility, and optional ASCII wireframes explicitly requested by the user.
---

# Flask UI & Template Agent Skill

This skill defines the role, scope, rules, and design guidelines of the specialized **Flask UI & Template Agent**. This agent is responsible for creating, editing, and maintaining the web interface views and ensuring separation of concerns between HTTP routes and core business domains.

## Role & Scope
* **Role**: Frontend Architect & UI/UX Engineer.
* **Scope**: Maintain the web visual layout, dashboard charts, signal tables, and routing controllers:
  - **Controllers**: Flask Blueprints (`app/routes/views/`) managing HTTP parameters, invoking service/repository layers, and passing variables to templates.
  - **Templates**: Jinja2 pages (`app/templates/`) rendering modular, responsive dashboard segments.
  - **Styling**: Follow the design system defined in `.agents/rules/html.md`.

---

## Strict Template & DRY Rules
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

> **Design System Reference:** Load and strictly enforce `.agents/rules/html.md` for all design tokens, 3-tier font weights, `font-mono` restrictions, `de-DE` formatting (`de_number`), template inheritance (`base.html`), and macros (`render_section_header` in `macros/cards.html`, `macros/navigation.html`). Templates must pass `test/unit/templates/test_template_typography_lint.py`.

---

## Platinum Standards for Quantitative Backtesting & EoD UIs

### 1. Quantitative Number & Column Alignment
- **Numerical Values**: Strict right-alignment (`text-right font-sans font-bold tabular-nums` or `font-medium`) for all market quotes, PnL values, notional amounts, contract sizes, position quantities, and quant ratios. Table headers (`<th>`) for numerical columns must be synchronously right-aligned (`text-right`).
- **Text & Date Columns**:
  - Text, tickers, strategy designations, and status badges must be strictly left-aligned (`text-left`).
  - EoD trade dates (`entry_date`, `exit_date`, trading days) must be centered (`text-center font-sans tabular-nums`) in `DD.MM.YYYY` format.
- **Rounding & Sign Conventions**:
  - **Quant Ratios**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor must be formatted with exactly 2 decimal places (e.g. `1,45`).
  - **Percentage Performance & Drawdown Values**: Mandatory explicit sign prefix (e.g. `+12,40 %` / `-3,15 %`).
  - **Neutral Values**: `0,00 %` or `0,00 $` must render in neutral text color (`text-slate-500 dark:text-slate-400`) without misleading gain/loss coloration.

### 2. PnL & Performance Semantics (Clean Text-Only & WCAG Sign Rules)
- **Strict Prohibition of Background Chips / Badges**: Numerical PnL figures, percentage returns, and slippage amounts must **NEVER** use background chips, pills, or highlight boxes (such as `bg-emerald-500/10`, `bg-rose-500/10`, `rounded-full`). These create visual irritation, clutter, and optical vibration in dense financial tables.
- **Clean Text-Only Presentation**: All financial PnL metrics in tables, cards, and summary footers must be rendered as clean, unboxed text (`font-sans font-bold tabular-nums text-xs` or `text-sm`).
- **Unambiguous Sign & 3-Way Neutrality**:
  - Differentiate profit and loss through explicit sign prefixes (`+` / `-`) combined with semantic text colors:
  - **Profit (Positive)**: `text-emerald-600 dark:text-emerald-400 font-sans font-bold tabular-nums` with explicit leading `+` (e.g. `+1.234,56 $` / `+12,40 %`).
  - **Loss (Negative)**: `text-rose-600 dark:text-rose-400 font-sans font-bold tabular-nums` with explicit leading `-` (e.g. `-450,20 $` / `-3,15 %`).
  - **Neutral (Zero / Break-Even)**: `0,00 %` or `0,00 $` must render strictly in `text-slate-500 dark:text-slate-400 font-sans font-bold tabular-nums` without green/red tinting or `+` sign.

### 3. Data-Dense Backtest Tables
- **Compact Row Spacing**: High information density for tabular trade histories, signal logs, and performance matrices using `py-1.5 px-2.5 text-xs` for data cells (`<td>`) and header cells (`<th>`).
- **Mandatory Sticky Headers**: All data tables with scrollable content must enforce sticky headers:
  `sticky top-0 bg-slate-50/95 dark:bg-slate-900/95 backdrop-blur z-10 border-b border-slate-200 dark:border-slate-800`
- **Standardized Row Hover States**: Every data row must provide visual guidance on hover:
  `hover:bg-slate-100/60 dark:hover:bg-slate-800/40 transition-colors`

### 4. Visualization & Chart Standards (Equity & Drawdown)
- **Color Tokens (Tailwind Slate Theme)**:
  - **Primary Strategy (Equity Curve)**: `emerald-500` (Stroke: `#10b981`) or `indigo-500` (Stroke: `#6366f1`).
  - **Benchmark / Buy & Hold**: `slate-400` (Stroke: `#94a3b8`, dashed line / secondary opacity).
  - **Underwater / Drawdown Fill**: `rose-500/20` (Fill: `rgba(244, 63, 94, 0.2)`) with border stroke in `rose-500` (`#f43f5e`).
- **Axis Labels & Tooltips**:
  - Axis ticks, axis labels, and tooltip numerical metrics must strictly use `font-sans text-xs tabular-nums` (strictly no `font-mono`/`monospace`).
  - Tooltip values must use German locale formatting (`de-DE`) with currency suffix.

---

## Context Isolation Invariants

- **Separation of Concerns**: Controllers/routes must **NOT** contain raw SQL, state updates, or quantitative mathematics logic.
- **Service Dependency Injection**: Routes must obtain instances of database repositories (`SignalRepository`, `TradeRepository`) or services (`ScreenerViewService`) via the helper functions defined in `app/routes/views/dependencies.py`.
- **Template Contexts**: Route responses should limit variables strictly to primitives, dataclasses, or structured dict mappings intended for UI injection.
- All references and imports must be repository-relative.

## ASCII Wireframe Mode

Use this mode only when the user explicitly requests a mockup, wireframe, or
layout proposal.

In this mode:

- remain read-only,
- create no files unless explicitly requested,
- produce no backend changes,
- use simple monospaced ASCII layouts,
- show hierarchy, content regions, actions, and responsive alternatives,
- do not select new UI frameworks,
- do not present the wireframe as implemented behavior.

