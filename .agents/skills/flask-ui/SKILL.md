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

> **Design System Reference:** All design tokens (colors, typography, status semantics, backgrounds, icons, component patterns, responsive layout rules, German locale standard `de-DE`, 3-tier font weights, and headline macros) are authoritatively defined in `.agents/rules/html.md`. Follow those rules strictly. Templates must pass static lint validation (`test/unit/templates/test_template_typography_lint.py`).

1. **German Locale Standard (`de-DE`):**
   - Format all numbers and currency amounts using German conventions (e.g. `{{ value | de_number }}` -> `1.234,56 $`).
   - Never use US raw format strings (`"{:,.2f}"`) or US currency prefix notation (`$1,234.56`).
2. **Typography & Headline Macros:**
   - Every primary page view must have exactly one `<h1>` (`text-2xl md:text-3xl font-bold text-slate-900 tracking-tight`).
   - Section headers (`<h2>`) must strictly invoke the `render_section_header` macro from `macros/cards.html`. Never write inline `<h2>` blocks.
   - Use only `font-normal`, `font-medium`, and `font-bold`. Never use `font-light` or `font-black`.
   - Never use arbitrary pixel classes (e.g. `text-[10px]`, `text-[11px]`). Use Tailwind-native sizes (`text-xs`).
   - Financial data and prices must use `font-sans font-bold tabular-nums`. Reserve `font-mono` exclusively for database IDs, logs, code, and ASCII tree connectors.
3. **Jinja2 Macros (`{% macro %}`):**
   - Reusable visual components (e.g. KPI cards, section headers, badges, symbol cards, order status tables, navigation lists) must be implemented inside reusable macros (such as `app/templates/macros/cards.html` or `app/templates/macros/navigation.html`).
   - Do **NOT** duplicate HTML code blocks for signals or trades lists. Always invoke the corresponding macro.
4. **Jinja2 Template Inheritance:**
   - Every layout file must inherit from `app/templates/base.html` to share unified metadata, favicon links, Lucide icon libraries, and configuration settings. Standalone `@import` font loading is forbidden.
5. **Partial Views:**
   - Use `{% include %}` for global snippets (e.g. `mobile_nav.html` or custom JS analytics blocks) to keep views compact.

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

### 2. Accessible PnL & Performance Semantics (WCAG Compliance)
- **Prohibition of Pure Color Coding**: Never differentiate profit and loss solely by green/red text colors (red-green color vision deficiency).
- **Mandatory Sign & Badge Pill Combination**:
  - Mandatory pairing of muted background chips and explicit sign prefixes (`+` / `-`).
  - **Profit (Positive)**: `bg-emerald-500/10 text-emerald-700 dark:text-emerald-400` with leading `+`.
  - **Loss (Negative)**: `bg-rose-500/10 text-rose-700 dark:text-rose-400` with leading `-`.
  - **Neutral**: `bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400`.

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

