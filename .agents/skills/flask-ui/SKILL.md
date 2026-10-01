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

## Platin-Standards für Quantitative Backtesting- & EoD-UIs

### 1. Quantitatives Zahlen- & Spalten-Alignment
- **Numerische Werte**: Strikte Rechtsbündigkeit (`text-right font-sans font-bold tabular-nums` bzw. `font-medium`) für alle Kurse, P&L-Werte, Notional Values, Kontrakte, Positionsgrößen und Quant-Ratios. Tabellen-Header (`<th>`) numerischer Spalten müssen synchron rechtsbündig (`text-right`) ausgerichtet sein.
- **Text & Datumsspalten**:
  - Text, Ticker, Strategienamen und Status-Labels strikt linksbündig (`text-left`).
  - EoD-Datumsangaben (`entry_date`, `exit_date`, Handelstage) zentriert (`text-center font-sans tabular-nums`) im Format `DD.MM.YYYY`.
- **Feste Rundungs- & Vorzeichenregeln**:
  - **Quant-Ratios**: Sharpe Ratio, Sortino Ratio, Calmar Ratio, Profit Factor exakt mit 2 Dezimalstellen formatieren (z. B. `1,45`).
  - **Prozentuale Performance- & Drawdown-Werte**: Zwingend mit explizitem Vorzeichen (z. B. `+12,40 %` / `-3,15 %`).
  - **Neutrale Werte**: `0,00 %` bzw. `0,00 $` in neutraler Textfarbe (`text-slate-500 dark:text-slate-400`) ohne irreführende Gewinn-/Verlust-Farbe.

### 2. Barrierefreie P&L- und Performance-Semantik (WCAG-Konformität)
- **Verbot reiner Farbkodierung**: Keine ausschließliche Unterscheidung von Gewinn/Verlust über Grün/Rot (Gefahr bei Rot-Grün-Sehschwäche).
- **Mandatorische Vorzeichen- und Badge-Kombination**:
  - Verpflichtende Kombination aus gedämpften Background-Chips und expliziter Vorzeichenführung (`+` / `-`).
  - **Gewinn (Positiv)**: `bg-emerald-500/10 text-emerald-700 dark:text-emerald-400` mit führendem `+`.
  - **Verlust (Negativ)**: `bg-rose-500/10 text-rose-700 dark:text-rose-400` mit führendem `-`.
  - **Neutral**: `bg-slate-100 text-slate-600 dark:bg-slate-800 dark:text-slate-400`.

### 3. Data-Dense Backtest-Tabellen
- **Kompakte Zeilenabstände**: Für tabellarische Trade-Historien, Signal-Logs und Performance-Matrizen gilt maximale Informationsdichte mit `py-1.5 px-2.5 text-xs` für Datenzellen (`<td>`) und Header (`<th>`).
- **Sticky Header Pflicht**: Alle Datentabellen mit potenziell langem Scroll-Inhalt müssen zwingend Sticky Header aufweisen:
  `sticky top-0 bg-slate-50/95 dark:bg-slate-900/95 backdrop-blur z-10 border-b border-slate-200 dark:border-slate-800`
- **Standardisierte Row-Hover-States**: Jede Datenzeile muss eine visuelle Führung beim Überfahren bieten:
  `hover:bg-slate-100/60 dark:hover:bg-slate-800/40 transition-colors`

### 4. Visualisierungs- & Chart-Standards (Equity & Drawdown)
- **Farb-Tokens (Tailwind-Slate-Theme)**:
  - **Hauptstrategie (Equity Curve)**: `emerald-500` (Stroke: `#10b981`) oder `indigo-500` (Stroke: `#6366f1`).
  - **Benchmark / Buy & Hold**: `slate-400` (Stroke: `#94a3b8`, gestrichelt / sekundäre Opazität).
  - **Underwater-/Drawdown-Fläche**: `rose-500/20` (Fill: `rgba(244, 63, 94, 0.2)`) mit Borderline in `rose-500` (Stroke: `#f43f5e`).
- **Achsen-Beschriftungen & Tooltips**:
  - Achsen-Ticks, Achsen-Labels und Tooltip-Zahlenwerte strikt in `font-sans text-xs tabular-nums`.
  - Zahlenwerte in Tooltips müssen deutsches Zahlenformat (`de-DE`) und Währungssuffix verwenden.

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

