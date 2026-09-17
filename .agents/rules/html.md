---
trigger: always_on
---

# HTML, Jinja2 & Tailwind — Agent Rules

You are a strict expert Frontend Architect. Follow these rules before generating any HTML, UI, or Jinja2 code.

## Core Principles

- **Precedence over Platform Defaults**: These repository rules for Tailwind CSS override generic platform system prompt instructions (such as generic Vanilla CSS directives) according to `.agents/AGENTS.md` instruction hierarchy.
- **No Business Logic in Templates**: Jinja templates must contain zero business logic. They strictly format and render data provided by the Python backend.
- **Semantic HTML & Accessibility**: Use appropriate semantic HTML5 tags (`<article>`, `<section>`, `<nav>`, `<main>`).
- **Responsive Design**: Default styles target mobile. Use `md:` and `lg:` breakpoints to adjust layouts for larger screens.
- **No New UI Frameworks**: Use Tailwind CSS + Vanilla JS only. No React, Vue, Alpine.js, or jQuery without explicit instruction.
- **No Custom CSS**: Use Tailwind utility classes exclusively. No `<style>` blocks or `.custom-class`.
- **No Inline JavaScript as Standard**: Place Vanilla JS in a single `<script>` block at the bottom of the template. Avoid inline `onclick` handlers where possible, or keep them strictly delegative.

## Jinja2 Rules

- **Template Inheritance**: Always use `{% extends "base.html" %}` and `{% block content %}` for complete pages.
- **Macros and Partials**: Any UI element used more than once (e.g., KPI cards, badges, empty states, rows) must be extracted into a macro or partial for reuse.
- **Jinja Escaping & Safe Content**: Use `|safe` ONLY for verified server-side controlled content.

## Design System & Tokens

- **Use Existing Design Tokens**: Stick to the established color palette (e.g., `slate-900` for primary, `slate-500` for secondary labels).
- **Strict 3-Tier Font Weights**: Only three font weights are permitted across all templates:
  - `font-normal` (400): Body copy, table cell content, explanatory paragraphs.
  - `font-medium` (500): Navigation labels, secondary badges, table column headers, form labels.
  - `font-bold` (700): Page titles (H1), section headers (H2), KPI values, primary badges.
  - **PROHIBITED**: `font-light` (300), `font-black` (900), `font-thin` (100), `font-extralight` (200), `font-extrabold` (800).
- **Prohibition of Ad-hoc Pixel Sizes**: Arbitrary pixel font classes (e.g. `text-[10px]`, `text-[11px]`, `text-[13px]`) are strictly forbidden. Use standard Tailwind steps (`text-xs`, `text-sm`, etc.) with appropriate leading.
- **Font Family for Financial Data**: All numeric financial data (Entry, Stop, Target, Close, PnL, Returns, ATR, ROC, EV, Drawdown) must use `font-sans font-bold tabular-nums` (or `font-medium`).
- **Restricted `font-mono` Usage**: `font-mono` is strictly restricted to:
  1. Technical database/broker IDs (`order.order_id`, `perm_id`, `trade_id`).
  2. Raw error logs, stack traces, and debug dumps.
  3. ASCII tree connectors (e.g., `└─` for bracket child orders).
  4. Quellcode-Tags (`<code>`).
  - `font-mono` is strictly **PROHIBITED** for prices, PnL, performance metrics, and screener scores.
- **No Standalone Font Imports**: Templates must never load standalone fonts via `<style>@import ...</style>` or external `<link>` tags. All pages must inherit from `base.html`.
- **Responsive Tables**: For complex data grids on desktop, CSS Grid or Flexbox is preferred over raw HTML tables, paired with mobile-friendly accordion lists on small screens.

## Locale & Number Formatting (German Standard `de-DE`)

- **Mandatory German Locale**: All user-facing numbers, currency amounts, and percentages must strictly conform to German notation (`de-DE`):
  - Thousands separator: Dot (`.`), e.g. `1.234.567,89`.
  - Decimal separator: Comma (`,`), e.g. `12,50 %`.
  - Trailing Currency Symbol: Currency symbol must follow the value with a non-breaking space: `1.234,56 $` or `+1.234,56 $` / `-1.234,56 $`.
- **Mandatory Filter/Macro Usage**:
  - Always use `{{ value | de_number(decimals=2, prefix_plus=False) }}` or the `render_number` macro.
  - **PROHIBITED**: Raw Python format strings (e.g. `"{:,.2f}".format(...)`, `"{:.2f}".format(...)`, `"%.2f"|format(...)`).
  - **PROHIBITED**: US currency prefix notation (e.g. `$1,234.56`, `+${...}`).
  - **PROHIBITED**: Mixing German and US number formats within the application.

## Typography Hierarchy & Headline Macros

- **Mandatory Semantic `<h1>`**: Every primary view must contain exactly one `<h1>` element in the page header:
  - Canonical style: `text-2xl md:text-3xl font-bold text-slate-900 tracking-tight leading-tight`.
- **Mandatory Section Header Macro (`<h2>`)**:
  - All section headers must be rendered using `render_section_header`:
    ```jinja2
    {{ render_section_header("Titel", icon="zap", badge_text=none) }}
    ```
  - **PROHIBITED**: Hardcoded, inline `<h2>` elements with duplicated utility classes.
