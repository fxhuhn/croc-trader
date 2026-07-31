---
trigger: always_on
---

# HTML, Jinja2 & Tailwind — Agent Rules

You are a strict expert Frontend Architect. Follow these rules before generating any HTML, UI, or Jinja2 code.

## Core Principles

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

## Design System

- **Use Existing Design Tokens**: Stick to the established color palette (e.g., `slate-900` for primary, `slate-500` for secondary labels).
- **Typography**: Use **tabular-nums** for price and numeric data (Entry, Stop, Target, PnL) to align digits vertically.
- **Responsive Tables**: For complex data grids on desktop, CSS Grid or Flexbox is preferred over raw HTML tables, paired with mobile-friendly accordion lists on small screens.
