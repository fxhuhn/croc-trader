---
name: flask-ui
description: "The absolute guardian of the presentation layer, managing Flask Blueprints, Jinja2 templates, and Tailwind CSS layouts."
---

# Flask UI & Template Agent Skill

* **Must strictly respect [.agent/rules/workspace.md](.agent/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill defines the role, scope, rules, and design guidelines of the specialized **Flask UI & Template Agent**. This agent is responsible for creating, editing, and maintaining the web interface views, styling layouts, and ensuring separation of concerns between HTTP routes and core business domains.

## Role & Scope
* **Role**: Frontend Architect & UI/UX Engineer.
* **Scope**: Maintain the web visual layout, dashboard charts, signal tables, and routing controllers:
  - **Controllers**: Flask Blueprints (`app/routes/views/`) managing HTTP parameters, invoking service/repository layers, and passing variables to templates.
  - **Templates**: Jinja2 pages (`app/templates/`) rendering modular, responsive dashboard segments.
  - **Styling**: Tailwind CSS classes coupled with Lucide icons to present a premium SaaS layout.

---

## Strict Template & DRY Rules
* **Strictly adhere to `.agent/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

1. **Jinja2 Macros (`{% macro %}`)**:
   - Reusable visual components (e.g. KPI badges, symbol cards, charts, order status tables, navigation lists) must be implemented inside reusable macros (such as [app/templates/macros/cards.html](app/templates/macros/cards.html) or [app/templates/macros/navigation.html](app/templates/macros/navigation.html)).
   - Do **NOT** duplicate HTML code blocks for signals or trades lists. Always invoke the corresponding macro.
2. **Jinja2 Template Inheritance**:
   - Every layout file must inherit from [app/templates/base.html](app/templates/base.html) to share unified metadata, favicon links, Lucide icon libraries, and configuration settings.
3. **Partial Views**:
   - Use `{% include %}` for global snippets (e.g. `mobile_nav.html` or custom JS analytics blocks) to keep views compact.

---

## Design System & Color Palette

The interface is configured to deliver a premium, high-fidelity quantitative dashboard experience. Maintain these styling conventions:

* **Backgrounds & Containers**:
  - Main background: Slate Slate-50 (`bg-slate-50`) or Dark Gray modes.
  - Cards & panels: White (`bg-white`), borders (`border-slate-100` or `border-slate-200`), rounded corners (`rounded-2xl` or `rounded-3xl`), and soft shadows (`shadow-soft` or `shadow-saas`).
* **Typography**:
  - Font Family: `Inter` loaded via Google Fonts. Use font weights like `font-black` for headers/KPIs, and `font-mono` for numerical lists.
* **Color Accents**:
  - Positive PnL / Success: Emerald (`text-emerald-500`, `bg-emerald-50`).
  - Negative PnL / Danger: Rose (`text-rose-500`, `bg-rose-50`).
  - Highlights / Targets: Blue / Indigo (`text-blue-600`, `text-indigo-500`).
* **Icons**:
  - Use `lucide` icons strictly using the `<i data-lucide="...">` markup, triggering `lucide.createIcons()` on DOM load.

---

## Context Isolation Invariants

- **Separation of Concerns**: Controllers/routes must **NOT** contain raw SQL, state updates, or quantitative mathematics logic.
- **Service Dependency Injection**: Routes must obtain instances of database repositories (`SignalRepository`, `TradeRepository`) or services (`ScreenerViewService`) via the helper functions defined in [app/routes/views/dependencies.py](app/routes/views/dependencies.py).
- **Template Contexts**: Route responses should limit variables strictly to primitives, dataclasses, or structured dict mappings intended for UI injection.
- All references and imports must be repository-relative.
