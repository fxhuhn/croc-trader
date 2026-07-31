---
name: flask-ui
description: Use for Flask views, Jinja templates, HTML, CSS, UI behavior, accessibility, and optional ASCII wireframes explicitly requested by the user.
---

# Flask UI & Template Agent Skill

* **Must strictly respect [.agents/rules/workspace.md](.agents/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill defines the role, scope, rules, and design guidelines of the specialized **Flask UI & Template Agent**. This agent is responsible for creating, editing, and maintaining the web interface views and ensuring separation of concerns between HTTP routes and core business domains.

## Role & Scope
* **Role**: Frontend Architect & UI/UX Engineer.
* **Scope**: Maintain the web visual layout, dashboard charts, signal tables, and routing controllers:
  - **Controllers**: Flask Blueprints (`app/routes/views/`) managing HTTP parameters, invoking service/repository layers, and passing variables to templates.
  - **Templates**: Jinja2 pages (`app/templates/`) rendering modular, responsive dashboard segments.
  - **Styling**: Follow the design system defined in [.agents/rules/html.md](.agents/rules/html.md).

---

## Strict Template & DRY Rules
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

> **Design System Reference:** All design tokens (colors, typography, status semantics, backgrounds, icons, component patterns, responsive layout rules) are authoritatively defined in [.agents/rules/html.md](.agents/rules/html.md). Follow those rules strictly. This skill focuses on Flask-specific architecture, not design tokens.

1. **Jinja2 Macros (`{% macro %}`):**
   - Reusable visual components (e.g. KPI badges, symbol cards, charts, order status tables, navigation lists) must be implemented inside reusable macros (such as [app/templates/macros/cards.html](app/templates/macros/cards.html) or [app/templates/macros/navigation.html](app/templates/macros/navigation.html)).
   - Do **NOT** duplicate HTML code blocks for signals or trades lists. Always invoke the corresponding macro.
2. **Jinja2 Template Inheritance:**
   - Every layout file must inherit from [app/templates/base.html](app/templates/base.html) to share unified metadata, favicon links, Lucide icon libraries, and configuration settings.
3. **Partial Views:**
   - Use `{% include %}` for global snippets (e.g. `mobile_nav.html` or custom JS analytics blocks) to keep views compact.

---

## Context Isolation Invariants

- **Separation of Concerns**: Controllers/routes must **NOT** contain raw SQL, state updates, or quantitative mathematics logic.
- **Service Dependency Injection**: Routes must obtain instances of database repositories (`SignalRepository`, `TradeRepository`) or services (`ScreenerViewService`) via the helper functions defined in [app/routes/views/dependencies.py](app/routes/views/dependencies.py).
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
