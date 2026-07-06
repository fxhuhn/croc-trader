## trigger: always_on

# HTML, Jinja2 & Tailwind — Agent Rules for EOD Trading & SaaS Dashboards

You are a strict expert Frontend Architect. These rules are **always active** (system-level constraints). The agent must follow them before generating any code, plan, or suggestion.

---

## 🚨 ZERO TOLERANCE — HARD BLOCKS (Read before every generation)

| # | Rule | Forbidden Pattern |
|---|------|-------------------|
| 1 | **NO LAYOUT DOM DUPLICATION** | Duplicating DOM for simple layout tweaks (use Tailwind responsive classes instead). *Note: Separating mobile lists (`md:hidden`) and desktop grid tables (`hidden md:block`) for complex tabular data is permitted to keep code clean.* |
| 2 | **NO HTML TABLES** | `<table>`, `<thead>`, `<tbody>`, `<tr>`, `<td>` — use CSS Grid or Flexbox exclusively to construct desktop tables and data grids. |
| 3 | **NO MICRO-TYPOGRAPHY** | `text-[8px]`, `text-[9px]`, `text-[10px]` — general minimum is `text-xs` (12px). *Note: `text-[11px]` is allowed exclusively for inline sub-rows or secondary metadata in dense desktop grids.* |
| 4 | **NO EXTERNAL FRAMEWORKS** | Bootstrap, jQuery, Alpine.js, Vue, React — Tailwind CSS + Vanilla JS only |
| 5 | **NO CUSTOM CSS** | `<style>` blocks, `.custom-class`, `@keyframes` — 100% Tailwind utility classes |
| 6 | **NO UPPERCASE/UNDERSCORE LABELS** | `"TIME_STOP"`, `"Target_Hit"` — use plain Title Case: `"Time Stop"`, `"Target Hit"` |

> These blocks are non-negotiable. If a user request conflicts with any rule above, refuse and explain.

---

## 1. Architecture & Layout

- **Mobile-First Responsive Layouts:** Default styles target mobile. Use `md:` and `lg:` breakpoints to adjust flex direction or grid columns. For complex tabular data lists (e.g., active positions or trade history tables), it is recommended to split the section into a mobile-specific accordion/list block (`md:hidden`) and a desktop-specific CSS grid table block (`hidden md:block`) to reduce formatting/spacing complexity.
- **CSS Grid/Flexbox for Data Grids:** Construct all data grids and tables using Tailwind CSS Grid (`grid grid-cols-X`) or Flexbox instead of semantic HTML `<table>` tags.
- **Card & Accordion UI:** Wrap mobile data rows in `bg-white rounded-2xl shadow-sm border border-slate-200` cards. Use accordions (hidden by default) for secondary details.
- **Dumb Templates:** Jinja/HTML contains zero business logic. It only formats and renders data provided by the Python backend.

---

## 2. Design System — Tokens & Hierarchy

### Backgrounds
- Page: `bg-slate-50`
- Cards & Panels: `bg-white`
- Expanded / Subtle: `bg-slate-50/30`

### Typography
- Font stack: **Inter** (sans) for all UI text; **tabular-nums** strictly for price and numeric data (Entry, Stop, Target, PnL) to align digits vertically without resorting to monospace fonts.
- **Primary:** `text-slate-900` — Symbols, main values, active nav
- **Secondary:** `text-slate-500` — Labels (Qty, Date, Entry), timestamps. **Never** use `slate-400`, `slate-600`, or `slate-700` here.
- **Tertiary:** `text-slate-400` / `text-slate-300` — Icons, subtle borders, empty states
- **Minimum size:** `text-xs` (12px). Standard labels: `text-sm` (14px).
- **Font weights:** `font-bold` for PnL values in lists. `font-black` reserved exclusively for large KPI cards.

### Financial Semantics (Strict)
| State | Value color | Label / badge color | Card background |
|-------|------------|---------------------|-----------------|
| Profit / Win | `text-emerald-500` | `text-emerald-600` | `bg-emerald-500` gradient |
| Loss / Negative | `text-rose-500` | `text-rose-600` | `bg-rose-500` |
| Warning | `text-amber-500` | `text-amber-600` | `bg-amber-50` |

### Win/Loss Ratio Format
```html
<span class="text-emerald-600 font-bold">13W</span>
<span class="text-slate-300 mx-1">—</span>
<span class="text-rose-600 font-bold">3L</span>
```

---

## 3. Component Patterns

### KPI Cards (Jinja Macro)
```jinja
{% macro kpi_card(title, value, is_currency=False, is_pnl=False) %}
{% set bg = 'bg-emerald-500 text-white' if is_pnl and value >= 0
            else ('bg-rose-500 text-white' if is_pnl and value < 0
            else 'bg-white text-slate-900 border border-slate-200') %}
{% set label_color = 'text-emerald-100' if is_pnl and value >= 0
                     else ('text-rose-100' if is_pnl and value < 0
                     else 'text-slate-500') %}
<article class="{{ bg }} rounded-2xl shadow-sm p-4 flex flex-col justify-center">
    <div class="text-2xl font-black mb-1">
        {{ "{:,.2f} $".format(value) if is_currency else value }}
    </div>
    <div class="{{ label_color }} text-xs uppercase font-semibold tracking-wide">{{ title }}</div>
</article>
{% endmacro %}
```

### Status Badges
- **History tags:** `px-2 py-0.5 rounded-md text-xs font-medium border border-slate-200 text-slate-600 bg-slate-50`
- **Strategy variants:** `px-2 py-0.5 rounded-md text-xs font-bold bg-indigo-50 text-indigo-600` — **never** uppercase

### Accordion Trade Row (Jinja Shared Macro)

Always use the centralized `trade_accordion_card` macro from `macros/cards.html` to render trade list accordions:

```html
{% call cards.trade_accordion_card(
    symbol=trade.symbol,
    badges=[{'text': trade.variant}],
    pnl_value=trade.pnl,
    pnl_pct=trade.pnl_pct,
    show_currency=false
) %}
    <!-- Strategy-Specific Accordion Details (Inside macro call) -->
    <div class="grid grid-cols-2 gap-8 items-start">
        <div class="flex flex-col">
            <div class="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span class="text-slate-500 font-medium text-xs">Date</span>
                <span class="text-slate-700 font-bold text-xs">{{ trade.exit_date }}</span>
            </div>
            <div class="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span class="text-slate-500 font-medium text-xs">Held</span>
                <span class="text-slate-700 font-bold text-xs">{{ trade.days_held }}d</span>
            </div>
        </div>
        <div class="flex flex-col">
            <div class="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span class="text-slate-500 font-medium text-xs">Entry</span>
                <span class="tabular-nums text-slate-700 font-bold text-xs">{{ trade.entry }}</span>
            </div>
            <div class="flex justify-between items-center py-1.5 border-b border-slate-100">
                <span class="text-slate-500 font-medium text-xs">Target</span>
                <span class="tabular-nums text-slate-700 font-bold text-xs">{{ trade.target }}</span>
            </div>
        </div>
    </div>
    <!-- Reusable TradingView Button inside details only -->
    <a href="{{ trade.chart_url }}" target="_blank"
       class="w-full flex items-center justify-center gap-2 py-2.5 mt-4 rounded-xl bg-slate-900 text-white text-sm font-semibold">
        <i data-lucide="bar-chart-2" class="w-4 h-4"></i> View on TradingView
    </a>
{% endcall %}
```

---

## 4. Jinja2 Rules

- **Template Inheritance:** Always use `{% extends "base.html" %}` + `{% block content %}`.
- **Macros:** Any UI element used more than once (KPI cards, badges, empty states) **must** be a macro.
- **Safe Rendering:** Always handle empty arrays with `{% else %}` producing a styled empty state. Use `context.get('key', 'default')` to prevent `KeyError`.
- **Formatting:** Currency/percentage formatting inline in Jinja:
  ```jinja
  {{ "{:+,.2f}".format(value) }}   {# signed float #}
  {{ "{:,.0f}".format(volume) }}   {# integer with thousands separator #}
  {{ "{:.1%}".format(ratio) }}     {# percentage #}
  ```

---

## 5. JavaScript

- **Vanilla JS only.** No React, Vue, jQuery, Alpine.js.
- **Event delegation via `this`:** Pass `this` in inline `onclick` handlers.
- **Single `<script>` block** at the bottom of the template.
- **Accordion toggle pattern:**
```javascript
function toggleTradeCard(header) {
    const details = header.nextElementSibling;
    const chevron = header.querySelector('.chevron-icon');
    const isOpen = !details.classList.contains('hidden');
    details.classList.toggle('hidden', isOpen);
    chevron.style.transform = isOpen ? '' : 'rotate(180deg)';
}
```

---

## 6. Icons

- **Lucide only** via `data-lucide` attribute: `<i data-lucide="trending-up" class="w-4 h-4"></i>`
- **No raw inline SVG paths** unless there is no Lucide equivalent.
- Initialize at bottom of page: `<script>lucide.createIcons();</script>`

---

## 7. Antigravity-Specific Behaviour

These rules apply as **passive, always-on constraints** injected into every agent task in this workspace:

- Before generating any HTML, verify compliance with all 6 Zero Tolerance rules.
- When the user requests a data table, automatically convert it to a flexbox list structure.
- When the user requests a font below `text-xs`, refuse and use `text-xs` instead.
- When the user requests a `<style>` block, refuse and suggest the equivalent Tailwind classes.
- Status tags must always pass through `| title | replace('_', ' ')` in Jinja before rendering.
- All price/numeric data columns must use `tabular-nums`.
- All key grid data columns (Date, Symbol, Signals, Quantity, Entry, Stop, Targets, PnL) must use `whitespace-nowrap` to prevent wrap-induced line breaks.
- Write Jinja tags and expressions on a single line inside inline/flex-item elements (like badges/spans) to prevent unwanted spacing or breaks.
- External links (TradingView, broker links) must always be placed inside the **expanded accordion area**, never in the clickable header row (fat-finger protection).
