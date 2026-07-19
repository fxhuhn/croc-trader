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
| 7 | **NO INLINE REPEATED UI BLOCKS** | Building KPI cards, status badges, or tree-timeline nodes inline when repeated >1 time is strictly forbidden. Use Jinja2 macros (`macros/cards.html`, `macros/timeline.html`). |

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
{% macro render_kpi_card(title, value, icon_name, value_classes="text-slate-900", bg_classes="bg-white border-slate-100", label_classes="text-slate-500", icon_bg_classes="bg-slate-50", icon_text_classes="text-indigo-500", card_id=none, value_id=none, icon_id=none) %}
<article id="{{ card_id }}" class="{{ bg_classes }} rounded-2xl shadow-sm p-4 md:p-6 flex flex-col md:flex-row items-center justify-center md:justify-start gap-1 md:gap-4 border">
    <div class="hidden md:flex w-12 h-12 rounded-xl {{ icon_bg_classes }} items-center justify-center {{ icon_text_classes }}">
        <i data-lucide="{{ icon_name }}" class="w-6 h-6" id="{{ icon_id }}"></i>
    </div>
    <div class="flex flex-col items-center md:items-start text-center md:text-left">
        <p class="text-lg md:text-2xl font-black leading-none mb-1 md:mb-0 md:order-last {{ value_classes }}" id="{{ value_id }}">{{ value }}</p>
        <p class="text-xs font-bold {{ label_classes }} uppercase tracking-wider md:mb-1">{{ title }}</p>
    </div>
</article>
{% endmacro %}
```

### Status Badges & Stammbaum-Timeline Nodes (`macros/timeline.html`)
Use `render_order_badge(status, is_child=False)` and `render_timeline_node(order_or_exec, is_child=False, is_timeline_first=False)` for any multi-leg order status tree or execution log. Never compute `dot_color` or `badge_style` inline in feature templates.

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
