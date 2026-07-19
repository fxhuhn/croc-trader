---
description: "Run mathematical and architectural audit of trading strategies using the strategy-screener skill."
trigger: "/strategy-audit"
---

# /strategy-audit Command Workflow

When the user invokes `/strategy-audit` or requests a strategy parameters review, you must:

1. **Activate the Strategy-Screener Skill**: Load and execute the instructions in [.agents/skills/strategy-screener/SKILL.md](.agents/skills/strategy-screener/SKILL.md).
2. **Audit Active Strategies**: Cross-reference the active configurations (e.g., `settings.yaml`, playbook files) with their implementations in [app/services/trade_manager/strategies/](app/services/trade_manager/strategies/) and [app/services/screener/strategies/](app/services/screener/strategies/).
3. **Verify Math Invariants**: Ensure Decimal precision rules, risk allocations ($R = 50$), and position sizing bounds are strictly adhered to.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
