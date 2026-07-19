---
description: "Run screening scan and write signals to signals.db using the strategy-screener skill."
trigger: "/generate-signals"
---

# /generate-signals Command Workflow

When the user invokes `/generate-signals` or triggers the daily screening run, you must:

1. **Activate the Strategy-Screener Skill**: Load and execute the instructions in [.agents/skills/strategy-screener/SKILL.md](.agents/skills/strategy-screener/SKILL.md).
2. **Execute Screener Engine**: Run the screening scan across active symbol lists.
3. **Log Proposals**: Write generated setup price limits and contexts to the `signals.db` signal repository.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
