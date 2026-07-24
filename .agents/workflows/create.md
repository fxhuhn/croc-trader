---
description: "Run visionary architecture & implementation using the python-creator skill."
trigger: manual
---

# /create Command Workflow

When the user invokes `/create` or requests a new Python module architecture, you must:

1. **Activate the Python Creator Skill**: Load and execute the instructions in [.agents/skills/python-creator/SKILL.md](.agents/skills/python-creator/SKILL.md).
2. **Execute Creation Process**: Follow the 2-phase process (Architectural Blueprint → Implementation) as defined in the skill.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
