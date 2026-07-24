---
description: "Run Quality Pyramid audit (Correctness → Readability → Maintainability → Changeability) using the python-auditor skill."
trigger: manual
---

# /auditor Command Workflow

When the user invokes `/auditor` or requests a code quality audit, you must:

1. **Activate the Python Auditor Skill**: Load and execute the instructions in [.agents/skills/python-auditor/SKILL.md](.agents/skills/python-auditor/SKILL.md).
2. **Execute Quality Pyramid Scan**: Follow the 5-step audit process (Correctness → Readability → Maintainability → Changeability → Report) as defined in the skill.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
