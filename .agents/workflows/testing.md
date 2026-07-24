---
description: "Run destructive pytest suite generation using the python-tester skill."
trigger: manual
---

# /testing Command Workflow

When the user invokes `/testing` or requests aggressive test suite creation, you must:

1. **Activate the Python Tester Skill**: Load and execute the instructions in [.agents/skills/python-tester/SKILL.md](.agents/skills/python-tester/SKILL.md).
2. **Execute Testing Protocol**: Follow the 2-step process (Attack Plan → Generate Test Code) as defined in the skill.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
