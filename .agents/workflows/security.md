---
description: "Run zero-trust security & penetration audit using the python-security skill."
trigger: manual
---

# /security Command Workflow

When the user invokes `/security` or requests a security audit, you must:

1. **Activate the Python Security Skill**: Load and execute the instructions in [.agents/skills/python-security/SKILL.md](.agents/skills/python-security/SKILL.md).
2. **Execute Security Audit**: Follow the 4-step penetration process (Layer 1 Scan → Layer 3 Scan → Business Logic Review → Penetration Report) as defined in the skill.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
