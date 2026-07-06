---
description: "Run standard 3-stage architect workflow (Interrogator -> Blueprint -> Builder) for requirements and coding."
trigger: "/architect"
---

# /architect Command Workflow

When the user invokes `/architect` or requests a technical specification/implementation flow, you must:

1. **Activate the Architect Workflow Skill**: Load and execute the instructions in [.agent/skills/architect-workflow/SKILL.md](.agent/skills/architect-workflow/SKILL.md).
2. **Execute Requirements & Specification**: Follow the 3 sequential stages (Interrogator -> Blueprint -> Builder) as defined in the skill.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
