---
description: "Run specification-first architect design workflow with Mermaid visualization."
trigger: "/architect-design"
---

# /architect-design Command Workflow

When the user invokes `/architect-design` or requests a specification-first design with Mermaid visualizations, you must:

1. **Activate the Architect Design Skill**: Load and execute the instructions in [.agent/skills/architect-design/SKILL.md](.agent/skills/architect-design/SKILL.md).
2. **Execute Specification & Visualization**: Follow the 3 sequential stages (Analyst -> Architect -> Developer) as defined in the skill.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.
