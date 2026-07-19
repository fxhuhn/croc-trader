---
description: "Run 3-stage full review pipeline (Auditor -> Quant -> QA Engineer) or localized quality & security checks."
trigger: "/full-review"
---

# /full-review Command Workflow

When the user invokes `/full-review` or requests a full code audit and optimization pipeline, you must:

1. **Activate the Full Review Skill**: Load and execute the instructions in [.agents/skills/full-review/SKILL.md](.agents/skills/full-review/SKILL.md).
2. **Execute Full Audit, Optimization, and Test Suite Generation**: Follow the 3 sequential stages (Auditor -> Quant -> QA Engineer) as defined in the skill.

---

# /audit Command Workflow

When the user invokes `/audit` or requests a localized quality and security audit, you must:

1. **Run Localized CLI Tools**: Execute the following commands sequentially:
   - Vulture (Dead code check): `vulture . --exclude .venv,test`
   - Pip-audit (Vulnerability check): `pip-audit -r requirements.txt`
   - Bandit (Security check): `bandit -r . -x ./.venv,./test`
2. **Report Anomalies**: Report any anomalies in a structured markdown table. If all tools pass without warnings or errors, output a structured table indicating "Status: Clean" for each tool.

---
Format Requirement: Return only repository-relative paths, direct code diffs, or structured markdown tables. No generic text summaries.

