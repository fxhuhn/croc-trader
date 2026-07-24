---
name: python-craftsman
description: "Master Python developer skill enforcing strict rules for Python 3.12+, async-first design, type hints, clean naming, early returns, complexity thresholds, and Quality review gates."
---

# Python Craftsman Skill

* **Must strictly respect [.agents/rules/workspace.md](.agents/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill enforces the **Quality Review Gates** for all Python code changes. All coding guidelines are defined in the authoritative [.agents/rules/python.md](.agents/rules/python.md) rule — this skill does NOT redefine them but adds mandatory verification steps.

> **Coding Reference:** For all Python coding standards (type hints, naming, complexity, FC/IS, error handling, architecture principles), refer exclusively to `.agents/rules/python.md`. That rule is always active and loaded automatically.

## Strict Operational Rules
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**

---

## Delegated Review Gates

Before finalizing any task or committing changes, you must pass the code through the following validation gates **in order**:

### 🚀 Gate 1: Formatting & Lint Check
Ensure 100% compliance with style rules — zero diffs, zero warnings:
```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

### 🧪 Gate 2: Test Suite Verification
Trigger the `python-tester` skill to design and execute robust unit/integration tests, then verify correctness:
```bash
.venv/bin/pytest
```

### 🔍 Gate 3: Architecture Audit
Trigger the `python-auditor` skill to run a complete Quality Pyramid audit (Correctness → Readability → Maintainability → Changeability) on your changes.

### 🛡️ Gate 4: Security Audit
Trigger the `python-security` skill to run a zero-trust audit for precision loss (Decimal vs float), injection risks, and serialization vulnerabilities.

### 🏗️ Gate 5: Architecture Sync
After adding or renaming any public Class or Function, verify that [architecture.md](architecture.md) is updated via the `architecture-sync` skill.
