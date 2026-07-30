---
name: python-craftsman
description: "Implementation skill for synchronous Python 3.12+ End-of-Day trading systems, enforcing repository rules, minimal task scope, strict typing, clear naming, controlled complexity, and mandatory verification gates."
---

# Python Craftsman Skill

## Responsibility

This skill implements approved Python changes.

It does not independently broaden requirements, redesign unrelated
architecture, or perform unrelated cleanup.

For new or architecturally significant solutions, it may consume a blueprint
from `python-creator`. It remains responsible for the actual repository
implementation and verification workflow.

---

## Operational Guidelines

* Must strictly respect `.agents/rules/workspace.md`. Do not reference or operate on files outside the active repository workspace.
* Adhere strictly to `.agents/AGENTS.md` and `.agents/rules/concise.md`.
* Refer exclusively to `.agents/rules/python.md` for coding standards (type hints, naming, complexity, FC/IS, error handling, architecture principles).

---

## Delegated Review Gates

Before finalizing any task or committing changes, pass the code through the following validation gates **in order**:

### 🚀 Gate 1: Formatting & Lint Check

Ensure 100% compliance with style rules — zero diffs, zero warnings:

```bash
.venv/bin/ruff format --check .
.venv/bin/ruff check .
```

Run checks against the smallest relevant scope first. Run repository-wide
checks before completion when required by project policy.

Do not modify unrelated files solely to make a repository-wide check pass.
Report pre-existing unrelated failures separately.

### 🧪 Gate 2: Behavior Verification

Use `python-tester` to design or review tests for the changed behavior.

Run the relevant tests first, followed by the full repository test suite when
available:

```bash
.venv/bin/pytest <relevant-test-paths>
.venv/bin/pytest
```

Do not report the gate as passed if tests were unavailable, skipped, failed,
or not executed.

### 🔍 Gate 3: Architecture Audit

Trigger the `python-auditor` skill to run a Quality Pyramid audit (Correctness → Readability → Maintainability → Changeability) on your changes.

Auditor findings outside the explicit task scope are reported but not automatically remediated. A finding becomes blocking only when it was introduced by the current change, directly affects the requested behavior, violates a mandatory contract in changed code, or makes the requested implementation unsafe.

### 🛡️ Gate 4: Security Audit

Trigger the `python-security` skill to run an audit for precision loss (Decimal vs float), injection risks, and serialization vulnerabilities.

Security findings outside the explicit task scope are reported but not automatically remediated. A finding becomes blocking only when it was introduced by the current change, directly affects the requested behavior, violates a mandatory contract in changed code, or makes the requested implementation unsafe.

### 🏗️ Gate 5: Architecture Documentation

Trigger `architecture-sync` only when the change affects an
architecture-relevant public component, module boundary, data flow, external
interface, database schema, or documented contract.

Do not add every public helper function to `architecture.md`.

---

## Unavailable Gate Handling

If a required tool or delegated skill is unavailable:

1. Do not mark the gate as passed.
2. Execute only equivalent checks that are actually available.
3. Report the unavailable gate under `Not validated`.
4. Do not simulate, invent, or infer successful results.
