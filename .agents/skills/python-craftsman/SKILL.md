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
from `architecture-specification`. It remains responsible for the actual repository
implementation and verification workflow.

---

## Operational Guidelines

* Must strictly respect `.agents/rules/workspace.md`. Do not reference or operate on files outside the active repository workspace.
* Adhere strictly to `.agents/AGENTS.md` and `.agents/rules/concise.md`.
* Refer exclusively to `.agents/rules/python.md` for coding standards (type hints, naming, complexity, FC/IS, error handling, architecture principles).

---

## Verification Gates

Before reporting an implementation task as complete, apply each gate only when
its applicability condition is met. A non-applicable gate is recorded as
`Not applicable`, not as `Passed`.

### Gate 0: Scope and Diff Verification — mandatory for file changes

- Inspect the final diff.
- Remove unrelated formatting, cleanup, debug output, generated files, and
  scope expansion.
- Map each meaningful change to the explicit task or a directly affected
  contract.

### Gate 1: Formatting and Lint — mandatory for Python changes

Run the smallest relevant scope first. Before completion, run repository-wide
checks only when project policy requires them and distinguish pre-existing
failures from newly introduced failures.

```bash
.venv/bin/ruff format --check <changed-python-paths>
.venv/bin/ruff check <changed-python-paths>
```

Do not modify unrelated files to make repository-wide checks pass.

Broad directory-level Ruff exclusions do not prove that changed code complies
with the excluded rules. When a changed file is covered by such an exclusion,
report which rule families were not effectively enforced.

### Gate 2: Static Typing — mandatory for typed Python changes

```bash
.venv/bin/mypy <changed-python-paths-or-configured-scope>
```

Do not report this gate as passed when MyPy globally suppresses errors or when
the changed modules are excluded. Report the effective checked scope.

A package-wide override such as `app.*` with `ignore_errors = true` means the
application scope is not statically validated. In that state:

- report the gate as `Not validated`,
- do not describe MyPy as strict for application code,
- and identify the suppressing override in the completion report.

A file or module may count as checked only when MyPy analyzes it without an
`ignore_errors = true` override.

### Gate 3: Behavior Verification — required for behavior changes

Use `python-tester` for new logic, behavior changes, and defect corrections.
Run relevant tests first. Run the full suite when available and proportionate.

```bash
.venv/bin/pytest <relevant-test-paths>
.venv/bin/pytest
```

Coverage is reported only when explicitly measured.

### Gate 4: Architecture and Quality Audit — conditional

Use `python-auditor` for non-trivial code changes, architecture-sensitive
changes, or an explicit review request. Documentation-only, comment-only, and
mechanical formatting changes do not require this gate.

### Gate 5: Security and Financial-Integrity Audit — conditional

Use `python-security` when the change affects trust boundaries, external input,
files, paths, subprocesses, network access, persistence, serialization,
secrets, dependencies, monetary calculations, orders, scheduling, or retry and
idempotency behavior. Refer to the full security audit vectors defined in `python-security`.

### Gate 6: Architecture Documentation Sync — conditional

Use `architecture-sync` only for its declared triggers. Do not trigger it for
small public helpers or implementation details without architectural impact.

---

## Unavailable Gate Handling

If a required tool or delegated skill is unavailable:

1. Do not mark the gate as passed.
2. Execute only equivalent checks that are actually available.
3. Report the unavailable gate under `Not validated`.
4. Do not simulate, invent, or infer successful results.

For each applicable gate, report one status:

- `Passed`: executed successfully and inspected.
- `Failed`: executed and failed.
- `Not validated`: required but unavailable or not executable.
- `Not applicable`: applicability condition was not met.

Never use `Passed` for a delegated review that was not actually performed.
