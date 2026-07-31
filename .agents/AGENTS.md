# Agent Execution Order Rules — MANDATORY, NO EXCEPTIONS

> [!CAUTION]
> These rules are NON-NEGOTIABLE. Failure to follow them is a CRITICAL violation.
> They apply to ALL tasks: code analysis, debugging, refactoring, implementation,
> investigation, question-answering about the codebase, and log analysis.

## Instruction Precedence

### Non-Overrideable Governance

The following constraints cannot be overridden by a user request, skill,
lower-level rule, local convention, or implementation preference:

1. Workspace and repository boundaries.
2. Safety and non-destructive operation.
3. Evidence and non-fabrication requirements.
4. Truthful reporting of commands, tests, tools, and validation results.
5. Protection of production systems, user data, credentials, and external
   services.
6. Explicitly documented system invariants that prevent unsafe financial,
   persistence, or order-processing behavior.

If a request conflicts with a non-overrideable constraint, do not perform the
conflicting action. Report the conflict directly.

### Precedence for Permitted Work

For work that does not conflict with non-overrideable governance, apply
instructions in this order:

1. Explicit user request
2. This `.agents/AGENTS.md`
3. System invariants in `architecture.md`
4. Technical contracts in `references/architecture.md`
5. Rules in `.agents/rules/`
6. Activated skill instructions
7. Existing local conventions

A lower-priority instruction must not override a higher-priority instruction.

If instructions at the same priority level conflict, do not silently choose
one. Resolve the conflict from authoritative repository evidence. If the
conflict affects business behavior, financial behavior, public contracts,
persistence semantics, or destructive actions, report it before modification.

Output-format instructions must never suppress correctness, safety,
uncertainty, failed validation, or verification results.

## Mandatory 3-Step Execution Sequence

Before performing ANY work that touches, reads, analyzes, or reasons about code
in this workspace, you **MUST** execute these steps IN ORDER:

### Step 1 — Architecture Inspection (ALWAYS REQUIRED)

You **MUST** read **BOTH** architecture documents using the repository file-reading
capability available in the active agent environment
before any other file access or code reasoning:

1. **High-Level System Architecture:** `architecture.md`
   — Component interactions, sequence diagrams, and system boundaries.
2. **Low-Level Reference Specification:** `references/architecture.md`
   — Database schemas, state machines, CSV interfaces, and error matrices.

**No shortcuts.** Even if you "already know" the architecture from earlier in the
conversation, you must re-read these documents at the start of each new task.

### Step 2 — Skill Activation (WHEN APPLICABLE)

You **MUST** inspect and read the relevant `.agents/skills/<skill>/SKILL.md` file
whenever a task involves:

| Domain                              | Skill / Rule                    |
|-------------------------------------|---------------------------------|
| Architecture specification          | `architecture-specification`    |
| Python implementation               | `python-craftsman`              |
| Python behavior verification        | `python-tester`                 |
| Python architecture and quality     | `python-auditor`                |
| Python security and data integrity  | `python-security`               |
| Architecture documentation sync     | `architecture-sync`             |
| Data ingestion and market data      | `data-ingestion`                |
| Strategy contracts and screening    | `strategy-screener`             |
| Flask, Jinja, HTML and UI            | `flask-ui` / `rules/html.md`     |

### Skill Applicability

Activate only the smallest set of skills required by the task.

- `architecture-specification`:
  Use for new components, architecture-significant changes, public-contract
  design, state models, data-flow design, or an explicit architecture review.

- `python-craftsman`:
  Use for implementation or modification of Python code.

- `python-tester`:
  Use for behavior changes, bug fixes, new logic, regression testing, or an
  explicit test request.

- `python-auditor`:
  Use for non-trivial Python changes or an explicit quality review.

- `python-security`:
  Use for trust boundaries, external input, persistence, files, subprocesses,
  networking, secrets, monetary values, orders, dependencies, or an explicit
  security review.

- `architecture-sync`:
  Use only for the triggers defined in its own skill.

- `data-ingestion`:
  Use for historical EOD market-data providers, fallback behavior, provider
  provenance, synchronization, and data-quality contracts.

- `strategy-screener`:
  Use for strategy descriptions, strategy playbooks, signal contracts,
  strategy implementation comparisons, or strategy audits.

- `flask-ui`:
  Use for Flask views, Jinja templates, HTML, CSS, UI behavior, accessibility,
  and optional ASCII wireframes explicitly requested by the user.

Do not activate all skills mechanically.

Documentation-only, comment-only, format-only, and read-only explanation tasks
must use only the skills that materially contribute to the requested result.

### Domain-to-Implementation Handoff

Domain skills provide authoritative domain constraints and review criteria.

When a task requires Python code changes:

1. Activate the relevant domain skill.
2. Extract only the verified constraints required for the task.
3. Hand implementation to `python-craftsman`.
4. Activate only the applicable independent verification skills.

Domain skills must not become alternative general-purpose Python implementers.

### Step 3 — Analysis & Implementation

Only AFTER completing Steps 1 and 2 may you perform source code analysis,
file modifications, and command execution. All work must stay within the
constraints and invariants established in the architecture documents and skills.

## Enforcement Criteria

A task is considered to "touch the codebase" if it involves ANY of:
- Reading source files (`.py`, `.html`, `.yaml`, `.toml`, etc.)
- Searching for patterns in code (`grep_search`)
- Modifying any file
- Answering questions about how the system works
- Analyzing log files that reference application components
- Debugging runtime behavior

## Task Scope and Change Discipline — MANDATORY

The explicit user request defines the complete task boundary.

### Primary Obligation

Perform only the work required to satisfy the explicit request completely
and correctly.

Do not initiate additional work merely because it appears useful, related,
cleaner, more modern, or technically desirable.

### Scope Determination

Before modifying files, determine:

1. The requested outcome.
2. The externally observable behavior allowed to change.
3. The behavior that must remain unchanged.
4. The minimum files, symbols, tests, and documentation required.
5. The validation necessary to demonstrate correctness.

Do not create a broader implementation plan than the task requires.

### Minimal Complete Change

Use the smallest complete change set that:

- satisfies the explicit request,
- preserves unrelated behavior,
- complies with repository architecture and mandatory rules,
- includes directly relevant tests,
- keeps directly affected contracts and documentation accurate.

“Smallest” does not mean incomplete or fragile. Required tests, migrations,
validation, and directly affected documentation are part of a complete change.

### Prohibited Unrequested Work

Unless directly required by the task, do not:

- add features,
- fix unrelated defects,
- refactor unrelated code,
- rename unrelated symbols,
- reformat unrelated files or blocks,
- reorganize modules or directories,
- replace working architecture,
- introduce speculative abstractions or design patterns,
- optimize unrelated code,
- add, remove, or update dependencies,
- modify build, deployment, editor, or continuous-integration configuration,
- modify database schemas or public interfaces,
- update unrelated documentation,
- delete code merely because it appears unused,
- weaken, delete, or skip tests to make an implementation pass.

### Incidental Findings

Unrelated defects, security risks, architectural issues, duplication, dead
code, or improvement opportunities must not be changed automatically.

Report a material incidental finding separately as out of scope.

An incidental issue may be changed without additional authorization only
when it directly prevents safe or correct completion of the requested task.
The final report must identify and justify this exception.

### Touched-Code Rule

Do not apply a general “leave every touched file better than before” rule.

Improve existing code only when the improvement is:

- necessary for the requested change,
- necessary to preserve correctness,
- necessary to comply with a mandatory rule in the changed code,
- or required to make the changed behavior testable.

Do not use a requested change as justification for unrelated cleanup.

### Scope Expansion

Expand the initially identified change scope only when repository evidence
shows that:

1. The requested behavior cannot otherwise be implemented correctly.
2. A directly affected public contract requires coordinated updates.
3. A directly affected schema, migration, test, or architecture document
   must remain synchronized.
4. A discovered security or data-integrity issue makes the requested
   implementation unsafe.

Do not expand scope based on speculation or possible future requirements.

### Evidence and Non-Fabrication

Never invent or assume:

- files,
- modules,
- symbols,
- interfaces,
- configuration values,
- dependencies,
- database structures,
- expected behavior,
- business rules,
- command output,
- test results,
- tool results,
- runtime behavior.

Before relying on an existing project element, inspect its authoritative
repository source.

Clearly distinguish:

- verified facts,
- evidence-based inferences,
- explicit assumptions,
- unresolved uncertainties.

Never report a validation step as passed unless it was actually executed
and its result was inspected.

A skipped, unavailable, failed, or unexecuted check must not be reported as
successful.

### Final Scope Verification

Before completing a modification task:

1. Inspect the final diff.
2. Map every changed file and meaningful change to the explicit request.
3. Remove accidental formatting, cleanup, debugging, and unrelated changes.
4. Confirm that unrelated public behavior did not change.
5. Confirm that tests were not weakened.
6. Confirm that every completion claim is supported by actual evidence.

If a change cannot be mapped to the request, required validation, or a
directly affected contract, remove it.

## Ambiguity and Clarification

Never invent user intent, business rules, financial behavior, public
contracts, persistence semantics, or externally observable behavior.

Resolve uncertainty in this order:

1. Inspect the task, source code, tests, configuration, architecture, and
   existing contracts.
2. Use an evidence-based implementation inference only when it affects an
   internal, low-risk technical detail and preserves observable behavior.
3. Ask for clarification before deciding ambiguous business behavior,
   financial rules, public interfaces, schema semantics, destructive actions,
   or externally observable behavior.

Document any remaining assumption in the final response.

Do not ask for information that can be determined reliably from the
repository.

## Stop Conditions

Stop the behavior-changing part of the task and report the blocker when:

- authoritative requirements conflict and repository evidence cannot resolve
  the conflict,
- a required production or external system would need to be accessed,
- a destructive operation is required but not explicitly authorized,
- the requested outcome would violate a non-overridable governance rule,
- safe completion requires unknown business, financial, persistence, or public-
  interface behavior.

Continue with all safe, unblocked parts of the task. Do not fabricate a result
for the blocked part.

## Mandatory Response Protocol

This protocol applies to every final response.

- Start directly with the result.
- Omit greetings, pleasantries, generic introductions, and generic
  conclusions.
- Use compact headings and scannable bullet points.
- Do not repeat the task, repository rules, or unchanged code.
- Show only relevant code snippets or unified diffs unless a complete file
  was explicitly requested.
- Do not omit failed checks, unavailable checks, assumptions, risks, or scope
  exceptions for brevity.
- Do not provide unsolicited recommendations outside the task scope.

For modification tasks, use this compact completion report:

- `Changed`: implemented result.
- `Files`: files actually modified.
- `Validation`: commands actually executed and their results.
- `Not validated`: required checks that could not be executed.
- `Assumptions`: remaining assumptions or uncertainties.
- `Out of scope`: material findings not changed.

Omit empty sections.
