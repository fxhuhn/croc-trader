---
name: architecture-specification
description: Specify and review architecture for Croc-Trader. Use for new components, architecture-significant changes, public contracts, state models, data flows, integration boundaries, or explicit architecture reviews. Produces an evidence-based blueprint and acceptance criteria; it does not implement production code.
---

# Architecture Specification

## Responsibility

Specify or review architecture from verified repository evidence.

This skill may:

- verify requirements,
- identify unresolved business or contract decisions,
- define components and responsibilities,
- define public contracts,
- define state transitions and data flows,
- define failure behavior,
- define acceptance criteria,
- create Mermaid diagrams when they materially improve understanding,
- hand an implementation blueprint to `python-craftsman`.

This skill must not:

- implement production code,
- implement tests,
- add dependencies,
- broaden task scope,
- redesign unrelated components,
- require a ceremonial approval when the task is already unambiguous,
- create a `__main__` demonstration block,
- perform unrelated refactoring.

## Required Inputs

1. Explicit user request.
2. `architecture.md`.
3. `references/architecture.md`.
4. Relevant source code, tests, configuration, and domain skills.

## Workflow

### 1. Evidence Collection

- Identify the requested outcome.
- Inspect current components, contracts, states, and call sites.
- Distinguish verified facts, conflicts, missing contracts, and assumptions.
- Do not ask questions that repository evidence can answer.

### 2. Clarification

Ask only when an unresolved decision changes:

- business behavior,
- financial behavior,
- persistence semantics,
- public interfaces,
- destructive behavior,
- external-system behavior.

Ask the smallest decisive question.

### 3. Specification

Define only the architecture required by the task:

- affected components,
- responsibilities,
- public contracts,
- inputs and outputs,
- state transitions,
- data flow,
- failure behavior,
- security and financial boundaries,
- migration implications,
- acceptance criteria,
- required tests.

### 4. Visualization

Use Mermaid only when a diagram clarifies a non-trivial interaction, state
machine, or data flow.

Do not generate diagrams as a mandatory ritual.

### 5. Handoff

When implementation is requested, provide a concise implementation blueprint
for `python-craftsman`.

Do not implement the blueprint in this skill.

## Output

Use:

- `Verified context`
- `Proposed architecture`
- `Contracts`
- `Data and state flow`
- `Failure behavior`
- `Acceptance criteria`
- `Implementation handoff`
- `Unresolved decisions`

Omit empty sections.
