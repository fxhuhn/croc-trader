---
name: strategy-screener
description: Define, explain, compare, and audit Croc-Trader screening strategies through canonical playbooks. Use for strategy contracts, indicators, entry and exit conditions, signal schemas, configuration mappings, implementation comparisons, and strategy-specific test requirements. Load only the relevant strategy playbook. Do not generate production signals or place orders.
---

# Strategy Screener Skill

This skill is the definitive navigation and process source for strategies. It serves as the authoritative entry point to strategy playbooks and the primary auditor of strategy code, configuration, and tests.

It is NOT a general Python implementer and NOT a production runner for signal generation.

## Source Hierarchy

For strategy behavior, use:

1. Explicit user-authorized strategy change.
2. Canonical strategy playbook.
3. `architecture.md` and `references/architecture.md`.
4. Current strategy configuration.
5. Existing implementation.
6. Existing tests.
7. Supporting research references.

The implementation is evidence to review. It is not automatically the
normative strategy definition.

Do not fill missing playbook behavior from implementation convenience or
plausible assumptions.

## Playbook Contract

Every supported strategy must have exactly one canonical playbook based on
`playbook/TEMPLATE.md`.

Before describing, implementing, or auditing a strategy:

1. Resolve the canonical strategy identifier from repository evidence.
2. Locate the matching playbook.
3. Load that playbook and only directly relevant architecture, configuration,
   code, tests, and references.
4. Verify whether all mandatory template sections are present.
5. Treat documented playbook behavior as the normative strategy contract.
6. Compare code, configuration, and tests against that contract.
7. Mark missing or contradictory behavior explicitly.
8. Do not invent missing strategy rules.

A strategy is not fully specified while mandatory template sections remain
missing or unresolved.

## Operating Modes

This skill operates in exactly four modes:

### Describe
- Explain the playbook.
- Do not modify files.
- Do not invent strategy parameters.

### Specify
- Design a new or modified playbook using `TEMPLATE.md`.
- Incorporate only verified content.
- Explicitly mark missing business/domain decisions.

### Audit
- Compare the playbook, architecture, configuration, code, and tests.
- Remain strictly read-only.
- Output findings categorized by severity and source.
- Do not automatically apply corrections.

### Implementation Handoff
- When code changes are explicitly requested, pass verified strategy requirements to `python-craftsman`.
- Pass relevant test requirements to `python-tester`.
- Do not execute production implementation directly.

## Forbidden Actions

This skill must NOT:
- Generate production signals.
- Write to `signals.db`.
- Generate broker orders.
- Send Telegram messages.
- Calculate final position sizes (unless architecturally assigned to the screener).
- Derive unknown parameter values from code or backtests.
- Silently align playbooks to match the existing implementation without verification.
