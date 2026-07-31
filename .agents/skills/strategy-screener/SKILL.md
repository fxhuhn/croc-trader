---
name: strategy-screener
description: Define, explain, compare, and audit Croc-Trader screening strategies through canonical playbooks. Use for strategy contracts, Screener and Trade Manager stages, indicators, entry and exit conditions, signal schemas, persisted runtime state, configuration mappings, implementation comparisons, and strategy-specific test requirements. Load only the relevant playbook. Do not generate production signals or place orders.
---

# Strategy Screener

## Responsibility

This skill manages the documentation and validation of strategy playbooks, which serve as the canonical contract bridging financial design and technical execution. The strategy screener is responsible for explaining, designing, or auditing strategies within the Croc-Trader architecture.

## Source Hierarchy

In the event of conflicting information, the following hierarchy applies strictly:

1. Explicit current user request, subject to non-overrideable governance
2. `.agents/AGENTS.md`
3. System invariants in `architecture.md`
4. Technical contracts in `references/architecture.md`
5. Applicable rules in `.agents/rules/`
6. Canonical Strategy Playbook for strategy-specific business rules
7. Current active configuration
8. Existing implementation
9. Existing tests
10. Supporting research or external descriptions


A strategy playbook is normative for strategy-specific indicators, conditions, timing, and business decisions only within the global architecture and persistence contracts. It must not override system-wide responsibilities, database schemas, state invariants, or CSV interfaces.

*Important Constraints*:
- Existing code is evidence, not automatically the normative contract.
- Missing strategy information must not be derived simply for implementation convenience.
- Contradictions must be visibly documented.
- If a normative contract value is missing, use `UNSPECIFIED`. If repository evidence has not been checked, use `NOT_REVIEWED`. Do not use prose variants such as `Not specified` or `Not verified`.

## Playbook Selection

Before working on a strategy:

1. Resolve the canonical strategy identifier from repository evidence.
2. Locate exactly one matching playbook.
3. Load `overview.md`.
4. Load only the selected strategy playbook.
5. Load `TEMPLATE.md` only when creating, migrating, or validating a playbook.
6. Load only directly relevant architecture, configuration, code, tests, and references.
7. Do not load all strategy playbooks unless the task explicitly requires a cross-strategy comparison.

## Operating Modes

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

### Migrate
- Structurally move an existing playbook into the current template format.
- Do not add missing domain logic.
- Mark missing normative values with `UNSPECIFIED`; mark unreviewed implementation, configuration, or test evidence with `NOT_REVIEWED`.

## Playbook Contract

All strategy playbooks must adhere to `TEMPLATE.md` and integrate flawlessly with the shared stage model outlined in `overview.md`.

## Mermaid and Visual Model Rules

- Playbooks must include at least one valid, renderable Mermaid diagram.
- `flowchart LR`, `flowchart TD`, or `stateDiagram-v2` are permitted.
- The diagram must reflect the data ingestion, screener calculation, setup decision, `CREATED` status transition, Trade Manager entry, `ACTIVE`/`INVALIDATED` states, runtime updates, and `CLOSED` states relevant to the strategy.
- Do not duplicate all shared components from `overview.md` if they do not change. Focus on strategy-specific logic.

## Implementation Handoff

- Hand off verified domain requirements to `python-craftsman`.
- Hand off required tests to `python-tester`.
- Do not execute direct code implementation from this skill.

## Prohibited Actions

This skill must NOT:
- Generate production signals.
- Write to databases.
- Generate broker orders.
- Generate CSV order files.
- Send Telegram messages.
- Invent strategy logic based on assumptions.
- Silently align playbooks to existing code without noting the conflict.
- Invent new strategy variants.
- Calculate production position sizes, create production orders, or export broker CSV files. The playbook may document the verified Trade Manager sizing and order contract.
- Modify application code (`app/`, `tests/`, etc.).

## Output Contract

Findings and designs must be presented cleanly. Do not emit empty tables or sections. Use only the canonical vocabulary from `overview.md`: `COMPLETE`, `INCOMPLETE`, `CONFLICTING`, `VERIFIED`, `PARTIALLY_VERIFIED`, `NOT_REVIEWED`, `NOT_APPLICABLE`, `PARTIAL`, `NONE`, and `UNSPECIFIED` as applicable.
