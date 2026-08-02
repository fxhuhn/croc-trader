# Strategy Playbooks

This directory contains the canonical strategy contracts used by the `strategy-screener` skill.

## Files

- `overview.md` defines shared lifecycle, responsibility, persistence, numerical, provider, and verification contracts.
- `TEMPLATE.md` defines the required 22-section structure for strategy playbooks.
- Each strategy file defines one canonical strategy contract and must use stable Contract IDs.

## Canonical vocabulary

- Missing normative value: `UNSPECIFIED`
- Unreviewed implementation or configuration evidence: `NOT_REVIEWED`
- Unreviewed tests: `NOT_REVIEWED`
- Confirmed incompatibility between normative sources or observed behavior: `CONFLICTING`
- Inapplicable section or rule: `NOT_APPLICABLE` with a sourced reason

Do not use prose substitutes such as `Not specified` or `Not verified`.

## Global architecture precedence

`architecture.md` and `references/architecture.md` govern system-wide responsibilities, state values, persistence, and CSV interfaces. Strategy playbooks are canonical only for strategy-specific business rules within those boundaries.

## Maintenance rules

1. Read `overview.md` before editing a strategy playbook.
2. Use `TEMPLATE.md` for new playbooks and structural migrations.
3. Preserve normative strategy rules unless an explicit user decision changes them.
4. Treat code, configuration, and tests as evidence; record mismatches instead of silently rewriting the playbook.
5. Keep Decision Flow, Execution Flow, implementation mapping, test mapping, and conflicts linked through Contract IDs.
6. Do not claim repository conformance without concrete code, configuration, or test evidence.
7. Keep `INVALIDATED` and `CLOSED` terminal and mutually exclusive.

## Strategy index

See [overview.md](overview.md#13-strategy-index), including the intentionally incomplete CrocSetup contract derived only from the API and package references supplied in this archive.
