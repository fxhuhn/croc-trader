---
name: architecture-sync
description: "Ensures system architecture documentation stays synchronized with architecture-relevant code components in the codebase."
---

# Architecture Sync Skill

This skill enforces synchronization between the active codebase structure and the system documentation (`architecture.md` and `references/architecture.md`).

---

## Triggers

Run this skill when a change affects one or more of:

- system components,
- module or package boundaries,
- architecture-relevant public services,
- external interfaces,
- data-flow topology,
- scheduler or batch orchestration,
- database schemas,
- persisted state transitions,
- CSV or API contracts,
- documented global invariants.

Do not trigger architecture documentation updates solely because a small public
helper function was added, renamed, or removed.

---

## Document Scope & Constraints

`references/architecture.md` is read-only unless the explicit task changes a
low-level technical contract that this file authoritatively defines.

Update:

- `architecture.md` for high-level components, interactions, data flows, and
  global invariants.
- `references/architecture.md` for authoritative database schemas, state
  machines, CSV interfaces, and low-level contracts.

Do not duplicate the same contract in both files unless each document needs a
different abstraction level.

This skill updates architecture documentation only. It must not modify
production code to make documentation appear synchronized.

---

## Consistency Verification

The synchronization script provides a mechanical consistency check only.

A successful exact-name scan does not prove semantic documentation accuracy.

The skill must additionally verify whether changed responsibilities,
interfaces, dependencies, schemas, or data flows remain correctly documented.

Verify additions, changes, renames, moves, and deletions. Remove stale
architecture references when an authoritative component or contract is removed.
Do not leave aliases or obsolete names merely to satisfy the mechanical scan.

To run the mechanical check:

```bash
.venv/bin/python .agents/skills/architecture-sync/scripts/check_sync.py
```

Follow `.agents/AGENTS.md` and `.agents/rules/concise.md`.
