# Agent Execution Order Rules — MANDATORY, NO EXCEPTIONS

> [!CAUTION]
> These rules are NON-NEGOTIABLE. Failure to follow them is a CRITICAL violation.
> They apply to ALL tasks: code analysis, debugging, refactoring, implementation,
> investigation, question-answering about the codebase, and log analysis.

## Mandatory 3-Step Execution Sequence

Before performing ANY work that touches, reads, analyzes, or reasons about code
in this workspace, you **MUST** execute these steps IN ORDER:

### Step 1 — Architecture Inspection (ALWAYS REQUIRED)

You **MUST** read **BOTH** architecture documents using the `view_file` tool
before any other file access or code reasoning:

1. **High-Level System Architecture**: [architecture.md](file:///Users/produktmanagement/Python/github/croc-trader/architecture.md)
   — Component interactions, sequence diagrams, system boundaries
2. **Low-Level Reference Specification**: [references/architecture.md](file:///Users/produktmanagement/Python/github/croc-trader/references/architecture.md)
   — DB schemas, state machines, CSV interfaces, error matrices

**No shortcuts.** Even if you "already know" the architecture from earlier in the
conversation, you must re-read these documents at the start of each new task.

### Step 2 — Skill Activation (WHEN APPLICABLE)

You **MUST** inspect and read the relevant `.agents/skills/<skill>/SKILL.md` file
whenever a task involves:

| Domain                              | Skill(s) / Rule(s)                                |
|-------------------------------------|---------------------------------------------------|
| Data Ingestion & Quality            | `data-ingestion`                                  |
| Strategy & Signal Generation        | `strategy-screener`                               |
| Python Architecture & Code Quality  | `python-craftsman` / `python-auditor` / `python-creator` |
| Presentation & UI                   | `flask-ui` / `python-designer`                    |
| API Design & Endpoints              | `rules/api.md`                                    |

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
