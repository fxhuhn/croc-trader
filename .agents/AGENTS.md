# Agent Execution Order Rules

* You **MUST** strictly follow this execution sequence before conducting any code analysis, debugging, refactoring, or implementation in this workspace:

1. **Step 1 (Architecture Inspection)**: You **MUST** inspect **BOTH** architecture documents first:
   - High-Level System Architecture: [architecture.md](file:///Users/produktmanagement/Python/github/croc-trader/architecture.md) (Component interactions, sequence diagrams, system boundaries)
   - Low-Level Reference Specification: [references/architecture.md](file:///Users/produktmanagement/Python/github/croc-trader/references/architecture.md) (DB schemas, state machines, CSV interfaces, error matrices)
2. **Step 2 (Skill Activation)**: You **MUST** inspect and read the relevant `.agent/skills/<skill>/SKILL.md` file whenever a task involves:
   - Data Ingestion & Quality: `data-ingestion`
   - Strategy & Signal Generation: `strategy-screener`
   - Python Architecture & Code Quality: `python-craftsman` / `python-auditor` / `python-creator`
   - Presentation & UI: `flask-ui` / `python-designer`
3. **Step 3 (Analysis & Implementation)**: You **MUST** perform source code analysis, file modifications, and command execution strictly within the constraints and invariants established in Steps 1 & 2.
