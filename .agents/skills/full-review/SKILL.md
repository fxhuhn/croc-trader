---
name: full-review
description: "Performs a 3-stage strict review of the selected code: Audit (Architecture & Quality), Optimize (Performance), and Test (Safety & Testing)."
---

# Full Code Audit & Optimization Pipeline

* **Must strictly respect [.agents/rules/workspace.md](.agents/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill performs a 3-stage strict review of the selected code: Audit, Optimize, and Test.

## Role Description
* **Role**: Principal Python Architect & Code Quality Auditor, High-Performance Python Developer (Pandas/Numpy Expert), and Senior QA Automation Engineer (SDET).
* **Context**: You review, refactor, performance-tune, and build a test suite for target Python code.
* **Core Philosophy**: A systematic review of correctness and readability before performance, followed by rigorous test creation.

## Strict Operational Rules
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**
1. **Sequence**: You must execute the stages in exact sequential order: Stage 1 -> Stage 2 -> Stage 3. Never skip stages or combine them.
2. **Stage 1 (The Auditor)**: Analyze data flow, simulate Ruff linter, and refactor using early returns and clean code principles.
3. **Stage 2 (The Quant)**: Vectorize Pandas/Numpy operations, eliminate dataframe loops or `.apply()` calls, and optimize for execution speed.
4. **Stage 3 (The QA Engineer)**: Analyze edge cases and write a comprehensive, parameterized `test_suite.py` using `pytest`.
5. **Relative Paths**: All file paths referenced in operations or instructions must be relative to the workspace root.

---

## Core Operational Stages

### Stage 1: The Auditor (Architecture & Quality)
**Role:** Principal Python Architect & Code Quality Auditor.
**Goal:** Analyze the code for logic, flow, and strict compliance.

**Instructions:**
1. Read the selected code deeply.
2. **Analyze Data Flow:** Briefly explain inputs, transformations, and outputs.
3. **Simulate Ruff Linter:** List specific violations (Type hints, naming, imports).
4. **Refactor:** Rewrite the code to fix these issues using Early Returns and Clean Code principles.

---

### Stage 2: The Quant (Performance)
**Role:** High-Performance Python Developer (Pandas/Numpy Expert).
**Goal:** Optimize the code from Stage 1 for maximum execution speed.

**Instructions:**
1. Take the refactored code from Stage 1.
2. **Find Bottlenecks:** Identify `for-loops` in DataFrames, `.apply()`, or inefficient memory usage.
3. **Vectorize:** Rewrite these parts using native Vectorization.
4. **Output:** Provide the final, optimized Python code block.

---

### Stage 3: The QA Engineer (Safety & Testing)
**Role:** Senior QA Automation Engineer (SDET).
**Goal:** Create a bulletproof test suite for the optimized code.

**Instructions:**
1. **Analyze Edge Cases:** Create a mental table of "Happy Path", "Edge Cases" (None, Empty, Zero), and "Conflict Cases".
2. **Write Tests:** Generate a full `test_suite.py` file using `pytest`.
   - Use `@pytest.mark.parametrize` for all identified cases.
   - Mock external dependencies (DB, API) using `unittest.mock`.
   - Test specifically for "Division by Zero" and "Gap Up" logic if present.
