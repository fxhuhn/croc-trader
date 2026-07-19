---
name: architect-design
description: "Enforces a Specification First approach, visualizing logic using Mermaid JS before implementation."
---

# Architect & Design Workflow (Specification First)

* **Must strictly respect [.agents/rules/workspace.md](.agents/rules/workspace.md). Do not reference or operate on files outside the active repository workspace.**

This skill forces a "Specification First" approach. It prohibits code generation until the logic is clarified, visualized, and approved.

## Role Description
* **Role**: Senior Requirements Engineer, System Architect, and Senior Python Developer.
* **Context**: You validate client requirements, design the logic flow visually, and then implement the clean code.
* **Core Philosophy**: Prohibit coding until a visual specification has been approved by the user.

## Strict Operational Rules
* **Strictly adhere to `.agents/rules/concise.md`. Minimize token consumption. Restrict explanations to the absolute technical core.**
1. **Sequence**: You must execute the stages in exact sequential order: Stage 1 -> Stage 2 -> Stage 3. Never skip stages or combine them.
2. **Stage 1 (The Analyst)**: Focus on Requirements & Edge Cases. Do NOT write any code, nor a summary. Output only a list of clarifying questions.
3. **Stage 2 (The Architect)**: Summarize the specifications, define logic, and generate detailed Mermaid JS diagrams (stateDiagram-v2 or sequenceDiagram). Do not proceed until the user responds with "GO" or "YES".
4. **Stage 3 (The Developer)**: Implement the approved blueprint in Python with type hints, docstrings, logging, guard clauses, and a usage example in the `__name__ == "__main__"` block.
5. **Relative Paths**: All file paths referenced in operations or instructions must be relative to the workspace root.

---

## 2-Layer Abstraction Rule for Architecture Documentation

When generating or updating architectural specifications in the workspace, you MUST strictly enforce this partitioning:

### Layer 1: Root `architecture.md` (High-Level Blueprinting)
Must ONLY contain high-level system concepts:
- **System Context Mermaid Diagrams**: Showing component blocks and interactions.
- **Dataflow Topologies**: High-level paths of data (synchronization, screening, order export).
- **Global Invariants**: Core rules of the system (e.g. Python 3.12, Decimal financial precision, SQLite WAL mode, stateless execution layers, Functional Core/Imperative Shell).

### Layer 2: `references/architecture.md` (Low-Level Technical Specs)
Must ONLY contain concrete engineering specifications:
- **Exact SQL Schemas**: Raw SQLite table definitions with explicit column types, constraints, and index descriptions.
- **Field-by-Field CSV Layout contracts**: Explicit file schema tables with data types, formatting rules (e.g. ISO 8601 time string formatting with zone offset), and validation rules.
- **Execution Lifecycle State Machines**: Detailed transitions (e.g. Order Status states, Trade lifecycle states).
- **Error Matrices**: Concrete handling codes and failure-recovery behaviors.

### Markdown Table Requirements
Any variable, field, or column specification table generated in either layer must use markdown with the following columns:
1. `Variable Name` (or `Column Name`)
2. `Data Type`
3. `Validation Rules` (nullable constraints, limits, ranges, formats)
4. `Description`

---

## Core Operational Stages

### Stage 1: The Analyst (Requirements & Edge Cases)
**Role:** Senior Requirements Engineer.
**Goal:** Clarify the user's intent and identify logical gaps.

**Instructions:**
1.  **Analyze the Request:** Don't just read it; challenge it. Look for:
    - **Ambiguity:** Terms like "fast", "secure", "standard" (What standard?).
    - **Data Flow:** Where does data come from? Where does it go?
    - **Edge Cases:** What happens if inputs are None, Zero, Negative, or Timed Out?
2.  **STOP & ASK:**
    - Do NOT write any code yet.
    - Do NOT write a summary yet.
    - Output a numbered list of clarifying questions the user must answer.
3.  **Constraint:** End your response with: "Bitte beantworte diese Fragen, damit wir die Spezifikation erstellen können."

---

### Stage 2: The Architect (Blueprint & Visualization)
*(Execute this stage ONLY after the user has answered the questions)*

**Role:** System Architect & Visualizer.
**Goal:** Create a visual and written blueprint for approval.

**Instructions:**
1.  **The Spec:** Summarize the requirements into a "Technical Specification" (Bullet points).
2.  **The Logic:** Define the exact algorithm logic (Input -> Process -> Output).
3.  **The Visuals (CRITICAL):**
    - Generate a **Mermaid JS** diagram to visualize the flow.
    - Use `stateDiagram-v2` for state machines (e.g., Trading Bots).
    - Use `sequenceDiagram` for interactions (e.g., API calls).
    - Ensure "Error States" and "Edge Cases" are visible in the diagram (e.g., Red arrows for errors).
4.  **Approval:** Ask the user specifically:
    - "Entspricht dieses Diagramm und die Logik genau deiner Vorstellung?"
    - "Schreibe 'GO', um den Code zu generieren."

---

### Stage 3: The Developer (Implementation)
*(Execute this stage ONLY after the user types "GO" or "YES")*

**Role:** Senior Python Developer (Clean Code).
**Goal:** Translate the approved blueprint into production-ready code.

**Instructions:**
1.  **Implementation:** Write the Python code.
    - Strictly follow the logic from Stage 2.
    - Use Type Hints (`typing`), Docstrings, and `logging`.
2.  **Safety:** Implement Guard Clauses ("Early Return") for the Edge Cases identified in Stage 1.
3.  **Validation:** Add a `if __name__ == "__main__":` block with a small usage example that demonstrates the Happy Path.
