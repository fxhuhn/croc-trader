---
description: "Workflow for inspecting live production containers, logs, and system health via Dozzle MCP."
trigger: "/status"
---

# /status Workflow — Production Inspection Protocol

When the user asks to check the production system, inspect containers, investigate logs, or invokes `/status`:

1. **Protocol & Execution Paths**:
   - **Path A (Deterministic & Mandatory Default for Initial Analysis)**: Run the consolidated plugin CLI:
     ```bash
     .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py status
     ```
     *(Or `python3 .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py status`)*
     *Benefits*: Executes the complete production inspection (Containers & Resources, EOD Scheduler & Tasks, Order Pipeline States, IBKR Gateway Connection, and Smart Error/Exception Scan) in **< 1 second**.
     *Invariant*: For initial analysis, **ONLY run `status`**. Do NOT run multiple exploratory log streams or create temporary scratch files unless `status` explicitly flags an anomaly that requires deep dives.
   - **Path B (Targeted Log Querying & Searching)**:
     ```bash
     .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py logs croc-trader --since 60 --tail 100
     .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py logs ibkr --since 30 --tail 50
     .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py search croc-trader error --since 180
     ```
     *Rule*: ALWAYS provide `--tail` (default 100) when fetching container logs to prevent unbounded SSE streaming delays.
   - Tool signatures, CLI commands, and parameter schemas are defined in [.agents/plugins/dozzle-mcp/instructions.md](file:///Users/produktmanagement/Python/github/croc-trader/.agents/plugins/dozzle-mcp/instructions.md).

2. **Synthesis & Reporting**:
   - Deliver the results directly to the user based on the structured 5-section `status` output:
     * 1. Production container states and health (`croc-trader`, `ibkr`) with CPU & RAM.
     * 2. Scheduler & EOD workflow lifecycle (Screener Engine, Trade Manager, Cache pre-warming).
     * 3. Order generation and export pipeline status.
     * 4. IBKR CapTrader Gateway connectivity and events.
     * 5. Real unhandled errors/exceptions vs. filtered benign notices.
