---
description: "Workflow for inspecting live production containers, logs, system health, and database metrics via Dozzle MCP & Croc MCP."
trigger: "/status"
---

# /status Workflow — Production Inspection Protocol

When the user asks to check the production system, inspect containers, investigate logs, check database status in PRO, or invokes `/status`:

## 1. Production Database Invariant (MANDATORY)

> [!IMPORTANT]
> **PRO DATABASE QUERIES VIA CROC MCP ONLY:**
> All database inspections, queries, and state verifications for the **PROD** environment (`signals.db`, `stocks.db`, `trading.db`) **MUST ALWAYS** be conducted via **Croc MCP** (`croc-trader-prod` via `call_mcp_tool` or `croc_cli.py`).
> - The `croc-sqlite-readonly` MCP server connects **strictly to local DEV SQLite files** on the local filesystem (`data/*.db`) and **MUST NEVER** be used for PROD queries.
> - Never execute raw `urllib`, `requests`, `curl`, or custom scripts to query or simulate production database endpoints.

---

## 2. Protocol & Execution Paths

- **Path A (Deterministic & Mandatory Default for Initial Status)**: Run the consolidated plugin CLI:
  ```bash
  .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py status
  ```
  *(Or `python3 .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py status`)*
  - *Benefits*: Executes concurrent multi-source production inspection (Containers & Resources via Dozzle, Real Database & Scheduler Health via Croc MCP, Live Portfolio Summary, IBKR Gateway Connection, and Smart Error Scan) in **< 0.5 seconds**.
  - *Invariant*: For initial analysis, **ONLY run `status`**. Do NOT run multiple exploratory log streams or create temporary scratch files unless `status` explicitly flags an anomaly.

- **Path B (Dedicated Croc MCP Domain & Database Inspection for PRO)**:
  - Via Official CLI:
    ```bash
    .venv/bin/python .agents/plugins/croc-mcp/scripts/croc_cli.py health
    .venv/bin/python .agents/plugins/croc-mcp/scripts/croc_cli.py summary
    .venv/bin/python .agents/plugins/croc-mcp/scripts/croc_cli.py positions
    .venv/bin/python .agents/plugins/croc-mcp/scripts/croc_cli.py orders
    ```
  - Via Native IDE Dispatcher (`call_mcp_tool`):
    - `ServerName`: `"croc-trader-prod"`
    - `ToolName`: `"get_system_health"` | `"get_portfolio_summary"` | `"get_active_positions"` | `"get_broker_active_orders"` | `"get_screener_candidates"`
    - `Arguments`: `{}`

- **Path C (Targeted Log Querying & Searching)**:
  ```bash
  .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py logs croc-trader --since 60 --tail 100
  .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py logs ibkr --since 30 --tail 50
  .venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py search croc-trader error --since 180
  ```
  - *Rule*: ALWAYS provide `--tail` (default 100) when fetching container logs to prevent unbounded SSE streaming delays.

---

## 3. Token-Minimal Synthesis & Reporting Protocol

To ensure minimal token usage and maximum density:
- **Direct Start**: Begin directly with the structured report. Omit pleasantries, conversational filler, and command repetitions.
- **Compact Headings & Tables**: Use short headings and concise bullet points / tables.
- **Structure (Max 25-30 Lines)**:
  1. **Container & Resource Health**: State, CPU, and RAM for `croc-trader`, `ibkr`, `trading-app`.
  2. **PROD Datenbank-Status (Croc MCP)**: File sizes and exact timestamps (`signals.db`, `stocks.db`, `trading.db`, Scheduler state).
  3. **Portfolio-Snapshot (Croc MCP)**: Active position count, invested capital, unrealized PnL, strategy distribution.
  4. **EOD Workflow & Gateway**: Cache pre-warming, scheduler activity, IBKR Gateway connectivity.
  5. **Fehler-Audit**: Number of unexpected errors vs. benign notices.
- Adhere strictly to [.agents/rules/concise.md](file:///Users/produktmanagement/Python/github/croc-trader/.agents/rules/concise.md).
