---
trigger: always_on
---

# Strict Workspace Boundary Protocol

## Supported Development Platforms

The supported development and execution environments are Linux and macOS
only.

Use POSIX-compatible paths, shell commands, scripts, and virtual-environment
layouts.

* **Strict Virtual Environment Execution & Zero Global Binaries:** Enforce Non-Overrideable Governance #2 from `.agents/AGENTS.md`. All tools (`.venv/bin/python`, `pytest`, `ruff`, `mypy`, `pip`, etc.) must use the repo-local virtual environment; global/system-wide binaries (`sudo`, global `pip`, `brew`) are strictly prohibited. Document dependencies in `pyproject.toml`.
* You are strictly locked into the current working directory from which you were invoked.
* **ZERO EXTERNAL REPOSITORY ACCESS:** Never inspect, read, write, or execute binaries/scripts from external or neighboring repositories (e.g., `github/TradeManager`, `croc-trader_2`, or any path outside the current workspace root).
* **STRICT COMMAND SCOPING:** Commands executed through the available shell or command-execution capability MUST NOT search or execute outside the workspace. Never run global search commands such as `find /` or `find ~`. All commands must target relative paths within `./` or `.venv/`.
* All file operations (read, write, list) and shell commands MUST use repository-relative paths starting from the current workspace root.
* Dynamically detect the current repository name using `git rev-parse --show-toplevel` if context confusion occurs.

## Command Safety

- Do not execute destructive commands such as recursive deletion, destructive
  database operations, forced Git resets, or history rewrites unless explicitly
  requested and strictly scoped.
- Do not execute downloaded, generated, or untrusted scripts without inspecting
  their contents first.
- Do not use `sudo` or modify system-wide packages or configuration.
- Never install or execute global or system-wide binaries.
- Do not start persistent background processes unless the task explicitly
  requires it.
- Use temporary directories for generated validation artifacts and remove them
  after use when safe.
- **Scratch Scripts:** Enforce Non-Overrideable Governance #1 from `.agents/AGENTS.md` (no scratch files in root; use `<appDataDir>/.../scratch/` or `scripts/one_off/` if persisted).

## Strategy & Constant References
* **Use Strategies Enum**: When checking or filtering trading strategies in the codebase, always use the system-wide canonical `Strategies` enum members from `app/const.py`. Do **NOT** hardcode raw strings (e.g., `"hold_target"`, `"split_target"`, `"DipBuyer"`, or `"TwoPercent"`) in strategy comparisons. Leverage the fact that `Strategies` is a `StrEnum` which compares equal to its raw string representation.

## Runtime & Environment Configurations
* **Always inspect `settings.yaml` at the repository root** before connecting to, launching, or testing local services (such as web servers, webhooks, or database paths).
* **Mandatory Endpoint & Port Verification**: Never state, suggest, or construct a cURL/API URL, port, or route endpoint from memory or default assumptions. Always use `view_file` or `grep_search` to inspect:
  1. `settings.yaml` for `webserver.host` and `webserver.port`.
  2. `app/routes/api.py` (or relevant blueprint files) for verified `@blueprint.route` decorator definitions.
* Do **NOT** rely on fallback default values defined inside Python dataclasses (e.g. `app/config.py`) when runtime configuration files (`settings.yaml`, `.env`) are present in the workspace.

## Production Inspection & Container Log Invariant
For inspecting production containers, logs, and metrics (Non-Overrideable Governance #11 in `.agents/AGENTS.md`):
- Never use `browser_subagent` or temporary scratch scripts in `/tmp` or the root directory.
- **Priority 1 (Official Plugin CLI):** `.venv/bin/python .agents/plugins/dozzle-mcp/scripts/dozzle_cli.py [status|list|logs|search|stats]` (or `python3 ...`).
- **Priority 2 (Native MCP Dispatcher):** `call_mcp_tool` with all 5 mandatory parameters (`ServerName: "dozzle"`, `ToolName`, `Arguments`, `toolSummary`, `toolAction`).

## Domain & Environment Tool Execution Invariant (Mandatory MCP Usage)
For inspecting or interacting with domain services and databases (Non-Overrideable Governance #12 in `.agents/AGENTS.md`):
- **Mandatory Native MCP Dispatcher:** Always use native MCP tools via `call_mcp_tool` (or plugin CLIs). Never craft ad-hoc raw HTTP/curl/urllib scripts.
- **PROD Database Queries via Croc MCP:** All PROD database queries (`signals.db`, `stocks.db`, `trading.db`) must use `croc-trader-prod` (`call_mcp_tool` or `.venv/bin/python .agents/plugins/croc-mcp/scripts/croc_cli.py [health|summary|positions|orders]`).
- **DEV vs. PROD Database Isolation:** `croc-sqlite-readonly` connects strictly to local development SQLite files (`data/*.db`). It has zero connectivity to PROD. Never use it to inspect or reason about PROD state.


Task scope, change discipline, evidence requirements, and completion reporting
are governed by `.agents/AGENTS.md`.
