---
trigger: always_on
---

# Strict Workspace Boundary Protocol

## Supported Development Platforms

The supported development and execution environments are Linux and macOS
only.

Use POSIX-compatible paths, shell commands, scripts, and virtual-environment
layouts.

The repository-local virtual environment uses:

- `.venv/bin/python`
- `.venv/bin/pytest`
- `.venv/bin/ruff`
- `.venv/bin/mypy`
- `.venv/bin/pip`
- other installed tools under `.venv/bin/`

* **Strict Virtual Environment Execution:** All commands, test runners, linters, formatters, type checkers, scripts, and package managers MUST strictly be invoked using their repository-relative path `.venv/bin/<tool>` (e.g. `.venv/bin/python`, `.venv/bin/pytest`, `.venv/bin/ruff`, `.venv/bin/mypy`, `.venv/bin/pip`). Never run un-prefixed bare commands like `python` or `pytest`.
* **Zero Global Installation & Usage:** Never install or use global binaries, packages, or system-wide tools (e.g., `pip install` without `.venv/bin/`, `brew install`, `pipx`, `npm install -g`, `sudo`, etc.). All dependencies MUST be installed exclusively into `.venv` via `.venv/bin/pip install` and documented in `pyproject.toml` / `requirements.txt`.
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
- **NO SCRATCH FILES IN ROOT:** Never create diagnostic, scratch, or test scripts (e.g., `test.py`, `debug.py`) in the repository root directory.
- **Dedicated Scratch Folders:** All temporary scripts MUST be created either in the system-provided scratch directory (`<appDataDir>/.../scratch/`) or within `scripts/one_off/` if they are to be persisted in the project.

## VS Code Environment Settings
* **Interpreter Path**: In `.vscode/settings.json`, always configure `"python.defaultInterpreterPath"` as a relative path (`".venv/bin/python"`).
* **Avoid Variables**: Do **NOT** use `${workspaceFolder}/.venv/bin/python` because VS Code's Python extension can fail to resolve `${workspaceFolder}` depending on the version or environment.

## Strategy & Constant References
* **Use Strategies Enum**: When checking or filtering trading strategies in the codebase, always use the system-wide canonical `Strategies` enum members from `app/const.py`. Do **NOT** hardcode raw strings (e.g., `"hold_target"`, `"split_target"`, `"DipBuyer"`, or `"TwoPercent"`) in strategy comparisons. Leverage the fact that `Strategies` is a `StrEnum` which compares equal to its raw string representation.

## Runtime & Environment Configurations
* **Always inspect `settings.yaml` at the repository root** before connecting to, launching, or testing local services (such as web servers, webhooks, or database paths).
* **Mandatory Endpoint & Port Verification**: Never state, suggest, or construct a cURL/API URL, port, or route endpoint from memory or default assumptions. Always use `view_file` or `grep_search` to inspect:
  1. `settings.yaml` for `webserver.host` and `webserver.port`.
  2. `app/routes/api.py` (or relevant blueprint files) for verified `@blueprint.route` decorator definitions.
* Do **NOT** rely on fallback default values defined inside Python dataclasses (e.g. `app/config.py`) when runtime configuration files (`settings.yaml`, `.env`) are present in the workspace.

Task scope, change discipline, evidence requirements, and completion reporting
are governed by `.agents/AGENTS.md`.
