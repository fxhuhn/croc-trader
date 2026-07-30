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
- other installed tools under `.venv/bin/`

Do not add Windows-specific paths, PowerShell commands, batch files, or
Windows compatibility code unless explicitly requested.

Prefer POSIX-compatible command behavior over GNU-specific behavior because
macOS commonly provides BSD variants of tools such as `sed`, `date`, `stat`,
and `find`.

* You are strictly locked into the current working directory from which you were invoked.
* **ZERO EXTERNAL REPOSITORY ACCESS:** Never inspect, read, write, or execute binaries/scripts from external or neighboring repositories (e.g., `github/TradeManager`, `croc-trader_2`, or any path outside the current workspace root).
* **STRICT COMMAND SCOPING:** Commands passed to `run_command` MUST NOT search or execute outside the workspace. Never run global search commands such as `find /` or `find ~`. All commands must target relative paths within `./` or `.venv/`.
* All file operations (read, write, list) and shell commands MUST use repository-relative paths starting from the current workspace root.
* Dynamically detect the current repository name using `git rev-parse --show-toplevel` if context confusion occurs.

## VS Code Environment Settings
* **Interpreter Path**: In `.vscode/settings.json`, always configure `"python.defaultInterpreterPath"` as a relative path (`".venv/bin/python"`).
* **Avoid Variables**: Do **NOT** use `${workspaceFolder}/.venv/bin/python` because VS Code's Python extension can fail to resolve `${workspaceFolder}` depending on the version or environment.

## Strategy & Constant References
* **Use Strategies Enum**: When checking or filtering trading strategies in the codebase, always use the system-wide canonical `Strategies` enum members from `app/const.py`. Do **NOT** hardcode raw strings (e.g., `"hold_target"`, `"split_target"`, `"DipBuyer"`, or `"TwoPercent"`) in strategy comparisons. Leverage the fact that `Strategies` is a `StrEnum` which compares equal to its raw string representation.

## Runtime & Environment Configurations
* **Always inspect `settings.yaml` at the repository root** before connecting to, launching, or testing local services (such as web servers, webhooks, or database paths).
* Do **NOT** rely on fallback default values defined inside Python dataclasses (e.g. `app/config.py`) when runtime configuration files (`settings.yaml`, `.env`) are present in the workspace.

Task scope, change discipline, evidence requirements, and completion reporting
are governed by `.agents/AGENTS.md`.
