# Strict Workspace Boundary Protocol

* You are strictly locked into the current working directory from which you were invoked.
* Never assume, hardcode, or switch to paths belonging to other repositories (e.g., do not mix TradeManager and croc-trader_2).
* All file operations (read, write, list) and shell commands MUST use repository-relative paths starting from the current workspace root.
* Dynamically detect the current repository name using `git rev-parse --show-toplevel` if context confusion occurs.

## VS Code Environment Settings
* **Interpreter Path**: In `.vscode/settings.json`, always configure `"python.defaultInterpreterPath"` as a relative path (`".venv/bin/python"`).
* **Avoid Variables**: Do **NOT** use `${workspaceFolder}/.venv/bin/python` because VS Code's Python extension can fail to resolve `${workspaceFolder}` depending on the version or environment.

## Strategy & Constant References
* **Use Strategies Enum**: When checking or filtering trading strategies in the codebase, always use the system-wide canonical `Strategies` enum members from [app/const.py](file:///Users/produktmanagement/Python/github/croc-trader/app/const.py). Do **NOT** hardcode raw strings (e.g., `"hold_target"`, `"split_target"`, `"DipBuyer"`, or `"TwoPercent"`) in strategy comparisons. Leverage the fact that `Strategies` is a `StrEnum` which compares equal to its raw string representation.

## Runtime & Environment Configurations
* **Always inspect `settings.yaml` at the repository root** before connecting to, launching, or testing local services (such as web servers, webhooks, or database paths).
* Do **NOT** rely on fallback default values defined inside Python dataclasses (e.g. `app/config.py`) when runtime configuration files (`settings.yaml`, `.env`) are present in the workspace.



