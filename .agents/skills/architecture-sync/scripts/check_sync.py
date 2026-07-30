"""
Verification script for architecture.md

Per the updated system rules, architecture.md no longer maintains a manual
inventory of every public function and class. Detailed function and class
inventories are derived from source code instead.

This script ensures that the architecture.md file exists, but skips the
legacy AST-based exact-name scanning.
"""

import sys
from pathlib import Path


def main() -> None:
    """Check that architecture.md exists without requiring manual API lists."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parents[3]

    arch_md_path = project_root / "architecture.md"

    if not arch_md_path.exists():
        print(
            f"Error: Project-root architecture.md not found at {arch_md_path.absolute()}",
            file=sys.stderr,
        )
        sys.exit(1)

    print(
        "\n✅ Success: architecture.md exists. Manual API component index is abolished."
    )
    sys.exit(0)


if __name__ == "__main__":
    main()
