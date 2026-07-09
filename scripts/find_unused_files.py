import os
import sys
from modulefinder import ModuleFinder


def find_unused_files():
    project_root = os.getcwd()
    # Add project root to sys.path so imports resolve
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    script_to_run = os.path.join(project_root, "run.py")

    print(f"Analyzing usage starting from: {script_to_run}")

    finder = ModuleFinder(path=[project_root] + sys.path)
    finder.run_script(script_to_run)

    # Collect all used files
    used_files = set()
    for _name, mod in finder.modules.items():
        if mod.__file__:
            # Resolve to absolute path
            path = os.path.abspath(mod.__file__)
            # Only care about files inside our project root
            if path.startswith(project_root):
                used_files.add(path)

    # Manually add run.py as it is the entry point (ModuleFinder might list it as __main__ but let's be safe)
    used_files.add(os.path.abspath(script_to_run))

    # Find ALL files in project
    all_files = set()
    # Extensions to look for - primarily Python for now as per "checking unused files" in code context
    # checking ALL files might list assets/configs which create_app might load dynamically (not via import)
    # So we should separate "Unused Python Files" vs "Unused Other Files"

    ignored_dirs = {
        ".git",
        ".venv",
        ".idea",
        "__pycache__",
        ".ruff_cache",
        "logs",
        "results",
        "out.sqlsource",
        "output",
        "in",
    }

    for root, dirs, files in os.walk(project_root):
        # Filter directories
        dirs[:] = [d for d in dirs if d not in ignored_dirs]

        for file in files:
            file_path = os.path.abspath(os.path.join(root, file))
            # Start relative from project root for cleaner output
            # rel_path = os.path.relpath(file_path, project_root)
            all_files.add(file_path)

    # Filter for Python files for the main logic
    all_py_files = {f for f in all_files if f.endswith(".py")}
    used_py_files = {f for f in used_files if f.endswith(".py")}

    unused_py_files = all_py_files - used_py_files

    # For non-python files, we can't easily track usage via ModuleFinder.
    # We will list them separately or just focus on Python files.
    # The prompt asks for "all files which are not needet".
    # It's risky to say a config file is unused.
    # trace usage of config files is hard. Use heuristics?

    # Print results in a format we can parse or copy to markdown
    print("\n--- BEGIN REPORT ---")
    print(f"Total Python files found: {len(all_py_files)}")
    print(f"Total Python files used: {len(used_py_files)}")
    print(f"Unused Python files: {len(unused_py_files)}")

    print("\n[UNUSED PYTHON FILES]")
    for f in sorted(unused_py_files):
        print(os.path.relpath(f, project_root))

    print("\n[USED PYTHON FILES]")
    for f in sorted(used_py_files):
        print(os.path.relpath(f, project_root))


if __name__ == "__main__":
    find_unused_files()
