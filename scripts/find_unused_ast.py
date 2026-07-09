import ast
import os


def get_imports(file_path):
    try:
        with open(file_path, encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=file_path)
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level > 0:
                # Relative import
                # We return a special marker or handle it later
                # For now, let's just store it as (level, module)
                imports.append((".", node.level, module))
            else:
                imports.append(module)
    return imports


def resolve_import(import_name, current_file_dir, project_root):
    """
    Resolve partial import names to possible file paths.
    Return list of found files.
    """
    candidates = []

    # helper to check if path exists and is file
    def check_file(path):
        return os.path.isfile(path)

    # Absolute imports (relative to project root)
    if isinstance(import_name, str):
        parts = import_name.split(".")
        base_path = os.path.join(project_root, *parts)

        # 1. package/__init__.py
        p1 = os.path.join(base_path, "__init__.py")
        if check_file(p1):
            candidates.append(p1)

        # 2. package.py
        p2 = base_path + ".py"
        if check_file(p2):
            candidates.append(p2)

        # 3. If import a.b.c, maybe a/b.py exists and c is a class
        # checking parents
        for i in range(len(parts) - 1, 0, -1):
            sub_parts = parts[:i]
            sub_path = os.path.join(project_root, *sub_parts)
            p_sub = sub_path + ".py"
            if check_file(p_sub):
                candidates.append(p_sub)
            p_sub_init = os.path.join(sub_path, "__init__.py")
            if check_file(p_sub_init):
                candidates.append(p_sub_init)

    # Relative imports
    elif isinstance(import_name, tuple) and import_name[0] == ".":
        level = import_name[1]
        module = import_name[2]

        # Resolve 'current_file_dir' up 'level' times
        target_dir = current_file_dir
        for _ in range(level - 1):  # -1 because level 1 is same dir
            target_dir = os.path.dirname(target_dir)

        if module:
            parts = module.split(".")
            base_path = os.path.join(target_dir, *parts)

            p1 = os.path.join(base_path, "__init__.py")
            if check_file(p1):
                candidates.append(p1)

            p2 = base_path + ".py"
            if check_file(p2):
                candidates.append(p2)
        else:
            # from . import x ... imports from __init__.py of current package
            p1 = os.path.join(target_dir, "__init__.py")
            if check_file(p1):
                candidates.append(p1)

    return candidates


def find_unused_files():
    project_root = os.getcwd()
    start_file = os.path.join(project_root, "run.py")

    if not os.path.exists(start_file):
        print(f"Start file not found: {start_file}")
        return

    used_files = set()
    queue = [start_file]
    used_files.add(start_file)

    processed = set()

    print(f"Tracing imports from {start_file}...")

    while queue:
        current_file = queue.pop(0)
        if current_file in processed:
            continue
        processed.add(current_file)

        current_dir = os.path.dirname(current_file)
        raw_imports = get_imports(current_file)

        for imp in raw_imports:
            # We don't distinguish what part of module is imported, just that the module IS imported.
            file_candidates = resolve_import(imp, current_dir, project_root)

            for cand in file_candidates:
                if cand not in used_files:
                    used_files.add(cand)
                    queue.append(cand)

    # Now find ALL python files
    all_py_files = set()
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
        dirs[:] = [d for d in dirs if d not in ignored_dirs]
        for file in files:
            if file.endswith(".py"):
                all_py_files.add(os.path.abspath(os.path.join(root, file)))

    unused_py_files = all_py_files - used_files

    print("\n--- REPORT ---")
    print(f"Total Python files: {len(all_py_files)}")
    print(f"Reachable Python files: {len(used_files)}")
    print(f"Unused/Unreachable Python files: {len(unused_py_files)}")

    print("\n[UNUSED PYTHON FILES]")
    for f in sorted(unused_py_files):
        print(os.path.relpath(f, project_root))


if __name__ == "__main__":
    find_unused_files()
