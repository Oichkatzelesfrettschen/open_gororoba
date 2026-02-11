#!/usr/bin/env python3
"""
Verify RUST-FIRST policy: no algorithm implementations in Python.

Allowed in Python:
- Data visualization (matplotlib, plotly, seaborn)
- CLI wrappers and glue code
- Jupyter notebooks for exploration

Forbidden in Python:
- Core algorithms (optimization, linear algebra, numerical solvers)
- Business logic (physics simulations, algebra operations)
- Duplicate implementations of Rust code
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


# Regex patterns for algorithm imports (forbidden)
ALGORITHM_PATTERNS = [
    r"scipy\.optimize",
    r"scipy\.integrate",
    r"scipy\.linalg",
    r"scipy\.special",
    r"numpy\.linalg",
    r"numpy\.fft",
    r"cvxpy",
    r"pulp",
    r"sklearn\.(?!datasets|preprocessing)",  # sklearn ML is forbidden
    r"sympy\.(?!printing)",  # sympy symbolic computation (OK: printing)
    r"odeint\(",
    r"solve_ivp\(",
]

# Regex patterns for visualization imports (allowed)
VISUALIZATION_PATTERNS = [
    r"matplotlib",
    r"plotly",
    r"seaborn",
    r"pandas\.plot",
    r"holoviews",
    r"bokeh",
]


def check_python_file(path: Path) -> list[str]:
    """Check a Python file for forbidden algorithm imports."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        return [f"Failed to read {path}: {e}"]

    violations = []
    for line_num, line in enumerate(content.split("\n"), 1):
        # Skip comments
        code_part = line.split("#")[0]

        # Check for algorithm imports
        for pattern in ALGORITHM_PATTERNS:
            if re.search(pattern, code_part):
                # Exception: allowed imports
                if any(
                    allowed in code_part
                    for allowed in ["datasets", "preprocessing", "printing"]
                ):
                    continue
                violations.append(f"{path}:{line_num}: {code_part.strip()}")

    return violations


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    # Scan all Python files (excluding venv, tests, notebooks)
    python_files = []
    for pattern in [
        "src/**/*.py",
        "crates/gororoba_py/**/*.py",
    ]:
        python_files.extend(repo_root.glob(pattern))

    # Filter out excluded directories
    excluded = {"venv", "__pycache__", ".venv", "env"}
    python_files = [
        f
        for f in python_files
        if not any(part in excluded for part in f.parts)
    ]

    violations = []
    for py_file in python_files:
        violations.extend(check_python_file(py_file))

    if violations:
        print("ERROR: RUST-FIRST policy violations (algorithm implementations in Python):")
        for violation in violations[:20]:  # Limit output
            print(f"  {violation}")
        if len(violations) > 20:
            print(f"  ... and {len(violations) - 20} more")
        return 1

    print(f"OK: RUST-FIRST policy verified ({len(python_files)} Python files scanned)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
