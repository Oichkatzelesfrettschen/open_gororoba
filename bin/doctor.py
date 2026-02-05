#!/usr/bin/env python3
from __future__ import annotations

import importlib.util
import shutil
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class Check:
    name: str
    kind: str  # "python" | "binary"


PYTHON_CHECKS = [
    Check("numpy", "python"),
    Check("scipy", "python"),
    Check("pandas", "python"),
    Check("matplotlib", "python"),
    Check("numba", "python"),
    Check("sympy", "python"),
    # optional
    Check("networkx", "python"),
    Check("ripser", "python"),
    Check("persim", "python"),
    Check("astroquery", "python"),
    Check("gwpy", "python"),
    Check("requests", "python"),
    Check("qiskit", "python"),
    # optional: advanced math foundations (see docs/requirements/algebra.md)
    Check("euclid3", "python"),
    Check("quaternion", "python"),  # numpy-quaternion
    Check("pyquaternion", "python"),
    Check("mutatorMath", "python"),
    Check("ipfn", "python"),
    Check("findiff", "python"),
    Check("pymultinest", "python"),
    Check("typedunits", "python"),
    Check("quaternionic", "python"),
    Check("unicodedata2", "python"),
    Check("defcon", "python"),
    Check("fontMath", "python"),
]

BINARY_CHECKS = [
    Check("docker", "binary"),
    Check("coqc", "binary"),
    Check("latexmk", "binary"),
]


def has_python_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def has_binary(name: str) -> bool:
    return shutil.which(name) is not None


def main() -> int:
    print("gemini-experiments doctor")
    print(f"python: {sys.version.split()[0]}")

    print("\nPython modules:")
    for check in PYTHON_CHECKS:
        ok = has_python_module(check.name)
        print(f"- {check.name}: {'OK' if ok else 'MISSING'}")

    print("\nSystem binaries:")
    for check in BINARY_CHECKS:
        ok = has_binary(check.name)
        print(f"- {check.name}: {'OK' if ok else 'MISSING'}")

    print("\nNext steps:")
    print("- Core: `make test`")
    print("- Optional requirements: `REQUIREMENTS.md`")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
