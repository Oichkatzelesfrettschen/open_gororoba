#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from importlib.util import find_spec


_PATCHES = [
    (
        '__import__("pkg_resources").declare_namespace(__name__)  # type: ignore\n',
        '__path__ = __import__("pkgutil").extend_path(__path__, __name__)  # type: ignore\n',
    ),
    (
        "__import__('pkg_resources').declare_namespace(__name__)  # type: ignore\n",
        "__path__ = __import__('pkgutil').extend_path(__path__, __name__)  # type: ignore\n",
    ),
]


def _patch_file(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return False

    for old, new in _PATCHES:
        if old in content:
            content = content.replace(old, new)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return True

    return False


def _patch_registry_py(path: str) -> bool:
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except FileNotFoundError:
        return False

    if "import pkg_resources" not in content:
        return False

    if "_importlib_metadata.entry_points" in content or "_iter_entry_points" in content:
        return False

    content = content.replace("import pkg_resources\n", "")

    # Drop pkg_resources usage in favor of importlib.metadata entry points.
    shim = (
        "try:\n"
        "    from importlib import metadata as _importlib_metadata\n"
        "except ImportError:  # pragma: no cover\n"
        "    import importlib_metadata as _importlib_metadata  # type: ignore\n"
        "\n"
        "\n"
        "def _iter_entry_points(group, name=None):\n"
        "    eps = _importlib_metadata.entry_points()\n"
        "    if hasattr(eps, 'select'):\n"
        "        eps = eps.select(group=group)\n"
        "    else:  # pragma: no cover\n"
        "        eps = eps.get(group, [])\n"
        "    if name is not None:\n"
        "        eps = [ep for ep in eps if ep.name == name]\n"
        "    return iter(eps)\n"
        "\n"
    )

    # Insert shim after stdlib imports.
    marker = "import contextlib\n"
    if marker not in content:
        return False
    content = content.replace(marker, marker + shim)

    content = content.replace(
        "pkg_resources.iter_entry_points(\"fs.opener\")",
        "_iter_entry_points(\"fs.opener\")",
    )
    content = content.replace(
        "pkg_resources.iter_entry_points(\"fs.opener\", protocol)",
        "_iter_entry_points(\"fs.opener\", protocol)",
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Patch pyfilesystem2 (fs) to avoid importing pkg_resources at import-time.\n\n"
            "Motivation: some distro builds of fs use pkg_resources.declare_namespace(), and\n"
            "newer setuptools/pkg_resources emit a deprecation warning which breaks\n"
            "PYTHONWARNINGS=error workflows."
        )
    )
    ap.add_argument(
        "--allow-system",
        action="store_true",
        help="Allow patching outside the current venv/sys.prefix (NOT recommended).",
    )
    args = ap.parse_args()

    spec = find_spec("fs")
    if spec is None or spec.origin is None:
        print("INFO: 'fs' is not installed; nothing to patch.")
        return 0

    fs_init = spec.origin
    fs_dir = os.path.dirname(fs_init)

    if not args.allow_system:
        prefix = os.path.realpath(sys.prefix)
        fs_init_real = os.path.realpath(fs_init)
        if not fs_init_real.startswith(prefix + os.sep):
            print(
                "ERROR: Refusing to patch system install. "
                f"fs is at {fs_init_real}, sys.prefix is {prefix}. "
                "Re-run with --allow-system if you really want this.",
                file=sys.stderr,
            )
            return 2

    targets = [
        fs_init,
        os.path.join(fs_dir, "opener", "__init__.py"),
        os.path.join(fs_dir, "opener", "registry.py"),
    ]

    changed = 0
    for path in targets:
        if path.endswith(os.path.join("opener", "registry.py")):
            if _patch_registry_py(path):
                changed += 1
        else:
            if _patch_file(path):
                changed += 1

    if changed == 0:
        print("INFO: No matching pkg_resources namespace line found; already patched or not applicable.")
        return 0

    print(f"OK: Patched {changed} file(s) to avoid pkg_resources.declare_namespace().")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
