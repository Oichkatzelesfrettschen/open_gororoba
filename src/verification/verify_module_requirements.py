#!/usr/bin/env python3
"""Verify module requirements have machine-checkable structure."""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    mod_req_path = repo_root / "registry" / "module_requirements.toml"

    try:
        data = tomllib.loads(mod_req_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"ERROR: Failed to parse module_requirements.toml: {e}")
        return 1

    modules = data.get("module", [])
    if not modules:
        print("OK: No modules to validate (empty registry)")
        return 0

    failures = []
    for mod in modules:
        mod_id = mod.get("id", "UNKNOWN")
        
        # Verify required fields
        if not mod.get("id"):
            failures.append(f"Module missing id field")
        if not mod.get("name"):
            failures.append(f"{mod_id}: missing name")
        if not mod.get("runtime_stack"):
            failures.append(f"{mod_id}: missing runtime_stack")

    if failures:
        print("ERROR: Module requirements structure violations:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"OK: Module requirements structure verified ({len(modules)} modules)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
