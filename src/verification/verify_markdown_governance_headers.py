#!/usr/bin/env python3
"""
Verify generated-header compliance for TOML-generated markdown mirrors.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    gov_path = repo_root / "registry/markdown_governance.toml"
    data = tomllib.loads(gov_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    for row in data.get("document", []):
        if not bool(row.get("header_required", False)):
            continue
        rel = str(row.get("path", "")).strip()
        path = repo_root / rel
        if not path.exists():
            failures.append(f"missing generated markdown file: {rel}")
            continue
        head = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:8])
        if "AUTO-GENERATED" not in head:
            failures.append(f"missing AUTO-GENERATED header: {rel}")

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
