#!/usr/bin/env python3
"""
Verify that knowledge_sources markdown is covered by markdown_governance.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    ks = tomllib.loads((repo_root / "registry/knowledge_sources.toml").read_text(encoding="utf-8"))
    gov = tomllib.loads(
        (repo_root / "registry/markdown_governance.toml").read_text(encoding="utf-8")
    )

    ks_paths = {
        str(row.get("path", "")).strip()
        for row in ks.get("document", [])
        if str(row.get("path", "")).strip().endswith(".md")
    }
    gov_paths = {
        str(row.get("path", "")).strip()
        for row in gov.get("document", [])
        if str(row.get("path", "")).strip().endswith(".md")
    }

    failures: list[str] = []
    missing_in_gov = sorted(ks_paths - gov_paths)
    if missing_in_gov:
        failures.append(
            "knowledge_sources paths missing in markdown_governance: "
            + ", ".join(missing_in_gov[:20])
        )

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
