#!/usr/bin/env python3
"""
Verify immutability contract for TOML-generated markdown mirrors.

Contract:
- Every markdown file classified as mode="toml_generated_mirror" must:
  1) Exist.
  2) Include "AUTO-GENERATED: DO NOT EDIT" in the header window.
  3) Include a "Source of truth:" marker in the header window.
  4) If source_toml_refs are recorded in governance, reference at least one in
     the header window.
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
        if str(row.get("mode", "")) != "toml_generated_mirror":
            continue
        rel = str(row.get("path", "")).strip()
        if not rel.endswith(".md"):
            continue
        path = repo_root / rel
        if not path.exists():
            failures.append(f"missing toml_generated_mirror file: {rel}")
            continue

        head = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:12])
        if "AUTO-GENERATED: DO NOT EDIT" not in head:
            failures.append(f"missing immutability marker in mirror header: {rel}")
        if "Source of truth:" not in head:
            failures.append(f"missing source-of-truth marker in mirror header: {rel}")

        refs = [str(item).strip() for item in row.get("source_toml_refs", []) if str(item).strip()]
        if refs:
            if not any(ref in head for ref in refs):
                failures.append(
                    f"mirror header does not reference governance source_toml_refs: {rel}"
                )

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
