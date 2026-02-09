#!/usr/bin/env python3
"""
Verify that markdown inventory remains TOML-first.

Policy:
- Project markdown must classify as toml_published_markdown.
- Non-project markdown may classify as third_party_markdown.
- No unbacked manual markdown is allowed.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


ALLOWED = {"toml_published_markdown", "third_party_markdown"}


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    inv_path = repo_root / "registry/markdown_inventory.toml"
    data = tomllib.loads(inv_path.read_text(encoding="utf-8"))

    failures: list[str] = []
    summary = data.get("markdown_inventory", {})
    if int(summary.get("unbacked_manual_count", 0)) != 0:
        failures.append(
            f"unbacked_manual_count={summary.get('unbacked_manual_count')} (expected 0)"
        )

    for row in data.get("document", []):
        path = str(row.get("path", "")).strip()
        classification = str(row.get("classification", "")).strip()
        if classification not in ALLOWED:
            failures.append(f"{path}: disallowed classification={classification}")

    if failures:
        print("ERROR: markdown inventory violates TOML-first policy.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("OK: markdown inventory is TOML-first (project markdown) with only third-party exceptions.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
