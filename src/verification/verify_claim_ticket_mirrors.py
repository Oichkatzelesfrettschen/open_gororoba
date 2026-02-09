#!/usr/bin/env python3
"""
Verify ticket markdown mirrors are present and generated from claim_tickets TOML.
"""

from __future__ import annotations

import tomllib
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    data = tomllib.loads((repo_root / "registry/claim_tickets.toml").read_text(encoding="utf-8"))
    tickets = data.get("ticket", [])

    expected_paths = set()
    failures: list[str] = []
    for row in tickets:
        rel = str(row.get("source_markdown", "")).strip()
        if not rel:
            failures.append(f"ticket missing source_markdown: {row.get('id', '(unknown)')}")
            continue
        expected_paths.add(rel)
        path = repo_root / rel
        if not path.exists():
            failures.append(f"missing ticket markdown mirror: {rel}")
            continue
        head = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:8])
        if "AUTO-GENERATED" not in head:
            failures.append(f"ticket file missing AUTO-GENERATED header: {rel}")

    tickets_dir = repo_root / "docs/tickets"
    existing_paths = {
        p.relative_to(repo_root).as_posix()
        for p in tickets_dir.glob("*.md")
        if p.name != "INDEX.md"
    }
    extra = sorted(existing_paths - expected_paths)
    if extra:
        failures.append(
            "ticket markdown files not declared in registry/claim_tickets.toml: "
            + ", ".join(extra)
        )

    index_path = tickets_dir / "INDEX.md"
    if not index_path.exists():
        failures.append("missing docs/tickets/INDEX.md")
    else:
        head = "\n".join(index_path.read_text(encoding="utf-8", errors="ignore").splitlines()[:8])
        if "AUTO-GENERATED" not in head:
            failures.append("docs/tickets/INDEX.md missing AUTO-GENERATED header")

    if failures:
        for msg in failures:
            print(f"ERROR: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
