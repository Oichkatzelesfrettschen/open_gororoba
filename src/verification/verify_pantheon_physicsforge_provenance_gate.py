#!/usr/bin/env python3
"""
Reject migrated files without provenance rows.

Policy:
- Every Rust file under required migration prefixes must have a row in
  registry/pantheon_physicsforge_ported_files.toml.
- Files marked origin pantheon/physicsforge must include mapping IDs.
- Verbatim copy mode is forbidden.
"""

from __future__ import annotations

from pathlib import Path
import tomllib


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "registry" / "pantheon_physicsforge_ported_files.toml"
    data = tomllib.loads(registry_path.read_text(encoding="utf-8"))

    gate = data.get("provenance_gate", {})
    required_prefixes = [str(p) for p in gate.get("required_prefixes", [])]
    reject_verbatim = bool(gate.get("reject_verbatim_copy", True))
    rows = data.get("ported_file", [])

    failures: list[str] = []
    by_path: dict[str, dict] = {}

    for row in rows:
        path = str(row.get("path", "")).strip()
        if not path:
            failures.append("ported_file row missing path")
            continue
        if path in by_path:
            failures.append(f"duplicate ported_file row for {path}")
            continue
        by_path[path] = row
        file_path = repo_root / path
        if not file_path.is_file():
            failures.append(f"ported_file path does not exist: {path}")

        copy_mode = str(row.get("copy_mode", "")).strip()
        if reject_verbatim and copy_mode == "verbatim":
            failures.append(f"{path}: forbidden copy_mode=verbatim")

        origin = str(row.get("origin", "")).strip().lower()
        mapping_ids = row.get("source_mapping_ids", [])
        if origin in {"pantheon", "physicsforge"} and not mapping_ids:
            failures.append(f"{path}: origin={origin} requires source_mapping_ids")

        if row.get("license_checked") is not True:
            failures.append(f"{path}: license_checked must be true")

    required_files: list[str] = []
    for prefix in required_prefixes:
        prefix_path = repo_root / prefix
        if not prefix_path.exists():
            failures.append(f"required prefix does not exist: {prefix}")
            continue
        for file_path in sorted(prefix_path.rglob("*.rs")):
            rel = file_path.relative_to(repo_root).as_posix()
            required_files.append(rel)

    for rel in required_files:
        if rel not in by_path:
            failures.append(f"missing provenance row for migrated file: {rel}")

    if failures:
        print("ERROR: provenance gate failed for Pantheon/PhysicsForge migration.")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "OK: provenance gate passed (all required migrated files have canonical provenance rows)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
