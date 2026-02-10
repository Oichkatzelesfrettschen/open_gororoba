#!/usr/bin/env python3
"""
Hard gate for Rust-maximal / PyO3 bridge governance.

Policy:
- Every in-scope Python core algorithm file must be explicitly mapped in
  registry/python_core_algorithms.toml.
- Each mapping must include:
  - target Rust crate + module
  - PyO3 binding plan
  - valid binding status token
- Files appearing in scope without a mapping fail the gate.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
import tomllib


def _assert_ascii(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    bad = sorted({ch for ch in text if ord(ch) > 127})
    if bad:
        sample = "".join(bad[:20])
        raise SystemExit(f"ERROR: Non-ASCII content in {path}: {sample!r}")


def _resolve_scoped_paths(
    repo_root: Path,
    watch_globs: list[str],
    exclude_globs: list[str],
) -> list[str]:
    paths: set[str] = set()
    for pattern in watch_globs:
        for path in repo_root.glob(pattern):
            if not path.is_file():
                continue
            rel = path.relative_to(repo_root).as_posix()
            if any(fnmatch.fnmatch(rel, ex) for ex in exclude_globs):
                continue
            paths.add(rel)
    return sorted(paths)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    registry_path = repo_root / "registry/python_core_algorithms.toml"
    if not registry_path.exists():
        print(f"ERROR: missing policy registry: {registry_path}")
        return 1
    _assert_ascii(registry_path)

    data = tomllib.loads(registry_path.read_text(encoding="utf-8"))
    policy = data.get("python_core_algorithms_policy", {})
    mappings = data.get("mapping", [])

    failures: list[str] = []

    watch_globs = [str(item).strip() for item in policy.get("watch_globs", []) if str(item).strip()]
    exclude_globs = [
        str(item).strip() for item in policy.get("exclude_globs", []) if str(item).strip()
    ]
    allowed_status = {
        str(item).strip()
        for item in policy.get("allowed_binding_status", [])
        if str(item).strip()
    }
    if not watch_globs:
        failures.append("policy.watch_globs is empty")
    if not allowed_status:
        failures.append("policy.allowed_binding_status is empty")

    scoped_paths = _resolve_scoped_paths(repo_root, watch_globs, exclude_globs) if watch_globs else []

    mapping_by_path: dict[str, dict] = {}
    ids: set[str] = set()
    for row in mappings:
        map_id = str(row.get("id", "")).strip()
        python_path = str(row.get("python_path", "")).strip()
        if not map_id:
            failures.append("mapping entry with empty id")
            continue
        if map_id in ids:
            failures.append(f"duplicate mapping id: {map_id}")
        ids.add(map_id)

        if not python_path:
            failures.append(f"{map_id}: empty python_path")
            continue
        if python_path in mapping_by_path:
            failures.append(f"{map_id}: duplicate python_path mapping: {python_path}")
        mapping_by_path[python_path] = row

    mapped_paths = set(mapping_by_path.keys())
    scoped_set = set(scoped_paths)

    missing_mappings = sorted(scoped_set - mapped_paths)
    stale_mappings = sorted(mapped_paths - scoped_set)

    for path in missing_mappings:
        failures.append(
            f"{path}: in-scope Python algorithm file lacks mapping in registry/python_core_algorithms.toml"
        )
    for path in stale_mappings:
        failures.append(
            f"{path}: mapped but not in current scope (stale mapping or scope misconfiguration)"
        )

    for path in sorted(scoped_set & mapped_paths):
        row = mapping_by_path[path]
        map_id = str(row.get("id", "")).strip()
        rust_crate = str(row.get("rust_crate", "")).strip()
        rust_module = str(row.get("rust_module", "")).strip()
        binding_status = str(row.get("binding_status", "")).strip()
        binding_plan = str(row.get("pyo3_binding_plan", "")).strip()

        if not rust_crate:
            failures.append(f"{map_id} ({path}): rust_crate is empty")
        else:
            crate_dir = repo_root / "crates" / rust_crate
            if not crate_dir.is_dir():
                failures.append(f"{map_id} ({path}): rust_crate directory missing: {crate_dir}")

        if not rust_module:
            failures.append(f"{map_id} ({path}): rust_module is empty")
        if not binding_plan:
            failures.append(f"{map_id} ({path}): pyo3_binding_plan is empty")
        elif "gororoba_py" not in binding_plan:
            failures.append(
                f"{map_id} ({path}): pyo3_binding_plan must reference gororoba_py integration"
            )

        if binding_status not in allowed_status:
            failures.append(
                f"{map_id} ({path}): binding_status={binding_status!r} not in allowed set"
            )
        if binding_status == "exempt":
            exemption_reason = str(row.get("exemption_reason", "")).strip()
            if not exemption_reason:
                failures.append(f"{map_id} ({path}): exempt status requires exemption_reason")

    expected_count = int(policy.get("mapping_count", len(mappings)))
    if expected_count != len(mappings):
        failures.append(
            f"policy.mapping_count={expected_count} but mapping entries={len(mappings)}"
        )

    if failures:
        print("ERROR: Python core algorithm -> Rust/PyO3 mapping verification failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    print(
        "OK: Python core algorithm mapping verified. "
        f"scoped_files={len(scoped_paths)} mappings={len(mappings)} "
        "all files have Rust crate/module + PyO3 binding plan."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
