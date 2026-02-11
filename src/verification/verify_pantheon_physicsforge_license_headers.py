#!/usr/bin/env python3
"""
Verify license header consistency for migrated Rust/Python files.

Policy intent:
- GPL-2.0-only remains the only allowed license token.
- Repo style currently allows source files without per-file license headers.
- If headers are present in migrated files, they must be consistently GPL-2.0-only.
"""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


PORTED_FILES_PATH = "registry/pantheon_physicsforge_ported_files.toml"
ALIGNMENT_PATH = "registry/pantheon_physicsforge_license_alignment.toml"

FORBIDDEN_PATTERNS = (
    re.compile(r"\\bGPL-3\\.0\\b", re.IGNORECASE),
    re.compile(r"\\bGPL-3\\.0-only\\b", re.IGNORECASE),
    re.compile(r"\\bGPL-3\\.0-or-later\\b", re.IGNORECASE),
    re.compile(r"\\bLGPL\\b", re.IGNORECASE),
    re.compile(r"\\bAGPL\\b", re.IGNORECASE),
    re.compile(r"\\bMIT\\b", re.IGNORECASE),
    re.compile(r"\\bApache\\b", re.IGNORECASE),
    re.compile(r"\\bBSD\\b", re.IGNORECASE),
    re.compile(r"\\bCC-BY\\b", re.IGNORECASE),
    re.compile(r"\\bproprietary\\b", re.IGNORECASE),
)

SPDX_RE = re.compile(r"SPDX-License-Identifier\s*:\s*([^\s*#]+)", re.IGNORECASE)


def _load(path: Path) -> dict:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _scan_header(path: Path, scan_lines: int) -> tuple[bool, list[str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    head = "\n".join(lines[:scan_lines])
    failures: list[str] = []
    has_header_signal = False

    spdx_match = SPDX_RE.search(head)
    if spdx_match:
        has_header_signal = True
        spdx_value = spdx_match.group(1).strip()
        if spdx_value != "GPL-2.0-only":
            failures.append(
                f"{path}: SPDX header must be GPL-2.0-only, found {spdx_value!r}"
            )

    if "gpl-2.0-only" in head.lower() or "gpl v2 only" in head.lower():
        has_header_signal = True

    for pattern in FORBIDDEN_PATTERNS:
        if pattern.search(head):
            failures.append(
                f"{path}: forbidden license token in header region: {pattern.pattern}"
            )

    return has_header_signal, failures


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]

    ported = _load(repo_root / PORTED_FILES_PATH)
    alignment = _load(repo_root / ALIGNMENT_PATH)

    required_license = str(
        alignment.get("license_alignment", {}).get("required_license", "GPL-2.0-only")
    ).strip()
    if required_license != "GPL-2.0-only":
        print(
            "ERROR: license alignment policy drift detected. "
            f"required_license={required_license!r}"
        )
        return 1

    scan_lines = 12
    migrated_rows = []
    for row in ported.get("ported_file", []):
        origin = str(row.get("origin", "")).strip().lower()
        rel_path = str(row.get("path", "")).strip()
        if origin not in {"pantheon", "physicsforge"}:
            continue
        if not rel_path.endswith((".rs", ".py")):
            continue
        migrated_rows.append(row)

    failures: list[str] = []
    header_present_paths: list[str] = []
    scanned_paths: list[str] = []

    for row in migrated_rows:
        rel_path = str(row.get("path", "")).strip()
        file_path = repo_root / rel_path
        if not file_path.is_file():
            failures.append(f"ported migrated file missing on disk: {rel_path}")
            continue

        has_header, scan_failures = _scan_header(file_path, scan_lines=scan_lines)
        if has_header:
            header_present_paths.append(rel_path)
        scanned_paths.append(rel_path)
        failures.extend(scan_failures)

    if scanned_paths:
        # Consistency policy: migrated files should be either all header-less (repo style)
        # or all explicitly tagged GPL-2.0-only.
        if 0 < len(header_present_paths) < len(scanned_paths):
            failures.append(
                "mixed license header style in migrated files: "
                f"{len(header_present_paths)} of {len(scanned_paths)} have explicit headers"
            )

    if failures:
        print("ERROR: Pantheon/PhysicsForge migrated license header verification failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    header_mode = "headerless_repo_style"
    if scanned_paths and len(header_present_paths) == len(scanned_paths):
        header_mode = "explicit_gpl2_headers"

    print(
        "OK: migrated Rust/Python license header consistency verified "
        f"(files={len(scanned_paths)}, mode={header_mode}, required_license={required_license})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
