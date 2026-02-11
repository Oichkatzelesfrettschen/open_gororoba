#!/usr/bin/env python3
"""
Verify Pantheon/PhysicsForge licensing remains GPL-2.0-only.

This verifier checks sibling repositories cloned under ../:
- ../pantheon
- ../PhysicsForge
"""

from __future__ import annotations

from pathlib import Path
import re
import tomllib


REQUIRED_LICENSE = "GPL-2.0-only"


def check_license_file(path: Path) -> list[str]:
    failures: list[str] = []
    if not path.is_file():
        failures.append(f"missing file: {path}")
        return failures
    text = path.read_text(encoding="utf-8", errors="replace")
    if "GNU GENERAL PUBLIC LICENSE" not in text or "Version 2, June 1991" not in text:
        failures.append(f"{path}: expected GPLv2 canonical text")
    return failures


def check_pyproject(path: Path) -> list[str]:
    failures: list[str] = []
    if not path.is_file():
        failures.append(f"missing file: {path}")
        return failures
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project", {})
    license_block = project.get("license", {})
    text = str(license_block.get("text", "")).strip()
    if text != REQUIRED_LICENSE:
        failures.append(f"{path}: project.license.text={text!r}, expected {REQUIRED_LICENSE!r}")
    return failures


def check_readme(path: Path) -> list[str]:
    failures: list[str] = []
    if not path.is_file():
        failures.append(f"missing file: {path}")
        return failures
    text = path.read_text(encoding="utf-8", errors="replace")
    if REQUIRED_LICENSE not in text and "GPL v2 only" not in text:
        failures.append(f"{path}: missing explicit GPL-2.0-only/GPL v2 only statement")
    return failures


def check_no_fallback_license_mentions(path: Path) -> list[str]:
    failures: list[str] = []
    if not path.is_file():
        failures.append(f"missing file: {path}")
        return failures
    text = path.read_text(encoding="utf-8", errors="replace")
    bad_patterns = [
        r"CC-BY-4\.0",
        r"\bMIT\b",
        r"\bApache\b",
        r"\bproprietary\b",
    ]
    for pat in bad_patterns:
        if re.search(pat, text, flags=re.IGNORECASE):
            failures.append(f"{path}: contains forbidden fallback license token {pat!r}")
    return failures


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    github_root = repo_root.parent
    pantheon = github_root / "pantheon"
    physicsforge = github_root / "PhysicsForge"

    failures: list[str] = []
    failures.extend(check_license_file(pantheon / "LICENSE"))
    failures.extend(check_pyproject(pantheon / "pyproject.toml"))
    failures.extend(check_readme(pantheon / "README.md"))

    failures.extend(check_license_file(physicsforge / "LICENSE"))
    failures.extend(check_pyproject(physicsforge / "pyproject.toml"))
    failures.extend(check_readme(physicsforge / "README.md"))
    failures.extend(
        check_no_fallback_license_mentions(physicsforge / "docs" / "RELEASE_NOTES_v1.0.md")
    )

    if failures:
        print("ERROR: Pantheon/PhysicsForge license consistency verification failed.")
        for item in failures:
            print(f"- {item}")
        return 1

    print("OK: Pantheon and PhysicsForge are aligned to GPL-2.0-only policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
