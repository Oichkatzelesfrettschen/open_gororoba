#!/usr/bin/env python3
"""Run Ruff on changed Python files only.

This repo carries legacy Python lint debt outside the actively maintained gate
surface. The required gate therefore ratchets Ruff on the current change set,
while `make lint-all` remains available for backlog burn-down and reporting.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON_PREFIXES = ("src/", "tests/", "bin/", "scripts/")
PYTHON_EXTENSIONS = (".py",)


class BaseRefError(RuntimeError):
    """Raised when the selected git diff base cannot be resolved safely."""


def git_result(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )


def git(*args: str) -> str:
    return git_result(*args).stdout.strip()


def in_ci() -> bool:
    return os.environ.get("GITHUB_ACTIONS", "").lower() == "true"


def ref_exists(ref: str) -> bool:
    return git_result("rev-parse", "--verify", ref).returncode == 0


def detect_base_ref() -> str:
    gh_base = os.environ.get("GITHUB_BASE_REF")
    if gh_base:
        return f"origin/{gh_base}"

    gh_before = os.environ.get("GITHUB_EVENT_BEFORE")
    if gh_before and gh_before != "0" * 40:
        return gh_before

    for candidate in ("origin/main", "origin/master"):
        if git("rev-parse", "--verify", candidate):
            return candidate

    return "HEAD~1"


def validate_base_ref(base: str) -> None:
    if ref_exists(base):
        return
    raise BaseRefError(
        f"cannot resolve diff base `{base}`. Fetch the base history before running "
        "changed-file Ruff (for GitHub Actions, use actions/checkout with fetch-depth: 0)."
    )


def _collect_committed_paths(base: str) -> set[str]:
    committed = git_result("diff", "--name-only", f"{base}...HEAD")
    if committed.returncode != 0 or not committed.stdout.strip():
        committed = git_result("diff", "--name-only", base, "HEAD")
    if committed.returncode != 0:
        raise BaseRefError(
            f"failed to diff against base `{base}`. Fetch the base history before "
            "running changed-file Ruff."
        )
    return {line for line in committed.stdout.splitlines() if line}


def changed_files(base: str | None) -> list[str]:
    paths: set[str] = set()

    if base is not None:
        paths.update(_collect_committed_paths(base))

    # Local gate runs must also lint staged and unstaged working-tree changes,
    # even when the branch already has committed deltas relative to the base.
    working_tree = git("diff", "--name-only", "HEAD")
    if working_tree:
        paths.update(line for line in working_tree.splitlines() if line)

    untracked = git(
        "ls-files",
        "--others",
        "--exclude-standard",
        *PYTHON_PREFIXES,
    )
    if untracked:
        paths.update(line for line in untracked.splitlines() if line)

    return sorted(paths)


def is_python_gate_file(path: str) -> bool:
    if not path.endswith(PYTHON_EXTENSIONS):
        return False
    if not any(path.startswith(prefix) for prefix in PYTHON_PREFIXES):
        return False
    return (REPO_ROOT / path).is_file()


def main() -> int:
    base = detect_base_ref()
    base_available = True
    try:
        validate_base_ref(base)
    except BaseRefError as exc:
        if in_ci():
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
        base_available = False
        print(
            f"WARNING: {exc} Falling back to working-tree and untracked files only.",
            file=sys.stderr,
        )
        base = None

    paths = sorted({path for path in changed_files(base) if is_python_gate_file(path)})

    if not paths:
        if not base_available:
            print(
                "OK: no working-tree Python gate files found; committed branch deltas "
                "were not linted because base history is unavailable."
            )
        else:
            print("OK: no changed Python gate files; skipping Ruff ratchet.")
        return 0

    command = [sys.executable, "-m", "ruff", "check", *paths]
    print(
        "Running changed-file Ruff ratchet against "
        f"{len(paths)} file(s) from base {base}: {', '.join(paths)}"
    )
    return subprocess.run(command, cwd=REPO_ROOT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
