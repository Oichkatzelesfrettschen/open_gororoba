#!/usr/bin/env python3
"""
Fast repository scanning helpers for control-plane builders and verifiers.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def worker_budget(repo_root: Path) -> int:
    raw = os.environ.get("WORKER_BUDGET", "").strip()
    if raw.isdigit():
        return max(1, int(raw))

    script = repo_root / "scripts" / "detect_worker_budget.sh"
    if script.is_file():
        proc = subprocess.run(
            ["sh", str(script)],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        value = proc.stdout.strip()
        if value.isdigit():
            return max(1, int(value))

    return 1


def _discover_files_with_rg(
    repo_root: Path,
    pattern: str,
    skip_prefixes: tuple[str, ...],
    skip_path_parts: set[str],
) -> list[str]:
    proc = subprocess.run(
        ["rg", "--files", "--hidden", "--no-ignore", "-g", pattern],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    out: list[str] = []
    for rel in proc.stdout.splitlines():
        if rel.startswith(".git/"):
            continue
        if any(rel.startswith(prefix) for prefix in skip_prefixes):
            continue
        if any(part in skip_path_parts for part in rel.split("/")):
            continue
        out.append(rel)
    out.sort()
    return out


def _discover_files_with_walk(
    repo_root: Path,
    suffix: str,
    skip_prefixes: tuple[str, ...],
    skip_path_parts: set[str],
) -> list[str]:
    out: list[str] = []
    pattern = f"*{suffix}"
    for path in repo_root.rglob(pattern):
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith(".git/"):
            continue
        if any(rel.startswith(prefix) for prefix in skip_prefixes):
            continue
        if any(part in skip_path_parts for part in rel.split("/")):
            continue
        out.append(rel)
    out.sort()
    return out


def discover_files(
    repo_root: Path,
    suffix: str,
    skip_prefixes: tuple[str, ...],
    skip_path_parts: set[str],
) -> list[str]:
    pattern = f"*{suffix}"
    if shutil.which("rg"):
        return _discover_files_with_rg(repo_root, pattern, skip_prefixes, skip_path_parts)
    return _discover_files_with_walk(repo_root, suffix, skip_prefixes, skip_path_parts)
