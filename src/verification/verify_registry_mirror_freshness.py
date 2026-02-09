#!/usr/bin/env python3
"""
Verify that TOML-driven markdown/csv mirrors are fresh.

This check is deterministic and warnings-as-errors friendly.
It fails if exported mirror content would differ from tracked files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    cmd = [
        sys.executable,
        "src/scripts/analysis/export_registry_markdown_mirrors.py",
        "--repo-root",
        str(repo_root),
        "--check",
    ]
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
