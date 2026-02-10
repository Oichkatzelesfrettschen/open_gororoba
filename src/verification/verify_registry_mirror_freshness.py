#!/usr/bin/env python3
"""
Verify that TOML-driven markdown/csv mirrors are fresh.

This check is deterministic and warnings-as-errors friendly.
It fails if exported mirror content would differ from tracked files.
"""

from __future__ import annotations

import subprocess
import sys
import os
from pathlib import Path


def main() -> int:
    if os.environ.get("MARKDOWN_EXPORT") != "1":
        print("SKIP: mirror freshness check disabled (set MARKDOWN_EXPORT=1)")
        return 0

    repo_root = Path(__file__).resolve().parents[2]
    out_dir = os.environ.get("MARKDOWN_EXPORT_OUT_DIR", "docs/generated")
    emit_legacy = os.environ.get("MARKDOWN_EXPORT_EMIT_LEGACY", "0") == "1"
    legacy_claims_sync = os.environ.get("MARKDOWN_EXPORT_LEGACY_CLAIMS_SYNC", "1") == "1"
    cmd = [
        sys.executable,
        "src/scripts/analysis/export_registry_markdown_mirrors.py",
        "--repo-root",
        str(repo_root),
        "--out-dir",
        out_dir,
        "--check",
    ]
    cmd.append("--emit-legacy" if emit_legacy else "--no-emit-legacy")
    cmd.append("--legacy-claims-sync" if legacy_claims_sync else "--no-legacy-claims-sync")
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
