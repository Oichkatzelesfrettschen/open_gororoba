#!/usr/bin/env python3
"""
Terminology gate: enforces banned-term policy from registry/terminology_standards.toml.

Reads the [[banned]] section, scans all git-tracked text files, and reports
violations with file:line context.  Exits non-zero if any banned term is found.

Usage:
  python3 bin/terminology_gate.py          # scan and report
  python3 bin/terminology_gate.py --quiet  # exit code only
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

# Try tomllib (3.11+), fall back to tomli
try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        print("ERROR: need Python >= 3.11 (tomllib) or `pip install tomli`", file=sys.stderr)
        sys.exit(2)

# Files and extensions to skip (binary, generated, or this script itself)
SKIP_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".svg",
    ".pdf", ".ttf", ".otf", ".woff", ".woff2",
    ".zip", ".gz", ".tar", ".xz", ".bz2", ".zst",
    ".wasm", ".o", ".a", ".so", ".dylib", ".dll",
    ".pyc", ".pyo", ".class",
    ".mp4", ".webm", ".ogg", ".mp3", ".wav",
    ".h5", ".hdf5", ".npy", ".npz",
    ".vo", ".vos", ".vok", ".glob",
})

SKIP_NAMES = frozenset({
    "Cargo.lock",
    "terminology_gate.py",  # do not scan ourselves
})

SKIP_SUFFIXES = (
    ".backup.phase2",  # historical registry snapshots
    ".backup",
)

# Historical / citation contexts where banned terms are legitimate
ALLOWLIST_PATTERNS = [
    re.compile(r"Toulouse\s*\(?1977\)?", re.IGNORECASE),
    re.compile(r"Harary.*frustration", re.IGNORECASE),
    re.compile(r"spin.glass\s+frustration", re.IGNORECASE),
    re.compile(r"Zaslavsky.*frustrat", re.IGNORECASE),
    re.compile(r"\\cite\{.*Toulouse", re.IGNORECASE),
    # TOML banned section itself (self-referential)
    re.compile(r'^pattern\s*=\s*"', re.IGNORECASE),
    re.compile(r'^replacement\s*=\s*"', re.IGNORECASE),
    re.compile(r'^reason\s*=\s*"', re.IGNORECASE),
]


def git_tracked_files(repo: Path) -> list[Path]:
    """Return list of git-tracked files relative to repo root."""
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"ERROR: git ls-files failed: {result.stderr}", file=sys.stderr)
        sys.exit(2)
    paths = [Path(p) for p in result.stdout.split("\0") if p]
    return paths


def should_skip(path: Path) -> bool:
    """Return True if path should be excluded from scanning."""
    if path.suffix.lower() in SKIP_EXTENSIONS:
        return True
    if path.name in SKIP_NAMES:
        return True
    name = path.name
    for sfx in SKIP_SUFFIXES:
        if name.endswith(sfx):
            return True
    # Skip anything under .git/ or build directories
    parts = path.parts
    if ".git" in parts or "target" in parts or "__pycache__" in parts:
        return True
    return False


def is_allowlisted(line: str) -> bool:
    """Return True if the line matches a known allowlist context."""
    for pat in ALLOWLIST_PATTERNS:
        if pat.search(line):
            return True
    return False


def load_banned_terms(toml_path: Path) -> list[dict[str, str]]:
    """Load [[banned]] entries from terminology_standards.toml."""
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    banned = data.get("banned", [])
    if not banned:
        print("WARNING: no [[banned]] entries in terminology_standards.toml", file=sys.stderr)
    return banned


def main() -> int:
    parser = argparse.ArgumentParser(description="Terminology gate")
    parser.add_argument("--quiet", action="store_true", help="suppress violation details")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parent.parent
    toml_path = repo / "registry" / "terminology_standards.toml"

    if not toml_path.exists():
        print(f"ERROR: {toml_path} not found", file=sys.stderr)
        return 2

    banned = load_banned_terms(toml_path)
    if not banned:
        return 0

    # Compile patterns (case-insensitive for prose terms, exact for identifiers)
    compiled = []
    for entry in banned:
        pat_str = entry["pattern"]
        # Use case-sensitive match for UPPER_CASE identifiers
        if pat_str == pat_str.upper() and "_" in pat_str:
            compiled.append((re.compile(re.escape(pat_str)), entry))
        else:
            compiled.append((re.compile(re.escape(pat_str), re.IGNORECASE), entry))

    files = git_tracked_files(repo)
    violations: list[tuple[Path, int, str, dict[str, str]]] = []

    for rel_path in files:
        if should_skip(rel_path):
            continue
        full_path = repo / rel_path
        if not full_path.is_file():
            continue
        try:
            text = full_path.read_text(encoding="utf-8", errors="replace")
        except (OSError, UnicodeDecodeError):
            continue

        for lineno, line in enumerate(text.splitlines(), start=1):
            if is_allowlisted(line):
                continue
            for regex, entry in compiled:
                if regex.search(line):
                    violations.append((rel_path, lineno, line.strip(), entry))

    if violations:
        if not args.quiet:
            print(f"FAIL: {len(violations)} terminology violation(s) found:\n")
            for rel_path, lineno, line_text, entry in violations:
                print(f"  {rel_path}:{lineno}")
                print(f"    found:   {entry['pattern']!r}")
                print(f"    replace: {entry['replacement']!r}")
                print(f"    reason:  {entry['reason']}")
                # Show truncated line context
                display = line_text[:120] + ("..." if len(line_text) > 120 else "")
                print(f"    line:    {display}")
                print()
        return 1
    else:
        if not args.quiet:
            print(f"OK: terminology gate passed ({len(compiled)} banned patterns, "
                  f"{len(files)} files scanned).")
        return 0


if __name__ == "__main__":
    sys.exit(main())
