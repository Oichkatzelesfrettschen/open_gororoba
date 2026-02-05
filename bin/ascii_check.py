#!/usr/bin/env python3
"""
Repo ASCII-only gate.

Policy:
- All repo-authored docs/code/data should be ASCII-only.
- Source transcripts under convos/ may contain Unicode; treat as immutable inputs.

Usage:
  python3 bin/ascii_check.py --check
  python3 bin/ascii_check.py --fix
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import unicodedata


REPLACEMENTS: dict[str, str] = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "--",
    "\u2212": "-",
    "\u2011": "-",
    "\u2192": "->",
    "\u21d2": "=>",
    "\u2026": "...",
    "\u00a0": " ",
    "\u221e": "infty",
    "\u00d7": "x",
    "\u00b2": "^2",
    "\u00b3": "^3",
    "\u00b9": "^1",
    "\u2080": "_0",
    "\u2081": "_1",
    "\u2082": "_2",
    "\u2083": "_3",
    "\u2084": "_4",
    "\u2085": "_5",
    "\u2086": "_6",
    "\u2087": "_7",
    "\u2088": "_8",
    "\u2089": "_9",
    "\u00c5": "Angstrom",
    "\u03b1": "\\alpha",
    "\u03b2": "\\beta",
    "\u03b3": "\\gamma",
    "\u03b4": "\\delta",
    "\u03b5": "\\epsilon",
    "\u03b8": "\\theta",
    "\u03bb": "\\lambda",
    "\u03bc": "\\mu",
    "\u03c0": "\\pi",
    "\u03c8": "\\psi",
    "\u0394": "\\Delta",
    "\u2206": "\\Delta",
    "\u2295": "\\oplus",
    "\u00f6": "o",
    "\u00fc": "u",
    "\u00e4": "a",
    "\u00e9": "e",
    "\u00f1": "n",
}


SKIP_DIRS = {
    ".git",
    "venv",
    "convos",
}

SKIP_PATH_PREFIXES = {
    "data/external/papers",
}

SKIP_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".pdf",
    ".xlsx",
    ".zip",
    ".tar",
    ".gz",
    ".bz2",
    ".xz",
    ".7z",
}


def iter_files(repo_root: Path) -> list[Path]:
    out: list[Path] = []
    for root, dirs, files in os.walk(repo_root):
        root_path = Path(root)
        rel = root_path.relative_to(repo_root)
        if rel.parts and rel.parts[0] in SKIP_DIRS:
            dirs[:] = []
            continue
        if any(part in SKIP_DIRS for part in rel.parts):
            dirs[:] = []
            continue
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            out.append(root_path / name)
    return out


def sanitize_text(text: str) -> str:
    for src, dst in REPLACEMENTS.items():
        text = text.replace(src, dst)
    # Drop accents and other combining marks (ASCII transliteration).
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    # Replace any remaining non-ASCII with explicit tokens so the output stays ASCII.
    out: list[str] = []
    for ch in text:
        if ord(ch) <= 127:
            out.append(ch)
        else:
            out.append(f"<U+{ord(ch):04X}>")
    return "".join(out)


def find_non_ascii(text: str) -> list[tuple[int, int, str]]:
    bad: list[tuple[int, int, str]] = []
    for line_idx, line in enumerate(text.splitlines(), start=1):
        for col_idx, ch in enumerate(line, start=1):
            if ord(ch) > 127:
                bad.append((line_idx, col_idx, ch))
                if len(bad) >= 10:
                    return bad
    return bad


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    if args.check == args.fix:
        raise SystemExit("Pass exactly one of: --check, --fix")

    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for path in iter_files(repo_root):
        if path.suffix.lower() in SKIP_EXTS:
            continue
        rel_posix = path.relative_to(repo_root).as_posix()
        if any(rel_posix.startswith(prefix) for prefix in SKIP_PATH_PREFIXES):
            continue
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue

        new_text = sanitize_text(text) if args.fix else text
        bad = find_non_ascii(new_text)
        if bad:
            failures.append(str(path.relative_to(repo_root)))
            continue

        if args.fix and new_text != text:
            path.write_text(new_text, encoding="utf-8")

    if failures:
        print("Non-ASCII files (first 10 chars per file not shown):")
        for fp in failures[:50]:
            print(f"  - {fp}")
        raise SystemExit(2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
