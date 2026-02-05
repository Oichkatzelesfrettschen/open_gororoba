#!/usr/bin/env python3
"""
Cleanup for ASCII placeholder tokens produced by bin/ascii_check.py --fix.

It replaces patterns like "<U+221A>" with readable ASCII approximations.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


TOKEN_MAP: dict[str, str] = {
    "<U+00B1>": "+/-",
    "<U+00B7>": "*",
    "<U+039B>": "Lambda",
    "<U+0393>": "Gamma",
    "<U+03A3>": "Sigma",
    "<U+03A8>": "\\Psi",
    "<U+03A6>": "\\Phi",
    "<U+03B7>": "eta",
    "<U+03BD>": "\\nu",
    "<U+03C1>": "\\rho",
    "<U+03C4>": "\\tau",
    "<U+03C6>": "\\phi",
    "<U+03C9>": "omega",
    "<U+2020>": "dagger",
    "<U+2194>": "<->",
    "<U+2193>": "down",
    "<U+21A6>": "|->",
    "<U+2202>": "\\partial",
    "<U+2203>": "exists",
    "<U+2207>": "\\nabla",
    "<U+2208>": "in",
    "<U+2218>": "circ",
    "<U+221A>": "sqrt",
    "<U+2212>": "-",
    "<U+222B>": "int",
    "<U+223C>": "~",
    "<U+2248>": "approx",
    "<U+2243>": "approx",
    "<U+2264>": "<=",
    "<U+2265>": ">=",
    "<U+2282>": "subseteq",
    "<U+2295>": "\\oplus",
    "<U+2297>": "\\otimes",
    "<U+22C6>": "\\star",
    "<U+22A5>": "bot",
    "<U+FE0F>": "",
    "<U+1F9E0>": "AI",
    "<U+2609>": "_sun",
}


SKIP_DIRS = {
    ".git",
    "venv",
    "convos",
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
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for name in files:
            if name.startswith("."):
                continue
            out.append(root_path / name)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--fix", action="store_true")
    args = parser.parse_args()

    if args.check == args.fix:
        raise SystemExit("Pass exactly one of: --check, --fix")

    repo_root = Path(__file__).resolve().parents[1]
    self_path = Path(__file__).resolve()
    pending: list[str] = []

    for path in iter_files(repo_root):
        if path.resolve() == self_path:
            continue
        if path.suffix.lower() in SKIP_EXTS:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if "<U+" not in text:
            continue

        new_text = text
        for k, v in TOKEN_MAP.items():
            new_text = new_text.replace(k, v)

        if args.check:
            if "<U+" in new_text:
                pending.append(str(path.relative_to(repo_root)))
            continue

        if new_text != text:
            path.write_text(new_text, encoding="utf-8")

    if args.check and pending:
        print("Files still containing <U+....> placeholders:")
        for fp in pending[:100]:
            print(f"  - {fp}")
        raise SystemExit(2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
