#!/usr/bin/env python3
"""CPD (Copy-Paste Detector) XML report formatter.

Reads PMD CPD XML output from stdin, prints a human-readable summary, and
exits with code 1 when --strict is given and any duplication clusters are found.

Usage:
  pmd cpd --language rust --minimum-tokens 42 --dir crates --format xml 2>/dev/null \
    | python3 scripts/cpd_report.py [--strict] [--top N]

Exit codes:
  0  No clusters found, or --strict not given.
  1  --strict given and at least one cluster found.
  2  PMD/XML parse error (empty or malformed input).
"""

import sys
import xml.etree.ElementTree as ET
import argparse


def short_path(path: str) -> str:
    for marker in ("crates/", "src/"):
        idx = path.find(marker)
        if idx >= 0:
            return path[idx:]
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strict", action="store_true",
                        help="Exit 1 if any clusters are found.")
    parser.add_argument("--top", type=int, default=20,
                        help="Show top N clusters by line count (default: 20).")
    args = parser.parse_args()

    raw = sys.stdin.read()
    if not raw.strip():
        print("cpd-report: no XML received (pmd produced no output).", file=sys.stderr)
        return 2

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        print(f"cpd-report: XML parse error: {exc}", file=sys.stderr)
        return 2

    dups = root.findall("duplication")
    total_lines = sum(int(d.get("lines", 0)) for d in dups)

    print(f"CPD audit  --minimum-tokens 42  Rust")
    print(f"Clusters : {len(dups)}")
    print(f"Dup lines: {total_lines}")

    if not dups:
        print("No duplication found.")
        return 0

    # Sort by line count descending, show top N.
    dups_sorted = sorted(dups, key=lambda d: int(d.get("lines", 0)), reverse=True)
    show = dups_sorted[: args.top]

    print()
    print(f"{'#':<4}  {'Lines':>6}  {'Tokens':>7}  Locations")
    print("-" * 72)
    for i, dup in enumerate(show, 1):
        lines = int(dup.get("lines", 0))
        tokens = int(dup.get("tokens", 0))
        locs = [
            f"{short_path(f.get('path','?'))}:{f.get('line','?')}"
            for f in dup.findall("file")
        ]
        loc_str = "  /  ".join(locs)
        print(f"{i:<4}  {lines:>6}  {tokens:>7}  {loc_str}")
        frag = (dup.findtext("codefragment") or "").strip()
        if frag:
            preview = frag.replace("\n", " ")[:90]
            print(f"       {preview!r}")

    if len(dups) > args.top:
        print(f"  ... {len(dups) - args.top} more clusters not shown.")

    if args.strict:
        print(f"\ncpd-audit-strict: FAIL -- {len(dups)} cluster(s) found.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
