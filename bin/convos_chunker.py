#!/usr/bin/env python3
"""
Convos helper: compute deterministic chunk boundaries for a transcript file.

Example:
  python3 bin/convos_chunker.py --path convos/1_read_nonuser_lines_cont.md --chunk-lines 800 --prefix C1
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--chunk-lines", type=int, default=800)
    parser.add_argument("--prefix", type=str, default="C1")
    args = parser.parse_args()

    if args.chunk_lines <= 0:
        raise SystemExit("chunk-lines must be > 0")
    if not args.path.exists():
        raise SystemExit(f"Missing file: {args.path}")

    total = sum(1 for _ in args.path.open("r", encoding="utf-8", errors="replace"))
    n_chunks = (total + args.chunk_lines - 1) // args.chunk_lines

    print(f"path: {args.path}")
    print(f"lines: {total}")
    print(f"chunk_lines: {args.chunk_lines}")
    print(f"chunks: {n_chunks}")
    print()

    for i in range(n_chunks):
        start = i * args.chunk_lines + 1
        end = min((i + 1) * args.chunk_lines, total)
        chunk_id = f"{args.prefix}-{i+1:04d}"
        print(f"{chunk_id}: L{start}-L{end}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

