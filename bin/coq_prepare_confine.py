#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


HEADER = """From Stdlib Require Import String.
Require Import ConfineModel.

Open Scope string_scope.

"""


def transform(src: Path, dst: Path) -> None:
    text = src.read_text(encoding="utf-8")
    out_lines = []
    for line in text.splitlines():
        if line.startswith("Theorem "):
            out_lines.append("Axiom " + line[len("Theorem ") :])
        else:
            out_lines.append(line)
    dst.write_text(HEADER + "\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description="Make confine_theorems_*.v compilable by stubbing axioms.")
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    args = p.parse_args()

    args.dst.parent.mkdir(parents=True, exist_ok=True)
    transform(args.src, args.dst)
    print(f"Wrote {args.dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
