#!/usr/bin/env python3
"""Extract const [f64; N] arrays from tabulated_nk.rs into per-material CSV files.

Usage:
  python3 scripts/extract_nk_arrays.py

Reads crates/materials_core/src/tabulated_nk.rs, finds every `const NAME: [f64; N] = [...]`
block, and writes the matched triples (EV, N, K) to crates/materials_data/data/nk/<name>.csv.
"""

import re
import sys
from pathlib import Path

SRC = Path("crates/materials_core/src/tabulated_nk.rs")
OUT = Path("crates/materials_data/data/nk")

# Regex: capture const name, size, and the array body (across multiple lines)
CONST_RE = re.compile(
    r"const\s+([A-Z0-9_]+)\s*:\s*\[f64\s*;\s*(\d+)\]\s*=\s*\[(.*?)\];",
    re.DOTALL,
)


def strip_comments(body: str) -> str:
    """Remove // ... line comments from an array body."""
    return re.sub(r"//[^\n]*", "", body)


def parse_floats(body: str) -> list[float]:
    body = strip_comments(body)
    tokens = re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?", body)
    return [float(t) for t in tokens]


def main() -> int:
    src = SRC.read_text()
    arrays: dict[str, list[float]] = {}

    for m in CONST_RE.finditer(src):
        name = m.group(1)
        size = int(m.group(2))
        body = m.group(3)
        values = parse_floats(body)
        if len(values) != size:
            print(f"WARNING: {name} expected {size} values, got {len(values)}", file=sys.stderr)
        arrays[name] = values

    print(f"Found {len(arrays)} const arrays: {list(arrays.keys())}")

    # Group arrays into materials.
    # Each material has a triple: <PREFIX>_EV, <PREFIX>_N, <PREFIX>_K.
    # Known prefixes and output filenames:
    groups = [
        ("JC_AU",      "au_jc1972"),
        ("JC_AG",      "ag_jc1972"),
        ("JC_CU",      "cu_jc1972"),
        ("SPLICED_AU", "au_ordal_jc"),
        ("SPLICED_CU", "cu_ordal_jc"),
        ("SPLICED3_AU","au_ordal_jc_henke"),
        ("SPLICED3_AG","ag_ordal_jc_henke"),
        ("SPLICED3_CU","cu_ordal_jc_henke"),
    ]

    OUT.mkdir(parents=True, exist_ok=True)
    written = 0

    for prefix, fname in groups:
        ev_key = f"{prefix}_EV"
        n_key  = f"{prefix}_N"
        k_key  = f"{prefix}_K"

        if ev_key not in arrays or n_key not in arrays or k_key not in arrays:
            print(f"SKIP {prefix}: missing one or more channels", file=sys.stderr)
            continue

        ev = arrays[ev_key]
        n  = arrays[n_key]
        k  = arrays[k_key]

        if not (len(ev) == len(n) == len(k)):
            print(
                f"ERROR {prefix}: mismatched lengths EV={len(ev)} N={len(n)} K={len(k)}",
                file=sys.stderr,
            )
            return 1

        out_path = OUT / f"{fname}.csv"
        with out_path.open("w") as f:
            f.write("energy_ev,n,k\n")
            for e, ni, ki in zip(ev, n, k):
                f.write(f"{e!r},{ni!r},{ki!r}\n")

        print(f"  Wrote {out_path} ({len(ev)} rows)")
        written += 1

    print(f"Done: {written} CSV files written to {OUT}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
