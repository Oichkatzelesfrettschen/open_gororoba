from __future__ import annotations

import csv
from pathlib import Path

from gemini_physics.m3_cd_transfer import classify_m3, compute_m3_octonion_basis


def main() -> int:
    out_path = Path("data/csv/m3_table.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["i", "j", "k", "kind", "index", "value"])

        for i in range(1, 8):
            for j in range(1, 8):
                for k in range(1, 8):
                    o_vec = compute_m3_octonion_basis(i, j, k)
                    c = classify_m3(o_vec)
                    counts[c.kind] = counts.get(c.kind, 0) + 1
                    w.writerow([i, j, k, c.kind, c.index, c.value])

    for kind in sorted(counts):
        print(f"{kind}: {counts[kind]}")
    print(f"Wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
