from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gemini_physics.de_marrais_boxkites import (
    box_kites,
    canonical_strut_table,
    diagonal_zero_products,
    edge_sign_type,
    primitive_assessors,
    strut_signature,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Export de Marrais assessor/box-kite structure to CSV.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/csv"),
        help="Output directory for CSVs (default: data/csv).",
    )
    args = parser.parse_args()

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    assessors = primitive_assessors()
    df_assessors = pd.DataFrame(assessors, columns=["low", "high"])
    assessors_path = out_dir / "de_marrais_assessors.csv"
    df_assessors.to_csv(assessors_path, index=False)

    bks = box_kites()
    rows_bk = []
    rows_edges = []
    rows_strut_table = []
    for bk_id, bk in enumerate(bks):
        sig = strut_signature(bk)
        for (low, high) in sorted(bk.assessors):
            rows_bk.append({"box_kite": bk_id, "strut_signature": sig, "low": low, "high": high})
        for a, b in sorted(bk.edges):
            sols = diagonal_zero_products(a, b)
            rows_edges.append(
                {
                    "box_kite": bk_id,
                    "strut_signature": sig,
                    "a_low": a[0],
                    "a_high": a[1],
                    "b_low": b[0],
                    "b_high": b[1],
                    "edge_type": edge_sign_type(a, b),
                    "sign_solutions": sols,
                }
            )
        tab = canonical_strut_table(bk)
        rows_strut_table.append(
            {
                "box_kite": bk_id,
                "strut_signature": sig,
                "A_low": tab["A"][0],
                "A_high": tab["A"][1],
                "B_low": tab["B"][0],
                "B_high": tab["B"][1],
                "C_low": tab["C"][0],
                "C_high": tab["C"][1],
                "D_low": tab["D"][0],
                "D_high": tab["D"][1],
                "E_low": tab["E"][0],
                "E_high": tab["E"][1],
                "F_low": tab["F"][0],
                "F_high": tab["F"][1],
            }
        )

    boxkites_path = out_dir / "de_marrais_boxkites.csv"
    pd.DataFrame(rows_bk).to_csv(boxkites_path, index=False)

    edges_path = out_dir / "de_marrais_boxkite_edges.csv"
    pd.DataFrame(rows_edges).to_csv(edges_path, index=False)

    struts_path = out_dir / "de_marrais_strut_table.csv"
    pd.DataFrame(rows_strut_table).to_csv(struts_path, index=False)

    print(f"Wrote: {assessors_path}")
    print(f"Wrote: {boxkites_path}")
    print(f"Wrote: {edges_path}")
    print(f"Wrote: {struts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
