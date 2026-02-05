from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gemini_physics.reggiani_replication import (
    standard_zero_divisor_partners,
    standard_zero_divisors,
)
from gemini_physics.sedenion_annihilator import annihilator_info


@dataclass(frozen=True)
class RenderSpec:
    width_px: int = 3160
    height_px: int = 2820
    dpi: int = 100
    facecolor: str = "#0d0f14"


def _apply_dark_style(spec: RenderSpec) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": spec.facecolor,
            "axes.facecolor": spec.facecolor,
            "axes.edgecolor": "#1f2937",
            "axes.labelcolor": "white",
            "xtick.color": "white",
            "ytick.color": "white",
            "text.color": "white",
            "grid.color": "#374151",
            "font.size": 16,
            "font.family": "sans-serif",
        }
    )


def main() -> None:
    out_csv = "data/csv/reggiani_standard_zero_divisors.csv"
    out_dist_csv = "data/csv/reggiani_annihilator_nullity_distribution.csv"
    out_pairs_csv = "data/csv/reggiani_standard_zero_divisor_pairs.csv"
    out_png = "data/artifacts/images/reggiani_annihilator_nullity_3160x2820.png"

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    rows = []
    pair_counts: dict[tuple[int, int], int] = {}
    pair_rows = []
    for zd in standard_zero_divisors():
        info = annihilator_info(zd.vector)
        rows.append(
            {
                "assessor_low": zd.assessor_low,
                "assessor_high": zd.assessor_high,
                "diagonal_sign": zd.diagonal_sign,
                "squared_norm": float(np.dot(zd.vector, zd.vector)),
                "left_nullity": info.left_nullity,
                "right_nullity": info.right_nullity,
            }
        )
        pair_counts[(info.left_nullity, info.right_nullity)] = pair_counts.get(
            (info.left_nullity, info.right_nullity), 0
        ) + 1

        for p in standard_zero_divisor_partners(zd):
            pair_rows.append(
                {
                    "u_low": zd.assessor_low,
                    "u_high": zd.assessor_high,
                    "u_sign": zd.diagonal_sign,
                    "v_low": p.assessor_low,
                    "v_high": p.assessor_high,
                    "v_sign": p.diagonal_sign,
                }
            )

    df = pd.DataFrame(rows).sort_values(
        by=["assessor_low", "assessor_high", "diagonal_sign"], ignore_index=True
    )
    df.to_csv(out_csv, index=False)

    dist_rows = [
        {"left_nullity": ln, "right_nullity": rn, "count": count}
        for (ln, rn), count in sorted(pair_counts.items())
    ]
    dist = pd.DataFrame(dist_rows)
    dist.to_csv(out_dist_csv, index=False)

    pairs_df = pd.DataFrame(pair_rows).sort_values(
        by=["u_low", "u_high", "u_sign", "v_low", "v_high", "v_sign"], ignore_index=True
    )
    pairs_df.to_csv(out_pairs_csv, index=False)

    # Plot distribution (most runs should show a single spike at (4,4) for standard ZDs).
    spec = RenderSpec()
    _apply_dark_style(spec)
    fig_w = spec.width_px / spec.dpi
    fig_h = spec.height_px / spec.dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=spec.dpi)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)

    labels = [f"({r['left_nullity']},{r['right_nullity']})" for r in dist_rows]
    counts = [r["count"] for r in dist_rows]
    ax.bar(labels, counts, color="#00f7ff", alpha=0.9)
    ax.set_xlabel("(left nullity, right nullity)")
    ax.set_ylabel("count")
    ax.set_title("Reggiani (2024) -- Standard Sedenion Zero Divisors: Annihilator Nullity Distribution")
    ax.text(
        0.01,
        0.01,
        "Computed from the 84 standard zero divisors (two diagonals for each of de Marrais' 42 assessors).",
        transform=ax.transAxes,
        fontsize=14,
        alpha=0.8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(out_png, dpi=spec.dpi, facecolor=spec.facecolor)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_dist_csv}")
    print(f"Wrote: {out_pairs_csv}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
