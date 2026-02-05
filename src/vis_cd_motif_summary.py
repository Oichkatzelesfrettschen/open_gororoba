from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def _collect_summary(dims: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dim in dims:
        comps_path = Path(f"data/csv/cd_motif_components_{dim}d.csv")
        if not comps_path.exists():
            continue
        df = pd.read_csv(comps_path)
        rows.append(
            {
                "dim": dim,
                "component_count": int(len(df)),
                "active_nodes_total": int(df["node_count"].sum()),
                "max_component_nodes": int(df["node_count"].max()),
                "max_component_edges": int(df["edge_count"].max()),
                "octahedron_k222_count": int(df.get("is_octahedron_k222", pd.Series(dtype=int)).sum())
                if "is_octahedron_k222" in df.columns
                else 0,
                "cuboctahedron_count": int(df.get("is_cuboctahedron", pd.Series(dtype=int)).sum())
                if "is_cuboctahedron" in df.columns
                else 0,
                "k2_multipartite_max_parts": int(df.get("k2_multipartite_part_count", pd.Series([0])).max())
                if "k2_multipartite_part_count" in df.columns
                else 0,
                "sampled": bool(df.get("sampled", pd.Series([False])).any())
                if "sampled" in df.columns
                else False,
                "sample_max_nodes": (
                    int(df["sample_max_nodes"].dropna().max())
                    if ("sample_max_nodes" in df.columns and not df["sample_max_nodes"].dropna().empty)
                    else 0
                ),
                "seed": (
                    int(df["seed"].dropna().max())
                    if ("seed" in df.columns and not df["seed"].dropna().empty)
                    else 0
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            [],
            columns=[
                "dim",
                "component_count",
                "active_nodes_total",
                "max_component_nodes",
                "max_component_edges",
                "octahedron_k222_count",
                "cuboctahedron_count",
                "k2_multipartite_max_parts",
                "sampled",
                "sample_max_nodes",
                "seed",
            ],
        )
    return pd.DataFrame(rows).sort_values(by=["dim"], ignore_index=True)


def main() -> None:
    dims = [16, 32, 64, 128, 256]
    summary = _collect_summary(dims)

    os.makedirs("data/csv", exist_ok=True)
    out_csv = Path("data/csv/cd_motif_summary_by_dim.csv")
    summary.to_csv(out_csv, index=False)

    spec = RenderSpec()
    _apply_dark_style(spec)
    fig_w = spec.width_px / spec.dpi
    fig_h = spec.height_px / spec.dpi
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h), dpi=spec.dpi)
    ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.35)

    if summary.empty:
        ax.set_title("Cayley-Dickson Motif Census Summary (no inputs found)")
        ax.text(
            0.02,
            0.98,
            "Missing: data/csv/cd_motif_components_{dim}d.csv",
            transform=ax.transAxes,
            va="top",
            alpha=0.85,
        )
    else:
        x = summary["dim"].tolist()
        y = summary["max_component_nodes"].tolist()
        bars = ax.bar(x, y, width=10, color="#00f7ff", alpha=0.9)

        for rect, row in zip(bars, summary.to_dict(orient="records"), strict=True):
            label = f"comps={row['component_count']}"
            cuboct = int(row.get("cuboctahedron_count", 0) or 0)
            k222 = int(row.get("octahedron_k222_count", 0) or 0)
            if cuboct:
                label += f", cuboct={cuboct}"
            if k222:
                label += f", k222={k222}"
            if row.get("sampled"):
                label += f", sampled<={row.get('sample_max_nodes', 0)}"
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + max(1.0, rect.get_height() * 0.01),
                label,
                ha="center",
                va="bottom",
                fontsize=12,
                alpha=0.85,
                rotation=0,
            )

        ax.set_xlabel("dimension (2^n)")
        ax.set_ylabel("max component node count")
        ax.set_title("Cayley-Dickson Cross-Assessor Motif Census -- Component Size vs Dimension")
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in x])

    out_png = Path("data/artifacts/images/cd_motif_max_component_nodes_3160x2820.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=spec.dpi, facecolor=spec.facecolor)

    print(f"Wrote: {out_csv}")
    print(f"Wrote: {out_png}")


if __name__ == "__main__":
    main()
