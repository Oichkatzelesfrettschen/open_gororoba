from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import pandas as pd

from gemini_physics.cd_motif_census import motif_components_for_cross_assessors


@dataclass(frozen=True)
class RunConfig:
    dims: tuple[int, ...]
    max_nodes: int | None
    seed: int
    summary_only: bool


def _write_dim(config: RunConfig, dim: int) -> None:
    comps = motif_components_for_cross_assessors(dim, max_nodes=config.max_nodes, seed=config.seed)

    comp_rows = []
    node_rows = []
    edge_rows = []
    for comp_id, comp in enumerate(comps):
        deg = comp.degree_sequence()
        k2_parts = comp.k2_multipartite_part_count()
        comp_rows.append(
            {
                "dim": dim,
                "component_id": comp_id,
                "node_count": len(comp.nodes),
                "edge_count": len(comp.edges),
                "degree_min": min(deg) if deg else 0,
                "degree_max": max(deg) if deg else 0,
                "degree_mean": (sum(deg) / len(deg)) if deg else 0.0,
                "is_octahedron_k222": comp.is_octahedron_graph(),
                "k2_multipartite_part_count": k2_parts,
                "is_cuboctahedron": comp.is_cuboctahedron_graph(),
                "sampled": config.max_nodes is not None,
                "sample_max_nodes": config.max_nodes,
                "seed": config.seed,
            }
        )
        if not config.summary_only:
            for low, high in sorted(comp.nodes):
                node_rows.append(
                    {"dim": dim, "component_id": comp_id, "low": low, "high": high}
                )
            for (a, b) in sorted(comp.edges):
                (a_low, a_high) = a
                (b_low, b_high) = b
                edge_rows.append(
                    {
                        "dim": dim,
                        "component_id": comp_id,
                        "a_low": a_low,
                        "a_high": a_high,
                        "b_low": b_low,
                        "b_high": b_high,
                    }
                )

    out_dir = "data/csv"
    os.makedirs(out_dir, exist_ok=True)
    comps_path = os.path.join(out_dir, f"cd_motif_components_{dim}d.csv")

    pd.DataFrame(comp_rows).to_csv(comps_path, index=False)

    print(f"Wrote: {comps_path}")
    if not config.summary_only:
        nodes_path = os.path.join(out_dir, f"cd_motif_nodes_{dim}d.csv")
        edges_path = os.path.join(out_dir, f"cd_motif_edges_{dim}d.csv")
        pd.DataFrame(node_rows).to_csv(nodes_path, index=False)
        pd.DataFrame(edge_rows).to_csv(edges_path, index=False)
        print(f"Wrote: {nodes_path}")
        print(f"Wrote: {edges_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Cayley-Dickson cross-assessor motif census CSVs.")
    ap.add_argument(
        "--dims",
        default="16,32",
        help="Comma-separated dimensions (powers of two), e.g. 16,32,64",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=None,
        help="Deterministic node sampling limit (for large dims).",
    )
    ap.add_argument("--seed", type=int, default=0, help="Sampling seed (only used with --max-nodes).")
    ap.add_argument(
        "--summary-only",
        action="store_true",
        help="Write only the components summary CSV (skip nodes/edges CSVs).",
    )
    args = ap.parse_args()

    dims = tuple(int(s.strip()) for s in args.dims.split(",") if s.strip())
    config = RunConfig(dims=dims, max_nodes=args.max_nodes, seed=args.seed, summary_only=args.summary_only)

    for dim in config.dims:
        _write_dim(config, dim)


if __name__ == "__main__":
    main()
