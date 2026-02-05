from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from PIL import Image

EXPECTED_PNG_SIZE = (3160, 2820)


def assert_png_size(path: Path) -> None:
    with Image.open(path) as img:
        if img.size != EXPECTED_PNG_SIZE:
            raise AssertionError(f"{path}: expected {EXPECTED_PNG_SIZE}, got {img.size}")


def main() -> int:
    checks = [
        Path("data/artifacts/images/dimensional_geometry_-4_to_16.png"),
        Path("data/artifacts/images/dimensional_geometry_0_to_32.png"),
        Path("data/artifacts/images/materials_pca_4d.png"),
        Path("data/artifacts/images/materials_pca_8d.png"),
        Path("data/artifacts/images/materials_pca_16d.png"),
        Path("data/artifacts/images/materials_pca_32d.png"),
        Path("data/artifacts/images/reggiani_annihilator_nullity_3160x2820.png"),
        Path("data/artifacts/images/cd_motif_max_component_nodes_3160x2820.png"),
    ]

    for p in checks:
        if not p.exists():
            print(f"SKIP: missing {p}")
            continue
        assert_png_size(p)
        print(f"OK: {p}")

    csv_checks = [
        ("data/csv/dimensional_geometry_-4_to_16.csv", ["d", "ball_volume_r1", "unit_sphere_area"]),
        ("data/csv/materials_jarvis_subset.csv", ["jid", "formula", "formation_energy_peratom"]),
        ("data/csv/materials_embedding_benchmarks.csv", ["k", "explained_variance_ratio_sum"]),
        ("data/csv/de_marrais_assessors.csv", ["low", "high"]),
        ("data/csv/de_marrais_boxkites.csv", ["box_kite", "strut_signature", "low", "high"]),
        ("data/csv/m3_table.csv", ["i", "j", "k", "kind", "index", "value"]),
        (
            "data/csv/de_marrais_boxkite_edges.csv",
            [
                "box_kite",
                "strut_signature",
                "a_low",
                "a_high",
                "b_low",
                "b_high",
                "edge_type",
                "sign_solutions",
            ],
        ),
        (
            "data/csv/de_marrais_strut_table.csv",
            [
                "box_kite",
                "strut_signature",
                "A_low",
                "A_high",
                "B_low",
                "B_high",
                "C_low",
                "C_high",
                "D_low",
                "D_high",
                "E_low",
                "E_high",
                "F_low",
                "F_high",
            ],
        ),
        (
            "data/csv/cd_motif_components_16d.csv",
            ["dim", "component_id", "node_count", "edge_count", "is_octahedron_k222", "k2_multipartite_part_count"],
        ),
        ("data/csv/cd_motif_nodes_16d.csv", ["dim", "component_id", "low", "high"]),
        ("data/csv/cd_motif_edges_16d.csv", ["dim", "component_id", "a_low", "a_high", "b_low", "b_high"]),
        (
            "data/csv/cd_motif_components_32d.csv",
            ["dim", "component_id", "node_count", "edge_count", "k2_multipartite_part_count"],
        ),
        ("data/csv/cd_motif_nodes_32d.csv", ["dim", "component_id", "low", "high"]),
        ("data/csv/cd_motif_edges_32d.csv", ["dim", "component_id", "a_low", "a_high", "b_low", "b_high"]),
        ("data/csv/cd_motif_components_64d.csv", ["dim", "component_id", "node_count", "edge_count"]),
        ("data/csv/cd_motif_components_128d.csv", ["dim", "component_id", "node_count", "edge_count"]),
        ("data/csv/cd_motif_components_256d.csv", ["dim", "component_id", "node_count", "edge_count", "sampled"]),
        (
            "data/csv/cd_motif_summary_by_dim.csv",
            ["dim", "component_count", "active_nodes_total", "max_component_nodes", "sampled"],
        ),
        (
            "data/csv/reggiani_standard_zero_divisors.csv",
            ["assessor_low", "assessor_high", "diagonal_sign", "left_nullity", "right_nullity"],
        ),
        (
            "data/csv/reggiani_standard_zero_divisor_pairs.csv",
            ["u_low", "u_high", "u_sign", "v_low", "v_high", "v_sign"],
        ),
    ]
    for path_str, required_cols in csv_checks:
        p = Path(path_str)
        if not p.exists():
            print(f"SKIP: missing {p}")
            continue
        df = pd.read_csv(p)
        for c in required_cols:
            if c not in df.columns:
                raise AssertionError(f"{p}: missing required column {c!r}")
        print(f"OK: {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
