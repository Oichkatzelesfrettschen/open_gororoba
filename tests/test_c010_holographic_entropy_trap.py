"""
Claim C-010 falsifiability hooks.

These tests pin the current structural result behind the
"holographic entropy trap" thesis:
  1) Sedenion root indices are partitioned into seven disconnected
     six-node groups (42 total roots).
  2) The absorber mapping table only proposes cross-group couplings,
     which is evidence that explicit non-local bridges are required.
"""
from __future__ import annotations

import csv
import re
from collections import Counter
from math import isclose
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BOXKITE_CLUSTERS_CSV = REPO_ROOT / "data/csv/sedenion_box_kites_clustered.csv"
ABSORBER_MAPPING_CSV = REPO_ROOT / "data/csv/cd_zd_absorber_mapping.csv"


def _parse_root_pair(raw: str) -> tuple[int, int]:
    digits = re.findall(r"\d+", raw)
    assert len(digits) == 2, f"Could not parse root pair from {raw!r}"
    a, b = int(digits[0]), int(digits[1])
    return (a, b) if a < b else (b, a)


def _pair_from_row(row: dict[str, str], first: str, second: str) -> tuple[int, int]:
    a, b = int(row[first]), int(row[second])
    return (a, b) if a < b else (b, a)


def test_c010_boxkite_clusters_form_disconnected_septet() -> None:
    with BOXKITE_CLUSTERS_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 42, "Expected 42 root-index rows (7 groups x 6 roots)."

    cluster_counts = Counter(int(row["Cluster_ID"]) for row in rows)
    assert set(cluster_counts) == set(range(7)), "Expected exactly seven cluster IDs (0..6)."
    assert set(cluster_counts.values()) == {6}, "Each cluster should contain exactly six roots."

    roots = [_parse_root_pair(row["Root_Indices"]) for row in rows]
    assert len(set(roots)) == 42, "Root index pairs should be unique."


def test_c010_absorber_mapping_requires_cross_cluster_links() -> None:
    with BOXKITE_CLUSTERS_CSV.open("r", encoding="utf-8", newline="") as handle:
        cluster_rows = list(csv.DictReader(handle))
    cluster_by_pair = {
        _parse_root_pair(row["Root_Indices"]): int(row["Cluster_ID"]) for row in cluster_rows
    }

    with ABSORBER_MAPPING_CSV.open("r", encoding="utf-8", newline="") as handle:
        mapping_rows = list(csv.DictReader(handle))

    assert len(mapping_rows) == 20, "Expected the canonical 20-row absorber bridge table."

    material_types = {row["material_type"].strip().lower() for row in mapping_rows}
    assert material_types == {"dielectric"}, "Current mapping should use dielectric cells only."
    assert all(
        row["is_physical"].strip().lower() == "true" for row in mapping_rows
    ), "All canonical bridge rows should be physical."
    layer_ids = sorted(int(row["layer_id"]) for row in mapping_rows)
    assert layer_ids == list(range(len(mapping_rows))), (
        "Expected contiguous layer IDs from 0..N-1 in canonical bridge table."
    )
    thicknesses = [float(row["thickness_nm"]) for row in mapping_rows]
    assert all(isclose(value, 10.0, rel_tol=0.0, abs_tol=1e-12) for value in thicknesses), (
        "Canonical bridge table should keep fixed 10 nm layer thickness."
    )
    assert all(
        float(row["n_real"]) > 0.0 and float(row["n_imag"]) >= 0.0 for row in mapping_rows
    ), "Expected physically admissible refractive-index parameters."

    edge_pairs = set()
    undirected_edges = set()
    cluster_graph: dict[int, set[int]] = {idx: set() for idx in range(7)}
    cross_cluster_edges = 0
    intra_cluster_edges = 0
    for row in mapping_rows:
        left = _pair_from_row(row, "i", "j")
        right = _pair_from_row(row, "k", "l")
        edge_pairs.add((left, right))
        undirected_edges.add(tuple(sorted((left, right))))
        assert left in cluster_by_pair, f"Missing cluster assignment for {left}."
        assert right in cluster_by_pair, f"Missing cluster assignment for {right}."
        c_left = cluster_by_pair[left]
        c_right = cluster_by_pair[right]
        if c_left != c_right:
            cross_cluster_edges += 1
            cluster_graph[c_left].add(c_right)
            cluster_graph[c_right].add(c_left)
        else:
            intra_cluster_edges += 1

    assert cross_cluster_edges == len(mapping_rows), (
        "All mapped absorber bridges currently cross cluster boundaries, "
        "so local intra-cluster-only coupling is insufficient."
    )
    assert intra_cluster_edges == 0, "No intra-cluster absorber bridge should appear."
    assert len(edge_pairs) == len(mapping_rows), "Directed bridge rows must be unique."
    assert len(undirected_edges) == len(mapping_rows), "Undirected bridge rows must be unique."

    degrees = {cluster: len(neighbors) for cluster, neighbors in cluster_graph.items()}
    isolated_clusters = [cluster for cluster, degree in degrees.items() if degree == 0]
    active_clusters = [cluster for cluster, degree in degrees.items() if degree > 0]
    assert len(isolated_clusters) == 1, (
        "Projected bridge graph should leave exactly one isolated ZD cluster."
    )
    assert len(active_clusters) == 6, "Expected six active clusters in bridge projection."
    assert all(degrees[cluster] == 4 for cluster in active_clusters), (
        "Each active projected cluster should have degree four."
    )

    visited: set[int] = set()
    stack = [active_clusters[0]]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(cluster_graph[node] - visited)
    assert len(visited) == len(active_clusters), (
        "Active projected cluster graph should be connected."
    )
