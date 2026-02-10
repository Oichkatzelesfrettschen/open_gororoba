#!/usr/bin/env python3
"""
Offline verifier for claims C-010 and C-011.

C-010 checks:
  - `sedenion_box_kites_clustered.csv` encodes 42 root indices split into
    seven six-node groups.
  - `cd_zd_absorber_mapping.csv` only proposes cross-group links, consistent
    with a non-local coupling requirement.

C-011 checks:
  - `genesis_gravastar_bridge.csv` contains complete gamma triplets
    (1.5, 2.0, 2.5) per soliton and monotone collapse of mass/radius.
  - `gravastar_radial_stability.csv` has no Harrison-Wheeler stable branch
    in the scanned region (all dM/drho_c < 0, all unstable).
"""
from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from math import isclose
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_root_pair(raw: str) -> tuple[int, int]:
    digits = re.findall(r"\d+", raw)
    if len(digits) != 2:
        raise ValueError(f"Could not parse root pair: {raw!r}")
    a, b = int(digits[0]), int(digits[1])
    return (a, b) if a < b else (b, a)


def _pair_from_row(row: dict[str, str], first: str, second: str) -> tuple[int, int]:
    a, b = int(row[first]), int(row[second])
    return (a, b) if a < b else (b, a)


def _check(condition: bool, errors: list[str], message: str) -> None:
    if not condition:
        errors.append(message)


def verify_c010(errors: list[str]) -> list[str]:
    messages: list[str] = []

    cluster_csv = REPO_ROOT / "data/csv/sedenion_box_kites_clustered.csv"
    mapping_csv = REPO_ROOT / "data/csv/cd_zd_absorber_mapping.csv"

    cluster_rows = _load_csv(cluster_csv)
    _check(len(cluster_rows) == 42, errors, "C-010: expected 42 box-kite root rows.")

    cluster_counts = Counter(int(row["Cluster_ID"]) for row in cluster_rows)
    _check(
        set(cluster_counts) == set(range(7)),
        errors,
        "C-010: expected cluster IDs 0..6.",
    )
    _check(
        set(cluster_counts.values()) == {6},
        errors,
        "C-010: expected 6 roots per cluster.",
    )

    cluster_by_pair = {
        _parse_root_pair(row["Root_Indices"]): int(row["Cluster_ID"]) for row in cluster_rows
    }
    _check(
        len(cluster_by_pair) == 42,
        errors,
        "C-010: root index pairs are not unique.",
    )

    mapping_rows = _load_csv(mapping_csv)
    _check(
        len(mapping_rows) == 20,
        errors,
        "C-010: expected 20 absorber bridge rows.",
    )

    material_types = {row["material_type"].strip().lower() for row in mapping_rows}
    _check(
        material_types == {"dielectric"},
        errors,
        "C-010: absorber bridge table should be dielectric-only.",
    )
    _check(
        all(row["is_physical"].strip().lower() == "true" for row in mapping_rows),
        errors,
        "C-010: all absorber bridge rows should be marked physical.",
    )
    layer_ids = sorted(int(row["layer_id"]) for row in mapping_rows)
    _check(
        layer_ids == list(range(len(mapping_rows))),
        errors,
        "C-010: expected contiguous layer_id values 0..N-1.",
    )
    thicknesses = [float(row["thickness_nm"]) for row in mapping_rows]
    _check(
        all(isclose(value, 10.0, rel_tol=0.0, abs_tol=1e-12) for value in thicknesses),
        errors,
        "C-010: expected fixed 10 nm bridge layers in canonical table.",
    )
    _check(
        all(float(row["n_real"]) > 0.0 and float(row["n_imag"]) >= 0.0 for row in mapping_rows),
        errors,
        "C-010: expected physically admissible refractive-index parameters.",
    )

    pair_edges = set()
    undirected_edges = set()
    cluster_graph: dict[int, set[int]] = {idx: set() for idx in range(7)}
    cross_cluster_edges = 0
    intra_cluster_edges = 0
    for row in mapping_rows:
        left = _pair_from_row(row, "i", "j")
        right = _pair_from_row(row, "k", "l")
        edge = (left, right)
        pair_edges.add(edge)
        undirected_edges.add(tuple(sorted((left, right))))

        if left not in cluster_by_pair:
            errors.append(f"C-010: missing cluster assignment for pair {left}.")
            continue
        if right not in cluster_by_pair:
            errors.append(f"C-010: missing cluster assignment for pair {right}.")
            continue
        c_left = cluster_by_pair[left]
        c_right = cluster_by_pair[right]
        if c_left != c_right:
            cross_cluster_edges += 1
            cluster_graph[c_left].add(c_right)
            cluster_graph[c_right].add(c_left)
        else:
            intra_cluster_edges += 1

    _check(
        cross_cluster_edges == len(mapping_rows),
        errors,
        "C-010: expected all absorber bridges to cross cluster boundaries.",
    )
    _check(
        intra_cluster_edges == 0,
        errors,
        "C-010: expected zero intra-cluster absorber bridges.",
    )
    _check(
        len(pair_edges) == len(mapping_rows) and len(undirected_edges) == len(mapping_rows),
        errors,
        "C-010: absorber bridge rows should be unique under directed and undirected pairing.",
    )

    degrees = {cluster: len(neighbors) for cluster, neighbors in cluster_graph.items()}
    isolated_clusters = [cluster for cluster, degree in degrees.items() if degree == 0]
    active_clusters = [cluster for cluster, degree in degrees.items() if degree > 0]
    _check(
        len(isolated_clusters) == 1,
        errors,
        f"C-010: expected exactly one isolated cluster in projected absorber graph, got {isolated_clusters}.",
    )
    _check(
        len(active_clusters) == 6,
        errors,
        f"C-010: expected 6 active clusters, got {active_clusters}.",
    )
    _check(
        all(degrees[cluster] == 4 for cluster in active_clusters),
        errors,
        "C-010: active cluster-projection degrees should all be exactly 4.",
    )

    if active_clusters:
        visited: set[int] = set()
        stack = [active_clusters[0]]
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            stack.extend(cluster_graph[node] - visited)
        _check(
            len(visited) == len(active_clusters),
            errors,
            "C-010: active projected cluster graph should be connected.",
        )

    messages.append(
        "C-010 OK: seven 6-node ZD groups verified; bridge graph is cross-group-only with one isolated cluster."
    )
    return messages


def verify_c011(errors: list[str]) -> list[str]:
    messages: list[str] = []

    bridge_csv = REPO_ROOT / "data/csv/genesis_gravastar_bridge.csv"
    radial_csv = REPO_ROOT / "data/csv/gravastar_radial_stability.csv"

    bridge_rows = _load_csv(bridge_csv)
    _check(
        len(bridge_rows) == 90,
        errors,
        "C-011: expected 90 bridge rows (30 soliton groups x 3 gammas).",
    )
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in bridge_rows:
        grouped[(row["seed"], row["soliton_id"])].append(row)
    _check(bool(grouped), errors, "C-011: bridge table has no soliton groups.")
    _check(
        len(grouped) == 30,
        errors,
        "C-011: expected 30 unique (seed, soliton_id) bridge groups.",
    )

    ratio_samples: list[float] = []
    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["gamma"]))
        gammas = [float(row["gamma"]) for row in ordered]
        _check(
            gammas == [1.5, 2.0, 2.5],
            errors,
            f"C-011: {key} missing gamma triplet 1.5/2.0/2.5.",
        )

        masses = [float(row["M_total"]) for row in ordered]
        radii = [float(row["R2"]) for row in ordered]
        r1 = float(ordered[0]["R1"])
        rho_v = [float(row["rho_v"]) for row in ordered]
        rho_shell = [float(row["rho_shell"]) for row in ordered]
        contrast = [vac / shell for vac, shell in zip(rho_v, rho_shell, strict=True)]
        ratio_samples.extend(contrast)

        _check(
            masses[0] > masses[1] > masses[2],
            errors,
            f"C-011: {key} mass is not monotone with gamma.",
        )
        _check(
            radii[0] > radii[1] > radii[2],
            errors,
            f"C-011: {key} radius is not monotone with gamma.",
        )
        _check(
            radii[2] / r1 <= 1.001,
            errors,
            f"C-011: {key} does not approach thin-shell limit at gamma=2.5.",
        )
        _check(
            all(row["is_stable"].strip().lower() == "true" for row in ordered),
            errors,
            f"C-011: {key} has non-stable bridge rows.",
        )
        _check(
            all(isclose(value, contrast[0], rel_tol=0.0, abs_tol=1e-12) for value in contrast),
            errors,
            f"C-011: {key} rho_v/rho_shell contrast is not gamma-invariant.",
        )

    _check(
        bool(ratio_samples),
        errors,
        "C-011: missing vacuum-to-shell contrast samples.",
    )
    if ratio_samples:
        contrast_ref = ratio_samples[0]
        _check(
            all(isclose(value, contrast_ref, rel_tol=0.0, abs_tol=1e-12) for value in ratio_samples),
            errors,
            "C-011: vacuum-to-shell contrast is not globally invariant across bridge rows.",
        )
        _check(
            isclose(contrast_ref, 10.0 / 3.0, rel_tol=0.0, abs_tol=1e-12),
            errors,
            f"C-011: expected rho_v/rho_shell = 10/3, observed {contrast_ref}.",
        )

    radial_rows = _load_csv(radial_csv)
    _check(bool(radial_rows), errors, "C-011: radial stability table is empty.")
    derivatives = [float(row["dM_drho_c"]) for row in radial_rows]
    _check(
        all(value < 0.0 for value in derivatives),
        errors,
        "C-011: expected dM/drho_c < 0 for all scanned radial points.",
    )
    _check(
        max(derivatives) < -100.0,
        errors,
        "C-011: radial branch should remain strongly negative (max dM/drho_c < -100).",
    )
    _check(
        all(row["harrison_wheeler_stable"].strip().lower() == "false" for row in radial_rows),
        errors,
        "C-011: expected no Harrison-Wheeler stable radial points.",
    )

    messages.append(
        "C-011 OK: bridge triplets verified; radial branch remains fully obstructed."
    )
    return messages


def main() -> int:
    errors: list[str] = []
    messages: list[str] = []
    messages.extend(verify_c010(errors))
    messages.extend(verify_c011(errors))

    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1

    for message in messages:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
