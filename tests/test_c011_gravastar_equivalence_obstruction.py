"""
Claim C-011 falsifiability hooks.

These tests capture two reproducible properties behind the current
"closed/obstructed" status:
  1) The bridge table has complete gamma triplets and a monotone
     collapse pattern as gamma increases.
  2) The radial-stability sweep in the current parameter region has
     no Harrison-Wheeler stable branch.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
GENESIS_BRIDGE_CSV = REPO_ROOT / "data/csv/genesis_gravastar_bridge.csv"
RADIAL_STABILITY_CSV = REPO_ROOT / "data/csv/gravastar_radial_stability.csv"


def test_c011_bridge_triplets_show_monotone_gamma_collapse() -> None:
    with GENESIS_BRIDGE_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["seed"], row["soliton_id"])].append(row)

    assert grouped, "Bridge table should contain at least one soliton group."

    for key, group in grouped.items():
        ordered = sorted(group, key=lambda row: float(row["gamma"]))
        gammas = [float(row["gamma"]) for row in ordered]
        assert gammas == [1.5, 2.0, 2.5], f"{key} is missing the expected gamma triplet."

        masses = [float(row["M_total"]) for row in ordered]
        radii = [float(row["R2"]) for row in ordered]
        r1 = float(ordered[0]["R1"])

        assert masses[0] > masses[1] > masses[2], (
            f"{key} does not show monotone mass collapse with gamma."
        )
        assert radii[0] > radii[1] > radii[2], (
            f"{key} does not show monotone outer-radius collapse with gamma."
        )
        assert radii[2] / r1 <= 1.001, (
            f"{key} does not approach the thin-shell limit at gamma=2.5."
        )
        assert all(
            row["is_stable"].strip().lower() == "true" for row in ordered
        ), f"{key} should remain numerically stable in the bridge table."


def test_c011_radial_stability_scan_remains_obstructed() -> None:
    with RADIAL_STABILITY_CSV.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert rows, "Radial stability sweep should not be empty."

    derivatives = [float(row["dM_drho_c"]) for row in rows]
    assert all(value < 0.0 for value in derivatives), (
        "Expected dM/drho_c < 0 throughout the scanned radial-stability branch."
    )

    hw_stability = [
        row["harrison_wheeler_stable"].strip().lower() == "true" for row in rows
    ]
    assert not any(hw_stability), (
        "No Harrison-Wheeler stable point is expected in this obstructed scan."
    )
