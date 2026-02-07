"""
Tests for C-090: SO(7) rotation-invariance drift.

Validates:
  1. Unrotated ZD pair has product norm < 1e-10.
  2. Any non-trivial SO(7) rotation breaks the ZD condition.
  3. Drift magnitude grows with rotation angle scale.
  4. At full random rotation (scale=1.0), all pairs are broken.
"""
from __future__ import annotations

import numpy as np

from scripts.analysis.c090_so7_rotation_drift_audit import (
    _cd_multiply,
    measure_zd_drift,
    run_angle_sweep,
    sedenion_zd_pair,
)


def test_base_pair_is_zero_divisor() -> None:
    """The canonical ZD pair should have product norm < 1e-10."""
    a, b = sedenion_zd_pair()
    ab = _cd_multiply(a, b, 16)
    norm = np.linalg.norm(ab)
    assert norm < 1e-10, f"||a*b||={norm:.2e}, expected < 1e-10"


def test_small_rotation_breaks_zd() -> None:
    """Even small SO(7) rotations (scale=0.05) should break the ZD condition."""
    results = measure_zd_drift(n_rotations=20, angle_scale=0.05, seed=42)
    broken = sum(1 for r in results if r["product_norm"] > 1e-6)
    assert broken == 20, (
        f"Only {broken}/20 pairs broken at scale=0.05, expected all"
    )


def test_drift_grows_with_angle() -> None:
    """Mean product norm should increase with rotation scale."""
    summary = run_angle_sweep(
        scales=[0.01, 0.1, 0.5, 1.0], n_rotations=20, seed=42,
    )
    norms = [r["mean_product_norm"] for r in summary]
    for i in range(len(norms) - 1):
        assert norms[i] < norms[i + 1], (
            f"Drift not monotone: scale={summary[i]['angle_scale']} "
            f"({norms[i]:.4f}) >= scale={summary[i+1]['angle_scale']} "
            f"({norms[i+1]:.4f})"
        )


def test_full_rotation_all_broken() -> None:
    """At scale=1.0, all rotated ZD pairs should be broken."""
    summary = run_angle_sweep(
        scales=[1.0], n_rotations=30, seed=42,
    )
    assert summary[0]["fraction_broken"] == 1.0, (
        f"fraction_broken={summary[0]['fraction_broken']:.2f}, expected 1.0"
    )
