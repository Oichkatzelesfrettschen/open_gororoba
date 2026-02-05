"""
C-090: SO(7) rotation-invariance drift audit for sedenion zero divisors.

Claim: The zero-divisor structure of the sedenion algebra is NOT invariant
under arbitrary SO(7) rotations of the imaginary octonion subspace.
The spectrum of the left-multiplication operator depends on the specific
embedding (choice of basis), and only the G2 automorphism subgroup of
SO(7) preserves the full octonionic multiplication table.

This script:
  1. Constructs a known sedenion zero-divisor pair (a, b) with a*b = 0.
  2. Applies random SO(7) rotations to the imaginary part (indices 1..7).
  3. Checks if the rotated pair still satisfies a'*b' = 0.
  4. Computes the "drift" ||a'*b'|| as a function of rotation angle.

Expected result:
  - Generic SO(7) rotations break the ZD condition (drift > 0).
  - G2 rotations (14-dim subgroup of SO(7)) preserve the ZD condition.
  - This demonstrates that octonionic structure depends on embedding.

Output: data/csv/c090_so7_rotation_drift_summary.csv

Refs:
  Baez, J.C. (2002), "The Octonions", Bull. AMS 39, 145-205.
  Harvey, F.R. (1990), "Spinors and Calibrations", Ch. 6.
"""
from __future__ import annotations

import csv
import os

import numpy as np
from numba import njit


@njit
def _cd_multiply(a, b, dim):
    """Cayley-Dickson multiplication on flat arrays."""
    if dim == 1:
        return np.array([a[0] * b[0]])
    half = dim // 2
    aL, aR = a[:half], a[half:]
    cL, cR = b[:half], b[half:]
    cR_conj = cR.copy()
    cR_conj[1:] = -cR_conj[1:]
    cL_conj = cL.copy()
    cL_conj[1:] = -cL_conj[1:]
    term1 = _cd_multiply(aL, cL, half)
    term2 = _cd_multiply(cR_conj, aR, half)
    L = term1 - term2
    term3 = _cd_multiply(cR, aL, half)
    term4 = _cd_multiply(aR, cL_conj, half)
    R = term3 + term4
    res = np.zeros(dim)
    res[:half] = L
    res[half:] = R
    return res


def sedenion_zd_pair():
    """
    Canonical sedenion zero-divisor pair from algebra.py.

    a = e1 + e10
    b = e4 - e15

    These satisfy a*b = 0 in the standard Cayley-Dickson basis
    (convention: e1*e2=e3, e1*e4=e5).
    """
    a = np.zeros(16)
    a[1] = 1.0
    a[10] = 1.0
    b = np.zeros(16)
    b[4] = 1.0
    b[15] = -1.0
    return a, b


def random_so7_rotation(rng, angle_scale=1.0):
    """
    Generate a random SO(7) rotation matrix.

    Acts on the 7-dimensional imaginary octonion subspace (indices 1..7
    within the first 8 components of a sedenion).

    Parameters
    ----------
    rng : numpy Generator
    angle_scale : float
        Controls the magnitude of the rotation (1.0 = full random).

    Returns
    -------
    R : ndarray, shape (7, 7)
        Orthogonal matrix with det +1.
    """
    # Random antisymmetric matrix -> matrix exponential gives SO(7).
    A = rng.standard_normal((7, 7))
    A = (A - A.T) / 2.0  # Antisymmetric.
    A *= angle_scale
    # Matrix exponential of antisymmetric matrix is orthogonal.
    from scipy.linalg import expm
    R = expm(A)
    return R


def apply_so7_rotation(x, R):
    """
    Apply SO(7) rotation R to the imaginary octonion part of a sedenion x.

    The sedenion is viewed as (octonion_L, octonion_R) where each
    octonion has components [real, im1, im2, ..., im7].
    The rotation acts on indices 1..7 of the LEFT octonion only.

    Parameters
    ----------
    x : ndarray, shape (16,)
    R : ndarray, shape (7, 7)

    Returns
    -------
    x_rot : ndarray, shape (16,)
    """
    x_rot = x.copy()
    # Rotate imaginary part of the left-half octonion.
    x_rot[1:8] = R @ x[1:8]
    return x_rot


def measure_zd_drift(n_rotations=100, angle_scale=1.0, seed=42):
    """
    Measure how much the ZD condition drifts under SO(7) rotations.

    Returns list of dicts with angle_scale, rotation_idx, product_norm.
    """
    rng = np.random.default_rng(seed)
    a, b = sedenion_zd_pair()

    # Verify unrotated pair is a ZD.
    ab = _cd_multiply(a, b, 16)
    assert np.linalg.norm(ab) < 1e-10, (
        f"Base pair not ZD: ||a*b||={np.linalg.norm(ab)}"
    )

    results = []
    for i in range(n_rotations):
        R = random_so7_rotation(rng, angle_scale)
        a_rot = apply_so7_rotation(a, R)
        b_rot = apply_so7_rotation(b, R)
        product = _cd_multiply(a_rot, b_rot, 16)
        norm = float(np.linalg.norm(product))
        results.append({
            "angle_scale": angle_scale,
            "rotation_idx": i,
            "product_norm": norm,
        })
    return results


def run_angle_sweep(scales=None, n_rotations=50, seed=42):
    """Sweep angle scales and measure drift at each."""
    if scales is None:
        scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0]
    all_results = []
    for s in scales:
        results = measure_zd_drift(n_rotations, s, seed)
        mean_norm = np.mean([r["product_norm"] for r in results])
        max_norm = np.max([r["product_norm"] for r in results])
        all_results.append({
            "angle_scale": s,
            "n_rotations": n_rotations,
            "mean_product_norm": float(mean_norm),
            "max_product_norm": float(max_norm),
            "fraction_broken": float(
                np.mean([r["product_norm"] > 1e-6 for r in results])
            ),
        })
    return all_results


def write_csv(results, path):
    """Write results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = run_angle_sweep()
    out = "data/csv/c090_so7_rotation_drift_summary.csv"
    write_csv(results, out)
    print(f"Wrote {out}")
    for r in results:
        print(
            f"  scale={r['angle_scale']:.2f}  "
            f"mean_norm={r['mean_product_norm']:.6f}  "
            f"broken={r['fraction_broken']:.0%}"
        )
