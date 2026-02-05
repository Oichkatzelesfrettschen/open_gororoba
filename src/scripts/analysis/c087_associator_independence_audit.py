"""
C-087: Associator norm independence audit.

Claim: For random unit elements a, b, c in a Cayley-Dickson algebra of
dimension d >= 16, the expected squared associator norm approaches 2:

  E[||[a,b,c]||^2] -> 2  as  d -> infinity

where [a,b,c] = (ab)c - a(bc).

Mechanism: By the parallelogram identity,
  ||[a,b,c]||^2 = ||(ab)c||^2 + ||a(bc)||^2 - 2*Re((ab)c . conj(a(bc)))

The cross-term Re((ab)c . conj(a(bc))) vanishes as d -> infinity because
the two products become statistically independent (their correlation
decays). Since the composition property fails for d >= 16, both
||(ab)c||^2 and ||a(bc)||^2 converge to 1 for unit inputs, giving
E[||A||^2] -> 2.

This script computes Monte Carlo estimates of:
  - E[||[a,b,c]||^2]
  - E[||(ab)c||^2]
  - E[||a(bc)||^2]
  - correlation = E[Re((ab)c . a(bc))]  (the cross-term)

for dimensions 4, 8, 16, 32, 64, 128.

Output: data/csv/c087_associator_independence_summary.csv

Refs:
  Cayley-Dickson construction: Schafer (1966), "An Introduction to
    Nonassociative Algebras".
"""
from __future__ import annotations

import csv
import os

import numpy as np
from numba import njit


@njit
def _cd_multiply(a, b, dim):
    """Cayley-Dickson multiplication: (a,b)(c,d) = (ac - d*b, da + bc*)."""
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


def random_unit(dim, rng):
    """Generate a random unit vector in R^dim."""
    v = rng.standard_normal(dim)
    return v / np.linalg.norm(v)


def compute_associator_stats(dim, n_trials=2000, seed=42):
    """
    Monte Carlo estimate of associator norm statistics.

    Returns dict with keys:
      dim, n_trials, mean_assoc_sq, mean_abc_sq, mean_ab_c_sq,
      mean_cross_term, correlation_coeff.
    """
    rng = np.random.default_rng(seed)

    assoc_sq_vals = np.empty(n_trials)
    abc_sq_vals = np.empty(n_trials)
    ab_c_sq_vals = np.empty(n_trials)
    cross_vals = np.empty(n_trials)

    for i in range(n_trials):
        a = random_unit(dim, rng)
        b = random_unit(dim, rng)
        c = random_unit(dim, rng)

        ab = _cd_multiply(a, b, dim)
        bc = _cd_multiply(b, c, dim)
        ab_c = _cd_multiply(ab, c, dim)
        a_bc = _cd_multiply(a, bc, dim)

        assoc = ab_c - a_bc
        assoc_sq_vals[i] = np.dot(assoc, assoc)
        ab_c_sq_vals[i] = np.dot(ab_c, ab_c)
        abc_sq_vals[i] = np.dot(a_bc, a_bc)
        cross_vals[i] = np.dot(ab_c, a_bc)

    return {
        "dim": dim,
        "n_trials": n_trials,
        "mean_assoc_sq": float(np.mean(assoc_sq_vals)),
        "std_assoc_sq": float(np.std(assoc_sq_vals)),
        "mean_ab_c_sq": float(np.mean(ab_c_sq_vals)),
        "mean_a_bc_sq": float(np.mean(abc_sq_vals)),
        "mean_cross_term": float(np.mean(cross_vals)),
        "correlation_coeff": float(
            np.mean(cross_vals)
            / np.sqrt(np.mean(ab_c_sq_vals) * np.mean(abc_sq_vals))
        ),
    }


def run_sweep(dims=None, n_trials=2000, seed=42):
    """Run associator stats for multiple dimensions."""
    if dims is None:
        dims = [4, 8, 16, 32, 64, 128]
    return [compute_associator_stats(d, n_trials, seed) for d in dims]


def write_csv(results, path):
    """Write results to CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(results[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    results = run_sweep()
    out = "data/csv/c087_associator_independence_summary.csv"
    write_csv(results, out)
    print(f"Wrote {out}")
    for r in results:
        print(
            f"  dim={r['dim']:4d}  E[||A||^2]={r['mean_assoc_sq']:.4f}  "
            f"corr={r['correlation_coeff']:.6f}"
        )
