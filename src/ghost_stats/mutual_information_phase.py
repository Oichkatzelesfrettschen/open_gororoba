#!/usr/bin/env python3
"""Cross-algebra mutual information analysis of per-galaxy DFT phases.

Tests whether the quasi-degeneracy (rho > 0.97) between algebraic frameworks
hides non-linear dependencies detectable by mutual information (MI) but
invisible to Pearson correlation.

Uses the KSG (Kraskov-Stoegbauer-Grassberger) mutual information estimator
with circular phase embedding: phi -> (cos(phi), sin(phi)) to handle the
periodic boundary.

Input: cross_algebra_correlation.galaxies.csv
  Columns: plateifu, cd_re, cd_im, g2_re, g2_im, j3o_re, j3o_im, sl2_re, sl2_im

Reference: Hypothesis H4 from novel hypothesis plan.
"""

import csv
import math
import sys

import numpy as np

ALGEBRA_NAMES = ["CD-ZD", "G2", "J3O", "sl2"]
RE_COLS = [1, 3, 5, 7]  # 0-indexed column positions for re components


def load_galaxy_phases(csv_path: str) -> dict[str, np.ndarray]:
    """Load per-galaxy complex DFT coefficients, return phases per algebra."""
    phases = {name: [] for name in ALGEBRA_NAMES}

    with open(csv_path) as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            for i, name in enumerate(ALGEBRA_NAMES):
                re = float(row[RE_COLS[i]])
                im = float(row[RE_COLS[i] + 1])
                phi = math.atan2(im, re)
                phases[name].append(phi)

    return {name: np.array(vals) for name, vals in phases.items()}


def ksg_mutual_information(x: np.ndarray, y: np.ndarray, k: int = 5) -> float:
    """KSG mutual information estimator (Algorithm 1 from Kraskov et al. 2004).

    For circular data, x and y should already be embedded on the unit circle
    as 2D points: (cos(phi), sin(phi)).

    Parameters
    ----------
    x : (N, d_x) array
    y : (N, d_y) array
    k : number of nearest neighbors

    Returns
    -------
    MI estimate in nats.
    """
    from scipy.spatial import cKDTree
    from scipy.special import digamma

    n = len(x)
    if n < k + 1:
        return 0.0

    # Joint space.
    xy = np.hstack([x, y])
    tree_xy = cKDTree(xy)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # For each point, find distance to k-th neighbor in joint space.
    # query returns (distances, indices); k+1 because point itself is included.
    dists, _ = tree_xy.query(xy, k=k + 1)
    eps = dists[:, -1]  # distance to k-th neighbor (excluding self)

    # Count neighbors within eps in marginal spaces.
    # Use strict inequality (< eps) per KSG Algorithm 1.
    nx = np.array([tree_x.query_ball_point(x[i], eps[i] - 1e-15, return_length=True) - 1
                   for i in range(n)])
    ny = np.array([tree_y.query_ball_point(y[i], eps[i] - 1e-15, return_length=True) - 1
                   for i in range(n)])

    # Clamp to avoid log(0).
    nx = np.maximum(nx, 1)
    ny = np.maximum(ny, 1)

    mi = digamma(k) + digamma(n) - np.mean(digamma(nx) + digamma(ny))
    return max(mi, 0.0)


def entropy_ksg(x: np.ndarray, k: int = 5) -> float:
    """KSG entropy estimator using k-NN distances."""
    from scipy.special import digamma

    n, d = x.shape
    if n < k + 1:
        return 0.0

    from scipy.spatial import cKDTree
    tree = cKDTree(x)
    dists, _ = tree.query(x, k=k + 1)
    eps = dists[:, -1]

    # Entropy estimate (Kozachenko-Leonenko).
    log_eps = np.log(eps + 1e-30)
    h = d * np.mean(log_eps) + np.log(n) - digamma(k) + d * np.log(2.0)
    # Add volume of unit ball correction.
    h += d / 2.0 * np.log(np.pi) - math.lgamma(d / 2.0 + 1)
    return h


def embed_phase_on_circle(phi: np.ndarray) -> np.ndarray:
    """Embed circular phase on unit circle: phi -> (cos(phi), sin(phi))."""
    return np.column_stack([np.cos(phi), np.sin(phi)])


def main():
    if len(sys.argv) < 2:
        print("Usage: mutual_information_phase.py <galaxies_csv> [k=5]", file=sys.stderr)
        sys.exit(1)

    csv_path = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    print("=== H4: Cross-Algebra Mutual Information ===", file=sys.stderr)
    print(f"Loading from {csv_path}...", file=sys.stderr)

    phases = load_galaxy_phases(csv_path)
    n_gal = len(next(iter(phases.values())))
    print(f"Loaded {n_gal} galaxies, k={k} neighbors", file=sys.stderr)

    # Embed phases on unit circle.
    embedded = {name: embed_phase_on_circle(phi) for name, phi in phases.items()}

    # Compute pairwise MI and NMI.
    print("algebra_a,algebra_b,mi_nats,h_a,h_b,nmi,pearson_r")

    results = []
    for i, name_a in enumerate(ALGEBRA_NAMES):
        for j, name_b in enumerate(ALGEBRA_NAMES):
            if j <= i:
                continue

            x = embedded[name_a]
            y = embedded[name_b]

            mi = ksg_mutual_information(x, y, k=k)
            h_a = entropy_ksg(x, k=k)
            h_b = entropy_ksg(y, k=k)
            min_h = min(h_a, h_b)
            nmi = mi / min_h if min_h > 1e-10 else 0.0

            # Pearson correlation on raw phases for comparison.
            r = np.corrcoef(phases[name_a], phases[name_b])[0, 1]

            print(f"{name_a},{name_b},{mi:.8f},{h_a:.6f},{h_b:.6f},{nmi:.6f},{r:.6f}")

            results.append((name_a, name_b, mi, nmi, r))
            print(f"  {name_a} vs {name_b}: MI={mi:.6f} nats, NMI={nmi:.6f}, r={r:.6f}",
                  file=sys.stderr)

    print(file=sys.stderr)
    print("=== Verdict ===", file=sys.stderr)

    # Check NMI values.
    max_nmi = max(nmi for _, _, _, nmi, _ in results)
    min_nmi = min(nmi for _, _, _, nmi, _ in results)

    if max_nmi < 0.05:
        print("  All NMI < 0.05: trivial independence.", file=sys.stderr)
        print("  FALSIFICATION: no non-linear coupling detected -> REJECT.", file=sys.stderr)
    elif max_nmi > 0.15:
        print(f"  Max NMI = {max_nmi:.4f} > 0.15: unexpected cross-algebra coupling!",
              file=sys.stderr)
        high_pairs = [(a, b, nmi) for a, b, _, nmi, _ in results if nmi > 0.15]
        for a, b, nmi in high_pairs:
            print(f"    {a} vs {b}: NMI = {nmi:.4f}", file=sys.stderr)
    else:
        print(f"  NMI range [{min_nmi:.4f}, {max_nmi:.4f}]: weak coupling, inconclusive.",
              file=sys.stderr)

    # Check if Pearson r captures all the dependence.
    for name_a, name_b, _mi, nmi, r in results:
        # If |r| is high but NMI is low, linear correlation dominates.
        # If NMI >> |r|^2, there is non-linear structure.
        r2 = r * r
        if nmi > 0.05 and nmi > 2 * r2:
            print(f"  Non-linear coupling: {name_a} vs {name_b}: "
                  f"NMI={nmi:.4f} >> r^2={r2:.4f}", file=sys.stderr)


if __name__ == "__main__":
    main()
