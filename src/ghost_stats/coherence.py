#!/usr/bin/env python3
"""Pairwise magnitude-squared coherence between catalog pairs.

Magnitude-squared coherence (MSC) measures the linear relationship between
two signals at each frequency:

    C_xy(f) = |S_xy(f)|^2 / (S_xx(f) * S_yy(f))

where S_xy is the cross-spectral density and S_xx, S_yy are auto-spectral
densities. C_xy in [0, 1], with 1 indicating perfect linear relationship.

For K catalogs, there are C(K,2) pairs. We test whether the number of
pairs with significant coherence at f=0.214 exceeds what's expected under
the null hypothesis (binomial with p=alpha).

Usage:
    python coherence.py --inputs dir_of_csvs/ --column value
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy import signal as scipy_signal
from scipy import stats as scipy_stats


ALIASED_GHOST_FREQ = 1.0 - 0.786_151_377_757_423
FREQ_TOL = 0.02


def load_csv_column(path: str, column: str) -> np.ndarray:
    """Load a single column from a CSV file."""
    values = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                v = float(row[column])
                if np.isfinite(v):
                    values.append(v)
            except (ValueError, KeyError):
                continue
    return np.array(values)


def coherence_at_freq(
    x: np.ndarray,
    y: np.ndarray,
    target_freq: float,
    freq_tol: float,
    nperseg: int | None = None,
) -> dict:
    """Compute magnitude-squared coherence between two signals at target frequency.

    Uses Welch's method (scipy.signal.coherence) with overlapping segments.
    Significance is tested against the null distribution for MSC:
      P(C > c | H0) = (1 - c)^{L-1}
    where L is the number of averaged segments.
    """
    # Match lengths by truncating to the shorter
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    if nperseg is None:
        nperseg = min(256, n // 4)
        nperseg = max(nperseg, 16)

    # Compute coherence using Welch's method
    # fs=1.0 so frequencies are normalized to [0, 0.5]
    freqs, coh = scipy_signal.coherence(
        x, y, fs=1.0, nperseg=nperseg, noverlap=nperseg // 2
    )

    # Find bin nearest to target
    mask = np.abs(freqs - target_freq) < freq_tol
    if not mask.any():
        return {
            "coherence": 0.0,
            "p_value": 1.0,
            "freq": None,
            "n_segments": 0,
            "significant": False,
        }

    target_coh = coh[mask]
    target_freqs = freqs[mask]
    peak_idx = np.argmax(target_coh)
    coh_val = float(target_coh[peak_idx])
    coh_freq = float(target_freqs[peak_idx])

    # Number of averaged segments (Welch's method)
    noverlap = nperseg // 2
    step = nperseg - noverlap
    n_segments = max(1, (n - nperseg) // step + 1)

    # p-value under null hypothesis (independent signals)
    # P(C > c) = (1 - c)^{L-1} where L = n_segments
    p_value = float((1.0 - coh_val) ** (n_segments - 1))

    return {
        "coherence": coh_val,
        "p_value": p_value,
        "freq": coh_freq,
        "n_segments": n_segments,
        "significant": p_value < 0.05,
    }


def run_pairwise_coherence(
    datasets: dict[str, np.ndarray],
    target_freq: float = ALIASED_GHOST_FREQ,
    freq_tol: float = FREQ_TOL,
    alpha: float = 0.05,
) -> dict:
    """Run pairwise coherence test across all dataset pairs.

    Tests whether the number of significant pairs exceeds binomial expectation.
    """
    names = sorted(datasets.keys())
    k = len(names)
    n_pairs = k * (k - 1) // 2

    if n_pairs == 0:
        return {
            "n_datasets": k,
            "n_pairs": 0,
            "significant_pairs": 0,
            "expected_under_null": 0.0,
            "binomial_p": 1.0,
            "pair_results": [],
        }

    pair_results = []
    n_significant = 0

    for i in range(k):
        for j in range(i + 1, k):
            result = coherence_at_freq(
                datasets[names[i]],
                datasets[names[j]],
                target_freq,
                freq_tol,
            )
            result["pair"] = f"{names[i]} vs {names[j]}"
            pair_results.append(result)
            if result["significant"]:
                n_significant += 1

    # Binomial test: is the number of significant pairs > expected?
    expected = n_pairs * alpha
    # P(X >= n_significant) under Binomial(n_pairs, alpha)
    binom_p = float(1.0 - scipy_stats.binom.cdf(n_significant - 1, n_pairs, alpha))

    return {
        "n_datasets": k,
        "n_pairs": n_pairs,
        "significant_pairs": n_significant,
        "expected_under_null": expected,
        "binomial_p": binom_p,
        "alpha": alpha,
        "target_freq": target_freq,
        "verdict": "EXCESS" if binom_p < alpha else "CONSISTENT WITH NULL",
        "pair_results": pair_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Pairwise magnitude-squared coherence"
    )
    parser.add_argument(
        "--inputs",
        required=True,
        help="Directory containing CSV files, or comma-separated file paths",
    )
    parser.add_argument("--column", default="value", help="Column name")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Load datasets
    datasets = {}
    if os.path.isdir(args.inputs):
        for fname in sorted(os.listdir(args.inputs)):
            if fname.endswith(".csv"):
                fpath = os.path.join(args.inputs, fname)
                data = load_csv_column(fpath, args.column)
                if len(data) > 0:
                    datasets[fname] = data
    else:
        for fpath in args.inputs.split(","):
            fpath = fpath.strip()
            if os.path.isfile(fpath):
                data = load_csv_column(fpath, args.column)
                if len(data) > 0:
                    datasets[os.path.basename(fpath)] = data

    if len(datasets) < 2:
        print("ERROR: Need at least 2 datasets for coherence analysis", file=sys.stderr)
        sys.exit(1)

    result = run_pairwise_coherence(datasets, alpha=args.alpha)

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"Pairwise Coherence Analysis (alpha={args.alpha})")
        print(f"  Datasets: {result['n_datasets']}")
        print(f"  Pairs: {result['n_pairs']}")
        print(f"  Target: f={ALIASED_GHOST_FREQ:.6f}")
        print()
        for pr in result["pair_results"]:
            sig = " *" if pr["significant"] else ""
            print(
                f"  {pr['pair']}: C={pr['coherence']:.4f}, "
                f"p={pr['p_value']:.4e}, L={pr['n_segments']}{sig}"
            )
        print()
        print(f"  Significant pairs: {result['significant_pairs']}/{result['n_pairs']}")
        print(f"  Expected under null: {result['expected_under_null']:.1f}")
        print(f"  Binomial p-value: {result['binomial_p']:.6e}")
        print(f"  Verdict: {result['verdict']}")


if __name__ == "__main__":
    main()
