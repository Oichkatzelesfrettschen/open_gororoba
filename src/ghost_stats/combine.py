#!/usr/bin/env python3
"""Meta-analysis p-value combination methods.

Combines per-dataset p-values from multiple independent tests into a single
summary statistic. Three methods are implemented:

1. Fisher's method: chi2 = -2 * sum(log(p_i)), distributed as chi-squared(2k)
   under H0. Optimal when effect sizes are similar. Sensitive to a single
   very small p-value.

2. Stouffer's weighted Z: Z = sum(w_i * Phi^{-1}(1-p_i)) / sqrt(sum(w_i^2))
   with weights w_i = sqrt(N_i). Optimal when studies have different sample
   sizes. Standard normal under H0.

3. Harmonic mean p-value (Wilson 2019, PNAS 116:1195): p_HM = sum(w_i) / sum(w_i/p_i)
   Robust combination with Landau distribution correction. Bounded by
   Bonferroni: p_HM <= K * min(p_i). Less sensitive to single extreme values.

Reference:
  - Fisher 1932, Statistical Methods for Research Workers
  - Stouffer et al. 1949, The American Soldier
  - Wilson 2019, PNAS 116:1195 (harmonic mean p-value)
  - Held et al. 2025, RSM 16:758 (method comparison)

Usage:
    python combine.py --input results.csv --p-column p_value --n-column n_samples --method all
"""
import argparse
import csv
import sys

import numpy as np
from scipy import stats as scipy_stats


def fisher_combine(p_values: list[float]) -> dict:
    """Fisher's method for combining p-values.

    chi2 = -2 * sum(log(p_i))
    Under H0, chi2 ~ chi-squared(2k).
    """
    k = len(p_values)
    # Clamp p-values away from 0 to avoid -inf
    p_safe = [max(p, 1e-300) for p in p_values]
    chi2_stat = -2.0 * sum(np.log(p) for p in p_safe)
    combined_p = float(1.0 - scipy_stats.chi2.cdf(chi2_stat, df=2 * k))

    return {
        "method": "fisher",
        "statistic": float(chi2_stat),
        "combined_p": combined_p,
        "df": 2 * k,
        "n_studies": k,
    }


def stouffer_weighted_z(
    p_values: list[float], weights: list[float] | None = None
) -> dict:
    """Stouffer's weighted Z-method for combining p-values.

    Z = sum(w_i * z_i) / sqrt(sum(w_i^2))
    where z_i = Phi^{-1}(1 - p_i) and w_i = sqrt(N_i).
    Under H0, Z ~ N(0, 1).
    """
    k = len(p_values)
    if weights is None:
        weights = [1.0] * k

    # Convert p-values to z-scores
    z_scores = []
    for p in p_values:
        p_clamped = np.clip(p, 1e-15, 1.0 - 1e-15)
        z = float(scipy_stats.norm.ppf(1.0 - p_clamped))
        z_scores.append(z)

    # Weighted combination
    w = np.array(weights)
    z = np.array(z_scores)
    combined_z = float(np.sum(w * z) / np.sqrt(np.sum(w**2)))
    combined_p = float(1.0 - scipy_stats.norm.cdf(combined_z))

    return {
        "method": "stouffer",
        "statistic": combined_z,
        "combined_p": combined_p,
        "z_scores": z_scores,
        "weights": list(weights),
        "n_studies": k,
    }


def harmonic_mean_p(p_values: list[float], weights: list[float] | None = None) -> dict:
    """Harmonic mean p-value (Wilson 2019).

    p_HM = sum(w_i) / sum(w_i / p_i)

    This has the useful property that p_HM <= K * min(p_i) (bounded by
    Bonferroni), making it a valid combined test without requiring
    independence.

    The corrected version accounts for the asymptotic Landau distribution.
    """
    k = len(p_values)
    if weights is None:
        weights = [1.0 / k] * k

    # Normalize weights
    w = np.array(weights)
    w = w / np.sum(w)

    # Harmonic mean
    p_arr = np.array(p_values)
    p_safe = np.maximum(p_arr, 1e-300)
    hm = float(np.sum(w) / np.sum(w / p_safe))

    # Asymptotically corrected p-value (Wilson 2019, Eq. 4)
    # For the corrected version, p_corrected ~ e * ln(K) * p_HM for large K
    # Simple bound: p_corrected = min(1, e * ln(K) * p_HM) for K >= 2
    if k >= 2:
        correction = np.e * np.log(k)
        corrected_p = min(1.0, correction * hm)
    else:
        corrected_p = hm

    return {
        "method": "harmonic_mean",
        "statistic": hm,
        "combined_p": float(corrected_p),
        "uncorrected_hm": float(hm),
        "correction_factor": float(correction) if k >= 2 else 1.0,
        "n_studies": k,
    }


def run_combination(
    p_values: list[float],
    sample_sizes: list[int] | None = None,
    methods: list[str] | None = None,
) -> dict:
    """Run all requested p-value combination methods."""
    if methods is None:
        methods = ["fisher", "stouffer", "harmonic_mean"]

    k = len(p_values)
    if sample_sizes is None:
        sample_sizes = [100] * k  # default equal weights

    # Stouffer weights = sqrt(N_i)
    stouffer_weights = [float(np.sqrt(n)) for n in sample_sizes]

    results = {}

    if "fisher" in methods:
        results["fisher"] = fisher_combine(p_values)

    if "stouffer" in methods:
        results["stouffer"] = stouffer_weighted_z(p_values, weights=stouffer_weights)

    if "harmonic_mean" in methods:
        results["harmonic_mean"] = harmonic_mean_p(p_values)

    # Summary
    all_combined_p = [r["combined_p"] for r in results.values()]
    min_combined_p = min(all_combined_p) if all_combined_p else 1.0
    max_combined_p = max(all_combined_p) if all_combined_p else 1.0

    # Consensus: all methods agree on significance?
    any_significant = any(r["combined_p"] < 0.05 for r in results.values())
    all_significant = all(r["combined_p"] < 0.05 for r in results.values())

    return {
        "input_p_values": p_values,
        "sample_sizes": sample_sizes,
        "results": results,
        "min_combined_p": min_combined_p,
        "max_combined_p": max_combined_p,
        "any_significant": any_significant,
        "all_significant": all_significant,
        "consensus": "SIGNIFICANT" if all_significant else (
            "MIXED" if any_significant else "NOT SIGNIFICANT"
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Meta-analysis p-value combination")
    parser.add_argument(
        "--input",
        help="CSV file with p-values (one row per dataset)",
    )
    parser.add_argument("--p-column", default="p_value", help="Column with p-values")
    parser.add_argument("--n-column", default="n_samples", help="Column with sample sizes")
    parser.add_argument(
        "--p-values",
        help="Comma-separated p-values (alternative to --input)",
    )
    parser.add_argument(
        "--n-values",
        help="Comma-separated sample sizes (alternative to --input)",
    )
    parser.add_argument(
        "--method",
        default="all",
        choices=["fisher", "stouffer", "harmonic_mean", "all"],
        help="Combination method",
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    # Load p-values
    p_values = []
    sample_sizes = []

    if args.p_values:
        p_values = [float(p) for p in args.p_values.split(",")]
        if args.n_values:
            sample_sizes = [int(n) for n in args.n_values.split(",")]
        else:
            sample_sizes = [100] * len(p_values)
    elif args.input:
        with open(args.input) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    p_values.append(float(row[args.p_column]))
                    if args.n_column in row:
                        sample_sizes.append(int(float(row[args.n_column])))
                    else:
                        sample_sizes.append(100)
                except (ValueError, KeyError):
                    continue
    else:
        print("ERROR: Provide --input or --p-values", file=sys.stderr)
        sys.exit(1)

    if len(p_values) == 0:
        print("ERROR: No p-values loaded", file=sys.stderr)
        sys.exit(1)

    methods = (
        ["fisher", "stouffer", "harmonic_mean"]
        if args.method == "all"
        else [args.method]
    )

    result = run_combination(p_values, sample_sizes, methods)

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"P-value Combination ({len(p_values)} datasets)")
        print(f"  Input p-values: {p_values}")
        print(f"  Sample sizes: {sample_sizes}")
        print()

        for name, r in result["results"].items():
            print(f"  {name.upper()}:")
            print(f"    Statistic: {r['statistic']:.4f}")
            print(f"    Combined p: {r['combined_p']:.6e}")
            verdict = "SIGNIFICANT" if r["combined_p"] < 0.05 else "NOT SIGNIFICANT"
            print(f"    Verdict: {verdict}")
            print()

        print(f"  Consensus: {result['consensus']}")
        print(f"  Range: [{result['min_combined_p']:.6e}, {result['max_combined_p']:.6e}]")


if __name__ == "__main__":
    main()
