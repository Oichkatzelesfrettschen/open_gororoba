#!/usr/bin/env python3
"""Quantile/Laplace periodogram for rank-based spectral analysis.

The quantile periodogram (Li 2012, Bernoulli 2015) performs spectral analysis
using quantile regression rather than least-squares, making it robust to outliers
and specifically designed for distributional data. This is the CORRECT method
for sorted catalog data where "frequency" describes distributional shape.

At each frequency f and quantile level tau, we fit:
    y_t = alpha + beta_c * cos(2*pi*f*t) + beta_s * sin(2*pi*f*t) + epsilon_t

using the check function rho_tau(u) = u*(tau - I(u<0)) as loss. The quantile
periodogram statistic is Q(f, tau) = ||beta||^2 normalized by scale.

Fisher's g-test for hidden periodicity: g = max(Q) / sum(Q), tested against
the exact null distribution (under white noise, Q values are iid exponential).

Reference: Li 2012, JASA 107:502; arXiv:2403.02060 (2024 extension)

Usage:
    python quantile_periodogram.py --input data.csv --column value --quantiles 0.1,0.25,0.5,0.75,0.9
"""
import argparse
import csv
import sys

import numpy as np
from scipy.optimize import linprog


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


def quantile_regression(
    X: np.ndarray, y: np.ndarray, tau: float
) -> np.ndarray:
    """Solve quantile regression via linear programming.

    Minimizes sum of rho_tau(y - X*beta) where rho_tau(u) = u*(tau - I(u<0)).

    This is reformulated as:
        min tau * sum(u+) + (1-tau) * sum(u-)
        s.t. X*beta + u+ - u- = y, u+>=0, u->=0

    Uses scipy.optimize.linprog with the HiGHS solver.
    """
    n, p = X.shape

    # Decision variables: [beta (p), u+ (n), u- (n)]
    # Objective: 0*beta + tau*u+ + (1-tau)*u-
    c = np.zeros(p + 2 * n)
    c[p : p + n] = tau
    c[p + n :] = 1.0 - tau

    # Equality constraints: X*beta + u+ - u- = y
    A_eq = np.zeros((n, p + 2 * n))
    A_eq[:, :p] = X
    A_eq[:, p : p + n] = np.eye(n)
    A_eq[:, p + n :] = -np.eye(n)
    b_eq = y

    # Bounds: beta unbounded, u+>=0, u->=0
    bounds = [(None, None)] * p + [(0, None)] * (2 * n)

    result = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        return result.x[:p]
    else:
        return np.zeros(p)


def quantile_periodogram_at_freq(
    signal: np.ndarray, freq: float, tau: float
) -> float:
    """Compute quantile periodogram statistic at a single frequency and quantile.

    Fits: y_t = alpha + beta_c*cos(2*pi*f*t) + beta_s*sin(2*pi*f*t)
    using quantile regression at level tau.

    Returns Q(f, tau) = beta_c^2 + beta_s^2 (unnormalized power).
    """
    n = len(signal)
    t = np.arange(n, dtype=float)

    # Design matrix: [1, cos(2*pi*f*t), sin(2*pi*f*t)]
    X = np.column_stack(
        [
            np.ones(n),
            np.cos(2.0 * np.pi * freq * t),
            np.sin(2.0 * np.pi * freq * t),
        ]
    )

    beta = quantile_regression(X, signal, tau)
    return float(beta[1] ** 2 + beta[2] ** 2)


def quantile_periodogram(
    signal: np.ndarray,
    freqs: np.ndarray,
    tau: float = 0.5,
) -> np.ndarray:
    """Compute quantile periodogram at all frequencies for a given quantile level.

    This is O(n * n_freq * LP_cost) -- expensive but correct for distributional data.
    """
    powers = np.zeros(len(freqs))
    for i, f in enumerate(freqs):
        powers[i] = quantile_periodogram_at_freq(signal, f, tau)
    return powers


def fisher_g_test(periodogram: np.ndarray) -> tuple[float, float]:
    """Fisher's exact g-test for hidden periodicity.

    g = max(I(f_j)) / sum(I(f_j))

    Under H0 (white noise), the p-value is:
    P(g > x) = sum_{k=1}^{M} (-1)^{k+1} * C(M,k) * max(1-k*x, 0)^{M-1}

    where M = number of frequency bins.
    """
    total = np.sum(periodogram)
    if total <= 0:
        return 0.0, 1.0

    g = float(np.max(periodogram) / total)
    m = len(periodogram)

    # Exact p-value computation
    p_value = 0.0
    for k in range(1, m + 1):
        term = 1.0 - k * g
        if term <= 0:
            break
        from scipy.special import comb

        sign = (-1.0) ** (k + 1)
        p_value += sign * comb(m, k, exact=False) * term ** (m - 1)

    p_value = float(np.clip(p_value, 0.0, 1.0))
    return g, p_value


def run_quantile_periodogram_test(
    signal: np.ndarray,
    target_freq: float = ALIASED_GHOST_FREQ,
    freq_tol: float = FREQ_TOL,
    quantiles: list[float] | None = None,
    n_freq: int = 100,
) -> dict:
    """Run quantile periodogram analysis at multiple quantile levels.

    Uses a coarse frequency grid (n_freq points) because quantile regression
    is O(n^2) per frequency. Focuses around target frequency.
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]

    n = len(signal)

    # Frequency grid: focus around target +/- tolerance, plus broader scan
    freq_min = max(1.0 / n, 0.01)
    freq_max = 0.5
    freqs = np.linspace(freq_min, freq_max, n_freq)

    # Add fine grid around target
    fine_freqs = np.linspace(
        max(freq_min, target_freq - freq_tol),
        min(freq_max, target_freq + freq_tol),
        20,
    )
    freqs = np.unique(np.sort(np.concatenate([freqs, fine_freqs])))

    results_per_quantile = {}
    combined_fisher_inputs = []

    for tau in quantiles:
        powers = quantile_periodogram(signal, freqs, tau=tau)

        # Find peak near target
        mask = np.abs(freqs - target_freq) < freq_tol
        if mask.any():
            target_powers = powers[mask]
            target_freqs_masked = freqs[mask]
            peak_idx = np.argmax(target_powers)
            peak_freq = float(target_freqs_masked[peak_idx])
            peak_power = float(target_powers[peak_idx])
        else:
            peak_freq = None
            peak_power = 0.0

        # Fisher's g-test on the full periodogram
        g, p_fisher = fisher_g_test(powers)

        # Check if the global max is near target
        global_max_idx = np.argmax(powers)
        global_max_freq = float(freqs[global_max_idx])
        global_max_near_target = abs(global_max_freq - target_freq) < freq_tol

        results_per_quantile[f"tau_{tau:.2f}"] = {
            "peak_freq": peak_freq,
            "peak_power": peak_power,
            "fisher_g": g,
            "fisher_p": p_fisher,
            "global_max_freq": global_max_freq,
            "global_max_near_target": global_max_near_target,
        }

        combined_fisher_inputs.append(p_fisher)

    # Combine across quantiles using Fisher's method
    combined_chi2 = -2.0 * sum(
        np.log(max(p, 1e-300)) for p in combined_fisher_inputs
    )
    from scipy.stats import chi2

    combined_p = float(1.0 - chi2.cdf(combined_chi2, df=2 * len(quantiles)))

    return {
        "quantiles": quantiles,
        "per_quantile": results_per_quantile,
        "combined_chi2": float(combined_chi2),
        "combined_p_fisher": combined_p,
        "n_frequencies": len(freqs),
        "n_samples": n,
        "target_freq": target_freq,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Quantile periodogram spectral analysis"
    )
    parser.add_argument("--input", required=True, help="CSV file path")
    parser.add_argument("--column", default="value", help="Column name")
    parser.add_argument(
        "--quantiles",
        default="0.1,0.25,0.5,0.75,0.9",
        help="Comma-separated quantile levels",
    )
    parser.add_argument(
        "--n-freq", type=int, default=100, help="Number of frequency grid points"
    )
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    values = load_csv_column(args.input, args.column)
    if len(values) == 0:
        print(f"ERROR: No valid data in column '{args.column}'", file=sys.stderr)
        sys.exit(1)

    quantiles = [float(q) for q in args.quantiles.split(",")]

    result = run_quantile_periodogram_test(
        values, quantiles=quantiles, n_freq=args.n_freq
    )

    if args.json:
        import json

        print(json.dumps(result, indent=2))
    else:
        print(f"Quantile Periodogram (tau = {quantiles})")
        print(f"  Input: {args.input} [{args.column}]")
        print(f"  N = {result['n_samples']}, {result['n_frequencies']} freq bins")
        print(f"  Target: f={ALIASED_GHOST_FREQ:.6f}")
        print()
        for tau_key, tau_result in result["per_quantile"].items():
            print(f"  {tau_key}:")
            print(f"    Peak near target: f={tau_result['peak_freq']}, power={tau_result['peak_power']:.6e}")
            print(f"    Fisher g={tau_result['fisher_g']:.6f}, p={tau_result['fisher_p']:.6e}")
            near = "YES" if tau_result["global_max_near_target"] else "NO"
            print(f"    Global max at f={tau_result['global_max_freq']:.6f} (near target: {near})")
        print()
        print(f"  Combined Fisher chi2: {result['combined_chi2']:.4f}")
        print(f"  Combined p-value: {result['combined_p_fisher']:.6e}")
        verdict = (
            "SIGNIFICANT"
            if result["combined_p_fisher"] < 0.05
            else "NOT SIGNIFICANT"
        )
        print(f"  Verdict: {verdict}")


if __name__ == "__main__":
    main()
