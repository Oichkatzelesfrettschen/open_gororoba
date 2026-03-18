#!/usr/bin/env python3
"""Gaussian Process model comparison: periodic vs smooth kernel.

Fits two GP models to the 16-bin stacked rotation curve residual:
  Model A (null): Matern kernel (smooth baryonic structure only)
  Model B (ZD):   Matern + ExpSineSquared kernel (periodic at CD-ZD frequency)

Compares via log marginal likelihood difference:
  Delta_LML = LML(B) - LML(A)

If Delta_LML > 3:  periodic component improves the fit (BIC-equivalent)
If Delta_LML < 1:  periodic component adds nothing
If Delta_LML < 0:  periodic component makes it WORSE (complexity penalty)

This is textbook GP model selection (Rasmussen & Williams 2006, Ch. 5).
16 data points is ideal for GP -- no scalability concerns, exact inference.

Reference: Hypothesis P2 from practical ML engineer plan.
"""

import csv
import sys

import numpy as np


def load_stacked_csv(path: str, min_per_bin: int = 10):
    """Load stacked CSV, filter to valid bins."""
    x, delta, err, n = [], [], [], []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            nc = int(row["n_contributing"])
            if nc >= min_per_bin:
                x.append(float(row["x"]))
                delta.append(float(row["delta_stack"]))
                err.append(float(row["delta_stack_err"]))
                n.append(nc)
    return np.array(x), np.array(delta), np.array(err), np.array(n)


def main():
    if len(sys.argv) < 2:
        print("Usage: gp_model_comparison.py <stacked_csv>", file=sys.stderr)
        sys.exit(1)

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import (
        ExpSineSquared,
        Matern,
        WhiteKernel,
    )

    csv_path = sys.argv[1]
    print("=== P2: GP Model Comparison (Periodic vs Smooth) ===", file=sys.stderr)

    x, delta, err, n = load_stacked_csv(csv_path)
    n_bins = len(x)
    print(f"Loaded {n_bins} valid bins (x={x[0]:.3f} to {x[-1]:.3f})", file=sys.stderr)

    X = x.reshape(-1, 1)
    y = delta

    # The CD-ZD fundamental period: k_1 = 2*pi/7, so period = 7 in x-units.
    zd_period = 7.0

    # --- Model A: Matern kernel (smooth baryonic structure) ---
    # Matern(nu=2.5) + WhiteKernel (measurement noise)
    kernel_a = (
        Matern(length_scale=0.3, length_scale_bounds=(0.05, 5.0), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )

    gp_a = GaussianProcessRegressor(
        kernel=kernel_a,
        n_restarts_optimizer=20,
        normalize_y=True,
    )
    gp_a.fit(X, y)
    lml_a = gp_a.log_marginal_likelihood_value_

    print("\nModel A (Matern only):", file=sys.stderr)
    print(f"  Kernel: {gp_a.kernel_}", file=sys.stderr)
    print(f"  LML = {lml_a:.4f}", file=sys.stderr)

    # --- Model B: Matern + ExpSineSquared (periodic at ZD frequency) ---
    kernel_b = (
        Matern(length_scale=0.3, length_scale_bounds=(0.05, 5.0), nu=2.5)
        + ExpSineSquared(
            length_scale=0.5,
            periodicity=zd_period,
            length_scale_bounds=(0.1, 5.0),
            periodicity_bounds=(3.0, 20.0),  # allow optimizer to explore
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )

    gp_b = GaussianProcessRegressor(
        kernel=kernel_b,
        n_restarts_optimizer=20,
        normalize_y=True,
    )
    gp_b.fit(X, y)
    lml_b = gp_b.log_marginal_likelihood_value_

    print("\nModel B (Matern + Periodic):", file=sys.stderr)
    print(f"  Kernel: {gp_b.kernel_}", file=sys.stderr)
    print(f"  LML = {lml_b:.4f}", file=sys.stderr)

    # --- Model C: Just periodic (no smooth component) ---
    kernel_c = (
        ExpSineSquared(
            length_scale=0.5,
            periodicity=zd_period,
            length_scale_bounds=(0.1, 5.0),
            periodicity_bounds=(3.0, 20.0),
        )
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )

    gp_c = GaussianProcessRegressor(
        kernel=kernel_c,
        n_restarts_optimizer=20,
        normalize_y=True,
    )
    gp_c.fit(X, y)
    lml_c = gp_c.log_marginal_likelihood_value_

    print("\nModel C (Periodic only):", file=sys.stderr)
    print(f"  Kernel: {gp_c.kernel_}", file=sys.stderr)
    print(f"  LML = {lml_c:.4f}", file=sys.stderr)

    # --- Comparison ---
    delta_lml_ba = lml_b - lml_a
    delta_lml_ca = lml_c - lml_a

    print("\n--- Model Comparison ---", file=sys.stderr)
    print(f"  Delta_LML (B-A, Matern+Periodic vs Matern): {delta_lml_ba:.4f}", file=sys.stderr)
    print(f"  Delta_LML (C-A, Periodic vs Matern):        {delta_lml_ca:.4f}", file=sys.stderr)

    # Bayes factor interpretation.
    # Delta_LML = ln(BF). BF > 20 (Delta > 3) = strong evidence.
    # BF > 150 (Delta > 5) = very strong evidence.
    # BF < 1 (Delta < 0) = evidence AGAINST periodic component.
    bf_ba = np.exp(delta_lml_ba)
    bf_ca = np.exp(delta_lml_ca)
    print(f"  Bayes factor (B vs A): {bf_ba:.4f}", file=sys.stderr)
    print(f"  Bayes factor (C vs A): {bf_ca:.4f}", file=sys.stderr)

    # Output CSV.
    print("model,kernel_description,lml,delta_lml_vs_a,bayes_factor_vs_a,n_params")
    n_params_a = len(gp_a.kernel_.get_params())
    n_params_b = len(gp_b.kernel_.get_params())
    n_params_c = len(gp_c.kernel_.get_params())
    print(f"A_matern,\"{gp_a.kernel_}\",{lml_a:.6f},0.000000,1.000000,{n_params_a}")
    print(f"B_matern_periodic,\"{gp_b.kernel_}\",{lml_b:.6f},{delta_lml_ba:.6f},{bf_ba:.6f},{n_params_b}")
    print(f"C_periodic_only,\"{gp_c.kernel_}\",{lml_c:.6f},{delta_lml_ca:.6f},{bf_ca:.6f},{n_params_c}")

    # --- GP predictions for visual inspection ---
    x_pred = np.linspace(x.min() - 0.1, x.max() + 0.1, 100).reshape(-1, 1)
    y_a, std_a = gp_a.predict(x_pred, return_std=True)
    y_b, std_b = gp_b.predict(x_pred, return_std=True)

    print("\n--- Predictive uncertainty ---", file=sys.stderr)
    print(f"  Model A mean RMS residual: {np.sqrt(np.mean((gp_a.predict(X) - y)**2)):.6f}",
          file=sys.stderr)
    print(f"  Model B mean RMS residual: {np.sqrt(np.mean((gp_b.predict(X) - y)**2)):.6f}",
          file=sys.stderr)

    # Verdict.
    print("\n=== Verdict ===", file=sys.stderr)
    if delta_lml_ba < 0:
        print(f"  Delta_LML = {delta_lml_ba:.4f} < 0: periodic component makes fit WORSE.",
              file=sys.stderr)
        print("  The complexity penalty outweighs any improvement.", file=sys.stderr)
        print("  CONFIRMED: smooth baryonic model is sufficient.", file=sys.stderr)
    elif delta_lml_ba < 1:
        print(f"  Delta_LML = {delta_lml_ba:.4f} < 1: periodic component adds nothing.",
              file=sys.stderr)
        print("  CONFIRMED: no evidence for periodic structure.", file=sys.stderr)
    elif delta_lml_ba < 3:
        print(f"  Delta_LML = {delta_lml_ba:.4f}: weak evidence for periodic component.",
              file=sys.stderr)
        print("  Inconclusive -- could be noise fitting.", file=sys.stderr)
    else:
        print(f"  Delta_LML = {delta_lml_ba:.4f} > 3: significant periodic component!",
              file=sys.stderr)
        opt_period = None
        for key, val in gp_b.kernel_.get_params().items():
            if "periodicity" in key and isinstance(val, float):
                opt_period = val
        if opt_period is not None:
            print(f"  Optimized period = {opt_period:.4f} (predicted = {zd_period:.4f})",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
