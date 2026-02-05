#!/usr/bin/env python3
"""
Polytropic gravastar parameter sweep for stability analysis.

Sweeps over:
  - Polytropic exponent gamma in [1.0, 2.5]
  - Target mass M_target in [5, 80] solar masses
  - Core compactness in [0.5, 0.9]
  - Anisotropy lambda_aniso in [0.0, 2.0]

Produces CSV artifacts for:
  1. Isotropic polytropic sweep
  2. Anisotropic extension (Bowers-Liang form)

The Harrison-Wheeler stability criterion (dM/d(rho_c) > 0) identifies
stable configurations. For stiff matter (gamma=1), all solutions lie on
an unstable branch. Polytropic shells with gamma >= 4/3 can yield stable
configurations. Anisotropic pressure further extends the stable domain.

References:
  Visser & Wiltshire CQG 21 (2004) -- Original gravastar model
  Cattoen, Faber & Visser gr-qc/0505137 (2005) -- Anisotropic gravastars
  Das, Debnath & Ray (2024) -- Polytropic thin-shell gravastars
  Bowers & Liang ApJ 188 (1974) -- Anisotropic stars

Output artifacts:
  data/csv/gravastar_polytropic_sweep.csv
  data/csv/gravastar_anisotropic_stability.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from gravastar_tov import (
    solve_gravastar,
    solve_gravastar_anisotropic,
    solve_gravastar_for_mass,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CSV_DIR = PROJECT_ROOT / "data" / "csv"


def compute_stability_derivative(rho_v, R1, rho_shell_center, gamma, K,
                                  delta_frac=0.01, aniso_lambda=None):
    """
    Compute dM/d(rho_shell) numerically to check Harrison-Wheeler stability.

    Parameters
    ----------
    rho_v : float
        Vacuum energy density.
    R1 : float
        Inner shell radius.
    rho_shell_center : float
        Central shell density for finite difference.
    gamma : float
        Polytropic exponent.
    K : float
        Polytropic constant.
    delta_frac : float
        Fractional perturbation for finite difference.
    aniso_lambda : float or None
        If not None, use anisotropic solver with this lambda.

    Returns
    -------
    dM_drho : float
        Numerical derivative dM/d(rho_shell).
    M_center : float
        Mass at central density.
    stable : bool
        True if dM/drho > 0 (stable branch).
    """
    delta = delta_frac * rho_shell_center
    rho_minus = rho_shell_center - delta
    rho_plus = rho_shell_center + delta

    solver = (solve_gravastar_anisotropic if aniso_lambda is not None
              else solve_gravastar)

    try:
        if aniso_lambda is not None:
            res_minus = solver(rho_v, R1, rho_minus, gamma=gamma, K=K,
                               lambda_aniso=aniso_lambda)
            res_center = solver(rho_v, R1, rho_shell_center, gamma=gamma, K=K,
                                lambda_aniso=aniso_lambda)
            res_plus = solver(rho_v, R1, rho_plus, gamma=gamma, K=K,
                              lambda_aniso=aniso_lambda)
        else:
            res_minus = solver(rho_v, R1, rho_minus, gamma=gamma, K=K)
            res_center = solver(rho_v, R1, rho_shell_center, gamma=gamma, K=K)
            res_plus = solver(rho_v, R1, rho_plus, gamma=gamma, K=K)

        M_minus = res_minus['M_total']
        M_center = res_center['M_total']
        M_plus = res_plus['M_total']

        # Central difference
        dM_drho = (M_plus - M_minus) / (2 * delta)
        stable = dM_drho > 0

        return dM_drho, M_center, stable
    except (ValueError, RuntimeError):
        return np.nan, np.nan, False


def run_polytropic_sweep():
    """Run isotropic polytropic parameter sweep."""
    print("Running polytropic gravastar sweep...")

    gamma_values = np.array([1.0, 1.1, 1.2, 1.3, 4.0/3.0, 1.4, 1.5, 1.75, 2.0, 2.5])
    M_targets = np.array([5, 10, 20, 35, 50, 65, 80])
    compactness_values = np.array([0.5, 0.6, 0.7, 0.8, 0.9])

    rows = []

    for gamma in gamma_values:
        for M_target in M_targets:
            for C in compactness_values:
                try:
                    result = solve_gravastar_for_mass(
                        m_target=M_target,
                        core_compactness=C,
                        gamma=gamma,
                        K=1.0,
                    )

                    # Compute stability
                    rho_v = result['rho'][0]  # vacuum density in core
                    R1 = result['R1']
                    # Estimate shell density from first shell point
                    shell_mask = (result['r'] > R1) & (result['rho'] > 0)
                    if np.any(shell_mask):
                        rho_shell = result['rho'][shell_mask][0]
                    else:
                        rho_shell = rho_v * 3.0

                    dM_drho, _, stable = compute_stability_derivative(
                        rho_v, R1, rho_shell, gamma, 1.0
                    )

                    rows.append({
                        'gamma': gamma,
                        'M_target': M_target,
                        'core_compactness': C,
                        'R1': result['R1'],
                        'R2': result['R2'],
                        'M_total': result['M_total'],
                        'compactness_2M_R2': 2.0 * result['M_total'] / result['R2'],
                        'equilibrium_satisfied': result['equilibrium_satisfied'],
                        'dM_drho_shell': dM_drho,
                        'harrison_wheeler_stable': stable,
                        'shell_thickness': result['R2'] - result['R1'],
                    })
                except (ValueError, RuntimeError) as e:
                    rows.append({
                        'gamma': gamma,
                        'M_target': M_target,
                        'core_compactness': C,
                        'R1': np.nan,
                        'R2': np.nan,
                        'M_total': np.nan,
                        'compactness_2M_R2': np.nan,
                        'equilibrium_satisfied': False,
                        'dM_drho_shell': np.nan,
                        'harrison_wheeler_stable': False,
                        'shell_thickness': np.nan,
                    })

    df = pd.DataFrame(rows)
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    outpath = CSV_DIR / "gravastar_polytropic_sweep.csv"
    df.to_csv(outpath, index=False)
    print(f"Wrote {len(df)} rows to {outpath}")

    # Summary stats
    n_stable = df['harrison_wheeler_stable'].sum()
    n_valid = (~df['M_total'].isna()).sum()
    print(f"  Valid solutions: {n_valid}/{len(df)}")
    print(f"  Stable (dM/drho > 0): {n_stable}/{n_valid}")

    # Check gamma < 4/3 stability
    sub_thresh = df[df['gamma'] < 4.0/3.0]
    n_sub_stable = sub_thresh['harrison_wheeler_stable'].sum()
    print(f"  Stable at gamma < 4/3: {n_sub_stable}/{len(sub_thresh)}")

    return df


def run_anisotropic_sweep():
    """Run anisotropic polytropic parameter sweep."""
    print("\nRunning anisotropic gravastar sweep...")

    gamma_values = np.array([1.0, 1.1, 1.2, 4.0/3.0, 1.5, 2.0])
    lambda_values = np.array([0.0, 0.5, 1.0, 1.5, 2.0])

    # Fixed configuration parameters
    rho_v = 1e-5
    R1 = 100.0
    rho_shell = 3e-5

    rows = []

    for gamma in gamma_values:
        for lam in lambda_values:
            try:
                result = solve_gravastar_anisotropic(
                    rho_v, R1, rho_shell,
                    gamma=gamma, K=1.0, lambda_aniso=lam
                )

                dM_drho, _, stable = compute_stability_derivative(
                    rho_v, R1, rho_shell, gamma, 1.0, aniso_lambda=lam
                )

                rows.append({
                    'gamma': gamma,
                    'lambda_aniso': lam,
                    'rho_v': rho_v,
                    'R1': R1,
                    'R2': result['R2'],
                    'M_total': result['M_total'],
                    'compactness_2M_R2': 2.0 * result['M_total'] / result['R2'],
                    'equilibrium_satisfied': result['equilibrium_satisfied'],
                    'dM_drho_shell': dM_drho,
                    'harrison_wheeler_stable': stable,
                    'shell_thickness': result['R2'] - result['R1'],
                })
            except (ValueError, RuntimeError):
                rows.append({
                    'gamma': gamma,
                    'lambda_aniso': lam,
                    'rho_v': rho_v,
                    'R1': R1,
                    'R2': np.nan,
                    'M_total': np.nan,
                    'compactness_2M_R2': np.nan,
                    'equilibrium_satisfied': False,
                    'dM_drho_shell': np.nan,
                    'harrison_wheeler_stable': False,
                    'shell_thickness': np.nan,
                })

    df = pd.DataFrame(rows)
    outpath = CSV_DIR / "gravastar_anisotropic_stability.csv"
    df.to_csv(outpath, index=False)
    print(f"Wrote {len(df)} rows to {outpath}")

    # Summary
    n_stable = df['harrison_wheeler_stable'].sum()
    n_valid = (~df['M_total'].isna()).sum()
    print(f"  Valid solutions: {n_valid}/{len(df)}")
    print(f"  Stable (dM/drho > 0): {n_stable}/{n_valid}")

    # Check if anisotropy enables stability at gamma < 4/3
    sub_thresh = df[df['gamma'] < 4.0/3.0]
    for lam in lambda_values:
        sub_lam = sub_thresh[sub_thresh['lambda_aniso'] == lam]
        n_sub_stable = sub_lam['harrison_wheeler_stable'].sum()
        print(f"  Stable at gamma < 4/3, lambda={lam}: {n_sub_stable}/{len(sub_lam)}")

    return df


if __name__ == "__main__":
    run_polytropic_sweep()
    run_anisotropic_sweep()
    print("\nDone.")
