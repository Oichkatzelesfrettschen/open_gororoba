"""
Gravastar TOV parameter sweep: mass x compactness grid.

Sweeps M_target over LIGO-relevant masses crossed with core compactness
values, collects equilibrium results, and computes Harrison-Wheeler
radial stability (dM/d(rho_c) > 0 on the stable branch).

Outputs:
    data/csv/gravastar_ligo_mass_sweep.csv
    data/csv/gravastar_radial_stability.csv

WHY:  Map the gravastar parameter space relevant to LIGO/Virgo detections.
WHAT: Two CSV artifacts covering equilibrium and radial stability.
HOW:  Run `python src/scripts/gravastar_sweep.py` from project root.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable when running as a script.
_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from gravastar_tov import solve_gravastar, solve_gravastar_for_mass  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

M_TARGETS = [5, 10, 15, 20, 25, 30, 35, 40, 50, 65, 80]
CORE_COMPACTNESSES = [0.5, 0.6, 0.7, 0.8, 0.9]
SEED = 42
RHO_PERTURBATION = 0.01  # +/-1% for Harrison-Wheeler check

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CSV_DIR = PROJECT_ROOT / "data" / "csv"


def run_mass_sweep():
    """
    Sweep (M_target, core_compactness) grid and return a DataFrame.

    Skips parameter combinations where the solver raises ValueError
    (over-compact cores that violate the horizon condition).
    """
    np.random.seed(SEED)
    rows = []

    for m_target in M_TARGETS:
        for cc in CORE_COMPACTNESSES:
            try:
                res = solve_gravastar_for_mass(
                    m_target=float(m_target),
                    core_compactness=cc,
                )
            except ValueError as exc:
                print(
                    f"  SKIP M={m_target}, C={cc}: {exc}",
                    file=sys.stderr,
                )
                continue

            compactness_2m_r2 = 2.0 * res["M_total"] / res["R2"]
            rows.append(
                {
                    "M_target": m_target,
                    "core_compactness": cc,
                    "R1": res["R1"],
                    "R2": res["R2"],
                    "M_total": res["M_total"],
                    "equilibrium_satisfied": res["equilibrium_satisfied"],
                    "compactness_2M_R2": compactness_2m_r2,
                }
            )

    df = pd.DataFrame(rows)
    return df


def harrison_wheeler_stability(sweep_df):
    """
    For each equilibrium-satisfying solution, perturb the central density
    by +/-1% and check dM/d(rho_c) > 0 (Harrison-Wheeler stable branch).

    The central density rho_c is the vacuum energy density rho_v, which
    is derived from (M_core, R1, core_compactness).  We perturb rho_v
    directly and re-solve the low-level TOV to measure the mass response.
    """
    stable_rows = []
    equilibrium_df = sweep_df[sweep_df["equilibrium_satisfied"]].copy()

    for _, row in equilibrium_df.iterrows():
        m_target = row["M_target"]
        cc = row["core_compactness"]

        # Reconstruct the core parameters that produced this solution.
        # From solve_gravastar_for_mass: R1 = 2*m_core/cc,
        # rho_v = 3*m_core/(4*pi*R1^3).
        # The M_total ~ m_target relation means m_core ~ m_target * f(cc).
        # Re-derive m_core from R1 and cc: m_core = cc * R1 / 2.
        r1 = row["R1"]
        m_core = cc * r1 / 2.0
        rho_v = 3.0 * m_core / (4.0 * np.pi * r1**3)

        masses = {}
        for label, factor in [("minus", 1.0 - RHO_PERTURBATION),
                              ("center", 1.0),
                              ("plus", 1.0 + RHO_PERTURBATION)]:
            rho_v_pert = rho_v * factor
            # Adjust R1 to keep m_core consistent with perturbed rho_v:
            # m_core = (4/3)*pi*R1^3*rho_v => R1 = (3*m_core/(4*pi*rho_v))^(1/3)
            r1_pert = (3.0 * m_core / (4.0 * np.pi * rho_v_pert)) ** (1.0 / 3.0)
            rho_shell_pert = 3.0 * rho_v_pert
            try:
                res = solve_gravastar(
                    rho_v_pert, r1_pert, rho_shell_pert, n_points=500,
                )
                masses[label] = res["M_total"]
            except ValueError:
                masses[label] = np.nan

        # Finite-difference derivative dM/d(rho_c).
        if np.isfinite(masses.get("plus", np.nan)) and np.isfinite(
            masses.get("minus", np.nan)
        ):
            d_rho = rho_v * 2.0 * RHO_PERTURBATION
            dm_drho = (masses["plus"] - masses["minus"]) / d_rho
            hw_stable = dm_drho > 0.0
        else:
            dm_drho = np.nan
            hw_stable = False

        stable_rows.append(
            {
                "M_target": m_target,
                "core_compactness": cc,
                "R1": r1,
                "R2": row["R2"],
                "M_total": row["M_total"],
                "rho_v": rho_v,
                "M_minus_1pct": masses.get("minus", np.nan),
                "M_center": masses.get("center", np.nan),
                "M_plus_1pct": masses.get("plus", np.nan),
                "dM_drho_c": dm_drho,
                "harrison_wheeler_stable": hw_stable,
            }
        )

    return pd.DataFrame(stable_rows)


def verify_compactness(sweep_df):
    """Assert all solutions have compactness < 1.0 (sub-horizon)."""
    violations = sweep_df[sweep_df["compactness_2M_R2"] >= 1.0]
    if len(violations) > 0:
        print(
            f"COMPACTNESS VIOLATION: {len(violations)} solutions have 2M/R2 >= 1.0",
            file=sys.stderr,
        )
        print(violations.to_string(), file=sys.stderr)
        return False
    return True


def main():
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Gravastar TOV Parameter Sweep ===")
    print(f"Mass targets: {M_TARGETS}")
    print(f"Core compactnesses: {CORE_COMPACTNESSES}")
    print(f"Total grid points: {len(M_TARGETS) * len(CORE_COMPACTNESSES)}")
    print()

    # --- Step 1: Mass sweep ---
    print("[1/3] Running mass x compactness sweep ...")
    sweep_df = run_mass_sweep()
    sweep_path = CSV_DIR / "gravastar_ligo_mass_sweep.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Saved {len(sweep_df)} solutions to {sweep_path}")
    print()

    # --- Step 2: Compactness verification ---
    print("[2/3] Verifying compactness < 1.0 ...")
    ok = verify_compactness(sweep_df)
    print(f"  All sub-horizon: {ok}")
    print()

    # --- Step 3: Harrison-Wheeler radial stability ---
    n_eq = sweep_df["equilibrium_satisfied"].sum()
    print(f"[3/3] Harrison-Wheeler stability for {n_eq} equilibrium solutions ...")
    stability_df = harrison_wheeler_stability(sweep_df)
    stability_path = CSV_DIR / "gravastar_radial_stability.csv"
    stability_df.to_csv(stability_path, index=False)
    n_stable = stability_df["harrison_wheeler_stable"].sum()
    print(f"  Saved {len(stability_df)} rows to {stability_path}")
    print(f"  Harrison-Wheeler stable: {n_stable}/{len(stability_df)}")
    print()

    # --- Summary ---
    print("=== Summary ===")
    print(f"  Grid points attempted: {len(M_TARGETS) * len(CORE_COMPACTNESSES)}")
    print(f"  Solutions found: {len(sweep_df)}")
    print(f"  Equilibrium satisfied: {n_eq}")
    print(f"  Radially stable (HW): {n_stable}")
    print(f"  Compactness < 1.0: {ok}")


if __name__ == "__main__":
    main()
