#!/usr/bin/env python3
"""Generate synthetic datasets for ghost spectral audit.

Creates datasets in data/csv/ghost_audit/ with known ground truth:
- Pure noise (Gaussian, log-normal, power-law) -- should be NULL
- Sorted distributions (Gaussian, log-normal) -- test sorting artifact
- Injected signal (sine at ghost freq + noise) -- should be DETECTED
- Injected signal at wrong freq (control) -- should be NULL at ghost freq

Usage:
    python generate_synthetic.py --output-dir data/csv/ghost_audit/
"""
import argparse
import os
import sys

import numpy as np


def write_csv(path: str, values: np.ndarray, column: str = "value"):
    """Write values to a standardized CSV with index,value columns."""
    with open(path, "w") as f:
        f.write(f"index,{column}\n")
        for i, v in enumerate(values):
            f.write(f"{i},{v:.12e}\n")
    print(f"  Wrote {len(values)} values to {path}")


def generate_all(output_dir: str, n: int = 2048, seed: int = 42):
    """Generate all synthetic datasets."""
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    ghost_freq = 1.0 - 0.786_151_377_757_423  # aliased ghost frequency

    # ---- NULL CONTROLS (unsorted) ----
    print("\n--- Null Controls (unsorted) ---")

    # 1. Gaussian white noise
    noise_gauss = rng.normal(0, 1, n)
    write_csv(os.path.join(output_dir, "null_gaussian.csv"), noise_gauss)

    # 2. Log-normal noise
    noise_lognorm = rng.lognormal(0, 1, n)
    write_csv(os.path.join(output_dir, "null_lognormal.csv"), noise_lognorm)

    # 3. Power-law noise (Pareto)
    noise_pareto = (rng.pareto(2.5, n) + 1) * 10
    write_csv(os.path.join(output_dir, "null_pareto.csv"), noise_pareto)

    # 4. Uniform noise
    noise_uniform = rng.uniform(0, 100, n)
    write_csv(os.path.join(output_dir, "null_uniform.csv"), noise_uniform)

    # ---- SORTED DISTRIBUTION TESTS ----
    # These mimic "sorted catalog data" where the ghost freq analysis
    # was originally applied. The key question: does sorting alone
    # produce a spectral peak near f=0.214?
    print("\n--- Sorted Distributions ---")

    # 5. Sorted Gaussian
    sorted_gauss = np.sort(rng.normal(50, 15, n))
    write_csv(os.path.join(output_dir, "sorted_gaussian.csv"), sorted_gauss)

    # 6. Sorted log-normal (similar to FRB DM distribution)
    sorted_lognorm = np.sort(rng.lognormal(5, 1.2, n))
    write_csv(os.path.join(output_dir, "sorted_lognormal.csv"), sorted_lognorm)

    # 7. Sorted power-law (similar to some astrophysical catalogs)
    sorted_pareto = np.sort((rng.pareto(2.0, n) + 1) * 100)
    write_csv(os.path.join(output_dir, "sorted_pareto.csv"), sorted_pareto)

    # 8. Sorted exponential
    sorted_exp = np.sort(rng.exponential(100, n))
    write_csv(os.path.join(output_dir, "sorted_exponential.csv"), sorted_exp)

    # 9. Sorted chi-squared (similar to some distance-modulus distributions)
    sorted_chi2 = np.sort(rng.chisquare(4, n) * 10)
    write_csv(os.path.join(output_dir, "sorted_chisq.csv"), sorted_chi2)

    # ---- SIGNAL INJECTION TESTS ----
    print("\n--- Signal Injections ---")

    # 10. Strong sine at ghost frequency (unsorted)
    t = np.arange(n, dtype=float)
    signal_ghost = 5.0 * np.sin(2 * np.pi * ghost_freq * t) + rng.normal(0, 1, n)
    write_csv(os.path.join(output_dir, "signal_ghost_strong.csv"), signal_ghost)

    # 11. Weak sine at ghost frequency (unsorted)
    signal_ghost_weak = 1.5 * np.sin(2 * np.pi * ghost_freq * t) + rng.normal(0, 1, n)
    write_csv(os.path.join(output_dir, "signal_ghost_weak.csv"), signal_ghost_weak)

    # 12. Strong sine at WRONG frequency (control)
    wrong_freq = 0.35
    signal_wrong = 5.0 * np.sin(2 * np.pi * wrong_freq * t) + rng.normal(0, 1, n)
    write_csv(os.path.join(output_dir, "signal_wrong_freq.csv"), signal_wrong)

    # 13. Mixed: sine at ghost freq embedded in red noise
    # AR(1) red noise with rho=0.7
    red_noise = np.zeros(n)
    red_noise[0] = rng.normal(0, 1)
    for i in range(1, n):
        red_noise[i] = 0.7 * red_noise[i - 1] + rng.normal(0, 1)
    signal_red = 3.0 * np.sin(2 * np.pi * ghost_freq * t) + red_noise
    write_csv(os.path.join(output_dir, "signal_ghost_red_noise.csv"), signal_red)

    # ---- MOCK CATALOG DATA ----
    # Simulate sorted catalog data WITH and WITHOUT periodic structure
    print("\n--- Mock Catalogs ---")

    # 14. Mock FRB DM catalog (log-normal, sorted, N~500)
    n_frb = 500
    mock_frb_dm = np.sort(rng.lognormal(np.log(400), 0.8, n_frb))
    write_csv(os.path.join(output_dir, "mock_frb_dm.csv"), mock_frb_dm)

    # 15. Mock pulsar period catalog (log-normal, sorted, N~4000)
    n_psr = 4000
    mock_psr = np.sort(rng.lognormal(np.log(0.5), 1.5, n_psr))
    write_csv(os.path.join(output_dir, "mock_pulsar_period.csv"), mock_psr)

    # 16. Mock SN distance modulus (Gaussian + scatter, sorted, N~1700)
    n_sn = 1700
    # Approximate Pantheon+ mu distribution: mean~38, sigma~3
    mock_sn = np.sort(rng.normal(38, 3, n_sn))
    write_csv(os.path.join(output_dir, "mock_sn_mu.csv"), mock_sn)

    # 17. Mock Gaia parallax (power-law, sorted, N~5000)
    n_gaia = 5000
    mock_gaia = np.sort(rng.exponential(2.0, n_gaia))
    write_csv(os.path.join(output_dir, "mock_gaia_parallax.csv"), mock_gaia)

    # ---- SUMMARY ----
    print(f"\n{'='*50}")
    n_files = len([f for f in os.listdir(output_dir) if f.endswith(".csv")])
    print(f"Generated {n_files} datasets in {output_dir}")
    print("\nCategories:")
    print(f"  Null controls (unsorted): 4")
    print(f"  Sorted distributions: 5")
    print(f"  Signal injections: 4")
    print(f"  Mock catalogs: 4")
    print(f"  Total: {n_files}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic datasets for ghost spectral audit"
    )
    parser.add_argument(
        "--output-dir",
        default="data/csv/ghost_audit",
        help="Output directory",
    )
    parser.add_argument("--n", type=int, default=2048, help="Default sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    generate_all(args.output_dir, n=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
