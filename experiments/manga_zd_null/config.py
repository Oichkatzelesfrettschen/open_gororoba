"""
Configuration for MaNGA ZD null result decisive experiment (E-201/E-202).

Hyperparameters, seed list, regime definitions, wavenumber generators,
and shared constants for the 20-condition experiment harness.
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Wavenumber generators from algebraic structures
# ---------------------------------------------------------------------------

def cd_zd_wavenumbers(cd_dim: int = 16) -> np.ndarray:
    """PG(n-2,2) zero-divisor graph wavenumbers for Cayley-Dickson dim D."""
    n_modes = max(cd_dim // 2 - 1, 1)
    return np.array([2.0 * math.pi * n / n_modes for n in range(1, n_modes + 1)])


def g2_wavenumbers() -> np.ndarray:
    """G2 = Aut(O) root system: 6 positive roots."""
    return np.array([2.0 * math.pi * n / 6.0 for n in range(1, 7)])


def albert_j3o_wavenumbers() -> np.ndarray:
    """Albert algebra J3(O): 3 Peirce idempotent modes."""
    return np.array([2.0 * math.pi * n / 3.0 for n in range(1, 4)])


def sl2_partner_wavenumbers() -> np.ndarray:
    """sl(2) partner graph: spin-2 weight, 2 modes."""
    k0 = 2.0 * math.pi / 7.0
    return np.array([2.0 * k0, 4.0 * k0])


ALGEBRA_WAVENUMBERS = {
    "CD-ZD": cd_zd_wavenumbers(16),
    "G2": g2_wavenumbers(),
    "J3(O)": albert_j3o_wavenumbers(),
    "sl(2)": sl2_partner_wavenumbers(),
}


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

HYPERPARAMETERS = {
    "n_galaxies": 6992,
    "x_min": 0.5,
    "x_max": 1.35,
    "n_points_per_galaxy": 18,
    "n_grid": 200,
    "min_per_bin": 10,
    "cd_dim": 16,
    "n_bootstrap_primary": 200,
    "n_bootstrap_diagnostic": 50,
    "gamma_prior_full": 0.808,
    "assessor_fraction": 0.5,
    "face_on_inc_max": 45.0,
    "mass_quartile_target": 3,
    "time_budget_seconds": 600,
    "inject_alphas": [0.001, 0.004, 0.01],
    "inject_alphas_stratified": [0.001, 0.002, 0.004, 0.006, 0.008, 0.01],
    "inclination_strata": [[0, 30], [30, 45], [45, 70]],
    "sigma_clip_n": 3.0,
    "sigma_clip_max_iter": 5,
    "dip_test_n_perm": 5000,
    "gp_n_restarts": 20,
    "gp_periodic_period": 7.0,
}


# ---------------------------------------------------------------------------
# Seeds and regimes
# ---------------------------------------------------------------------------

SEEDS = [
    42, 123, 456, 789, 1024, 2025, 3141, 4096, 5555, 6174,
    7071, 8191, 9001, 10007, 11111, 12345, 13579, 14159, 15485, 16384,
]

REGIMES = {
    "full_sample": {"filter": None},
    "face_on": {"filter": "inclination", "inc_min": 0.0, "inc_max": 45.0},
    "mass_Q3": {"filter": "mass_quartile", "quartile": 3},
}


# ---------------------------------------------------------------------------
# Condition priority ordering (baselines -> proposed -> ablations)
# ---------------------------------------------------------------------------

CONDITION_PRIORITY = [
    # Baselines
    "mean_stack_full_sample",
    "nfw_only_chi2",
    "relatores_composite",
    "multi_algebra_mean_stack",
    # Proposed methods
    "face_on_rednoise_subtracted",
    "face_on_whitened_matched_filter",
    "face_on_gp_model_comparison",
    "q3_injection_recovery",
    # Ablations: H1 face-on
    "face_on_no_rednoise",
    "face_on_free_gamma",
    "face_on_median_stack",
    "face_on_trimmed_05",
    "face_on_trimmed_20",
    # Ablations: H2 injection
    "q1_injection_control",
    "q1_synthetic_6bin",
    "inclination_stratified_injection",
    # Ablations: cross-cutting diagnostics
    "bispectral_fano",
    "persistence_topology",
    "mutual_information",
    "variance_quantization",
]
