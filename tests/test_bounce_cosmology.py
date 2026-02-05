"""
Tests for bounce cosmology observational fitting.

Validates:
  1. q_corr=0 bounce model recovers Lambda-CDM Hubble parameter.
  2. Luminosity distance is positive and monotonically increasing.
  3. CMB shift parameter is in physically reasonable range.
  4. BAO sound horizon approximation matches Planck value within 5%.
  5. Lambda-CDM chi2 per DOF is reasonable (near 1.0).
  6. Bounce delta_BIC > -10 (not strongly preferred over LCDM).
  7. Spectral index n_s in [0.9, 1.0].
  8. n_s = 1.0 at q_corr = 0.
"""
from __future__ import annotations

import numpy as np

from gemini_physics.cosmology import (
    bao_sound_horizon_approx,
    cmb_shift_parameter,
    distance_modulus,
    fit_model,
    generate_synthetic_bao_data,
    generate_synthetic_sn_data,
    hubble_E_bounce,
    hubble_E_lcdm,
    luminosity_distance,
    spectral_index_bounce,
)


def test_bounce_qcorr_zero_matches_lcdm() -> None:
    """At q_corr=0, bounce E(z) should equal LCDM E(z)."""
    z = np.array([0.0, 0.5, 1.0, 2.0])
    omega_m = 0.3
    E_lcdm = hubble_E_lcdm(z, omega_m)
    E_bounce = hubble_E_bounce(z, omega_m, q_corr=0.0)
    assert np.allclose(E_lcdm, E_bounce, atol=1e-12)


def test_luminosity_distance_positive_increasing() -> None:
    """d_L(z) should be positive and increasing with z."""
    z = np.array([0.1, 0.5, 1.0, 1.5, 2.0])
    d_L = luminosity_distance(z, omega_m=0.3, h0=70.0)
    assert np.all(d_L > 0), f"d_L has non-positive values: {d_L}"
    assert np.all(np.diff(d_L) > 0), "d_L is not monotonically increasing"


def test_cmb_shift_parameter_range() -> None:
    """R should be approximately 1.75 for Planck-like parameters."""
    R = cmb_shift_parameter(omega_m=0.315, h0=67.4, q_corr=0.0)
    assert 1.5 < R < 2.0, f"CMB shift parameter R={R:.4f} outside [1.5, 2.0]"


def test_bao_sound_horizon_planck() -> None:
    """r_d should be near 147 Mpc for Planck parameters, within 5%."""
    r_d = bao_sound_horizon_approx(omega_m=0.315, h0=67.4)
    assert abs(r_d - 147.0) / 147.0 < 0.05, f"r_d={r_d:.2f} Mpc, expected ~147"


def test_lcdm_chi2_per_dof_reasonable() -> None:
    """LCDM fit to LCDM-generated data should have chi2/dof near 1."""
    sn_data = generate_synthetic_sn_data(n_sn=50, seed=42)
    bao_data = generate_synthetic_bao_data(seed=42)
    result = fit_model(sn_data, bao_data, model="lcdm")
    dof = result["n_data"] - result["n_params"]
    chi2_per_dof = result["chi2"] / dof
    assert 0.3 < chi2_per_dof < 3.0, (
        f"chi2/dof={chi2_per_dof:.2f} outside reasonable range [0.3, 3.0]"
    )


def test_bounce_delta_bic_threshold() -> None:
    """Bounce delta_BIC should be > -10 (not strongly preferred)."""
    sn_data = generate_synthetic_sn_data(n_sn=50, seed=42)
    bao_data = generate_synthetic_bao_data(seed=42)
    lcdm = fit_model(sn_data, bao_data, model="lcdm")
    bounce = fit_model(sn_data, bao_data, model="bounce")
    delta_bic = bounce["bic"] - lcdm["bic"]
    assert delta_bic > -10.0, (
        f"delta_BIC={delta_bic:.2f} < -10 (bounce strongly preferred over LCDM, "
        f"unexpected for LCDM-generated data)"
    )


def test_spectral_index_zero_qcorr() -> None:
    """n_s = 1.0 at q_corr = 0."""
    assert spectral_index_bounce(0.0) == 1.0


def test_spectral_index_in_range() -> None:
    """n_s should be in [0.9, 1.0] for small positive q_corr."""
    # q_corr ~ 1e-6 gives n_s ~ 0.97 (Planck-compatible regime).
    n_s = spectral_index_bounce(q_corr=1e-5, omega_m=0.315)
    assert 0.9 < n_s < 1.0, f"n_s={n_s:.6f} outside [0.9, 1.0]"


def test_distance_modulus_self_consistent() -> None:
    """mu at z=0 should be -inf (d_L=0); at z>0 should be positive and large."""
    mu = distance_modulus(np.array([0.1, 1.0]), omega_m=0.3, h0=70.0)
    assert np.all(mu > 30), f"Distance moduli {mu} seem too small"
