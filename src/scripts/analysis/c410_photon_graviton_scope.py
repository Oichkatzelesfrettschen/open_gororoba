"""
C-410: Scope limits for photon-graviton mixing (Ahmadiniaz et al. 2026)
relative to C-402 (metamaterial gravity coupling).

Key result from the paper:
  - Three one-loop diagrams contribute to photon-graviton conversion
    in a constant EM field: irreducible, reducible (tadpole), and a
    third diagram involving vacuum-polarization subdiagram.
  - The tadpole diagram contributes to the amplitude but NOT to
    magnetic dichroism.
  - The gravitational Ward identity k_mu T^{mu nu} = 0 provides a
    consistency check.

Scope limits for C-402:
  - C-402 (Rodal 2025) REFUTED "cheap warp drives via EM-gravity
    coupling" because modifying G via metamaterials violates energy-
    momentum conservation under standard GR+SM.
  - C-410 does NOT refute Rodal. It refines the perturbative expansion
    for photon-graviton mixing at one loop, but the coupling remains
    gravitationally weak (kappa ~ sqrt(16*pi*G/c^4)).
  - The mixing amplitude scales as kappa * (alpha/pi) * (B/B_cr)^2,
    where B_cr = m_e^2*c^3/(e*hbar) ~ 4.4e9 T (Schwinger limit).
    This is unmeasurably small for laboratory fields.

Numeric check: Schwinger critical field and coupling constants.

Refs:
  Ahmadiniaz, N. et al. (2026), arXiv:2601.23279.
  Rodal, J. (2025), "Metamaterial Gravitational Coupling".
"""
from __future__ import annotations

import numpy as np


def schwinger_critical_field():
    """
    Compute the Schwinger critical field B_cr = m_e^2 c^2 / (e hbar).

    Returns
    -------
    B_cr : float
        Critical magnetic field in Tesla.
    """
    m_e = 9.1093837015e-31   # kg
    c = 2.99792458e8         # m/s
    e = 1.602176634e-19      # C
    hbar = 1.054571817e-34   # J*s

    B_cr = m_e ** 2 * c ** 2 / (e * hbar)
    return B_cr


def gravitational_coupling():
    """
    Compute kappa = sqrt(16*pi*G/c^4) in SI units.

    Returns
    -------
    kappa : float
        Gravitational coupling constant [m / (kg * m/s^2)]
        = [s^2 / (kg * m)] effectively.
    """
    G = 6.67430e-11   # m^3 / (kg * s^2)
    c = 2.99792458e8  # m/s

    kappa = np.sqrt(16.0 * np.pi * G / c ** 4)
    return kappa


def mixing_amplitude_estimate(B_lab=1.0):
    """
    Order-of-magnitude estimate for one-loop photon-graviton mixing
    amplitude in a laboratory magnetic field.

    A ~ kappa * (alpha/pi) * (B_lab/B_cr)^2

    Parameters
    ----------
    B_lab : float
        Laboratory magnetic field in Tesla.

    Returns
    -------
    dict with B_cr, kappa, alpha_em, amplitude_ratio, and scope note.
    """
    alpha_em = 1.0 / 137.036
    B_cr = schwinger_critical_field()
    kappa = gravitational_coupling()

    # Dimensionless amplitude ratio (suppression factor).
    ratio = (alpha_em / np.pi) * (B_lab / B_cr) ** 2

    return {
        "B_cr_tesla": B_cr,
        "kappa_SI": kappa,
        "alpha_em": alpha_em,
        "B_lab_tesla": B_lab,
        "amplitude_ratio": ratio,
        "scope_note": (
            "C-410 refines the perturbative expansion for photon-graviton "
            "mixing at one loop. It does NOT overturn Rodal (2025) C-402 "
            "refutation of metamaterial gravity coupling. The mixing "
            "amplitude is suppressed by (B/B_cr)^2 ~ {:.2e}, making it "
            "unmeasurable for laboratory fields.".format(ratio)
        ),
    }


if __name__ == "__main__":
    result = mixing_amplitude_estimate(B_lab=10.0)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
