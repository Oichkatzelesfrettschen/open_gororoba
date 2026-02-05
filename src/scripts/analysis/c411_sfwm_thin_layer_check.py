"""
C-411: Minimal numeric check for SFWM thin-layer scaling
(Son & Chekhova 2026, arXiv:2601.23137).

Key testable claim:
  In a thin nonlinear layer, direct SFWM dominates over cascaded
  SHG+SPDC because |Delta_k_SFWM| << |Delta_k_SHG|, |Delta_k_SPDC|.

  The phase-matching function for each process is:
    F(L) = sin(Delta_k * L / 2) / (Delta_k / 2)

  Peak efficiency scales as |F(L)|^2 ~ (2/|Delta_k|)^2 for thick
  crystals but as L^2 for L << L_coh = pi/|Delta_k|.

  For L = 10 um (experimental):
    L_coh_SFWM  = 33.3 um   (L/L_coh ~ 0.30: nearly phase-matched)
    L_coh_SHG   =  3.1 um   (L/L_coh ~ 3.23: oscillating, suppressed)
    L_coh_SPDC  =  3.4 um   (L/L_coh ~ 2.94: oscillating, suppressed)

  Therefore |F_SFWM(L)|^2 >> |F_SHG(L)|^2 * |F_SPDC(L)|^2.

Refs:
  Son, C. & Chekhova, M. (2026), arXiv:2601.23137.
"""
from __future__ import annotations

import numpy as np


def phase_matching_function(delta_k, L):
    """
    Phase-matching function F(L) = sin(delta_k * L / 2) / (delta_k / 2).

    Parameters
    ----------
    delta_k : float
        Wavevector mismatch [1/um].
    L : float
        Crystal thickness [um].

    Returns
    -------
    F : float
        Phase-matching function value [um].
    """
    arg = delta_k * L / 2.0
    if abs(arg) < 1e-12:
        return L
    return np.sin(arg) / (delta_k / 2.0)


def coherence_length(delta_k):
    """L_coh = pi / |delta_k| in um."""
    return np.pi / abs(delta_k)


def sfwm_dominance_check(L=10.0):
    """
    Check that direct SFWM dominates over cascaded SHG+SPDC
    for a thin layer of thickness L (um).

    Uses experimental values from Son & Chekhova (2026):
      L_coh_SFWM  = 33.3 um -> delta_k_SFWM  = pi/33.3
      L_coh_SHG   =  3.1 um -> delta_k_SHG   = pi/3.1
      L_coh_SPDC  =  3.4 um -> delta_k_SPDC  = pi/3.4

    Returns
    -------
    dict with phase-matching values and dominance ratio.
    """
    # Wavevector mismatches from measured coherence lengths.
    dk_sfwm = np.pi / 33.3
    dk_shg = np.pi / 3.1
    dk_spdc = np.pi / 3.4

    F_sfwm = phase_matching_function(dk_sfwm, L)
    F_shg = phase_matching_function(dk_shg, L)
    F_spdc = phase_matching_function(dk_spdc, L)

    # Direct SFWM efficiency ~ |F_sfwm|^2.
    # Cascaded efficiency ~ |F_shg|^2 * |F_spdc|^2 (product of two steps).
    eff_direct = F_sfwm ** 2
    eff_cascaded = F_shg ** 2 * F_spdc ** 2

    dominance_ratio = eff_direct / max(eff_cascaded, 1e-30)

    return {
        "L_um": L,
        "L_coh_SFWM_um": coherence_length(dk_sfwm),
        "L_coh_SHG_um": coherence_length(dk_shg),
        "L_coh_SPDC_um": coherence_length(dk_spdc),
        "F_SFWM": F_sfwm,
        "F_SHG": F_shg,
        "F_SPDC": F_spdc,
        "eff_direct": eff_direct,
        "eff_cascaded": eff_cascaded,
        "dominance_ratio": dominance_ratio,
    }


def thickness_sweep(L_values=None):
    """
    Sweep crystal thickness and compute direct/cascaded ratio.

    Returns list of dicts, one per thickness.
    """
    if L_values is None:
        L_values = np.linspace(1.0, 100.0, 100)
    return [sfwm_dominance_check(L) for L in L_values]


if __name__ == "__main__":
    result = sfwm_dominance_check(L=10.0)
    for k, v in result.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6e}")
        else:
            print(f"  {k}: {v}")
    print()
    print(f"  Direct SFWM dominates by factor: {result['dominance_ratio']:.1f}x")
