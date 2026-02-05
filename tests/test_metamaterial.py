"""
Tests for metamaterial effective medium theory.

Validates:
  1. Maxwell-Garnett: f=0 gives host, f->0 perturbative.
  2. Bruggeman: equal components gives geometric mean; f=0,1 limits.
  3. Quarter-wave antireflection: R < 1e-4.
  4. Drude-Lorentz: KK consistency within 5%.
"""
from __future__ import annotations

import numpy as np

from gemini_physics.metamaterial import (
    bruggeman,
    drude_lorentz,
    kramers_kronig_check,
    maxwell_garnett,
    tmm_reflection,
)


def test_maxwell_garnett_zero_fraction() -> None:
    """At f=0, eps_eff should equal eps_host."""
    eps_host = 2.0 + 0.1j
    eps_inc = 10.0 + 1.0j
    eps_eff = maxwell_garnett(eps_host, eps_inc, f=0.0)
    assert abs(eps_eff - eps_host) < 1e-14


def test_maxwell_garnett_dilute_limit() -> None:
    """At small f, MG should give a small perturbation from host."""
    eps_host = 2.25 + 0j
    eps_inc = 12.0 + 0j
    f = 0.01
    eps_eff = maxwell_garnett(eps_host, eps_inc, f)
    # Should be close to eps_host but slightly shifted toward eps_inc.
    assert abs(eps_eff.real - eps_host.real) < 0.3
    assert eps_eff.real > eps_host.real


def test_bruggeman_equal_components() -> None:
    """When eps_1 == eps_2, Bruggeman should return that value."""
    eps = 4.0 + 0j
    eps_eff = bruggeman(np.array([eps]), np.array([eps]), f=0.5)
    assert abs(eps_eff[0] - eps) < 1e-10


def test_bruggeman_f0_gives_component2() -> None:
    """At f=0, should return eps_2."""
    eps_1 = 2.0 + 0j
    eps_2 = 10.0 + 0j
    eps_eff = bruggeman(np.array([eps_1]), np.array([eps_2]), f=0.0)
    assert abs(eps_eff[0] - eps_2) < 1e-10


def test_bruggeman_f1_gives_component1() -> None:
    """At f=1, should return eps_1."""
    eps_1 = 2.0 + 0j
    eps_2 = 10.0 + 0j
    eps_eff = bruggeman(np.array([eps_1]), np.array([eps_2]), f=1.0)
    assert abs(eps_eff[0] - eps_1) < 1e-10


def test_quarter_wave_antireflection() -> None:
    """Quarter-wave AR coating should give R < 1e-4 at design wavelength."""
    # Glass substrate n=1.5, AR coating n=sqrt(1.5) ~ 1.2247.
    n_sub = 1.5
    n_ar = np.sqrt(n_sub)
    wavelength = 550.0  # nm
    # Quarter-wave thickness: d = lambda / (4 * n_ar)
    d_ar = wavelength / (4.0 * n_ar)

    # Stack: air | AR coating | glass
    _, R = tmm_reflection(
        n_layers=[1.0, n_ar, n_sub],
        d_layers=[d_ar],
        wavelength=wavelength,
    )

    assert R < 1e-4, f"Quarter-wave AR reflectance {R:.6e} exceeds 1e-4"


def test_drude_lorentz_kk_consistency() -> None:
    """Drude-Lorentz dielectric should satisfy KK relations within 5%."""
    # Single Lorentz oscillator.  Wide grid so tails are captured.
    omega = np.linspace(0.05, 100.0, 4000)
    eps = drude_lorentz(
        omega, eps_inf=1.0,
        oscillators=[(2.0, 5.0, 0.5)],  # strength, resonance, damping
    )

    _, max_err = kramers_kronig_check(omega, eps, component="real", eps_inf=1.0)
    assert max_err < 0.05, f"KK error {max_err:.4f} exceeds 5%"


def test_tmm_bare_interface_fresnel() -> None:
    """TMM with no layers should match Fresnel equations."""
    n1 = 1.0
    n2 = 1.5
    # At normal incidence: r = (n1 - n2) / (n1 + n2)
    r_expected = (n1 - n2) / (n1 + n2)
    R_expected = r_expected ** 2

    _, R = tmm_reflection(
        n_layers=[n1, n2],
        d_layers=[],
        wavelength=550.0,
    )
    assert abs(R - R_expected) < 1e-10
