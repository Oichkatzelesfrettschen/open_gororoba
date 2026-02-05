"""
Tests for fractional Schrodinger equation analytical benchmarks.

Validates:
  1. alpha=2.0 free-particle recovers Gaussian propagator (L2 < 0.01).
  2. alpha=1.5 free-particle L2 is bounded.
  3. alpha=2.0 harmonic oscillator recovers E_0 = omega/2 within 1%.
"""
from __future__ import annotations

from quantum.fractional_schrodinger_benchmarks import (
    fractional_ho_variational_energy,
    free_particle_l2_error,
)


def test_free_particle_alpha2_recovers_gaussian() -> None:
    """At alpha=2, the Levy propagator should match the Gaussian."""
    l2_err = free_particle_l2_error(alpha=2.0, D=0.5, t=1.0)
    assert l2_err < 0.01, f"L2 error {l2_err:.6f} exceeds 0.01"


def test_free_particle_alpha15_bounded() -> None:
    """At alpha=1.5, the L2 error vs Gaussian should be > 0 (different physics)."""
    l2_err = free_particle_l2_error(alpha=1.5, D=0.5, t=1.0)
    assert l2_err > 0.1, "alpha=1.5 should differ substantially from Gaussian"


def test_ho_variational_alpha2_recovers_exact() -> None:
    """At alpha=2, variational E_0 should be omega/2 within 1%."""
    E_var, _ = fractional_ho_variational_energy(alpha=2.0, D=0.5, omega=1.0, m=1.0)
    exact = 0.5
    rel_err = abs(E_var - exact) / exact
    assert rel_err < 0.01, f"Variational E_0={E_var:.6f}, expected {exact}, err={rel_err:.6f}"


def test_ho_variational_alpha15_positive() -> None:
    """At alpha=1.5, ground state energy should be positive and well-defined."""
    E_var, beta = fractional_ho_variational_energy(alpha=1.5, D=0.5, omega=1.0, m=1.0)
    assert E_var > 0, f"E_var={E_var} should be positive"
    assert beta > 0, f"beta={beta} should be positive"
