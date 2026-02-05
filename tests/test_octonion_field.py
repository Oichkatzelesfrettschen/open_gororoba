"""
Tests for octonion-valued scalar field Hamiltonian.

Validates:
  1. Free-field energy conservation < 0.1% over 1000 Verlet steps.
  2. Free-field dispersion omega^2 = k^2 + m^2 within 1%.
  3. Quartic interaction energy conservation < 0.1%.
  4. Noether charge conservation < 0.1%.
  5. Octonion multiplication is non-associative.
  6. Octonion multiplication is alternative.
  7. Octonion norm is multiplicative (composition algebra).
"""
from __future__ import annotations

import numpy as np

from gemini_physics.octonion_field import (
    evolve,
    measure_dispersion,
    noether_charges,
    oct_multiply,
    stormer_verlet_step,
)


def test_free_field_energy_conservation() -> None:
    """Energy conserved < 0.1% over 1000 Verlet steps."""
    N, L, mass = 64, 2 * np.pi, 1.0
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    env = np.exp(-0.5 * ((x - np.pi) / 0.5) ** 2)
    phi0 = np.zeros((N, 8))
    phi0[:, 0] = env
    phi0[:, 1] = 0.5 * env
    pi0 = np.zeros((N, 8))
    pi0[:, 2] = 0.3 * env

    _, _, energies = evolve(phi0, pi0, dt=0.01, n_steps=1000, dx=dx, mass=mass)
    E0 = energies[0]
    max_rel = np.max(np.abs(energies - E0)) / abs(E0)
    assert max_rel < 0.001, f"Energy drift {max_rel:.6f} exceeds 0.1%"


def test_free_field_dispersion() -> None:
    """omega matches sqrt(k^2 + m^2) within 1% for first 3 modes."""
    results = measure_dispersion(mass=1.0, N=128, n_modes=3)
    for r in results:
        assert r["rel_err"] < 0.01, (
            f"Mode {r['mode']}: omega_meas={r['omega_measured']:.4f}, "
            f"omega_exact={r['omega_exact']:.4f}, err={r['rel_err']:.4f}"
        )


def test_quartic_energy_conservation() -> None:
    """With small coupling, energy still conserved < 0.1%."""
    N, L, mass = 64, 2 * np.pi, 1.0
    dx = L / N
    x = np.linspace(0.0, L, N, endpoint=False)
    env = np.exp(-0.5 * ((x - np.pi) / 0.5) ** 2)
    phi0 = np.zeros((N, 8))
    phi0[:, 0] = 0.1 * env
    pi0 = np.zeros((N, 8))
    pi0[:, 1] = 0.05 * env

    _, _, energies = evolve(
        phi0, pi0, dt=0.005, n_steps=1000,
        dx=dx, mass=mass, coupling=0.1,
    )
    E0 = energies[0]
    max_rel = np.max(np.abs(energies - E0)) / abs(E0)
    assert max_rel < 0.001, f"Energy drift {max_rel:.6f} exceeds 0.1%"


def test_noether_charge_conservation() -> None:
    """U(1) Noether charge Q_1 conserved < 0.1% over 1000 steps."""
    N, L, mass = 64, 2 * np.pi, 1.0
    dx = L / N
    # Homogeneous field: phi_0=1 (const), pi_1=1 (const).
    # Q_1 = sum pi . (e_1*phi) = sum [pi_1*phi_0 - pi_0*phi_1] = N*1 = N.
    phi0 = np.zeros((N, 8))
    phi0[:, 0] = 1.0
    pi0 = np.zeros((N, 8))
    pi0[:, 1] = 1.0

    Q0 = noether_charges(phi0, pi0, dx)
    phi, pi = phi0.copy(), pi0.copy()
    for _ in range(1000):
        phi, pi = stormer_verlet_step(phi, pi, dt=0.01, dx=dx, mass=mass)
    Qf = noether_charges(phi, pi, dx)

    assert abs(Q0[0]) > 1.0, f"Q_1(0)={Q0[0]:.4f} too small"
    rel_err = abs(Qf[0] - Q0[0]) / abs(Q0[0])
    assert rel_err < 0.001, f"Noether charge Q_1 drift {rel_err:.6f} exceeds 0.1%"


def test_octonion_nonassociative() -> None:
    """Octonion multiplication is non-associative for generic elements."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    c = rng.standard_normal(8)
    lhs = oct_multiply(oct_multiply(a, b), c)
    rhs = oct_multiply(a, oct_multiply(b, c))
    assoc = np.linalg.norm(lhs - rhs)
    assert assoc > 0.01, f"Associator norm {assoc:.6f} unexpectedly small"


def test_octonion_alternative() -> None:
    """Alternativity: (a*a)*b = a*(a*b) and (b*a)*a = b*(a*a)."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    # Left alternativity.
    lhs = oct_multiply(oct_multiply(a, a), b)
    rhs = oct_multiply(a, oct_multiply(a, b))
    assert np.allclose(lhs, rhs, atol=1e-12), (
        f"Left alternativity failed: max diff={np.max(np.abs(lhs - rhs)):.2e}"
    )
    # Right alternativity.
    lhs2 = oct_multiply(oct_multiply(b, a), a)
    rhs2 = oct_multiply(b, oct_multiply(a, a))
    assert np.allclose(lhs2, rhs2, atol=1e-12), (
        f"Right alternativity failed: max diff={np.max(np.abs(lhs2 - rhs2)):.2e}"
    )


def test_octonion_composition() -> None:
    """||a*b|| = ||a|| * ||b|| (composition algebra property)."""
    rng = np.random.default_rng(42)
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    ab = oct_multiply(a, b)
    norm_ab = np.linalg.norm(ab)
    norm_a_b = np.linalg.norm(a) * np.linalg.norm(b)
    rel_err = abs(norm_ab - norm_a_b) / norm_a_b
    assert rel_err < 1e-12, f"Composition property failed: rel_err={rel_err:.2e}"
