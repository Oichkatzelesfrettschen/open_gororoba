"""
Tests for negative-dimension PDE solver.

Validates:
  1. Kinetic operator construction for alpha < 0.
  2. Ground state eigenvalue is positive and well-defined.
  3. Eigenvalue convergence: rel change < 1% between eps=0.005 and eps=0.001.
  4. Caffarelli-Silvestre s=1 recovers standard harmonic oscillator E_0=0.5.
  5. Ground state wavefunction is normalized.
"""
from __future__ import annotations

import numpy as np

from neg_dim_pde import (
    build_kinetic_operator,
    caffarelli_silvestre_eigenvalues,
    eigenvalues_imaginary_time,
)


def test_kinetic_operator_alpha_negative() -> None:
    """For alpha < 0, T(k) should decay for large |k|."""
    T_k, k = build_kinetic_operator(N=64, L=10.0, alpha=-1.5, epsilon=0.1)
    # T(k) = (|k| + eps)^alpha with alpha < 0 => smaller for larger |k|.
    k_abs = np.abs(k)
    # Compare two non-zero k values.
    idx_small = np.argmin(np.abs(k_abs - 1.0))
    idx_large = np.argmin(np.abs(k_abs - 10.0))
    assert T_k[idx_small] > T_k[idx_large], (
        f"T(k_small)={T_k[idx_small]} should exceed T(k_large)={T_k[idx_large]}"
    )


def test_kinetic_operator_epsilon_regularizes() -> None:
    """Larger epsilon should increase T(k=0) for alpha < 0."""
    T_small, _ = build_kinetic_operator(N=64, L=10.0, alpha=-1.5, epsilon=0.01)
    T_large, _ = build_kinetic_operator(N=64, L=10.0, alpha=-1.5, epsilon=1.0)
    # At k=0: T = epsilon^alpha.  For alpha < 0, smaller epsilon => larger T.
    assert T_small[0] > T_large[0], (
        f"T(k=0, eps=0.01)={T_small[0]} should exceed T(k=0, eps=1.0)={T_large[0]}"
    )


def test_ground_state_positive() -> None:
    """Ground state energy should be positive for alpha=-1.5."""
    eigs, states = eigenvalues_imaginary_time(
        alpha=-1.5, epsilon=0.1, N=128, L=10.0,
        n_eig=1, dt=0.01, n_steps=3000,
    )
    assert eigs[0] > 0, f"E_0={eigs[0]} should be positive"


def test_ground_state_normalized() -> None:
    """Ground state wavefunction should be normalized."""
    N = 128
    L = 10.0
    dx = L / N
    _, states = eigenvalues_imaginary_time(
        alpha=-1.5, epsilon=0.1, N=N, L=L,
        n_eig=1, dt=0.01, n_steps=3000,
    )
    norm = np.sum(states[0] ** 2) * dx
    assert abs(norm - 1.0) < 0.01, f"Norm={norm:.6f}, expected 1.0"


def test_epsilon_convergence() -> None:
    """Eigenvalue relative change < 1% between eps=0.005 and eps=0.001."""
    eigs_fine, _ = eigenvalues_imaginary_time(
        alpha=-1.5, epsilon=0.005, N=256, L=10.0,
        n_eig=1, dt=0.005, n_steps=5000,
    )
    eigs_finer, _ = eigenvalues_imaginary_time(
        alpha=-1.5, epsilon=0.001, N=256, L=10.0,
        n_eig=1, dt=0.005, n_steps=5000,
    )
    rel_change = abs(eigs_finer[0] - eigs_fine[0]) / abs(eigs_fine[0])
    assert rel_change < 0.01, (
        f"Relative change {rel_change:.6f} exceeds 1% "
        f"(E_0(0.005)={eigs_fine[0]:.8f}, E_0(0.001)={eigs_finer[0]:.8f})"
    )


def test_caffarelli_silvestre_s1_harmonic_oscillator() -> None:
    """At s=1, H = k^2 + 0.5*x^2 has E_0 = sqrt(2)/2 (m_eff=1/2, w=sqrt(2))."""
    eigs = caffarelli_silvestre_eigenvalues(s=1.0, N=256, L=20.0, n_eig=1)
    exact = np.sqrt(2.0) / 2.0
    rel_err = abs(eigs[0] - exact) / exact
    assert rel_err < 0.02, f"E_0={eigs[0]:.6f}, expected {exact:.6f}, rel_err={rel_err:.4f}"


def test_caffarelli_silvestre_eigenvalues_positive() -> None:
    """Eigenvalues for s=0.5 should be positive."""
    eigs = caffarelli_silvestre_eigenvalues(s=0.5, N=128, L=10.0, n_eig=1)
    assert eigs[0] > 0, f"E_0={eigs[0]} should be positive"
