"""
Negative-Dimension PDE Solver: Regularized Fractional Operators.

Provides:
  1. Regularized fractional kinetic operator: T(k) = (|k| + epsilon)^alpha
     for alpha < 0 (inverse Laplacian / smoothing operators).
  2. Eigenvalue spectrum computation via imaginary-time evolution.
  3. Epsilon convergence sweep: eigenvalues as epsilon -> 0.
  4. Caffarelli-Silvestre extension comparison for s in (0,1).

The "negative dimension" interpretation: when alpha < 0, the kinetic
operator T(k) = |k|^alpha decays for large |k|, making the operator
a smoothing (integral) operator rather than a differential operator.
The regularization epsilon prevents the k=0 singularity.

As epsilon -> 0, the eigenvalue spectrum should converge.  The rate
of convergence and the limiting spectrum characterize the physics.

Refs:
  Caffarelli, L. & Silvestre, L. (2007), Comm. PDE 32, 1245.
  Laskin, N. (2000), Phys. Lett. A 268, 298.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_kinetic_operator(N, L, alpha, epsilon):
    """
    Build the regularized fractional kinetic operator in k-space.

    T(k) = (|k| + epsilon)^alpha

    Parameters
    ----------
    N : int
        Number of grid points.
    L : float
        Domain size [-L/2, L/2].
    alpha : float
        Fractional exponent (alpha < 0 for negative-dimension).
    epsilon : float
        Regularization parameter (epsilon > 0).

    Returns
    -------
    T_k : ndarray, shape (N,)
        Kinetic operator values on the FFT frequency grid.
    k : ndarray, shape (N,)
        Wavenumber grid.
    """
    dx = L / N
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    T_k = (np.abs(k) + epsilon) ** alpha
    return T_k, k


def eigenvalues_imaginary_time(alpha, epsilon, N=256, L=10.0, n_eig=5,
                               dt=0.005, n_steps=5000):
    """
    Compute lowest eigenvalues of H = T(k) + V(x) via imaginary-time evolution.

    The Hamiltonian is:
      H = (|k| + epsilon)^alpha + V(x),   V(x) = 0.5 * x^2

    Imaginary-time Strang splitting projects out successive eigenstates
    via Gram-Schmidt orthogonalization.

    Parameters
    ----------
    alpha : float
        Fractional exponent.
    epsilon : float
        Regularization parameter.
    N : int
        Grid points.
    L : float
        Domain size.
    n_eig : int
        Number of eigenvalues to compute.
    dt : float
        Imaginary time step.
    n_steps : int
        Number of imaginary-time steps.

    Returns
    -------
    eigenvalues : ndarray, shape (n_eig,)
        Approximate eigenvalues in ascending order.
    eigenstates : ndarray, shape (n_eig, N)
        Corresponding eigenstates (normalized, real).
    """
    dx = L / N
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    V = 0.5 * x ** 2

    T_k, _ = build_kinetic_operator(N, L, alpha, epsilon)

    expV_half = np.exp(-V * dt / 2.0)
    expT = np.exp(-T_k * dt)

    eigenvalues = np.zeros(n_eig)
    eigenstates = np.zeros((n_eig, N))

    for m in range(n_eig):
        # Initial trial: Hermite-Gaussian-like state for diversity.
        if m == 0:
            psi = np.exp(-x ** 2 / 2.0)
        else:
            # Use x^m * gaussian for higher states.
            psi = (x ** m) * np.exp(-x ** 2 / 2.0)

        psi = psi.astype(complex)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        if norm > 1e-30:
            psi /= norm

        for _ in range(n_steps):
            # Strang splitting (imaginary time).
            psi *= expV_half
            psi_k = np.fft.fft(psi)
            psi_k *= expT
            psi = np.fft.ifft(psi_k)
            psi *= expV_half

            # Project to real (imaginary-time evolution preserves real states).
            psi = psi.real.astype(complex)

            # Gram-Schmidt: orthogonalize against previously found states.
            for j in range(m):
                overlap = np.sum(eigenstates[j] * psi.real) * dx
                psi -= overlap * eigenstates[j]

            # Renormalize.
            norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
            if norm > 1e-30:
                psi /= norm

        # Compute energy expectation: <H> = <T> + <V>
        psi_real = psi.real
        psi_k = np.fft.fft(psi_real)
        T_exp = np.sum(T_k * np.abs(psi_k) ** 2) * dx / N
        V_exp = np.sum(V * psi_real ** 2) * dx

        eigenvalues[m] = float(T_exp + V_exp)
        eigenstates[m] = psi_real

    return eigenvalues, eigenstates


def epsilon_convergence_sweep(alpha=-1.5, epsilon_values=None, N=256, L=10.0,
                              n_eig=5, dt=0.005, n_steps=5000):
    """
    Sweep epsilon and track eigenvalue convergence.

    Parameters
    ----------
    alpha : float
        Fractional exponent.
    epsilon_values : list of float or None
        Regularization values to sweep.  Default:
        [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001].
    N, L, n_eig, dt, n_steps : see eigenvalues_imaginary_time.

    Returns
    -------
    results : list of dict
        Each entry has 'epsilon', 'eigenvalue_index', 'eigenvalue'.
    """
    if epsilon_values is None:
        epsilon_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]

    results = []
    for eps_val in epsilon_values:
        eigs, _ = eigenvalues_imaginary_time(
            alpha=alpha, epsilon=eps_val, N=N, L=L,
            n_eig=n_eig, dt=dt, n_steps=n_steps,
        )
        for i, e in enumerate(eigs):
            results.append({
                "epsilon": eps_val,
                "eigenvalue_index": i,
                "eigenvalue": e,
            })

    return results


def caffarelli_silvestre_eigenvalues(s, N=256, L=10.0, n_eig=5):
    """
    Compute eigenvalues of H = (-Delta)^s + V(x) using the standard
    fractional Laplacian (no regularization).

    For s in (0,1), the fractional Laplacian (-Delta)^s has Fourier
    symbol |k|^{2s}.  The Caffarelli-Silvestre extension characterizes
    this operator as the boundary trace of a degenerate elliptic PDE
    in one higher dimension.

    Here we compute eigenvalues directly via the Fourier multiplier
    |k|^{2s} using imaginary-time evolution (no extension needed for
    the eigenvalue problem, but we name it after the reference).

    Parameters
    ----------
    s : float
        Fractional exponent in (0, 1).
    N : int
        Grid points.
    L : float
        Domain size.
    n_eig : int
        Number of eigenvalues.

    Returns
    -------
    eigenvalues : ndarray, shape (n_eig,)
    """
    dx = L / N
    x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    V = 0.5 * x ** 2

    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
    # Standard fractional Laplacian: |k|^{2s}, with k=0 set to 0.
    T_k = np.abs(k) ** (2.0 * s)
    T_k[0] = 0.0

    dt = 0.005
    n_steps = 5000
    expV_half = np.exp(-V * dt / 2.0)
    expT = np.exp(-T_k * dt)

    eigenvalues = np.zeros(n_eig)
    eigenstates = np.zeros((n_eig, N))

    for m in range(n_eig):
        if m == 0:
            psi = np.exp(-x ** 2 / 2.0)
        else:
            psi = (x ** m) * np.exp(-x ** 2 / 2.0)

        psi = psi.astype(complex)
        norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
        if norm > 1e-30:
            psi /= norm

        for _ in range(n_steps):
            psi *= expV_half
            psi_k = np.fft.fft(psi)
            psi_k *= expT
            psi = np.fft.ifft(psi_k)
            psi *= expV_half

            psi = psi.real.astype(complex)

            for j in range(m):
                overlap = np.sum(eigenstates[j] * psi.real) * dx
                psi -= overlap * eigenstates[j]

            norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
            if norm > 1e-30:
                psi /= norm

        psi_real = psi.real
        psi_k = np.fft.fft(psi_real)
        T_exp = np.sum(T_k * np.abs(psi_k) ** 2) * dx / N
        V_exp = np.sum(V * psi_real ** 2) * dx

        eigenvalues[m] = float(T_exp + V_exp)
        eigenstates[m] = psi_real

    return eigenvalues


def run_convergence_analysis(output_csv="data/csv/neg_dim_convergence_metrics.csv"):
    """
    Run the full epsilon convergence sweep and Caffarelli-Silvestre
    comparison, saving results to CSV.

    Returns
    -------
    df : DataFrame
        All convergence metrics.
    """
    alpha = -1.5
    epsilon_values = [1.0, 0.5, 0.1, 0.05, 0.01, 0.005, 0.001]
    n_eig = 5

    # Epsilon sweep.
    results = epsilon_convergence_sweep(
        alpha=alpha, epsilon_values=epsilon_values, n_eig=n_eig,
    )

    # Add relative change between consecutive epsilon values.
    df = pd.DataFrame(results)
    for i in range(n_eig):
        mask = df["eigenvalue_index"] == i
        eigs = df.loc[mask, "eigenvalue"].values
        rel_change = np.zeros(len(eigs))
        for j in range(1, len(eigs)):
            if abs(eigs[j - 1]) > 1e-30:
                rel_change[j] = abs(eigs[j] - eigs[j - 1]) / abs(eigs[j - 1])
        df.loc[mask, "rel_change"] = rel_change

    # Caffarelli-Silvestre comparison (s=0.5 corresponds to alpha=1.0 kinetic).
    cs_eigs = caffarelli_silvestre_eigenvalues(s=0.5, n_eig=n_eig)
    cs_rows = []
    for i, e in enumerate(cs_eigs):
        cs_rows.append({
            "epsilon": 0.0,
            "eigenvalue_index": i,
            "eigenvalue": e,
            "rel_change": 0.0,
        })
    df = pd.concat([df, pd.DataFrame(cs_rows)], ignore_index=True)

    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    df = run_convergence_analysis()
    print(df.to_string(index=False))
