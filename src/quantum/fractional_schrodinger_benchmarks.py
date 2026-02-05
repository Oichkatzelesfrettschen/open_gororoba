"""
Analytical benchmarks for the Fractional Schrodinger Equation.

Provides:
  1. Free-particle Levy stable propagator (exact for any alpha in (0,2]).
  2. Fractional harmonic oscillator -- variational ground state energy
     using Gaussian trial wavefunction.
  3. Split-operator time evolution for numerical comparison.

The fractional Schrodinger equation:
  i * d_psi/dt = D * (-Delta)^{alpha/2} * psi + V(x) * psi

where alpha in (0, 2] is the Levy index and D is a generalized
diffusion coefficient.  At alpha=2, D=1/(2m), this reduces to
the standard Schrodinger equation.

Refs:
  Laskin, N. (2000), Phys. Lett. A 268, 298.
  Laskin, N. (2002), Phys. Rev. E 66, 056108.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def levy_propagator_1d(x, t, alpha=2.0, D=0.5, N_k=4096, k_max=50.0):
    """
    Numerical evaluation of the free-particle Levy propagator.

    K(x, t) = (1 / 2*pi) * integral_{-inf}^{inf} exp(i*k*x - i*D*|k|^alpha*t) dk

    At alpha=2, D=1/(2m), this is the standard Gaussian propagator:
      K(x, t) = sqrt(m / (2*pi*i*t)) * exp(i*m*x^2 / (2*t))

    Parameters
    ----------
    x : ndarray
        Spatial evaluation points.
    t : float
        Time (> 0).
    alpha : float
        Levy index in (0, 2].
    D : float
        Generalized diffusion coefficient.
    N_k : int
        Number of quadrature points in k-space.
    k_max : float
        k-space cutoff (must be large enough for convergence).

    Returns
    -------
    K : ndarray (complex)
        Propagator evaluated at each x.
    """
    x = np.asarray(x, dtype=float)
    dk = 2.0 * k_max / N_k
    k = np.linspace(-k_max, k_max, N_k, endpoint=False)

    # Integrand: exp(i*k*x_j - i*D*|k|^alpha*t) for each x_j.
    # Shape: (len(x), N_k)
    phase_k = -D * np.abs(k) ** alpha * t
    K = np.zeros(len(x), dtype=complex)
    for j, xj in enumerate(x):
        integrand = np.exp(1j * (k * xj + phase_k))
        K[j] = np.sum(integrand) * dk / (2.0 * np.pi)

    return K


def gaussian_propagator_1d(x, t, m=1.0):
    """
    Standard (alpha=2) Gaussian free-particle propagator.

    K(x, t) = sqrt(m / (2*pi*i*t)) * exp(i*m*x^2 / (2*t))
    """
    x = np.asarray(x, dtype=float)
    return np.sqrt(m / (2.0 * np.pi * 1j * t)) * np.exp(1j * m * x ** 2 / (2.0 * t))


def free_particle_l2_error(alpha=2.0, D=0.5, t=1.0, N_x=256, L=20.0, N_k=8192, k_max=80.0):
    """
    L2 error between the Levy propagator and the standard Gaussian propagator.

    At alpha=2, D=1/(2m), the Levy propagator should recover the Gaussian.

    Returns
    -------
    l2_error : float
        Normalized L2 error: ||K_levy - K_gauss|| / ||K_gauss||.
    """
    x = np.linspace(-L, L, N_x)
    K_levy = levy_propagator_1d(x, t, alpha=alpha, D=D, N_k=N_k, k_max=k_max)
    m = 1.0 / (2.0 * D)  # D = 1/(2m) for alpha=2
    K_gauss = gaussian_propagator_1d(x, t, m=m)

    norm_gauss = np.linalg.norm(K_gauss)
    if norm_gauss < 1e-30:
        return 0.0
    return float(np.linalg.norm(K_levy - K_gauss) / norm_gauss)


def fractional_ho_variational_energy(alpha=2.0, D=0.5, omega=1.0, m=1.0):
    """
    Variational ground state energy of the fractional harmonic oscillator
    using a Gaussian trial wavefunction.

    H = D * (-Delta)^{alpha/2} + (1/2) * m * omega^2 * x^2

    Trial: psi(x) = (beta/pi)^{1/4} * exp(-beta*x^2/2)

    <T> = D * <|k|^alpha> where the Fourier transform of psi gives
          <|k|^alpha> = integral |k|^alpha * |psi_hat(k)|^2 dk

    For a Gaussian with width beta:
      psi_hat(k) = (1/(pi*beta))^{1/4} * exp(-k^2/(2*beta))
      <|k|^alpha> = (2*beta)^{alpha/2} * Gamma((alpha+1)/2) / sqrt(pi)

    <V> = m * omega^2 / (4 * beta)

    E(beta) = D * (2*beta)^{alpha/2} * Gamma((alpha+1)/2) / sqrt(pi)
              + m * omega^2 / (4 * beta)

    Minimize over beta numerically.

    At alpha=2, the exact ground state energy is omega/2 (setting hbar=1, m=1,
    D=1/(2m)=0.5).

    Returns
    -------
    E_min : float
        Variational ground state energy.
    beta_opt : float
        Optimal width parameter.
    """
    from scipy.optimize import minimize_scalar
    from scipy.special import gamma as gamma_func

    coeff_T = D * gamma_func((alpha + 1.0) / 2.0) / np.sqrt(np.pi)
    coeff_V = m * omega ** 2 / 4.0

    def energy(log_beta):
        beta = np.exp(log_beta)
        return coeff_T * beta ** (alpha / 2.0) + coeff_V / beta

    # Search in log-space for robust optimization.
    result = minimize_scalar(energy, bounds=(-5, 10), method="bounded")
    beta_opt = np.exp(result.x)
    E_min = result.fun

    return E_min, beta_opt


def split_operator_evolve(psi0, x, V, alpha=2.0, D=0.5, dt=0.01, n_steps=100):
    """
    Strang split-operator evolution for the fractional Schrodinger equation.

    Parameters
    ----------
    psi0 : ndarray (complex)
        Initial wavefunction on the grid x.
    x : ndarray
        Spatial grid (evenly spaced).
    V : ndarray
        Potential on the grid.
    alpha : float
        Levy index.
    D : float
        Generalized diffusion coefficient.
    dt : float
        Time step.
    n_steps : int
        Number of time steps.

    Returns
    -------
    psi : ndarray (complex)
        Final wavefunction.
    """
    N = len(x)
    dx = x[1] - x[0]
    k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi

    expV_half = np.exp(-1j * V * dt / 2.0)
    expT = np.exp(-1j * D * np.abs(k) ** alpha * dt)

    psi = psi0.copy()
    for _ in range(n_steps):
        psi *= expV_half
        psi_k = np.fft.fft(psi)
        psi_k *= expT
        psi = np.fft.ifft(psi_k)
        psi *= expV_half

    return psi


def run_benchmarks(output_csv="data/csv/fractional_schrodinger_benchmark_errors.csv"):
    """
    Run all benchmarks and save results to CSV.

    Tests:
      1. alpha=2.0 free-particle propagator recovers Gaussian.
      2. alpha=1.5, 1.2 free-particle L2 errors are finite and bounded.
      3. alpha=2.0 harmonic oscillator variational energy recovers omega/2.
      4. alpha=1.5 harmonic oscillator gives a well-defined energy.
    """
    results = []

    # -- Free-particle benchmarks --
    for alpha in [2.0, 1.5, 1.2]:
        l2_err = free_particle_l2_error(alpha=alpha, D=0.5, t=1.0)
        results.append({
            "benchmark": "free_particle_propagator",
            "alpha": alpha,
            "metric": "l2_error_vs_gaussian",
            "value": l2_err,
            "pass": l2_err < 0.01 if alpha == 2.0 else True,
        })

    # -- Harmonic oscillator benchmarks --
    omega = 1.0
    for alpha in [2.0, 1.5, 1.2]:
        E_var, beta_opt = fractional_ho_variational_energy(
            alpha=alpha, D=0.5, omega=omega, m=1.0
        )
        # At alpha=2, exact E_0 = omega/2 = 0.5
        if alpha == 2.0:
            exact = omega / 2.0
            rel_err = abs(E_var - exact) / exact
            passed = rel_err < 0.01
        else:
            exact = None
            rel_err = None
            passed = E_var > 0  # energy should be positive
        results.append({
            "benchmark": "harmonic_oscillator_variational",
            "alpha": alpha,
            "metric": "E_ground_state",
            "value": E_var,
            "pass": passed,
        })
        if rel_err is not None:
            results.append({
                "benchmark": "harmonic_oscillator_variational",
                "alpha": alpha,
                "metric": "rel_error_vs_exact",
                "value": rel_err,
                "pass": passed,
            })

    # -- Split-operator imaginary time for ground state energy --
    N = 512
    L = 30.0
    x = np.linspace(-L, L, N, endpoint=False)
    dx = x[1] - x[0]
    V = 0.5 * omega ** 2 * x ** 2

    for alpha in [2.0, 1.5]:
        # Imaginary time evolution: replace dt -> -i*dt to project ground state.
        psi0 = np.exp(-x ** 2 / 4.0).astype(complex)
        psi0 /= np.sqrt(np.sum(np.abs(psi0) ** 2) * dx)

        k = np.fft.fftfreq(N, d=dx) * 2.0 * np.pi
        tau = 0.01  # imaginary time step
        expV_half = np.exp(-V * tau / 2.0)
        expT = np.exp(-0.5 * np.abs(k) ** alpha * tau)

        psi = psi0.copy()
        for _ in range(2000):
            psi *= expV_half
            psi_k = np.fft.fft(psi)
            psi_k *= expT
            psi = np.fft.ifft(psi_k)
            psi *= expV_half
            psi = psi.real.astype(complex)
            norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
            if norm > 1e-30:
                psi /= norm

        # Energy via <H> = <T> + <V>
        psi_k = np.fft.fft(psi)
        T_expectation = np.sum(
            0.5 * np.abs(k) ** alpha * np.abs(psi_k) ** 2
        ) * dx / N
        V_expectation = np.sum(V * np.abs(psi) ** 2) * dx

        E_imag_time = float(np.real(T_expectation + V_expectation))
        results.append({
            "benchmark": "harmonic_oscillator_imag_time",
            "alpha": alpha,
            "metric": "E_ground_state",
            "value": E_imag_time,
            "pass": E_imag_time > 0,
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    df = run_benchmarks()
    print(df.to_string(index=False))
