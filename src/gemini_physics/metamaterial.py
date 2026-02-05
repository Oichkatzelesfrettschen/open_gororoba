"""
Metamaterial Effective Medium Theory.

Provides rigorous implementations of:
  1. Maxwell-Garnett mixing formula (dilute inclusions in host).
  2. Bruggeman self-consistent EMA (arbitrary fill fractions).
  3. Drude-Lorentz dielectric model with Kramers-Kronig consistency.
  4. Transfer Matrix Method (TMM) for multilayer thin-film optics.
  5. Kramers-Kronig Hilbert-transform consistency check.

All permittivities are complex: eps = eps' + i*eps''.
Frequency is in eV unless stated otherwise.

Refs:
  Sihvola, A. (1999), "Electromagnetic Mixing Formulas and Applications", IEE.
  Pozar, D. (2012), "Microwave Engineering", 4th ed., Wiley.
  Born & Wolf (2019), "Principles of Optics", 7th ed., Cambridge.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 1. Maxwell-Garnett Mixing
# ---------------------------------------------------------------------------

def maxwell_garnett(eps_host, eps_inc, f):
    """
    Maxwell-Garnett effective permittivity for spherical inclusions.

    Parameters
    ----------
    eps_host : complex or ndarray
        Host medium permittivity.
    eps_inc : complex or ndarray
        Inclusion permittivity.
    f : float
        Volume fraction of inclusions (0 < f < 1).

    Returns
    -------
    eps_eff : complex or ndarray
        Effective permittivity.
    """
    beta = (eps_inc - eps_host) / (eps_inc + 2.0 * eps_host)
    return eps_host * (1.0 + 2.0 * f * beta) / (1.0 - f * beta)


# ---------------------------------------------------------------------------
# 2. Bruggeman Self-Consistent EMA
# ---------------------------------------------------------------------------

def bruggeman(eps_1, eps_2, f, tol=1e-12, max_iter=200):
    """
    Bruggeman effective medium approximation (self-consistent).

    Solves: f*(eps_1 - eps_eff)/(eps_1 + 2*eps_eff)
            + (1-f)*(eps_2 - eps_eff)/(eps_2 + 2*eps_eff) = 0

    Parameters
    ----------
    eps_1, eps_2 : complex or ndarray
        Permittivities of the two components.
    f : float
        Volume fraction of component 1.

    Returns
    -------
    eps_eff : complex or ndarray
    """
    eps_1 = np.asarray(eps_1, dtype=complex)
    eps_2 = np.asarray(eps_2, dtype=complex)

    # Multiply through by (eps_1 + 2*eps_eff)(eps_2 + 2*eps_eff) and collect
    # powers of eps_eff to get: 2*eps_eff^2 + B*eps_eff + C = 0
    # where B = -[(3f-1)*eps_1 + (2-3f)*eps_2], C = -eps_1*eps_2.
    A = np.full_like(eps_1, 2.0)
    B = -((3.0 * f - 1.0) * eps_1 + (2.0 - 3.0 * f) * eps_2)
    C = -eps_1 * eps_2

    disc = B ** 2 - 4.0 * A * C
    sqrt_disc = np.sqrt(disc)

    # Two roots; choose the one with positive imaginary part (causal).
    root1 = (-B + sqrt_disc) / (2.0 * A)
    root2 = (-B - sqrt_disc) / (2.0 * A)

    eps_eff = np.where(root1.imag >= 0, root1, root2)

    # For purely real inputs, choose the root with positive real part.
    real_mask = (eps_1.imag == 0) & (eps_2.imag == 0)
    if np.any(real_mask):
        eps_eff = np.where(
            real_mask & (root1.real > 0), root1,
            np.where(real_mask & (root2.real > 0), root2, eps_eff)
        )

    return eps_eff


# ---------------------------------------------------------------------------
# 3. Drude-Lorentz Dielectric Model
# ---------------------------------------------------------------------------

def drude_lorentz(omega, eps_inf=1.0, omega_p=0.0, gamma_d=0.0,
                  oscillators=None):
    """
    Drude-Lorentz dielectric function.

    eps(omega) = eps_inf - omega_p^2 / (omega^2 + i*gamma_d*omega)
                 + sum_j S_j * omega_j^2 / (omega_j^2 - omega^2 - i*gamma_j*omega)

    Parameters
    ----------
    omega : ndarray
        Frequency array (eV or rad/s -- consistent units required).
    eps_inf : float
        High-frequency dielectric constant.
    omega_p : float
        Plasma frequency (Drude term).
    gamma_d : float
        Drude damping rate.
    oscillators : list of (S_j, omega_j, gamma_j) or None
        Lorentz oscillator parameters: strength, resonance, damping.

    Returns
    -------
    eps : ndarray (complex)
    """
    omega = np.asarray(omega, dtype=float)
    eps = np.full_like(omega, eps_inf, dtype=complex)

    # Drude term.
    if omega_p > 0:
        denom = omega ** 2 + 1j * gamma_d * omega
        # Avoid division by zero at omega=0.
        safe = np.where(np.abs(denom) > 1e-30, denom, 1e-30 + 0j)
        eps -= omega_p ** 2 / safe

    # Lorentz oscillators.
    if oscillators is not None:
        for S_j, omega_j, gamma_j in oscillators:
            denom = omega_j ** 2 - omega ** 2 - 1j * gamma_j * omega
            safe = np.where(np.abs(denom) > 1e-30, denom, 1e-30 + 0j)
            eps += S_j * omega_j ** 2 / safe

    return eps


# ---------------------------------------------------------------------------
# 4. Kramers-Kronig Consistency Check
# ---------------------------------------------------------------------------

def kramers_kronig_check(omega, eps, component="real", eps_inf=None):
    """
    Verify Kramers-Kronig consistency of a dielectric function.

    Uses an FFT-based Hilbert transform on the full frequency line.
    The one-sided data chi''(w) (w > 0) is odd-extended to negative
    frequencies, then the Hilbert transform H is computed in O(N log N)
    via FFT:  H[f] = IFFT[-i * sign(freq) * FFT[f]].

    The KK relation for a causal susceptibility chi = eps - eps_inf is:
      chi'(w) = -H[chi''](w)    (real part from imaginary)
      chi''(w) = H[chi'](w)     (imaginary part from real)

    Parameters
    ----------
    omega : ndarray
        Frequency grid (positive, evenly spaced).
    eps : ndarray (complex)
        Dielectric function.
    component : str
        "real" -- reconstruct eps' from eps'', compare to actual eps'.
        "imag" -- reconstruct eps'' from eps', compare to actual eps''.
    eps_inf : float or None
        High-frequency permittivity.  If None, uses eps.real[-1].

    Returns
    -------
    reconstructed : ndarray
        The KK-reconstructed component.
    max_rel_error : float
        Maximum relative error between actual and reconstructed.
    """
    omega = np.asarray(omega, dtype=float)
    eps = np.asarray(eps, dtype=complex)
    N = len(omega)

    if eps_inf is None:
        eps_inf = float(eps.real[-1])

    if component == "real":
        # Reconstruct eps' from eps'' via chi' = -H[chi'']
        chi_imag = eps.imag  # chi''(w) for w > 0

        # Odd-extend to negative frequencies: chi''(-w) = -chi''(w)
        M = 2 * N
        signal = np.zeros(M)
        signal[:N] = -chi_imag[::-1]  # negative-w half (reversed)
        signal[N:] = chi_imag          # positive-w half

        # FFT Hilbert: H[f] = IFFT[-i * sgn(freq) * FFT[f]]
        F = np.fft.fft(signal)
        freqs = np.fft.fftfreq(M)
        sgn = np.sign(freqs)
        sgn[freqs == 0] = 0.0
        hilbert_signal = np.fft.ifft(-1j * sgn * F).real

        # Extract positive-frequency half; chi' = -H[chi'']
        reconstructed = eps_inf - hilbert_signal[N:]
        actual = eps.real

    else:
        # Reconstruct eps'' from eps' via chi'' = H[chi']
        chi_real = eps.real - eps_inf  # chi'(w) for w > 0

        # Even-extend to negative frequencies: chi'(-w) = chi'(w)
        M = 2 * N
        signal = np.zeros(M)
        signal[:N] = chi_real[::-1]   # negative-w half (reversed)
        signal[N:] = chi_real          # positive-w half

        F = np.fft.fft(signal)
        freqs = np.fft.fftfreq(M)
        sgn = np.sign(freqs)
        sgn[freqs == 0] = 0.0
        hilbert_signal = np.fft.ifft(-1j * sgn * F).real

        reconstructed = hilbert_signal[N:]
        actual = eps.imag

    # Relative error (skip near-zero regions).
    mask = np.abs(actual) > 0.01 * np.max(np.abs(actual))
    if np.any(mask):
        max_rel_error = float(np.max(
            np.abs(reconstructed[mask] - actual[mask]) / np.abs(actual[mask])
        ))
    else:
        max_rel_error = 0.0

    return reconstructed, max_rel_error


# ---------------------------------------------------------------------------
# 5. Transfer Matrix Method (TMM) for Multilayer Optics
# ---------------------------------------------------------------------------

def tmm_reflection(n_layers, d_layers, wavelength, theta_i=0.0, polarization="s"):
    """
    Transfer Matrix Method for a multilayer thin-film stack.

    Computes the complex reflection coefficient for a stack of layers
    bounded by semi-infinite incidence and substrate media.

    Parameters
    ----------
    n_layers : list of complex
        Refractive indices [n_incidence, n_1, n_2, ..., n_substrate].
        First and last are the semi-infinite bounding media.
    d_layers : list of float
        Thicknesses of the intermediate layers [d_1, d_2, ...].
        Length = len(n_layers) - 2 (bounding media have no thickness).
    wavelength : float
        Free-space wavelength (same units as d_layers).
    theta_i : float
        Angle of incidence (radians).
    polarization : str
        "s" (TE) or "p" (TM).

    Returns
    -------
    r : complex
        Complex reflection coefficient.
    R : float
        Reflectance |r|^2.
    """
    n_layers = [complex(n) for n in n_layers]
    n_inc = n_layers[0]

    # Snell's law for each layer.
    cos_theta = []
    for n in n_layers:
        sin_t = n_inc * np.sin(theta_i) / n
        cos_t = np.sqrt(1.0 - sin_t ** 2 + 0j)
        if cos_t.imag < 0:
            cos_t = -cos_t
        cos_theta.append(cos_t)

    # Admittance (eta) depends on polarization.
    if polarization == "s":
        eta = [n * ct for n, ct in zip(n_layers, cos_theta, strict=True)]
    else:
        eta = [n / ct if abs(ct) > 1e-30 else n * 1e30
               for n, ct in zip(n_layers, cos_theta, strict=True)]

    # Build transfer matrix M = M_1 * M_2 * ... * M_N
    M = np.eye(2, dtype=complex)
    for j in range(1, len(n_layers) - 1):
        n_j = n_layers[j]
        d_j = d_layers[j - 1]
        ct_j = cos_theta[j]
        delta_j = 2.0 * np.pi * n_j * ct_j * d_j / wavelength
        cos_d = np.cos(delta_j)
        sin_d = np.sin(delta_j)
        eta_j = eta[j]

        layer_matrix = np.array([
            [cos_d, 1j * sin_d / eta_j],
            [1j * eta_j * sin_d, cos_d],
        ], dtype=complex)
        M = M @ layer_matrix

    # Reflection coefficient.
    eta_inc = eta[0]
    eta_sub = eta[-1]
    r = ((M[0, 0] + M[0, 1] * eta_sub) * eta_inc
         - (M[1, 0] + M[1, 1] * eta_sub))
    r /= ((M[0, 0] + M[0, 1] * eta_sub) * eta_inc
           + (M[1, 0] + M[1, 1] * eta_sub))

    R = float(abs(r) ** 2)
    return r, R
