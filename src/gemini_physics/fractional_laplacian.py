"""
Fractional Laplacian operators.

This module provides spectral fractional Laplacian operators for periodic and
Dirichlet boundary conditions in 1D, 2D, and 3D.

The periodic 1D case uses a Rust backend (gororoba_py) when available for
improved performance, falling back to NumPy FFT otherwise.
"""
from __future__ import annotations

import numpy as np
from scipy.fft import dst, idst

# Try to import Rust bindings first
_USE_RUST = False
try:
    import gororoba_py as _gp
    _USE_RUST = True
except ImportError:
    pass


def fractional_laplacian_periodic_1d(
    u: np.ndarray, *, s: float, L: float = 1.0
) -> np.ndarray:
    """
    Discrete 1D periodic (-\\Delta)^s via Fourier multiplier on an evenly spaced grid.

    This corresponds to a torus/periodic-domain model (not a bounded-domain Dirichlet model).

    Uses Rust backend when gororoba_py is available for improved performance.

    Args:
        u: Samples on a periodic grid (length N).
        s: Fractional power in (0, 1] (also works for s > 0, but interpret with care).
        L: Domain length.

    Returns:
        Array of same shape as u: (-\\Delta)^s u.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 1:
        raise ValueError("Expected a 1D array.")
    if L <= 0:
        raise ValueError("Expected L > 0.")

    n = u.shape[0]
    if n < 2:
        return np.zeros_like(u)

    if _USE_RUST:
        # Use Rust backend
        result = _gp.py_fractional_laplacian_1d(list(u), s, L)
        return np.asarray(result)

    # NumPy FFT fallback
    # Frequencies (cycles per unit length) -> angular frequencies.
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)
    mult = np.abs(k) ** (2.0 * s)
    uhat = np.fft.fft(u)
    out = np.fft.ifft(mult * uhat)
    return out.real


def _dirichlet_laplacian_eigs_1d(n: int, h: float) -> np.ndarray:
    # Eigenvalues of the standard 2nd-difference Dirichlet Laplacian on n interior points.
    k = np.arange(1, n + 1)
    return (4.0 / (h * h)) * (np.sin(0.5 * np.pi * k / (n + 1)) ** 2)


def fractional_laplacian_dirichlet_1d(
    u_interior: np.ndarray, *, s: float, L: float = 1.0
) -> np.ndarray:
    """
    Discrete 1D Dirichlet (-\\Delta)^s as a fractional power of the *discrete* Dirichlet Laplacian.

    We work on n interior points (boundary values u(0)=u(L)=0 are not included).
    This is the spectral definition in the discrete sine basis.

    Args:
        u_interior: Values on interior grid points (length n).
        s: Fractional power in (0, 1] (also works for s > 0).
        L: Domain length.

    Returns:
        Array of same shape as u_interior: (-\\Delta)^s u (discrete-spectral).
    """
    u = np.asarray(u_interior, dtype=float)
    if u.ndim != 1:
        raise ValueError("Expected a 1D array.")
    if L <= 0:
        raise ValueError("Expected L > 0.")

    n = u.shape[0]
    if n < 1:
        return np.zeros_like(u)

    h = L / (n + 1)
    lam = _dirichlet_laplacian_eigs_1d(n, h) ** s

    # DST-I diagonalizes the Dirichlet Laplacian on interior points.
    coeff = dst(u, type=1, norm="ortho")
    out = idst(lam * coeff, type=1, norm="ortho")
    return out


def fractional_laplacian_periodic_2d(
    u: np.ndarray, *, s: float, Lx: float = 1.0, Ly: float = 1.0
) -> np.ndarray:
    """
    Discrete 2D periodic (-Delta)^s via Fourier multiplier.

    Args:
        u: Samples on a periodic grid of shape (Nx, Ny).
        s: Fractional power (s > 0).
        Lx, Ly: Domain lengths in each direction.

    Returns:
        Array of same shape as u: (-Delta)^s u.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 2:
        raise ValueError("Expected a 2D array.")

    nx, ny = u.shape
    if nx < 2 or ny < 2:
        return np.zeros_like(u)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    mult = (KX**2 + KY**2) ** s
    uhat = np.fft.fft2(u)
    out = np.fft.ifft2(mult * uhat)
    return out.real


def fractional_laplacian_periodic_3d(
    u: np.ndarray, *, s: float, Lx: float = 1.0, Ly: float = 1.0, Lz: float = 1.0
) -> np.ndarray:
    """
    Discrete 3D periodic (-Delta)^s via Fourier multiplier.

    Args:
        u: Samples on a periodic grid of shape (Nx, Ny, Nz).
        s: Fractional power (s > 0).
        Lx, Ly, Lz: Domain lengths.

    Returns:
        Array of same shape as u: (-Delta)^s u.
    """
    u = np.asarray(u, dtype=float)
    if u.ndim != 3:
        raise ValueError("Expected a 3D array.")

    nx, ny, nz = u.shape
    if nx < 2 or ny < 2 or nz < 2:
        return np.zeros_like(u)

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=Lx / nx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=Ly / ny)
    kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=Lz / nz)
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    mult = (KX**2 + KY**2 + KZ**2) ** s
    uhat = np.fft.fftn(u)
    out = np.fft.ifftn(mult * uhat)
    return out.real


def fractional_laplacian_dirichlet_2d(
    u_interior: np.ndarray, *, s: float, Lx: float = 1.0, Ly: float = 1.0
) -> np.ndarray:
    """
    Discrete 2D Dirichlet (-Delta)^s via tensor-product DST-I.

    The 2D Dirichlet Laplacian eigenvalues factor as
    lambda_{j,k} = lambda_j^x + lambda_k^y, and the eigenvectors are
    tensor products of 1D sine modes.  We apply (-Delta)^s by raising
    these eigenvalues to the power s in the sine basis.

    Args:
        u_interior: Values on interior grid points, shape (nx, ny).
        s: Fractional power (s > 0).
        Lx, Ly: Domain lengths.

    Returns:
        Array of same shape: (-Delta)^s u (discrete-spectral).
    """
    u = np.asarray(u_interior, dtype=float)
    if u.ndim != 2:
        raise ValueError("Expected a 2D array.")

    nx, ny = u.shape
    if nx < 1 or ny < 1:
        return np.zeros_like(u)

    hx = Lx / (nx + 1)
    hy = Ly / (ny + 1)
    lam_x = _dirichlet_laplacian_eigs_1d(nx, hx)
    lam_y = _dirichlet_laplacian_eigs_1d(ny, hy)

    # Tensor-product eigenvalues: lambda_{j,k} = lam_x[j] + lam_y[k]
    LAM = lam_x[:, None] + lam_y[None, :]
    mult = LAM ** s

    # 2D DST-I: apply along each axis (scipy dst works on last axis).
    coeff = dst(dst(u, type=1, norm="ortho", axis=0), type=1, norm="ortho", axis=1)
    out = idst(idst(mult * coeff, type=1, norm="ortho", axis=0), type=1, norm="ortho", axis=1)
    return out


def dirichlet_laplacian_1d(u_interior: np.ndarray, *, L: float = 1.0) -> np.ndarray:
    """
    Standard 1D second-difference Dirichlet Laplacian on interior points.
    """
    u = np.asarray(u_interior, dtype=float)
    n = u.shape[0]
    if n < 1:
        return np.zeros_like(u)
    h = L / (n + 1)
    out = np.zeros_like(u)
    for i in range(n):
        left = u[i - 1] if i - 1 >= 0 else 0.0
        right = u[i + 1] if i + 1 < n else 0.0
        out[i] = (left - 2.0 * u[i] + right) / (h * h)
    return out
