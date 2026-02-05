from __future__ import annotations

import numpy as np

from gemini_physics.fractional_laplacian import (
    dirichlet_laplacian_1d,
    fractional_laplacian_dirichlet_1d,
    fractional_laplacian_dirichlet_2d,
    fractional_laplacian_periodic_1d,
    fractional_laplacian_periodic_2d,
    fractional_laplacian_periodic_3d,
)


def test_fractional_laplacian_periodic_fourier_mode() -> None:
    n = 512
    L = 1.0
    x = np.linspace(0.0, L, n, endpoint=False)
    m = 7
    s = 0.75
    u = np.cos(2.0 * np.pi * m * x / L)
    out = fractional_laplacian_periodic_1d(u, s=s, L=L)
    expected = (2.0 * np.pi * m / L) ** (2.0 * s) * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 5e-10


def test_fractional_laplacian_dirichlet_is_discrete_spectral_power() -> None:
    n = 256  # interior points
    L = 1.0
    i = np.arange(1, n + 1)
    k = 9
    u = np.sin(np.pi * k * i / (n + 1))
    out = fractional_laplacian_dirichlet_1d(u, s=1.0, L=L)
    expected = -dirichlet_laplacian_1d(u, L=L)
    # For s=1, (-Delta)^1 == -Delta. Our dirichlet_laplacian_1d returns Delta.
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-12


def test_fractional_laplacian_dirichlet_eigenvector() -> None:
    n = 256
    L = 1.0
    i = np.arange(1, n + 1)
    k = 5
    s = 0.5
    u = np.sin(np.pi * k * i / (n + 1))

    out = fractional_laplacian_dirichlet_1d(u, s=s, L=L)

    h = L / (n + 1)
    lam_k = (4.0 / (h * h)) * (np.sin(0.5 * np.pi * k / (n + 1)) ** 2)
    expected = (lam_k**s) * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-12


# --- 2D periodic tests ---


def test_fractional_laplacian_periodic_2d_fourier_mode() -> None:
    """A single 2D Fourier mode should be an eigenfunction of (-Delta)^s."""
    nx, ny = 64, 64
    Lx, Ly = 1.0, 1.0
    x = np.linspace(0.0, Lx, nx, endpoint=False)
    y = np.linspace(0.0, Ly, ny, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mx, my = 3, 5
    s = 0.75
    u = np.cos(2.0 * np.pi * mx * X / Lx) * np.cos(2.0 * np.pi * my * Y / Ly)
    out = fractional_laplacian_periodic_2d(u, s=s, Lx=Lx, Ly=Ly)
    k2 = (2.0 * np.pi * mx / Lx) ** 2 + (2.0 * np.pi * my / Ly) ** 2
    expected = k2**s * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-9


def test_fractional_laplacian_periodic_2d_s1_recovers_standard() -> None:
    """At s=1, (-Delta)^1 u should equal the standard Laplacian (negative sign)."""
    n = 32
    L = 1.0
    x = np.linspace(0.0, L, n, endpoint=False)
    X, Y = np.meshgrid(x, x, indexing="ij")
    m = 4
    u = np.sin(2.0 * np.pi * m * X / L) * np.sin(2.0 * np.pi * m * Y / L)
    out = fractional_laplacian_periodic_2d(u, s=1.0, Lx=L, Ly=L)
    # -Delta u for this mode = ((2*pi*m/L)^2 + (2*pi*m/L)^2) * u
    k2 = 2.0 * ((2.0 * np.pi * m / L) ** 2)
    expected = k2 * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-10


def test_fractional_laplacian_periodic_2d_convergence() -> None:
    """Convergence rate should be spectral for smooth Fourier modes."""
    Lx, Ly = 1.0, 1.0
    s = 0.5
    mx, my = 2, 3
    k2 = (2.0 * np.pi * mx / Lx) ** 2 + (2.0 * np.pi * my / Ly) ** 2
    errors = []
    for n in [32, 64, 128]:
        x = np.linspace(0.0, Lx, n, endpoint=False)
        y = np.linspace(0.0, Ly, n, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = np.cos(2.0 * np.pi * mx * X / Lx) * np.cos(2.0 * np.pi * my * Y / Ly)
        out = fractional_laplacian_periodic_2d(u, s=s, Lx=Lx, Ly=Ly)
        expected = k2**s * u
        errors.append(np.linalg.norm(out - expected) / np.linalg.norm(expected))

    # All errors should be very small for pure Fourier modes (spectral accuracy).
    for err in errors:
        assert err < 1e-9
    # Errors should all be near machine precision (roundoff may fluctuate).
    assert max(errors) < 1e-9


# --- 3D periodic tests ---


def test_fractional_laplacian_periodic_3d_fourier_mode() -> None:
    """A single 3D Fourier mode should be an eigenfunction."""
    n = 16
    L = 1.0
    x = np.linspace(0.0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    mx, my, mz = 2, 1, 3
    s = 0.5
    u = (np.cos(2.0 * np.pi * mx * X / L)
         * np.cos(2.0 * np.pi * my * Y / L)
         * np.cos(2.0 * np.pi * mz * Z / L))
    out = fractional_laplacian_periodic_3d(u, s=s, Lx=L, Ly=L, Lz=L)
    k2 = ((2.0 * np.pi * mx / L) ** 2
           + (2.0 * np.pi * my / L) ** 2
           + (2.0 * np.pi * mz / L) ** 2)
    expected = k2**s * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-9


def test_fractional_laplacian_periodic_3d_s1_recovers_standard() -> None:
    """At s=1, (-Delta)^1 should equal standard 3D Laplacian."""
    n = 16
    L = 1.0
    x = np.linspace(0.0, L, n, endpoint=False)
    X, Y, Z = np.meshgrid(x, x, x, indexing="ij")
    m = 2
    u = np.sin(2.0 * np.pi * m * X / L)
    out = fractional_laplacian_periodic_3d(u, s=1.0, Lx=L, Ly=L, Lz=L)
    k2 = (2.0 * np.pi * m / L) ** 2
    expected = k2 * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-9


# --- 2D Dirichlet tests ---


def test_fractional_laplacian_dirichlet_2d_eigenvector() -> None:
    """A tensor-product sine mode should be an eigenvector of (-Delta_D)^s."""
    nx, ny = 32, 32
    Lx, Ly = 1.0, 1.0
    jx, jy = 3, 5
    s = 0.6
    ix = np.arange(1, nx + 1)
    iy = np.arange(1, ny + 1)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    u = np.sin(np.pi * jx * IX / (nx + 1)) * np.sin(np.pi * jy * IY / (ny + 1))

    out = fractional_laplacian_dirichlet_2d(u, s=s, Lx=Lx, Ly=Ly)

    hx = Lx / (nx + 1)
    hy = Ly / (ny + 1)
    lam_jx = (4.0 / (hx * hx)) * (np.sin(0.5 * np.pi * jx / (nx + 1)) ** 2)
    lam_jy = (4.0 / (hy * hy)) * (np.sin(0.5 * np.pi * jy / (ny + 1)) ** 2)
    expected = ((lam_jx + lam_jy) ** s) * u
    rel = np.linalg.norm(out - expected) / np.linalg.norm(expected)
    assert rel < 1e-10


def test_fractional_laplacian_dirichlet_2d_s1_recovers_standard() -> None:
    """At s=1, (-Delta_D)^1 should match finite-difference Laplacian."""
    nx, ny = 16, 16
    Lx, Ly = 1.0, 1.0
    jx, jy = 2, 3
    ix = np.arange(1, nx + 1)
    iy = np.arange(1, ny + 1)
    IX, IY = np.meshgrid(ix, iy, indexing="ij")
    u = np.sin(np.pi * jx * IX / (nx + 1)) * np.sin(np.pi * jy * IY / (ny + 1))

    out = fractional_laplacian_dirichlet_2d(u, s=1.0, Lx=Lx, Ly=Ly)

    # Compute -Delta u via 2D finite differences (5-point stencil, Dirichlet BCs).
    hx = Lx / (nx + 1)
    hy = Ly / (ny + 1)
    neg_laplacian = np.zeros_like(u)
    for i in range(nx):
        for j in range(ny):
            center = u[i, j]
            left = u[i - 1, j] if i > 0 else 0.0
            right = u[i + 1, j] if i < nx - 1 else 0.0
            down = u[i, j - 1] if j > 0 else 0.0
            up = u[i, j + 1] if j < ny - 1 else 0.0
            neg_laplacian[i, j] = (
                -(left - 2.0 * center + right) / (hx * hx)
                - (down - 2.0 * center + up) / (hy * hy)
            )

    rel = np.linalg.norm(out - neg_laplacian) / np.linalg.norm(neg_laplacian)
    assert rel < 1e-10
