"""
D2Q9 Lattice Boltzmann Method (BGK) with streaming, boundary conditions,
and macroscopic variable extraction.

Supports:
  - Periodic BCs (default in x-direction for Poiseuille benchmark).
  - Bounce-back BCs (no-slip walls, applied in y-direction for Poiseuille).
  - Body-force driven flow for Poiseuille channel benchmark.

The D2Q9 velocity set uses the standard ordering:
  i:   0   1   2   3   4   5   6   7   8
  cx:  0   1   0  -1   0   1  -1  -1   1
  cy:  0   0   1   0  -1   1   1  -1  -1
  w: 4/9 1/9 1/9 1/9 1/9 1/36 1/36 1/36 1/36

The main simulation function uses a Rust backend (gororoba_py) when available
for improved performance, falling back to pure NumPy otherwise.

Ref: Succi, "The Lattice Boltzmann Equation" (OUP, 2018), Ch. 10-12.
"""

from __future__ import annotations

import numpy as np

# Try to import Rust bindings first
_USE_RUST = False
try:
    import gororoba_py as _gp
    _USE_RUST = True
except ImportError:
    pass

# D2Q9 lattice constants.
W = np.array([4.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9, 1.0 / 9,
              1.0 / 36, 1.0 / 36, 1.0 / 36, 1.0 / 36])
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
# Opposite direction index for bounce-back.
OPP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])


def equilibrium(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray) -> np.ndarray:
    """
    Compute D2Q9 equilibrium distributions.

    Parameters
    ----------
    rho : ndarray, shape (nx, ny)
        Macroscopic density field.
    ux, uy : ndarray, shape (nx, ny)
        Macroscopic velocity components.

    Returns
    -------
    feq : ndarray, shape (9, nx, ny)
        Equilibrium distribution functions.
    """
    u_sq = ux**2 + uy**2
    feq = np.zeros((9, rho.shape[0], rho.shape[1]))
    for i in range(9):
        cu = CX[i] * ux + CY[i] * uy
        feq[i] = rho * W[i] * (1.0 + 3.0 * cu + 4.5 * cu**2 - 1.5 * u_sq)
    return feq


def macroscopic(f: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract macroscopic variables from distribution functions.

    Returns
    -------
    rho, ux, uy : each ndarray, shape (nx, ny)
    """
    rho = np.sum(f, axis=0)
    # Guard against division by zero in vacuum cells.
    rho_safe = np.where(rho > 1e-30, rho, 1.0)
    ux = np.sum(f * CX[:, None, None], axis=0) / rho_safe
    uy = np.sum(f * CY[:, None, None], axis=0) / rho_safe
    return rho, ux, uy


def stream(f: np.ndarray) -> np.ndarray:
    """
    Streaming step: propagate each population along its lattice velocity.

    Uses np.roll for periodic boundaries in both x and y.
    Walls (bounce-back) are applied *after* streaming by the caller.
    """
    f_out = np.empty_like(f)
    for i in range(9):
        f_out[i] = np.roll(np.roll(f[i], CX[i], axis=0), CY[i], axis=1)
    return f_out


def bounce_back_top_bottom(f: np.ndarray) -> np.ndarray:
    """
    Apply full-way bounce-back on the top and bottom rows (y=0, y=ny-1).

    After streaming, populations that would have left through a wall
    are reflected back by swapping with their opposite direction at
    the wall site.
    """
    ny = f.shape[2]
    for i in range(9):
        if CY[i] == 1:
            # Population streamed *into* the top wall -- reflect.
            f[OPP[i], :, ny - 1] = f[i, :, ny - 1]
        elif CY[i] == -1:
            # Population streamed *into* the bottom wall -- reflect.
            f[OPP[i], :, 0] = f[i, :, 0]
    return f


def add_body_force(f: np.ndarray, rho: np.ndarray, fx: float) -> np.ndarray:
    """
    Add a constant body force in the x-direction (Guo forcing scheme, 1st order).

    For Poiseuille flow, this drives the flow between walls.
    Ref: Guo, Zheng & Shi, PRE 65 (2002) 046308.
    """
    for i in range(9):
        f[i] += 3.0 * W[i] * CX[i] * fx * rho
    return f


def simulate_poiseuille(
    nx: int = 3,
    ny: int = 41,
    tau: float = 0.8,
    fx: float = 1e-5,
    n_steps: int = 5000,
) -> dict:
    """
    Simulate 2D Poiseuille flow between parallel walls.

    Walls at y=0 and y=ny-1 (bounce-back), periodic in x.
    Body force fx drives flow in x-direction.

    Uses Rust backend when gororoba_py is available for improved performance.

    Analytical solution (channel half-width H = (ny-2)/2):
      u_x(y) = fx / (2*nu) * y * (ny - 1 - y)
    where nu = (tau - 0.5) / 3.

    Parameters
    ----------
    nx : int
        Grid points in x (periodic; 3 is enough for 1D profile).
    ny : int
        Grid points in y (including two wall layers).
    tau : float
        BGK relaxation time (> 0.5 for stability).
    fx : float
        Body force in x-direction per lattice unit.
    n_steps : int
        Number of LBM time steps.

    Returns
    -------
    result : dict with keys:
        'y' : y-coordinate array (interior points).
        'ux_numerical' : simulated x-velocity profile at x=nx//2.
        'ux_analytical' : analytical parabolic profile.
        'max_rel_error' : max |u_num - u_ana| / max(u_ana).
        'mass_history' : total mass at each step (Python backend only).
        'rho_final' : final density field (Python backend only).
    """
    if _USE_RUST:
        # Use Rust backend (returns core arrays, no mass_history/rho_final)
        y_arr, ux_num, ux_ana, max_err = _gp.py_simulate_poiseuille(
            nx, ny, tau, fx, n_steps
        )
        return {
            "y": np.asarray(y_arr),
            "ux_numerical": np.asarray(ux_num),
            "ux_analytical": np.asarray(ux_ana),
            "max_rel_error": max_err,
            "mass_history": None,  # Not available in Rust backend
            "initial_mass": None,
            "rho_final": None,
        }

    # Python fallback
    omega = 1.0 / tau
    nu = (tau - 0.5) / 3.0

    # Initialize: uniform density, zero velocity.
    rho = np.ones((nx, ny))
    ux = np.zeros((nx, ny))
    uy = np.zeros((nx, ny))
    f = equilibrium(rho, ux, uy)

    mass_history = np.empty(n_steps)
    initial_mass = np.sum(f)

    for step in range(n_steps):
        # Record mass.
        mass_history[step] = np.sum(f)

        # Macroscopic variables (all nodes, for diagnostics).
        rho, ux, uy = macroscopic(f)

        # Collision (BGK) -- interior fluid nodes only.
        feq = equilibrium(rho, ux, uy)
        f[:, :, 1:ny - 1] = (
            f[:, :, 1:ny - 1]
            + omega * (feq[:, :, 1:ny - 1] - f[:, :, 1:ny - 1])
        )

        # Body force -- interior fluid nodes only.
        for i in range(9):
            f[i, :, 1:ny - 1] += (
                3.0 * W[i] * CX[i] * fx * rho[:, 1:ny - 1]
            )

        # Streaming.
        f = stream(f)

        # Bounce-back walls at y=0 and y=ny-1.
        f = bounce_back_top_bottom(f)

    # Final macroscopic fields.
    rho, ux, uy = macroscopic(f)

    # Extract profile at the middle x-slice (should be uniform in x).
    mid_x = nx // 2
    ux_profile = ux[mid_x, :]

    # Analytical Poiseuille profile.
    # With bounce-back on wall nodes y=0 and y=ny-1, the effective
    # no-slip boundary sits at y=0.5 and y=ny-1.5 (halfway convention
    # intrinsic to the simple bounce-back rule).
    # u(y) = fx/(2*nu) * (y - 0.5) * (ny - 1.5 - y).
    y = np.arange(ny)
    ux_analytical = np.zeros(ny)
    for j in range(ny):
        yf = float(j)
        val = fx / (2.0 * nu) * (yf - 0.5) * (ny - 1.5 - yf)
        ux_analytical[j] = max(val, 0.0)

    # Interior points only (skip wall cells).
    interior = slice(1, ny - 1)
    max_ana = np.max(ux_analytical[interior])
    if max_ana > 1e-30:
        max_rel_error = np.max(np.abs(
            ux_profile[interior] - ux_analytical[interior]
        )) / max_ana
    else:
        max_rel_error = 0.0

    return {
        "y": y,
        "ux_numerical": ux_profile,
        "ux_analytical": ux_analytical,
        "max_rel_error": max_rel_error,
        "mass_history": mass_history,
        "initial_mass": initial_mass,
        "rho_final": rho,
    }
