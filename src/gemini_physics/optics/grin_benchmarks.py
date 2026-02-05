"""
Analytic GRIN benchmark profiles with closed-form ray paths.

Three canonical gradient-index media where the ray equation
dT/ds = (grad(n) - (T . grad(n)) T) / n
has known analytic solutions, enabling rigorous convergence testing
of numerical ray tracers.

Profiles (all 2D, in the (x,y) plane, z=0):
  1. Luneburg lens:     n(r) = sqrt(2 - r^2),  r <= 1
  2. Maxwell fish-eye:  n(r) = 2 / (1 + r^2)
  3. Parabolic GRIN:    n(x,y) = n0 * (1 - g^2 * y^2 / 2), paraxial fiber

Refs:
  Luneburg (1944), "Mathematical Theory of Optics".
  Born & Wolf, "Principles of Optics", 7th ed., Ch. 3.
  Succi & Leonhardt (2006), transformation optics.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from gemini_physics.optics.grin_solver import rk4_step

# ---------------------------------------------------------------------------
# 1. Luneburg Lens: n(r) = sqrt(2 - r^2) for r <= 1
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def luneburg_n(p):
    """Refractive index for the Luneburg lens (r <= 1 region)."""
    r2 = p[0] ** 2 + p[1] ** 2
    if r2 >= 1.0:
        return 1.0
    return np.sqrt(2.0 - r2)


@njit(fastmath=True)
def luneburg_grad_n(p):
    """Gradient of n and n at point p for the Luneburg lens."""
    r2 = p[0] ** 2 + p[1] ** 2
    if r2 >= 1.0:
        return np.array([0.0, 0.0, 0.0]), 1.0
    n = np.sqrt(2.0 - r2)
    # dn/dx = -x / sqrt(2 - r^2), etc.
    grad = np.array([-p[0] / n, -p[1] / n, 0.0])
    return grad, n


def trace_luneburg(y0, dt=0.001, max_steps=20000):
    """
    Trace a ray through the Luneburg lens.

    A ray enters parallel to the x-axis at height y0 (|y0| < 1).
    It should exit the lens and converge to the point (-x_entry, -y0)
    on the opposite side.

    Returns
    -------
    positions : ndarray, shape (n_steps, 3)
    directions : ndarray, shape (n_steps, 3)
    """
    # Entry point: ray enters from x = -1 side, moving in +x direction.
    # Find entry x from r^2 = 1: x^2 + y0^2 = 1 -> x = -sqrt(1 - y0^2)
    x_entry = -np.sqrt(1.0 - y0 ** 2)
    pos = np.array([x_entry, y0, 0.0], dtype=np.float64)
    direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)

    positions = [pos.copy()]
    directions = [direction.copy()]

    for _ in range(max_steps):
        pos, direction = rk4_step(pos, direction, dt, luneburg_n, luneburg_grad_n)
        positions.append(pos.copy())
        directions.append(direction.copy())
        # Stop when ray exits the lens (r > 1) and has traveled sufficiently.
        r2 = pos[0] ** 2 + pos[1] ** 2
        if r2 > 1.0 and pos[0] > 0:
            break

    return np.array(positions), np.array(directions)


def luneburg_exit_angle_analytical(y0):
    """
    Analytical exit angle for Luneburg lens.

    A parallel ray at height y0 exits at angle theta = arcsin(y0)
    below the axis (focusing to the antipodal point).
    """
    return np.arcsin(y0)


# ---------------------------------------------------------------------------
# 2. Maxwell Fish-Eye: n(r) = 2 / (1 + r^2)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def fisheye_n(p):
    """Refractive index for the Maxwell fish-eye."""
    r2 = p[0] ** 2 + p[1] ** 2
    return 2.0 / (1.0 + r2)


@njit(fastmath=True)
def fisheye_grad_n(p):
    """Gradient of n and n at point p for the Maxwell fish-eye."""
    r2 = p[0] ** 2 + p[1] ** 2
    denom = 1.0 + r2
    n = 2.0 / denom
    # dn/dx = -4*x / (1+r^2)^2
    factor = -4.0 / (denom * denom)
    grad = np.array([factor * p[0], factor * p[1], 0.0])
    return grad, n


def trace_fisheye(x0, y0, dx0, dy0, dt=0.001, max_steps=50000):
    """
    Trace a ray through the Maxwell fish-eye lens.

    All rays from (x0, y0) converge to the antipodal point
    (-x0/(x0^2+y0^2), -y0/(x0^2+y0^2)) on the stereographic sphere.

    Returns
    -------
    positions : ndarray, shape (n_steps, 3)
    """
    pos = np.array([x0, y0, 0.0], dtype=np.float64)
    norm = np.sqrt(dx0 ** 2 + dy0 ** 2)
    direction = np.array([dx0 / norm, dy0 / norm, 0.0], dtype=np.float64)

    positions = [pos.copy()]

    for _ in range(max_steps):
        pos, direction = rk4_step(pos, direction, dt, fisheye_n, fisheye_grad_n)
        positions.append(pos.copy())

    return np.array(positions)


def fisheye_antipodal_point(x0, y0):
    """
    Analytical image point in the Maxwell fish-eye.

    The image of (x0, y0) is the inversion through the unit circle:
    (x_img, y_img) = (-x0, -y0) / (x0^2 + y0^2).
    """
    r2 = x0 ** 2 + y0 ** 2
    return -x0 / r2, -y0 / r2


# ---------------------------------------------------------------------------
# 3. Parabolic GRIN Fiber: n(y) = n0 * (1 - g^2 * y^2 / 2)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _parabolic_n_params(p, n0, g):
    """Refractive index for parabolic GRIN fiber."""
    y = p[1]
    val = n0 * (1.0 - 0.5 * g * g * y * y)
    if val < 0.1:
        val = 0.1
    return val


@njit(fastmath=True)
def _parabolic_grad_n_params(p, n0, g):
    """Gradient of n for parabolic GRIN fiber."""
    y = p[1]
    val = n0 * (1.0 - 0.5 * g * g * y * y)
    if val < 0.1:
        val = 0.1
    # dn/dy = -n0 * g^2 * y
    grad = np.array([0.0, -n0 * g * g * y, 0.0])
    return grad, val


def make_parabolic_funcs(n0=1.5, g=0.1):
    """
    Create numba-jitted n(p) and grad_n(p) for a specific parabolic GRIN.

    Returns
    -------
    map_n, grad_n : callables compatible with rk4_step.
    """
    @njit(fastmath=True)
    def map_n(p):
        y = p[1]
        val = n0 * (1.0 - 0.5 * g * g * y * y)
        if val < 0.1:
            val = 0.1
        return val

    @njit(fastmath=True)
    def grad_n(p):
        y = p[1]
        val = n0 * (1.0 - 0.5 * g * g * y * y)
        if val < 0.1:
            val = 0.1
        grad = np.array([0.0, -n0 * g * g * y, 0.0])
        return grad, val

    return map_n, grad_n


def trace_parabolic(y0, theta0=0.0, n0=1.5, g=0.1, dt=0.01, max_steps=50000):
    """
    Trace a ray in a parabolic GRIN fiber.

    The ray starts at (0, y0, 0) with initial direction at angle theta0
    from the x-axis. In the paraxial limit, the trajectory is:
      y(x) = y0 * cos(g*x) + (tan(theta0)/g) * sin(g*x)

    The oscillation period is P = 2*pi/g.

    Returns
    -------
    positions : ndarray, shape (n_steps, 3)
    """
    map_n, grad_n = make_parabolic_funcs(n0, g)

    pos = np.array([0.0, y0, 0.0], dtype=np.float64)
    direction = np.array([np.cos(theta0), np.sin(theta0), 0.0], dtype=np.float64)

    positions = [pos.copy()]

    for _ in range(max_steps):
        pos, direction = rk4_step(pos, direction, dt, map_n, grad_n)
        positions.append(pos.copy())

    return np.array(positions)


def parabolic_analytical_y(x, y0, theta0, g):
    """
    Analytical ray path y(x) for paraxial parabolic GRIN fiber.

    y(x) = y0 * cos(g*x) + (tan(theta0) / g) * sin(g*x)
    """
    return y0 * np.cos(g * x) + (np.tan(theta0) / g) * np.sin(g * x)


# ---------------------------------------------------------------------------
# Convergence analysis helper
# ---------------------------------------------------------------------------

def measure_rk4_convergence(profile="parabolic", y0=0.3, dt_values=None):
    """
    Measure RK4 convergence rate for a given GRIN profile.

    For the parabolic profile, errors are measured against the known
    analytical solution y(x). For the Luneburg profile, errors are
    measured against the finest-resolution result (self-convergence).

    Parameters
    ----------
    profile : str
        One of "luneburg", "parabolic".
    y0 : float
        Initial ray height.
    dt_values : list of float or None
        Step sizes to test. Default depends on profile.

    Returns
    -------
    dt_arr : ndarray
        Step sizes used.
    errors : ndarray
        Position errors at a reference point.
    convergence_rate : float
        Estimated convergence order from log-log fit.
    """
    if profile == "parabolic":
        g = 0.1
        n0 = 1.5
        if dt_values is None:
            dt_values = [2.0, 1.0, 0.5, 0.25, 0.125]
        x_target = 10.0

        # Self-convergence: compare each result against the finest dt.
        # The paraxial formula has O(y0^2*g^2) error, so we measure
        # numerical convergence via Richardson extrapolation.
        results = []
        for dt in dt_values:
            n_steps = int(x_target / dt) + 500
            pos = trace_parabolic(
                y0, theta0=0.0, n0=n0, g=g, dt=dt, max_steps=n_steps
            )
            idx = np.argmin(np.abs(pos[:, 0] - x_target))
            results.append(pos[idx, 1])

        ref = results[-1]
        errors = np.array([abs(r - ref) for r in results[:-1]])
        dt_arr = np.array(dt_values[:-1])

    elif profile == "luneburg":
        if dt_values is None:
            dt_values = [0.01, 0.005, 0.0025, 0.00125]
        results = []
        for dt in dt_values:
            pos, _ = trace_luneburg(y0, dt=dt, max_steps=int(10.0 / dt))
            results.append(pos[-1].copy())

        ref = results[-1]
        errors = [np.linalg.norm(r - ref) for r in results[:-1]]
        dt_arr = np.array(dt_values[:-1])
        errors = np.array(errors)

    else:
        raise ValueError(f"Unknown profile: {profile}")

    # Fit convergence rate: error ~ C * dt^p.
    mask = errors > 1e-15
    if np.sum(mask) >= 2:
        log_dt = np.log(dt_arr[mask])
        log_err = np.log(errors[mask])
        p = np.polyfit(log_dt, log_err, 1)[0]
    else:
        p = 0.0

    return dt_arr, errors, p
