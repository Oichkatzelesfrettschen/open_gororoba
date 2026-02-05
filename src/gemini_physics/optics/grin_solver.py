"""
Rigorous Gradient-Index (GRIN) Ray Solver.
Solves dT/ds = (grad(n) - (T . grad(n)) T) / n via RK4.

Supports both real and complex refractive indices:
  - Real n: standard GRIN ray bending (rk4_step).
  - Complex n = n_r + i*k: ray path from grad(n_r),
    Beer-Lambert amplitude attenuation from k (rk4_step_absorbing).

At 1550nm in a Si/Au/Sapphire stack the skin depth in gold is ~12nm,
so rays entering the metallic ring are captured within a single step.
The complex solver tracks this absorption quantitatively rather than
using a phenomenological capture probability.

Ref: CSC KTH (2011) "Ray Tracing in Gradient-Index Media".
"""

import math

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
# Real-valued n: original solver (backward-compatible)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def rk4_step(pos, dir, dt, map_func_n, grad_func_n):
    """
    Performs one RK4 step for the ray equation (real n).
    y = [pos, dir]
    dy/ds = [dir, (grad_n - (dir.grad_n)dir)/n]
    """

    def get_derivatives(p, d):
        grad_n, n = grad_func_n(p)
        dot_vd = d[0]*grad_n[0] + d[1]*grad_n[1] + d[2]*grad_n[2]
        k = (grad_n - dot_vd * d) / n
        return d, k

    # k1
    v1, a1 = get_derivatives(pos, dir)

    # k2
    v2, a2 = get_derivatives(pos + v1 * dt * 0.5, dir + a1 * dt * 0.5)
    v2 = v2 / np.linalg.norm(v2)

    # k3
    v3, a3 = get_derivatives(pos + v2 * dt * 0.5, dir + a2 * dt * 0.5)
    v3 = v3 / np.linalg.norm(v3)

    # k4
    v4, a4 = get_derivatives(pos + v3 * dt, dir + a3 * dt)
    v4 = v4 / np.linalg.norm(v4)

    new_pos = pos + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    new_dir = dir + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)

    new_dir = new_dir / np.linalg.norm(new_dir)

    return new_pos, new_dir


@njit(fastmath=True)
def get_gradient_central(p, map_func_n, eps=0.001):
    """Central difference gradient estimation (real n)."""
    p_px = p.copy(); p_px[0] += eps
    p_mx = p.copy(); p_mx[0] -= eps
    nx = (map_func_n(p_px) - map_func_n(p_mx)) / (2*eps)

    p_py = p.copy(); p_py[1] += eps
    p_my = p.copy(); p_my[1] -= eps
    ny = (map_func_n(p_py) - map_func_n(p_my)) / (2*eps)

    p_pz = p.copy(); p_pz[2] += eps
    p_mz = p.copy(); p_mz[2] -= eps
    nz = (map_func_n(p_pz) - map_func_n(p_mz)) / (2*eps)

    n0 = map_func_n(p)
    return np.array([nx, ny, nz]), n0


# ---------------------------------------------------------------------------
# Complex-valued n: absorptive GRIN media
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def get_gradient_complex(p, map_func_n_complex, eps=0.001):
    """
    Central-difference gradient for complex refractive index.

    Returns (grad_re_n, n_complex) where grad_re_n is the 3-vector
    gradient of Re(n) (which drives ray curvature) and n_complex is
    the full complex n at the query point.
    """
    p_px = p.copy(); p_px[0] += eps
    p_mx = p.copy(); p_mx[0] -= eps
    nx = (map_func_n_complex(p_px).real - map_func_n_complex(p_mx).real) / (2*eps)

    p_py = p.copy(); p_py[1] += eps
    p_my = p.copy(); p_my[1] -= eps
    ny = (map_func_n_complex(p_py).real - map_func_n_complex(p_my).real) / (2*eps)

    p_pz = p.copy(); p_pz[2] += eps
    p_mz = p.copy(); p_mz[2] -= eps
    nz = (map_func_n_complex(p_pz).real - map_func_n_complex(p_mz).real) / (2*eps)

    n0 = map_func_n_complex(p)
    return np.array([nx, ny, nz]), n0


@njit(fastmath=True)
def rk4_step_absorbing(pos, dir, dt, grad_func_n_complex, wavelength_nm):
    """
    One RK4 step in absorptive GRIN media (complex n).

    The ray path follows grad(Re(n)); the amplitude decays via
    Beer-Lambert: dA/ds = -(2*pi/lambda)*Im(n)*A.

    Parameters
    ----------
    pos, dir : ndarray(3)
        Current position and unit tangent direction.
    dt : float
        Step size in arc length.
    grad_func_n_complex : callable
        p -> (grad_re_n, n_complex), as from get_gradient_complex.
    wavelength_nm : float
        Free-space wavelength in nm (e.g. 1550.0).

    Returns
    -------
    new_pos : ndarray(3)
    new_dir : ndarray(3)
        Updated unit tangent.
    atten : float
        Amplitude survival fraction for this step (in [0, 1]).
    phase_advance : float
        Optical phase accumulated over this step (radians).
    """
    def get_derivatives(p, d):
        grad_n_real, n_complex = grad_func_n_complex(p)
        n_real = n_complex.real
        if n_real < 1e-10:
            n_real = 1e-10
        dot_vd = (d[0]*grad_n_real[0] + d[1]*grad_n_real[1]
                  + d[2]*grad_n_real[2])
        curvature = (grad_n_real - dot_vd * d) / n_real
        return d, curvature, n_complex

    # k1
    v1, a1, n1 = get_derivatives(pos, dir)

    # k2
    d2 = dir + a1 * dt * 0.5
    d2 = d2 / np.linalg.norm(d2)
    v2, a2, n2 = get_derivatives(pos + v1 * dt * 0.5, d2)

    # k3
    d3 = dir + a2 * dt * 0.5
    d3 = d3 / np.linalg.norm(d3)
    v3, a3, n3 = get_derivatives(pos + v2 * dt * 0.5, d3)

    # k4
    d4 = dir + a3 * dt
    d4 = d4 / np.linalg.norm(d4)
    v4, a4, n4 = get_derivatives(pos + v3 * dt, d4)

    new_pos = pos + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
    new_dir = dir + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
    new_dir = new_dir / np.linalg.norm(new_dir)

    # Attenuation: Beer-Lambert using midpoint k (Simpson-like estimate).
    k_mid = n2.imag
    alpha = 2.0 * math.pi * k_mid / wavelength_nm
    exponent = alpha * dt
    if exponent < 500.0:
        atten = math.exp(-exponent)
    else:
        atten = 0.0

    # Phase advance using midpoint Re(n).
    n_real_mid = max(n2.real, 1e-10)
    phase_advance = 2.0 * math.pi * n_real_mid * dt / wavelength_nm

    return new_pos, new_dir, atten, phase_advance
