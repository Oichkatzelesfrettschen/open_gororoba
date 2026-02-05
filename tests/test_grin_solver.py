"""
Tests for the Rigorous GRIN Ray Solver.
Verifies arclength parameterization and unit tangent preservation.
"""

import numpy as np
from numba import njit

from gemini_physics.optics.grin_solver import get_gradient_central, rk4_step


def test_unit_tangent_preservation():
    """RK4 step must preserve |T|=1."""
    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dir = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    dt = 0.1

    # Linear gradient n = 1 + x
    @njit
    def map_n(p):
        return 1.0 + p[0]

    @njit
    def grad_n(p):
        return np.array([1.0, 0.0, 0.0], dtype=np.float64), map_n(p)

    new_p, new_d = rk4_step(pos, dir, dt, map_n, grad_n)

    assert np.isclose(np.linalg.norm(new_d), 1.0, atol=1e-12)
    assert new_p[0] > pos[0] # Moving forward

def test_central_gradient():
    """Verifies gradient accuracy on a sphere."""
    # n = x^2 + y^2 + z^2
    @njit
    def map_n(p):
        return p[0]**2 + p[1]**2 + p[2]**2

    p = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    grad, n = get_gradient_central(p, map_n, eps=0.0001)

    # Analytical grad = 2*p = [2, 4, 6]
    expected_grad = 2.0 * p
    assert np.allclose(grad, expected_grad, atol=1e-6)
    assert np.isclose(n, 14.0)

def test_linear_bending():
    """Verify ray bends towards higher index."""
    # n = 1 + y (higher at y > 0)
    @njit
    def map_n(p):
        return 1.0 + p[1]

    @njit
    def grad_n(p):
        return np.array([0.0, 1.0, 0.0], dtype=np.float64), map_n(p)

    pos = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    dir = np.array([1.0, 0.0, 0.0], dtype=np.float64) # Moving along X
    dt = 0.1

    p, d = rk4_step(pos, dir, dt, map_n, grad_n)

    # Bending towards +Y (higher n)
    assert d[1] > 0.0
    assert d[0] < 1.0 # Tangent component in X must decrease to keep norm 1
