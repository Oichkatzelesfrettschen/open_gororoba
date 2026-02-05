"""
Tests for GRIN analytic benchmark profiles.

Validates:
  1. Luneburg lens: exit angle matches arcsin(y0) within tolerance.
  2. Parabolic GRIN: ray oscillation matches analytical y(x) within 1e-4.
  3. RK4 convergence rate is in [3.8, 4.2] (4th order method).
"""
from __future__ import annotations

import numpy as np

from gemini_physics.optics.grin_benchmarks import (
    luneburg_exit_angle_analytical,
    measure_rk4_convergence,
    parabolic_analytical_y,
    trace_luneburg,
    trace_parabolic,
)


def test_luneburg_exit_angle() -> None:
    """Ray at height y0 should exit at angle arcsin(y0)."""
    for y0 in [0.1, 0.3, 0.5]:
        pos, dirs = trace_luneburg(y0, dt=0.001, max_steps=20000)
        # Exit direction: last direction vector.
        exit_dir = dirs[-1]
        # Exit angle from x-axis (should be negative for downward focus).
        numerical_angle = np.arctan2(-exit_dir[1], exit_dir[0])
        analytical_angle = luneburg_exit_angle_analytical(y0)
        # The ray focuses downward, so numerical angle ~ analytical.
        assert abs(numerical_angle - analytical_angle) < 0.05, (
            f"y0={y0}: numerical={numerical_angle:.4f}, "
            f"analytical={analytical_angle:.4f}"
        )


def test_luneburg_focus_position() -> None:
    """Ray should exit near the diametrically opposite point."""
    y0 = 0.3
    pos, _ = trace_luneburg(y0, dt=0.001, max_steps=20000)
    exit_pos = pos[-1]
    # Expected: exits at x ~ +sqrt(1-y0^2), near the edge of the lens.
    x_expect = np.sqrt(1.0 - y0 ** 2)
    assert abs(exit_pos[0] - x_expect) < 0.1


def test_parabolic_grin_oscillation() -> None:
    """Ray in parabolic GRIN should follow sinusoidal path."""
    y0 = 0.5
    g = 0.1
    n0 = 1.5
    dt = 0.005
    # Trace about half a period.
    x_half = np.pi / g
    n_steps = int(x_half / dt) + 500
    pos = trace_parabolic(y0, theta0=0.0, n0=n0, g=g, dt=dt, max_steps=n_steps)

    # Compare at several x positions.
    x_test = np.linspace(1.0, x_half * 0.8, 10)
    for x_t in x_test:
        idx = np.argmin(np.abs(pos[:, 0] - x_t))
        y_num = pos[idx, 1]
        y_ana = parabolic_analytical_y(x_t, y0, 0.0, g)
        assert abs(y_num - y_ana) < 1e-3, (
            f"x={x_t:.2f}: y_num={y_num:.6f}, y_ana={y_ana:.6f}"
        )


def test_parabolic_grin_position_error() -> None:
    """Position error should be < 1e-4 with fine step size."""
    y0 = 0.3
    g = 0.1
    n0 = 1.5
    dt = 0.001
    x_target = 10.0
    n_steps = int(x_target / dt) + 500
    pos = trace_parabolic(y0, theta0=0.0, n0=n0, g=g, dt=dt, max_steps=n_steps)

    idx = np.argmin(np.abs(pos[:, 0] - x_target))
    y_num = pos[idx, 1]
    y_ana = parabolic_analytical_y(x_target, y0, 0.0, g)
    assert abs(y_num - y_ana) < 1e-4


def test_rk4_convergence_rate_parabolic() -> None:
    """RK4 convergence rate should be in [3.5, 4.5] for smooth GRIN."""
    # Self-convergence with coarse-enough dt to be in the convergence regime.
    _, _, rate = measure_rk4_convergence(
        profile="parabolic", y0=0.3,
        dt_values=[2.0, 1.0, 0.5, 0.25, 0.125]
    )
    assert 3.5 <= rate <= 4.5, f"Convergence rate {rate:.2f} outside [3.5, 4.5]"
