import math

import numpy as np

from gemini_physics.dimensional_geometry import ball_volume, unit_sphere_surface_area


def test_ball_volume_known_integers() -> None:
    r = 1.0
    assert np.isclose(ball_volume(0, r=r), 1.0)
    assert np.isclose(ball_volume(1, r=r), 2.0)
    assert np.isclose(ball_volume(2, r=r), math.pi)
    assert np.isclose(ball_volume(3, r=r), 4.0 * math.pi / 3.0)


def test_unit_sphere_surface_area_known_integers() -> None:
    assert np.isclose(unit_sphere_surface_area(1), 2.0)  # S^0 has 2 points
    assert np.isclose(unit_sphere_surface_area(2), 2.0 * math.pi)  # circle
    assert np.isclose(unit_sphere_surface_area(3), 4.0 * math.pi)  # sphere


def test_area_volume_relation() -> None:
    # S_{d-1} = d * V_d(1) for analytic continuation (away from poles).
    d = 2.5
    lhs = unit_sphere_surface_area(d)
    rhs = d * ball_volume(d, r=1.0)
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)

