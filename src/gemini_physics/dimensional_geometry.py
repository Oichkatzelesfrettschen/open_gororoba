from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import special


@dataclass(frozen=True)
class DimensionalQuantity:
    d: float
    value: complex


def unit_sphere_surface_area(d: float) -> complex:
    """
    Surface area of the unit (d-1)-sphere embedded in R^d, analytically continued in d.

    Formula:
        S_{d-1} = 2 * pi^(d/2) / Gamma(d/2)

    Notes:
    - For d <= 0 and at gamma poles, the result is not finite (meromorphic continuation).
    - For integer d >= 1 this agrees with the standard geometric quantity.
    """
    z = d / 2.0
    return 2.0 * (math.pi**z) / complex(special.gamma(z))


def ball_volume(d: float, r: float = 1.0) -> complex:
    """
    Volume of a d-dimensional ball of radius r, analytically continued in d.

    Formula:
        V_d(r) = pi^(d/2) / Gamma(d/2 + 1) * r^d
    """
    z = d / 2.0
    return (math.pi**z) / complex(special.gamma(z + 1.0)) * (r**d)


def sample_dimensional_range(
    d_min: float, d_max: float, *, n: int = 2001, r: float = 1.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns (d, V_d(r), S_{d-1}) for a uniform grid of d in [d_min, d_max].
    """
    ds = np.linspace(d_min, d_max, n, dtype=float)
    vols = np.array([ball_volume(float(d), r=r) for d in ds], dtype=complex)
    areas = np.array([unit_sphere_surface_area(float(d)) for d in ds], dtype=complex)
    return ds, vols, areas

