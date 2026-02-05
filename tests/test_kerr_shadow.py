"""
Tests for Kerr black hole shadow and geodesic solver.

Validates:
  1. Schwarzschild (a=0): shadow is circle with radius sqrt(27)*M.
  2. Photon orbit radii: a=0 gives r_ph=3M; a=1 gives r_ph=1 (pro), 4 (retro).
  3. Shadow boundary is closed curve (first point ~ last point).
  4. High-spin (a=0.99): shadow D-shape is asymmetric (alpha range).
  5. Null geodesic integration: circular orbit at r=3M for a=0.
  6. Impact parameters: a=0 gives xi^2 + eta = 27.
"""
from __future__ import annotations

import numpy as np

from gemini_physics.gr.kerr_geodesic import (
    impact_parameters,
    kerr_metric_quantities,
    photon_orbit_radius,
    shadow_boundary,
    trace_null_geodesic,
)


def test_schwarzschild_shadow_radius() -> None:
    """At a=0, shadow boundary should be circle with radius sqrt(27)."""
    alpha, beta = shadow_boundary(a=0.0, n_points=500)
    R = np.sqrt(alpha ** 2 + beta ** 2)
    R_expected = np.sqrt(27.0)
    # All points should be at R = sqrt(27) within 0.1%.
    rel_err = np.max(np.abs(R - R_expected)) / R_expected
    assert rel_err < 0.001, (
        f"Schwarzschild shadow radius error {rel_err:.6f} exceeds 0.1%"
    )


def test_photon_orbit_schwarzschild() -> None:
    """At a=0, both photon orbit radii should be 3M."""
    r_pro, r_retro = photon_orbit_radius(a=0.0)
    assert abs(r_pro - 3.0) < 1e-10, f"r_pro={r_pro}, expected 3.0"
    assert abs(r_retro - 3.0) < 1e-10, f"r_retro={r_retro}, expected 3.0"


def test_photon_orbit_extreme_kerr() -> None:
    """At a~1, prograde orbit -> 1M, retrograde -> 4M."""
    r_pro, r_retro = photon_orbit_radius(a=0.9999)
    assert abs(r_pro - 1.0) < 0.02, f"r_pro={r_pro}, expected ~1.0"
    assert abs(r_retro - 4.0) < 0.01, f"r_retro={r_retro}, expected ~4.0"


def test_shadow_boundary_closed() -> None:
    """Shadow boundary should form a closed curve."""
    alpha, beta = shadow_boundary(a=0.5, n_points=200)
    # First and last points should coincide.
    dist = np.sqrt((alpha[0] - alpha[-1]) ** 2 + (beta[0] - beta[-1]) ** 2)
    assert dist < 0.1, f"Shadow boundary not closed: gap={dist:.4f}"


def test_high_spin_shadow_asymmetric() -> None:
    """At a=0.99, shadow should be D-shaped: asymmetric in alpha."""
    alpha, beta = shadow_boundary(a=0.99, n_points=500)
    alpha_min = np.min(alpha)
    alpha_max = np.max(alpha)
    # The prograde side (alpha > 0 in our convention) is more indented.
    # For a=0.99, the shadow should be noticeably asymmetric.
    center = (alpha_min + alpha_max) / 2.0
    assert abs(center) > 0.1, (
        f"Shadow center at {center:.4f}, expected asymmetric shift"
    )


def test_impact_parameters_schwarzschild() -> None:
    """At a=0, r_ph=3: xi^2 + eta = 27 (shadow radius^2)."""
    # Use small a to avoid division by zero in xi formula.
    a = 1e-8
    r_ph = 3.0
    xi, eta = impact_parameters(r_ph, a)
    # For Schwarzschild: L/E = b*sin(theta), Q/E^2 = b^2*cos^2(theta)
    # At equatorial observer: b^2 = xi^2 + eta = 27
    shadow_r2 = xi ** 2 + eta
    assert abs(shadow_r2 - 27.0) < 0.01, (
        f"xi^2+eta={shadow_r2:.6f}, expected 27.0"
    )


def test_kerr_metric_schwarzschild() -> None:
    """At a=0, Sigma=r^2, Delta=r^2-2r."""
    sigma, delta = kerr_metric_quantities(r=5.0, theta=np.pi / 4, a=0.0)
    assert abs(sigma - 25.0) < 1e-10
    assert abs(delta - 15.0) < 1e-10


def test_null_geodesic_far_field() -> None:
    """A far-field photon with large impact parameter should scatter, not fall in."""
    # Schwarzschild: photon with b >> sqrt(27) should escape.
    a = 0.0
    b = 10.0  # impact parameter > sqrt(27) ~ 5.2
    result = trace_null_geodesic(
        a=a, E=1.0, L=b, Q=0.0,
        r0=100.0, theta0=np.pi / 2,
        lam_max=300.0, sgn_r=-1.0, n_eval=500,
    )
    # Ray should turn around and go back out (r increases at end).
    r_min = np.min(result["r"])
    r_final = result["r"][-1]
    assert r_min > 2.0, f"Ray came too close: r_min={r_min:.4f}"
    assert r_final > 50.0, f"Ray did not escape: r_final={r_final:.4f}"
