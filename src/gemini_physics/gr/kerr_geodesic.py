"""
Kerr black hole geodesic integrator in Boyer-Lindquist coordinates.

Integrates null geodesics (photon orbits) around a Kerr black hole
using the constants of motion: energy E, angular momentum L, and
Carter constant Q.  The equations are written in Mino time (lambda)
to regularize the coordinate singularity at the horizon.

The Kerr metric in Boyer-Lindquist coordinates (G = c = M = 1):
  ds^2 = -(1 - 2r/Sigma) dt^2 - (4ar sin^2(theta)/Sigma) dt dphi
         + (Sigma/Delta) dr^2 + Sigma dtheta^2
         + (r^2 + a^2 + 2a^2 r sin^2(theta)/Sigma) sin^2(theta) dphi^2

where:
  Sigma = r^2 + a^2 cos^2(theta)
  Delta = r^2 - 2r + a^2

Geodesic equations in Mino time (Sigma * dlambda = dtau):
  Sigma dr/dlambda = +/- sqrt(R(r))
  Sigma dtheta/dlambda = +/- sqrt(Theta(theta))
  Sigma dphi/dlambda = -(aE - L/sin^2(theta)) + a*T/Delta
  Sigma dt/dlambda = -a(aE sin^2(theta) - L) + (r^2 + a^2)*T/Delta

where:
  R(r) = ((r^2 + a^2)E - aL)^2 - Delta*(Q + (L - aE)^2)
         = T^2 - Delta * (Q + (L - aE)^2)
  Theta(theta) = Q - cos^2(theta) * (a^2(1 - E^2) + L^2/sin^2(theta))
         (simplified: for null geodesics with E=1)
  T = E(r^2 + a^2) - aL

Refs:
  Bardeen, J.M. (1973), in "Black Holes", Les Houches.
  Chandrasekhar, S. (1983), "The Mathematical Theory of Black Holes", Ch. 7.
  Teo, E. (2003), Gen. Relativ. Gravit. 35, 1909 [Kerr shadow].
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def kerr_metric_quantities(r, theta, a):
    """
    Compute Kerr metric functions Sigma, Delta.

    Parameters
    ----------
    r : float
        Boyer-Lindquist radial coordinate.
    theta : float
        Boyer-Lindquist polar angle.
    a : float
        Spin parameter (0 <= a < 1 for M=1).

    Returns
    -------
    sigma, delta : float
    """
    sigma = r ** 2 + a ** 2 * np.cos(theta) ** 2
    delta = r ** 2 - 2.0 * r + a ** 2
    return sigma, delta


def photon_orbit_radius(a):
    """
    Radii of prograde and retrograde circular photon orbits.

    For a Kerr black hole with spin a (M=1), the photon orbit radii are:
      r_ph = 2 * (1 + cos(2/3 * arccos(-/+a)))

    Parameters
    ----------
    a : float
        Spin parameter.

    Returns
    -------
    r_pro, r_retro : float
        Prograde and retrograde photon orbit radii.
    """
    r_pro = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(-a)))
    r_retro = 2.0 * (1.0 + np.cos(2.0 / 3.0 * np.arccos(a)))
    return r_pro, r_retro


def impact_parameters(r_ph, a):
    """
    Critical impact parameters (alpha, beta) for a photon orbit at radius r_ph.

    These define the boundary of the black hole shadow as seen by a
    distant observer in the equatorial plane.

    xi = L/E = (r_ph^2(r_ph - 3) + a^2(r_ph + 1)) / (a(r_ph - 1))
    eta = Q/E^2 = r_ph^3 * (-r_ph*(r_ph - 3)^2 + 4*a^2) / (a^2*(r_ph - 1)^2)

    (Algebraically equivalent to the Delta_ph form but avoids catastrophic
    cancellation when a << 1 and r_ph ~ 3.)

    For an observer at inclination theta_o:
      alpha = -xi / sin(theta_o)
      beta = +/- sqrt(eta + a^2*cos^2(theta_o) - xi^2*cot^2(theta_o))

    Parameters
    ----------
    r_ph : float or ndarray
        Photon orbit radius.
    a : float
        Spin parameter.

    Returns
    -------
    xi, eta : float or ndarray
        Reduced impact parameters (L/E and Q/E^2).
    """
    r = np.asarray(r_ph, dtype=float)
    xi = (r ** 2 * (r - 3.0) + a ** 2 * (r + 1.0)) / (a * (r - 1.0))
    eta = r ** 3 * (-r * (r - 3.0) ** 2 + 4.0 * a ** 2) / (
        a ** 2 * (r - 1.0) ** 2
    )

    return xi, eta


def shadow_boundary(a, n_points=500, theta_o=np.pi / 2):
    """
    Compute the Kerr black hole shadow boundary (Bardeen curve).

    The shadow boundary is parametrized by the photon orbit radius r_ph
    ranging from the prograde to retrograde orbit.  Each r_ph maps to
    celestial coordinates (alpha, beta) via the critical impact parameters.

    Parameters
    ----------
    a : float
        Spin parameter (0 <= a < 1).
    n_points : int
        Number of points on the boundary.
    theta_o : float
        Observer inclination (pi/2 = equatorial).

    Returns
    -------
    alpha : ndarray, shape (2*n_points,)
        Horizontal celestial coordinate.
    beta : ndarray, shape (2*n_points,)
        Vertical celestial coordinate (upper + lower halves).
    """
    if abs(a) < 1e-6:
        # Schwarzschild: circular shadow of radius sqrt(27)*M.
        angle = np.linspace(0.0, 2.0 * np.pi, 2 * n_points, endpoint=False)
        R_shadow = np.sqrt(27.0)
        return R_shadow * np.cos(angle), R_shadow * np.sin(angle)

    r_pro, r_retro = photon_orbit_radius(a)
    r_ph = np.linspace(r_pro, r_retro, n_points)

    xi, eta = impact_parameters(r_ph, a)

    sin_o = np.sin(theta_o)
    cos_o = np.cos(theta_o)

    alpha = -xi / sin_o

    beta_sq = eta + a ** 2 * cos_o ** 2 - xi ** 2 * cos_o ** 2 / sin_o ** 2
    beta_sq = np.maximum(beta_sq, 0.0)
    beta_pos = np.sqrt(beta_sq)

    # Full boundary: upper half + lower half (reversed for closed curve).
    alpha_full = np.concatenate([alpha, alpha[::-1]])
    beta_full = np.concatenate([beta_pos, -beta_pos[::-1]])

    return alpha_full, beta_full


def geodesic_rhs(lam, state, a, E, L, Q):
    """
    Right-hand side of the Kerr geodesic equations in Mino time.

    State = [t, r, theta, phi, v_r, v_theta]
    where v_r = dr/dlambda, v_theta = dtheta/dlambda (actual velocities).

    Uses the second-order form dv/dlambda = (1/2)*dV/dq, which naturally
    handles turning points without explicit sign tracking.

    Parameters
    ----------
    lam : float
        Mino time parameter.
    state : array, shape (6,)
        [t, r, theta, phi, v_r, v_theta].
    a : float
        Spin.
    E, L, Q : float
        Constants of motion.
    """
    t, r, theta, phi, v_r, v_theta = state

    _, delta = kerr_metric_quantities(r, theta, a)
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)

    # Avoid coordinate singularity at poles.
    sin2 = max(sin_th ** 2, 1e-30)
    sin3 = max(abs(sin_th) ** 3, 1e-30)

    T = E * (r ** 2 + a ** 2) - a * L

    # Coordinate time and azimuthal angle evolution.
    dphi = -(a * E - L / sin2) + a * T / delta
    dt = -a * (a * E * sin2 - L) + (r ** 2 + a ** 2) * T / delta

    # Radial acceleration: dv_r/dlambda = (1/2) dR/dr.
    # R(r) = T^2 - Delta*(Q + (L-aE)^2), so
    # dR/dr = 2*T*dT/dr - dDelta/dr*(Q + (L-aE)^2)
    #       = 4*E*r*T - (2r-2)*(Q + (L-aE)^2).
    dR_dr = 4.0 * E * r * T - (2.0 * r - 2.0) * (Q + (L - a * E) ** 2)

    # Polar acceleration: dv_theta/dlambda = (1/2) dTheta/dtheta.
    # Theta = Q - cos^2(th)*(a^2*(1-E^2) + L^2/sin^2(th)), so
    # dTheta/dtheta = sin(2th)*a^2*(1-E^2) + 2*L^2*cos(th)/sin^3(th).
    dTheta_dth = (
        np.sin(2.0 * theta) * a ** 2 * (1.0 - E ** 2)
        + 2.0 * L ** 2 * cos_th / sin3
    )

    return [dt, v_r, v_theta, dphi, 0.5 * dR_dr, 0.5 * dTheta_dth]


def trace_null_geodesic(a, E, L, Q, r0, theta0, lam_max=100.0,
                        sgn_r=-1.0, sgn_theta=1.0, n_eval=2000):
    """
    Trace a null geodesic in a Kerr spacetime.

    Parameters
    ----------
    a : float
        Spin parameter.
    E, L, Q : float
        Constants of motion (energy, angular momentum, Carter constant).
    r0 : float
        Initial radial coordinate.
    theta0 : float
        Initial polar angle.
    lam_max : float
        Maximum Mino time for integration.
    sgn_r : float
        Initial sign of dr/dlambda (+1 outgoing, -1 ingoing).
    sgn_theta : float
        Initial sign of dtheta/dlambda.
    n_eval : int
        Number of output points.

    Returns
    -------
    result : dict with keys 't', 'r', 'theta', 'phi', 'lam'.
    """
    # Compute initial radial and polar velocities from the potentials.
    _, delta0 = kerr_metric_quantities(r0, theta0, a)
    T0 = E * (r0 ** 2 + a ** 2) - a * L
    R0 = T0 ** 2 - delta0 * (Q + (L - a * E) ** 2)
    sin_th0 = np.sin(theta0)
    cos_th0 = np.cos(theta0)
    sin2_0 = max(sin_th0 ** 2, 1e-30)
    Theta0 = Q - cos_th0 ** 2 * (a ** 2 * (1.0 - E ** 2) + L ** 2 / sin2_0)

    v_r0 = sgn_r * np.sqrt(max(R0, 0.0))
    v_theta0 = sgn_theta * np.sqrt(max(Theta0, 0.0))

    state0 = [0.0, r0, theta0, 0.0, v_r0, v_theta0]

    r_horizon = 1.0 + np.sqrt(1.0 - a ** 2)

    def hit_horizon(lam, state, a, E, L, Q):
        return state[1] - (r_horizon + 0.01)
    hit_horizon.terminal = True
    hit_horizon.direction = -1

    def escaped(lam, state, a, E, L, Q):
        return state[1] - 5.0 * r0
    escaped.terminal = True
    escaped.direction = 1

    lam_eval = np.linspace(0, lam_max, n_eval)
    sol = solve_ivp(
        geodesic_rhs, [0.0, lam_max], state0,
        args=(a, E, L, Q),
        t_eval=lam_eval,
        events=[hit_horizon, escaped],
        rtol=1e-10, atol=1e-12,
        method="RK45",
        max_step=lam_max / 500,
    )

    return {
        "t": sol.y[0],
        "r": sol.y[1],
        "theta": sol.y[2],
        "phi": sol.y[3],
        "lam": sol.t,
    }


def shadow_ray_traced(a, r_obs=1000.0, theta_obs=np.pi / 2,
                      n_alpha=200, n_beta=200,
                      alpha_range=(-8, 8), beta_range=(-8, 8)):
    """
    Ray-trace the black hole shadow by backward-tracing photons.

    For each pixel (alpha, beta) in the observer's sky, launch a null
    geodesic backward and check if it falls into the horizon.

    Parameters
    ----------
    a : float
        Spin parameter.
    r_obs : float
        Observer distance (large = far-field).
    theta_obs : float
        Observer inclination.
    n_alpha, n_beta : int
        Grid resolution.
    alpha_range, beta_range : tuple
        Celestial coordinate ranges.

    Returns
    -------
    alpha_grid, beta_grid : ndarray, shape (n_alpha,), (n_beta,)
    shadow_mask : ndarray, shape (n_alpha, n_beta), bool
        True where the photon falls into the black hole.
    """
    alpha_arr = np.linspace(alpha_range[0], alpha_range[1], n_alpha)
    beta_arr = np.linspace(beta_range[0], beta_range[1], n_beta)

    sin_o = np.sin(theta_obs)
    cos_o = np.cos(theta_obs)

    shadow_mask = np.zeros((n_alpha, n_beta), dtype=bool)

    for i, alpha in enumerate(alpha_arr):
        for j, beta in enumerate(beta_arr):
            # Map (alpha, beta) to (L, Q) at the observer.
            # For a distant observer:
            #   L = -alpha * sin(theta_obs)
            #   Q = beta^2 + cos^2(theta_obs) * (alpha^2 - a^2)
            L_val = -alpha * sin_o
            Q_val = beta ** 2 + cos_o ** 2 * (alpha ** 2 - a ** 2)
            E_val = 1.0

            if Q_val < 0:
                continue

            result = trace_null_geodesic(
                a, E_val, L_val, Q_val,
                r0=r_obs, theta0=theta_obs,
                lam_max=2.0 * r_obs,
                sgn_r=-1.0,
                n_eval=100,
            )

            # If ray hit the horizon (terminated early), it's in the shadow.
            r_horizon = 1.0 + np.sqrt(1.0 - a ** 2)
            if result["r"][-1] < r_horizon + 0.1:
                shadow_mask[i, j] = True

    return alpha_arr, beta_arr, shadow_mask
