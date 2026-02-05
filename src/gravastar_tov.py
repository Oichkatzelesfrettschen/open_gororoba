"""
Gravastar TOV Solver (Mazur-Mottola 2001/2004 three-layer model).

Solves the Tolman-Oppenheimer-Volkoff equation for hydrostatic equilibrium
in each layer, matching boundary conditions at layer transitions.

Three-layer structure (geometrized units G = c = 1):
  I.   Interior (0 < r < R1):  de Sitter vacuum, p = -rho_v (constant).
  II.  Shell (R1 < r < R2):    polytropic matter, p = K * rho^gamma.
  III. Exterior (r > R2):      Schwarzschild vacuum, p = rho = 0.

The shell outer radius R2 is an eigenvalue determined by the TOV integration:
it is the coordinate where the shell pressure drops to zero.  This avoids
over-constraining the model by independently fixing R1, R2, and shell density.

Stability analysis:
  The Harrison-Wheeler criterion (dM/d(rho_c) > 0) identifies stable branches.
  For stiff matter (gamma=1), all solutions lie on an unstable branch.
  Polytropic shells with gamma >= 4/3 can yield stable configurations.
  Anisotropic pressure (p_t != p_r) further extends the stable domain.

References:
  Mazur & Mottola (2001), gr-qc/0109035.
  Mazur & Mottola (2004), PNAS 101, 9545-9550.
  Visser & Wiltshire (2004), gr-qc/0310107.
  Cattoen, Faber & Visser (2005), arXiv:0707.1636.
  Das, Debnath & Ray (2024), Polytropic thin-shell gravastar models.
  Bowers & Liang (1974), ApJ 188, 657 -- Anisotropic stars.
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def tov_rhs(r, y, rho_func):
    """
    Right-hand side of the TOV system:
      dp/dr = -(rho + p)(m + 4*pi*r^3*p) / (r*(r - 2*m))
      dm/dr = 4*pi*r^2*rho

    Parameters
    ----------
    r : float
        Radial coordinate (geometrized units).
    y : array_like
        State vector [p, m] -- pressure and enclosed mass.
    rho_func : callable
        rho(p) -- equation of state giving density from pressure.
    """
    p, m = y

    if r < 1e-30:
        # At the origin, dm/dr = 0 and dp/dr = 0 by L'Hopital.
        return np.array([0.0, 0.0])

    rho = rho_func(p)

    # Check for surface (p <= 0 in non-vacuum regions)
    if p <= 0 and rho >= 0:
        return np.array([0.0, 0.0])

    denom = r * (r - 2.0 * m)
    if abs(denom) < 1e-30:
        # Near horizon; integration should stop before this.
        return np.array([0.0, 0.0])

    dpdr = -(rho + p) * (m + 4.0 * np.pi * r ** 3 * p) / denom
    dmdr = 4.0 * np.pi * r ** 2 * rho

    return np.array([dpdr, dmdr])


def solve_gravastar(rho_v, R1, rho_shell, n_points=2000, gamma=1.0, K=1.0):
    """
    Solve the gravastar structure with three layers (polytropic shell EoS).

    The shell outer radius R2 is found by integrating the TOV equation
    outward from R1 until the shell pressure drops to zero (surface event).

    Parameters
    ----------
    rho_v : float
        Vacuum energy density in the de Sitter interior (geometrized).
    R1 : float
        Inner radius of the shell (end of de Sitter core).
    rho_shell : float
        Initial shell density at r = R1.
    n_points : int
        Number of radial grid points for output.
    gamma : float
        Polytropic exponent for the shell EoS: p = K * rho^gamma.
        gamma = 1.0 recovers the stiff EoS (p = rho when K = 1).
        gamma >= 4/3 is required for stable configurations in isotropic case.
        Default: 1.0 (backward-compatible stiff EoS).
    K : float
        Polytropic constant. For gamma = 1 and K = 1, recovers p = rho.
        Default: 1.0.

    Returns
    -------
    result : dict with keys:
        'r' : radial coordinate array
        'p' : pressure profile
        'rho' : density profile
        'm' : enclosed mass profile
        'R1' : inner shell radius
        'R2' : outer shell radius (eigenvalue from TOV)
        'M_total' : total gravitational mass
        'equilibrium_satisfied' : bool
        'shell_solution' : OdeResult from solve_ivp
        'gamma' : polytropic exponent used
        'K' : polytropic constant used

    Raises
    ------
    ValueError
        If the core exceeds the horizon condition (2*m_core/R1 >= 1).
    """
    # --- Layer I: de Sitter interior (0 < r < R1) ---
    # p = -rho_v (constant).  m(r) = (4/3)*pi*r^3*rho_v.
    # TOV is trivially satisfied: dp/dr = 0 because
    # (rho + p) = (rho_v + (-rho_v)) = 0.
    m_at_R1 = (4.0 / 3.0) * np.pi * R1 ** 3 * rho_v

    # Subcritical compactness check: 2m/R1 must be < 1.
    compactness_R1 = 2.0 * m_at_R1 / R1
    if compactness_R1 >= 1.0:
        raise ValueError(
            f"Core exceeds horizon condition: 2m/R1 = {compactness_R1:.6f}. "
            f"Reduce rho_v or increase R1."
        )

    # --- Layer II: Polytropic shell (R1 < r < R2) ---
    # EoS: p = K * rho^gamma (polytropic matter).
    # For gamma = 1, K = 1: reduces to stiff EoS p = rho.
    # The junction at R1 involves a pressure discontinuity
    # from p = -rho_v (interior) to p_shell (shell).
    # Per Visser & Wiltshire (2004), the thin-shell formalism
    # uses Israel junction conditions.  The shell pressure is
    # determined by the surface tension at the phase boundary.
    p_shell_start = K * rho_shell ** gamma  # polytropic EoS

    def rho_from_p_shell(p):
        """Polytropic EoS inversion: rho = (p / K)^(1/gamma) for p > 0."""
        if p > 0:
            return (p / K) ** (1.0 / gamma)
        return 0.0

    def shell_rhs(r, y):
        return tov_rhs(r, y, rho_from_p_shell)

    # Event: pressure drops to zero (defines the shell surface R2).
    def pressure_zero(r, y):
        return y[0]
    pressure_zero.terminal = True
    pressure_zero.direction = -1

    # Practical surface: pressure decays below 1e-3 of initial value.
    # The stiff EoS produces a power-law tail; this gives a clean cutoff
    # at the point where 99.9% of the shell pressure has been dissipated.
    p_floor = 1e-3 * p_shell_start

    def pressure_negligible(r, y):
        return y[0] - p_floor
    pressure_negligible.terminal = True
    pressure_negligible.direction = -1

    # Safety: terminate if approaching a horizon inside the shell.
    def horizon_approach(r, y):
        return r - 2.002 * y[1]
    horizon_approach.terminal = True
    horizon_approach.direction = -1

    R_max = 10.0 * R1

    sol_shell = solve_ivp(
        shell_rhs,
        t_span=(R1, R_max),
        y0=[p_shell_start, m_at_R1],
        method='RK45',
        events=[pressure_zero, pressure_negligible, horizon_approach],
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01 * R1,
        dense_output=True,
    )

    # Determine R2: prefer exact p=0, then practical surface, then R_max.
    if sol_shell.t_events[0].size > 0:
        R2 = float(sol_shell.t_events[0][0])
    elif sol_shell.t_events[1].size > 0:
        R2 = float(sol_shell.t_events[1][0])
    else:
        R2 = float(sol_shell.t[-1])

    M_total = float(sol_shell.sol(R2)[1]) if sol_shell.sol else float(
        sol_shell.y[1, -1])

    # Re-evaluate shell on a uniform grid via dense output.
    n_shell = max(200, n_points // 3)
    r_shell_uniform = np.linspace(R1, R2, n_shell)
    if sol_shell.sol:
        shell_vals = sol_shell.sol(r_shell_uniform)
        p_shell = shell_vals[0]
        m_shell = shell_vals[1]
    else:
        r_shell_uniform = sol_shell.t
        p_shell = sol_shell.y[0]
        m_shell = sol_shell.y[1]

    # Clip small negative pressures from interpolation artifacts.
    p_shell = np.maximum(p_shell, 0.0)

    # --- Layer III: Exterior vacuum (r > R2) ---
    # p = rho = 0.  m = M_total = const.

    # --- Assemble full profile ---
    r_int = np.linspace(1e-6, R1, n_points // 3)
    r_ext = np.linspace(R2, R2 * 3.0, n_points // 3)

    r_full = np.concatenate([r_int, r_shell_uniform, r_ext])
    p_full = np.concatenate([
        np.full_like(r_int, -rho_v),       # interior: p = -rho_v
        p_shell,                            # shell: from integration
        np.zeros_like(r_ext),               # exterior: p = 0
    ])
    m_full = np.concatenate([
        (4.0 / 3.0) * np.pi * r_int ** 3 * rho_v,  # interior
        m_shell,                                      # shell
        np.full_like(r_ext, M_total),                 # exterior
    ])
    # For polytropic EoS: rho = (p / K)^(1/gamma)
    rho_shell_arr = np.where(
        p_shell > 0,
        (p_shell / K) ** (1.0 / gamma),
        0.0
    )
    rho_full = np.concatenate([
        np.full_like(r_int, rho_v),         # interior: rho = rho_v
        rho_shell_arr,                       # shell: polytropic
        np.zeros_like(r_ext),               # exterior: rho = 0
    ])

    # --- Verify hydrostatic equilibrium in the shell ---
    # Compare numerical dp/dr (central differences) against the TOV formula.
    equilibrium_ok = True
    if len(r_shell_uniform) >= 5:
        r_s = r_shell_uniform
        p_s = p_shell
        m_s = m_shell

        dpdr_num = np.gradient(p_s, r_s)

        dpdr_tov = np.zeros_like(r_s)
        for i in range(len(r_s)):
            ri = r_s[i]
            if ri > 1e-30 and abs(ri - 2.0 * m_s[i]) > 1e-30 and p_s[i] > 0:
                rho_i = (p_s[i] / K) ** (1.0 / gamma)  # polytropic EoS
                dpdr_tov[i] = (-(rho_i + p_s[i]) *
                               (m_s[i] + 4.0 * np.pi * ri ** 3 * p_s[i]) /
                               (ri * (ri - 2.0 * m_s[i])))

        # Skip boundary points where finite-difference gradient is poor.
        margin = 0.05 * (R2 - R1)
        mask = (p_s > 1e-10) & (r_s > R1 + margin) & (r_s < R2 - margin)
        if np.any(mask):
            rel_err = np.abs(dpdr_num[mask] - dpdr_tov[mask])
            max_err = np.max(rel_err / (np.abs(dpdr_tov[mask]) + 1e-30))
            equilibrium_ok = max_err < 0.01  # 1% tolerance

    return {
        'r': r_full,
        'p': p_full,
        'rho': rho_full,
        'm': m_full,
        'R1': R1,
        'R2': R2,
        'M_total': M_total,
        'equilibrium_satisfied': equilibrium_ok,
        'shell_solution': sol_shell,
        'gamma': gamma,
        'K': K,
    }


def solve_gravastar_for_mass(m_target=35.0, core_compactness=0.7,
                              shell_density_factor=3.0, gamma=1.0, K=1.0):
    """
    Convenience wrapper: construct a gravastar with target total mass.

    Uses Brent's method to find the core mass fraction that yields the
    desired M_total after TOV integration through the shell.

    Parameters
    ----------
    m_target : float
        Target total gravitational mass in geometrized units.
    core_compactness : float
        Compactness of the de Sitter core: C_core = 2*m_core/R1.
        Must be in (0, 1).  Default 0.7 balances realism and stability.
    shell_density_factor : float
        Initial shell density as multiple of the vacuum energy density.
        Higher values give thinner, denser shells.
    gamma : float
        Polytropic exponent for the shell EoS. Default 1.0.
    K : float
        Polytropic constant. Default 1.0.

    Returns
    -------
    result : dict (same as solve_gravastar)
    """
    def _build(m_core):
        """Construct a gravastar from core mass; return total mass."""
        R1 = 2.0 * m_core / core_compactness
        rho_v = 3.0 * m_core / (4.0 * np.pi * R1 ** 3)
        rho_shell = shell_density_factor * rho_v
        try:
            res = solve_gravastar(rho_v, R1, rho_shell, n_points=500,
                                  gamma=gamma, K=K)
            return res['M_total']
        except ValueError:
            return np.inf

    # Bracket: total mass is monotone increasing with core mass.
    lo = 0.3 * m_target
    hi = 0.99 * m_target

    M_lo = _build(lo)
    M_hi = _build(hi)

    # If the bracket doesn't straddle m_target, widen it.
    if M_lo > m_target:
        lo = 0.1 * m_target
    if M_hi < m_target:
        hi = m_target

    def residual(m_core):
        return _build(m_core) - m_target

    try:
        m_core_opt = brentq(residual, lo, hi, xtol=1e-6 * m_target,
                            maxiter=50)
    except ValueError:
        # Fallback: use hi (closest achievable mass).
        m_core_opt = hi

    # Final solve at full resolution.
    R1 = 2.0 * m_core_opt / core_compactness
    rho_v = 3.0 * m_core_opt / (4.0 * np.pi * R1 ** 3)
    rho_shell = shell_density_factor * rho_v

    return solve_gravastar(rho_v, R1, rho_shell, gamma=gamma, K=K)


def tov_anisotropic_rhs(r, y, rho_func, lambda_aniso):
    """
    Anisotropic TOV equation with Bowers-Liang pressure anisotropy.

    The generalized TOV with anisotropy (sigma = p_t - p_r):
      dp_r/dr = -(rho + p_r)(m + 4*pi*r^3*p_r) / (r*(r - 2*m)) + 2*sigma/r
      dm/dr = 4*pi*r^2*rho

    With Bowers-Liang parameterization: sigma = lambda_aniso * rho * p_r

    Parameters
    ----------
    r : float
        Radial coordinate.
    y : array_like
        State vector [p_r, m].
    rho_func : callable
        rho(p_r) -- equation of state.
    lambda_aniso : float
        Anisotropy parameter. lambda_aniso = 0 recovers isotropic TOV.
        Positive values give tangential pressure excess (p_t > p_r).

    Returns
    -------
    dydr : ndarray
        Derivatives [dp_r/dr, dm/dr].
    """
    p_r, m = y

    if r < 1e-30:
        return np.array([0.0, 0.0])

    rho = rho_func(p_r)

    if p_r <= 0 and rho >= 0:
        return np.array([0.0, 0.0])

    denom = r * (r - 2.0 * m)
    if abs(denom) < 1e-30:
        return np.array([0.0, 0.0])

    # Bowers-Liang anisotropy: sigma = lambda * rho * p_r
    sigma = lambda_aniso * rho * p_r

    # Anisotropic TOV: standard term + 2*sigma/r
    dpdr = (-(rho + p_r) * (m + 4.0 * np.pi * r ** 3 * p_r) / denom
            + 2.0 * sigma / r)
    dmdr = 4.0 * np.pi * r ** 2 * rho

    return np.array([dpdr, dmdr])


def solve_gravastar_anisotropic(rho_v, R1, rho_shell, n_points=2000,
                                 gamma=1.0, K=1.0, lambda_aniso=0.0):
    """
    Solve gravastar structure with anisotropic pressure in the shell.

    Anisotropy enables stable configurations at lower gamma values than
    the isotropic case (Cattoen, Faber & Visser 2005).

    Parameters
    ----------
    rho_v : float
        Vacuum energy density in the de Sitter interior.
    R1 : float
        Inner radius of the shell.
    rho_shell : float
        Initial shell density at r = R1.
    n_points : int
        Number of radial grid points.
    gamma : float
        Polytropic exponent (p = K * rho^gamma).
    K : float
        Polytropic constant.
    lambda_aniso : float
        Bowers-Liang anisotropy parameter. lambda_aniso = 0 gives isotropic.
        Positive values increase tangential pressure (stabilizing effect).

    Returns
    -------
    result : dict
        Same as solve_gravastar, plus 'lambda_aniso' and 'p_tangential'.
    """
    # --- Layer I: de Sitter interior (identical to isotropic) ---
    m_at_R1 = (4.0 / 3.0) * np.pi * R1 ** 3 * rho_v

    compactness_R1 = 2.0 * m_at_R1 / R1
    if compactness_R1 >= 1.0:
        raise ValueError(
            f"Core exceeds horizon: 2m/R1 = {compactness_R1:.6f}."
        )

    # --- Layer II: Anisotropic polytropic shell ---
    p_shell_start = K * rho_shell ** gamma

    def rho_from_p_shell(p):
        if p > 0:
            return (p / K) ** (1.0 / gamma)
        return 0.0

    def shell_rhs(r, y):
        return tov_anisotropic_rhs(r, y, rho_from_p_shell, lambda_aniso)

    def pressure_zero(r, y):
        return y[0]
    pressure_zero.terminal = True
    pressure_zero.direction = -1

    p_floor = 1e-3 * p_shell_start

    def pressure_negligible(r, y):
        return y[0] - p_floor
    pressure_negligible.terminal = True
    pressure_negligible.direction = -1

    def horizon_approach(r, y):
        return r - 2.002 * y[1]
    horizon_approach.terminal = True
    horizon_approach.direction = -1

    R_max = 10.0 * R1

    sol_shell = solve_ivp(
        shell_rhs,
        t_span=(R1, R_max),
        y0=[p_shell_start, m_at_R1],
        method='RK45',
        events=[pressure_zero, pressure_negligible, horizon_approach],
        rtol=1e-10,
        atol=1e-12,
        max_step=0.01 * R1,
        dense_output=True,
    )

    if sol_shell.t_events[0].size > 0:
        R2 = float(sol_shell.t_events[0][0])
    elif sol_shell.t_events[1].size > 0:
        R2 = float(sol_shell.t_events[1][0])
    else:
        R2 = float(sol_shell.t[-1])

    M_total = float(sol_shell.sol(R2)[1]) if sol_shell.sol else float(
        sol_shell.y[1, -1])

    n_shell = max(200, n_points // 3)
    r_shell_uniform = np.linspace(R1, R2, n_shell)
    if sol_shell.sol:
        shell_vals = sol_shell.sol(r_shell_uniform)
        p_shell = shell_vals[0]
        m_shell = shell_vals[1]
    else:
        r_shell_uniform = sol_shell.t
        p_shell = sol_shell.y[0]
        m_shell = sol_shell.y[1]

    p_shell = np.maximum(p_shell, 0.0)

    # Compute tangential pressure: p_t = p_r + sigma = p_r + lambda*rho*p_r
    rho_shell_arr = np.where(
        p_shell > 0,
        (p_shell / K) ** (1.0 / gamma),
        0.0
    )
    sigma_shell = lambda_aniso * rho_shell_arr * p_shell
    p_tangential_shell = p_shell + sigma_shell

    # --- Assemble profiles ---
    r_int = np.linspace(1e-6, R1, n_points // 3)
    r_ext = np.linspace(R2, R2 * 3.0, n_points // 3)

    r_full = np.concatenate([r_int, r_shell_uniform, r_ext])
    p_full = np.concatenate([
        np.full_like(r_int, -rho_v),
        p_shell,
        np.zeros_like(r_ext),
    ])
    m_full = np.concatenate([
        (4.0 / 3.0) * np.pi * r_int ** 3 * rho_v,
        m_shell,
        np.full_like(r_ext, M_total),
    ])
    rho_full = np.concatenate([
        np.full_like(r_int, rho_v),
        rho_shell_arr,
        np.zeros_like(r_ext),
    ])
    # Tangential pressure profile
    p_t_full = np.concatenate([
        np.full_like(r_int, -rho_v),  # isotropic in interior
        p_tangential_shell,
        np.zeros_like(r_ext),
    ])

    # Verify equilibrium
    equilibrium_ok = True
    if len(r_shell_uniform) >= 5:
        r_s = r_shell_uniform
        p_s = p_shell
        m_s = m_shell
        dpdr_num = np.gradient(p_s, r_s)
        dpdr_tov = np.zeros_like(r_s)
        for i in range(len(r_s)):
            ri = r_s[i]
            if ri > 1e-30 and abs(ri - 2.0 * m_s[i]) > 1e-30 and p_s[i] > 0:
                rho_i = (p_s[i] / K) ** (1.0 / gamma)
                sigma_i = lambda_aniso * rho_i * p_s[i]
                denom = ri * (ri - 2.0 * m_s[i])
                dpdr_tov[i] = (-(rho_i + p_s[i]) *
                               (m_s[i] + 4.0 * np.pi * ri ** 3 * p_s[i]) /
                               denom + 2.0 * sigma_i / ri)
        margin = 0.05 * (R2 - R1)
        mask = (p_s > 1e-10) & (r_s > R1 + margin) & (r_s < R2 - margin)
        if np.any(mask):
            rel_err = np.abs(dpdr_num[mask] - dpdr_tov[mask])
            max_err = np.max(rel_err / (np.abs(dpdr_tov[mask]) + 1e-30))
            equilibrium_ok = max_err < 0.01

    return {
        'r': r_full,
        'p': p_full,
        'p_tangential': p_t_full,
        'rho': rho_full,
        'm': m_full,
        'R1': R1,
        'R2': R2,
        'M_total': M_total,
        'equilibrium_satisfied': equilibrium_ok,
        'shell_solution': sol_shell,
        'gamma': gamma,
        'K': K,
        'lambda_aniso': lambda_aniso,
    }


if __name__ == "__main__":
    result = solve_gravastar_for_mass(m_target=35.0)
    print("Gravastar TOV Solution (isotropic, stiff EoS):")
    print(f"  R1 (core boundary) = {result['R1']:.4f}")
    print(f"  R2 (shell boundary) = {result['R2']:.4f}")
    print(f"  M_total = {result['M_total']:.6f}")
    print(f"  Equilibrium satisfied: {result['equilibrium_satisfied']}")
    print(f"  Shell thickness = {result['R2'] - result['R1']:.4f}")
    n_shell = result['shell_solution'].t.shape[0]
    print(f"  Shell integration points: {n_shell}")
    compactness = 2.0 * result['M_total'] / result['R2']
    print(f"  Surface compactness 2M/R2 = {compactness:.6f}")

    # Demo polytropic
    print("\nPolytropic (gamma=1.5) solution:")
    result_poly = solve_gravastar_for_mass(m_target=35.0, gamma=1.5)
    print(f"  M_total = {result_poly['M_total']:.6f}")
    print(f"  gamma = {result_poly['gamma']}")

    # Demo anisotropic
    print("\nAnisotropic (lambda=0.5, gamma=1.2) solution:")
    result_aniso = solve_gravastar_anisotropic(
        rho_v=0.01, R1=50.0, rho_shell=0.03,
        gamma=1.2, lambda_aniso=0.5
    )
    print(f"  M_total = {result_aniso['M_total']:.6f}")
    print(f"  lambda_aniso = {result_aniso['lambda_aniso']}")
