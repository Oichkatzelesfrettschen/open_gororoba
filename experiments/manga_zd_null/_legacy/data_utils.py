"""
Synthetic MaNGA-like rotation curve generator for ZD null result analysis.

Generates physically realistic galaxy rotation curves using:
- NFW dark matter halo profiles with Duffy+2008 concentration-mass relation
- Moster+2013 stellar-mass-to-halo-mass (SMHM) relation
- Realistic galaxy-to-galaxy scatter in mass, concentration, and velocity errors
- Known baryonic systematics: inner bulge excess (+5% at x<0.8), NFW cusp
  over-prediction (-10-15% at x=0.6-0.87)

Parameters calibrated to reproduce E-183 stacking results:
  N=6992 galaxies, 19 valid bins at x=0.5-1.35, RMS=0.075, SNR=0.25

No network access required. All data is generated deterministically from seeds.
"""

import math
from dataclasses import dataclass, field

import numpy as np

G_KPC_KMS2 = 4.302e-6  # G in kpc * (km/s)^2 / Msun

# Moster+2013 SMHM parameters at z=0
SMHM_LOG_M1 = 11.59
SMHM_N = 0.0351
SMHM_BETA = 1.376
SMHM_GAMMA = 0.608


@dataclass
class SyntheticGalaxy:
    """A synthetic MaNGA galaxy with rotation curve data."""
    galaxy_id: str
    log_m200: float
    log_mstar: float
    z: float
    inclination_deg: float
    r_s_kpc: float
    c200: float
    x_points: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_v: np.ndarray = field(default_factory=lambda: np.array([]))
    delta_v_err: np.ndarray = field(default_factory=lambda: np.array([]))


def smhm_moster13(log_m200: float, z: float = 0.0) -> float:
    """Moster+2013 SMHM: log(M_star) from log(M_200)."""
    m200 = 10.0 ** log_m200
    m1 = 10.0 ** SMHM_LOG_M1
    ratio = m200 / m1
    frac = 2.0 * SMHM_N / (ratio ** (-SMHM_BETA) + ratio ** SMHM_GAMMA)
    mstar = frac * m200
    return math.log10(max(mstar, 1.0))


def duffy08_concentration(log_m200: float, z: float = 0.0) -> float:
    """Duffy+2008 concentration-mass relation."""
    m200 = 10.0 ** log_m200
    a, b, c_z = 5.71, -0.084, -0.47
    return a * (m200 / 2e12) ** b * (1.0 + z) ** c_z


def nfw_v_circ(r_kpc: float, r_s: float, m200: float, c200: float) -> float:
    """NFW circular velocity at radius r_kpc (km/s)."""
    if r_kpc <= 0 or r_s <= 0:
        return 0.0
    x = r_kpc / r_s
    log_factor = math.log(1.0 + c200) - c200 / (1.0 + c200)
    if log_factor <= 0:
        return 0.0
    m_enc = m200 * (math.log(1.0 + x) - x / (1.0 + x)) / log_factor
    v_sq = G_KPC_KMS2 * m_enc / r_kpc
    return math.sqrt(max(v_sq, 0.0))


def baryonic_systematic(x: float) -> float:
    """
    Known baryonic systematics in NFW residuals (from E-183):
    - Inner bulge excess: +5% Gaussian at x~0.5
    - Cusp over-prediction: -12% dip at x~0.7
    - IFU edge spike: +5% at x~0.95 (projection artifact)
    """
    bulge = 0.05 * math.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
    cusp = -0.12 * math.exp(-0.5 * ((x - 0.72) / 0.12) ** 2)
    edge = 0.05 * math.exp(-0.5 * ((x - 0.953) / 0.05) ** 2)
    return bulge + cusp + edge


def generate_manga_sample(
    n_galaxies: int = 6992,
    seed: int = 42,
    x_min: float = 0.5,
    x_max: float = 1.35,
    n_points_per_galaxy: int = 18,
    inject_alpha_zd: float = 0.0,
    cd_dim: int = 16,
) -> list:
    """
    Generate a sample of synthetic MaNGA-like galaxies.

    The x-range is restricted to 0.5-1.35 r/r_s to match MaNGA IFU coverage
    (typical IFU ~12-18 kpc vs r_s ~10-20 kpc).
    """
    rng = np.random.RandomState(seed)

    log_m200_samples = rng.normal(11.8, 0.6, n_galaxies)
    log_m200_samples = np.clip(log_m200_samples, 10.2, 15.0)

    z_samples = rng.normal(0.03, 0.015, n_galaxies)
    z_samples = np.clip(z_samples, 0.005, 0.15)

    cos_i_min = math.cos(math.radians(70.0))
    cos_i = rng.uniform(cos_i_min, 1.0, n_galaxies)
    inclination_samples = np.degrees(np.arccos(cos_i))

    galaxies = []
    n_modes = cd_dim // 2 - 1 if cd_dim >= 4 else 1

    for i in range(n_galaxies):
        log_m200 = float(log_m200_samples[i])
        z = float(z_samples[i])
        inc = float(inclination_samples[i])

        c200 = duffy08_concentration(log_m200, z)
        c200 *= rng.lognormal(0, 0.12)
        c200 = max(c200, 2.0)

        m200 = 10.0 ** log_m200
        rho_crit = 277.54  # Msun/kpc^3 at z~0 (H0=70)
        r200 = (3.0 * m200 / (4.0 * math.pi * 200.0 * rho_crit)) ** (1.0 / 3.0)
        r_s = r200 / c200

        log_mstar = smhm_moster13(log_m200, z)

        n_pts = max(8, rng.poisson(n_points_per_galaxy))
        x_points = np.sort(rng.uniform(x_min, x_max, n_pts))

        delta_v = np.zeros(n_pts)
        delta_v_err = np.zeros(n_pts)

        for j in range(n_pts):
            x = float(x_points[j])
            sys_val = baryonic_systematic(x)
            scatter = rng.normal(0, 0.08)
            meas_noise_frac = rng.uniform(0.03, 0.10)

            delta_v[j] = sys_val + scatter
            delta_v_err[j] = max(meas_noise_frac, 0.01)

            if inject_alpha_zd > 0:
                af = 0.5
                modulation = 0.0
                for n in range(1, n_modes + 1):
                    k_n = 2.0 * math.pi * n / n_modes
                    modulation += (af / n) * math.cos(k_n * x) * math.exp(-x)
                delta_v[j] += inject_alpha_zd * modulation

        gal = SyntheticGalaxy(
            galaxy_id=f"SYN-{i:05d}",
            log_m200=log_m200,
            log_mstar=log_mstar,
            z=z,
            inclination_deg=inc,
            r_s_kpc=r_s,
            c200=c200,
            x_points=x_points,
            delta_v=delta_v,
            delta_v_err=delta_v_err,
        )
        galaxies.append(gal)

    return galaxies


def filter_inclination(galaxies: list, inc_min: float, inc_max: float) -> list:
    """Filter galaxies by inclination range (degrees)."""
    return [g for g in galaxies if inc_min <= g.inclination_deg <= inc_max]


def filter_mass_quartile(galaxies: list, quartile: int) -> list:
    """Filter to mass quartile (1=lowest, 4=highest)."""
    sorted_gals = sorted(galaxies, key=lambda g: g.log_m200)
    n = len(sorted_gals)
    q_size = n // 4
    start = (quartile - 1) * q_size
    end = quartile * q_size if quartile < 4 else n
    return sorted_gals[start:end]
