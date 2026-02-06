"""
Quantum Cosmology with Regularized Bounce.

The Bohmian quantum potential Q ~ l_P^4 / a^6 (from the Wheeler-DeWitt
equation for a dust-filled universe) gives a repulsive force F_q ~ a^{-7}
in the Raychaudhuri equation, preventing the classical Big Bang singularity
and producing a "Big Bounce" at a = a_min > 0.

All equations use dimensionless time tau = H_0 * t, with the Hubble
constant H_0 absorbed into the time coordinate.  Density parameters
Omega_m and Omega_Lambda are the standard LCDM values.

Observational fitting:
  1. Luminosity distance d_L(z) via numerical quadrature of 1/E(z).
  2. CMB shift parameter R = sqrt(Omega_m) * integral_0^{z_*} dz/E(z).
  3. BAO sound horizon r_d approximation (Eisenstein & Hu 1998).
  4. Chi-square fitting to synthetic Pantheon+/DESI-like data.
  5. AIC/BIC model comparison vs Lambda-CDM.
  6. Primordial spectral index n_s from bounce perturbation theory.

Some functions use a Rust backend (gororoba_py) when available for
improved performance.

References:
  Pinto-Neto & Fabris (2013), CQG 30, 143001 [arXiv:1306.0820].
  Ashtekar & Singh (2011), CQG 28, 213001 [LQC review].
  Peter & Pinto-Neto (2008), PRD 78, 063506 [Bohmian bounce].
  Eisenstein & Hu (1998), ApJ 496, 605 [BAO fitting formulae].
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy.integrate import odeint, quad
from scipy.optimize import minimize

# Try to import Rust bindings first
_USE_RUST = False
try:
    import gororoba_py as _gp
    _USE_RUST = True
except ImportError:
    pass

# Fixed seed for reproducibility across runs.
_RNG_SEED = 42

# Physical constants for cosmological fitting.
_C_KM_S = 299792.458  # speed of light in km/s


class QuantumCosmology:
    """
    Simulates Quantum Cosmology with a Regularized Bounce.

    The Quantum Potential Q ~ 1/a^6 causes a 'Big Bounce' avoiding the
    singularity a=0.  The quantum force F_q = -dQ/da ~ q_corr * a^{-7}
    enters the modified Raychaudhuri equation.

    All dynamical equations are written in dimensionless time tau = H_0*t,
    so H and H_dot carry factors of H_0 implicitly.
    """
    def __init__(self, omega_m=0.3, omega_l=0.7, h0=70.0):
        self.omega_m = omega_m
        self.omega_l = omega_l
        self.h0 = h0

    def system(self, state, t, q_corr=0.01):
        """
        ODE system in dimensionless time tau = H_0 * t.

        State = [a, H] where H is in units of H_0.

        Modified Raychaudhuri equation (dust-dominated, p=0, Lambda neglected):
          dH/dtau = -H^2 - (1/2) * Omega_m * a^{-3} + q_corr * a^{-7}

        The -H^2 term is the Hubble friction that ensures self-consistent
        deceleration.  Lambda is omitted because this model focuses on the
        near-bounce regime where the quantum correction dominates.
        The a^{-7} term is the Bohmian quantum force derived from the
        WDW quantum potential Q ~ l_P^4 / a^6 (Pinto-Neto & Fabris 2013).
        The coefficient q_corr encodes l_P^4 * H_0^2 in dimensionless units.
        """
        a, H = state

        if a < 1e-4:
            a = 1e-4

        # Bohmian quantum force: F_q = -dQ/da where Q ~ a^{-6}
        q_force = q_corr * (a ** -7)

        # Full Raychaudhuri equation: -H^2 provides Hubble friction.
        H_dot = -H**2 - 0.5 * self.omega_m * a**-3 + q_force
        a_dot = a * H

        return [a_dot, H_dot]

    def simulate(self, t_end=10.0, steps=1000, q_corr=0.001, seed=None):
        """
        Integrate the bounce cosmology ODE.

        Parameters
        ----------
        t_end : float
            End time in dimensionless units (tau = H_0 * t).
        steps : int
            Number of output time steps.
        q_corr : float
            Dimensionless quantum correction strength.
        seed : int or None
            RNG seed for vacuum-fluctuation noise.  If None, uses the
            module-level default seed for reproducibility.
        """
        t = np.linspace(0, t_end, steps)
        # Start at the "Bounce" point (minimum size), not a=0.
        a0 = 0.1
        H0 = 0.0  # Momentarily static at bounce

        sol = odeint(self.system, [a0, H0], t, args=(q_corr,))
        a_sol = sol[:, 0]
        H_sol = sol[:, 1]

        # Deceleration parameter: q = -1 - H_dot / H^2
        H_safe = np.where(np.abs(H_sol) < 1e-5, 1e-5, H_sol)
        q_param = -1 - np.gradient(H_sol, t[1] - t[0]) / (H_safe**2)

        # Vacuum-fluctuation noise (seeded for reproducibility).
        # This is a phenomenological proxy for primordial metric
        # perturbations; it does not replace a full perturbation theory.
        rng = np.random.default_rng(seed if seed is not None else _RNG_SEED)
        noise = rng.normal(0, 0.005, size=len(t))
        a_final = a_sol + noise

        return pd.DataFrame({
            'time_Gyr': t,
            'scale_factor_a': a_final,
            'scale_factor_a_clean': a_sol,
            'hubble_H': H_sol,
            'deceleration_q': q_param,
        })


# ---------------------------------------------------------------------------
# Observational Fitting: Luminosity Distance and Hubble Parameter
# ---------------------------------------------------------------------------

def hubble_E_lcdm(z, omega_m):
    """
    Dimensionless Hubble parameter E(z) = H(z)/H_0 for flat Lambda-CDM.

    E^2(z) = Omega_m * (1+z)^3 + (1 - Omega_m)
    """
    return np.sqrt(omega_m * (1.0 + z) ** 3 + (1.0 - omega_m))


def hubble_E_bounce(z, omega_m, q_corr):
    """
    Dimensionless Hubble parameter for the bounce model.

    E^2(z) = Omega_m * (1+z)^3 + (1 - Omega_m) + q_corr * (1+z)^7

    The q_corr*(1+z)^7 term is the Bohmian quantum correction from the
    WDW equation.  At low z, it is negligible; at high z (near bounce),
    it provides the repulsive force.  For observational fitting at
    z < 2.5, this term is vanishingly small unless q_corr is large.
    """
    zp1 = 1.0 + z
    E2 = omega_m * zp1 ** 3 + (1.0 - omega_m) + q_corr * zp1 ** 7
    return np.sqrt(np.maximum(E2, 1e-30))


def luminosity_distance(z, omega_m, h0, q_corr=0.0, n_quad=500):
    """
    Luminosity distance d_L(z) in Mpc.

    d_L(z) = (c/H_0) * (1+z) * integral_0^z dz' / E(z')

    Uses Rust backend when gororoba_py is available.

    Parameters
    ----------
    z : float or ndarray
        Redshift(s).
    omega_m : float
        Matter density parameter.
    h0 : float
        Hubble constant in km/s/Mpc.
    q_corr : float
        Bounce quantum correction (0 = Lambda-CDM).
    n_quad : int
        Quadrature points for numerical integration.

    Returns
    -------
    d_L : float or ndarray
        Luminosity distance(s) in Mpc.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    d_L = np.zeros_like(z)

    if _USE_RUST:
        # Use Rust backend (handles one z at a time)
        for i, zi in enumerate(z):
            if zi <= 0:
                d_L[i] = 0.0
            else:
                d_L[i] = _gp.py_luminosity_distance(zi, omega_m, h0, q_corr)
        return d_L.squeeze()

    # Python fallback using scipy.integrate.quad
    for i, zi in enumerate(z):
        if zi <= 0:
            d_L[i] = 0.0
            continue

        def integrand(zp):
            if q_corr == 0.0:
                return 1.0 / hubble_E_lcdm(zp, omega_m)
            return 1.0 / hubble_E_bounce(zp, omega_m, q_corr)

        val, _ = quad(integrand, 0.0, zi)
        d_L[i] = (_C_KM_S / h0) * (1.0 + zi) * val

    return d_L.squeeze()


def distance_modulus(z, omega_m, h0, q_corr=0.0):
    """
    Distance modulus mu(z) = 5 * log10(d_L / 10 pc).

    Returns
    -------
    mu : ndarray
        Distance modulus in magnitudes.
    """
    d_L = luminosity_distance(z, omega_m, h0, q_corr=q_corr)
    d_L = np.atleast_1d(d_L)
    # d_L is in Mpc; convert to pc for the modulus formula.
    d_L_pc = d_L * 1e6
    d_L_pc = np.maximum(d_L_pc, 1e-30)
    return 5.0 * np.log10(d_L_pc / 10.0)


def cmb_shift_parameter(omega_m, h0, q_corr=0.0, z_star=1089.0):
    """
    CMB shift parameter R = sqrt(Omega_m) * d_C(z_*) * H_0 / c.

    d_C(z) = (c/H_0) * integral_0^z dz'/E(z') is the comoving distance.
    """
    def integrand(z):
        if q_corr == 0.0:
            return 1.0 / hubble_E_lcdm(z, omega_m)
        return 1.0 / hubble_E_bounce(z, omega_m, q_corr)

    val, _ = quad(integrand, 0.0, z_star)
    return np.sqrt(omega_m) * val


def bao_sound_horizon_approx(omega_m, h0):
    """
    BAO sound horizon r_d via Eisenstein & Hu (1998) fitting formula.

    r_d ~ 147.05 * (Omega_m * h^2 / 0.1326)^{-0.255}
          * (Omega_b * h^2 / 0.02273)^{-0.128}  [Mpc]

    We fix Omega_b * h^2 = 0.02237 (Planck 2018 value).
    """
    h = h0 / 100.0
    omega_mh2 = omega_m * h ** 2
    omega_bh2 = 0.02237
    r_d = 147.05 * (omega_mh2 / 0.1326) ** (-0.255) * (omega_bh2 / 0.02273) ** (-0.128)
    return r_d


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_sn_data(n_sn=200, omega_m_true=0.315, h0_true=67.4,
                               seed=_RNG_SEED):
    """
    Generate synthetic Type Ia supernova data following Lambda-CDM.

    Mimics Pantheon+ distribution: redshifts in [0.01, 2.3] with
    Gaussian distance modulus errors of 0.10-0.15 mag.

    Returns
    -------
    data : dict with keys 'z', 'mu_obs', 'mu_err'.
    """
    rng = np.random.default_rng(seed)
    z = np.sort(rng.uniform(0.01, 2.3, n_sn))
    mu_true = distance_modulus(z, omega_m_true, h0_true, q_corr=0.0)
    mu_err = rng.uniform(0.10, 0.15, n_sn)
    mu_obs = mu_true + rng.normal(0, mu_err)
    return {"z": z, "mu_obs": mu_obs, "mu_err": mu_err}


def generate_synthetic_bao_data(omega_m_true=0.315, h0_true=67.4,
                                seed=_RNG_SEED):
    """
    Generate synthetic BAO data mimicking DESI DR1 measurements.

    Returns distance ratios D_V(z)/r_d at effective redshifts.
    D_V(z) = [z * d_L(z)^2 / ((1+z)^2 * E(z))]^{1/3} * (c/H_0)^{-1/3}
    """
    rng = np.random.default_rng(seed + 100)
    z_eff = np.array([0.30, 0.51, 0.71, 0.93, 1.32, 1.49, 2.33])
    r_d = bao_sound_horizon_approx(omega_m_true, h0_true)

    dv_rd_true = np.zeros_like(z_eff)
    for i, zi in enumerate(z_eff):
        d_L_val = luminosity_distance(zi, omega_m_true, h0_true, q_corr=0.0)
        d_C = d_L_val / (1.0 + zi)
        E_val = hubble_E_lcdm(zi, omega_m_true)
        d_H = _C_KM_S / (h0_true * E_val)
        D_V = (zi * d_C ** 2 * d_H) ** (1.0 / 3.0)
        dv_rd_true[i] = D_V / r_d

    dv_rd_err = 0.02 * dv_rd_true  # 2% errors
    dv_rd_obs = dv_rd_true + rng.normal(0, dv_rd_err)

    return {
        "z_eff": z_eff,
        "dv_rd_obs": dv_rd_obs,
        "dv_rd_err": dv_rd_err,
    }


# ---------------------------------------------------------------------------
# Chi-square and Model Fitting
# ---------------------------------------------------------------------------

def chi2_sn(params, sn_data, model="lcdm"):
    """
    Chi-square for supernova distance modulus data.

    Parameters
    ----------
    params : array-like
        For 'lcdm': [omega_m, h0].
        For 'bounce': [omega_m, h0, q_corr].
    sn_data : dict
        Keys: 'z', 'mu_obs', 'mu_err'.
    model : str
        'lcdm' or 'bounce'.
    """
    if model == "lcdm":
        omega_m, h0 = params
        q_corr = 0.0
    else:
        omega_m, h0, q_corr = params

    if omega_m < 0.01 or omega_m > 0.99 or h0 < 50.0 or h0 > 90.0:
        return 1e10
    if model == "bounce" and q_corr < 0:
        return 1e10

    mu_model = distance_modulus(sn_data["z"], omega_m, h0, q_corr=q_corr)
    residuals = (sn_data["mu_obs"] - mu_model) / sn_data["mu_err"]
    return float(np.sum(residuals ** 2))


def chi2_bao(params, bao_data, model="lcdm"):
    """
    Chi-square for BAO distance ratio data.
    """
    if model == "lcdm":
        omega_m, h0 = params
        q_corr = 0.0
    else:
        omega_m, h0, q_corr = params

    if omega_m < 0.01 or omega_m > 0.99 or h0 < 50.0 or h0 > 90.0:
        return 1e10

    r_d = bao_sound_horizon_approx(omega_m, h0)

    dv_rd_model = np.zeros_like(bao_data["z_eff"])
    for i, zi in enumerate(bao_data["z_eff"]):
        d_L_val = luminosity_distance(zi, omega_m, h0, q_corr=q_corr)
        d_C = d_L_val / (1.0 + zi)
        if q_corr == 0.0:
            E_val = hubble_E_lcdm(zi, omega_m)
        else:
            E_val = hubble_E_bounce(zi, omega_m, q_corr)
        d_H = _C_KM_S / (h0 * E_val)
        D_V = (zi * d_C ** 2 * d_H) ** (1.0 / 3.0)
        dv_rd_model[i] = D_V / r_d

    residuals = (bao_data["dv_rd_obs"] - dv_rd_model) / bao_data["dv_rd_err"]
    return float(np.sum(residuals ** 2))


def fit_model(sn_data, bao_data, model="lcdm"):
    """
    Joint fit of SN + BAO data.

    Parameters
    ----------
    sn_data : dict
    bao_data : dict
    model : str
        'lcdm' or 'bounce'.

    Returns
    -------
    result : dict
        Best-fit parameters, chi2, n_params, AIC, BIC.
    """
    n_data = len(sn_data["z"]) + len(bao_data["z_eff"])

    def total_chi2(params):
        return chi2_sn(params, sn_data, model) + chi2_bao(params, bao_data, model)

    if model == "lcdm":
        x0 = [0.3, 70.0]
        bounds = [(0.1, 0.5), (60.0, 80.0)]
        n_params = 2
    else:
        x0 = [0.3, 70.0, 1e-6]
        bounds = [(0.1, 0.5), (60.0, 80.0), (0.0, 1e-2)]
        n_params = 3

    res = minimize(total_chi2, x0, method="L-BFGS-B", bounds=bounds)

    chi2_val = res.fun
    aic = chi2_val + 2.0 * n_params
    bic = chi2_val + n_params * np.log(n_data)

    if model == "lcdm":
        best = {"omega_m": res.x[0], "h0": res.x[1], "q_corr": 0.0}
    else:
        best = {"omega_m": res.x[0], "h0": res.x[1], "q_corr": res.x[2]}

    return {
        "model": model,
        "best_fit": best,
        "chi2": chi2_val,
        "n_params": n_params,
        "n_data": n_data,
        "aic": aic,
        "bic": bic,
    }


# ---------------------------------------------------------------------------
# Primordial Spectral Index from Bounce Perturbation Theory
# ---------------------------------------------------------------------------

def spectral_index_bounce(q_corr, omega_m=0.315):
    """
    Approximate primordial spectral index n_s for the bounce model.

    In Bohmian bounce cosmology (Pinto-Neto & Fabris 2013), the scalar
    perturbation spectrum is nearly scale-invariant with a small
    correction from the quantum potential:

      n_s ~ 1 - 2 * (q_corr / omega_m)^{1/3}

    This is derived from the mode equation for perturbations crossing
    the bounce, where the quantum potential modifies the effective
    potential felt by each Fourier mode.

    At q_corr=0, n_s=1 (exact Harrison-Zeldovich).  A small positive
    q_corr gives n_s < 1 (red tilt), consistent with Planck data
    (n_s = 0.965 +/- 0.004).

    Parameters
    ----------
    q_corr : float
        Quantum correction parameter.
    omega_m : float
        Matter density parameter.

    Returns
    -------
    n_s : float
        Spectral index.
    """
    if q_corr <= 0:
        return 1.0
    ratio = q_corr / omega_m
    return 1.0 - 2.0 * ratio ** (1.0 / 3.0)


# ---------------------------------------------------------------------------
# Full Observational Fitting Pipeline
# ---------------------------------------------------------------------------

def run_observational_fit(output_csv="data/csv/bounce_cosmology_fit_results.csv"):
    """
    Run the full observational fitting pipeline.

    Steps:
      1. Generate synthetic SN Ia and BAO data from Lambda-CDM truth.
      2. Fit Lambda-CDM model (2 params: Omega_m, H_0).
      3. Fit bounce model (3 params: Omega_m, H_0, q_corr).
      4. Compare AIC/BIC.
      5. Compute spectral index n_s for best-fit q_corr.

    Returns
    -------
    df : DataFrame
        Fit results including chi2, AIC, BIC, delta_BIC.
    """
    sn_data = generate_synthetic_sn_data()
    bao_data = generate_synthetic_bao_data()

    lcdm_result = fit_model(sn_data, bao_data, model="lcdm")
    bounce_result = fit_model(sn_data, bao_data, model="bounce")

    delta_bic = bounce_result["bic"] - lcdm_result["bic"]
    delta_aic = bounce_result["aic"] - lcdm_result["aic"]

    n_s = spectral_index_bounce(bounce_result["best_fit"]["q_corr"])

    rows = [
        {
            "model": "lcdm",
            "omega_m": lcdm_result["best_fit"]["omega_m"],
            "h0": lcdm_result["best_fit"]["h0"],
            "q_corr": 0.0,
            "chi2": lcdm_result["chi2"],
            "n_params": lcdm_result["n_params"],
            "aic": lcdm_result["aic"],
            "bic": lcdm_result["bic"],
            "delta_bic": 0.0,
            "delta_aic": 0.0,
            "n_s": 1.0,
        },
        {
            "model": "bounce",
            "omega_m": bounce_result["best_fit"]["omega_m"],
            "h0": bounce_result["best_fit"]["h0"],
            "q_corr": bounce_result["best_fit"]["q_corr"],
            "chi2": bounce_result["chi2"],
            "n_params": bounce_result["n_params"],
            "aic": bounce_result["aic"],
            "bic": bounce_result["bic"],
            "delta_bic": delta_bic,
            "delta_aic": delta_aic,
            "n_s": n_s,
        },
    ]

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def synthesize_cosmology_data():
    print("Simulating Regularized Quantum Big Bounce...")
    qc = QuantumCosmology()

    combined_data = {}
    master_df = None

    for i in range(1, 9):
        # Varying the "Bounce Stiffness" (quantum correction strength)
        q_c = 0.005 * i
        df = qc.simulate(q_corr=q_c, seed=_RNG_SEED + i)
        combined_data[f"QCosmo_{i}"] = df["scale_factor_a"]
        if master_df is None:
            master_df = df[["time_Gyr"]].copy()

    final_df = pd.concat([master_df, pd.DataFrame(combined_data)], axis=1)

    target_path = "curated/02_simulations_pde_quantum/flrw_quantum_potential_evolution.csv"
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    final_df.to_csv(target_path, index=False)
    print(f"Saved Regularized Cosmology Data to {target_path}")


if __name__ == "__main__":
    df = run_observational_fit()
    print(df.to_string(index=False))
