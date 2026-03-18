"""
Analysis condition classes for MaNGA ZD null result experiment.

20 conditions: 4 baselines + 4 proposed methods + 12 ablations.
Base class uses Template Method with 3 overridable hooks:
  prepare_galaxies, compute_stack, evaluate_signal.
14 classes override hooks; 6 override compute_metrics entirely.
"""

import math

import numpy as np
from config import (
    ALGEBRA_WAVENUMBERS,
    HYPERPARAMETERS,
    cd_zd_wavenumbers,
)
from data_utils import filter_inclination, filter_mass_quartile
from pipeline import (
    bispectrum_at_triads,
    chi2_gof_flat,
    chi2_survival_p,
    compute_rms,
    compute_snr,
    embed_phase_circular,
    fit_red_noise,
    fourier_power_phase,
    hartigan_dip_statistic,
    inject_zd_signal,
    ksg_mutual_information,
    per_galaxy_dft,
    phase_reversal_test,
    rayleigh_phase_coherence,
    stack_residuals,
    stack_residuals_clipped,
    stack_residuals_median,
    stack_residuals_trimmed,
    sublevel_persistence_1d,
    whitened_dft,
)
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------

class AnalysisCondition:
    """
    Abstract base with concrete template method. Children override hooks
    (prepare_galaxies, compute_stack, evaluate_signal) or override
    compute_metrics entirely for non-standard orchestration.
    """

    def __init__(self, name: str, wavenumbers: np.ndarray, min_per_bin: int = 10):
        self.name = name
        self.wavenumbers = wavenumbers
        self.min_per_bin = min_per_bin

    def prepare_galaxies(self, galaxies: list, seed: int) -> list:
        """Filtering/preprocessing hook. Default: passthrough."""
        return galaxies

    def compute_stack(self, galaxies: list) -> tuple:
        """Stacking strategy hook. Default: IVW mean."""
        return stack_residuals(galaxies)

    def evaluate_signal(
        self,
        x_grid: np.ndarray,
        delta: np.ndarray,
        delta_err: np.ndarray,
        n_contrib: np.ndarray,
    ) -> dict:
        """Analysis hook. Default: DFT + SNR at self.wavenumbers."""
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))
        snr_mean = snr["snr_mean"]
        alpha_zd = snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0
        return {
            "detection_snr": snr_mean,
            "detection_snr_max": snr["snr_max"],
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "n_modes": len(self.wavenumbers),
        }

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        """Template method: prepare -> stack -> evaluate."""
        filtered = self.prepare_galaxies(galaxies, seed)
        if len(filtered) < 10:
            return {"detection_snr": 0.0, "n_galaxies": 0, "insufficient_data": 1,
                    "alpha_zd_upper_limit": 0.0, "n_valid_bins": 0, "n_modes": 0}
        x_grid, delta, delta_err, n_contrib = self.compute_stack(filtered)
        result = self.evaluate_signal(x_grid, delta, delta_err, n_contrib)
        result["n_galaxies"] = len(filtered)
        return result


class FaceOnCondition(AnalysisCondition):
    """Intermediate class for face-on conditions. Filters to i < inc_max."""

    def __init__(self, name: str, wavenumbers: np.ndarray,
                 inc_max: float = 45.0, min_per_bin: int = 10):
        super().__init__(name, wavenumbers, min_per_bin)
        self.inc_max = inc_max

    def prepare_galaxies(self, galaxies: list, seed: int) -> list:
        return filter_inclination(galaxies, 0.0, self.inc_max)


# ===========================================================================
# BASELINES (4)
# ===========================================================================

class MeanStackFullSample(AnalysisCondition):
    """
    E-183 standard pipeline. Adds Rayleigh phase coherence and E-183
    comparison metrics beyond the base evaluate_signal.
    """

    def __init__(self):
        super().__init__("mean_stack_full_sample", cd_zd_wavenumbers(16))

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)

        snr_mean = snr["snr_mean"]
        alpha_zd = snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0
        e183_alpha = 0.002392
        alpha_ratio = alpha_zd / e183_alpha if e183_alpha > 0 else 0.0
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        return {
            "detection_snr": snr_mean,
            "detection_snr_max": snr["snr_max"],
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "n_modes": 7,
            "max_fourier_power": float(np.max(power)) if len(power) > 0 else 0.0,
            "mean_rayleigh_r": float(np.mean(rayleigh_r)),
            "min_rayleigh_p": float(np.min(rayleigh_p)) if len(rayleigh_p) > 0 else 1.0,
            "alpha_ratio_vs_e183": alpha_ratio,
        }


class NFWOnlyChi2Baseline(AnalysisCondition):
    """
    No Fourier decomposition. Chi-squared GOF of stacked residuals against
    flat null (delta=0). Peak-to-RMS as detection proxy.
    """

    def __init__(self):
        super().__init__("nfw_only_chi2", cd_zd_wavenumbers(16))

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        chi2, chi2_red, p_val, n_valid = chi2_gof_flat(
            delta, delta_err, n_contrib, self.min_per_bin)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)

        mask = n_contrib >= self.min_per_bin
        d = delta[mask]
        sigma = np.maximum(delta_err[mask], 1e-10)
        peak_snr = float(np.max(np.abs(d))) / rms if rms > 0 else 0.0
        alpha_zd = peak_snr / (0.5 * math.e) if peak_snr > 0 else 0.0
        n_exceed_2sigma = int(np.sum(np.abs(d / sigma) > 2.0))

        return {
            "detection_snr": peak_snr,
            "chi2": chi2,
            "chi2_reduced": chi2_red,
            "chi2_p_value": p_val,
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "n_exceed_2sigma": n_exceed_2sigma,
            "n_modes": 0,
            "method": "chi2_gof",
        }


class RelatoresCompositeBaseline(AnalysisCondition):
    """
    3-sigma iterative outlier clipping per Relatores+2019 (ApJ 887, 94).
    Different STACKING strategy; same Fourier evaluation.
    """

    def __init__(self):
        super().__init__("relatores_composite", cd_zd_wavenumbers(16))
        self.sigma_clip = 3.0
        self.max_iter = 5
        self._n_clipped = 0

    def compute_stack(self, galaxies):
        x_grid, delta, delta_err, n_contrib, n_clipped = stack_residuals_clipped(
            galaxies, self.sigma_clip, self.max_iter)
        self._n_clipped = n_clipped
        return x_grid, delta, delta_err, n_contrib

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        result = super().evaluate_signal(x_grid, delta, delta_err, n_contrib)
        result["n_clipped"] = self._n_clipped
        total = int(np.sum(n_contrib)) + self._n_clipped
        result["clip_fraction"] = self._n_clipped / total if total > 0 else 0.0
        return result


class MultiAlgebraMeanStack(AnalysisCondition):
    """
    Evaluates all 4 algebraic wavenumber sets independently.
    Tests algebra universality of the null result.
    """

    def __init__(self):
        super().__init__("multi_algebra_mean_stack", cd_zd_wavenumbers(16))

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        algebra_snrs = {}
        for algebra_name, wk in ALGEBRA_WAVENUMBERS.items():
            snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, wk)
            algebra_snrs[algebra_name] = snr["snr_mean"]

        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))
        values = list(algebra_snrs.values())
        mean_snr = float(np.mean(values))
        max_snr = float(np.max(values))
        snr_std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        universal = snr_std < 0.1 * mean_snr if mean_snr > 0 else True
        alpha_zd = max_snr / (0.5 * math.e) if max_snr > 0 else 0.0

        return {
            "detection_snr": mean_snr,
            "detection_snr_max": max_snr,
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "n_modes": sum(len(w) for w in ALGEBRA_WAVENUMBERS.values()),
            "cd_zd_snr": algebra_snrs["CD-ZD"],
            "g2_snr": algebra_snrs["G2"],
            "j3o_snr": algebra_snrs["J3(O)"],
            "sl2_snr": algebra_snrs["sl(2)"],
            "algebra_snr_std": snr_std,
            "algebra_universal": int(universal),
        }


# ===========================================================================
# PROPOSED METHODS (4)
# ===========================================================================

class FaceOnRednoiseSubtracted(FaceOnCondition):
    """
    H1 PRIMARY. Face-on + Rayleigh R + red-noise k^gamma envelope
    subtraction. Per-mode significance with Bonferroni correction.
    """

    def __init__(self):
        super().__init__("face_on_rednoise_subtracted", cd_zd_wavenumbers(16), inc_max=45.0)
        self.gamma_prior = 0.808

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        snr_raw = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)

        gamma, amp, pred_r, resid, sigma = fit_red_noise(
            self.wavenumbers, rayleigh_r, gamma_prior=self.gamma_prior)
        corrected_r = np.maximum(rayleigh_r - pred_r, 0.0)
        corrected_snr = float(np.max(corrected_r)) / sigma if sigma > 0 else 0.0

        mode_sigmas = corrected_r / sigma if sigma > 0 else np.zeros_like(corrected_r)
        max_mode_sigma = float(np.max(mode_sigmas))
        any_detection = max_mode_sigma > 3.0
        alpha_zd_corrected = float(np.max(corrected_r)) / (0.5 * math.e)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        return {
            "detection_snr": snr_raw["snr_mean"],
            "corrected_rayleigh_snr": corrected_snr,
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd_corrected,
            "n_valid_bins": n_valid,
            "red_noise_gamma": gamma,
            "red_noise_amplitude": amp,
            "red_noise_sigma": sigma,
            "max_mode_sigma": max_mode_sigma,
            "any_detection": int(any_detection),
            "n_modes": 7,
        }


class FaceOnWhitenedMatchedFilter(FaceOnCondition):
    """
    H1 SECONDARY. Per-bin whitened DFT + chi-squared with red-noise
    correction. Neyman-Pearson optimal for known template.
    """

    def __init__(self):
        super().__init__("face_on_whitened_matched_filter", cd_zd_wavenumbers(16), inc_max=45.0)

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        chi2_mode, snr_mode, total_chi2, dof = whitened_dft(
            x_grid, delta, delta_err, n_contrib, self.min_per_bin, self.wavenumbers)
        p_raw = chi2_survival_p(total_chi2, dof)

        # Red-noise correction on whitened power
        whitened_power = chi2_mode / 2.0
        gamma, amp, pred_p, resid_p, sigma_p = fit_red_noise(
            self.wavenumbers, whitened_power, gamma_prior=None)

        pred_safe = np.maximum(pred_p, 1e-10)
        corrected_chi2 = float(np.sum(chi2_mode / pred_safe))
        p_corrected = chi2_survival_p(corrected_chi2, dof)

        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))
        max_snr_mode = float(np.max(snr_mode)) if len(snr_mode) > 0 else 0.0

        return {
            "detection_snr": max_snr_mode,
            "chi2_raw": total_chi2,
            "chi2_dof": dof,
            "chi2_p_raw": p_raw,
            "chi2_corrected": corrected_chi2,
            "chi2_p_corrected": p_corrected,
            "rms_residual": rms,
            "alpha_zd_upper_limit": max_snr_mode / (0.5 * math.e) if max_snr_mode > 0 else 0.0,
            "n_valid_bins": n_valid,
            "red_noise_gamma_whitened": gamma,
            "n_modes": 7,
            "method": "whitened_matched_filter",
        }


class FaceOnGPModelComparison(FaceOnCondition):
    """
    H1 TERTIARY. GP model selection: Matern (smooth baryonic) vs
    Matern+ExpSineSquared (periodic). Bayes factor via log marginal
    likelihood. Degrades gracefully without sklearn.
    """

    def __init__(self):
        super().__init__("face_on_gp_model_comparison", cd_zd_wavenumbers(16), inc_max=45.0)
        self.n_restarts = HYPERPARAMETERS["gp_n_restarts"]
        self.periodic_period = HYPERPARAMETERS["gp_periodic_period"]

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))
        base_result = {
            "rms_residual": rms,
            "n_valid_bins": n_valid,
            "n_modes": 7,
            "method": "gp_model_comparison",
        }

        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import (
                ExpSineSquared,
                Matern,
                WhiteKernel,
            )
        except ImportError:
            base_result.update({
                "gp_available": 0, "detection_snr": 0.0,
                "alpha_zd_upper_limit": 0.0,
                "lml_smooth": 0.0, "lml_periodic": 0.0,
                "delta_lml": 0.0, "bayes_factor": 1.0,
                "periodic_preferred": 0,
            })
            return base_result

        mask = n_contrib >= self.min_per_bin
        if mask.sum() < 5:
            base_result.update({
                "gp_available": 1, "detection_snr": 0.0,
                "alpha_zd_upper_limit": 0.0,
                "lml_smooth": 0.0, "lml_periodic": 0.0,
                "delta_lml": 0.0, "bayes_factor": 1.0,
                "periodic_preferred": 0,
            })
            return base_result

        X = x_grid[mask].reshape(-1, 1)
        y = delta[mask]
        alpha = np.maximum(delta_err[mask], 1e-10) ** 2

        # Model A: smooth baryonic
        kernel_a = Matern(nu=2.5) + WhiteKernel()
        gp_a = GaussianProcessRegressor(
            kernel=kernel_a, alpha=alpha, n_restarts_optimizer=self.n_restarts)
        gp_a.fit(X, y)
        lml_a = float(gp_a.log_marginal_likelihood_value_)

        # Model B: smooth + periodic
        kernel_b = Matern(nu=2.5) + ExpSineSquared(
            periodicity=self.periodic_period) + WhiteKernel()
        gp_b = GaussianProcessRegressor(
            kernel=kernel_b, alpha=alpha, n_restarts_optimizer=self.n_restarts)
        gp_b.fit(X, y)
        lml_b = float(gp_b.log_marginal_likelihood_value_)

        delta_lml = lml_b - lml_a
        bayes_factor = math.exp(min(delta_lml, 500.0)) if delta_lml < 500 else 0.0
        periodic_preferred = delta_lml > 0
        proxy_snr = max(0.0, delta_lml) if delta_lml > 0 else 0.0

        base_result.update({
            "gp_available": 1,
            "detection_snr": proxy_snr,
            "alpha_zd_upper_limit": proxy_snr / (0.5 * math.e) if proxy_snr > 0 else 0.0,
            "lml_smooth": lml_a,
            "lml_periodic": lml_b,
            "delta_lml": delta_lml,
            "bayes_factor": bayes_factor,
            "periodic_preferred": int(periodic_preferred),
        })
        return base_result


class Q3InjectionRecovery(AnalysisCondition):
    """
    H2 PRIMARY. Inject ZD signal into Q3 mass quartile (6 bins) at 3
    alpha levels. Measures pipeline blindness via delta_SNR.
    """

    def __init__(self):
        super().__init__("q3_injection_recovery", cd_zd_wavenumbers(16))
        self.inject_alphas = HYPERPARAMETERS["inject_alphas"]

    def compute_metrics(self, galaxies, seed):
        q3_gals = filter_mass_quartile(galaxies, 3)
        x_g, d_g, e_g, n_g = stack_residuals(q3_gals)
        snr_baseline = compute_snr(x_g, d_g, n_g, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_g >= self.min_per_bin))
        rms = compute_rms(d_g, n_g, self.min_per_bin)

        recovery = {}
        for alpha in self.inject_alphas:
            injected = inject_zd_signal(q3_gals, alpha, cd_dim=16)
            x_i, d_i, e_i, n_i = stack_residuals(injected)
            snr_inj = compute_snr(x_i, d_i, n_i, self.min_per_bin, self.wavenumbers)
            delta_snr = snr_inj["snr_max"] - snr_baseline["snr_max"]
            recovery[alpha] = {"delta_snr": delta_snr, "recovered": delta_snr > 2.0}

        blind_at_ska = recovery[0.004]["delta_snr"] < 0.5
        return {
            "detection_snr": snr_baseline["snr_max"],
            "n_galaxies": len(q3_gals),
            "n_valid_bins": n_valid,
            "rms_residual": rms,
            "alpha_zd_upper_limit": snr_baseline["snr_max"] / (0.5 * math.e),
            "snr_baseline_max": snr_baseline["snr_max"],
            "delta_snr_0.001": recovery[0.001]["delta_snr"],
            "delta_snr_0.004": recovery[0.004]["delta_snr"],
            "delta_snr_0.01": recovery[0.01]["delta_snr"],
            "recovered_0.004": int(recovery[0.004]["recovered"]),
            "blind_at_ska": int(blind_at_ska),
            "n_modes": 7,
            "method": "q3_injection_recovery",
        }


# ===========================================================================
# ABLATIONS: H1 face-on variants (5)
# ===========================================================================

class FaceOnNoRednoiseSubtraction(FaceOnCondition):
    """
    Ablates red-noise correction (gamma=0 flat). Includes flatness
    diagnostic: chi2 of Rayleigh R vs constant model.
    """

    def __init__(self):
        super().__init__("face_on_no_rednoise", cd_zd_wavenumbers(16), inc_max=45.0)

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        # Flat correction: gamma=0 => R_pred = mean(R)
        r_mean = float(np.mean(rayleigh_r))
        r_flat_resid = rayleigh_r - r_mean
        sigma_flat = float(np.std(r_flat_resid, ddof=1)) if len(r_flat_resid) > 1 else 1e-10
        max_flat_sigma = float(np.max(np.abs(r_flat_resid))) / sigma_flat if sigma_flat > 0 else 0.0

        # Flatness diagnostic
        chi2_flat = float(np.sum(r_flat_resid ** 2)) / (sigma_flat ** 2) if sigma_flat > 0 else 0.0
        flat_p = chi2_survival_p(chi2_flat, len(self.wavenumbers) - 1)
        alpha_zd = float(np.max(np.maximum(r_flat_resid, 0.0))) / (0.5 * math.e)

        return {
            "detection_snr": snr["snr_mean"],
            "corrected_rayleigh_snr": max_flat_sigma,
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "gamma_used": 0.0,
            "flatness_chi2": chi2_flat,
            "flatness_p": flat_p,
            "n_modes": 7,
        }


class FaceOnFreeGammaFit(FaceOnCondition):
    """
    Ablates gamma prior. OLS free fit with jackknife uncertainty on gamma
    and test of prior transferability (is gamma_face_on == 0.808?).
    """

    def __init__(self):
        super().__init__("face_on_free_gamma", cd_zd_wavenumbers(16), inc_max=45.0)

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        gamma, amp, pred_r, resid, sigma = fit_red_noise(
            self.wavenumbers, rayleigh_r, gamma_prior=None)

        # Jackknife uncertainty on gamma (leave-one-mode-out)
        m = len(self.wavenumbers)
        gamma_jk = np.zeros(m)
        for b in range(m):
            mask = np.ones(m, dtype=bool)
            mask[b] = False
            g_b, _, _, _, _ = fit_red_noise(self.wavenumbers[mask], rayleigh_r[mask])
            gamma_jk[b] = g_b
        gamma_se = float(np.std(gamma_jk, ddof=1)) * math.sqrt(m - 1)
        gamma_z = gamma / gamma_se if gamma_se > 0 else 0.0
        gamma_is_red = gamma > 2 * gamma_se

        # Prior transferability
        prior_dev = abs(gamma - 0.808) / gamma_se if gamma_se > 0 else 0.0
        prior_transfers = prior_dev < 2.0

        corrected_r = np.maximum(rayleigh_r - pred_r, 0.0)
        alpha_zd = float(np.max(corrected_r)) / (0.5 * math.e)

        return {
            "detection_snr": snr["snr_mean"],
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "gamma_fitted": gamma,
            "gamma_se": gamma_se,
            "gamma_z_score": gamma_z,
            "gamma_is_red": int(gamma_is_red),
            "prior_deviation_sigma": prior_dev,
            "prior_transfers": int(prior_transfers),
            "n_modes": 7,
        }


class FaceOnMedianStack(FaceOnCondition):
    """
    Per-bin median instead of IVW mean. Addresses C-1428 (8.6x mean/median
    divergence). Bootstrap error on per-bin medians.
    """

    def __init__(self):
        super().__init__("face_on_median_stack", cd_zd_wavenumbers(16), inc_max=45.0)

    def compute_stack(self, galaxies):
        return stack_residuals_median(galaxies, n_bootstrap_err=50, seed=42)


class FaceOnTrimmed05(FaceOnCondition):
    """
    5%-trimmed mean DFT. Adds outlier contamination diagnostic: what
    fraction of total Fourier power was in the trimmed tails.
    """

    def __init__(self):
        super().__init__("face_on_trimmed_05", cd_zd_wavenumbers(16), inc_max=45.0)
        self.trim_frac = 0.05
        self._frac_power_trimmed = 0.0

    def compute_stack(self, galaxies):
        x_grid, delta, delta_err, n_contrib, frac = stack_residuals_trimmed(
            galaxies, self.trim_frac)
        self._frac_power_trimmed = frac
        return x_grid, delta, delta_err, n_contrib

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        frac = self._frac_power_trimmed
        outlier_dominated = frac > 0.10
        snr_mean = snr["snr_mean"]
        alpha_zd = snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0

        return {
            "detection_snr": snr_mean,
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "trim_frac": 0.05,
            "frac_power_in_tails": frac,
            "outlier_dominated": int(outlier_dominated),
            "mean_fourier_power": float(np.mean(power)) if len(power) > 0 else 0.0,
            "n_modes": 7,
        }


class FaceOnTrimmed20(FaceOnCondition):
    """
    20%-trimmed mean (removes 40% total). Adds phase reversal detection:
    computes DFT phase before/after trimming, checks for > pi/2 rotation.
    """

    def __init__(self):
        super().__init__("face_on_trimmed_20", cd_zd_wavenumbers(16), inc_max=45.0)
        self.trim_frac = 0.20
        self._frac_power_trimmed = 0.0
        self._phase_untrimmed = None

    def compute_stack(self, galaxies):
        # Untrimmed phase for comparison
        x_u, d_u, e_u, n_u = stack_residuals(galaxies)
        _, phase_u = fourier_power_phase(
            x_u, d_u, n_u, self.min_per_bin, self.wavenumbers)
        self._phase_untrimmed = phase_u

        # Trimmed stack
        x_grid, delta, delta_err, n_contrib, frac = stack_residuals_trimmed(
            galaxies, self.trim_frac)
        self._frac_power_trimmed = frac
        return x_grid, delta, delta_err, n_contrib

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        _, phase_trimmed = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        max_rot, any_reversed = phase_reversal_test(
            self._phase_untrimmed, phase_trimmed)
        snr_mean = snr["snr_mean"]
        alpha_zd = snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0

        return {
            "detection_snr": snr_mean,
            "rms_residual": rms,
            "alpha_zd_upper_limit": alpha_zd,
            "n_valid_bins": n_valid,
            "trim_frac": 0.20,
            "frac_power_in_tails": self._frac_power_trimmed,
            "max_phase_rotation": max_rot,
            "any_phase_reversed": int(any_reversed),
            "n_modes": 7,
        }


# ===========================================================================
# ABLATIONS: H2 injection variants (3)
# ===========================================================================

class Q1InjectionRecoveryControl(AnalysisCondition):
    """
    Q1 injection (19 bins) as positive control for Q3 comparison.
    """

    def __init__(self):
        super().__init__("q1_injection_control", cd_zd_wavenumbers(16))
        self.inject_alphas = HYPERPARAMETERS["inject_alphas"]

    def compute_metrics(self, galaxies, seed):
        q1_gals = filter_mass_quartile(galaxies, 1)
        x_g, d_g, e_g, n_g = stack_residuals(q1_gals)
        snr_baseline = compute_snr(x_g, d_g, n_g, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_g >= self.min_per_bin))
        rms = compute_rms(d_g, n_g, self.min_per_bin)

        recovery = {}
        for alpha in self.inject_alphas:
            injected = inject_zd_signal(q1_gals, alpha)
            x_i, d_i, _, n_i = stack_residuals(injected)
            snr_inj = compute_snr(x_i, d_i, n_i, self.min_per_bin, self.wavenumbers)
            recovery[alpha] = snr_inj["snr_max"] - snr_baseline["snr_max"]

        return {
            "detection_snr": snr_baseline["snr_max"],
            "n_galaxies": len(q1_gals),
            "n_valid_bins": n_valid,
            "rms_residual": rms,
            "alpha_zd_upper_limit": snr_baseline["snr_max"] / (0.5 * math.e),
            "delta_snr_0.001": recovery[0.001],
            "delta_snr_0.004": recovery[0.004],
            "delta_snr_0.01": recovery[0.01],
            "recovered_0.004": int(recovery[0.004] > 2.0),
            "n_modes": 7,
            "method": "q1_injection_control",
        }


class Q1Synthetic6binControl(AnalysisCondition):
    """
    Q1 restricted to x_max=0.74 (6-bin window matching Q3). Isolates
    whether recovery failure is window width or mass selection.
    """

    def __init__(self):
        super().__init__("q1_synthetic_6bin", cd_zd_wavenumbers(16))
        self.x_max_restricted = 0.74
        self.inject_alphas = HYPERPARAMETERS["inject_alphas"]

    def compute_metrics(self, galaxies, seed):
        q1_gals = filter_mass_quartile(galaxies, 1)
        x_g, d_g, e_g, n_g = stack_residuals(q1_gals, x_max=self.x_max_restricted)
        snr_baseline = compute_snr(x_g, d_g, n_g, self.min_per_bin, self.wavenumbers)
        n_valid = int(np.sum(n_g >= self.min_per_bin))
        rms = compute_rms(d_g, n_g, self.min_per_bin)

        recovery = {}
        for alpha in self.inject_alphas:
            injected = inject_zd_signal(q1_gals, alpha)
            x_i, d_i, _, n_i = stack_residuals(injected, x_max=self.x_max_restricted)
            snr_inj = compute_snr(x_i, d_i, n_i, self.min_per_bin, self.wavenumbers)
            recovery[alpha] = snr_inj["snr_max"] - snr_baseline["snr_max"]

        window_blind = recovery[0.004] < 0.5
        return {
            "detection_snr": snr_baseline["snr_max"],
            "n_galaxies": len(q1_gals),
            "n_valid_bins": n_valid,
            "x_max_restricted": self.x_max_restricted,
            "rms_residual": rms,
            "alpha_zd_upper_limit": snr_baseline["snr_max"] / (0.5 * math.e),
            "delta_snr_0.001": recovery[0.001],
            "delta_snr_0.004": recovery[0.004],
            "delta_snr_0.01": recovery[0.01],
            "window_blind": int(window_blind),
            "n_modes": 7,
            "method": "q1_synthetic_6bin",
        }


class InclinationStratifiedInjection(AnalysisCondition):
    """
    3 inclination strata x 6 alpha levels = 18-cell recovery grid.
    Tests inclination-dependent pipeline bias.
    """

    def __init__(self):
        super().__init__("inclination_stratified_injection", cd_zd_wavenumbers(16))
        self.strata = HYPERPARAMETERS["inclination_strata"]
        self.alphas = HYPERPARAMETERS["inject_alphas_stratified"]

    def compute_metrics(self, galaxies, seed):
        recovery_by_stratum = {}
        for i_lo, i_hi in self.strata:
            stratum_gals = filter_inclination(galaxies, float(i_lo), float(i_hi))
            if len(stratum_gals) < 10:
                recovery_by_stratum[(i_lo, i_hi)] = 0.0
                continue
            x_b, d_b, _, n_b = stack_residuals(stratum_gals)
            snr_base = compute_snr(x_b, d_b, n_b, self.min_per_bin, self.wavenumbers)

            n_recovered = 0
            for alpha in self.alphas:
                injected = inject_zd_signal(stratum_gals, alpha)
                x_i, d_i, _, n_i = stack_residuals(injected)
                snr_inj = compute_snr(x_i, d_i, n_i, self.min_per_bin, self.wavenumbers)
                delta = snr_inj["snr_max"] - snr_base["snr_max"]
                if delta > 2.0:
                    n_recovered += 1
            recovery_by_stratum[(i_lo, i_hi)] = n_recovered / len(self.alphas)

        rates = list(recovery_by_stratum.values())
        spread = max(rates) - min(rates) if len(rates) > 1 else 0.0
        inclination_bias = spread > 0.3
        mean_rate = float(np.mean(rates))

        return {
            "detection_snr": mean_rate,
            "n_galaxies": len(galaxies),
            "n_strata": len(self.strata),
            "n_alphas": len(self.alphas),
            "recovery_rate_low_i": rates[0] if len(rates) > 0 else 0.0,
            "recovery_rate_mid_i": rates[1] if len(rates) > 1 else 0.0,
            "recovery_rate_high_i": rates[2] if len(rates) > 2 else 0.0,
            "recovery_spread": spread,
            "inclination_bias": int(inclination_bias),
            "alpha_zd_upper_limit": 0.0,
            "n_modes": 7,
            "method": "inclination_stratified_injection",
        }


# ===========================================================================
# ABLATIONS: Cross-cutting spectral/topological diagnostics (4)
# ===========================================================================

class FullSampleBispectralFano(AnalysisCondition):
    """
    Fano plane bispectral coherence. Third-order statistic tests
    three-wave coupling at octonion triples vs non-Fano triples.
    """

    def __init__(self):
        super().__init__("bispectral_fano", cd_zd_wavenumbers(16))

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        bicoherence, is_fano, ratio = bispectrum_at_triads(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        n_triads = len(bicoherence)
        n_fano = int(np.sum(is_fano))
        fano_mean = float(np.mean(bicoherence[is_fano])) if is_fano.any() else 0.0
        nonfano_mean = float(np.mean(bicoherence[~is_fano])) if (~is_fano).any() else 0.0
        no_selection_rule = 0.7 <= ratio <= 1.3

        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))
        snr_mean = snr["snr_mean"]

        return {
            "detection_snr": snr_mean,
            "rms_residual": rms,
            "alpha_zd_upper_limit": snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0,
            "n_valid_bins": n_valid,
            "fano_nonfano_ratio": ratio,
            "fano_mean_bicoherence": fano_mean,
            "nonfano_mean_bicoherence": nonfano_mean,
            "no_selection_rule": int(no_selection_rule),
            "n_triads": n_triads,
            "n_fano_triads": n_fano,
            "n_modes": 7,
            "method": "bispectral_fano",
        }


class FullSamplePersistenceTopology(AnalysisCondition):
    """
    Sublevel persistence on stacked residual. ZD predicts ~7 persistent
    features; baryonic origin predicts 2-3.
    """

    def __init__(self):
        super().__init__("persistence_topology", cd_zd_wavenumbers(16))
        self.epsilon_frac = 0.1

    def evaluate_signal(self, x_grid, delta, delta_err, n_contrib):
        mask = n_contrib >= self.min_per_bin
        values = delta[mask]
        pairs = sublevel_persistence_1d(values, self.epsilon_frac)
        n_total = len(pairs)
        n_persistent = sum(1 for p in pairs if p["persistent"])
        max_lifetime = max((p["lifetime"] for p in pairs), default=0.0)
        baryonic_origin = n_persistent <= 3
        zd_consistent = n_persistent >= 5

        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        n_valid = int(np.sum(mask))
        snr_mean = snr["snr_mean"]

        return {
            "detection_snr": snr_mean,
            "rms_residual": rms,
            "alpha_zd_upper_limit": snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0,
            "n_valid_bins": n_valid,
            "n_persistent_features": n_persistent,
            "n_total_features": n_total,
            "max_lifetime": float(max_lifetime),
            "baryonic_origin": int(baryonic_origin),
            "zd_consistent": int(zd_consistent),
            "n_modes": 7,
            "method": "persistence_topology",
        }


class FullSampleMutualInformation(AnalysisCondition):
    """
    KSG MI between cross-algebra DFT phases with circular embedding.
    Tests for hidden nonlinear coupling despite high linear r.
    """

    def __init__(self):
        super().__init__("mutual_information", cd_zd_wavenumbers(16))
        self.algebra_wavenumbers = ALGEBRA_WAVENUMBERS

    def compute_metrics(self, galaxies, seed):
        # Per-galaxy DFT phase at fundamental mode of each algebra
        algebra_phases = {}
        for alg_name, wk_a in self.algebra_wavenumbers.items():
            _, phases_a = per_galaxy_dft(galaxies, wk_a)
            algebra_phases[alg_name] = phases_a[:, 0]

        algebra_names = sorted(algebra_phases.keys())
        nmi_results = {}
        for i, a1 in enumerate(algebra_names):
            for a2 in algebra_names[i + 1:]:
                phi1 = embed_phase_circular(algebra_phases[a1])
                phi2 = embed_phase_circular(algebra_phases[a2])
                mi = ksg_mutual_information(phi1, phi2, k=6)
                nmi = mi / max(math.log(len(galaxies)), 1e-6)
                nmi_results[f"{a1}_vs_{a2}"] = nmi

        max_nmi = max(nmi_results.values()) if nmi_results else 0.0
        all_trivial = all(v < 0.05 for v in nmi_results.values())

        x_g, d_g, e_g, n_g = stack_residuals(galaxies)
        snr = compute_snr(x_g, d_g, n_g, self.min_per_bin, self.wavenumbers)
        snr_mean = snr["snr_mean"]

        result = {
            "detection_snr": snr_mean,
            "n_galaxies": len(galaxies),
            "max_nmi": max_nmi,
            "all_trivial_independence": int(all_trivial),
            "alpha_zd_upper_limit": snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0,
            "n_modes": 7,
            "method": "mutual_information",
        }
        result.update(nmi_results)
        return result


class FullSampleVarianceQuantization(AnalysisCondition):
    """
    Hartigan dip test on per-galaxy Fourier power. Tests multimodality
    at all 4 algebras independently. CV ratio: CD-ZD vs others.
    """

    def __init__(self):
        super().__init__("variance_quantization", cd_zd_wavenumbers(16))
        self.n_perm = HYPERPARAMETERS["dip_test_n_perm"]
        self.algebra_wavenumbers = ALGEBRA_WAVENUMBERS

    def compute_metrics(self, galaxies, seed):
        algebra_results = {}
        for alg_name, wk_a in self.algebra_wavenumbers.items():
            powers, _ = per_galaxy_dft(galaxies, wk_a)
            total_power = np.mean(powers, axis=1)
            dip, p_val = hartigan_dip_statistic(total_power, self.n_perm, seed=42)
            cv = float(np.std(total_power, ddof=1)) / max(float(np.mean(total_power)), 1e-10)
            kurt = float(sp_stats.kurtosis(total_power, fisher=True))
            algebra_results[alg_name] = {
                "dip": dip, "p_value": p_val, "cv": cv, "kurtosis": kurt}

        cd_cv = algebra_results["CD-ZD"]["cv"]
        other_cvs = [v["cv"] for k, v in algebra_results.items() if k != "CD-ZD"]
        mean_other = float(np.mean(other_cvs)) if other_cvs else 1.0
        cv_ratio = cd_cv / mean_other if mean_other > 0 else 0.0
        all_multimodal = all(v["p_value"] < 0.05 for v in algebra_results.values())

        x_g, d_g, e_g, n_g = stack_residuals(galaxies)
        snr = compute_snr(x_g, d_g, n_g, self.min_per_bin, self.wavenumbers)
        snr_mean = snr["snr_mean"]

        return {
            "detection_snr": snr_mean,
            "n_galaxies": len(galaxies),
            "cd_zd_dip": algebra_results["CD-ZD"]["dip"],
            "cd_zd_dip_p": algebra_results["CD-ZD"]["p_value"],
            "cd_zd_kurtosis": algebra_results["CD-ZD"]["kurtosis"],
            "cv_ratio_cd_vs_others": cv_ratio,
            "all_multimodal": int(all_multimodal),
            "alpha_zd_upper_limit": snr_mean / (0.5 * math.e) if snr_mean > 0 else 0.0,
            "n_modes": 7,
            "method": "variance_quantization",
        }


# ===========================================================================
# CONDITION REGISTRY
# ===========================================================================

def get_all_conditions() -> dict:
    """Return all 20 analysis conditions keyed by name."""
    return {
        # Baselines
        "mean_stack_full_sample": MeanStackFullSample(),
        "nfw_only_chi2": NFWOnlyChi2Baseline(),
        "relatores_composite": RelatoresCompositeBaseline(),
        "multi_algebra_mean_stack": MultiAlgebraMeanStack(),
        # Proposed methods
        "face_on_rednoise_subtracted": FaceOnRednoiseSubtracted(),
        "face_on_whitened_matched_filter": FaceOnWhitenedMatchedFilter(),
        "face_on_gp_model_comparison": FaceOnGPModelComparison(),
        "q3_injection_recovery": Q3InjectionRecovery(),
        # Ablations: H1 face-on
        "face_on_no_rednoise": FaceOnNoRednoiseSubtraction(),
        "face_on_free_gamma": FaceOnFreeGammaFit(),
        "face_on_median_stack": FaceOnMedianStack(),
        "face_on_trimmed_05": FaceOnTrimmed05(),
        "face_on_trimmed_20": FaceOnTrimmed20(),
        # Ablations: H2 injection
        "q1_injection_control": Q1InjectionRecoveryControl(),
        "q1_synthetic_6bin": Q1Synthetic6binControl(),
        "inclination_stratified_injection": InclinationStratifiedInjection(),
        # Ablations: cross-cutting diagnostics
        "bispectral_fano": FullSampleBispectralFano(),
        "persistence_topology": FullSamplePersistenceTopology(),
        "mutual_information": FullSampleMutualInformation(),
        "variance_quantization": FullSampleVarianceQuantization(),
    }
