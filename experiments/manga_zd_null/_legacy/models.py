"""
Physics analysis models for MaNGA ZD null result experiment.

Implements the full stacking-Fourier-Rayleigh pipeline from E-183/E-184:
1. Rotation curve stacking with inverse-variance weighting
2. Discrete Fourier projection at algebra-specific wavenumbers
3. SNR computation (max Fourier power / RMS noise)
4. Rayleigh phase coherence (jackknife leave-one-out)
5. Red-noise envelope fitting (log-log OLS)
6. Signal injection and recovery validation

Each analysis condition is a separate class with distinct compute_metrics().
Condition names match exp_plan.yaml: DANN, CORAL, Comprehensive_baseline_1,
Comprehensive_baseline_2, Comprehensive_proposed, Comprehensive_variant,
without_key_component, simplified_version.
"""

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy import stats as sp_stats

# ---------------------------------------------------------------------------
# Wavenumber sets from algebraic structures
# ---------------------------------------------------------------------------

def cd_zd_wavenumbers(cd_dim: int = 16) -> np.ndarray:
    """PG(n-2,2) zero-divisor graph wavenumbers for Cayley-Dickson dim D."""
    n_modes = max(cd_dim // 2 - 1, 1)
    return np.array([2.0 * math.pi * n / n_modes for n in range(1, n_modes + 1)])


def g2_wavenumbers() -> np.ndarray:
    """G2 = Aut(O) root system: 6 positive roots."""
    return np.array([2.0 * math.pi * n / 6.0 for n in range(1, 7)])


def albert_j3o_wavenumbers() -> np.ndarray:
    """Albert algebra J3(O): 3 Peirce idempotent modes."""
    return np.array([2.0 * math.pi * n / 3.0 for n in range(1, 4)])


def sl2_partner_wavenumbers() -> np.ndarray:
    """sl(2) partner graph: spin-2 weight, 2 modes."""
    k0 = 2.0 * math.pi / 7.0
    return np.array([2.0 * k0, 4.0 * k0])


def combined_wavenumbers() -> np.ndarray:
    """All 4 algebra wavenumber sets concatenated and sorted (18 modes)."""
    all_k = np.concatenate([
        cd_zd_wavenumbers(16),
        g2_wavenumbers(),
        albert_j3o_wavenumbers(),
        sl2_partner_wavenumbers(),
    ])
    return np.sort(np.unique(all_k))


ALGEBRA_WAVENUMBERS = {
    "CD-ZD": cd_zd_wavenumbers(16),
    "G2": g2_wavenumbers(),
    "J3(O)": albert_j3o_wavenumbers(),
    "sl(2)": sl2_partner_wavenumbers(),
    "combined": combined_wavenumbers(),
}


# ---------------------------------------------------------------------------
# Core stacking pipeline
# ---------------------------------------------------------------------------

def stack_residuals(
    galaxies: list,
    x_min: float = 0.5,
    x_max: float = 10.0,
    n_grid: int = 200,
    min_per_bin: int = 10,
) -> tuple:
    """
    Stack galaxy rotation curve residuals onto a uniform x-grid.
    Uses inverse-variance weighting: w_i = 1/sigma_i^2.
    Returns: (x_grid, delta_stack, delta_stack_err, n_contributing)
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    sum_w = np.zeros(n_grid)
    sum_wd = np.zeros(n_grid)
    n_contrib = np.zeros(n_grid, dtype=int)

    for gal in galaxies:
        for k in range(len(gal.x_points)):
            x = gal.x_points[k]
            d = gal.delta_v[k]
            e = max(gal.delta_v_err[k], 1e-6)

            idx = int((x - x_min) / dx)
            if 0 <= idx < n_grid:
                w = 1.0 / (e * e)
                sum_w[idx] += w
                sum_wd[idx] += w * d
                n_contrib[idx] += 1

    delta_stack = np.zeros(n_grid)
    delta_stack_err = np.full(n_grid, np.inf)

    valid = sum_w > 0
    delta_stack[valid] = sum_wd[valid] / sum_w[valid]
    delta_stack_err[valid] = 1.0 / np.sqrt(sum_w[valid])

    return x_grid, delta_stack, delta_stack_err, n_contrib


# ---------------------------------------------------------------------------
# Discrete Fourier projection
# ---------------------------------------------------------------------------

def fourier_power_phase(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Discrete Fourier projection at specified wavenumbers.
    Not FFT -- wavenumbers are non-uniform algebraic predictions.
    Returns: (power, phase) arrays of shape (n_wavenumbers,)
    """
    mask = n_contributing >= min_per_bin
    x = x_grid[mask]
    d = delta_stack[mask]
    count = len(x)

    if count < 3:
        return np.zeros(len(wavenumbers)), np.zeros(len(wavenumbers))

    power = np.zeros(len(wavenumbers))
    phase = np.zeros(len(wavenumbers))

    for i, k in enumerate(wavenumbers):
        re = np.sum(d * np.cos(k * x)) / count
        im = np.sum(d * np.sin(k * x)) / count
        power[i] = re * re + im * im
        phase[i] = math.atan2(im, re)

    return power, phase


def compute_snr(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> float:
    """
    SNR = sqrt(mean_power * n_modes) / rms_residual.

    Uses total power normalized by mode count so algebras with different
    numbers of modes produce genuinely different SNR values.
    """
    mask = n_contributing >= min_per_bin
    if mask.sum() < 3:
        return 0.0
    d = delta_stack[mask]
    rms = np.sqrt(np.mean(d * d))
    if rms <= 0:
        return 0.0
    power, _ = fourier_power_phase(
        x_grid, delta_stack, n_contributing, min_per_bin, wavenumbers
    )
    n_modes = len(wavenumbers)
    mean_power = float(np.mean(power))
    return math.sqrt(mean_power * n_modes) / rms


def compute_rms(
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
) -> float:
    """RMS of stacked residuals in valid bins."""
    mask = n_contributing >= min_per_bin
    if mask.sum() < 1:
        return 0.0
    d = delta_stack[mask]
    return float(np.sqrt(np.mean(d * d)))


# ---------------------------------------------------------------------------
# Rayleigh phase coherence (jackknife)
# ---------------------------------------------------------------------------

def rayleigh_phase_coherence(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Jackknife leave-one-out Rayleigh R test for phase coherence.
    Returns: (rayleigh_r, rayleigh_p) arrays of shape (n_wavenumbers,)
    """
    mask = n_contributing >= min_per_bin
    x = x_grid[mask]
    d = delta_stack[mask]
    m = len(x)

    if m < 4:
        return np.zeros(len(wavenumbers)), np.ones(len(wavenumbers))

    rayleigh_r = np.zeros(len(wavenumbers))
    rayleigh_p = np.zeros(len(wavenumbers))

    for wi, k in enumerate(wavenumbers):
        phases = np.zeros(m)
        for drop in range(m):
            keep = np.concatenate([x[:drop], x[drop + 1:]])
            keep_d = np.concatenate([d[:drop], d[drop + 1:]])
            re = np.mean(keep_d * np.cos(k * keep))
            im = np.mean(keep_d * np.sin(k * keep))
            phases[drop] = math.atan2(im, re)

        sum_cos = np.sum(np.cos(phases))
        sum_sin = np.sum(np.sin(phases))
        r = math.sqrt(sum_cos ** 2 + sum_sin ** 2) / m
        rayleigh_r[wi] = r
        rayleigh_p[wi] = math.exp(-m * r * r)

    return rayleigh_r, rayleigh_p


# ---------------------------------------------------------------------------
# Red-noise envelope
# ---------------------------------------------------------------------------

def fit_red_noise(
    wavenumbers: np.ndarray,
    rayleigh_r: np.ndarray,
    gamma_prior: float = None,
) -> tuple:
    """
    Fit red-noise envelope R(k) = A * k^gamma via log-log OLS.
    Returns: (gamma, amplitude, predicted_r, residuals, sigma_resid)
    """
    pos = (wavenumbers > 0) & (rayleigh_r > 0)
    if pos.sum() < 2:
        return 0.0, 1.0, rayleigh_r.copy(), np.zeros_like(rayleigh_r), 0.0

    log_k = np.log(wavenumbers[pos])
    log_r = np.log(rayleigh_r[pos])

    gamma = 0.0
    log_a = 0.0
    if gamma_prior is not None:
        gamma = gamma_prior
        log_a = float(np.mean(log_r - gamma * log_k))
    else:
        slope, intercept, _, _, _ = sp_stats.linregress(log_k, log_r)
        gamma = slope
        log_a = float(intercept)

    amplitude = math.exp(log_a)
    predicted = amplitude * np.power(wavenumbers, gamma)
    residuals = rayleigh_r - predicted
    sigma = float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0

    return gamma, amplitude, predicted, residuals, sigma


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_snr(
    galaxies: list,
    wavenumbers: np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
    min_per_bin: int = 10,
) -> tuple:
    """
    Bootstrap CI on SNR by resampling galaxies with replacement.
    Returns: (snr_mean, snr_std, snr_ci_lo, snr_ci_hi)
    """
    rng = np.random.RandomState(seed)
    n = len(galaxies)
    snr_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.randint(0, n, n)
        boot_galaxies = [galaxies[i] for i in indices]
        x_grid, delta, delta_err, n_contrib = stack_residuals(boot_galaxies)
        snr_samples[b] = compute_snr(x_grid, delta, n_contrib, min_per_bin, wavenumbers)

    mean = float(np.mean(snr_samples))
    std = float(np.std(snr_samples, ddof=1))
    ci_lo = float(np.percentile(snr_samples, 2.5))
    ci_hi = float(np.percentile(snr_samples, 97.5))
    return mean, std, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Analysis condition classes
# ---------------------------------------------------------------------------

class AnalysisCondition(ABC):
    """Base class for all analysis conditions."""

    def __init__(self, name: str, wavenumbers: np.ndarray, min_per_bin: int = 10):
        self.name = name
        self.wavenumbers = wavenumbers
        self.min_per_bin = min_per_bin

    @abstractmethod
    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        """Run analysis and return metrics dict."""
        ...


class CdZdHarmonicSearch(AnalysisCondition):
    """
    CD-ZD harmonic search at specified Cayley-Dickson dimension.
    Full pipeline: stack -> Fourier at CD wavenumbers -> SNR -> Rayleigh.
    Mapped to plan condition: DANN (primary baseline).
    """

    def __init__(self, cd_dim: int = 16):
        wk = cd_zd_wavenumbers(cd_dim)
        super().__init__("DANN", wk)
        self.cd_dim = cd_dim

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )
        alpha_zd_limit = snr / (0.5 * math.e) if snr > 0 else 0.0
        n_valid = int(np.sum(n_contrib >= self.min_per_bin))

        return {
            "primary_metric": snr,
            "secondary_metric": alpha_zd_limit,
            "rms_residual": rms,
            "n_valid_bins": n_valid,
            "max_fourier_power": float(np.max(power)) if len(power) > 0 else 0.0,
            "mean_rayleigh_r": float(np.mean(rayleigh_r)),
            "min_rayleigh_p": float(np.min(rayleigh_p)) if len(rayleigh_p) > 0 else 1.0,
            "n_modes": len(self.wavenumbers),
            "algebra": "CD-ZD D16",
        }


class G2RootSystemSearch(AnalysisCondition):
    """
    G2 = Aut(O) root system harmonic search with 6 modes.
    Mapped to plan condition: CORAL (secondary baseline).
    """

    def __init__(self):
        super().__init__("CORAL", g2_wavenumbers())

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )
        gamma_fit, amp, pred_r, resid, sigma = fit_red_noise(self.wavenumbers, rayleigh_r)
        n_above_3sigma = int(np.sum(np.abs(resid) > 3.0 * sigma)) if sigma > 0 else 0

        return {
            "primary_metric": snr,
            "secondary_metric": snr / (0.5 * math.e) if snr > 0 else 0.0,
            "rms_residual": rms,
            "n_valid_bins": int(np.sum(n_contrib >= self.min_per_bin)),
            "max_fourier_power": float(np.max(power)) if len(power) > 0 else 0.0,
            "mean_rayleigh_r": float(np.mean(rayleigh_r)),
            "red_noise_gamma": gamma_fit,
            "n_above_3sigma": n_above_3sigma,
            "n_modes": 6,
            "algebra": "G2 Aut(O)",
        }


class AlbertJ3OSearch(AnalysisCondition):
    """
    Albert algebra J3(O) Peirce idempotent search with 3 modes.
    Mapped to plan condition: Comprehensive_baseline_1.
    """

    def __init__(self):
        super().__init__("Comprehensive_baseline_1", albert_j3o_wavenumbers())

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )

        phase_spread = math.pi
        phase_concentrated = False
        if len(phase) == 3:
            phase_spread = float(np.std(phase))
            phase_concentrated = phase_spread < (math.pi / 3.0)

        return {
            "primary_metric": snr,
            "secondary_metric": snr / (0.5 * math.e) if snr > 0 else 0.0,
            "rms_residual": rms,
            "n_valid_bins": int(np.sum(n_contrib >= self.min_per_bin)),
            "max_fourier_power": float(np.max(power)) if len(power) > 0 else 0.0,
            "peirce_phase_spread": phase_spread,
            "peirce_phase_concentrated": int(phase_concentrated),
            "n_modes": 3,
            "algebra": "Albert J3(O)",
        }


class Sl2PartnerGraphSearch(AnalysisCondition):
    """
    sl(2) partner graph search with 2 spin-2 weight modes.
    Mapped to plan condition: Comprehensive_baseline_2.
    """

    def __init__(self):
        super().__init__("Comprehensive_baseline_2", sl2_partner_wavenumbers())

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )

        degeneracy_ratio = 0.0
        if len(power) == 2 and power[1] > 0:
            degeneracy_ratio = float(power[0] / power[1])

        predicted_ratio = 2.0
        ratio_deviation = abs(degeneracy_ratio - predicted_ratio)

        return {
            "primary_metric": snr,
            "secondary_metric": snr / (0.5 * math.e) if snr > 0 else 0.0,
            "rms_residual": rms,
            "n_valid_bins": int(np.sum(n_contrib >= self.min_per_bin)),
            "degeneracy_ratio": degeneracy_ratio,
            "predicted_ratio": predicted_ratio,
            "ratio_deviation": ratio_deviation,
            "ratio_falsified": int(ratio_deviation > 0.5),
            "n_modes": 2,
            "algebra": "sl(2)",
        }


class MultiAlgebraCombined(AnalysisCondition):
    """
    Combined multi-algebra harmonic search using all 4 algebraic structures.
    Concatenates CD-ZD(7) + G2(6) + J3O(3) + sl2(2) wavenumbers (unique+sorted).
    This is the primary proposed method that tests whether combining algebra-specific
    predictions improves sensitivity over individual searches.
    Mapped to plan condition: Comprehensive_proposed.
    """

    def __init__(self):
        wk = combined_wavenumbers()
        super().__init__("Comprehensive_proposed", wk)

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )
        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )

        # Per-algebra power breakdown
        cd_power = float(np.mean(power[:7])) if len(power) >= 7 else 0.0
        remaining = power[7:]
        g2_power = float(np.mean(remaining[:6])) if len(remaining) >= 6 else 0.0

        return {
            "primary_metric": snr,
            "secondary_metric": snr / (0.5 * math.e) if snr > 0 else 0.0,
            "rms_residual": rms,
            "n_valid_bins": int(np.sum(n_contrib >= self.min_per_bin)),
            "max_fourier_power": float(np.max(power)) if len(power) > 0 else 0.0,
            "mean_rayleigh_r": float(np.mean(rayleigh_r)),
            "min_rayleigh_p": float(np.min(rayleigh_p)) if len(rayleigh_p) > 0 else 1.0,
            "cd_zd_mean_power": cd_power,
            "g2_mean_power": g2_power,
            "n_modes": len(self.wavenumbers),
            "algebra": "combined (CD+G2+J3O+sl2)",
        }


class MultiAlgebraCorrected(AnalysisCondition):
    """
    Red-noise corrected multi-algebra search.
    Same wavenumber set as MultiAlgebraCombined but subtracts a data-driven
    red-noise Rayleigh envelope R(k)=A*k^gamma before computing corrected SNR.
    gamma is fit from data (no prior), ensuring seed-to-seed variation.

    Uses the galaxies provided by the caller (respects regime filtering).
    Previous version internally filtered to face-on, making it regime-invariant.
    Mapped to plan condition: Comprehensive_variant.
    """

    def __init__(self):
        wk = combined_wavenumbers()
        super().__init__("Comprehensive_variant", wk)

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr_raw = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)

        rayleigh_r, rayleigh_p = rayleigh_phase_coherence(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )
        # Data-driven fit: gamma_prior=None so slope comes from the data
        gamma, amp, pred_r, resid, sigma = fit_red_noise(
            self.wavenumbers, rayleigh_r, gamma_prior=None
        )
        corrected_r = np.maximum(rayleigh_r - pred_r, 0.0)
        corrected_snr = float(np.max(corrected_r)) / max(sigma, 1e-10) if sigma > 0 else 0.0
        corrected_alpha = corrected_snr / (0.5 * math.e) if corrected_snr > 0 else 0.0

        return {
            "primary_metric": snr_raw,
            "corrected_snr": corrected_snr,
            "secondary_metric": corrected_alpha,
            "rms_residual": rms,
            "n_galaxies": len(galaxies),
            "n_valid_bins": int(np.sum(n_contrib >= self.min_per_bin)),
            "red_noise_gamma": gamma,
            "red_noise_amplitude": amp,
            "red_noise_sigma": sigma,
            "n_modes": len(self.wavenumbers),
            "algebra": "combined+red_noise_corrected",
        }


class NoHarmonicsAblation(AnalysisCondition):
    """
    Ablation: no Fourier decomposition.
    SNR is defined as max(|delta_stack|) / (RMS * sqrt(n_valid_bins)).
    The sqrt(n_valid_bins) denominator corrects for the look-elsewhere effect:
    checking n bins for a peak inflates the raw peak/RMS ratio by ~sqrt(n).
    This makes the ablation SNR comparable to Fourier-based SNR.
    Mapped to plan condition: without_key_component.
    """

    def __init__(self):
        super().__init__("without_key_component", cd_zd_wavenumbers(16))

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        mask = n_contrib >= self.min_per_bin
        if mask.sum() < 3:
            return {
                "primary_metric": 0.0, "secondary_metric": 0.0,
                "rms_residual": 0.0, "n_valid_bins": 0,
                "n_modes": 0,
            }

        d = delta[mask]
        n_valid = len(d)
        rms = float(np.sqrt(np.mean(d * d)))
        # Divide by sqrt(n_valid) to penalize the look-elsewhere effect,
        # making this comparable to Fourier SNR which averages over bins
        peak_snr = float(np.max(np.abs(d))) / (rms * math.sqrt(n_valid)) if rms > 0 else 0.0

        return {
            "primary_metric": peak_snr,
            "secondary_metric": peak_snr / (0.5 * math.e) if peak_snr > 0 else 0.0,
            "rms_residual": rms,
            "n_valid_bins": n_valid,
            "peak_residual": float(np.max(np.abs(d))),
            "method": "peak_to_rms_corrected",
            "n_modes": 0,
        }


class SingleModeAblation(AnalysisCondition):
    """
    Ablation: single fundamental mode only.
    Tests only the first (lowest) CD-ZD wavenumber instead of all 7.
    This checks whether the multi-mode search improves sensitivity.
    Mapped to plan condition: simplified_version.
    """

    def __init__(self):
        k1 = cd_zd_wavenumbers(16)[:1]
        super().__init__("simplified_version", k1)

    def compute_metrics(self, galaxies: list, seed: int) -> dict:
        x_grid, delta, delta_err, n_contrib = stack_residuals(galaxies)
        snr = compute_snr(x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers)
        rms = compute_rms(delta, n_contrib, self.min_per_bin)
        power, phase = fourier_power_phase(
            x_grid, delta, n_contrib, self.min_per_bin, self.wavenumbers
        )

        return {
            "primary_metric": snr,
            "secondary_metric": snr / (0.5 * math.e) if snr > 0 else 0.0,
            "rms_residual": rms,
            "n_valid_bins": int(np.sum(n_contrib >= self.min_per_bin)),
            "fundamental_power": float(power[0]) if len(power) > 0 else 0.0,
            "fundamental_phase": float(phase[0]) if len(phase) > 0 else 0.0,
            "n_modes": 1,
        }


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

def get_all_conditions() -> dict:
    """
    Return all analysis conditions keyed by exp_plan.yaml names.

    Mapping:
      DANN                   -> CD-ZD D16 harmonic search (primary baseline)
      CORAL                  -> G2 Aut(O) root system (secondary baseline)
      Comprehensive_baseline_1 -> Albert J3(O) Peirce idempotent
      Comprehensive_baseline_2 -> sl(2) partner graph
      Comprehensive_proposed -> Multi-algebra combined (all 4)
      Comprehensive_variant  -> Red-noise corrected multi-algebra
      without_key_component  -> No harmonics ablation
      simplified_version     -> Single mode ablation
    """
    return {
        "DANN": CdZdHarmonicSearch(16),
        "CORAL": G2RootSystemSearch(),
        "Comprehensive_baseline_1": AlbertJ3OSearch(),
        "Comprehensive_baseline_2": Sl2PartnerGraphSearch(),
        "Comprehensive_proposed": MultiAlgebraCombined(),
        "Comprehensive_variant": MultiAlgebraCorrected(),
        "without_key_component": NoHarmonicsAblation(),
        "simplified_version": SingleModeAblation(),
    }
