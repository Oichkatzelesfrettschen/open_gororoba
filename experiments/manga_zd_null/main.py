"""
MaNGA IFU Rotation Curve ZD Null Result Experiment
===================================================
Synthetic null experiment: N=6992 galaxies, x=0.5..1.35 r/r_s, 20 radial bins.
Conditions test whether ZD harmonic signals are detectable over baryonic systematics.
Primary metric: SNR (signal-to-noise ratio of ZD detection statistic).
"""

import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExperimentConfig:
    n_galaxies: int = 6992
    n_bins: int = 20
    x_min: float = 0.5
    x_max: float = 1.35
    # CD dimensions to sweep for harmonic basis
    cd_dims: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256, 512, 1024])
    # ZD harmonics per CD dimension (n_modes ~ log2(dim))
    n_seeds: int = 12
    data_seed: int = 999
    device: str = "cpu"
    # NFW scale: r_s uniform in [8, 20] kpc, IFU radius ~12-18 kpc
    rs_min_kpc: float = 8.0
    rs_max_kpc: float = 20.0
    # Inclination range (degrees)
    inc_min_deg: float = 30.0
    inc_max_deg: float = 70.0
    # Velocity dispersion (fractional)
    vel_disp_frac: float = 0.05
    # Baryonic template amplitudes
    disk_amp_mean: float = 0.3
    disk_amp_std: float = 0.1
    bulge_amp_mean: float = 0.1
    bulge_amp_std: float = 0.05
    # ZD signal amplitude (zero = pure null)
    zd_signal_amp: float = 0.0


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Per-seed noise realization for all galaxies."""
    x_bins: torch.Tensor           # (n_bins,)   radial bin centers in r/r_s
    v_nfw: torch.Tensor            # (N, n_bins) NFW circular velocity
    v_disk: torch.Tensor           # (N, n_bins) disk template
    v_bulge: torch.Tensor          # (N, n_bins) bulge template
    v_true: torch.Tensor           # (N, n_bins) true velocity (NFW+baryonic)
    v_obs: torch.Tensor            # (N, n_bins) observed velocity (incl. corrected)
    err_obs: torch.Tensor          # (N, n_bins) per-bin error (incl. corrected)
    mask: torch.Tensor             # (N, n_bins) bool valid bins
    inclinations: np.ndarray       # (N,) degrees
    rs_kpc: np.ndarray             # (N,) NFW scale radii
    disk_amps: np.ndarray          # (N,) disk template amplitudes
    bulge_amps: np.ndarray         # (N,) bulge template amplitudes


def _nfw_circular_velocity(x: torch.Tensor, v200: float = 1.0) -> torch.Tensor:
    """NFW circular velocity profile, normalized so v(x=1)=v200."""
    # v^2(x) = v200^2 * [ln(1+x) - x/(1+x)] / [ln(2) - 0.5] / x
    log_term = torch.log(1.0 + x) - x / (1.0 + x)
    norm = math.log(2.0) - 0.5
    return torch.sqrt((log_term / (norm * x)).clamp(min=1e-8))


def _disk_template(x: torch.Tensor) -> torch.Tensor:
    """Exponential disk rotation curve template."""
    # Freeman disk: v ~ x * exp(-x/2) * (I0 K0 - I1 K1) at x/2
    # Approximation: rises then falls
    y = x / 2.0
    return x * torch.exp(-y) * (1.0 + 0.5 * y)


def _bulge_template(x: torch.Tensor) -> torch.Tensor:
    """Bulge (de Vaucouleurs-like) rotation curve template."""
    return x / (1.0 + x ** 2) ** 0.75


def generate_data_bundle(config: ExperimentConfig, seed: int) -> DataBundle:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    N = config.n_galaxies
    B = config.n_bins
    dev = config.device

    # Radial bins
    x_bins = torch.linspace(config.x_min, config.x_max, B, dtype=torch.float64)

    # Per-galaxy parameters
    inclinations = rng.uniform(config.inc_min_deg, config.inc_max_deg, size=N)
    rs_kpc = rng.uniform(config.rs_min_kpc, config.rs_max_kpc, size=N)
    disk_amps = rng.normal(config.disk_amp_mean, config.disk_amp_std, size=N).clip(0.05, 0.8)
    bulge_amps = rng.normal(config.bulge_amp_mean, config.bulge_amp_std, size=N).clip(0.01, 0.3)

    sin_i = np.sin(np.radians(inclinations)).clip(np.sin(np.radians(30.0)), 1.0)

    # NFW profile (shape: N x B, broadcast)
    v_nfw = _nfw_circular_velocity(x_bins.unsqueeze(0)).expand(N, -1)  # (N, B)

    # Baryonic templates
    v_disk_t = _disk_template(x_bins)      # (B,)
    v_bulge_t = _bulge_template(x_bins)    # (B,)

    d_amp = torch.tensor(disk_amps, dtype=torch.float64).unsqueeze(1)   # (N,1)
    b_amp = torch.tensor(bulge_amps, dtype=torch.float64).unsqueeze(1)  # (N,1)

    v_disk = d_amp * v_disk_t.unsqueeze(0)    # (N, B)
    v_bulge = b_amp * v_bulge_t.unsqueeze(0)  # (N, B)

    v_true = v_nfw + v_disk + v_bulge  # (N, B)

    # LOS projection + noise
    sin_i_t = torch.tensor(sin_i, dtype=torch.float64).unsqueeze(1)  # (N,1)
    v_los = v_true * sin_i_t
    noise_std = config.vel_disp_frac * v_true  # (N, B)
    noise = torch.tensor(
        rng.normal(0.0, 1.0, size=(N, B)), dtype=torch.float64
    ) * noise_std

    v_los_obs = v_los + noise

    # Inclination correction: v_obs = v_los_obs / sin(i)
    v_obs = v_los_obs / sin_i_t
    err_los = noise_std  # measurement error in LOS space
    err_obs = err_los / sin_i_t  # error after inclination correction

    # Mask: all bins valid (uniform coverage for synthetic data)
    mask = torch.ones(N, B, dtype=torch.bool)

    return DataBundle(
        x_bins=x_bins,
        v_nfw=v_nfw,
        v_disk=v_disk,
        v_bulge=v_bulge,
        v_true=v_true,
        v_obs=v_obs,
        err_obs=err_obs,
        mask=mask,
        inclinations=inclinations,
        rs_kpc=rs_kpc,
        disk_amps=disk_amps,
        bulge_amps=bulge_amps,
    )


# ---------------------------------------------------------------------------
# ZD harmonic basis construction
# ---------------------------------------------------------------------------

def build_zd_basis(n_bins: int, cd_dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Gram-Schmidt orthonormal cosine modes as ZD harmonic basis.
    n_modes = max(1, floor(log2(cd_dim)))
    Returns: (n_modes, n_bins) orthonormal basis.
    """
    n_modes = max(1, int(math.log2(cd_dim)))
    n_modes = min(n_modes, n_bins // 2)  # cannot exceed Nyquist
    t = torch.linspace(0, math.pi, n_bins, dtype=torch.float64, device=device)
    # Initial cosine modes
    vecs = []
    for k in range(1, n_modes + 1):
        vecs.append(torch.cos(k * t))
    # Gram-Schmidt
    basis = []
    for v in vecs:
        w = v.clone()
        for b in basis:
            w = w - (w @ b) * b
        nrm = w.norm()
        if nrm > 1e-12:
            basis.append(w / nrm)
    return torch.stack(basis, dim=0)  # (n_modes, n_bins)


# ---------------------------------------------------------------------------
# Profile fitting utilities
# ---------------------------------------------------------------------------

def _wls_nfw(v_obs: torch.Tensor, v_nfw: torch.Tensor,
             err_obs: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    WLS fit: scale NFW template to each galaxy.
    Returns: (scale, raw_residuals) both (N, B).
    """
    w = mask.float() / (err_obs ** 2 + 1e-12)
    num = (w * v_obs * v_nfw).sum(dim=1, keepdim=True)
    den = (w * v_nfw ** 2).sum(dim=1, keepdim=True)
    scale = num / (den + 1e-12)
    v_fit = scale * v_nfw
    raw_residuals = (v_obs - v_fit) * mask.float()
    return scale, raw_residuals


def _wls_baryonic(
    v_obs: torch.Tensor,
    v_nfw: torch.Tensor,
    v_disk: torch.Tensor,
    v_bulge: torch.Tensor,
    err_obs: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    WLS fit: NFW + disk + bulge templates via Cramer's rule (3x3).
    Returns: raw_residuals (N, B).
    """
    w = mask.float() / (err_obs ** 2 + 1e-12)
    templates = [v_nfw, v_disk, v_bulge]
    # Build 3x3 normal equations for each galaxy
    # A_ij = sum_b w_b * T_i_b * T_j_b,  rhs_i = sum_b w_b * v_obs_b * T_i_b
    M = 3
    A = torch.zeros(v_obs.shape[0], M, M, dtype=torch.float64)
    rhs = torch.zeros(v_obs.shape[0], M, dtype=torch.float64)
    for i, ti in enumerate(templates):
        rhs[:, i] = (w * v_obs * ti).sum(dim=1)
        for j, tj in enumerate(templates):
            A[:, i, j] = (w * ti * tj).sum(dim=1)

    # Solve via torch.linalg.solve (batched)
    # Add small regularization to avoid singular matrices
    eye = torch.eye(M, dtype=torch.float64).unsqueeze(0) * 1e-8
    coeffs = torch.linalg.solve(A + eye, rhs)  # (N, 3)

    v_fit = (
        coeffs[:, 0:1] * v_nfw
        + coeffs[:, 1:2] * v_disk
        + coeffs[:, 2:3] * v_bulge
    )
    return (v_obs - v_fit) * mask.float()


# ---------------------------------------------------------------------------
# Base condition
# ---------------------------------------------------------------------------

class BaseCondition:
    """Abstract base for all experimental conditions."""

    name: str = "base"
    description: str = ""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = config.device

    # ------------------------------------------------------------------
    # Physical metric helpers
    # ------------------------------------------------------------------

    def compute_rms_physical(
        self,
        raw_residuals: torch.Tensor,
        v_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        RMS of fractional residuals: sqrt(mean((res/|v_obs|)^2)).
        Normalization by per-galaxy |v_obs|.mean prevents scale bias.
        """
        v_scale = v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (raw_residuals / v_scale) * mask.float()
        sq = frac ** 2
        n_valid = mask.float().sum(dim=1).clamp(min=1.0)
        rms_per_gal = (sq.sum(dim=1) / n_valid).sqrt()
        return float(rms_per_gal.mean())

    def compute_detection_threshold_physical(
        self,
        raw_residuals: torch.Tensor,
        v_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> float:
        """
        Detection threshold = 3 * max_bin(SEM of per-galaxy fractional residuals).
        SEM_b = std_{galaxies}(frac_{i,b}) / sqrt(N_b).
        This reflects the sensitivity limit from galaxy-to-galaxy scatter.
        """
        v_scale = v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (raw_residuals / v_scale) * mask.float()
        N_b = mask.float().sum(dim=0).clamp(min=2.0)  # (B,)
        std_b = frac.std(dim=0)                         # (B,)
        sem_b = std_b / N_b.sqrt()                      # (B,)
        return float(3.0 * sem_b.max())

    def _subtract_harmonic_projection_population(
        self,
        raw_residuals: torch.Tensor,
        v_obs: torch.Tensor,
        mask: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Subtract population-mean harmonic pattern from all galaxies.
        NOTE: does NOT change per-galaxy variance or SEM-based threshold.
        Useful for RMS reduction (removes coherent bias), not threshold.
        """
        v_scale = v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (raw_residuals / v_scale) * mask.float()
        # Population-mean fractional residual
        mean_frac = frac.mean(dim=0, keepdim=True)  # (1, B)
        # Project mean onto basis
        coeffs = torch.einsum("b,kb->k", mean_frac[0], basis)  # (n_modes,)
        mean_proj = torch.einsum("k,kb->b", coeffs, basis)     # (B,)
        # Subtract same pattern from all galaxies
        frac_clean = frac - mean_proj.unsqueeze(0)
        return frac_clean * v_scale

    def _subtract_harmonic_per_galaxy(
        self,
        raw_residuals: torch.Tensor,
        v_obs: torch.Tensor,
        mask: torch.Tensor,
        basis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-galaxy harmonic projection subtraction.
        Reduces within-galaxy variance by factor (1 - n_modes/n_bins)
        via orthogonal projector trace identity tr(I-P)/n_bins = 1 - n_modes/n_bins.
        This CHANGES the SEM-based threshold (unlike population-mean subtraction).
        Returns residuals in velocity units.
        """
        v_scale = v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (raw_residuals / v_scale) * mask.float()
        # Per-galaxy projection onto basis
        coeffs_per_gal = torch.einsum("nb,kb->nk", frac, basis)  # (N, n_modes)
        proj_per_gal = torch.einsum("nk,kb->nb", coeffs_per_gal, basis)  # (N, B)
        post_frac = (frac - proj_per_gal) * mask.float()
        return post_frac * v_scale

    def _remove_g2_per_galaxy(
        self,
        raw_residuals: torch.Tensor,
        v_obs: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Per-galaxy G2 Aut(O) mode removal: zero out rfft bins k=1..7.
        G2 has 7 roots; zeroing 7 complex bins removes 14/20 DOF in frequency space,
        leaving variance fraction ~ (n_bins - 14) / n_bins = 6/20 = 0.3.
        Returns residuals in velocity units.
        """
        v_scale = v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (raw_residuals / v_scale) * mask.float()

        n_bins = frac.shape[1]
        fft_g = torch.fft.rfft(frac, dim=1)           # (N, n_bins//2+1) complex
        n_rfft = fft_g.shape[1]

        # G2 selector: zero out k=1..7 (G2 Aut(O) 7 roots)
        g2_keep = torch.ones(n_rfft, dtype=torch.float64, device=self.device)
        for k in range(1, min(8, n_rfft)):
            g2_keep[k] = 0.0

        non_g2 = torch.fft.irfft(fft_g * g2_keep[None, :], n=n_bins)  # (N, B)
        return (non_g2 * mask.float()) * v_scale

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        raise NotImplementedError

    def compute_snr(self, residuals: torch.Tensor, basis: torch.Tensor,
                    mask: torch.Tensor, v_obs: torch.Tensor) -> float:
        """
        ZD detection SNR: ratio of projected signal power to noise floor.
        signal = |mean_gal(proj onto basis)| / std_gal(proj onto basis)
        """
        v_scale = v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (residuals / v_scale) * mask.float()
        coeffs = torch.einsum("nb,kb->nk", frac, basis)  # (N, n_modes)
        signal_power = coeffs.mean(dim=0)                  # (n_modes,)
        noise_power = coeffs.std(dim=0).clamp(min=1e-12)  # (n_modes,)
        snr_per_mode = (signal_power / noise_power).abs()
        return float(snr_per_mode.max())


# ---------------------------------------------------------------------------
# Conditions
# ---------------------------------------------------------------------------

class NfwBaseline(BaseCondition):
    """
    Reference: NFW-only fit (incomplete model).
    Has large coherent baryonic residuals -- NOT used in ZD detection gate.
    Purpose: show magnitude of baryonic systematic if ignored.
    """
    name = "nfw_baseline"
    description = "NFW-only WLS fit; baryonic residuals retained as reference"

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        _, raw_residuals = _wls_nfw(bundle.v_obs, bundle.v_nfw, bundle.err_obs, bundle.mask)
        basis = build_zd_basis(self.config.n_bins, 64, self.device)
        snr = self.compute_snr(raw_residuals, basis, bundle.mask, bundle.v_obs)
        rms = self.compute_rms_physical(raw_residuals, bundle.v_obs, bundle.mask)
        thr = self.compute_detection_threshold_physical(raw_residuals, bundle.v_obs, bundle.mask)
        return {"SNR": snr, "RMS_mean": rms, "detection_threshold_mean": thr}


class BaryonicBaseline(BaseCondition):
    """
    Full 3-component WLS (NFW + disk + bulge). Baseline for ZD search.
    """
    name = "baryonic_baseline"
    description = "Full 3-template WLS; baseline for all ZD detections"

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        raw_residuals = _wls_baryonic(
            bundle.v_obs, bundle.v_nfw, bundle.v_disk, bundle.v_bulge,
            bundle.err_obs, bundle.mask
        )
        basis = build_zd_basis(self.config.n_bins, 64, self.device)
        snr = self.compute_snr(raw_residuals, basis, bundle.mask, bundle.v_obs)
        rms = self.compute_rms_physical(raw_residuals, bundle.v_obs, bundle.mask)
        thr = self.compute_detection_threshold_physical(raw_residuals, bundle.v_obs, bundle.mask)
        return {"SNR": snr, "RMS_mean": rms, "detection_threshold_mean": thr}


class HarmonicHaloStack(BaseCondition):
    """
    ZD harmonic subtraction at CD dimension 1024 (10 modes).
    RMS: population-mean ZD pattern subtracted (removes coherent bias).
    Threshold: per-galaxy D=1024 subtraction -> variance x (1 - 10/20) = x0.5
               -> threshold x sqrt(0.5) ~ 0.707 of baryonic baseline.
    """
    name = "harmonic_halo_stack"
    description = "D=1024 ZD per-galaxy subtraction; 10-mode orthonormal basis"

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        raw_residuals = _wls_baryonic(
            bundle.v_obs, bundle.v_nfw, bundle.v_disk, bundle.v_bulge,
            bundle.err_obs, bundle.mask
        )
        basis_1024 = build_zd_basis(self.config.n_bins, 1024, self.device)

        # RMS: population-mean subtraction (removes coherent ZD bias)
        res_pop = self._subtract_harmonic_projection_population(
            raw_residuals, bundle.v_obs, bundle.mask, basis_1024
        )
        rms = self.compute_rms_physical(res_pop, bundle.v_obs, bundle.mask)

        # SNR on population-cleaned residuals
        snr = self.compute_snr(res_pop, basis_1024, bundle.mask, bundle.v_obs)

        # Threshold: per-galaxy subtraction changes within-galaxy scatter
        res_per_gal = self._subtract_harmonic_per_galaxy(
            raw_residuals, bundle.v_obs, bundle.mask, basis_1024
        )
        thr = self.compute_detection_threshold_physical(res_per_gal, bundle.v_obs, bundle.mask)

        return {"SNR": snr, "RMS_mean": rms, "detection_threshold_mean": thr}


class MultiAlgebraDFT(BaseCondition):
    """
    G2 Aut(O) DFT selector: zero out rfft bins k=1..7 per galaxy.
    14/20 DOF removed -> variance fraction ~0.3 -> threshold x sqrt(0.3) ~ 0.548.
    RMS uses population-mean G2-component subtraction.
    """
    name = "multi_algebra_dft"
    description = "G2 Aut(O) 7-root DFT mode removal; per-galaxy variance reduction"

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        raw_residuals = _wls_baryonic(
            bundle.v_obs, bundle.v_nfw, bundle.v_disk, bundle.v_bulge,
            bundle.err_obs, bundle.mask
        )

        # Per-galaxy G2 removal for threshold
        res_g2 = self._remove_g2_per_galaxy(raw_residuals, bundle.v_obs, bundle.mask)
        thr = self.compute_detection_threshold_physical(res_g2, bundle.v_obs, bundle.mask)

        # For RMS and SNR: use G2-only component (what was removed = G2 signal)
        # G2 component = raw_residuals - non_g2_component
        v_scale = bundle.v_obs.abs().mean(dim=1, keepdim=True).clamp(min=1e-6)
        frac = (raw_residuals / v_scale) * bundle.mask.float()
        n_bins = frac.shape[1]
        fft_g = torch.fft.rfft(frac, dim=1)
        n_rfft = fft_g.shape[1]
        g2_keep = torch.ones(n_rfft, dtype=torch.float64, device=self.device)
        for k in range(1, min(8, n_rfft)):
            g2_keep[k] = 0.0
        non_g2_frac = torch.fft.irfft(fft_g * g2_keep[None, :], n=n_bins)
        g2_component_frac = frac - non_g2_frac
        g2_res = (g2_component_frac * bundle.mask.float()) * v_scale

        rms = self.compute_rms_physical(res_g2, bundle.v_obs, bundle.mask)

        # SNR: project G2 component onto a G2-compatible basis
        basis = build_zd_basis(self.config.n_bins, 64, self.device)
        snr = self.compute_snr(g2_res, basis, bundle.mask, bundle.v_obs)

        return {"SNR": snr, "RMS_mean": rms, "detection_threshold_mean": thr}


class NoInclinationAblation(BaseCondition):
    """
    Ablation: inclination correction removed (use raw LOS velocities).
    Threshold boosted by inclination scatter penalty:
      thr = thr_base * (1 + std(1/sin(i)))
    This quantifies how much inclination uncertainty inflates the noise floor.
    """
    name = "no_inclination_ablation"
    description = "LOS-only fit; threshold penalized by inclination scatter"

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        # Reconstruct LOS velocity (undo inclination correction)
        sin_i = np.sin(np.radians(bundle.inclinations)).clip(
            np.sin(np.radians(self.config.inc_min_deg)), 1.0
        )
        sin_i_t = torch.tensor(sin_i, dtype=torch.float64).unsqueeze(1)

        v_los = bundle.v_obs * sin_i_t
        err_los = bundle.err_obs * sin_i_t
        v_nfw_los = bundle.v_nfw * sin_i_t
        v_disk_los = bundle.v_disk * sin_i_t
        v_bulge_los = bundle.v_bulge * sin_i_t

        raw_residuals_los = _wls_baryonic(
            v_los, v_nfw_los, v_disk_los, v_bulge_los, err_los, bundle.mask
        )

        basis = build_zd_basis(self.config.n_bins, 64, self.device)
        snr = self.compute_snr(raw_residuals_los, basis, bundle.mask, v_los)
        rms = self.compute_rms_physical(raw_residuals_los, v_los, bundle.mask)

        # Base threshold from LOS residuals
        thr_base = self.compute_detection_threshold_physical(raw_residuals_los, v_los, bundle.mask)

        # Inclination scatter penalty: std(1/sin(i)) over the galaxy sample
        inv_sin = 1.0 / sin_i
        inc_penalty = float(np.std(inv_sin))
        thr = thr_base * (1.0 + inc_penalty)

        return {"SNR": snr, "RMS_mean": rms, "detection_threshold_mean": thr}


class SingleCDDimAblation(BaseCondition):
    """
    Ablation: low CD dimension D=16 (4 modes) instead of D=1024 (10 modes).
    Per-galaxy subtraction removes only 4/20 DOF -> variance x (1 - 4/20) = x0.8
    -> threshold x sqrt(0.8) ~ 0.894 of baryonic baseline.
    Contrast with HarmonicHaloStack (0.707) shows benefit of higher CD dimension.
    """
    name = "single_cd_dim_ablation"
    description = "D=16 ZD basis (4 modes); per-galaxy subtraction ablation"

    def evaluate(self, bundle: DataBundle) -> Dict[str, float]:
        raw_residuals = _wls_baryonic(
            bundle.v_obs, bundle.v_nfw, bundle.v_disk, bundle.v_bulge,
            bundle.err_obs, bundle.mask
        )
        basis_16 = build_zd_basis(self.config.n_bins, 16, self.device)
        basis_64 = build_zd_basis(self.config.n_bins, 64, self.device)

        # RMS: population-mean subtraction with D=16 basis
        res_pop = self._subtract_harmonic_projection_population(
            raw_residuals, bundle.v_obs, bundle.mask, basis_16
        )
        rms = self.compute_rms_physical(res_pop, bundle.v_obs, bundle.mask)
        snr = self.compute_snr(res_pop, basis_64, bundle.mask, bundle.v_obs)

        # Threshold: per-galaxy D=16 subtraction (4 modes out of 20 bins)
        res_per_gal = self._subtract_harmonic_per_galaxy(
            raw_residuals, bundle.v_obs, bundle.mask, basis_16
        )
        thr = self.compute_detection_threshold_physical(res_per_gal, bundle.v_obs, bundle.mask)

        return {"SNR": snr, "RMS_mean": rms, "detection_threshold_mean": thr}


# ---------------------------------------------------------------------------
# Condition registry
# ---------------------------------------------------------------------------

ALL_CONDITIONS = [
    NfwBaseline,
    BaryonicBaseline,
    HarmonicHaloStack,
    MultiAlgebraDFT,
    NoInclinationAblation,
    SingleCDDimAblation,
]


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------

def run_condition(
    cond_cls,
    config: ExperimentConfig,
    seeds: List[int],
) -> Dict[str, float]:
    """Run a condition over multiple seeds, compute mean/std of each metric."""
    cond = cond_cls(config)
    per_seed: Dict[str, List[float]] = {}

    for seed in seeds:
        # Fresh data bundle per seed (different noise realization)
        bundle = generate_data_bundle(config, seed)
        metrics = cond.evaluate(bundle)
        for k, v in metrics.items():
            per_seed.setdefault(k, []).append(v)

    result: Dict[str, float] = {}
    for k, vals in per_seed.items():
        result[f"{k}_mean"] = float(np.mean(vals))
        result[f"{k}_std"] = float(np.std(vals))
    return result


def assess_null_result(results: Dict[str, Dict[str, float]]) -> Dict[str, object]:
    """
    ZD detection gate: SNR < 3 across all ZD-sensitive conditions.
    nfw_baseline excluded (incomplete model with known coherent residuals).
    """
    zd_conditions = [k for k in results if k != "nfw_baseline"]
    max_snr = max(results[k].get("SNR_mean", 0.0) for k in zd_conditions)
    detected = max_snr >= 3.0
    return {
        "null_result": not detected,
        "max_SNR": max_snr,
        "detected_conditions": [
            k for k in zd_conditions if results[k].get("SNR_mean", 0.0) >= 3.0
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    config = ExperimentConfig()
    seeds = list(range(config.data_seed, config.data_seed + config.n_seeds))

    print(f"Running {len(ALL_CONDITIONS)} conditions over {len(seeds)} seeds...")
    print(f"N_galaxies={config.n_galaxies}, N_bins={config.n_bins}, "
          f"x=[{config.x_min},{config.x_max}]")
    print()

    results: Dict[str, Dict[str, float]] = {}
    for cond_cls in ALL_CONDITIONS:
        cond = cond_cls(config)
        print(f"  [{cond.name}] {cond.description}")
        metrics = run_condition(cond_cls, config, seeds)
        results[cond.name] = metrics
        snr = metrics.get("SNR_mean", float("nan"))
        rms = metrics.get("RMS_mean_mean", float("nan"))
        thr = metrics.get("detection_threshold_mean_mean", float("nan"))
        snr_s = metrics.get("SNR_std", float("nan"))
        print(f"    SNR={snr:.3f} +/- {snr_s:.3f}  "
              f"RMS={rms:.5f}  threshold={thr:.6f}")

    print()
    assessment = assess_null_result({k: {
        "SNR_mean": v.get("SNR_mean", 0.0)
    } for k, v in results.items()})

    print("=== NULL RESULT ASSESSMENT ===")
    print(f"  Null result: {assessment['null_result']}")
    print(f"  Max SNR (ZD conditions): {assessment['max_SNR']:.3f}")
    if assessment["detected_conditions"]:
        print(f"  Detected conditions: {assessment['detected_conditions']}")
    else:
        print("  No ZD signal detected above SNR=3 threshold.")

    print()
    print("=== DETECTION THRESHOLD SUMMARY ===")
    print(f"  {'Condition':<30}  {'threshold_mean':>16}  {'threshold_std':>14}")
    for cond_cls in ALL_CONDITIONS:
        k = cond_cls.name
        tm = results[k].get("detection_threshold_mean_mean", float("nan"))
        ts = results[k].get("detection_threshold_mean_std", float("nan"))
        print(f"  {k:<30}  {tm:>16.6f}  {ts:>14.6f}")

    return results, assessment


if __name__ == "__main__":
    main()
