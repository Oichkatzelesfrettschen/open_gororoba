"""
Core analysis functions shared by all 20 conditions.

Stacking variants (IVW mean, median, trimmed, sigma-clipped), Fourier
projection at non-uniform algebraic wavenumbers, SNR, Rayleigh phase
coherence, red-noise envelope fitting, bootstrap CI, whitened DFT,
chi-squared tests, signal injection, per-galaxy DFT, sublevel persistence,
bispectral coherence, KSG mutual information, and Hartigan dip test.
"""

import copy
import math

import numpy as np
from scipy import stats as sp_stats
from scipy.spatial import cKDTree
from scipy.special import digamma

# ---------------------------------------------------------------------------
# Stacking: inverse-variance weighted mean (default)
# ---------------------------------------------------------------------------

def stack_residuals(
    galaxies: list,
    x_min: float = 0.5,
    x_max: float = 1.35,
    n_grid: int = 200,
    min_per_bin: int = 10,
) -> tuple:
    """
    Stack galaxy rotation curve residuals onto a uniform x-grid.
    Uses inverse-variance weighting: w_i = 1/sigma_i^2.
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
# Stacking: per-bin median (robust to outliers)
# ---------------------------------------------------------------------------

def stack_residuals_median(
    galaxies: list,
    x_min: float = 0.5,
    x_max: float = 1.35,
    n_grid: int = 200,
    min_per_bin: int = 10,
    n_bootstrap_err: int = 50,
    seed: int = 42,
) -> tuple:
    """
    Per-bin median stacking. Error estimated via bootstrap of per-bin medians.
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    # Collect values per bin
    bins = [[] for _ in range(n_grid)]
    for gal in galaxies:
        for k in range(len(gal.x_points)):
            x = gal.x_points[k]
            idx = int((x - x_min) / dx)
            if 0 <= idx < n_grid:
                bins[idx].append(gal.delta_v[k])

    delta_stack = np.zeros(n_grid)
    delta_stack_err = np.full(n_grid, np.inf)
    n_contrib = np.zeros(n_grid, dtype=int)

    rng = np.random.RandomState(seed)
    for i in range(n_grid):
        vals = np.array(bins[i])
        n_contrib[i] = len(vals)
        if len(vals) < 1:
            continue
        delta_stack[i] = float(np.median(vals))
        # Bootstrap error on the median
        if len(vals) >= 3 and n_bootstrap_err > 0:
            boot_medians = np.zeros(n_bootstrap_err)
            for b in range(n_bootstrap_err):
                boot_idx = rng.randint(0, len(vals), len(vals))
                boot_medians[b] = np.median(vals[boot_idx])
            delta_stack_err[i] = float(np.std(boot_medians, ddof=1))
        elif len(vals) >= 1:
            delta_stack_err[i] = float(np.std(vals, ddof=1)) / max(math.sqrt(len(vals)), 1.0)

    return x_grid, delta_stack, delta_stack_err, n_contrib


# ---------------------------------------------------------------------------
# Stacking: trimmed mean
# ---------------------------------------------------------------------------

def stack_residuals_trimmed(
    galaxies: list,
    trim_frac: float,
    x_min: float = 0.5,
    x_max: float = 1.35,
    n_grid: int = 200,
    min_per_bin: int = 10,
) -> tuple:
    """
    Per-bin trimmed mean stacking. Returns extra diagnostic: fraction of
    total power removed by trimming.
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    bins = [[] for _ in range(n_grid)]
    for gal in galaxies:
        for k in range(len(gal.x_points)):
            x = gal.x_points[k]
            idx = int((x - x_min) / dx)
            if 0 <= idx < n_grid:
                bins[idx].append(gal.delta_v[k])

    delta_stack = np.zeros(n_grid)
    delta_stack_err = np.full(n_grid, np.inf)
    n_contrib = np.zeros(n_grid, dtype=int)

    total_power_all = 0.0
    total_power_trimmed = 0.0

    for i in range(n_grid):
        vals = np.sort(np.array(bins[i]))
        n_all = len(vals)
        if n_all < 1:
            continue
        total_power_all += float(np.sum(vals ** 2))
        n_trim = int(trim_frac * n_all)
        if n_trim > 0 and 2 * n_trim < n_all:
            trimmed = vals[n_trim:n_all - n_trim]
        else:
            trimmed = vals
        total_power_trimmed += float(np.sum(trimmed ** 2))
        n_contrib[i] = len(trimmed)
        if len(trimmed) >= 1:
            delta_stack[i] = float(np.mean(trimmed))
            if len(trimmed) >= 2:
                delta_stack_err[i] = float(np.std(trimmed, ddof=1)) / math.sqrt(len(trimmed))

    frac_power_trimmed = 1.0 - total_power_trimmed / total_power_all if total_power_all > 0 else 0.0
    return x_grid, delta_stack, delta_stack_err, n_contrib, frac_power_trimmed


# ---------------------------------------------------------------------------
# Stacking: iterative sigma-clipping (Relatores+2019)
# ---------------------------------------------------------------------------

def stack_residuals_clipped(
    galaxies: list,
    sigma_clip: float = 3.0,
    max_iter: int = 5,
    x_min: float = 0.5,
    x_max: float = 1.35,
    n_grid: int = 200,
    min_per_bin: int = 10,
) -> tuple:
    """
    Iterative sigma-clipped weighted mean stacking.
    Returns extra diagnostic: total number of clipped contributions.
    """
    x_grid = np.linspace(x_min, x_max, n_grid)
    dx = x_grid[1] - x_grid[0]

    bins_vals = [[] for _ in range(n_grid)]
    bins_errs = [[] for _ in range(n_grid)]
    for gal in galaxies:
        for k in range(len(gal.x_points)):
            x = gal.x_points[k]
            idx = int((x - x_min) / dx)
            if 0 <= idx < n_grid:
                bins_vals[idx].append(gal.delta_v[k])
                bins_errs[idx].append(max(gal.delta_v_err[k], 1e-6))

    delta_stack = np.zeros(n_grid)
    delta_stack_err = np.full(n_grid, np.inf)
    n_contrib = np.zeros(n_grid, dtype=int)
    total_clipped = 0

    for i in range(n_grid):
        vals = np.array(bins_vals[i])
        errs = np.array(bins_errs[i])
        if len(vals) < 1:
            continue
        n_before = len(vals)
        weights = 1.0 / (errs ** 2)

        for _ in range(max_iter):
            w_sum = np.sum(weights)
            if w_sum <= 0:
                break
            mu = np.sum(weights * vals) / w_sum
            # Weighted standard deviation
            var = np.sum(weights * (vals - mu) ** 2) / w_sum
            sigma = math.sqrt(max(var, 1e-20))
            keep = np.abs(vals - mu) <= sigma_clip * sigma
            if np.all(keep):
                break
            vals = vals[keep]
            errs = errs[keep]
            weights = weights[keep]

        n_contrib[i] = len(vals)
        total_clipped += n_before - len(vals)
        if len(vals) >= 1:
            w_sum = np.sum(1.0 / (errs ** 2))
            delta_stack[i] = np.sum(vals / (errs ** 2)) / w_sum
            delta_stack_err[i] = 1.0 / math.sqrt(w_sum)

    return x_grid, delta_stack, delta_stack_err, n_contrib, total_clipped


# ---------------------------------------------------------------------------
# Discrete Fourier projection at algebraic wavenumbers
# ---------------------------------------------------------------------------

def fourier_power_phase(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Non-FFT discrete Fourier at specified wavenumbers.
    Returns (power, phase) arrays of shape (n_modes,).
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
) -> dict:
    """SNR from Fourier power. Returns snr_mean and snr_max."""
    zero = {"snr_mean": 0.0, "snr_max": 0.0}
    mask = n_contributing >= min_per_bin
    if mask.sum() < 3:
        return zero
    d = delta_stack[mask]
    rms = np.sqrt(np.mean(d * d))
    if rms <= 0:
        return zero
    power, _ = fourier_power_phase(
        x_grid, delta_stack, n_contributing, min_per_bin, wavenumbers
    )
    mean_power = float(np.mean(power))
    max_power = float(np.max(power))
    return {
        "snr_mean": math.sqrt(mean_power) / rms,
        "snr_max": math.sqrt(max_power) / rms,
    }


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
# Rayleigh phase coherence (jackknife leave-one-out)
# ---------------------------------------------------------------------------

def rayleigh_phase_coherence(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Jackknife Rayleigh R test. For each valid bin, drop it and recompute
    DFT phase. R = |mean(exp(i*phi))| measures phase consistency.
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
# Red-noise envelope fitting
# ---------------------------------------------------------------------------

def fit_red_noise(
    wavenumbers: np.ndarray,
    rayleigh_r: np.ndarray,
    gamma_prior: float = None,
) -> tuple:
    """
    Fit R(k) = A * k^gamma via log-log OLS.
    If gamma_prior is given, uses it as fixed gamma.
    """
    pos = (wavenumbers > 0) & (rayleigh_r > 0)
    if pos.sum() < 2:
        return 0.0, 1.0, rayleigh_r.copy(), np.zeros_like(rayleigh_r), 0.0

    log_k = np.log(wavenumbers[pos])
    log_r = np.log(rayleigh_r[pos])

    if gamma_prior is not None:
        gamma = gamma_prior
        log_a = np.mean(log_r - gamma * log_k)
    else:
        slope, intercept, _, _, _ = sp_stats.linregress(log_k, log_r)
        gamma = slope
        log_a = intercept

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
    """Bootstrap CI on SNR by resampling galaxies with replacement."""
    rng = np.random.RandomState(seed)
    n = len(galaxies)
    snr_samples = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        indices = rng.randint(0, n, n)
        boot_galaxies = [galaxies[i] for i in indices]
        x_grid, delta, delta_err, n_contrib = stack_residuals(boot_galaxies)
        snr_samples[b] = compute_snr(
            x_grid, delta, n_contrib, min_per_bin, wavenumbers
        )["snr_mean"]

    mean = float(np.mean(snr_samples))
    std = float(np.std(snr_samples, ddof=1))
    ci_lo = float(np.percentile(snr_samples, 2.5))
    ci_hi = float(np.percentile(snr_samples, 97.5))
    return mean, std, ci_lo, ci_hi


# ---------------------------------------------------------------------------
# Whitened DFT and chi-squared tests
# ---------------------------------------------------------------------------

def whitened_dft(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    delta_stack_err: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Variance-normalized DFT. Accounts for per-bin noise variation (33x in MaNGA).
    Returns (chi2_per_mode, snr_per_mode, total_chi2, dof).
    """
    mask = n_contributing >= min_per_bin
    x = x_grid[mask]
    d = delta_stack[mask]
    sigma = delta_stack_err[mask]
    sigma = np.maximum(sigma, 1e-10)

    n_modes = len(wavenumbers)
    if len(x) < 3:
        return np.zeros(n_modes), np.zeros(n_modes), 0.0, 2 * n_modes

    d_white = d / sigma
    chi2_per_mode = np.zeros(n_modes)
    snr_per_mode = np.zeros(n_modes)

    for i, k in enumerate(wavenumbers):
        cos_k = np.cos(k * x)
        sin_k = np.sin(k * x)
        re_w = np.sum(d_white * cos_k)
        im_w = np.sum(d_white * sin_k)
        var_re = np.sum(cos_k ** 2 / sigma ** 2)
        var_im = np.sum(sin_k ** 2 / sigma ** 2)
        if var_re > 0 and var_im > 0:
            chi2_per_mode[i] = re_w ** 2 / var_re + im_w ** 2 / var_im
            snr_per_mode[i] = math.sqrt(chi2_per_mode[i] / 2.0)

    total_chi2 = float(np.sum(chi2_per_mode))
    dof = 2 * n_modes
    return chi2_per_mode, snr_per_mode, total_chi2, dof


def chi2_survival_p(chi2_val: float, dof: int) -> float:
    """P(X > chi2_val) where X ~ chi2(dof)."""
    if dof <= 0:
        return 1.0
    return float(sp_stats.chi2.sf(chi2_val, dof))


def chi2_gof_flat(
    delta_stack: np.ndarray,
    delta_stack_err: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
) -> tuple:
    """Chi-squared GOF against flat null (delta=0)."""
    mask = n_contributing >= min_per_bin
    d = delta_stack[mask]
    sigma = np.maximum(delta_stack_err[mask], 1e-10)
    n_valid = len(d)
    if n_valid < 1:
        return 0.0, 0.0, 1.0, 0

    chi2 = float(np.sum((d / sigma) ** 2))
    chi2_reduced = chi2 / n_valid
    p_value = chi2_survival_p(chi2, n_valid)
    return chi2, chi2_reduced, p_value, n_valid


# ---------------------------------------------------------------------------
# Signal injection
# ---------------------------------------------------------------------------

def inject_zd_signal(
    galaxies: list,
    alpha_zd: float,
    cd_dim: int = 16,
) -> list:
    """Deep-copy galaxies and inject ZD harmonic signal at alpha_zd amplitude."""
    n_modes = max(cd_dim // 2 - 1, 1)
    af = 0.5
    injected = copy.deepcopy(galaxies)
    for gal in injected:
        for j in range(len(gal.x_points)):
            x = float(gal.x_points[j])
            modulation = 0.0
            for n in range(1, n_modes + 1):
                k_n = 2.0 * math.pi * n / n_modes
                modulation += (af / n) * math.cos(k_n * x) * math.exp(-x)
            gal.delta_v[j] += alpha_zd * modulation
    return injected


# ---------------------------------------------------------------------------
# Per-galaxy DFT (for MI and dip test conditions)
# ---------------------------------------------------------------------------

def per_galaxy_dft(
    galaxies: list,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Compute DFT at algebraic wavenumbers for each galaxy individually.
    Returns (powers, phases) each shape (N, n_modes).
    """
    n_gal = len(galaxies)
    n_modes = len(wavenumbers)
    powers = np.zeros((n_gal, n_modes))
    phases = np.zeros((n_gal, n_modes))

    for i, gal in enumerate(galaxies):
        x = gal.x_points
        d = gal.delta_v
        n = len(x)
        if n < 2:
            continue
        for j, k in enumerate(wavenumbers):
            re = np.sum(d * np.cos(k * x)) / n
            im = np.sum(d * np.sin(k * x)) / n
            powers[i, j] = re * re + im * im
            phases[i, j] = math.atan2(im, re)

    return powers, phases


# ---------------------------------------------------------------------------
# Bispectral coherence at Fano plane triples
# ---------------------------------------------------------------------------

def _is_fano_triple(a: int, b: int, c: int) -> bool:
    """Check if (a, b, c) is a Fano plane triple in PG(2,2) (mod 7)."""
    fano = {
        frozenset({1, 2, 4}), frozenset({2, 3, 5}), frozenset({3, 4, 6}),
        frozenset({4, 5, 7}), frozenset({5, 6, 1}), frozenset({6, 7, 2}),
        frozenset({7, 1, 3}),
    }
    return frozenset({a, b, c}) in fano


def bispectrum_at_triads(
    x_grid: np.ndarray,
    delta_stack: np.ndarray,
    n_contributing: np.ndarray,
    min_per_bin: int,
    wavenumbers: np.ndarray,
) -> tuple:
    """
    Bispectral coherence at wavenumber triads. Classifies each triad as
    Fano or non-Fano. Returns (bicoherence, is_fano, fano_nonfano_ratio).
    """
    mask = n_contributing >= min_per_bin
    x = x_grid[mask]
    d = delta_stack[mask]
    count = len(x)
    n_modes = len(wavenumbers)

    if count < 3:
        return np.array([]), np.array([], dtype=bool), 1.0

    # Compute complex DFT coefficients
    dft = np.zeros(n_modes, dtype=complex)
    for i, k in enumerate(wavenumbers):
        re = np.sum(d * np.cos(k * x)) / count
        im = np.sum(d * np.sin(k * x)) / count
        dft[i] = complex(re, im)

    # Build wavenumber-to-index map (tolerance-based)
    k_to_idx = {}
    for i, k in enumerate(wavenumbers):
        k_to_idx[i] = k

    bicoherence_list = []
    is_fano_list = []

    for i in range(n_modes):
        for j in range(i, n_modes):
            k_sum = wavenumbers[i] + wavenumbers[j]
            # Check if k_sum matches any wavenumber
            for m in range(n_modes):
                if abs(wavenumbers[m] - k_sum) < 0.01:
                    # B(i,j) = F(i) * F(j) * conj(F(m))
                    bispectrum = dft[i] * dft[j] * np.conj(dft[m])
                    denom = abs(dft[i]) ** 2 * abs(dft[j]) ** 2 * abs(dft[m]) ** 2
                    bic = abs(bispectrum) ** 2 / denom if denom > 1e-30 else 0.0
                    bicoherence_list.append(bic)
                    is_fano_list.append(_is_fano_triple(i + 1, j + 1, m + 1))
                    break

    bicoherence = np.array(bicoherence_list) if bicoherence_list else np.array([0.0])
    is_fano = np.array(is_fano_list, dtype=bool) if is_fano_list else np.array([False])

    fano_mask = is_fano
    nonfano_mask = ~is_fano
    fano_mean = float(np.mean(bicoherence[fano_mask])) if fano_mask.any() else 0.0
    nonfano_mean = float(np.mean(bicoherence[nonfano_mask])) if nonfano_mask.any() else 1.0
    ratio = fano_mean / nonfano_mean if nonfano_mean > 1e-30 else 1.0

    return bicoherence, is_fano, ratio


# ---------------------------------------------------------------------------
# Sublevel-set persistence (1D, union-find)
# ---------------------------------------------------------------------------

def sublevel_persistence_1d(
    values: np.ndarray,
    epsilon_frac: float = 0.1,
) -> list:
    """
    1D sublevel persistence via union-find. Returns list of dicts:
    {birth, death, lifetime, persistent}.
    """
    n = len(values)
    if n < 2:
        return []

    epsilon = epsilon_frac * (float(np.max(values)) - float(np.min(values)))
    order = np.argsort(values)

    parent = np.full(n, -1, dtype=int)
    rank = np.zeros(n, dtype=int)
    birth = np.full(n, np.inf)
    visited = np.zeros(n, dtype=bool)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []

    for idx in order:
        parent[idx] = idx
        birth[idx] = values[idx]
        visited[idx] = True

        for neighbor in [idx - 1, idx + 1]:
            if 0 <= neighbor < n and visited[neighbor]:
                root_a = find(idx)
                root_b = find(neighbor)
                if root_a != root_b:
                    # Merge: component born later dies now
                    if birth[root_a] < birth[root_b]:
                        older, younger = root_a, root_b
                    else:
                        older, younger = root_b, root_a

                    death_val = values[idx]
                    lifetime = death_val - birth[younger]
                    pairs.append({
                        "birth": float(birth[younger]),
                        "death": float(death_val),
                        "lifetime": float(lifetime),
                        "persistent": lifetime > epsilon,
                    })

                    if rank[older] < rank[younger]:
                        older, younger = younger, older
                    parent[younger] = older
                    if rank[older] == rank[younger]:
                        rank[older] += 1

    return pairs


# ---------------------------------------------------------------------------
# KSG mutual information estimator
# ---------------------------------------------------------------------------

def embed_phase_circular(phases: np.ndarray) -> np.ndarray:
    """Embed angular data on unit circle: phi -> (cos phi, sin phi)."""
    return np.column_stack([np.cos(phases), np.sin(phases)])


def ksg_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    k: int = 6,
) -> float:
    """
    Kraskov-Stoegbauer-Grassberger MI estimator (Algorithm 1).
    x: shape (N, D1), y: shape (N, D2).
    Returns MI in nats (non-negative).
    """
    n = x.shape[0]
    if n < k + 1:
        return 0.0

    # Joint space
    z = np.hstack([x, y])
    tree_z = cKDTree(z)
    tree_x = cKDTree(x)
    tree_y = cKDTree(y)

    # For each point, find k-th neighbor distance in joint space
    # query k+1 because the point itself is included
    dists, _ = tree_z.query(z, k=k + 1)
    eps = dists[:, -1]  # k-th neighbor distance

    # Count neighbors within eps in marginal spaces
    nx = np.zeros(n, dtype=int)
    ny = np.zeros(n, dtype=int)
    for i in range(n):
        eps_i = eps[i]
        if eps_i <= 0:
            eps_i = 1e-10
        nx[i] = max(tree_x.query_ball_point(x[i], eps_i, return_length=True) - 1, 1)
        ny[i] = max(tree_y.query_ball_point(y[i], eps_i, return_length=True) - 1, 1)

    mi = digamma(k) - float(np.mean(digamma(nx + 1) + digamma(ny + 1))) + digamma(n)
    return max(mi, 0.0)


# ---------------------------------------------------------------------------
# Hartigan dip test (simplified permutation-based)
# ---------------------------------------------------------------------------

def _dip_statistic(sorted_vals: np.ndarray) -> float:
    """
    Compute Hartigan's dip statistic: maximum deviation of empirical CDF
    from the best-fitting unimodal CDF (greatest convex minorant / least
    concave majorant).
    """
    n = len(sorted_vals)
    if n < 4:
        return 0.0

    # Empirical CDF
    ecdf = np.arange(1.0, n + 1.0) / n

    # Greatest convex minorant (GCM) from left
    gcm = np.zeros(n)
    gcm[0] = ecdf[0]
    for i in range(1, n):
        gcm[i] = ecdf[i]
        # Walk back to ensure convexity
        j = i - 1
        while j >= 0 and gcm[j] > gcm[i] - (ecdf[i] - ecdf[j]) * (i - j) / (i - j + 1e-30):
            j -= 1
        if j < i - 1:
            for m in range(j + 1, i):
                t = (m - j) / (i - j)
                gcm[m] = gcm[j] * (1 - t) + gcm[i] * t if j >= 0 else ecdf[0] * (1 - t) + gcm[i] * t

    # Least concave majorant (LCM) from right
    lcm = np.zeros(n)
    lcm[-1] = ecdf[-1]
    for i in range(n - 2, -1, -1):
        lcm[i] = ecdf[i]
        j = i + 1
        while j < n and lcm[j] < lcm[i] + (ecdf[j] - ecdf[i]) * (j - i) / (j - i + 1e-30):
            j += 1
        if j > i + 1:
            for m in range(i + 1, j):
                t = (m - i) / (j - i)
                lcm[m] = lcm[i] * (1 - t) + lcm[j] * t if j < n else lcm[i] * (1 - t) + ecdf[-1] * t

    # Dip = max(LCM - GCM) / 2
    dip = float(np.max(lcm - gcm)) / 2.0
    return max(dip, 0.0)


def hartigan_dip_statistic(
    values: np.ndarray,
    n_perm: int = 5000,
    seed: int = 42,
) -> tuple:
    """
    Hartigan dip test with permutation p-value.
    Returns (dip_statistic, p_value).
    """
    sorted_vals = np.sort(values)
    dip = _dip_statistic(sorted_vals)
    n = len(values)

    rng = np.random.RandomState(seed)
    n_exceed = 0
    mu = float(np.mean(values))
    sigma = float(np.std(values, ddof=1))
    if sigma <= 0:
        return dip, 1.0

    for _ in range(n_perm):
        perm_vals = np.sort(rng.normal(mu, sigma, n))
        d_perm = _dip_statistic(perm_vals)
        if d_perm >= dip:
            n_exceed += 1

    p_value = n_exceed / n_perm
    return dip, p_value


# ---------------------------------------------------------------------------
# Phase reversal test (for trimmed ablations)
# ---------------------------------------------------------------------------

def phase_reversal_test(
    phase_before: np.ndarray,
    phase_after: np.ndarray,
) -> tuple:
    """
    Test if DFT phases reversed after trimming.
    Returns (max_rotation, any_reversed).
    """
    delta = phase_after - phase_before
    # Wrap to [-pi, pi]
    delta = (delta + math.pi) % (2.0 * math.pi) - math.pi
    max_rotation = float(np.max(np.abs(delta)))
    any_reversed = bool(np.any(np.abs(delta) > math.pi / 2.0))
    return max_rotation, any_reversed
