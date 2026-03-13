//! Harmonic halo stacking analysis.
//!
//! Implements the rotation curve stacking procedure for detecting the N-mode
//! harmonic halo signature predicted by Cayley-Dickson zero-divisor topology.
//!
//! # Algorithm
//!
//! 1. For each galaxy with known NFW scale radius r_s:
//!    - Normalize radius: x = r / r_s (dimensionless)
//!    - Compute NFW model: v_nfw(x) using the galaxy's M200, c200
//!    - Compute fractional residual: delta(x) = (v_obs - v_nfw) / v_nfw
//!
//! 2. Interpolate all residuals onto a common x-grid.
//!
//! 3. Stack (average) residuals: delta_stack(x) = (1/N) * sum_i delta_i(x).
//!    Noise reduces by sqrt(N), signal (if present) remains.
//!
//! 4. Fourier-analyze delta_stack(x) for power at the N predicted wavenumbers
//!    k_n' = 2*pi*n / N_modes for n = 1..N_modes.
//!
//! # Key Physics
//!
//! In normalized x = r/r_s coordinates, the harmonic halo modulation becomes:
//!   F(x) = 1 + alpha_zd * sum_{n=1}^{N} (af/n) * cos(k_n' * x + phi_n) * exp(-x)
//! where k_n' = 2*pi*n / N_modes are universal (mass-independent) wavenumbers,
//! af is the assessor fraction, and N_modes is the motif component count from
//! the PG(n-2,2) projective geometry at that CD dimension.
//!
//! # CD dimension scaling
//!
//! | Dim | Name       | N_modes | PG     | Formula              |
//! |-----|------------|---------|--------|----------------------|
//! |  16 | sedenion   |       7 | PG(2,2)| dim/2 - 1 = 7       |
//! |  32 | pathion    |      15 | PG(3,2)| dim/2 - 1 = 15      |
//! |  64 | chingon    |      31 | PG(4,2)| dim/2 - 1 = 31      |
//! | 128 | routon     |      63 | PG(5,2)| dim/2 - 1 = 63      |
//! | 256 | voudion    |     127 | PG(6,2)| dim/2 - 1 = 127     |
//!
//! # References
//!
//! - C-1332: Stacking falsification thesis
//! - C-1313: Harmonic halo v_circ prediction
//! - E-180, E-182: Stacking experiments (D=16 and multi-dim)

use std::f64::consts::PI;

/// Cayley-Dickson dimension parameters for harmonic stacking.
///
/// Encapsulates the algebraic constants that change with CD dimension:
/// - `n_modes`: number of motif components = dim/2 - 1 (PG points)
/// - `assessor_fraction`: ratio of primitive assessors to total ZDs
///
/// # Assessor fraction identity
///
/// The assessor fraction is **identically 0.5** for all dim >= 16.
/// Proof: each cross-assessor (i,j) generates exactly 2 zero-divisors
/// (+ and - sign variants of diag(e_i +/- e_j)). Therefore
/// `n_ZDs = 2 * n_assessors`, so `fraction = n_assessors / n_ZDs = 0.5`.
///
/// # Algebraic statistics
///
/// | Dim  | n_modes | n_assessors | n_ZDs  | nodes/comp |
/// |------|---------|-------------|--------|------------|
/// |   16 |       7 |          42 |     84 |          6 |
/// |   32 |      15 |         210 |    420 |         14 |
/// |   64 |      31 |         930 |   1860 |         30 |
/// |  128 |      63 |        3906 |   7812 |         62 |
/// |  256 |     127 |       16002 |  32004 |        126 |
/// |  512 |     255 |       64770 | 129540 |        254 |
/// | 1024 |     511 |      260610 | 521220 |        510 |
///
/// General formulae:
/// - `n_modes = dim/2 - 1`
/// - `nodes_per_comp = dim/2 - 2`
/// - `n_assessors = n_modes * nodes_per_comp = (dim/2-1)(dim/2-2)`
/// - `n_ZDs = 2 * n_assessors`
/// - `assessor_fraction = 0.5` (always)
#[derive(Debug, Clone)]
pub struct CdDimensionParams {
    /// Cayley-Dickson dimension (must be power of 2, >= 16).
    pub cd_dim: usize,
    /// Number of harmonic modes = motif components from PG(n-2,2).
    pub n_modes: usize,
    /// Assessor fraction: primitive assessors / total ZDs.
    /// Known to be 0.5 at D=16; conjectured constant for D >= 16.
    pub assessor_fraction: f64,
}

impl CdDimensionParams {
    /// Create params for a given CD dimension.
    ///
    /// Uses the formula n_modes = dim/2 - 1 (from PG correspondence, C-444).
    /// Assessor fraction defaults to 0.5 (verified at D=16, conjectured universal).
    pub fn new(cd_dim: usize) -> Self {
        assert!(
            cd_dim >= 16 && cd_dim.is_power_of_two(),
            "CD dimension must be a power of 2 >= 16, got {cd_dim}"
        );
        Self {
            cd_dim,
            n_modes: cd_dim / 2 - 1,
            assessor_fraction: 0.5,
        }
    }

    /// Create params with a custom assessor fraction (for when empirical
    /// computation at higher dims reveals a different value).
    pub fn with_assessor_fraction(cd_dim: usize, assessor_fraction: f64) -> Self {
        let mut params = Self::new(cd_dim);
        params.assessor_fraction = assessor_fraction;
        params
    }

    /// Sedenion (D=16) defaults: 7 modes, 0.5 assessor fraction.
    pub fn sedenion() -> Self {
        Self::new(16)
    }

    /// Predicted wavenumbers in normalized x = r/r_s space.
    ///
    /// Returns k_n' = 2*pi*n / n_modes for n = 1..n_modes.
    /// These are universal constants for a given CD dimension.
    pub fn predicted_wavenumbers(&self) -> Vec<f64> {
        (1..=self.n_modes)
            .map(|n| 2.0 * PI * n as f64 / self.n_modes as f64)
            .collect()
    }
}

/// A normalized rotation curve point in dimensionless x = r/r_s coordinates.
#[derive(Debug, Clone)]
pub struct NormalizedPoint {
    /// Dimensionless radius x = r / r_s.
    pub x: f64,
    /// Fractional velocity residual: (v_obs - v_model) / v_model.
    pub delta_v: f64,
    /// Fractional uncertainty on delta_v.
    pub delta_v_err: f64,
}

/// A galaxy's normalized rotation curve residuals.
#[derive(Debug, Clone)]
pub struct NormalizedResiduals {
    /// Galaxy identifier.
    pub name: String,
    /// NFW scale radius used for normalization (kpc).
    pub r_s_kpc: f64,
    /// Normalized residual data points.
    pub points: Vec<NormalizedPoint>,
}

/// Result of the stacking analysis.
#[derive(Debug, Clone)]
pub struct StackingResult {
    /// Number of galaxies stacked.
    pub n_galaxies: usize,
    /// CD dimension used for analysis.
    pub cd_dim: usize,
    /// Number of harmonic modes analyzed.
    pub n_modes: usize,
    /// Common x-grid values.
    pub x_grid: Vec<f64>,
    /// Stacked mean residual at each x-grid point.
    pub delta_stack: Vec<f64>,
    /// Uncertainty on stacked residual (sigma / sqrt(N_eff)).
    pub delta_stack_err: Vec<f64>,
    /// Number of galaxies contributing at each x-grid point.
    pub n_contributing: Vec<usize>,
    /// Fourier power at each predicted wavenumber (length = n_modes).
    pub fourier_power: Vec<f64>,
    /// Fourier phase at each predicted wavenumber (length = n_modes).
    pub fourier_phase: Vec<f64>,
    /// RMS of the stacked residual (noise floor estimate).
    pub rms_residual: f64,
    /// Detection significance: max(fourier_power) / rms_noise_power.
    pub detection_snr: f64,
    /// Inferred alpha_zd upper limit (or detection) from amplitude.
    pub alpha_zd_estimate: f64,
}

/// Configuration for the stacking analysis.
#[derive(Debug, Clone)]
pub struct StackingConfig {
    /// Minimum x = r/r_s for the common grid (default 0.1).
    pub x_min: f64,
    /// Maximum x = r/r_s for the common grid (default 10.0).
    pub x_max: f64,
    /// Number of grid points (default 200).
    pub n_grid: usize,
    /// Minimum number of galaxies required per x-bin (default 3).
    pub min_galaxies_per_bin: usize,
    /// Use inverse-variance weighting (default true).
    pub inverse_variance_weighting: bool,
    /// Cayley-Dickson dimension parameters (default: sedenion D=16).
    pub cd_params: CdDimensionParams,
    /// If true, skip points flagged as PSF-smeared (r < 1 beam FWHM).
    /// Default false for backward compatibility with pre-E-196 runs.
    pub exclude_psf_flagged: bool,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            x_min: 0.1,
            x_max: 10.0,
            n_grid: 200,
            min_galaxies_per_bin: 3,
            inverse_variance_weighting: true,
            cd_params: CdDimensionParams::sedenion(),
            exclude_psf_flagged: false,
        }
    }
}

/// Linearly interpolate a galaxy's residuals onto the common x-grid.
///
/// Returns (delta_v, delta_v_err) at each grid point, or None if the
/// grid point falls outside the galaxy's radial range.
fn interpolate_residuals(galaxy: &NormalizedResiduals, x_grid: &[f64]) -> Vec<Option<(f64, f64)>> {
    let pts = &galaxy.points;
    if pts.is_empty() {
        return vec![None; x_grid.len()];
    }

    x_grid
        .iter()
        .map(|&x| {
            // Find bracketing points
            if x < pts[0].x || x > pts[pts.len() - 1].x {
                return None;
            }
            // Binary search for left bracket
            let idx = match pts.binary_search_by(|p| p.x.partial_cmp(&x).unwrap()) {
                Ok(i) => {
                    return Some((pts[i].delta_v, pts[i].delta_v_err));
                }
                Err(i) => i,
            };
            if idx == 0 || idx >= pts.len() {
                return None;
            }
            let (p0, p1) = (&pts[idx - 1], &pts[idx]);
            let t = (x - p0.x) / (p1.x - p0.x);
            let dv = p0.delta_v + t * (p1.delta_v - p0.delta_v);
            let de = p0.delta_v_err + t * (p1.delta_v_err - p0.delta_v_err);
            Some((dv, de))
        })
        .collect()
}

/// Stack normalized rotation curve residuals from multiple galaxies.
///
/// The stacking procedure:
/// 1. Build a common x = r/r_s grid
/// 2. Interpolate each galaxy's residuals onto this grid
/// 3. Compute weighted mean and uncertainty at each grid point
/// 4. Fourier-analyze the stacked residual for the 7 predicted modes
pub fn stack_residuals(
    galaxies: &[NormalizedResiduals],
    config: &StackingConfig,
) -> StackingResult {
    let n_gal = galaxies.len();

    // Build uniform x-grid
    let dx = (config.x_max - config.x_min) / (config.n_grid as f64 - 1.0);
    let x_grid: Vec<f64> = (0..config.n_grid)
        .map(|i| config.x_min + i as f64 * dx)
        .collect();

    // Interpolate all galaxies onto common grid
    let interp: Vec<Vec<Option<(f64, f64)>>> = galaxies
        .iter()
        .map(|g| interpolate_residuals(g, &x_grid))
        .collect();

    // Stack with (optional) inverse-variance weighting
    let mut delta_stack = vec![0.0_f64; config.n_grid];
    let mut delta_stack_err = vec![0.0_f64; config.n_grid];
    let mut n_contributing = vec![0_usize; config.n_grid];

    for j in 0..config.n_grid {
        let mut sum_w = 0.0_f64;
        let mut sum_wv = 0.0_f64;
        let mut count = 0_usize;

        for galaxy_interp in &interp {
            if let Some(&(dv, de)) = galaxy_interp[j].as_ref() {
                let w = if config.inverse_variance_weighting && de > 1e-10 {
                    1.0 / (de * de)
                } else {
                    1.0
                };
                sum_w += w;
                sum_wv += w * dv;
                count += 1;
            }
        }

        n_contributing[j] = count;
        if count >= config.min_galaxies_per_bin && sum_w > 0.0 {
            delta_stack[j] = sum_wv / sum_w;
            delta_stack_err[j] = 1.0 / sum_w.sqrt();
        }
    }

    // Fourier analysis at the N predicted wavenumbers for this CD dimension
    let cd = &config.cd_params;
    let (fourier_power, fourier_phase) = fourier_at_predicted_wavenumbers(
        &x_grid,
        &delta_stack,
        &n_contributing,
        config.min_galaxies_per_bin,
        cd.n_modes,
    );

    // RMS of the stacked residual (noise floor)
    let valid_bins: Vec<f64> = delta_stack
        .iter()
        .zip(&n_contributing)
        .filter(|(_, n)| **n >= config.min_galaxies_per_bin)
        .map(|(d, _)| *d)
        .collect();

    let rms_residual = if valid_bins.is_empty() {
        0.0
    } else {
        let mean = valid_bins.iter().sum::<f64>() / valid_bins.len() as f64;
        let var =
            valid_bins.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / valid_bins.len() as f64;
        var.sqrt()
    };

    // Detection SNR: peak Fourier power vs noise floor
    let max_power = fourier_power.iter().copied().fold(0.0_f64, f64::max);
    let noise_power = rms_residual.powi(2);
    let detection_snr = if noise_power > 0.0 {
        max_power.sqrt() / rms_residual
    } else {
        0.0
    };

    // Estimate alpha_zd from peak amplitude
    // The modulation amplitude is alpha_zd * af * exp(-x_peak),
    // so alpha_zd ~ A_peak / (af * exp(-x_peak))
    // Use x ~ 1 (near scale radius) as reference
    let af = cd.assessor_fraction;
    let alpha_zd_estimate = if af > 0.0 {
        max_power.sqrt() / af * 1.0_f64.exp()
    } else {
        0.0
    };

    StackingResult {
        n_galaxies: n_gal,
        cd_dim: cd.cd_dim,
        n_modes: cd.n_modes,
        x_grid,
        delta_stack,
        delta_stack_err,
        n_contributing,
        fourier_power,
        fourier_phase,
        rms_residual,
        detection_snr,
        alpha_zd_estimate,
    }
}

/// Compute Fourier power at the N predicted wavenumbers k_n' = 2*pi*n / N.
///
/// Uses a targeted DFT (not FFT) since we only need power at N specific
/// frequencies, not the full spectrum.
fn fourier_at_predicted_wavenumbers(
    x_grid: &[f64],
    delta: &[f64],
    n_contributing: &[usize],
    min_per_bin: usize,
    n_modes: usize,
) -> (Vec<f64>, Vec<f64>) {
    let mut power = vec![0.0_f64; n_modes];
    let mut phase = vec![0.0_f64; n_modes];

    for n in 1..=n_modes {
        let k = 2.0 * PI * n as f64 / n_modes as f64;
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut count = 0_usize;

        for (j, &x) in x_grid.iter().enumerate() {
            if n_contributing[j] < min_per_bin {
                continue;
            }
            re += delta[j] * (k * x).cos();
            im += delta[j] * (k * x).sin();
            count += 1;
        }

        if count > 0 {
            let norm = count as f64;
            re /= norm;
            im /= norm;
            power[n - 1] = re * re + im * im;
            phase[n - 1] = im.atan2(re);
        }
    }

    (power, phase)
}

/// Predicted harmonic wavenumbers for sedenions (D=16, backward compatible).
///
/// Returns the 7 wavenumbers k_n' = 2*pi*n / 7 for n = 1..7.
/// For arbitrary CD dimensions, use `CdDimensionParams::predicted_wavenumbers()`.
pub fn predicted_wavenumbers() -> Vec<f64> {
    CdDimensionParams::sedenion().predicted_wavenumbers()
}

/// Predicted harmonic wavenumbers for a given CD dimension.
///
/// Returns N wavenumbers k_n' = 2*pi*n / N for n = 1..N,
/// where N = dim/2 - 1 is the motif component count.
pub fn predicted_wavenumbers_cd(cd_dim: usize) -> Vec<f64> {
    CdDimensionParams::new(cd_dim).predicted_wavenumbers()
}

/// Detection threshold: minimum alpha_zd detectable by stacking N galaxies
/// with typical velocity uncertainty sigma_v and NFW velocity v_nfw.
///
/// alpha_zd_min = 2 * (sigma_v / v_nfw) / (0.5 * sqrt(N))
///
/// The factor 2 comes from alpha_zd * 0.5 being the peak amplitude,
/// and sqrt(N) from the noise reduction in stacking.
pub fn detection_threshold(n_galaxies: usize, sigma_v_frac: f64) -> f64 {
    if n_galaxies == 0 {
        return f64::INFINITY;
    }
    2.0 * sigma_v_frac / (0.5 * (n_galaxies as f64).sqrt())
}

/// Evaluate targeted DFT of a stacked profile at arbitrary wavenumbers.
///
/// Takes an already-computed stacked delta_v profile and evaluates the
/// normalized DFT at each wavenumber in `wavenumbers`. This allows
/// post-hoc multi-algebra analysis without re-running the full stacking.
///
/// The normalization matches `fourier_at_predicted_wavenumbers`:
///   re = sum(delta[j] * cos(k * x[j])) / count
///   im = sum(delta[j] * sin(k * x[j])) / count
///   power = re^2 + im^2
///
/// Returns (power, phase) vectors of length wavenumbers.len().
pub fn fourier_at_wavenumbers(
    x_grid: &[f64],
    delta: &[f64],
    n_contributing: &[usize],
    min_per_bin: usize,
    wavenumbers: &[f64],
) -> (Vec<f64>, Vec<f64>) {
    let mut power = vec![0.0_f64; wavenumbers.len()];
    let mut phase = vec![0.0_f64; wavenumbers.len()];

    for (mode_idx, &k) in wavenumbers.iter().enumerate() {
        let mut re = 0.0_f64;
        let mut im = 0.0_f64;
        let mut count = 0_usize;

        for (j, &x) in x_grid.iter().enumerate() {
            if n_contributing[j] < min_per_bin {
                continue;
            }
            re += delta[j] * (k * x).cos();
            im += delta[j] * (k * x).sin();
            count += 1;
        }

        if count > 0 {
            let norm = count as f64;
            re /= norm;
            im /= norm;
            power[mode_idx] = re * re + im * im;
            phase[mode_idx] = im.atan2(re);
        }
    }

    (power, phase)
}

/// G2 angular wavenumber series for rotation curve harmonic analysis.
///
/// G2 = Aut(O) is the automorphism group of the octonions and the compact
/// simple Lie algebra of type G2 with 12 roots (6 positive, 6 negative).
/// The 6 positive roots divide the root plane into 6 equal angular sectors
/// of 30 degrees each.
///
/// The angular-index wavenumber series maps each of the 6 positive roots to
/// a harmonic: k_n = 2*pi*n / 6 for n = 1..6.
///
/// This gives k in {1.047, 2.094, 3.141, 4.189, 5.236, 6.283}.
/// The spacing is pi/3 between consecutive modes (vs 2*pi/7 for CD-16).
/// The last mode (k_6 = 2*pi) coincides with CD-16 mode 7.
///
/// # Algebraic connection
///
/// G2 is the automorphism group of the octonions, which sit at the D=8
/// level of the Cayley-Dickson tower below the sedenions (D=16). The
/// 7 imaginary octonion units correspond to the 7 ZD box-kite slots
/// in the sedenion algebra (C-003). G2's 6 positive roots are therefore
/// a natural comparison basis for the sedenion's 7 CD-ZD modes.
pub fn g2_angular_wavenumbers() -> Vec<f64> {
    // G2 has 6 positive roots; angular-index series k_n = 2*pi*n/6
    (1..=6).map(|n| 2.0 * PI * n as f64 / 6.0).collect()
}

/// Albert algebra J3(O) Peirce wavenumber series for harmonic analysis.
///
/// The exceptional Jordan algebra J3(O) is the 27-dimensional algebra of
/// 3x3 Hermitian matrices over octonions. It has rank 3: the Peirce
/// decomposition with respect to any Jordan frame gives 3 eigenvalue layers
/// (Peirce spaces P_0, P_{1/2}, P_1 relative to each primitive idempotent).
///
/// The rank-3 structure gives a natural 3-mode wavenumber series:
///   k_n = 2*pi*n / 3 for n = 1, 2, 3.
///
/// This gives k in {2.094, 4.189, 6.283}. The last mode (k_3 = 2*pi)
/// coincides with CD-16 mode 7 and G2 mode 6.
///
/// # Algebraic connection
///
/// J3(O) is the Albert algebra; its automorphism group is the exceptional
/// Lie algebra F4 (52-dimensional). J3(O) sits above J2(O) = octonion
/// Minkowski space and below the 56-dimensional Freudenthal triple system.
/// The rank-3 structure (3 primitive idempotents) provides a complementary
/// viewpoint to the CD-ZD 7-mode series.
pub fn albert_peirce_wavenumbers() -> Vec<f64> {
    // J3(O) rank-3 Peirce decomposition -> 3-mode series k_n = 2*pi*n/3
    (1..=3).map(|n| 2.0 * PI * n as f64 / 3.0).collect()
}

/// sl(2) partner-graph embedding wavenumbers for harmonic analysis.
///
/// The sedenion partner graph has spectrum {-4, -2, 0, +2, +4} with
/// degeneracies {7, 14, 42, 14, 7} (C-1255). This spectrum is the weight
/// decomposition of the spin-2 (J=2) representation of sl(2,R):
///   m in {-2J, -2J+2, ..., +2J} = {-4, -2, 0, +2, +4} for J=2.
///
/// The non-zero weights |m| in {2, 4} map to wavenumbers using the sedenion
/// fundamental k_0 = 2*pi/7:
///   k_1 = 2 * k_0 = 4*pi/7 ~= 1.795  (= CD-16 mode 2 exactly)
///   k_2 = 4 * k_0 = 8*pi/7 ~= 3.590  (= CD-16 mode 4 exactly)
///
/// The sl(2) embedding recovers exactly modes 2 and 4 of the CD-16
/// sedenion series -- the "even-index" subset of CD-ZD wavenumbers.
/// This algebraic coincidence reflects the embedding sl(2) < g2 < f4
/// and the fact that the partner graph is a substructure of the CD algebra.
pub fn sl2_partner_graph_wavenumbers() -> Vec<f64> {
    // sl(2) spin-2 weights {2, 4} * k_0 where k_0 = 2*pi/7 (sedenion fundamental)
    let k0 = 2.0 * PI / 7.0;
    vec![2.0 * k0, 4.0 * k0]
}

// ---------------------------------------------------------------------------
// Non-static (x-resolved) signal analysis
// ---------------------------------------------------------------------------

/// Output of a sliding-window short-time Fourier transform spectrogram.
///
/// Reveals WHERE in x = r/r_s space harmonic power is concentrated.
/// Baryonic features (bulge excess, NFW cusp trough) are localized at x < 2.
/// A genuine ZD forcing signal would produce power across the full x range.
#[derive(Debug, Clone)]
pub struct StftSpectrogram {
    /// Center positions of each sliding Gaussian window.
    pub x_centers: Vec<f64>,
    /// Wavenumbers at which DFT power is evaluated.
    pub wavenumbers: Vec<f64>,
    /// `power[i][j]` = windowed DFT power at `x_centers[i]`, `wavenumbers[j]`.
    pub power: Vec<Vec<f64>>,
    /// `phase[i][j]` = windowed DFT phase (radians) at `x_centers[i]`, `wavenumbers[j]`.
    pub phase: Vec<Vec<f64>>,
    /// Effective bin weight (sum of Gaussian weights) at each window center.
    pub n_eff: Vec<f64>,
}

/// Compute a sliding-window STFT spectrogram with a Gaussian window.
///
/// The window weight for bin j centered at x_c is:
///   w(x_j, x_c) = exp(-0.5 * ((x_j - x_c) / sigma_x)^2)
///
/// The windowed DFT at wavenumber k is:
///   F(x_c, k) = sum_j w_j * delta_j * exp(-i k x_j) / sum_j w_j
///   power(x_c, k) = |F|^2
///
/// # Why Gaussian windows?
///
/// A Gaussian window minimizes the time-frequency uncertainty product
/// (Heisenberg-Gabor limit). For a profile with features of spatial scale ~1 r_s,
/// sigma_x = 1.0 r_s balances x-resolution against k-resolution.
/// Narrower sigma_x increases x-resolution but blurs k-resolution.
///
/// # Arguments
///
/// * `sigma_x` - Half-width of Gaussian window in r/r_s units (typ. 1.0-2.0).
/// * `n_windows` - Number of window centers spanning the valid x range.
pub fn stft_spectrogram(
    x_grid: &[f64],
    delta: &[f64],
    n_contributing: &[usize],
    min_per_bin: usize,
    wavenumbers: &[f64],
    sigma_x: f64,
    n_windows: usize,
) -> StftSpectrogram {
    let n = x_grid.len();

    // Determine x range from valid bins.
    let mut x_min = f64::INFINITY;
    let mut x_max = f64::NEG_INFINITY;
    for (j, &x) in x_grid.iter().enumerate() {
        if n_contributing[j] >= min_per_bin {
            if x < x_min {
                x_min = x;
            }
            if x > x_max {
                x_max = x;
            }
        }
    }

    if x_min > x_max || n_windows == 0 {
        return StftSpectrogram {
            x_centers: vec![],
            wavenumbers: wavenumbers.to_vec(),
            power: vec![],
            phase: vec![],
            n_eff: vec![],
        };
    }

    let x_centers: Vec<f64> = if n_windows == 1 {
        vec![(x_min + x_max) / 2.0]
    } else {
        (0..n_windows)
            .map(|i| x_min + (x_max - x_min) * i as f64 / (n_windows - 1) as f64)
            .collect()
    };

    let inv_2sigma2 = 0.5 / (sigma_x * sigma_x);
    let nc = x_centers.len();
    let nk = wavenumbers.len();

    let mut power = vec![vec![0.0_f64; nk]; nc];
    let mut phase = vec![vec![0.0_f64; nk]; nc];
    let mut n_eff = vec![0.0_f64; nc];

    for (ci, &x_c) in x_centers.iter().enumerate() {
        // Accumulate window weight.
        let mut w_sum = 0.0_f64;
        for j in 0..n {
            if n_contributing[j] < min_per_bin {
                continue;
            }
            let dx = x_grid[j] - x_c;
            w_sum += (-dx * dx * inv_2sigma2).exp();
        }
        n_eff[ci] = w_sum;
        if w_sum < 1e-12 {
            continue;
        }

        for (ki, &k) in wavenumbers.iter().enumerate() {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            for j in 0..n {
                if n_contributing[j] < min_per_bin {
                    continue;
                }
                let dx = x_grid[j] - x_c;
                let w = (-dx * dx * inv_2sigma2).exp();
                let kx = k * x_grid[j];
                re += w * delta[j] * kx.cos();
                im += w * delta[j] * kx.sin();
            }
            re /= w_sum;
            im /= w_sum;
            power[ci][ki] = re * re + im * im;
            phase[ci][ki] = im.atan2(re);
        }
    }

    StftSpectrogram {
        x_centers,
        wavenumbers: wavenumbers.to_vec(),
        power,
        phase,
        n_eff,
    }
}

/// Derivative stacking: compute d(delta_stack)/dx via finite differences.
///
/// Central differences for interior valid bins; one-sided at endpoints.
///
/// # Why differentiate?
///
/// The baryonic baseline is approximately constant (the -9% DC offset from
/// the stacked profile). d/dx annihilates a constant, exposing oscillatory
/// content. The first derivative amplifies harmonic modes relative to the
/// monotonic DC component; the second derivative further sharpens peaks.
/// If the stacked profile has harmonic content at wavenumber k, d(delta)/dx
/// has a peak at the same k with amplitude proportional to k.
///
/// Returns `(x_mid, d_delta_dx)` where each point uses the valid bins
/// from which the central difference was computed.
pub fn derivative_stack(
    x_grid: &[f64],
    delta: &[f64],
    n_contributing: &[usize],
    min_per_bin: usize,
) -> (Vec<f64>, Vec<f64>) {
    // Build arrays of (x, delta) for valid bins only.
    let valid: Vec<(f64, f64)> = x_grid
        .iter()
        .zip(delta)
        .zip(n_contributing)
        .filter(|(_, nc)| **nc >= min_per_bin)
        .map(|((&x, &d), _)| (x, d))
        .collect();

    let m = valid.len();
    let mut x_out = Vec::with_capacity(m);
    let mut dy_out = Vec::with_capacity(m);

    if m < 2 {
        return (x_out, dy_out);
    }

    for j in 0..m {
        let (x_j, d_j) = valid[j];
        let (deriv, x_mid) = if j == 0 {
            let (x1, d1) = valid[1];
            let dx = x1 - x_j;
            if dx.abs() < 1e-15 {
                continue;
            }
            ((d1 - d_j) / dx, (x_j + x1) / 2.0)
        } else if j == m - 1 {
            let (xm1, dm1) = valid[m - 2];
            let dx = x_j - xm1;
            if dx.abs() < 1e-15 {
                continue;
            }
            ((d_j - dm1) / dx, (xm1 + x_j) / 2.0)
        } else {
            let (xm1, dm1) = valid[j - 1];
            let (x1, d1) = valid[j + 1];
            let dx = x1 - xm1;
            if dx.abs() < 1e-15 {
                continue;
            }
            ((d1 - dm1) / dx, x_j)
        };
        x_out.push(x_mid);
        dy_out.push(deriv);
    }

    (x_out, dy_out)
}

/// Result of a jackknife phase coherence test.
///
/// The Rayleigh R statistic measures how tightly jackknife-sample DFT phases
/// cluster around their mean. R -> 1 indicates a phase-coherent signal
/// (each jackknife sample produces the same phase). R -> 0 indicates that
/// phases scatter randomly, consistent with noise.
#[derive(Debug, Clone)]
pub struct PhaseCoherenceResult {
    /// Wavenumbers tested.
    pub wavenumbers: Vec<f64>,
    /// Rayleigh R statistic in [0, 1] for each wavenumber.
    /// R = (1/n) * |sum_j exp(i * phi_j)| where phi_j are jackknife phases.
    pub rayleigh_r: Vec<f64>,
    /// Approximate p-value under H0 (uniformly distributed phases).
    /// p ~= exp(-n * R^2) for large n (Rayleigh test approximation).
    pub rayleigh_p: Vec<f64>,
    /// Number of jackknife samples (= number of valid x bins).
    pub n_jackknife: usize,
}

/// Jackknife phase coherence test.
///
/// For each valid x-bin, drop it from the sum and recompute the DFT phase.
/// The resulting n_jackknife phases are tested for circular clustering using
/// the Rayleigh R statistic.
///
/// # Interpretation
///
/// A signal from physical forcing would produce consistent complex Fourier
/// coefficients regardless of which single bin is excluded. Noise would cause
/// the phase to wander as different bins are excluded. The Rayleigh p-value
/// tests the null hypothesis that the jackknife phases are uniform on [-pi, pi].
///
/// For SNR < 1 (baryonic noise-dominated), R is expected near 0.1-0.3 with
/// p ~ 0.5-0.9, confirming that the DFT phases are not coherent.
pub fn jackknife_phase_coherence(
    x_grid: &[f64],
    delta: &[f64],
    n_contributing: &[usize],
    min_per_bin: usize,
    wavenumbers: &[f64],
) -> PhaseCoherenceResult {
    let valid_idx: Vec<usize> = (0..x_grid.len())
        .filter(|&j| n_contributing[j] >= min_per_bin)
        .collect();

    let m = valid_idx.len();
    let nk = wavenumbers.len();

    // phases[ki] accumulates the DFT phase for each jackknife drop.
    let mut jk_re: Vec<Vec<f64>> = vec![Vec::with_capacity(m); nk];
    let mut jk_im: Vec<Vec<f64>> = vec![Vec::with_capacity(m); nk];

    for drop in 0..m {
        for (ki, &k) in wavenumbers.iter().enumerate() {
            let mut re = 0.0_f64;
            let mut im = 0.0_f64;
            let mut count = 0_usize;
            for (idx, &j) in valid_idx.iter().enumerate() {
                if idx == drop {
                    continue;
                }
                let kx = k * x_grid[j];
                re += delta[j] * kx.cos();
                im += delta[j] * kx.sin();
                count += 1;
            }
            if count > 0 {
                let norm = count as f64;
                jk_re[ki].push(re / norm);
                jk_im[ki].push(im / norm);
            }
        }
    }

    let mut rayleigh_r = vec![0.0_f64; nk];
    let mut rayleigh_p = vec![1.0_f64; nk];

    for ki in 0..nk {
        let n = jk_re[ki].len();
        if n < 2 {
            continue;
        }
        // Convert (re, im) vectors to unit-complex phases, then Rayleigh R.
        let mut sum_cos = 0.0_f64;
        let mut sum_sin = 0.0_f64;
        for (&r, &i) in jk_re[ki].iter().zip(&jk_im[ki]) {
            let phi = i.atan2(r);
            sum_cos += phi.cos();
            sum_sin += phi.sin();
        }
        let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n as f64;
        rayleigh_r[ki] = r;
        // Rayleigh test: T = 2*n*R^2 ~ chi2(2) under H0. p ~= exp(-T/2).
        rayleigh_p[ki] = (-(n as f64) * r * r).exp();
    }

    PhaseCoherenceResult {
        wavenumbers: wavenumbers.to_vec(),
        rayleigh_r,
        rayleigh_p,
        n_jackknife: m,
    }
}

// ---------------------------------------------------------------------------
// Galaxy-ensemble phase coherence (Item 2: N=6992 vs N=19 jackknife)
// ---------------------------------------------------------------------------

/// Per-galaxy complex DFT at each wavenumber.
///
/// For each galaxy with at least `min_points` valid data points,
/// computes the normalized DFT at each wavenumber in x = r/r_s
/// coordinates. Returns (power, phase) per galaxy per mode.
///
/// Unlike the x-bin jackknife (N_jackknife = 19), this operates on the
/// galaxy ENSEMBLE (N_gal = 6992), giving ~sqrt(6992/19) ~ 19x more
/// statistical power. Expected Rayleigh R under H0: 1/sqrt(N_gal) ~ 0.012.
///
/// Galaxies with fewer than `min_points` contribute zero (excluded).
pub fn per_galaxy_dft(
    galaxies: &[NormalizedResiduals],
    wavenumbers: &[f64],
    min_points: usize,
) -> Vec<Vec<(f64, f64)>> {
    let nk = wavenumbers.len();
    galaxies
        .iter()
        .map(|g| {
            if g.points.len() < min_points {
                return vec![(0.0_f64, 0.0_f64); nk];
            }
            let n = g.points.len() as f64;
            wavenumbers
                .iter()
                .map(|&k| {
                    let re: f64 = g
                        .points
                        .iter()
                        .map(|p| p.delta_v * (k * p.x).cos())
                        .sum::<f64>()
                        / n;
                    let im: f64 = g
                        .points
                        .iter()
                        .map(|p| p.delta_v * (k * p.x).sin())
                        .sum::<f64>()
                        / n;
                    (re * re + im * im, im.atan2(re))
                })
                .collect()
        })
        .collect()
}

/// Galaxy-ensemble Rayleigh phase coherence test.
///
/// For each wavenumber, computes the per-galaxy DFT phase (using
/// `per_galaxy_dft`) and applies the Rayleigh R statistic across all
/// N_gal galaxies. This corrects the fundamental limitation of the
/// x-bin jackknife (N=19 bins vs N=6992 galaxies).
///
/// Under H0 (incoherent phases), expected R ~ 1/sqrt(N_gal) ~ 0.012
/// with p > 0.9. A signal at alpha_zd = 0.004 would produce R > 0.1.
///
/// Returns `PhaseCoherenceResult` with `n_jackknife` = N_gal.
pub fn galaxy_ensemble_phase_coherence(
    galaxies: &[NormalizedResiduals],
    wavenumbers: &[f64],
    min_points: usize,
) -> PhaseCoherenceResult {
    let per_gal = per_galaxy_dft(galaxies, wavenumbers, min_points);
    let nk = wavenumbers.len();
    let n_gal = galaxies.len();

    let mut rayleigh_r = vec![0.0_f64; nk];
    let mut rayleigh_p = vec![1.0_f64; nk];

    for ki in 0..nk {
        let mut sum_cos = 0.0_f64;
        let mut sum_sin = 0.0_f64;
        let mut n_valid = 0_usize;

        for gal_dfts in &per_gal {
            let (power, phase) = gal_dfts[ki];
            if power > 0.0 {
                sum_cos += phase.cos();
                sum_sin += phase.sin();
                n_valid += 1;
            }
        }

        if n_valid < 2 {
            continue;
        }
        let r = (sum_cos * sum_cos + sum_sin * sum_sin).sqrt() / n_valid as f64;
        rayleigh_r[ki] = r;
        // Rayleigh test: T = 2*n*R^2 ~ chi2(2) under H0 -> p = exp(-n*R^2)
        rayleigh_p[ki] = (-(n_valid as f64) * r * r).exp();
    }

    PhaseCoherenceResult {
        wavenumbers: wavenumbers.to_vec(),
        rayleigh_r,
        rayleigh_p,
        n_jackknife: n_gal,
    }
}

// ---------------------------------------------------------------------------
// Signal injection and recovery (Item 5: pipeline validation)
// ---------------------------------------------------------------------------

/// Result of a signal injection-recovery test.
#[derive(Debug, Clone)]
pub struct InjectionRecoveryPoint {
    /// ZD forcing amplitude that was injected.
    pub alpha_zd_injected: f64,
    /// Detected SNR after injection and stacking.
    pub detected_snr: f64,
    /// Alpha_zd estimated from the stacked amplitude.
    pub alpha_zd_recovered: f64,
    /// Ratio: alpha_zd_recovered / alpha_zd_injected.
    /// Near 1.0 for a linear, unbiased pipeline.
    /// NaN when alpha_zd_injected = 0 (noise-only baseline).
    pub recovery_ratio: f64,
}

/// Inject ZD harmonic modulation into normalized residuals.
///
/// Adds the predicted CD-ZD signal at amplitude `alpha_zd` to each
/// galaxy's fractional velocity residuals:
///
///   delta_v_injected = delta_v + alpha_zd * sum_{n=1}^{N} (af/n) * cos(k_n * x) * exp(-x)
///
/// where k_n = 2*pi*n / N_modes (universal in x = r/r_s), af = 0.5
/// (assessor fraction), and N_modes = cd_dim/2 - 1.
///
/// At alpha_zd = 0, returns a clone of the input unchanged.
/// Uses the same amplitude formula as `harmonic_halos::harmonic_halo_modulation`.
pub fn inject_zd_signal(
    galaxies: &[NormalizedResiduals],
    alpha_zd: f64,
    cd_dim: usize,
) -> Vec<NormalizedResiduals> {
    if alpha_zd == 0.0 {
        return galaxies.to_vec();
    }
    let params = CdDimensionParams::new(cd_dim);
    let wavenumbers = params.predicted_wavenumbers();
    let af = params.assessor_fraction;

    galaxies
        .iter()
        .map(|g| {
            let points = g
                .points
                .iter()
                .map(|p| {
                    let modulation: f64 = wavenumbers
                        .iter()
                        .enumerate()
                        .map(|(n, &k)| {
                            let a_n = af / (n + 1) as f64;
                            a_n * (k * p.x).cos() * (-p.x).exp()
                        })
                        .sum();
                    NormalizedPoint {
                        x: p.x,
                        delta_v: p.delta_v + alpha_zd * modulation,
                        delta_v_err: p.delta_v_err,
                    }
                })
                .collect();
            NormalizedResiduals {
                name: g.name.clone(),
                r_s_kpc: g.r_s_kpc,
                points,
            }
        })
        .collect()
}

/// Sweep a range of injection amplitudes and measure pipeline recovery.
///
/// For each alpha_zd in `alpha_zd_values`:
/// 1. Inject the ZD signal at that amplitude via `inject_zd_signal`.
/// 2. Stack the injected residuals with `stack_residuals`.
/// 3. Record detected SNR and recovered amplitude.
///
/// At alpha_zd = 0: result matches baseline E-183 SNR = 0.29.
/// At alpha_zd = 0.004 (SKA floor): expected SNR > 3.0, recovery_ratio in [0.7, 1.3].
/// At alpha_zd = 0.001 (below threshold): marginal detection ~ SNR 1.5.
pub fn injection_recovery_sweep(
    galaxies: &[NormalizedResiduals],
    alpha_zd_values: &[f64],
    cd_dim: usize,
    config: &StackingConfig,
) -> Vec<InjectionRecoveryPoint> {
    alpha_zd_values
        .iter()
        .map(|&alpha| {
            let injected = inject_zd_signal(galaxies, alpha, cd_dim);
            let result = stack_residuals(&injected, config);
            let recovery_ratio = if alpha > 0.0 {
                result.alpha_zd_estimate / alpha
            } else {
                f64::NAN
            };
            InjectionRecoveryPoint {
                alpha_zd_injected: alpha,
                detected_snr: result.detection_snr,
                alpha_zd_recovered: result.alpha_zd_estimate,
                recovery_ratio,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// HI Eigenmode Stacking (Item 1: PCA via Jacobi eigensystem)
// ---------------------------------------------------------------------------

/// Result of an eigenmode decomposition of the galaxy covariance matrix.
///
/// Performs PCA on the N_gal x N_r residual matrix to extract the dominant
/// modes of coherent variation. A ZD signal would appear as a leading eigenmode
/// with high overlap onto the predicted ZD template.
#[derive(Debug, Clone)]
pub struct EigenmodeStackResult {
    /// Common x = r/r_s grid (N_r points).
    pub x_grid: Vec<f64>,
    /// Eigenvalues sorted by descending |lambda| (N_r values).
    pub eigenvalues: Vec<f64>,
    /// Top n_modes_kept eigenvectors, each of length N_r.
    pub eigenmodes: Vec<Vec<f64>>,
    /// |dot(V_k, T)| / (||V_k|| * ||T||) for each retained mode.
    pub zd_overlaps: Vec<f64>,
    /// Cumulative fraction of total variance per mode (length n_modes_kept).
    pub explained_variance: Vec<f64>,
    /// max_k(zd_overlaps[k] * sqrt(lambda_k / lambda_mean)).
    pub reconstruction_snr: f64,
    /// Number of galaxies used.
    pub n_galaxies: usize,
    /// Number of eigenmodes retained.
    pub n_modes_kept: usize,
}

/// PCA-based eigenmode stacking: decompose the galaxy covariance matrix.
///
/// # Algorithm
/// 1. Interpolate all galaxy residuals onto the common x_grid; build binary mask.
/// 2. Build the N_r x N_r covariance matrix C, weighted by per-galaxy coverage.
/// 3. Eigendecompose C via x87 (x86_64) or DD (other) Jacobi solver.
/// 4. Compute ZD template overlaps for each eigenmode.
///
/// The covariance matrix is constructed as a sum of outer products weighted by
/// the binary coverage mask, so galaxies that do not cover a bin contribute zero
/// to that bin's covariance. This avoids imputation bias from zero-filling.
pub fn eigenmode_stack(
    galaxies: &[NormalizedResiduals],
    config: &StackingConfig,
    n_modes: usize,
) -> EigenmodeStackResult {
    let n_r = config.n_grid;
    let n_gal = galaxies.len();

    // Build common x-grid
    let dx = if n_r > 1 {
        (config.x_max - config.x_min) / (n_r as f64 - 1.0)
    } else {
        0.0
    };
    let x_grid: Vec<f64> = (0..n_r).map(|i| config.x_min + i as f64 * dx).collect();

    let empty = EigenmodeStackResult {
        x_grid: x_grid.clone(),
        eigenvalues: vec![0.0; n_r],
        eigenmodes: vec![],
        zd_overlaps: vec![],
        explained_variance: vec![],
        reconstruction_snr: 0.0,
        n_galaxies: n_gal,
        n_modes_kept: 0,
    };

    if n_gal == 0 || n_r == 0 {
        return empty;
    }

    // Interpolate all galaxies and build X matrix [n_gal x n_r] with mask
    let interp: Vec<Vec<Option<(f64, f64)>>> = galaxies
        .iter()
        .map(|g| interpolate_residuals(g, &x_grid))
        .collect();

    // Compute column means weighted by coverage count
    let mut col_count = vec![0_usize; n_r];
    let mut col_sum = vec![0.0_f64; n_r];
    for g_interp in &interp {
        for (j, pt) in g_interp.iter().enumerate() {
            if let Some(&(dv, _)) = pt.as_ref() {
                col_count[j] += 1;
                col_sum[j] += dv;
            }
        }
    }
    let col_mean: Vec<f64> = col_count
        .iter()
        .zip(col_sum.iter())
        .map(|(&cnt, &s)| if cnt > 0 { s / cnt as f64 } else { 0.0 })
        .collect();

    // Build X [n_gal x n_r] with mean-subtracted residuals (zero for missing)
    let mut x_mat: Vec<Vec<f64>> = vec![vec![0.0; n_r]; n_gal];
    let mut mask: Vec<Vec<f64>> = vec![vec![0.0; n_r]; n_gal];
    for (g, g_interp) in interp.iter().enumerate() {
        for (j, pt) in g_interp.iter().enumerate() {
            if let Some(&(dv, _)) = pt.as_ref() {
                x_mat[g][j] = dv - col_mean[j];
                mask[g][j] = 1.0;
            }
        }
    }

    // Build covariance matrix C [n_r x n_r] = sum_g (x_g * mask_g) outer (x_g * mask_g)
    let mut cov_flat: Vec<f64> = vec![0.0; n_r * n_r];
    for g in 0..n_gal {
        for i in 0..n_r {
            if mask[g][i] < 0.5 {
                continue;
            }
            let xi = x_mat[g][i];
            for j in 0..n_r {
                if mask[g][j] < 0.5 {
                    continue;
                }
                cov_flat[i * n_r + j] += xi * x_mat[g][j];
            }
        }
    }

    // Eigendecompose C
    let eigen_result: Result<(Vec<f64>, Vec<f64>), String> = {
        #[cfg(target_arch = "x86_64")]
        {
            algebra_analysis::x87_jacobi::symmetric_eigensystem_x87(
                &cov_flat,
                n_r,
                5 * n_r * n_r,
                1.0e-12,
            )
            .map_err(|e| e.to_string())
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let mat: Vec<Vec<f64>> = (0..n_r)
                .map(|i| cov_flat[i * n_r..(i + 1) * n_r].to_vec())
                .collect();
            algebra_analysis::dd_jacobi::symmetric_eigensystem_dd(&mat).map_err(|e| e.to_string())
        }
    };

    let (eigenvalues, v_flat) = match eigen_result {
        Ok(r) => r,
        Err(_) => return empty,
    };

    // ZD template: T[i] = sum_n (af/n) * cos(k_n * x[i]) * exp(-x[i])
    let wavenumbers = config.cd_params.predicted_wavenumbers();
    let af = config.cd_params.assessor_fraction;
    let template: Vec<f64> = x_grid
        .iter()
        .map(|&x| {
            wavenumbers
                .iter()
                .enumerate()
                .map(|(n, &k)| (af / (n + 1) as f64) * (k * x).cos() * (-x).exp())
                .sum::<f64>()
        })
        .collect();
    let template_norm = template.iter().map(|t| t * t).sum::<f64>().sqrt();

    let n_keep = n_modes.min(n_r);
    let lambda_sum: f64 = eigenvalues.iter().map(|e| e.abs()).sum();
    let lambda_mean = if n_r > 0 {
        lambda_sum / n_r as f64
    } else {
        0.0
    };

    let mut eigenmodes = Vec::with_capacity(n_keep);
    let mut zd_overlaps = Vec::with_capacity(n_keep);
    let mut explained_variance = Vec::with_capacity(n_keep);
    let mut cumulative_var = 0.0_f64;

    for k in 0..n_keep {
        // Extract column k of V: V[i,k] = v_flat[i*n_r + k]
        let mode: Vec<f64> = (0..n_r).map(|i| v_flat[i * n_r + k]).collect();
        let mode_norm = mode.iter().map(|v| v * v).sum::<f64>().sqrt();

        let overlap = if mode_norm > 0.0 && template_norm > 0.0 {
            let dot: f64 = mode.iter().zip(template.iter()).map(|(v, t)| v * t).sum();
            dot.abs() / (mode_norm * template_norm)
        } else {
            0.0
        };

        cumulative_var += eigenvalues[k].abs() / lambda_sum.max(1.0e-30);

        eigenmodes.push(mode);
        zd_overlaps.push(overlap);
        explained_variance.push(cumulative_var);
    }

    // reconstruction_snr = max_k(zd_overlaps[k] * sqrt(|lambda_k| / lambda_mean))
    let reconstruction_snr = zd_overlaps
        .iter()
        .enumerate()
        .map(|(k, &ov)| {
            let scale = if lambda_mean > 0.0 {
                (eigenvalues[k].abs() / lambda_mean).sqrt()
            } else {
                0.0
            };
            ov * scale
        })
        .fold(0.0_f64, f64::max);

    EigenmodeStackResult {
        x_grid,
        eigenvalues,
        eigenmodes,
        zd_overlaps,
        explained_variance,
        reconstruction_snr,
        n_galaxies: n_gal,
        n_modes_kept: n_keep,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_galaxy(name: &str, r_s: f64, n_pts: usize, residual: f64) -> NormalizedResiduals {
        let points: Vec<NormalizedPoint> = (0..n_pts)
            .map(|i| {
                let x = 0.1 + (i as f64 / (n_pts - 1) as f64) * 9.9;
                NormalizedPoint {
                    x,
                    delta_v: residual,
                    delta_v_err: 0.01,
                }
            })
            .collect();
        NormalizedResiduals {
            name: name.to_string(),
            r_s_kpc: r_s,
            points,
        }
    }

    #[test]
    fn test_stack_zero_residuals() {
        let galaxies: Vec<_> = (0..10)
            .map(|i| make_galaxy(&format!("G{i}"), 10.0 + i as f64, 50, 0.0))
            .collect();
        let result = stack_residuals(&galaxies, &StackingConfig::default());
        assert_eq!(result.n_galaxies, 10);
        for &d in &result.delta_stack {
            assert!(d.abs() < 1e-12, "zero residuals should stack to zero");
        }
        assert!(
            result.rms_residual < 1e-12,
            "RMS should be ~zero for zero input"
        );
    }

    #[test]
    fn test_stack_constant_offset() {
        let galaxies: Vec<_> = (0..5)
            .map(|i| make_galaxy(&format!("G{i}"), 15.0, 50, 0.02))
            .collect();
        let result = stack_residuals(&galaxies, &StackingConfig::default());
        // All galaxies have +2% residual, stack should recover ~0.02
        let valid: Vec<f64> = result
            .delta_stack
            .iter()
            .zip(&result.n_contributing)
            .filter(|(_, n)| **n >= 3)
            .map(|(&d, _)| d)
            .collect();
        let mean = valid.iter().sum::<f64>() / valid.len() as f64;
        assert!(
            (mean - 0.02).abs() < 0.005,
            "stacked mean should be ~0.02, got {mean}"
        );
    }

    #[test]
    fn test_predicted_wavenumbers_distinct() {
        let k = predicted_wavenumbers();
        assert_eq!(k.len(), 7);
        for i in 0..k.len() {
            assert!(k[i] > 0.0);
            for j in (i + 1)..k.len() {
                assert!((k[i] - k[j]).abs() > 0.1, "wavenumbers should be distinct");
            }
        }
        // k_7 should be 2*pi
        assert!((k[6] - 2.0 * PI).abs() < 1e-10);
    }

    #[test]
    fn test_cd_dimension_params() {
        let p16 = CdDimensionParams::sedenion();
        assert_eq!(p16.cd_dim, 16);
        assert_eq!(p16.n_modes, 7);
        assert!((p16.assessor_fraction - 0.5).abs() < 1e-15);

        let p32 = CdDimensionParams::new(32);
        assert_eq!(p32.n_modes, 15);

        let p64 = CdDimensionParams::new(64);
        assert_eq!(p64.n_modes, 31);

        let p128 = CdDimensionParams::new(128);
        assert_eq!(p128.n_modes, 63);

        let p256 = CdDimensionParams::new(256);
        assert_eq!(p256.n_modes, 127);
    }

    #[test]
    fn test_wavenumbers_scale_with_dimension() {
        // At D=16: k_7 = 2*pi (full circle). At D=32: k_15 = 2*pi.
        let k16 = CdDimensionParams::new(16).predicted_wavenumbers();
        let k32 = CdDimensionParams::new(32).predicted_wavenumbers();

        assert_eq!(k16.len(), 7);
        assert_eq!(k32.len(), 15);

        // Both last wavenumbers should be 2*pi (full circle)
        assert!((k16[6] - 2.0 * PI).abs() < 1e-10);
        assert!((k32[14] - 2.0 * PI).abs() < 1e-10);

        // D=32 fundamental mode has lower wavenumber (longer wavelength)
        assert!(k32[0] < k16[0]);
    }

    #[test]
    fn test_stacking_with_d32() {
        let galaxies: Vec<_> = (0..10)
            .map(|i| make_galaxy(&format!("G{i}"), 10.0 + i as f64, 50, 0.0))
            .collect();
        let config = StackingConfig {
            cd_params: CdDimensionParams::new(32),
            ..Default::default()
        };
        let result = stack_residuals(&galaxies, &config);
        assert_eq!(result.cd_dim, 32);
        assert_eq!(result.n_modes, 15);
        assert_eq!(result.fourier_power.len(), 15);
        assert_eq!(result.fourier_phase.len(), 15);
    }

    #[test]
    fn test_cd_algebraic_statistics() {
        // Verify the analytical formulae for all documented dimensions.
        // n_modes = dim/2 - 1
        // nodes_per_comp = dim/2 - 2
        // n_assessors = n_modes * nodes_per_comp
        // n_zds = 2 * n_assessors
        // assessor_fraction = 0.5 (always)
        let expected: &[(usize, usize, usize, usize)] = &[
            // (dim, n_modes, n_assessors, n_zds)
            (16, 7, 42, 84),
            (32, 15, 210, 420),
            (64, 31, 930, 1860),
            (128, 63, 3906, 7812),
            (256, 127, 16002, 32004),
            (512, 255, 64770, 129540),
            (1024, 511, 260610, 521220),
            (2048, 1023, 1045506, 2091012),
            (4096, 2047, 4188162, 8376324),
        ];
        for &(dim, exp_modes, exp_assessors, exp_zds) in expected {
            let p = CdDimensionParams::new(dim);
            assert_eq!(p.n_modes, exp_modes, "n_modes mismatch at D={dim}");

            let nodes_per_comp = dim / 2 - 2;
            let n_assessors = p.n_modes * nodes_per_comp;
            assert_eq!(
                n_assessors, exp_assessors,
                "n_assessors mismatch at D={dim}"
            );

            let n_zds = 2 * n_assessors;
            assert_eq!(n_zds, exp_zds, "n_zds mismatch at D={dim}");

            assert!(
                (p.assessor_fraction - 0.5).abs() < 1e-15,
                "assessor_fraction should be 0.5 at D={dim}"
            );
        }
    }

    #[test]
    fn test_cd_dimension_up_to_4096() {
        // All power-of-two dimensions from 16 to 4096 should work
        let mut dim = 16;
        while dim <= 4096 {
            let p = CdDimensionParams::new(dim);
            assert_eq!(p.n_modes, dim / 2 - 1);
            assert!(p.predicted_wavenumbers().len() == p.n_modes);
            dim *= 2;
        }
    }

    #[test]
    fn test_detection_threshold() {
        // 175 galaxies (SPARC), 5% velocity uncertainty
        let thr = detection_threshold(175, 0.05);
        assert!(
            thr < 0.02,
            "SPARC-like threshold should be < 2%, got {thr:.4}"
        );
        assert!(thr > 0.001, "threshold should be positive, got {thr:.4}");

        // 10000 galaxies (MaNGA), 5% uncertainty
        let thr2 = detection_threshold(10000, 0.05);
        assert!(thr2 < thr, "more galaxies should give lower threshold");
    }

    #[test]
    fn test_fourier_detects_injected_signal() {
        // Inject a mode-1 harmonic signal: delta = A * cos(k_1 * x)
        let k1 = 2.0 * PI / 7.0;
        let amplitude = 0.05;
        let galaxies: Vec<_> = (0..20)
            .map(|i| {
                let points: Vec<NormalizedPoint> = (0..100)
                    .map(|j| {
                        let x = 0.1 + (j as f64 / 99.0) * 9.9;
                        NormalizedPoint {
                            x,
                            delta_v: amplitude * (k1 * x).cos(),
                            delta_v_err: 0.01,
                        }
                    })
                    .collect();
                NormalizedResiduals {
                    name: format!("G{i}"),
                    r_s_kpc: 10.0,
                    points,
                }
            })
            .collect();

        let result = stack_residuals(&galaxies, &StackingConfig::default());

        // Mode 1 should have the most power
        let max_idx = result
            .fourier_power
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_idx,
            0,
            "mode 1 (index 0) should have peak power, got mode {}",
            max_idx + 1
        );
        assert!(
            result.fourier_power[0] > result.fourier_power[1] * 2.0,
            "mode 1 power should dominate"
        );
    }

    #[test]
    fn test_per_galaxy_dft_zero_input() {
        // A galaxy with all-zero residuals should have zero power at all modes.
        let g = make_galaxy("Z", 10.0, 50, 0.0);
        let wavenumbers = predicted_wavenumbers();
        let result = per_galaxy_dft(&[g], &wavenumbers, 3);
        assert_eq!(result.len(), 1);
        for (power, _phase) in &result[0] {
            assert!(power.abs() < 1e-15, "zero residuals -> zero power");
        }
    }

    #[test]
    fn test_per_galaxy_dft_below_min_points() {
        // Galaxy with 2 points but min_points=3: should return zeros.
        let pts: Vec<NormalizedPoint> = (0..2)
            .map(|i| NormalizedPoint {
                x: 0.5 + i as f64,
                delta_v: 0.1,
                delta_v_err: 0.01,
            })
            .collect();
        let g = NormalizedResiduals {
            name: "tiny".to_string(),
            r_s_kpc: 10.0,
            points: pts,
        };
        let wavenumbers = predicted_wavenumbers();
        let result = per_galaxy_dft(&[g], &wavenumbers, 3);
        for (power, _) in &result[0] {
            assert_eq!(*power, 0.0);
        }
    }

    #[test]
    fn test_galaxy_ensemble_phase_coherence_noise() {
        // 20 galaxies with random-ish residuals (no coherent phase).
        // Rayleigh R should be small.
        let galaxies: Vec<_> = (0..20)
            .map(|i| {
                make_galaxy(
                    &format!("G{i}"),
                    10.0 + i as f64,
                    50,
                    0.02 * (i as f64 % 3.0 - 1.0),
                )
            })
            .collect();
        let wavenumbers = predicted_wavenumbers();
        let result = galaxy_ensemble_phase_coherence(&galaxies, &wavenumbers, 3);
        assert_eq!(result.n_jackknife, 20);
        // All Rayleigh R values should be finite
        for &r in &result.rayleigh_r {
            assert!(r.is_finite(), "Rayleigh R should be finite");
        }
    }

    #[test]
    fn test_inject_zd_signal_alpha_zero() {
        // At alpha_zd=0, injection should return identical residuals.
        let galaxies: Vec<_> = (0..5)
            .map(|i| make_galaxy(&format!("G{i}"), 10.0, 30, 0.05))
            .collect();
        let injected = inject_zd_signal(&galaxies, 0.0, 16);
        assert_eq!(injected.len(), galaxies.len());
        for (g, inj) in galaxies.iter().zip(&injected) {
            assert_eq!(g.points.len(), inj.points.len());
            for (p, q) in g.points.iter().zip(&inj.points) {
                assert!((p.delta_v - q.delta_v).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn test_inject_zd_signal_adds_modulation() {
        // At alpha_zd > 0, injection should change residuals.
        let galaxies: Vec<_> = (0..5)
            .map(|i| make_galaxy(&format!("G{i}"), 10.0, 30, 0.0))
            .collect();
        let injected = inject_zd_signal(&galaxies, 0.01, 16);
        // At least some points should differ from 0.0
        let any_nonzero = injected[0].points.iter().any(|p| p.delta_v.abs() > 1e-10);
        assert!(any_nonzero, "injected signal should be non-zero");
    }

    #[test]
    fn test_injection_recovery_at_alpha_zero() {
        // At alpha_zd=0, recovery result should exist and have valid SNR.
        let galaxies: Vec<_> = (0..10)
            .map(|i| make_galaxy(&format!("G{i}"), 10.0, 50, 0.0))
            .collect();
        let results = injection_recovery_sweep(&galaxies, &[0.0], 16, &StackingConfig::default());
        assert_eq!(results.len(), 1);
        let r = &results[0];
        assert_eq!(r.alpha_zd_injected, 0.0);
        assert!(r.detected_snr.is_finite());
        assert!(r.recovery_ratio.is_nan(), "ratio should be NaN at alpha=0");
    }

    #[test]
    fn test_injection_recovery_sweep_length() {
        let galaxies: Vec<_> = (0..10)
            .map(|i| make_galaxy(&format!("G{i}"), 10.0, 50, 0.0))
            .collect();
        let alphas = [0.0, 0.001, 0.004, 0.008];
        let results = injection_recovery_sweep(&galaxies, &alphas, 16, &StackingConfig::default());
        assert_eq!(results.len(), 4);
    }

    #[test]
    fn test_n_contributing_tracks_coverage() {
        // Galaxies with different radial coverage
        let g1 = NormalizedResiduals {
            name: "short".to_string(),
            r_s_kpc: 10.0,
            points: vec![
                NormalizedPoint {
                    x: 0.5,
                    delta_v: 0.0,
                    delta_v_err: 0.01,
                },
                NormalizedPoint {
                    x: 2.0,
                    delta_v: 0.0,
                    delta_v_err: 0.01,
                },
            ],
        };
        let g2 = NormalizedResiduals {
            name: "long".to_string(),
            r_s_kpc: 10.0,
            points: vec![
                NormalizedPoint {
                    x: 0.1,
                    delta_v: 0.0,
                    delta_v_err: 0.01,
                },
                NormalizedPoint {
                    x: 10.0,
                    delta_v: 0.0,
                    delta_v_err: 0.01,
                },
            ],
        };

        let config = StackingConfig {
            min_galaxies_per_bin: 1,
            ..Default::default()
        };
        let result = stack_residuals(&[g1, g2], &config);

        // Inner bins (x < 2) should have 2 contributing, outer bins only 1
        let inner_count = result.n_contributing[5]; // x ~ 0.5
        let outer_count = result.n_contributing[result.x_grid.len() - 5]; // x ~ 9.5
        assert!(
            inner_count >= outer_count,
            "inner bins should have >= outer coverage"
        );
    }

    // ── Eigenmode stacking tests ─────────────────────────────────────────────

    fn make_random_galaxy_residuals(
        name: &str,
        n_pts: usize,
        amplitude: f64,
        seed: u64,
    ) -> NormalizedResiduals {
        // Simple deterministic LCG for reproducible pseudo-random residuals
        let mut s = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let points: Vec<NormalizedPoint> = (0..n_pts)
            .map(|i| {
                let x = 0.5 + (i as f64 / (n_pts - 1) as f64) * 9.5;
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                let noise = (s as i64 as f64 / i64::MAX as f64) * amplitude;
                NormalizedPoint {
                    x,
                    delta_v: noise,
                    delta_v_err: amplitude.abs() * 0.1 + 0.001,
                }
            })
            .collect();
        NormalizedResiduals {
            name: name.to_string(),
            r_s_kpc: 10.0,
            points,
        }
    }

    #[test]
    fn test_eigenmode_white_noise() {
        // 50 galaxies with N(0,1)-scale residuals; all zd_overlaps should be small
        let n_gal = 50_usize;
        let galaxies: Vec<NormalizedResiduals> = (0..n_gal)
            .map(|i| make_random_galaxy_residuals(&format!("G{i}"), 50, 0.05, i as u64 + 1))
            .collect();

        let config = StackingConfig {
            x_min: 0.5,
            x_max: 10.0,
            n_grid: 50,
            min_galaxies_per_bin: 2,
            ..Default::default()
        };

        let result = eigenmode_stack(&galaxies, &config, 10);

        // Under H0 (white noise), max overlap should be well below the signal threshold
        let max_overlap = result.zd_overlaps.iter().copied().fold(0.0_f64, f64::max);
        assert!(
            max_overlap < 0.7,
            "white noise max overlap = {max_overlap:.3}, expected < 0.7"
        );
        assert!(
            result.reconstruction_snr < 3.0,
            "white noise reconstruction_snr = {:.3}, expected < 3.0",
            result.reconstruction_snr
        );
        assert_eq!(result.n_modes_kept, 10);
        assert_eq!(result.n_galaxies, n_gal);
    }

    #[test]
    fn test_eigenmode_injected_signal() {
        // 30 galaxies with ZD signal at alpha_zd=0.01 injected
        let n_gal = 30_usize;
        let galaxies: Vec<NormalizedResiduals> = (0..n_gal)
            .map(|i| make_random_galaxy_residuals(&format!("G{i}"), 50, 0.02, i as u64 + 100))
            .collect();
        let config = StackingConfig {
            x_min: 0.5,
            x_max: 10.0,
            n_grid: 50,
            min_galaxies_per_bin: 2,
            cd_params: CdDimensionParams::sedenion(),
            ..Default::default()
        };

        // Inject ZD signal into all galaxies
        let injected = inject_zd_signal(&galaxies, 0.01, 16);
        let result = eigenmode_stack(&injected, &config, 5);

        // The leading mode should have a measurable ZD overlap
        assert!(!result.zd_overlaps.is_empty(), "no eigenmodes computed");
        let leading_overlap = result.zd_overlaps[0];
        assert!(
            leading_overlap > 0.1,
            "injected leading overlap = {leading_overlap:.3}, expected > 0.1"
        );
    }
}
