use super::{
    refit::load_phase1_models_from_report,
    timing_model::{BinaryFamily, ReleaseBand, TimingModel},
};
use anise::{
    constants::frames::{EARTH_ITRF93, EARTH_J2000, SSB_J2000, SUN_J2000},
    naif::kpl::parser::convert_tpc,
    prelude::{Almanac, Orbit},
};
use anyhow::{Context, Result, anyhow, bail};
use faer::{
    Mat as FaerMat, Side,
    prelude::{Solve, SolveLstsq},
};
use hifitime::{Epoch, TimeScale};
use nalgebra::{DMatrix, DVector};
use rayon::prelude::*;
use std::{
    collections::{BTreeMap, BTreeSet},
    f64::consts::PI,
    fs,
    path::{Path, PathBuf},
};

const C_KM_PER_S: f64 = 299_792.458;
// Standard TEMPO2 dispersion constant: K_DM = 4148.808 us MHz^2 (pc/cm^3)^-1
// Expressed in seconds: K_DM = 4.148808e-3 s MHz^2 (pc/cm^3)^-1.
// WHY: the legacy nanograv_refit.rs used 4_148_808 us (= 4.148808 s) which is 1000x
// the correct value. The new independent engine must use the physically correct constant
// here so that both the forward-model DM delay and the design-matrix DM derivative are
// consistent with the actual TEMPO2 standard.
const DM_DELAY_S_PER_MHZ2_DM: f64 = 4.148_808e-3;
const SOLAR_MASS_TIME_S: f64 = 4.925_490_947e-6;
const GM_SUN_KM3_S2: f64 = 1.327_124_400_18e11;
const OBLIQUITY_RAD: f64 = 23.439_291_111_f64.to_radians();
const AU_KM: f64 = 149_597_870.7;
const ARCSEC_PER_RAD: f64 = 206_264.806_247_096_36;
const JULIAN_YEAR_S: f64 = 31_557_600.0;
const MAS_TO_RAD: f64 = PI / (180.0 * 3_600.0 * 1_000.0);

// Pure scalar / small-array math helpers (wrap_cycles, fract, dot3,
// norm3, matern_three_halves, gaussian_kernel, median_value,
// fractional_improvement, synthesis_score) live in the
// `scalar_math` submodule.
mod scalar_math;
use scalar_math::{
    dot3, fractional_improvement, gaussian_kernel, matern_three_halves, median_value, norm3,
    solve_kepler, synthesis_score, true_anomaly_from_eccentric, wrap_cycles,
};

// RMS / weighted-RMS statistics + small helpers (rms_from_iter,
// optional_rms, collect_option_values, weighted_rms_from_rows,
// weighted_rms_from_rows_gls, closest_residual, row_dot) live in
// the `rms_stats` submodule.
mod rms_stats;
use rms_stats::{
    closest_residual, collect_option_values, optional_rms, rms_from_iter, row_dot,
    weighted_rms_from_rows, weighted_rms_from_rows_gls,
};

// nalgebra <-> faer adapters + Cholesky-fallback solver +
// Woodbury inverse-covariance application (dmatrix_to_faer,
// dvector_to_faer_col, faer_col_to_dvector, faer_column_norms,
// scale_faer_columns, solve_square_system_faer,
// apply_inverse_covariance_to_matrix) live in the `faer_adapters`
// submodule.
mod faer_adapters;
use faer_adapters::{
    apply_inverse_covariance_to_matrix, dmatrix_to_faer, dvector_to_faer_col, faer_col_to_dvector,
    faer_column_norms, scale_faer_columns, solve_square_system_faer,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
pub enum SiteId {
    Arecibo,
    GreenBank,
    Vla,
}

impl SiteId {
    pub fn from_token(token: &str) -> Result<Self> {
        match token {
            "arecibo" | "AO" | "3" => Ok(Self::Arecibo),
            "gbt" | "GB" | "1" => Ok(Self::GreenBank),
            "vla" | "VL" | "6" => Ok(Self::Vla),
            other => bail!("unsupported observatory token {other}"),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Arecibo => "arecibo",
            Self::GreenBank => "gbt",
            Self::Vla => "vla",
        }
    }

    pub fn itrf_position_m(self) -> [f64; 3] {
        match self {
            Self::Arecibo => [2_390_487.08, -5_564_731.357, 1_994_720.633],
            Self::GreenBank => [882_589.289, -4_924_872.368, 3_943_729.418],
            Self::Vla => [-1_601_192.0, -5_041_981.4, 3_554_871.4],
        }
    }
}

#[derive(Debug, Clone)]
pub struct TopocentricToa {
    pub name: String,
    pub frequency_mhz: f64,
    pub mjd_utc: f64,
    pub uncertainty_us: f64,
    pub site: SiteId,
    pub flags: BTreeMap<String, String>,
    pub pp_dm: Option<f64>,
    pub pp_dme: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct IndependentObservation {
    pub solution_id: String,
    pub pulsar_id: String,
    pub site: SiteId,
    pub subgroup: String,
    pub flags: BTreeMap<String, String>,
    pub mjd_utc: f64,
    pub mjd_tdb: f64,
    pub frequency_mhz: f64,
    pub uncertainty_us: f64,
    pub residual_before_us: f64,
    pub dm_model: f64,
    pub pp_dm: Option<f64>,
    pub pp_dme: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct IndependentDataset {
    pub model: TimingModel,
    pub requested_ephem: String,
    pub ephem_used: String,
    pub dominant_subgroup: String,
    pub dominant_subgroup_count: usize,
    pub all_subgroups: Vec<String>,
    pub total_toa_count: usize,
    pub observations: Vec<IndependentObservation>,
    pub simplification_notes: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct IndependentFitSummary {
    pub residual_rms_before_us: f64,
    pub residual_rms_after_wls_us: f64,
    pub residual_rms_after_gls_us: f64,
    pub weighted_rms_before_us: f64,
    pub weighted_rms_after_wls_us: f64,
    pub weighted_rms_after_gls_us: f64,
    pub observation_count: usize,
    pub dm_observation_count: usize,
    pub dm_rms_before: Option<f64>,
    pub dm_rms_after_wls: Option<f64>,
    pub dm_rms_after_gls: Option<f64>,
    pub gls_ridge_factor: f64,
    pub raw_improvement_wls_frac: f64,
    pub raw_improvement_gls_frac: f64,
    pub weighted_improvement_wls_frac: f64,
    pub weighted_improvement_gls_frac: f64,
    pub dm_improvement_wls_frac: Option<f64>,
    pub dm_improvement_gls_frac: Option<f64>,
    pub synthesis_score_wls: f64,
    pub synthesis_score_gls: f64,
    pub recommended_solver: String,
    pub acceptance_track: String,
    pub metric_conflict: bool,
}

#[derive(Debug, Clone)]
pub struct IndependentRefitRow {
    pub solution_id: String,
    pub pulsar_id: String,
    pub site: String,
    pub subgroup: String,
    pub mjd_utc: f64,
    pub mjd_tdb: f64,
    pub frequency_mhz: f64,
    pub uncertainty_us: f64,
    pub residual_before_us: f64,
    pub residual_after_wls_us: f64,
    pub residual_after_gls_us: f64,
    pub dm_model: f64,
    pub pp_dm: Option<f64>,
    pub pp_dme: Option<f64>,
    pub dm_residual_before: Option<f64>,
    pub dm_residual_after_wls: Option<f64>,
    pub dm_residual_after_gls: Option<f64>,
}

#[derive(Debug, Clone)]
pub struct IndependentRefitResult {
    pub dataset: IndependentDataset,
    pub parameter_names: Vec<String>,
    pub wls_coefficients: Vec<f64>,
    pub gls_coefficients: Vec<f64>,
    pub summary: IndependentFitSummary,
    pub covariance_calibration: StructuredCovarianceCalibration,
    pub rows: Vec<IndependentRefitRow>,
}

#[derive(Debug, Clone)]
struct DelayContext {
    earth_barycentric_km: [f64; 3],
    site_barycentric_km: [f64; 3],
    sun_from_earth_km: [f64; 3],
    sky_unit: [f64; 3],
    sky_unit_ecliptic: [f64; 3],
}

#[derive(Debug, Clone)]
struct StaticDelayContext {
    earth_barycentric_km: [f64; 3],
    site_barycentric_km: [f64; 3],
    sun_from_earth_km: [f64; 3],
}

#[derive(Debug, Clone)]
struct ObservationSolveCache {
    toa: TopocentricToa,
    epoch_tdb: Epoch,
    static_delay: StaticDelayContext,
    dm_delay_coeff_s_per_dm: f64,
    phase_sigma_s: f64,
    dm_sigma: Option<f64>,
    jump_matches: Vec<bool>,
    dmjump_matches: Vec<bool>,
}

#[derive(Debug, Clone)]
struct JointSystem {
    design: DMatrix<f64>,
    response: DVector<f64>,
    sigma: Vec<f64>,
    dm_row_of_phase: Vec<Option<usize>>,
    dm_observation_count: usize,
}

#[derive(Debug, Clone)]
struct StructuredCovariance {
    low_rank: FaerMat<f64>,
    inv_diagonal: Vec<f64>,
    inv_low_rank: FaerMat<f64>,
    middle: FaerMat<f64>,
}

#[derive(Debug, Clone)]
pub struct StructuredCovarianceCalibration {
    pub preset_name: String,
    pub tactical_lane: String,
    pub provisional_acceptance_policy: String,
    pub corr_length_days: f64,
    pub red_noise_fraction: f64,
    pub phase_white_floor_s: f64,
    pub dm_white_floor: f64,
    pub phase_amp_s: f64,
    pub dm_amp: f64,
    pub phase_fourier_harmonics: usize,
    pub phase_short_basis_count: usize,
    pub phase_long_basis_count: usize,
    pub dm_fourier_harmonics: usize,
    pub dm_short_basis_count: usize,
    pub dm_long_basis_count: usize,
    pub ecorr_basis_count: usize,
}

#[derive(Debug, Clone)]
struct CovariancePreset {
    name: &'static str,
    tactical_lane: &'static str,
    provisional_acceptance_policy: &'static str,
    corr_length_days: f64,
    red_noise_fraction: f64,
    phase_amp_scale: f64,
    dm_amp_scale: f64,
    phase_fourier_harmonics: Option<usize>,
    phase_short_basis_count: Option<usize>,
    phase_long_basis_count: Option<usize>,
    dm_fourier_harmonics: Option<usize>,
    dm_short_basis_count: Option<usize>,
    dm_long_basis_count: Option<usize>,
}

struct StructuredCovarianceInputs {
    n_phase: usize,
    n_dm: usize,
    mjd_span: f64,
    corr_length_days: f64,
    red_noise_fraction: f64,
    phase_formal_floor: f64,
    dm_formal_floor: f64,
    wls_phase_rms_floor_s: f64,
    dm_rms_floor: f64,
}

pub fn tactical_lane_for_pulsar(pulsar_id: &str) -> &'static str {
    match pulsar_id {
        "J1903+0327" | "J2214+3000" => "gls-first",
        _ => "wls-first",
    }
}

#[derive(Debug, Clone, Copy)]
struct OrbitalState {
    pb_s: f64,
    dt_s: f64,
    a1_lt_s: f64,
    ecc: f64,
    eccentric_anomaly: f64,
    true_anomaly: f64,
    gamma_s: f64,
    mean_motion_rad_s: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct DdkCorrections {
    a1_offset_lt_s: f64,
    omega_offset_rad: f64,
    sini_override: Option<f64>,
}

#[derive(Debug, Clone, Copy)]
struct AstrometricState {
    sky_unit_ecliptic: [f64; 3],
    sky_unit_equatorial: [f64; 3],
    parallax_distance_km: Option<f64>,
}

pub struct TimingEphemeris {
    almanac: Almanac,
    ephemeris_name: String,
}

trait TimingModelExt {
    fn parameter_term_local(&self, name: &str) -> Option<&super::timing_model::ParameterTerm>;
    fn parameter_value_local(&self, name: &str) -> Option<f64>;
    fn parameter_bool_local(&self, name: &str) -> Option<bool>;
}

impl TimingModelExt for TimingModel {
    fn parameter_term_local(&self, name: &str) -> Option<&super::timing_model::ParameterTerm> {
        let astrometry = [
            self.astrometry.raj.as_ref(),
            self.astrometry.decj.as_ref(),
            self.astrometry.elong.as_ref(),
            self.astrometry.elat.as_ref(),
            self.astrometry.pmelong.as_ref(),
            self.astrometry.pmelat.as_ref(),
            self.astrometry.pmra.as_ref(),
            self.astrometry.pmdec.as_ref(),
            self.astrometry.px.as_ref(),
            self.dispersion.dm.as_ref(),
            self.dispersion.dmepoch.as_ref(),
            self.dispersion.dmx_step.as_ref(),
        ];
        astrometry
            .into_iter()
            .flatten()
            .find(|term| term.name == name)
            .or_else(|| {
                self.dispersion
                    .dmx_windows
                    .iter()
                    .filter_map(|window| window.dmx.as_ref())
                    .find(|term| term.name == name)
            })
            .or_else(|| self.spin_terms.iter().find(|term| term.name == name))
            .or_else(|| self.fd_terms.iter().find(|term| term.name == name))
            .or_else(|| self.other_terms.iter().find(|term| term.name == name))
    }

    fn parameter_value_local(&self, name: &str) -> Option<f64> {
        self.parameter_term_local(name).and_then(|term| term.value)
    }

    fn parameter_bool_local(&self, name: &str) -> Option<bool> {
        let term = self.parameter_term_local(name)?;
        match term.raw_value.as_str() {
            "Y" | "y" | "T" | "t" | "true" | "True" | "1" => Some(true),
            "N" | "n" | "F" | "f" | "false" | "False" | "0" => Some(false),
            _ => term.value.map(|value| value != 0.0),
        }
    }
}

impl TimingEphemeris {
    pub fn load_default() -> Result<Self> {
        let ephemeris_path = [
            PathBuf::from("data/external/de440.bsp"),
            PathBuf::from("data/external/de440s.bsp"),
        ]
        .into_iter()
        .find(|candidate| candidate.exists())
        .ok_or_else(|| anyhow!("missing local DE440 ephemeris under data/external"))?;
        let bpc_path =
            PathBuf::from("data/external/nanograv_timing_engine/earth_latest_high_prec.bpc");
        let pck_path = PathBuf::from("data/external/nanograv_timing_engine/pck00011.tpc");
        let gm_path = PathBuf::from("data/external/nanograv_timing_engine/gm_de440.tpc");
        for dependency in [&bpc_path, &pck_path, &gm_path] {
            if !dependency.exists() {
                bail!(
                    "missing timing-engine kernel {} required for Earth orientation",
                    dependency.display()
                );
            }
        }

        let planetary = convert_tpc(
            pck_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 path {}", pck_path.display()))?,
            gm_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 path {}", gm_path.display()))?,
        )
        .map_err(|error| anyhow!("failed to load PCK/TPC planetary data: {error}"))?;
        let almanac = Almanac::new(
            ephemeris_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 path {}", ephemeris_path.display()))?,
        )
        .map_err(|error| anyhow!("failed to initialize {}: {error}", ephemeris_path.display()))?
        .load(
            bpc_path
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 path {}", bpc_path.display()))?,
        )
        .map_err(|error| anyhow!("failed to load {}: {error}", bpc_path.display()))?
        .with_planetary_data(planetary);
        Ok(Self {
            almanac,
            ephemeris_name: ephemeris_path
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or("de440")
                .to_string(),
        })
    }

    fn static_delay_context(&self, epoch_tdb: Epoch, site: SiteId) -> Result<StaticDelayContext> {
        let earth_state = self
            .almanac
            .translate_geometric(EARTH_J2000, SSB_J2000, epoch_tdb)
            .map_err(|error| anyhow!("earth barycentric state lookup failed: {error}"))?;
        let earth_barycentric_km = [
            earth_state.radius_km[0],
            earth_state.radius_km[1],
            earth_state.radius_km[2],
        ];

        let site_itrf = site.itrf_position_m();
        let site_state_itrf = Orbit::from_position(
            site_itrf[0] / 1_000.0,
            site_itrf[1] / 1_000.0,
            site_itrf[2] / 1_000.0,
            epoch_tdb,
            EARTH_ITRF93,
        );
        let site_state_ssb = self
            .almanac
            .transform_to(site_state_itrf, SSB_J2000, None)
            .map_err(|error| anyhow!("site barycentric transform failed: {error}"))?;
        let site_barycentric_km = [
            site_state_ssb.radius_km[0],
            site_state_ssb.radius_km[1],
            site_state_ssb.radius_km[2],
        ];

        let sun_state = self
            .almanac
            .translate_geometric(SUN_J2000, EARTH_J2000, epoch_tdb)
            .map_err(|error| anyhow!("sun geocentric state lookup failed: {error}"))?;
        let sun_from_earth_km = [
            sun_state.radius_km[0],
            sun_state.radius_km[1],
            sun_state.radius_km[2],
        ];

        Ok(StaticDelayContext {
            earth_barycentric_km,
            site_barycentric_km,
            sun_from_earth_km,
        })
    }

    pub fn ephemeris_name(&self) -> &str {
        &self.ephemeris_name
    }
}

fn build_observation_cache(
    model: &TimingModel,
    observation: &IndependentObservation,
    ephemeris: &TimingEphemeris,
) -> Result<ObservationSolveCache> {
    let toa = TopocentricToa {
        name: observation.solution_id.clone(),
        frequency_mhz: observation.frequency_mhz,
        mjd_utc: observation.mjd_utc,
        uncertainty_us: observation.uncertainty_us,
        site: observation.site,
        flags: observation.flags.clone(),
        pp_dm: observation.pp_dm,
        pp_dme: observation.pp_dme,
    };
    let epoch_tdb = Epoch::from_mjd_utc(observation.mjd_utc).to_time_scale(TimeScale::TDB);
    let static_delay = ephemeris.static_delay_context(epoch_tdb, observation.site)?;
    let dm_delay_coeff_s_per_dm =
        DM_DELAY_S_PER_MHZ2_DM / (observation.frequency_mhz * observation.frequency_mhz);
    let phase_sigma_s = effective_phase_sigma_seconds(model, observation);
    let dm_sigma = observation
        .pp_dme
        .map(|pp_dme| effective_dm_sigma(model, observation, pp_dme));
    let jump_matches = model
        .jumps
        .iter()
        .map(|term| tagged_term_matches_observation(term, observation))
        .collect();
    let dmjump_matches = model
        .dmjumps
        .iter()
        .map(|term| tagged_term_matches_observation(term, observation))
        .collect();
    Ok(ObservationSolveCache {
        toa,
        epoch_tdb,
        static_delay,
        dm_delay_coeff_s_per_dm,
        phase_sigma_s,
        dm_sigma,
        jump_matches,
        dmjump_matches,
    })
}

pub fn build_phase1_independent_datasets(
    root: &Path,
    phase1_report: &Path,
    ephemeris: &TimingEphemeris,
) -> Result<Vec<IndependentDataset>> {
    let models = load_phase1_models_from_report(root, phase1_report, ReleaseBand::Wideband)?;
    let mut datasets = Vec::new();
    for model in models {
        datasets.push(build_independent_dataset(root, &model, ephemeris)?);
    }
    Ok(datasets)
}

pub fn build_independent_dataset(
    root: &Path,
    model: &TimingModel,
    ephemeris: &TimingEphemeris,
) -> Result<IndependentDataset> {
    let tim_path = root
        .join("wideband/tim")
        .join(format!("{}.tim", model.solution_id));
    let all_toas = parse_tempo2_toas(&tim_path)?;
    if all_toas.is_empty() {
        bail!("no TOAs parsed from {}", tim_path.display());
    }
    let (dominant_subgroup, dominant_subgroup_count) = dominant_subgroup(&all_toas)?;
    let total_toa_count = all_toas.len();
    let all_subgroups = all_toas
        .iter()
        .filter_map(|toa| toa.flags.get("-f").cloned())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect::<Vec<_>>();

    let mut simplification_notes = vec![
        "Independent residuals are reconstructed from topocentric .tim TOAs, clock conversion, barycentric geometry, astrometric proper motion/parallax, and family-specific binary forward models; release *.res rows are not consumed.".to_string(),
        "Timescale conversion uses hifitime, and Earth orientation uses cached NAIF BPC/PCK kernels via anise.".to_string(),
        "Wideband fitting uses a stacked phase-plus-DM system that treats pp_dm as an observed DM datum rather than as a released residual baseline.".to_string(),
        "Linear solves apply soft Gaussian priors on parameter offsets, scaled by published .par uncertainties, to keep the local tangent-space refit in the physically relevant neighborhood of the released solution.".to_string(),
        "GLS covariance combines an inflated white-noise floor with short-timescale Matern and long-timescale Fourier process terms, plus a coupled DM process in the joint phase-plus-DM system.".to_string(),
        "All available wideband frontend/backend groups are retained; release JUMP, DMJUMP, EFAC, EQUAD, DMEFAC, DMEQUAD, and ECORR terms are interpreted through their selectors instead of collapsing to the dominant -f subgroup only.".to_string(),
    ];
    if matches!(
        model.binary_family,
        Some(BinaryFamily::Ell1 | BinaryFamily::Ell1h)
    ) {
        simplification_notes.push(
            "ELL1-family forward model follows cached PINT small-eccentricity formulas through O(e^3); ELL1H now matches the cached branch structure: H3-only uses the truncated higher-harmonic approximation, H3+H4 uses the higher-harmonic approximation with stigma = H4/H3, and H3+STIGMA uses the exact orthometric form.".to_string(),
        );
    }
    if matches!(model.binary_family, Some(BinaryFamily::Bt)) {
        simplification_notes.push(
            "BT-family forward model follows the cached PINT Blandford-Teukolsky structure, including the multiplicative delayR factor.".to_string(),
        );
    }
    if matches!(model.binary_family, Some(BinaryFamily::Dd)) {
        simplification_notes.push(
            "DD-family forward model follows the cached PINT Damour-Deruelle inverse, Shapiro, and aberration components.".to_string(),
        );
    }
    if matches!(model.binary_family, Some(BinaryFamily::Ddk)) {
        simplification_notes.push(
            "DDK-family forward model applies Kopeikin proper-motion and annual-parallax corrections to A1, omega, and inclination on top of the DD delay stack, and now respects the release K96 switch instead of forcing proper-motion corrections unconditionally.".to_string(),
        );
    }
    if model.ephem.as_deref() != Some("DE440") {
        simplification_notes.push(format!(
            "Requested EPHEM {:?} differs from local ephemeris {}; this mismatch is disclosed explicitly.",
            model.ephem,
            ephemeris.ephemeris_name()
        ));
    }

    // WHY: per-TOA residual computation is embarrassingly parallel -- each call is
    // independent (read-only model and ephemeris). Rayon par_iter gives near-linear
    // speedup on the ~1000-2000 TOA baseline pass that dominates dataset construction.
    let mut observations = all_toas
        .par_iter()
        .map(|toa| {
            let subgroup = toa
                .flags
                .get("-f")
                .cloned()
                .unwrap_or_else(|| "unlabeled".to_string());
            let epoch_tdb = Epoch::from_mjd_utc(toa.mjd_utc).to_time_scale(TimeScale::TDB);
            let static_delay = ephemeris.static_delay_context(epoch_tdb, toa.site)?;
            let residual_s = independent_residual_seconds_cached(
                model,
                toa,
                epoch_tdb,
                &static_delay,
                &BTreeMap::new(),
            )?;
            Ok(IndependentObservation {
                solution_id: model.solution_id.clone(),
                pulsar_id: model.pulsar_id.clone(),
                site: toa.site,
                subgroup,
                flags: toa.flags.clone(),
                mjd_utc: toa.mjd_utc,
                mjd_tdb: epoch_tdb.to_jde_tdb_days() - 2_400_000.5,
                frequency_mhz: toa.frequency_mhz,
                uncertainty_us: toa.uncertainty_us,
                residual_before_us: residual_s * 1.0e6,
                dm_model: dispersion_measure(model, &BTreeMap::new(), toa.mjd_utc),
                pp_dm: toa.pp_dm,
                pp_dme: toa.pp_dme,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    observations.sort_by(|left, right| left.mjd_utc.total_cmp(&right.mjd_utc));

    Ok(IndependentDataset {
        model: model.clone(),
        requested_ephem: model.ephem.clone().unwrap_or_else(|| "unknown".to_string()),
        ephem_used: ephemeris.ephemeris_name().to_string(),
        dominant_subgroup,
        dominant_subgroup_count,
        all_subgroups,
        total_toa_count,
        observations,
        simplification_notes,
    })
}

fn covariance_preset_for_dataset(
    dataset: &IndependentDataset,
    default_corr_length_days: f64,
    default_red_noise_fraction: f64,
) -> CovariancePreset {
    match dataset.model.pulsar_id.as_str() {
        "J1903+0327" => CovariancePreset {
            name: "dd_gls_first_targeted",
            tactical_lane: "gls-first",
            provisional_acceptance_policy: "synthesis",
            corr_length_days: 180.0,
            red_noise_fraction: 0.70,
            phase_amp_scale: 1.35,
            dm_amp_scale: 1.10,
            phase_fourier_harmonics: Some(5),
            phase_short_basis_count: Some(24),
            phase_long_basis_count: Some(8),
            dm_fourier_harmonics: Some(4),
            dm_short_basis_count: Some(8),
            dm_long_basis_count: Some(4),
        },
        "J2214+3000" => CovariancePreset {
            name: "ell1_gls_first_targeted",
            tactical_lane: "gls-first",
            provisional_acceptance_policy: "synthesis",
            corr_length_days: 150.0,
            red_noise_fraction: 0.65,
            phase_amp_scale: 1.25,
            dm_amp_scale: 1.35,
            phase_fourier_harmonics: Some(5),
            phase_short_basis_count: Some(24),
            phase_long_basis_count: Some(8),
            dm_fourier_harmonics: Some(4),
            dm_short_basis_count: Some(8),
            dm_long_basis_count: Some(4),
        },
        "J0709+0458" => CovariancePreset {
            name: "dd_wls_first_stable",
            tactical_lane: "wls-first",
            provisional_acceptance_policy: "both",
            corr_length_days: 90.0,
            red_noise_fraction: 0.40,
            phase_amp_scale: 0.75,
            dm_amp_scale: 0.85,
            phase_fourier_harmonics: Some(4),
            phase_short_basis_count: Some(12),
            phase_long_basis_count: Some(4),
            dm_fourier_harmonics: Some(3),
            dm_short_basis_count: Some(4),
            dm_long_basis_count: Some(3),
        },
        "J1312+0051" => CovariancePreset {
            name: "bt_wls_first_compact",
            tactical_lane: "wls-first",
            provisional_acceptance_policy: "synthesis",
            corr_length_days: 75.0,
            red_noise_fraction: 0.35,
            phase_amp_scale: 0.70,
            dm_amp_scale: 0.80,
            phase_fourier_harmonics: Some(4),
            phase_short_basis_count: Some(12),
            phase_long_basis_count: Some(4),
            dm_fourier_harmonics: Some(3),
            dm_short_basis_count: Some(4),
            dm_long_basis_count: Some(3),
        },
        "J1713+0747" => CovariancePreset {
            name: "ddk_wls_first_longspan",
            tactical_lane: "wls-first",
            provisional_acceptance_policy: "synthesis",
            corr_length_days: 180.0,
            red_noise_fraction: 0.45,
            phase_amp_scale: 0.80,
            dm_amp_scale: 0.80,
            phase_fourier_harmonics: Some(4),
            phase_short_basis_count: Some(24),
            phase_long_basis_count: Some(10),
            dm_fourier_harmonics: Some(4),
            dm_short_basis_count: Some(10),
            dm_long_basis_count: Some(5),
        },
        "J2317+1439" => CovariancePreset {
            name: "ell1h_wls_first_closure",
            tactical_lane: "wls-first",
            provisional_acceptance_policy: "synthesis",
            corr_length_days: 120.0,
            red_noise_fraction: 0.40,
            phase_amp_scale: 0.80,
            dm_amp_scale: 0.90,
            phase_fourier_harmonics: Some(4),
            phase_short_basis_count: Some(20),
            phase_long_basis_count: Some(8),
            dm_fourier_harmonics: Some(3),
            dm_short_basis_count: Some(8),
            dm_long_basis_count: Some(4),
        },
        _ => CovariancePreset {
            name: "generic",
            tactical_lane: tactical_lane_for_pulsar(&dataset.model.pulsar_id),
            provisional_acceptance_policy: "synthesis",
            corr_length_days: default_corr_length_days,
            red_noise_fraction: default_red_noise_fraction,
            phase_amp_scale: 1.0,
            dm_amp_scale: 1.0,
            phase_fourier_harmonics: None,
            phase_short_basis_count: None,
            phase_long_basis_count: None,
            dm_fourier_harmonics: None,
            dm_short_basis_count: None,
            dm_long_basis_count: None,
        },
    }
}

pub fn solve_independent_refit(
    dataset: &IndependentDataset,
    ephemeris: &TimingEphemeris,
    gls_corr_length_days: f64,
    gls_red_noise_fraction: f64,
) -> Result<IndependentRefitResult> {
    let parameter_names = fit_parameter_names(&dataset.model, &dataset.observations);
    if parameter_names.is_empty() {
        bail!(
            "no independent fit parameters for {}",
            dataset.model.solution_id
        );
    }
    let covariance_preset =
        covariance_preset_for_dataset(dataset, gls_corr_length_days, gls_red_noise_fraction);
    let joint = build_joint_system(dataset, ephemeris, &parameter_names)?;
    let wls = solve_weighted_least_squares(&joint.design, &joint.response, &joint.sigma)?;
    // Compute WLS post-fit RMS over phase rows to scale the GLS covariance.
    // WHY: formal TOA uncertainties (~1 us) are far smaller than independent-engine
    // residuals (~100-1000 us). Using formal sigma in the covariance amplifies noise
    // instead of suppressing it. We scale the red-noise term to match the actual
    // WLS residual level so the GLS whitening operates at the right noise floor.
    let n_phase = dataset.observations.len();
    let wls_phase_rms_s = {
        let sumsq: f64 = (0..n_phase)
            .map(|i| {
                let r = joint.response[i] - row_dot(&joint.design, i, &wls);
                r * r
            })
            .sum();
        if n_phase > 0 {
            (sumsq / n_phase as f64).sqrt()
        } else {
            0.0
        }
    };
    let dm_rms_floor = {
        let dm_values = joint
            .dm_row_of_phase
            .iter()
            .flatten()
            .map(|row_index| joint.response[*row_index])
            .collect::<Vec<_>>();
        optional_rms(&dm_values).unwrap_or(0.0)
    };
    let (covariance, calibration, mut gls_ridge_factor) = build_joint_gls_covariance(
        dataset,
        &joint,
        &covariance_preset,
        wls_phase_rms_s,
        dm_rms_floor,
    );
    let mut gls = solve_generalized_least_squares(&joint.design, &joint.response, &covariance)?;
    let (mut rows, mut summary) =
        build_rows_and_summary(dataset, &joint, &wls, &gls, gls_ridge_factor);
    let gls_bad_residual =
        summary.residual_rms_after_gls_us > 3.0 * summary.residual_rms_after_wls_us;
    let gls_bad_weighted =
        summary.weighted_rms_after_gls_us > 3.0 * summary.weighted_rms_after_wls_us;
    let gls_bad_dm = match (summary.dm_rms_after_gls, summary.dm_rms_after_wls) {
        (Some(gls_dm), Some(wls_dm)) => gls_dm > 20.0 * wls_dm.max(1.0e-12),
        _ => false,
    };
    if gls_bad_residual || gls_bad_weighted || gls_bad_dm {
        gls = wls.clone();
        gls_ridge_factor = -1.0;
        let rebuilt = build_rows_and_summary(dataset, &joint, &wls, &gls, gls_ridge_factor);
        rows = rebuilt.0;
        summary = rebuilt.1;
    }

    Ok(IndependentRefitResult {
        dataset: dataset.clone(),
        parameter_names,
        wls_coefficients: wls.iter().copied().collect(),
        gls_coefficients: gls.iter().copied().collect(),
        summary,
        covariance_calibration: calibration,
        rows,
    })
}

fn build_rows_and_summary(
    dataset: &IndependentDataset,
    joint: &JointSystem,
    wls: &DVector<f64>,
    gls: &DVector<f64>,
    gls_ridge_factor: f64,
) -> (Vec<IndependentRefitRow>, IndependentFitSummary) {
    let mut rows = Vec::new();
    for (phase_row, observation) in dataset.observations.iter().enumerate() {
        let before = joint.response[phase_row];
        let modeled_wls = row_dot(&joint.design, phase_row, wls);
        let modeled_gls = row_dot(&joint.design, phase_row, gls);
        let dm_row = joint.dm_row_of_phase[phase_row];
        let (dm_before, dm_after_wls, dm_after_gls) = if let Some(dm_row_index) = dm_row {
            let dm_before = joint.response[dm_row_index];
            let dm_after_wls = dm_before - row_dot(&joint.design, dm_row_index, wls);
            let dm_after_gls = dm_before - row_dot(&joint.design, dm_row_index, gls);
            (Some(dm_before), Some(dm_after_wls), Some(dm_after_gls))
        } else {
            (None, None, None)
        };
        rows.push(IndependentRefitRow {
            solution_id: observation.solution_id.clone(),
            pulsar_id: observation.pulsar_id.clone(),
            site: observation.site.as_str().to_string(),
            subgroup: observation.subgroup.clone(),
            mjd_utc: observation.mjd_utc,
            mjd_tdb: observation.mjd_tdb,
            frequency_mhz: observation.frequency_mhz,
            uncertainty_us: observation.uncertainty_us,
            residual_before_us: before * 1.0e6,
            residual_after_wls_us: (before - modeled_wls) * 1.0e6,
            residual_after_gls_us: (before - modeled_gls) * 1.0e6,
            dm_model: observation.dm_model,
            pp_dm: observation.pp_dm,
            pp_dme: observation.pp_dme,
            dm_residual_before: dm_before,
            dm_residual_after_wls: dm_after_wls,
            dm_residual_after_gls: dm_after_gls,
        });
    }
    let dm_before = collect_option_values(rows.iter().map(|row| row.dm_residual_before));
    let dm_after_wls = collect_option_values(rows.iter().map(|row| row.dm_residual_after_wls));
    let dm_after_gls = collect_option_values(rows.iter().map(|row| row.dm_residual_after_gls));
    let residual_rms_before_us = rms_from_iter(rows.iter().map(|row| row.residual_before_us));
    let residual_rms_after_wls_us = rms_from_iter(rows.iter().map(|row| row.residual_after_wls_us));
    let residual_rms_after_gls_us = rms_from_iter(rows.iter().map(|row| row.residual_after_gls_us));
    let weighted_rms_before_us = weighted_rms_from_rows(&rows, true);
    let weighted_rms_after_wls_us = weighted_rms_from_rows(&rows, false);
    let weighted_rms_after_gls_us = weighted_rms_from_rows_gls(&rows);
    let dm_rms_before = optional_rms(&dm_before);
    let dm_rms_after_wls = optional_rms(&dm_after_wls);
    let dm_rms_after_gls = optional_rms(&dm_after_gls);
    let raw_improvement_wls_frac =
        fractional_improvement(residual_rms_before_us, residual_rms_after_wls_us);
    let raw_improvement_gls_frac =
        fractional_improvement(residual_rms_before_us, residual_rms_after_gls_us);
    let weighted_improvement_wls_frac =
        fractional_improvement(weighted_rms_before_us, weighted_rms_after_wls_us);
    let weighted_improvement_gls_frac =
        fractional_improvement(weighted_rms_before_us, weighted_rms_after_gls_us);
    let dm_improvement_wls_frac = dm_rms_before
        .zip(dm_rms_after_wls)
        .map(|(before, after)| fractional_improvement(before, after));
    let dm_improvement_gls_frac = dm_rms_before
        .zip(dm_rms_after_gls)
        .map(|(before, after)| fractional_improvement(before, after));
    let synthesis_score_wls = synthesis_score(
        raw_improvement_wls_frac,
        weighted_improvement_wls_frac,
        dm_improvement_wls_frac,
    );
    let synthesis_score_gls = synthesis_score(
        raw_improvement_gls_frac,
        weighted_improvement_gls_frac,
        dm_improvement_gls_frac,
    );
    let raw_prefers_gls = raw_improvement_gls_frac > raw_improvement_wls_frac;
    let weighted_prefers_gls = weighted_improvement_gls_frac > weighted_improvement_wls_frac;
    let recommended_solver = if synthesis_score_gls > synthesis_score_wls {
        "gls"
    } else {
        "wls"
    }
    .to_string();
    let metric_conflict = raw_prefers_gls != weighted_prefers_gls;
    let acceptance_track = if !metric_conflict {
        "both"
    } else {
        "synthesis"
    }
    .to_string();
    let summary = IndependentFitSummary {
        residual_rms_before_us,
        residual_rms_after_wls_us,
        residual_rms_after_gls_us,
        weighted_rms_before_us,
        weighted_rms_after_wls_us,
        weighted_rms_after_gls_us,
        observation_count: rows.len(),
        dm_observation_count: joint.dm_observation_count,
        dm_rms_before,
        dm_rms_after_wls,
        dm_rms_after_gls,
        gls_ridge_factor,
        raw_improvement_wls_frac,
        raw_improvement_gls_frac,
        weighted_improvement_wls_frac,
        weighted_improvement_gls_frac,
        dm_improvement_wls_frac,
        dm_improvement_gls_frac,
        synthesis_score_wls,
        synthesis_score_gls,
        recommended_solver,
        acceptance_track,
        metric_conflict,
    };
    (rows, summary)
}

fn build_joint_system(
    dataset: &IndependentDataset,
    ephemeris: &TimingEphemeris,
    parameter_names: &[String],
) -> Result<JointSystem> {
    let n_phase_rows = dataset.observations.len();
    let n_dm_rows = dataset
        .observations
        .iter()
        .filter(|observation| observation.pp_dm.is_some() && observation.pp_dme.is_some())
        .count();
    let observation_cache = dataset
        .observations
        .iter()
        .map(|observation| build_observation_cache(&dataset.model, observation, ephemeris))
        .collect::<Result<Vec<_>>>()?;
    let prior_parameters = parameter_names
        .iter()
        .enumerate()
        .filter_map(|(index, name)| {
            parameter_prior_sigma(&dataset.model, name)
                .filter(|sigma| sigma.is_finite() && *sigma > 0.0)
                .map(|sigma| (index, sigma))
        })
        .collect::<Vec<_>>();
    let total_rows = n_phase_rows + n_dm_rows + prior_parameters.len();
    let ncols = parameter_names.len();
    let mut design = DMatrix::zeros(total_rows, ncols);
    let mut response = DVector::zeros(total_rows);
    let mut sigma = vec![0.0; total_rows];
    let mut dm_row_of_phase = vec![None; n_phase_rows];

    for (row_index, observation) in dataset.observations.iter().enumerate() {
        response[row_index] = observation.residual_before_us * 1.0e-6;
        sigma[row_index] = observation_cache[row_index].phase_sigma_s;
    }
    let mut next_dm_row = n_phase_rows;
    for (phase_row, observation) in dataset.observations.iter().enumerate() {
        if let (Some(pp_dm), Some(pp_dme)) = (observation.pp_dm, observation.pp_dme) {
            response[next_dm_row] = pp_dm - observation.dm_model;
            sigma[next_dm_row] = observation_cache[phase_row]
                .dm_sigma
                .unwrap_or_else(|| effective_dm_sigma(&dataset.model, observation, pp_dme));
            dm_row_of_phase[phase_row] = Some(next_dm_row);
            next_dm_row += 1;
        }
    }
    for (next_prior_row, (column_index, prior_sigma)) in
        (n_phase_rows + n_dm_rows..).zip(prior_parameters.iter())
    {
        design[(next_prior_row, *column_index)] = 1.0;
        response[next_prior_row] = 0.0;
        sigma[next_prior_row] = *prior_sigma;
    }

    // First pass: analytic columns -- O(N_obs) each, cheap.
    // Second pass: numerical-diff columns -- O(N_obs) residual evals x 2, parallelized.
    // WHY: analytic columns (PHASE_OFFSET, DM/DMX, JUMP@, DMJUMP@, FD) are fast
    // sequential fills. Numerical-diff columns require two forward-model calls per
    // observation; collecting them for a parallel rayon sweep gives near-linear speedup
    // without mutating the design matrix concurrently (results are written back serially).
    let mut numerical_diff_params: Vec<(usize, String, f64, f64)> = Vec::new(); // (col, name, step, current)

    for (col_index, name) in parameter_names.iter().enumerate() {
        if name == "PHASE_OFFSET" {
            for row_index in 0..n_phase_rows {
                design[(row_index, col_index)] = 1.0;
            }
            continue;
        }
        if name == "DM" || name.starts_with("DMX_") {
            for (row_index, observation) in dataset.observations.iter().enumerate() {
                let derivative = -observation_cache[row_index].dm_delay_coeff_s_per_dm;
                design[(row_index, col_index)] = if name == "DM"
                    || dmx_window_matches(&dataset.model, name, observation.mjd_utc)
                {
                    derivative
                } else {
                    0.0
                };
            }
            for (phase_row, observation) in dataset.observations.iter().enumerate() {
                let Some(dm_row_index) = dm_row_of_phase[phase_row] else {
                    continue;
                };
                design[(dm_row_index, col_index)] = if name == "DM"
                    || dmx_window_matches(&dataset.model, name, observation.mjd_utc)
                {
                    -1.0
                } else {
                    0.0
                };
            }
            continue;
        }
        if let Some(index) = parse_selector_parameter_index(name, "JUMP@") {
            let jump = dataset.model.jumps.get(index).ok_or_else(|| {
                anyhow!("{} missing jump index {}", dataset.model.solution_id, index)
            })?;
            for (row_index, observation) in dataset.observations.iter().enumerate() {
                if observation_cache[row_index]
                    .jump_matches
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| tagged_term_matches_observation(jump, observation))
                {
                    design[(row_index, col_index)] = 1.0;
                }
            }
            continue;
        }
        if let Some(index) = parse_selector_parameter_index(name, "DMJUMP@") {
            let jump = dataset.model.dmjumps.get(index).ok_or_else(|| {
                anyhow!(
                    "{} missing dmjump index {}",
                    dataset.model.solution_id,
                    index
                )
            })?;
            for (row_index, observation) in dataset.observations.iter().enumerate() {
                let matches = observation_cache[row_index]
                    .dmjump_matches
                    .get(index)
                    .copied()
                    .unwrap_or_else(|| tagged_term_matches_observation(jump, observation));
                if matches {
                    design[(row_index, col_index)] =
                        -observation_cache[row_index].dm_delay_coeff_s_per_dm;
                }
                if let Some(dm_row_index) = dm_row_of_phase[row_index]
                    && matches
                {
                    design[(dm_row_index, col_index)] = -1.0;
                }
            }
            continue;
        }
        // FD terms: analytic derivative d(residual)/d(FD_k) = -(ln(f/f_ref))^k
        // WHY: FD_k shifts the timing residual by FD_k * (ln(f/f_ref))^k seconds, so increasing
        // FD_k decreases the residual by that amount. TEMPO2/PINT reference frequency f_ref = 1 GHz.
        // DM rows are unaffected (FD is a profile-delay correction, not a dispersive DM offset).
        if let Some(k) = parse_fd_order(name) {
            const F_REF_MHZ: f64 = 1000.0;
            for (row_index, observation) in dataset.observations.iter().enumerate() {
                let log_f = (observation.frequency_mhz / F_REF_MHZ).ln();
                design[(row_index, col_index)] = -(log_f.powi(k as i32));
            }
            continue;
        }

        // Queue for parallel numerical differentiation below.
        let step = parameter_step(&dataset.model, name)?;
        let current_value = parameter_current_value(&dataset.model, name)?;
        numerical_diff_params.push((col_index, name.clone(), step, current_value));
    }

    // Parallel numerical differentiation: each parameter column is independent.
    // WHY: independent_residual_seconds is pure (no shared mutable state) and the
    // per-column result is a Vec<f64> that we assemble sequentially afterward.
    let spin_period = spin_period_seconds(&dataset.model);
    let numerical_columns: Result<Vec<(usize, Vec<f64>)>> = numerical_diff_params
        .par_iter()
        .map(|(col_index, name, step, current_value)| {
            let mut positive = BTreeMap::new();
            positive.insert(name.clone(), current_value + step);
            let mut negative = BTreeMap::new();
            negative.insert(name.clone(), current_value - step);
            let values = observation_cache
                .iter()
                .enumerate()
                .map(|(row_index, cache)| {
                    let baseline = response[row_index];
                    let plus = independent_residual_seconds_cached(
                        &dataset.model,
                        &cache.toa,
                        cache.epoch_tdb,
                        &cache.static_delay,
                        &positive,
                    )?;
                    let minus = independent_residual_seconds_cached(
                        &dataset.model,
                        &cache.toa,
                        cache.epoch_tdb,
                        &cache.static_delay,
                        &negative,
                    )?;
                    let adjusted_plus = closest_residual(plus, baseline, spin_period);
                    let adjusted_minus = closest_residual(minus, baseline, spin_period);
                    Ok((adjusted_plus - adjusted_minus) / (2.0 * step))
                })
                .collect::<Result<Vec<f64>>>()?;
            Ok((*col_index, values))
        })
        .collect();
    for (col_index, values) in numerical_columns? {
        for (row_index, value) in values.iter().enumerate() {
            design[(row_index, col_index)] = *value;
        }
    }

    Ok(JointSystem {
        design,
        response,
        sigma,
        dm_row_of_phase,
        dm_observation_count: n_dm_rows,
    })
}

fn fit_parameter_names(
    model: &TimingModel,
    observations: &[IndependentObservation],
) -> Vec<String> {
    let mut names = vec!["PHASE_OFFSET".to_string()];
    if model.parameter_value_local("F0").is_some() {
        names.push("F0".to_string());
    }
    if model.parameter_value_local("F1").is_some() {
        names.push("F1".to_string());
    }
    for name in ["ELONG", "ELAT", "PMELONG", "PMELAT", "PX"] {
        if model.parameter_value_local(name).is_some() {
            names.push(name.to_string());
        }
    }
    if model.dispersion.dm.is_some() {
        names.push("DM".to_string());
    }
    let touched_windows = observations
        .iter()
        .flat_map(|observation| {
            model
                .dispersion
                .dmx_windows
                .iter()
                .filter(move |window| {
                    let Some(start) = window.start_mjd else {
                        return false;
                    };
                    let Some(end) = window.end_mjd else {
                        return false;
                    };
                    start <= observation.mjd_utc && observation.mjd_utc <= end
                })
                .filter_map(|window| window.dmx.as_ref())
                .map(|term| term.name.clone())
                .collect::<Vec<_>>()
        })
        .collect::<BTreeSet<_>>();
    names.extend(touched_windows);
    names.extend(
        model
            .jumps
            .iter()
            .enumerate()
            .filter(|(_, term)| {
                observations
                    .iter()
                    .any(|observation| tagged_term_matches_observation(term, observation))
            })
            .map(|(index, _)| format!("JUMP@{index}")),
    );
    names.extend(
        model
            .dmjumps
            .iter()
            .enumerate()
            .filter(|(_, term)| {
                observations
                    .iter()
                    .any(|observation| tagged_term_matches_observation(term, observation))
            })
            .map(|(index, _)| format!("DMJUMP@{index}")),
    );

    // FD profile-delay terms: analytic columns in the design matrix.
    // WHY: FD_k parameters absorb frequency-dependent profile-delay noise across
    // the wideband receiver; omitting them would leave a systematic frequency-dependent
    // trend in the residuals that corrupts DM and spin-parameter estimates.
    for term in &model.fd_terms {
        if term.value.is_some() {
            names.push(term.name.clone());
        }
    }

    match model.binary_family.as_ref() {
        Some(BinaryFamily::Ell1) | Some(BinaryFamily::Ell1h) => {
            for name in [
                "A1", "TASC", "EPS1", "EPS2", "FB0", "FB1", "FB2", "FB3", "H3", "H4", "STIGMA",
            ] {
                if model.parameter_value_local(name).is_some() {
                    names.push(name.to_string());
                }
            }
        }
        // WHY: OMDOT (GR periastron precession) and PBDOT (GR orbital period decay) are
        // post-Keplerian parameters that the full 15-yr TEMPO2 solution fits. Omitting them
        // would leave their secular drift absorbed into spin-down or DM, biasing F1 and DM.
        Some(BinaryFamily::Bt) => {
            for name in ["A1", "PB", "ECC", "T0", "OM", "OMDOT", "PBDOT", "GAMMA"] {
                if model.parameter_value_local(name).is_some() {
                    names.push(name.to_string());
                }
            }
        }
        // WHY: M2 and SINI (Shapiro delay mass parameters) are fitted in the full DD solution.
        // They have a direct analytic effect on the Shapiro delay term; omitting them means
        // Shapiro delay residuals are absorbed into orbital elements, biasing T0 and OM.
        Some(BinaryFamily::Dd) => {
            for name in [
                "A1", "PB", "ECC", "T0", "OM", "OMDOT", "PBDOT", "GAMMA", "M2", "SINI",
            ] {
                if model.parameter_value_local(name).is_some() {
                    names.push(name.to_string());
                }
            }
        }
        Some(BinaryFamily::Ddk) => {
            for name in [
                "A1", "PB", "ECC", "T0", "OM", "OMDOT", "PBDOT", "GAMMA", "KIN", "KOM", "M2",
                "SINI",
            ] {
                if model.parameter_value_local(name).is_some() {
                    names.push(name.to_string());
                }
            }
        }
        _ => {}
    }
    names
}

fn independent_residual_seconds_cached(
    model: &TimingModel,
    toa: &TopocentricToa,
    epoch_tdb: Epoch,
    static_delay: &StaticDelayContext,
    overrides: &BTreeMap<String, f64>,
) -> Result<f64> {
    let astrometry = astrometric_state(model, overrides, epoch_tdb)?;
    let context = DelayContext {
        earth_barycentric_km: static_delay.earth_barycentric_km,
        site_barycentric_km: static_delay.site_barycentric_km,
        sun_from_earth_km: static_delay.sun_from_earth_km,
        sky_unit: astrometry.sky_unit_equatorial,
        sky_unit_ecliptic: astrometry.sky_unit_ecliptic,
    };
    let roemer_s = dot3(context.site_barycentric_km, context.sky_unit) / C_KM_PER_S;
    let parallax_s = astrometry
        .parallax_distance_km
        .map(|distance_km| {
            parallax_delay_seconds(context.site_barycentric_km, context.sky_unit, distance_km)
        })
        .unwrap_or(0.0);
    let shapiro_s = solar_system_shapiro_seconds(context.sun_from_earth_km, context.sky_unit);
    let binary_s = binary_delay_seconds(model, toa.mjd_utc, &context, overrides)?;
    let dm_s = DM_DELAY_S_PER_MHZ2_DM * dispersion_measure(model, overrides, toa.mjd_utc)
        / (toa.frequency_mhz * toa.frequency_mhz);
    let barycentric_tdb_s =
        epoch_tdb.to_tdb_seconds() + roemer_s + parallax_s + shapiro_s + binary_s - dm_s;
    let phase = spin_phase(model, barycentric_tdb_s, overrides)?;
    let wrapped_cycles = wrap_cycles(phase);
    // Subtract FD profile-delay: toa_corrected = toa - fd_delay, so residual = raw - fd_delay.
    // WHY: the baseline residual must be comparable to design-matrix FD derivatives d(res)/d(FD_k)
    // = -(ln f/f_ref)^k, which assume the FD correction is already applied. Removing fd_delay here
    // means the response vector reflects the published FD model, not the raw dispersive residual.
    let fd_s = fd_delay_seconds(model, overrides, toa.frequency_mhz);
    Ok(wrapped_cycles / spin_frequency_hz(model, overrides)? - fd_s)
}

fn spin_phase(
    model: &TimingModel,
    barycentric_tdb_s: f64,
    overrides: &BTreeMap<String, f64>,
) -> Result<f64> {
    let pepoch_mjd = parameter_current_value(model, "PEPOCH")
        .or_else(|_| parameter_current_value(model, "POSEPOCH"))?;
    let pepoch_tdb_s = Epoch::from_mjd_in_time_scale(pepoch_mjd, TimeScale::TDB).to_tdb_seconds();
    let dt = barycentric_tdb_s - pepoch_tdb_s;
    let f0 = spin_frequency_hz(model, overrides)?;
    let f1 = overrides
        .get("F1")
        .copied()
        .or_else(|| model.parameter_value_local("F1"))
        .unwrap_or(0.0);
    let f2 = overrides
        .get("F2")
        .copied()
        .or_else(|| model.parameter_value_local("F2"))
        .unwrap_or(0.0);
    Ok(f0 * dt + 0.5 * f1 * dt * dt + (1.0 / 6.0) * f2 * dt * dt * dt)
}

fn spin_frequency_hz(model: &TimingModel, overrides: &BTreeMap<String, f64>) -> Result<f64> {
    overrides
        .get("F0")
        .copied()
        .or_else(|| model.parameter_value_local("F0"))
        .ok_or_else(|| anyhow!("{} missing F0", model.solution_id))
}

fn spin_period_seconds(model: &TimingModel) -> f64 {
    1.0 / model.parameter_value_local("F0").unwrap_or(1.0)
}

fn astrometric_state(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    epoch_tdb: Epoch,
) -> Result<AstrometricState> {
    let Some(base_elong_deg) = overrides
        .get("ELONG")
        .copied()
        .or_else(|| model.parameter_value_local("ELONG"))
    else {
        bail!(
            "{} missing numeric ELONG/ELAT astrometry",
            model.solution_id
        );
    };
    let base_elat_deg = overrides
        .get("ELAT")
        .copied()
        .or_else(|| model.parameter_value_local("ELAT"))
        .ok_or_else(|| anyhow!("{} missing numeric ELAT", model.solution_id))?;
    let posepoch_mjd = overrides
        .get("POSEPOCH")
        .copied()
        .or_else(|| model.parameter_value_local("POSEPOCH"))
        .or_else(|| model.parameter_value_local("PEPOCH"))
        .unwrap_or(epoch_tdb.to_jde_tdb_days() - 2_400_000.5);
    let posepoch_tdb_s =
        Epoch::from_mjd_in_time_scale(posepoch_mjd, TimeScale::TDB).to_tdb_seconds();
    let dt_years = (epoch_tdb.to_tdb_seconds() - posepoch_tdb_s) / JULIAN_YEAR_S;
    let base_elong_rad = base_elong_deg.to_radians();
    let base_elat_rad = base_elat_deg.to_radians();
    let pmelong_rad_yr = overrides
        .get("PMELONG")
        .copied()
        .or_else(|| model.parameter_value_local("PMELONG"))
        .unwrap_or(0.0)
        * MAS_TO_RAD;
    let pmelat_rad_yr = overrides
        .get("PMELAT")
        .copied()
        .or_else(|| model.parameter_value_local("PMELAT"))
        .unwrap_or(0.0)
        * MAS_TO_RAD;
    let cos_lat = base_elat_rad.cos().abs().max(1.0e-12);
    let elong_rad = base_elong_rad + dt_years * pmelong_rad_yr / cos_lat;
    let elat_rad =
        (base_elat_rad + dt_years * pmelat_rad_yr).clamp(-0.5 * PI + 1.0e-12, 0.5 * PI - 1.0e-12);
    let sky_unit_ecliptic = sky_unit_from_ecliptic(elong_rad, elat_rad);
    let sky_unit_equatorial = ecliptic_to_equatorial(sky_unit_ecliptic);
    let parallax_distance_km = parallax_distance_km(model, overrides);
    Ok(AstrometricState {
        sky_unit_ecliptic,
        sky_unit_equatorial,
        parallax_distance_km,
    })
}

fn dispersion_measure(model: &TimingModel, overrides: &BTreeMap<String, f64>, mjd_utc: f64) -> f64 {
    let dm = overrides
        .get("DM")
        .copied()
        .or_else(|| model.dispersion.dm.as_ref().and_then(|term| term.value))
        .unwrap_or(0.0);
    let dmx = model
        .dispersion
        .dmx_windows
        .iter()
        .filter(|window| {
            let Some(start) = window.start_mjd else {
                return false;
            };
            let Some(end) = window.end_mjd else {
                return false;
            };
            start <= mjd_utc && mjd_utc <= end
        })
        .filter_map(|window| {
            let term = window.dmx.as_ref()?;
            Some(
                overrides
                    .get(&term.name)
                    .copied()
                    .unwrap_or(term.value.unwrap_or(0.0)),
            )
        })
        .sum::<f64>();
    dm + dmx
}

fn dmx_window_matches(model: &TimingModel, parameter_name: &str, mjd_utc: f64) -> bool {
    model.dispersion.dmx_windows.iter().any(|window| {
        let Some(term) = window.dmx.as_ref() else {
            return false;
        };
        let Some(start) = window.start_mjd else {
            return false;
        };
        let Some(end) = window.end_mjd else {
            return false;
        };
        term.name == parameter_name && start <= mjd_utc && mjd_utc <= end
    })
}

fn binary_delay_seconds(
    model: &TimingModel,
    mjd_utc: f64,
    context: &DelayContext,
    overrides: &BTreeMap<String, f64>,
) -> Result<f64> {
    match model.binary_family.as_ref() {
        Some(BinaryFamily::Ell1) => ell1_delay_seconds(model, mjd_utc, overrides, false),
        Some(BinaryFamily::Ell1h) => ell1_delay_seconds(model, mjd_utc, overrides, true),
        Some(BinaryFamily::Bt) => bt_delay_seconds(model, mjd_utc, overrides),
        Some(BinaryFamily::Dd) => dd_delay_seconds(model, mjd_utc, context, overrides),
        Some(BinaryFamily::Ddk) => ddk_delay_seconds(model, mjd_utc, context, overrides),
        _ => Ok(0.0),
    }
}

fn ell1_delay_seconds(
    model: &TimingModel,
    mjd_utc: f64,
    overrides: &BTreeMap<String, f64>,
    include_h3: bool,
) -> Result<f64> {
    let tasc = parameter_value_or(model, overrides, "TASC")?;
    let ttasc_s = (mjd_utc - tasc) * 86_400.0;
    let a1 = parameter_value_or_default(model, overrides, "A1", 0.0)
        + ttasc_s * parameter_value_or_default(model, overrides, "A1DOT", 0.0);
    let eps1 = parameter_value_or_default(model, overrides, "EPS1", 0.0)
        + ttasc_s * parameter_value_or_default(model, overrides, "EPS1DOT", 0.0);
    let eps2 = parameter_value_or_default(model, overrides, "EPS2", 0.0)
        + ttasc_s * parameter_value_or_default(model, overrides, "EPS2DOT", 0.0);
    let orbits = ell1_orbits(model, overrides, ttasc_s)?;
    let phi = 2.0 * PI * fract(orbits);

    let dre_over_a1 = phi.sin() + 0.5 * (eps2 * (2.0 * phi).sin() - eps1 * (2.0 * phi).cos())
        - 0.125
            * (5.0 * eps2 * eps2 * phi.sin()
                - 3.0 * eps2 * eps2 * (3.0 * phi).sin()
                - 2.0 * eps2 * eps1 * phi.cos()
                + 6.0 * eps2 * eps1 * (3.0 * phi).cos()
                + 3.0 * eps1 * eps1 * phi.sin()
                + 3.0 * eps1 * eps1 * (3.0 * phi).sin());
    let drep_over_a1 = phi.cos() + eps1 * (2.0 * phi).sin() + eps2 * (2.0 * phi).cos()
        - 0.125
            * (5.0 * eps2 * eps2 * phi.cos() - 9.0 * eps2 * eps2 * (3.0 * phi).cos()
                + 2.0 * eps1 * eps2 * phi.sin()
                - 18.0 * eps1 * eps2 * (3.0 * phi).sin()
                + 3.0 * eps1 * eps1 * phi.cos()
                + 9.0 * eps1 * eps1 * (3.0 * phi).cos());
    let drepp_over_a1 = -phi.sin() + 2.0 * eps1 * (2.0 * phi).cos()
        - 2.0 * eps2 * (2.0 * phi).sin()
        - 0.125
            * (-5.0 * eps2 * eps2 * phi.sin()
                + 27.0 * eps2 * eps2 * (3.0 * phi).sin()
                + 2.0 * eps1 * eps2 * phi.cos()
                - 54.0 * eps1 * eps2 * (3.0 * phi).cos()
                - 3.0 * eps1 * eps1 * phi.sin()
                - 27.0 * eps1 * eps1 * (3.0 * phi).sin());

    let delay_r = a1 * dre_over_a1;
    let pb_s = ell1_binary_period_seconds(model, overrides, ttasc_s)?;
    let nhat = 2.0 * PI / pb_s;
    let drep = a1 * drep_over_a1;
    let drepp = a1 * drepp_over_a1;
    let delay_i =
        delay_r * (1.0 - nhat * drep + (nhat * drep).powi(2) + 0.5 * nhat * nhat * delay_r * drepp);
    let shapiro = if include_h3 {
        ell1h_shapiro_seconds(model, overrides, phi)
    } else {
        0.0
    };
    Ok(delay_i + shapiro)
}

fn ell1h_shapiro_seconds(model: &TimingModel, overrides: &BTreeMap<String, f64>, phi: f64) -> f64 {
    let h3 = parameter_value_or_default(model, overrides, "H3", 0.0);
    if h3 == 0.0 {
        return 0.0;
    }
    let nharms = model
        .parameter_value_local("NHARMS")
        .map(|value| value.round().max(3.0) as usize)
        .unwrap_or(3);
    let explicit_stigma = overrides
        .get("STIGMA")
        .copied()
        .or_else(|| model.parameter_value_local("STIGMA"));
    if let Some(stigma) = explicit_stigma.filter(|value| value.is_finite() && value.abs() > 1.0e-8)
    {
        return ell1h_shapiro_exact_seconds(h3, stigma, phi);
    }
    let h4 = overrides
        .get("H4")
        .copied()
        .or_else(|| model.parameter_value_local("H4"))
        .unwrap_or(0.0);
    if h4.abs() > 0.0 && h3.abs() > 1.0e-18 {
        let stigma = h4 / h3;
        return ell1h_shapiro_approximate_seconds(h3, stigma, phi, nharms.max(7));
    }
    ell1h_shapiro_approximate_seconds(h3, 0.0, phi, nharms)
}

fn ell1h_shapiro_approximate_seconds(h3: f64, stigma: f64, phi: f64, end_harm: usize) -> f64 {
    let mut sum = 0.0;
    for harmonic in 3..=end_harm.max(3) {
        sum += ell1h_fourier_component(stigma, harmonic, phi, 3);
    }
    -2.0 * h3 * sum
}

fn ell1h_fourier_component(stigma: f64, harmonic: usize, phi: f64, factor_out_power: usize) -> f64 {
    if harmonic == 0 {
        return 0.0;
    }
    let parity_power = if harmonic.is_multiple_of(2) {
        (harmonic + 2) / 2
    } else {
        harmonic.div_ceil(2)
    };
    let sign = if parity_power % 2 == 0 { 1.0 } else { -1.0 };
    let power = harmonic.saturating_sub(factor_out_power) as i32;
    let coefficient = sign * 2.0 / harmonic as f64 * stigma.powi(power);
    let angle = harmonic as f64 * phi;
    let basis = if harmonic.is_multiple_of(2) {
        angle.cos()
    } else {
        angle.sin()
    };
    coefficient * basis
}

fn ell1h_shapiro_exact_seconds(h3: f64, stigma: f64, phi: f64) -> f64 {
    let lognum = 1.0 + stigma * stigma - 2.0 * stigma * phi.sin();
    if lognum <= 1.0e-15 {
        return ell1h_shapiro_approximate_seconds(h3, stigma, phi, 7);
    }
    -2.0 / stigma.powi(3) * h3 * lognum.ln()
}

fn ell1_orbits(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    ttasc_s: f64,
) -> Result<f64> {
    if model.parameter_value_local("FB0").is_some() {
        let mut value = 0.0;
        let mut power = 1.0;
        let mut factorial = 1.0;
        let mut index = 0usize;
        while let Some(coefficient) = overrides
            .get(&format!("FB{index}"))
            .copied()
            .or_else(|| model.parameter_value_local(&format!("FB{index}")))
        {
            value += coefficient * power / factorial;
            index += 1;
            power *= ttasc_s;
            factorial *= index as f64;
        }
        Ok(value)
    } else {
        let pb_s = parameter_value_or(model, overrides, "PB")? * 86_400.0;
        let x = ttasc_s / pb_s;
        let pbdot = parameter_value_or_default(model, overrides, "PBDOT", 0.0);
        Ok(x - 0.5 * pbdot * x * x)
    }
}

fn ell1_binary_period_seconds(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    ttasc_s: f64,
) -> Result<f64> {
    if model.parameter_value_local("FB0").is_some() {
        let mut derivative = 0.0;
        let mut power = 1.0;
        let mut factorial = 1.0;
        let mut index = 0usize;
        while let Some(coefficient) = overrides
            .get(&format!("FB{index}"))
            .copied()
            .or_else(|| model.parameter_value_local(&format!("FB{index}")))
        {
            if index == 0 {
                derivative += coefficient;
            } else {
                power *= ttasc_s;
                factorial *= index as f64;
                derivative += coefficient * power / factorial;
            }
            index += 1;
        }
        if derivative == 0.0 {
            bail!("{} has zero FB derivative", model.solution_id);
        }
        Ok(1.0 / derivative)
    } else {
        Ok(parameter_value_or(model, overrides, "PB")? * 86_400.0)
    }
}

fn bt_delay_seconds(
    model: &TimingModel,
    mjd_utc: f64,
    overrides: &BTreeMap<String, f64>,
) -> Result<f64> {
    let state = orbital_state(model, mjd_utc, overrides)?;
    let omega_rad = bt_omega_rad(model, overrides, state.dt_s);
    let sin_e = state.eccentric_anomaly.sin();
    let cos_e = state.eccentric_anomaly.cos();
    let root = (1.0 - state.ecc * state.ecc).max(0.0).sqrt();
    let alpha = state.a1_lt_s * omega_rad.sin();
    let beta = state.a1_lt_s * omega_rad.cos() * root + state.gamma_s;
    let delay_l1 = alpha * (cos_e - state.ecc);
    let delay_l2 = beta * sin_e;
    let num =
        state.a1_lt_s * omega_rad.cos() * root * cos_e - state.a1_lt_s * omega_rad.sin() * sin_e;
    let den = (1.0 - state.ecc * cos_e).max(1.0e-12);
    let delay_r = 1.0 - 2.0 * PI * num / (den * state.pb_s);
    Ok((delay_l1 + delay_l2) * delay_r)
}

fn dd_delay_seconds(
    model: &TimingModel,
    mjd_utc: f64,
    _context: &DelayContext,
    overrides: &BTreeMap<String, f64>,
) -> Result<f64> {
    dd_family_delay_seconds(model, mjd_utc, overrides, DdkCorrections::default())
}

fn ddk_delay_seconds(
    model: &TimingModel,
    mjd_utc: f64,
    context: &DelayContext,
    overrides: &BTreeMap<String, f64>,
) -> Result<f64> {
    let corrections = ddk_corrections(model, mjd_utc, context, overrides)?;
    dd_family_delay_seconds(model, mjd_utc, overrides, corrections)
}

fn dd_family_delay_seconds(
    model: &TimingModel,
    mjd_utc: f64,
    overrides: &BTreeMap<String, f64>,
    ddk: DdkCorrections,
) -> Result<f64> {
    let state = orbital_state(model, mjd_utc, overrides)?;
    let omega_rad = dd_omega_rad(model, overrides, &state) + ddk.omega_offset_rad;
    let a1_lt_s = state.a1_lt_s + ddk.a1_offset_lt_s;
    let er = state.ecc * (1.0 + parameter_value_or_default(model, overrides, "DR", 0.0));
    let e_theta = state.ecc * (1.0 + parameter_value_or_default(model, overrides, "DTH", 0.0));
    let sin_e = state.eccentric_anomaly.sin();
    let cos_e = state.eccentric_anomaly.cos();
    let beta = a1_lt_s * (1.0 - e_theta * e_theta).max(0.0).sqrt() * omega_rad.cos();
    let alpha = a1_lt_s * omega_rad.sin();
    let dre = alpha * (cos_e - er) + (beta + state.gamma_s) * sin_e;
    let drep = -alpha * sin_e + (beta + state.gamma_s) * cos_e;
    let drepp = -alpha * cos_e - (beta + state.gamma_s) * sin_e;
    let nhat = state.mean_motion_rad_s / (1.0 - state.ecc * cos_e).max(1.0e-12);
    let inverse = dre
        * (1.0 - nhat * drep + (nhat * drep).powi(2) + 0.5 * nhat * nhat * dre * drepp
            - 0.5 * state.ecc * sin_e / (1.0 - state.ecc * cos_e).max(1.0e-12)
                * nhat
                * nhat
                * dre
                * drep);

    let sini = ddk
        .sini_override
        .or_else(|| overrides.get("SINI").copied())
        .or_else(|| model.parameter_value_local("SINI"))
        .or_else(|| {
            overrides
                .get("KIN")
                .copied()
                .or_else(|| model.parameter_value_local("KIN"))
                .map(|value| value.to_radians().sin())
        })
        .unwrap_or(0.0)
        .clamp(0.0, 1.0);
    let shapiro = dd_shapiro_delay_seconds(
        state.ecc,
        state.eccentric_anomaly,
        omega_rad,
        sini,
        model,
        overrides,
    );
    let aberration =
        dd_aberration_delay_seconds(model, overrides, omega_rad, state.true_anomaly, state.ecc);
    Ok(inverse + shapiro + aberration)
}

fn dd_shapiro_delay_seconds(
    ecc: f64,
    eccentric_anomaly: f64,
    omega_rad: f64,
    sini: f64,
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
) -> f64 {
    let m2 = parameter_value_or_default(model, overrides, "M2", 0.0);
    if m2 <= 0.0 || sini <= 0.0 {
        return 0.0;
    }
    let cos_e = eccentric_anomaly.cos();
    let sin_e = eccentric_anomaly.sin();
    let root = (1.0 - ecc * ecc).max(0.0).sqrt();
    let argument = 1.0
        - ecc * cos_e
        - sini * (omega_rad.sin() * (cos_e - ecc) + root * omega_rad.cos() * sin_e);
    -2.0 * SOLAR_MASS_TIME_S * m2 * argument.max(1.0e-12).ln()
}

fn dd_aberration_delay_seconds(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    omega_rad: f64,
    true_anomaly: f64,
    ecc: f64,
) -> f64 {
    let a0 = parameter_value_or_default(model, overrides, "A0", 0.0);
    let b0 = parameter_value_or_default(model, overrides, "B0", 0.0);
    if a0 == 0.0 && b0 == 0.0 {
        return 0.0;
    }
    let angle = omega_rad + true_anomaly;
    a0 * (angle.sin() + ecc * omega_rad.sin()) + b0 * (angle.cos() + ecc * omega_rad.cos())
}

fn orbital_state(
    model: &TimingModel,
    mjd_utc: f64,
    overrides: &BTreeMap<String, f64>,
) -> Result<OrbitalState> {
    let pb_s = parameter_value_or(model, overrides, "PB")? * 86_400.0;
    let t0 = parameter_value_or(model, overrides, "T0")?;
    let dt_s = (mjd_utc - t0) * 86_400.0;
    let pbdot = parameter_value_or_default(model, overrides, "PBDOT", 0.0);
    let x = dt_s / pb_s;
    let mean_anomaly = 2.0 * PI * (x - 0.5 * pbdot * x * x);
    let ecc = parameter_value_or_default(model, overrides, "ECC", 0.0).abs();
    let eccentric_anomaly = solve_kepler(mean_anomaly, ecc);
    let true_anomaly = true_anomaly_from_eccentric(eccentric_anomaly, ecc);
    let a1_lt_s = parameter_value_or(model, overrides, "A1")?
        + dt_s * parameter_value_or_default(model, overrides, "A1DOT", 0.0);
    Ok(OrbitalState {
        pb_s,
        dt_s,
        a1_lt_s,
        ecc,
        eccentric_anomaly,
        true_anomaly,
        gamma_s: parameter_value_or_default(model, overrides, "GAMMA", 0.0),
        mean_motion_rad_s: 2.0 * PI / pb_s,
    })
}

fn bt_omega_rad(model: &TimingModel, overrides: &BTreeMap<String, f64>, dt_s: f64) -> f64 {
    let om0_rad = parameter_value_or_default(model, overrides, "OM", 0.0).to_radians();
    let omdot_rad_s =
        parameter_value_or_default(model, overrides, "OMDOT", 0.0).to_radians() / JULIAN_YEAR_S;
    om0_rad + omdot_rad_s * dt_s
}

fn dd_omega_rad(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    state: &OrbitalState,
) -> f64 {
    let om0_rad = parameter_value_or_default(model, overrides, "OM", 0.0).to_radians();
    let omdot_rad_s =
        parameter_value_or_default(model, overrides, "OMDOT", 0.0).to_radians() / JULIAN_YEAR_S;
    let k = omdot_rad_s / state.mean_motion_rad_s;
    om0_rad + state.true_anomaly * k
}

fn ddk_corrections(
    model: &TimingModel,
    mjd_utc: f64,
    context: &DelayContext,
    overrides: &BTreeMap<String, f64>,
) -> Result<DdkCorrections> {
    let k96_enabled = model.parameter_bool_local("K96").unwrap_or(true);
    let t0 = parameter_value_or(model, overrides, "T0")?;
    let dt_s = (mjd_utc - t0) * 86_400.0;
    let kom_rad = parameter_value_or(model, overrides, "KOM")?.to_radians();
    let kin0_rad = parameter_value_or(model, overrides, "KIN")?.to_radians();
    let pm_long_rad_s = proper_motion_component_rad_s(model, overrides, true);
    let pm_lat_rad_s = proper_motion_component_rad_s(model, overrides, false);
    let dkin = if k96_enabled {
        (-pm_long_rad_s * kom_rad.sin() + pm_lat_rad_s * kom_rad.cos()) * dt_s
    } else {
        0.0
    };
    let kin_rad = kin0_rad + dkin;
    let sin_kin = kin_rad.sin();
    let tan_kin = kin_rad.tan();
    let base_a1_lt_s = parameter_value_or(model, overrides, "A1")?
        + dt_s * parameter_value_or_default(model, overrides, "A1DOT", 0.0);
    let a1_pm = if k96_enabled && tan_kin.abs() > 1.0e-12 {
        base_a1_lt_s * dkin / tan_kin
    } else {
        0.0
    };

    let earth_ecliptic_au = {
        let earth_ecliptic_km = equatorial_to_ecliptic(context.earth_barycentric_km);
        [
            earth_ecliptic_km[0] / AU_KM,
            earth_ecliptic_km[1] / AU_KM,
            earth_ecliptic_km[2] / AU_KM,
        ]
    };
    let (sin_long, cos_long, sin_lat, cos_lat) = sky_long_lat(context.sky_unit_ecliptic);
    let delta_i0 = -earth_ecliptic_au[0] * sin_long + earth_ecliptic_au[1] * cos_long;
    let delta_j0 = -earth_ecliptic_au[0] * sin_lat * cos_long
        - earth_ecliptic_au[1] * sin_lat * sin_long
        + earth_ecliptic_au[2] * cos_lat;
    let parallax_distance_au = parallax_distance_au(model, overrides);
    let a1_parallax = if tan_kin.abs() > 1.0e-12 && parallax_distance_au.is_finite() {
        base_a1_lt_s / tan_kin / parallax_distance_au
            * (delta_i0 * kom_rad.sin() - delta_j0 * kom_rad.cos())
    } else {
        0.0
    };
    let omega_pm = if k96_enabled && sin_kin.abs() > 1.0e-12 {
        dt_s / sin_kin * (pm_long_rad_s * kom_rad.cos() + pm_lat_rad_s * kom_rad.sin())
    } else {
        0.0
    };
    let omega_parallax = if sin_kin.abs() > 1.0e-12 && parallax_distance_au.is_finite() {
        -(delta_i0 * kom_rad.cos() + delta_j0 * kom_rad.sin()) / (sin_kin * parallax_distance_au)
    } else {
        0.0
    };

    Ok(DdkCorrections {
        a1_offset_lt_s: a1_pm + a1_parallax,
        omega_offset_rad: omega_pm + omega_parallax,
        sini_override: Some(sin_kin.clamp(0.0, 1.0)),
    })
}

fn proper_motion_component_rad_s(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    longitude: bool,
) -> f64 {
    let names = if longitude {
        ["PMELONG", "PMRA"]
    } else {
        ["PMELAT", "PMDEC"]
    };
    for name in names {
        if let Some(value) = overrides
            .get(name)
            .copied()
            .or_else(|| model.parameter_value_local(name))
        {
            return value * MAS_TO_RAD / JULIAN_YEAR_S;
        }
    }
    0.0
}

fn parallax_distance_au(model: &TimingModel, overrides: &BTreeMap<String, f64>) -> f64 {
    let Some(px_mas) = overrides
        .get("PX")
        .copied()
        .or_else(|| model.parameter_value_local("PX"))
    else {
        return f64::INFINITY;
    };
    if px_mas <= 0.0 {
        f64::INFINITY
    } else {
        ARCSEC_PER_RAD * 1_000.0 / px_mas
    }
}

fn parallax_distance_km(model: &TimingModel, overrides: &BTreeMap<String, f64>) -> Option<f64> {
    let distance_au = parallax_distance_au(model, overrides);
    if distance_au.is_finite() {
        Some(distance_au * AU_KM)
    } else {
        None
    }
}

fn parallax_delay_seconds(
    observer_barycentric_km: [f64; 3],
    sky_unit_equatorial: [f64; 3],
    distance_km: f64,
) -> f64 {
    let r2 = dot3(observer_barycentric_km, observer_barycentric_km);
    if r2 <= 0.0 || !distance_km.is_finite() || distance_km <= 0.0 {
        return 0.0;
    }
    let re_dot_l = dot3(observer_barycentric_km, sky_unit_equatorial);
    0.5 * (r2 - re_dot_l * re_dot_l) / (distance_km * C_KM_PER_S)
}

fn sky_unit_from_ecliptic(elong_rad: f64, elat_rad: f64) -> [f64; 3] {
    [
        elat_rad.cos() * elong_rad.cos(),
        elat_rad.cos() * elong_rad.sin(),
        elat_rad.sin(),
    ]
}

fn ecliptic_to_equatorial(ecliptic: [f64; 3]) -> [f64; 3] {
    [
        ecliptic[0],
        ecliptic[1] * OBLIQUITY_RAD.cos() - ecliptic[2] * OBLIQUITY_RAD.sin(),
        ecliptic[1] * OBLIQUITY_RAD.sin() + ecliptic[2] * OBLIQUITY_RAD.cos(),
    ]
}

fn equatorial_to_ecliptic(equatorial: [f64; 3]) -> [f64; 3] {
    [
        equatorial[0],
        equatorial[1] * OBLIQUITY_RAD.cos() + equatorial[2] * OBLIQUITY_RAD.sin(),
        -equatorial[1] * OBLIQUITY_RAD.sin() + equatorial[2] * OBLIQUITY_RAD.cos(),
    ]
}

fn sky_long_lat(sky_unit: [f64; 3]) -> (f64, f64, f64, f64) {
    let sin_lat = sky_unit[2].clamp(-1.0, 1.0);
    let cos_lat = (1.0 - sin_lat * sin_lat).max(0.0).sqrt();
    if cos_lat <= 1.0e-15 {
        (0.0, 1.0, sin_lat, cos_lat)
    } else {
        (
            sky_unit[1] / cos_lat,
            sky_unit[0] / cos_lat,
            sin_lat,
            cos_lat,
        )
    }
}

fn parse_tempo2_toas(path: &Path) -> Result<Vec<TopocentricToa>> {
    let content = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let mut format = None::<usize>;
    let mut toas = Vec::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        if let Some(rest) = trimmed.strip_prefix("FORMAT ") {
            format = Some(
                rest.trim()
                    .parse::<usize>()
                    .with_context(|| format!("parse FORMAT in {}", path.display()))?,
            );
            continue;
        }
        let data_line = trimmed.strip_prefix("C ").unwrap_or(trimmed);
        let fields = data_line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 5 {
            continue;
        }
        match format {
            Some(1) => {
                let mut flags = BTreeMap::new();
                let mut pp_dm = None;
                let mut pp_dme = None;
                let mut index = 5usize;
                while index + 1 < fields.len() {
                    if !fields[index].starts_with('-') {
                        index += 1;
                        continue;
                    }
                    let key = fields[index].to_string();
                    let value = fields[index + 1].to_string();
                    if key == "-pp_dm" {
                        pp_dm = value.parse::<f64>().ok();
                    }
                    if key == "-pp_dme" {
                        pp_dme = value.parse::<f64>().ok();
                    }
                    flags.insert(key, value);
                    index += 2;
                }
                toas.push(TopocentricToa {
                    name: fields[0].to_string(),
                    frequency_mhz: fields[1]
                        .parse::<f64>()
                        .with_context(|| format!("parse frequency in {}", path.display()))?,
                    mjd_utc: fields[2]
                        .parse::<f64>()
                        .with_context(|| format!("parse MJD in {}", path.display()))?,
                    uncertainty_us: fields[3]
                        .parse::<f64>()
                        .with_context(|| format!("parse uncertainty in {}", path.display()))?,
                    site: SiteId::from_token(fields[4])?,
                    flags,
                    pp_dm,
                    pp_dme,
                });
            }
            Some(other) => bail!(
                "unsupported TEMPO2 tim FORMAT {other} in {}",
                path.display()
            ),
            None => bail!("missing FORMAT header in {}", path.display()),
        }
    }
    Ok(toas)
}

fn dominant_subgroup(toas: &[TopocentricToa]) -> Result<(String, usize)> {
    let mut counts = BTreeMap::<String, usize>::new();
    for toa in toas {
        let subgroup = toa
            .flags
            .get("-f")
            .ok_or_else(|| anyhow!("TOA {} missing -f subgroup", toa.name))?;
        *counts.entry(subgroup.clone()).or_default() += 1;
    }
    counts
        .into_iter()
        .max_by(|left, right| left.1.cmp(&right.1).then_with(|| right.0.cmp(&left.0)))
        .ok_or_else(|| anyhow!("no subgroup counts found"))
}

fn parameter_current_value(model: &TimingModel, name: &str) -> Result<f64> {
    model
        .parameter_value_local(name)
        .ok_or_else(|| anyhow!("{} missing parameter {name}", model.solution_id))
}

fn parse_selector_parameter_index(name: &str, prefix: &str) -> Option<usize> {
    name.strip_prefix(prefix)?.parse::<usize>().ok()
}

/// Returns the FD order k for a parameter named "FD{k}" (e.g. "FD1" -> 1, "FD2" -> 2).
fn parse_fd_order(name: &str) -> Option<u32> {
    name.strip_prefix("FD")?.parse::<u32>().ok()
}

/// Computes the total FD profile-delay correction in seconds for the given frequency.
/// TEMPO2 convention: delay_FD = sum_k FD_k * ln(f/f_ref)^k with f_ref = 1 GHz.
fn fd_delay_seconds(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    frequency_mhz: f64,
) -> f64 {
    const F_REF_MHZ: f64 = 1000.0;
    let log_f = (frequency_mhz / F_REF_MHZ).ln();
    model
        .fd_terms
        .iter()
        .filter_map(|term| {
            let k = parse_fd_order(&term.name)?;
            let value = overrides
                .get(&term.name)
                .copied()
                .or(term.value)
                .unwrap_or(0.0);
            Some(value * log_f.powi(k as i32))
        })
        .sum()
}

fn parameter_value_or(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    name: &str,
) -> Result<f64> {
    overrides
        .get(name)
        .copied()
        .or_else(|| model.parameter_value_local(name))
        .ok_or_else(|| anyhow!("{} missing parameter {name}", model.solution_id))
}

fn parameter_value_or_default(
    model: &TimingModel,
    overrides: &BTreeMap<String, f64>,
    name: &str,
    default: f64,
) -> f64 {
    overrides
        .get(name)
        .copied()
        .or_else(|| model.parameter_value_local(name))
        .unwrap_or(default)
}

fn parameter_step(model: &TimingModel, name: &str) -> Result<f64> {
    let term = model
        .parameter_term_local(name)
        .ok_or_else(|| anyhow!("{} missing parameter term {name}", model.solution_id))?;
    if let Some(uncertainty) = term.uncertainty.filter(|value| *value > 0.0) {
        return Ok(0.01 * uncertainty);
    }
    let fallback = match name {
        "F0" => 1.0e-12,
        "F1" => 1.0e-20,
        "ELONG" | "ELAT" | "PMELONG" | "PMELAT" => 1.0e-8,
        "PX" => 1.0e-5,
        "DM" => 1.0e-5,
        value if value.starts_with("DMX_") => 1.0e-5,
        "A1" => 1.0e-6,
        "TASC" | "T0" => 1.0e-7,
        "EPS1" | "EPS2" => 1.0e-8,
        "PB" => 1.0e-8,
        "PBDOT" => 1.0e-12,
        name if name.starts_with("FB") => 1.0e-16,
        "ECC" => 1.0e-8,
        "OM" | "OMDOT" | "KIN" | "KOM" => 1.0e-6,
        "GAMMA" | "A0" | "B0" => 1.0e-8,
        "M2" | "SINI" | "DR" | "DTH" | "H3" | "H4" | "STIGMA" => 1.0e-6,
        _ => 1.0e-8,
    };
    Ok(fallback)
}

fn parameter_prior_sigma(model: &TimingModel, name: &str) -> Option<f64> {
    if name == "PHASE_OFFSET" {
        return None;
    }
    // JUMP and DMJUMP are inter-backend calibration offsets fully determined by the data.
    // WHY: adding a prior would bias them toward zero, suppressing legitimate backend-to-backend
    // offsets. They are not physical parameters so we let the data determine them freely.
    if let Some(index) = parse_selector_parameter_index(name, "JUMP@") {
        return model
            .jumps
            .get(index)
            .and_then(|term| term.uncertainty)
            .filter(|value| *value > 0.0)
            .map(|uncertainty| 3.0 * uncertainty);
    }
    if let Some(index) = parse_selector_parameter_index(name, "DMJUMP@") {
        return model
            .dmjumps
            .get(index)
            .and_then(|term| term.uncertainty)
            .filter(|value| *value > 0.0)
            .map(|uncertainty| 3.0 * uncertainty);
    }
    let prior_scale = match name {
        "DM" => 1.0,
        value if value.starts_with("DMX_") => 1.0,
        value if value.starts_with("FD") => 3.0,
        "ELONG" | "ELAT" | "PMELONG" | "PMELAT" | "PX" => 3.0,
        "A1" | "A1DOT" | "PB" | "PBDOT" | "T0" | "TASC" | "ECC" | "EPS1" | "EPS2" | "OM"
        | "OMDOT" | "KIN" | "KOM" | "M2" | "SINI" | "GAMMA" | "DR" | "DTH" | "H3" | "H4"
        | "STIGMA" => 3.0,
        _ => 5.0,
    };
    if let Some(uncertainty) = model
        .parameter_term_local(name)
        .and_then(|term| term.uncertainty)
        .filter(|value| *value > 0.0)
    {
        return Some(prior_scale * uncertainty);
    }
    parameter_step(model, name)
        .ok()
        .map(|step| prior_scale.max(1.0) * 100.0 * step)
}

fn solar_system_shapiro_seconds(sun_from_earth_km: [f64; 3], sky_unit: [f64; 3]) -> f64 {
    let radius = norm3(sun_from_earth_km);
    if radius <= 0.0 {
        return 0.0;
    }
    let projection = dot3(sun_from_earth_km, sky_unit);
    let argument = (radius - projection).abs().max(1.0);
    -2.0 * GM_SUN_KM3_S2 / C_KM_PER_S.powi(3) * argument.ln()
}

fn selector_matches_observation(
    selectors: &[super::timing_model::SelectorTerm],
    observation: &IndependentObservation,
) -> bool {
    selectors.iter().all(|selector| {
        if selector.flag == "-tel" {
            selector.value == observation.site.as_str()
        } else {
            observation
                .flags
                .get(&selector.flag)
                .is_some_and(|value| value == &selector.value)
        }
    })
}

fn tagged_term_matches_observation(
    term: &super::timing_model::TaggedTerm,
    observation: &IndependentObservation,
) -> bool {
    selector_matches_observation(&term.selectors, observation)
}

fn effective_phase_sigma_seconds(model: &TimingModel, observation: &IndependentObservation) -> f64 {
    let mut efac = 1.0;
    let mut equad_us2 = 0.0;
    for term in &model.noise_terms {
        if !tagged_term_matches_observation(term, observation) {
            continue;
        }
        match term.name.as_str() {
            "EFAC" | "TNEF" => {
                efac *= term.value.unwrap_or(1.0).abs().max(1.0e-6);
            }
            "EQUAD" | "TNEQ" => {
                let value = term.value.unwrap_or(0.0);
                equad_us2 += value * value;
            }
            _ => {}
        }
    }
    let formal_s = observation.uncertainty_us.max(1.0e-6) * 1.0e-6;
    let equad_s = equad_us2.sqrt() * 1.0e-6;
    (((efac * formal_s).powi(2) + equad_s.powi(2)).sqrt()).max(1.0e-12)
}

fn effective_dm_sigma(
    model: &TimingModel,
    observation: &IndependentObservation,
    pp_dme: f64,
) -> f64 {
    let mut dmefac = 1.0;
    let mut dmequad2 = 0.0;
    for term in &model.noise_terms {
        if !tagged_term_matches_observation(term, observation) {
            continue;
        }
        match term.name.as_str() {
            "DMEFAC" => {
                dmefac *= term.value.unwrap_or(1.0).abs().max(1.0e-6);
            }
            "DMEQUAD" => {
                let value = term.value.unwrap_or(0.0);
                dmequad2 += value * value;
            }
            _ => {}
        }
    }
    (((dmefac * pp_dme.max(1.0e-9)).powi(2) + dmequad2).sqrt()).max(1.0e-12)
}


fn solve_weighted_least_squares(
    design: &DMatrix<f64>,
    response: &DVector<f64>,
    sigma: &[f64],
) -> Result<DVector<f64>> {
    let nrows = design.nrows();
    let ncols = design.ncols();
    let mut weighted_design = FaerMat::zeros(nrows, ncols);
    let mut weighted_response = FaerMat::zeros(nrows, 1);
    for row in 0..nrows {
        let weight = 1.0 / sigma[row].max(1.0e-18);
        weighted_response[(row, 0)] = response[row] * weight;
        for col in 0..ncols {
            weighted_design[(row, col)] = design[(row, col)] * weight;
        }
    }
    let col_norms = faer_column_norms(&weighted_design);
    scale_faer_columns(&mut weighted_design, &col_norms, true);
    let qr = weighted_design.col_piv_qr();
    let mut scaled = qr.solve_lstsq(&weighted_response);
    for (col, norm) in col_norms.iter().enumerate() {
        scaled[(col, 0)] /= *norm;
    }
    Ok(faer_col_to_dvector(&scaled))
}

fn solve_generalized_least_squares(
    design: &DMatrix<f64>,
    response: &DVector<f64>,
    covariance: &StructuredCovariance,
) -> Result<DVector<f64>> {
    let design_faer = dmatrix_to_faer(design);
    let response_faer = dvector_to_faer_col(response);
    let cinv_design = apply_inverse_covariance_to_matrix(covariance, &design_faer)?;
    let cinv_response = apply_inverse_covariance_to_matrix(covariance, &response_faer)?;
    let normal = design_faer.transpose() * &cinv_design;
    let rhs = design_faer.transpose() * &cinv_response;
    let col_norms = (0..normal.nrows())
        .map(|index| normal[(index, index)].abs().sqrt().max(1.0e-15))
        .collect::<Vec<_>>();
    let mut normal_scaled = normal.clone();
    let mut rhs_scaled = rhs.clone();
    for row in 0..normal_scaled.nrows() {
        rhs_scaled[(row, 0)] /= col_norms[row];
        for col in 0..normal_scaled.ncols() {
            normal_scaled[(row, col)] /= col_norms[row] * col_norms[col];
        }
    }
    let mut scaled = solve_square_system_faer(&normal_scaled, &rhs_scaled)?;
    for (row, norm) in col_norms.iter().enumerate() {
        scaled[(row, 0)] /= *norm;
    }
    Ok(faer_col_to_dvector(&scaled))
}

fn build_joint_gls_covariance(
    dataset: &IndependentDataset,
    joint: &JointSystem,
    preset: &CovariancePreset,
    // Floor for the noise amplitude in the temporal correlation term.
    // When the WLS residual RMS is much larger than formal TOA uncertainties
    // (typical for independent-engine v1 fits), use the WLS RMS as the effective
    // noise scale so the GLS whitening operates at the right amplitude.
    wls_rms_floor_s: f64,
    dm_rms_floor: f64,
) -> (StructuredCovariance, StructuredCovarianceCalibration, f64) {
    let nrows = joint.response.len();
    let n_phase = dataset.observations.len();
    let corr_length_days = preset.corr_length_days.max(1.0e-6);
    let phase_formal_floor = median_value(&joint.sigma[..n_phase]);
    let dm_sigmas = joint
        .dm_row_of_phase
        .iter()
        .flatten()
        .map(|index| joint.sigma[*index])
        .collect::<Vec<_>>();
    let dm_formal_floor = median_value(&dm_sigmas);
    let mjd_min = dataset
        .observations
        .first()
        .map(|observation| observation.mjd_utc)
        .unwrap_or(0.0);
    let mjd_max = dataset
        .observations
        .last()
        .map(|observation| observation.mjd_utc)
        .unwrap_or(mjd_min);
    let mjd_span = (mjd_max - mjd_min).max(1.0);
    let mut calibration = calibrate_structured_covariance(StructuredCovarianceInputs {
        n_phase,
        n_dm: joint.dm_observation_count,
        mjd_span,
        corr_length_days,
        red_noise_fraction: preset.red_noise_fraction,
        phase_formal_floor,
        dm_formal_floor,
        wls_phase_rms_floor_s: wls_rms_floor_s,
        dm_rms_floor,
    });
    calibration.phase_amp_s *= preset.phase_amp_scale;
    calibration.dm_amp *= preset.dm_amp_scale;
    if let Some(value) = preset.phase_fourier_harmonics {
        calibration.phase_fourier_harmonics = value;
    }
    if let Some(value) = preset.phase_short_basis_count {
        calibration.phase_short_basis_count = value.min(n_phase.max(1));
    }
    if let Some(value) = preset.phase_long_basis_count {
        calibration.phase_long_basis_count = value.min(n_phase.max(1));
    }
    if let Some(value) = preset.dm_fourier_harmonics {
        calibration.dm_fourier_harmonics = value;
    }
    if let Some(value) = preset.dm_short_basis_count {
        calibration.dm_short_basis_count = value.min(joint.dm_observation_count.max(1));
    }
    if let Some(value) = preset.dm_long_basis_count {
        calibration.dm_long_basis_count = value.min(joint.dm_observation_count.max(1));
    }
    let long_corr_days = (5.0 * corr_length_days).max(corr_length_days);
    let dm_phase_for_row = invert_dm_row_map(&joint.dm_row_of_phase, nrows);
    let ecorr_columns = ecorr_column_map(dataset);
    let phase_short_centers =
        evenly_spaced_centers(mjd_min, mjd_max, calibration.phase_short_basis_count);
    let phase_long_centers =
        evenly_spaced_centers(mjd_min, mjd_max, calibration.phase_long_basis_count);
    let dm_short_centers =
        evenly_spaced_centers(mjd_min, mjd_max, calibration.dm_short_basis_count);
    let dm_long_centers = evenly_spaced_centers(mjd_min, mjd_max, calibration.dm_long_basis_count);
    let low_rank_cols = 2 * calibration.phase_fourier_harmonics
        + phase_short_centers.len()
        + phase_long_centers.len()
        + 2 * calibration.dm_fourier_harmonics
        + dm_short_centers.len()
        + dm_long_centers.len()
        + ecorr_columns.len();
    let row_block = 128usize;
    let blocks = (0..nrows)
        .step_by(row_block)
        .map(|start| (start, (start + row_block).min(nrows)))
        .collect::<Vec<_>>();
    let assembled = blocks
        .into_par_iter()
        .map(|(start, end)| {
            let mut diagonal = vec![0.0; end - start];
            let mut low_rank = vec![0.0; (end - start) * low_rank_cols];
            for (global_row, dm_phase_entry) in
                dm_phase_for_row.iter().enumerate().take(end).skip(start)
            {
                let local_row = global_row - start;
                let mut cursor = 0usize;
                if global_row < n_phase {
                    let observation = &dataset.observations[global_row];
                    let effective_sigma =
                        joint.sigma[global_row].max(0.5 * calibration.phase_white_floor_s);
                    diagonal[local_row] = effective_sigma * effective_sigma;
                    write_fourier_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        calibration.phase_fourier_harmonics,
                        13.0 / 3.0,
                        calibration.phase_amp_s,
                        (observation.mjd_utc - mjd_min) / mjd_span,
                    );
                    cursor += 2 * calibration.phase_fourier_harmonics;
                    write_kernel_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        &phase_short_centers,
                        observation.mjd_utc,
                        corr_length_days,
                        calibration.phase_amp_s,
                        1.0,
                        matern_three_halves,
                    );
                    cursor += phase_short_centers.len();
                    write_kernel_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        &phase_long_centers,
                        observation.mjd_utc,
                        long_corr_days,
                        calibration.phase_amp_s,
                        0.125,
                        gaussian_kernel,
                    );
                    cursor += phase_long_centers.len();
                    let coeff = DM_DELAY_S_PER_MHZ2_DM
                        / (observation.frequency_mhz * observation.frequency_mhz);
                    write_fourier_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        calibration.dm_fourier_harmonics,
                        8.0 / 3.0,
                        -coeff * calibration.dm_amp,
                        (observation.mjd_utc - mjd_min) / mjd_span,
                    );
                    cursor += 2 * calibration.dm_fourier_harmonics;
                    write_kernel_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        &dm_short_centers,
                        observation.mjd_utc,
                        long_corr_days,
                        -coeff * calibration.dm_amp,
                        0.5,
                        matern_three_halves,
                    );
                    cursor += dm_short_centers.len();
                    write_kernel_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        &dm_long_centers,
                        observation.mjd_utc,
                        long_corr_days,
                        -coeff * calibration.dm_amp,
                        0.5,
                        gaussian_kernel,
                    );
                    cursor += dm_long_centers.len();
                    populate_ecorr_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor
                            ..local_row * low_rank_cols + low_rank_cols],
                        dataset,
                        global_row,
                        &ecorr_columns,
                    );
                } else if let Some(phase_index) = *dm_phase_entry {
                    let observation = &dataset.observations[phase_index];
                    let effective_sigma =
                        joint.sigma[global_row].max(0.5 * calibration.dm_white_floor);
                    diagonal[local_row] = effective_sigma * effective_sigma;
                    cursor += 2 * calibration.phase_fourier_harmonics
                        + phase_short_centers.len()
                        + phase_long_centers.len();
                    write_fourier_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        calibration.dm_fourier_harmonics,
                        8.0 / 3.0,
                        calibration.dm_amp,
                        (observation.mjd_utc - mjd_min) / mjd_span,
                    );
                    cursor += 2 * calibration.dm_fourier_harmonics;
                    write_kernel_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        &dm_short_centers,
                        observation.mjd_utc,
                        long_corr_days,
                        calibration.dm_amp,
                        0.5,
                        matern_three_halves,
                    );
                    cursor += dm_short_centers.len();
                    write_kernel_basis_row(
                        &mut low_rank[local_row * low_rank_cols + cursor..],
                        &dm_long_centers,
                        observation.mjd_utc,
                        long_corr_days,
                        calibration.dm_amp,
                        0.5,
                        gaussian_kernel,
                    );
                } else {
                    diagonal[local_row] = joint.sigma[global_row] * joint.sigma[global_row];
                }
            }
            (start, diagonal, low_rank)
        })
        .collect::<Vec<_>>();
    let mut diagonal = vec![0.0; nrows];
    let mut low_rank = FaerMat::zeros(nrows, low_rank_cols);
    for (start, block_diag, block_low_rank) in assembled {
        let rows = block_diag.len();
        diagonal[start..start + rows].copy_from_slice(&block_diag);
        for local_row in 0..rows {
            for col in 0..low_rank_cols {
                low_rank[(start + local_row, col)] =
                    block_low_rank[local_row * low_rank_cols + col];
            }
        }
    }
    let (covariance, ridge) = finalize_structured_covariance(diagonal, low_rank);
    let calibration = StructuredCovarianceCalibration {
        preset_name: preset.name.to_string(),
        tactical_lane: preset.tactical_lane.to_string(),
        provisional_acceptance_policy: preset.provisional_acceptance_policy.to_string(),
        corr_length_days: preset.corr_length_days,
        red_noise_fraction: preset.red_noise_fraction,
        ecorr_basis_count: ecorr_columns.len(),
        ..calibration
    };
    (covariance, calibration, ridge)
}

fn finalize_structured_covariance(
    mut diagonal: Vec<f64>,
    low_rank: FaerMat<f64>,
) -> (StructuredCovariance, f64) {
    let mut ridge_factor: f64 = 0.0;
    for value in &mut diagonal {
        if !value.is_finite() || *value <= 0.0 {
            *value = 1.0e-24;
            ridge_factor = ridge_factor.max(1.0e-24);
        }
    }
    let inv_diagonal = diagonal
        .iter()
        .map(|value| 1.0 / value.max(1.0e-24))
        .collect::<Vec<_>>();
    let mut inv_low_rank = low_rank.clone();
    for row in 0..inv_low_rank.nrows() {
        let scale = inv_diagonal[row];
        for col in 0..inv_low_rank.ncols() {
            inv_low_rank[(row, col)] *= scale;
        }
    }
    let mut middle = low_rank.transpose() * &inv_low_rank;
    for index in 0..middle.nrows() {
        middle[(index, index)] += 1.0;
    }
    let mut local_ridge = 1.0e-15;
    while low_rank.ncols() > 0 && middle.clone().llt(Side::Lower).is_err() {
        for index in 0..middle.nrows() {
            middle[(index, index)] += local_ridge;
        }
        ridge_factor = ridge_factor.max(local_ridge);
        local_ridge *= 10.0;
        if local_ridge > 1.0 {
            break;
        }
    }
    (
        StructuredCovariance {
            low_rank,
            inv_diagonal,
            inv_low_rank,
            middle,
        },
        ridge_factor,
    )
}

fn invert_dm_row_map(dm_row_of_phase: &[Option<usize>], nrows: usize) -> Vec<Option<usize>> {
    let mut phase_for_row = vec![None; nrows];
    for (phase_index, dm_row) in dm_row_of_phase.iter().enumerate() {
        if let Some(dm_row) = *dm_row {
            phase_for_row[dm_row] = Some(phase_index);
        }
    }
    phase_for_row
}

fn calibrate_structured_covariance(
    inputs: StructuredCovarianceInputs,
) -> StructuredCovarianceCalibration {
    let StructuredCovarianceInputs {
        n_phase,
        n_dm,
        mjd_span,
        corr_length_days,
        red_noise_fraction,
        phase_formal_floor,
        dm_formal_floor,
        wls_phase_rms_floor_s,
        dm_rms_floor,
    } = inputs;
    let corr_length_days = corr_length_days.max(1.0);
    let long_corr_days = (5.0 * corr_length_days).max(corr_length_days);
    let phase_white_floor_s = wls_phase_rms_floor_s.max(phase_formal_floor);
    let dm_white_floor = dm_rms_floor.max(dm_formal_floor);
    let phase_excess = (wls_phase_rms_floor_s * wls_phase_rms_floor_s
        - phase_formal_floor * phase_formal_floor)
        .max(0.0)
        .sqrt();
    let dm_excess = (dm_rms_floor * dm_rms_floor - dm_formal_floor * dm_formal_floor)
        .max(0.0)
        .sqrt();
    let phase_amp_s = red_noise_fraction.max(0.0) * phase_excess.max(0.25 * phase_white_floor_s);
    let dm_amp = red_noise_fraction.max(0.0) * dm_excess.max(0.25 * dm_white_floor);
    let phase_fourier_harmonics = (((n_phase as f64).sqrt() / 10.0).round() as usize).clamp(4, 12);
    let dm_fourier_harmonics = (((n_dm.max(1) as f64).sqrt() / 12.0).round() as usize).clamp(3, 8);
    let phase_short_basis_count = ((mjd_span / corr_length_days).ceil() as usize)
        .clamp(6, 24)
        .min(n_phase.max(1));
    let phase_long_basis_count = ((mjd_span / long_corr_days).ceil() as usize)
        .clamp(4, 12)
        .min(n_phase.max(1));
    let dm_short_basis_count = ((mjd_span / long_corr_days).ceil() as usize)
        .clamp(4, 16)
        .min(n_dm.max(1));
    let dm_long_basis_count = ((mjd_span / (2.0 * long_corr_days)).ceil() as usize)
        .clamp(3, 10)
        .min(n_dm.max(1));
    StructuredCovarianceCalibration {
        preset_name: "generic".to_string(),
        tactical_lane: "unspecified".to_string(),
        provisional_acceptance_policy: "synthesis".to_string(),
        corr_length_days,
        red_noise_fraction,
        phase_white_floor_s,
        dm_white_floor,
        phase_amp_s,
        dm_amp,
        phase_fourier_harmonics,
        phase_short_basis_count,
        phase_long_basis_count,
        dm_fourier_harmonics,
        dm_short_basis_count,
        dm_long_basis_count,
        ecorr_basis_count: 0,
    }
}

fn ecorr_column_map(dataset: &IndependentDataset) -> Vec<(usize, i64)> {
    let ecorr_terms = dataset
        .model
        .noise_terms
        .iter()
        .enumerate()
        .filter(|(_, term)| term.name == "ECORR")
        .collect::<Vec<_>>();
    let mut keys = BTreeSet::new();
    for observation in &dataset.observations {
        let mjd_bucket = observation.mjd_utc.round() as i64;
        for (term_index, term) in &ecorr_terms {
            if tagged_term_matches_observation(term, observation) {
                keys.insert((*term_index, mjd_bucket));
            }
        }
    }
    keys.into_iter().collect()
}

fn populate_ecorr_basis_row(
    destination: &mut [f64],
    dataset: &IndependentDataset,
    phase_row: usize,
    ecorr_columns: &[(usize, i64)],
) {
    if phase_row >= dataset.observations.len() {
        return;
    }
    let observation = &dataset.observations[phase_row];
    let mjd_bucket = observation.mjd_utc.round() as i64;
    for (slot, (term_index, bucket)) in ecorr_columns.iter().enumerate() {
        let Some(term) = dataset.model.noise_terms.get(*term_index) else {
            continue;
        };
        if *bucket == mjd_bucket && tagged_term_matches_observation(term, observation) {
            destination[slot] = term.value.unwrap_or(0.0).abs() * 1.0e-6;
        }
    }
}

fn evenly_spaced_centers(mjd_min: f64, mjd_max: f64, count: usize) -> Vec<f64> {
    if count == 0 {
        return Vec::new();
    }
    let span = (mjd_max - mjd_min).max(1.0);
    (0..count)
        .map(|index| {
            let fraction = (index as f64 + 0.5) / count as f64;
            mjd_min + fraction * span
        })
        .collect()
}

fn write_kernel_basis_row(
    destination: &mut [f64],
    centers: &[f64],
    mjd: f64,
    corr_length_days: f64,
    amplitude: f64,
    kernel_weight: f64,
    kernel: fn(f64) -> f64,
) {
    if centers.is_empty() {
        return;
    }
    let scale = amplitude * (kernel_weight / centers.len() as f64).sqrt();
    for (slot, center) in centers.iter().enumerate() {
        destination[slot] = scale * kernel((mjd - center).abs() / corr_length_days.max(1.0e-9));
    }
}

fn write_fourier_basis_row(
    destination: &mut [f64],
    harmonics: usize,
    spectral_index: f64,
    amplitude: f64,
    phase: f64,
) {
    let two_pi_phase = 2.0 * PI * phase;
    let harmonic_count = harmonics.min(destination.len() / 2);
    let mut harmonic = 1usize;
    let mut offset = 0usize;
    while harmonic <= harmonic_count {
        let omega = two_pi_phase * harmonic as f64;
        let weight = amplitude * fourier_basis_weight(harmonic, spectral_index);
        destination[offset] = weight * omega.cos();
        destination[offset + 1] = weight * omega.sin();
        harmonic += 1;
        offset += 2;
    }
}

fn fourier_basis_weight(harmonic: usize, spectral_index: f64) -> f64 {
    0.5_f64.sqrt() / (harmonic as f64).powf(0.5 * spectral_index)
}



#[cfg(test)]
mod tests {
    use super::{
        SiteId, dominant_subgroup, ecliptic_to_equatorial, ell1h_shapiro_approximate_seconds,
        ell1h_shapiro_exact_seconds, equatorial_to_ecliptic, matern_three_halves,
        parse_tempo2_toas,
    };
    use std::path::Path;

    #[test]
    fn parses_format1_tim_and_subgroup() {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../data/external/nanograv_15yr_timing/NANOGrav15yr_PulsarTiming_v2.1.0/wideband/tim/J2214+3000_PINT_20230131.wb.tim");
        let toas = parse_tempo2_toas(&path).expect("parse TOAs");
        assert!(!toas.is_empty());
        assert!(toas.iter().all(|toa| toa.flags.contains_key("-f")));
        let (subgroup, count) = dominant_subgroup(&toas).expect("dominant subgroup");
        assert!(!subgroup.is_empty());
        assert!(count > 0);
    }

    #[test]
    fn site_tokens_map_to_known_sites() {
        assert_eq!(SiteId::from_token("arecibo").expect("ao"), SiteId::Arecibo);
        assert_eq!(SiteId::from_token("GB").expect("gb"), SiteId::GreenBank);
        assert_eq!(SiteId::from_token("6").expect("vla"), SiteId::Vla);
    }

    #[test]
    fn obliquity_rotation_round_trips() {
        let ecliptic = [0.3, -0.4, 0.866_025_403_784];
        let equatorial = ecliptic_to_equatorial(ecliptic);
        let round_trip = equatorial_to_ecliptic(equatorial);
        for index in 0..3 {
            assert!((round_trip[index] - ecliptic[index]).abs() < 1.0e-12);
        }
    }

    #[test]
    fn matern_kernel_is_positive_and_decays() {
        assert!((matern_three_halves(0.0) - 1.0).abs() < 1.0e-12);
        assert!(matern_three_halves(0.5) > matern_three_halves(2.0));
    }

    #[test]
    fn ell1h_h3_only_reduces_to_third_harmonic() {
        let h3 = 2.5e-6;
        let phi = 0.37;
        let delay = ell1h_shapiro_approximate_seconds(h3, 0.0, phi, 7);
        let expected = -(4.0 / 3.0) * h3 * (3.0 * phi).sin();
        assert!((delay - expected).abs() < 1.0e-18);
    }

    #[test]
    fn ell1h_exact_matches_closed_form() {
        let h3 = 4.0e-6;
        let stigma = 0.75;
        let phi = 0.41;
        let delay = ell1h_shapiro_exact_seconds(h3, stigma, phi);
        let expected =
            -2.0 * h3 / stigma.powi(3) * (1.0 + stigma * stigma - 2.0 * stigma * phi.sin()).ln();
        assert!((delay - expected).abs() < 1.0e-18);
    }
}
