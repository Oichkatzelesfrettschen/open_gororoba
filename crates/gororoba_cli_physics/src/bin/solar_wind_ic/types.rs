//! Cli struct for the `solar-wind-ic` binary. ~190 lines of clap
//! Parser-derived configuration options. Fields are pub(crate) so
//! the bin root can read them. Uses #[path] indirection because
//! this binary has explicit Cargo.toml path.

use clap::Parser;
use std::path::PathBuf;

/// Generate solar wind initial conditions from real spacecraft data.
///
/// Maps hourly measurements to a 3D LBM grid via Taylor's frozen-in
/// hypothesis. Transverse (y,z) planes are uniform (single spacecraft
/// has no spatial resolution perpendicular to the flow).
#[derive(Parser)]
#[command(name = "solar-wind-ic")]
pub(crate) struct Cli {
    /// Path to NASA OMNI2 hourly data file (preferred: includes B-field).
    /// If not provided, uses built-in OMNI sample.
    #[arg(long)]
    pub(crate) omni_file: Option<PathBuf>,

    /// Path to ACE SWEPAM file (legacy: plasma only, no B-field).
    /// When used, B-field falls back to Parker spiral model.
    #[arg(long)]
    pub(crate) swepam_file: Option<PathBuf>,

    /// Path to ACE MAG L2 file (16-sec B-field data, averaged to hourly).
    /// Provides independent B-field measurements; plasma uses defaults.
    #[arg(long)]
    pub(crate) ace_mag_file: Option<PathBuf>,

    /// Path to SOHO CELIAS Proton Monitor mission-long tar.gz bundle.
    /// Uses either native cadence or hourly median downsampling plus Parker
    /// spiral B-field fallback.
    #[arg(long)]
    pub(crate) soho_celias_file: Option<PathBuf>,

    /// SOHO CELIAS cadence selection.
    /// `auto` uses native cadence when the requested time resolution is
    /// 15 minutes or finer, and hourly medians otherwise.
    #[arg(long, default_value = "auto")]
    pub(crate) soho_celias_cadence: String,

    /// Path to WIND SWE key-parameter file (plasma: density, speed, temp).
    #[arg(long)]
    pub(crate) wind_swe_file: Option<PathBuf>,

    /// Path to WIND MFI file (magnetic field in GSE, hourly averaged).
    /// If --wind-swe-file is also provided, merges plasma + B-field.
    #[arg(long)]
    pub(crate) wind_mfi_file: Option<PathBuf>,

    /// Path to STEREO-A PLASTIC file (plasma in RTN coordinates).
    #[arg(long)]
    pub(crate) stereo_file: Option<PathBuf>,

    /// Path to STEREO-A IMPACT/MAG file (B-field in RTN coordinates).
    #[arg(long)]
    pub(crate) stereo_mag_file: Option<PathBuf>,

    /// STEREO-A heliocentric separation angle from Earth (degrees).
    /// Required for RTN -> GSE coordinate transform.
    #[arg(long, default_value_t = 0.0)]
    pub(crate) stereo_sep_deg: f64,

    /// Enable L1+STEREO-A 3D triangulation mode.
    /// Y-axis maps to heliocentric longitude: y=0 is pure L1 data,
    /// y=ny-1 is pure STEREO-A data, intermediate slices are linearly
    /// interpolated. Requires both L1 data (OMNI/ACE/WIND) and STEREO.
    #[arg(long, default_value_t = false)]
    pub(crate) triangulate: bool,

    /// Enable radial profile IC mode: x-axis spans r_min..r_max AU
    /// using multi-spacecraft data at different heliocentric distances.
    /// Requires at least 2 spacecraft files at different distances.
    #[arg(long, default_value_t = false)]
    pub(crate) radial_mode: bool,

    /// Minimum heliocentric distance for radial mode (AU).
    #[arg(long, default_value_t = 1.0)]
    pub(crate) r_min_au: f64,

    /// Maximum heliocentric distance for radial mode (AU).
    #[arg(long, default_value_t = 100.0)]
    pub(crate) r_max_au: f64,

    /// Path to Voyager 1 SPDF merged hourly file.
    #[arg(long)]
    pub(crate) voyager1_file: Option<PathBuf>,

    /// Path to Voyager 2 SPDF merged hourly file.
    #[arg(long)]
    pub(crate) voyager2_file: Option<PathBuf>,

    /// Path to Pioneer 10 SPDF merged hourly file.
    #[arg(long)]
    pub(crate) pioneer10_file: Option<PathBuf>,

    /// Path to Pioneer 11 SPDF merged hourly file.
    #[arg(long)]
    pub(crate) pioneer11_file: Option<PathBuf>,

    /// Path to New Horizons SWAP hourly file (no magnetometer).
    #[arg(long)]
    pub(crate) nh_swap_file: Option<PathBuf>,

    /// Path to Juno cruise SPDF merged hourly file.
    #[arg(long)]
    pub(crate) juno_file: Option<PathBuf>,

    /// Path to Cassini cruise SPDF merged hourly file.
    #[arg(long)]
    pub(crate) cassini_file: Option<PathBuf>,

    /// Path to Ulysses SWOOPS plasma file (has heliographic latitude).
    #[arg(long)]
    pub(crate) ulysses_swoops_file: Option<PathBuf>,

    /// Path to Ulysses VHM/FGM magnetic field file (RTN coordinates).
    #[arg(long)]
    pub(crate) ulysses_mag_file: Option<PathBuf>,

    /// Time resolution per x-slice in seconds (default: 3600 = hourly).
    /// For WIND MFI 3-second data, use --time-resolution 3 to resolve
    /// CME shock ramps (nx=128 at 3s covers 384s = 6.4 min of shock transit).
    #[arg(long, default_value_t = 3600)]
    pub(crate) time_resolution: u32,

    /// Density clamp range [min, max] in LBM units for shock mode.
    /// Default: 0.1,10.0. For high-variability shock data: 0.01,50.0.
    #[arg(long, default_value = "0.1,10.0")]
    pub(crate) clamp_density_range: String,

    /// Speed clamp range [min, max] in LBM units for shock mode.
    /// Default: 0.001,0.15. For CME shock data: 0.0001,0.25.
    #[arg(long, default_value = "0.001,0.15")]
    pub(crate) clamp_speed_range: String,

    /// Start hour index within the data file (0-based).
    #[arg(long, default_value_t = 0)]
    pub(crate) start_hour: usize,

    /// Number of hours to map along x-axis.
    /// 0 = auto (use min(nx, available_hours)).
    #[arg(long, default_value_t = 0)]
    pub(crate) num_hours: usize,

    /// Grid size in x (radial/Sun-Earth direction)
    #[arg(long, default_value_t = 128)]
    pub(crate) nx: usize,

    /// Grid size in y (ecliptic transverse)
    #[arg(long, default_value_t = 32)]
    pub(crate) ny: usize,

    /// Grid size in z (ecliptic north)
    #[arg(long, default_value_t = 32)]
    pub(crate) nz: usize,

    /// B-field scale factor for LBM units.
    /// Physical B (nT) is multiplied by this to get LBM B.
    /// Typical: 0.001 maps 5 nT -> 0.005 LBM units.
    #[arg(long, default_value_t = 0.001)]
    pub(crate) b_scale: f64,

    /// Parker spiral omega (only used for SWEPAM fallback, rad/s).
    #[arg(long, default_value_t = 2.662e-6)]
    pub(crate) omega: f64,

    /// Output format: "volume" (single 3D CSV) or "slices" (one CSV per z)
    #[arg(long, default_value = "volume")]
    pub(crate) format: String,

    /// Output path (file for volume, directory for slices)
    #[arg(long, default_value = "solar_wind_ic.csv")]
    pub(crate) out: PathBuf,

    /// Optional CSV output for measured radial profile bins.
    #[arg(long)]
    pub(crate) radial_profile_out: Option<PathBuf>,

    /// Optional CSV output for interpolated radial profile samples.
    #[arg(long)]
    pub(crate) radial_sample_out: Option<PathBuf>,

    /// Optional CSV output for radial scaling fit diagnostics.
    #[arg(long)]
    pub(crate) radial_fit_out: Option<PathBuf>,

    /// Enable latitudinal Z-axis gradient modulation.
    /// Z-axis maps to heliographic latitude: z=0 is -lat_max,
    /// z=nz/2 is equator, z=nz-1 is +lat_max. Fast polar wind
    /// (750 km/s, 3 cm^-3) transitions to slow equatorial wind
    /// (400 km/s, 7 cm^-3) via tanh profile based on Ulysses data.
    #[arg(long, default_value_t = false)]
    pub(crate) latitudinal: bool,

    /// Maximum heliographic latitude for latitudinal mode (degrees).
    #[arg(long, default_value_t = 30.0)]
    pub(crate) lat_max_deg: f64,
}
