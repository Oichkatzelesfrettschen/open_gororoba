//! Real-data solar wind initial condition generator.
//!
//! Loads NASA OMNI2 hourly data (proton density, bulk speed, ion temperature,
//! Bx/By/Bz in GSE coordinates) and maps them to 3D LBM grid initial
//! conditions using Taylor's frozen-in hypothesis.
//!
//! Supports multiple spacecraft data sources via the adapter pattern:
//!   - NASA OMNI2 (preferred: merged multi-source, includes B-field)
//!   - ACE SWEPAM (plasma only, Parker spiral B-field fallback)
//!   - SOHO CELIAS Proton Monitor mission-long bundle (5-min -> hourly medians)
//!   - ACE MAG L2 (B-field only, 16-sec -> hourly averaged)
//!   - WIND SWE + MFI (independent L1 spacecraft, plasma + B-field)
//!   - STEREO-A PLASTIC + IMPACT/MAG (different heliocentric longitude)
//!
//! All data flows through OmniRecord as the universal interchange format.
//!
//! GSE coordinate convention:
//!   X -> Sun-Earth line (radial, our LBM x-axis)
//!   Y -> ecliptic plane, dawn side (our LBM y-axis)
//!   Z -> ecliptic north (our LBM z-axis)
//!
//! Output: CSV files with columns x,y,z,rho,ux,uy,uz,bx,by,bz matching the
//! snapshot format used by solar-wind-mhd-sim and solar-wind-dm-mhd.

use clap::Parser;
use data_core::catalogs::{
    ace_mag::{ace_mag_to_omni, average_to_hourly, parse_ace_mag_file},
    cassini::{cassini_to_omni, parse_cassini_cruise_file},
    juno::{juno_to_omni, parse_juno_cruise_file},
    new_horizons::{nh_swap_to_omni, parse_nh_swap_file},
    omni::{OmniRecord, parse_omni_file, parse_omni_hourly},
    pioneer::{PioneerSpacecraft, parse_pioneer_file, pioneer_to_omni},
    soho_celias::{parse_soho_celias_bundle_file, soho_to_hourly_omni, soho_to_native_omni},
    solar_wind::parse_swepam_file,
    stereo_plastic::{
        average_stereo_mag_hourly, parse_stereo_magplasma_file, parse_stereo_plastic_file,
        stereo_to_omni,
    },
    ulysses::{parse_ulysses_file, ulysses_to_omni},
    voyager::{VoyagerSpacecraft, parse_voyager_file, voyager_to_omni},
    wind_swe::{
        KnudsenRegime, classify_knudsen, knudsen_number, merge_wind_swe_mfi, parse_wind_mfi_file,
        parse_wind_swe_file, wind_mfi_to_omni,
    },
};
use std::{fs, io::Write, path::PathBuf};

/// Generate solar wind initial conditions from real spacecraft data.
///
/// Maps hourly measurements to a 3D LBM grid via Taylor's frozen-in
/// hypothesis. Transverse (y,z) planes are uniform (single spacecraft
/// has no spatial resolution perpendicular to the flow).
#[derive(Parser)]
#[command(name = "solar-wind-ic")]
struct Cli {
    /// Path to NASA OMNI2 hourly data file (preferred: includes B-field).
    /// If not provided, uses built-in OMNI sample.
    #[arg(long)]
    omni_file: Option<PathBuf>,

    /// Path to ACE SWEPAM file (legacy: plasma only, no B-field).
    /// When used, B-field falls back to Parker spiral model.
    #[arg(long)]
    swepam_file: Option<PathBuf>,

    /// Path to ACE MAG L2 file (16-sec B-field data, averaged to hourly).
    /// Provides independent B-field measurements; plasma uses defaults.
    #[arg(long)]
    ace_mag_file: Option<PathBuf>,

    /// Path to SOHO CELIAS Proton Monitor mission-long tar.gz bundle.
    /// Uses either native cadence or hourly median downsampling plus Parker
    /// spiral B-field fallback.
    #[arg(long)]
    soho_celias_file: Option<PathBuf>,

    /// SOHO CELIAS cadence selection.
    /// `auto` uses native cadence when the requested time resolution is
    /// 15 minutes or finer, and hourly medians otherwise.
    #[arg(long, default_value = "auto")]
    soho_celias_cadence: String,

    /// Path to WIND SWE key-parameter file (plasma: density, speed, temp).
    #[arg(long)]
    wind_swe_file: Option<PathBuf>,

    /// Path to WIND MFI file (magnetic field in GSE, hourly averaged).
    /// If --wind-swe-file is also provided, merges plasma + B-field.
    #[arg(long)]
    wind_mfi_file: Option<PathBuf>,

    /// Path to STEREO-A PLASTIC file (plasma in RTN coordinates).
    #[arg(long)]
    stereo_file: Option<PathBuf>,

    /// Path to STEREO-A IMPACT/MAG file (B-field in RTN coordinates).
    #[arg(long)]
    stereo_mag_file: Option<PathBuf>,

    /// STEREO-A heliocentric separation angle from Earth (degrees).
    /// Required for RTN -> GSE coordinate transform.
    #[arg(long, default_value_t = 0.0)]
    stereo_sep_deg: f64,

    /// Enable L1+STEREO-A 3D triangulation mode.
    /// Y-axis maps to heliocentric longitude: y=0 is pure L1 data,
    /// y=ny-1 is pure STEREO-A data, intermediate slices are linearly
    /// interpolated. Requires both L1 data (OMNI/ACE/WIND) and STEREO.
    #[arg(long, default_value_t = false)]
    triangulate: bool,

    /// Enable radial profile IC mode: x-axis spans r_min..r_max AU
    /// using multi-spacecraft data at different heliocentric distances.
    /// Requires at least 2 spacecraft files at different distances.
    #[arg(long, default_value_t = false)]
    radial_mode: bool,

    /// Minimum heliocentric distance for radial mode (AU).
    #[arg(long, default_value_t = 1.0)]
    r_min_au: f64,

    /// Maximum heliocentric distance for radial mode (AU).
    #[arg(long, default_value_t = 100.0)]
    r_max_au: f64,

    /// Path to Voyager 1 SPDF merged hourly file.
    #[arg(long)]
    voyager1_file: Option<PathBuf>,

    /// Path to Voyager 2 SPDF merged hourly file.
    #[arg(long)]
    voyager2_file: Option<PathBuf>,

    /// Path to Pioneer 10 SPDF merged hourly file.
    #[arg(long)]
    pioneer10_file: Option<PathBuf>,

    /// Path to Pioneer 11 SPDF merged hourly file.
    #[arg(long)]
    pioneer11_file: Option<PathBuf>,

    /// Path to New Horizons SWAP hourly file (no magnetometer).
    #[arg(long)]
    nh_swap_file: Option<PathBuf>,

    /// Path to Juno cruise SPDF merged hourly file.
    #[arg(long)]
    juno_file: Option<PathBuf>,

    /// Path to Cassini cruise SPDF merged hourly file.
    #[arg(long)]
    cassini_file: Option<PathBuf>,

    /// Path to Ulysses SWOOPS plasma file (has heliographic latitude).
    #[arg(long)]
    ulysses_swoops_file: Option<PathBuf>,

    /// Path to Ulysses VHM/FGM magnetic field file (RTN coordinates).
    #[arg(long)]
    ulysses_mag_file: Option<PathBuf>,

    /// Time resolution per x-slice in seconds (default: 3600 = hourly).
    /// For WIND MFI 3-second data, use --time-resolution 3 to resolve
    /// CME shock ramps (nx=128 at 3s covers 384s = 6.4 min of shock transit).
    #[arg(long, default_value_t = 3600)]
    time_resolution: u32,

    /// Density clamp range [min, max] in LBM units for shock mode.
    /// Default: 0.1,10.0. For high-variability shock data: 0.01,50.0.
    #[arg(long, default_value = "0.1,10.0")]
    clamp_density_range: String,

    /// Speed clamp range [min, max] in LBM units for shock mode.
    /// Default: 0.001,0.15. For CME shock data: 0.0001,0.25.
    #[arg(long, default_value = "0.001,0.15")]
    clamp_speed_range: String,

    /// Start hour index within the data file (0-based).
    #[arg(long, default_value_t = 0)]
    start_hour: usize,

    /// Number of hours to map along x-axis.
    /// 0 = auto (use min(nx, available_hours)).
    #[arg(long, default_value_t = 0)]
    num_hours: usize,

    /// Grid size in x (radial/Sun-Earth direction)
    #[arg(long, default_value_t = 128)]
    nx: usize,

    /// Grid size in y (ecliptic transverse)
    #[arg(long, default_value_t = 32)]
    ny: usize,

    /// Grid size in z (ecliptic north)
    #[arg(long, default_value_t = 32)]
    nz: usize,

    /// B-field scale factor for LBM units.
    /// Physical B (nT) is multiplied by this to get LBM B.
    /// Typical: 0.001 maps 5 nT -> 0.005 LBM units.
    #[arg(long, default_value_t = 0.001)]
    b_scale: f64,

    /// Parker spiral omega (only used for SWEPAM fallback, rad/s).
    #[arg(long, default_value_t = 2.662e-6)]
    omega: f64,

    /// Output format: "volume" (single 3D CSV) or "slices" (one CSV per z)
    #[arg(long, default_value = "volume")]
    format: String,

    /// Output path (file for volume, directory for slices)
    #[arg(long, default_value = "solar_wind_ic.csv")]
    out: PathBuf,

    /// Optional CSV output for measured radial profile bins.
    #[arg(long)]
    radial_profile_out: Option<PathBuf>,

    /// Optional CSV output for interpolated radial profile samples.
    #[arg(long)]
    radial_sample_out: Option<PathBuf>,

    /// Optional CSV output for radial scaling fit diagnostics.
    #[arg(long)]
    radial_fit_out: Option<PathBuf>,

    /// Enable latitudinal Z-axis gradient modulation.
    /// Z-axis maps to heliographic latitude: z=0 is -lat_max,
    /// z=nz/2 is equator, z=nz-1 is +lat_max. Fast polar wind
    /// (750 km/s, 3 cm^-3) transitions to slow equatorial wind
    /// (400 km/s, 7 cm^-3) via tanh profile based on Ulysses data.
    #[arg(long, default_value_t = false)]
    latitudinal: bool,

    /// Maximum heliographic latitude for latitudinal mode (degrees).
    #[arg(long, default_value_t = 30.0)]
    lat_max_deg: f64,
}

/// Built-in OMNI2 sample: real data from 2024 DOY 1, hours 0-23.
/// Includes actual measured Bx/By/Bz (GSE) alongside plasma parameters.
/// Source: https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_2024.dat
const BUILTIN_OMNI_SAMPLE: &str = "\
2024   1  0 2596 51 52  60  36   5.3   5.1  -7.3 322.9   4.0  -3.0  -0.6  -2.8  -1.3   0.1   1.5   0.3   0.4   1.5   31114.   7.4  306.  -2.3   2.8 0.042  1.35    4691.   0.4    2.   0.3   1.3 0.007   0.40   1.75   7.9  7  55     0   20 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3    -9    11  5.0
2024   1  1 2596 51 52  53  33   5.4   5.2 -21.3 333.4   4.4  -2.2  -1.9  -1.7  -2.3   0.1   1.1   0.5   0.4   1.0   28455.   6.5  301.  -2.1   1.9 0.040  1.14    4102.   0.4    4.   0.4   1.2 0.008   0.69   1.45   7.1  7  55     2   25 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3   -11    14  4.7
2024   1  2 2596 51 52  63  33   4.5   4.1 -42.3 324.4   2.5  -1.8  -2.8  -1.3  -3.1   0.7   2.0   0.6   1.2   1.4   36413.   7.9  309.  -0.8   1.7 0.034  1.43    2767.   1.0    2.   0.6   1.0 0.008   0.96   2.67   9.7  7  55     4   32 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3   -11    21  5.3
2024   1  3 2596 51 52  58  34   4.2   3.9 -12.2 329.6   3.3  -1.9  -0.8  -1.8  -1.1   0.1   1.4   0.3   0.4   1.3   32023.   8.9  306.  -0.8   1.7 0.037  1.60    1565.   0.3    2.   0.2   0.3 0.003   0.34   3.37  10.9  3  55     5   29 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.5   -10    19  5.5
2024   1  4 2596 51 52  62  34   4.2   4.1 -23.0 341.0   3.6  -1.2  -1.6  -1.0  -1.7   0.1   1.0   0.3   0.6   0.8   33006.   9.1  307.  -0.4   1.1 0.035  1.63    1953.   0.3    2.   0.4   0.4 0.003   0.52   3.46  11.0  3  55     3   33 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3   -13    21  5.5
2024   1  5 2596 51 52  61  37   4.3   4.1 -14.3 335.8   3.5  -1.5  -1.1  -1.4  -1.3   0.2   1.0   0.3   0.3   1.0   37437.   8.3  313.  -0.2   0.2 0.031  1.60    4297.   0.5    5.   0.2   0.4 0.005   0.77   3.07  11.2  3  55     2   22 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3   -12    15  5.8
2024   1  6 2596 51 52  56  35   3.7   3.6  -2.6 332.5   3.0  -1.7  -0.2  -1.7  -0.3   0.1   0.6   0.2   0.3   0.3   38710.   8.2  315.   0.2  -0.3 0.032  1.45    3218.   0.4    3.   0.1   0.2 0.001   0.14   4.05  13.6  3  55     0   17 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3   -11    11  6.0
2024   1  7 2596 51 52  56  35   3.9   3.7 -15.9 330.7   3.1  -1.7  -1.0  -1.5  -1.3   0.2   1.2   0.3   0.5   0.4   40476.   8.0  318.   0.4  -0.4 0.029  1.44    4010.   0.2    3.   0.2   0.6 0.002   0.49   3.57  12.8  3  55    -1   22 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3   -11    14  6.1
2024   1  8 2596 51 52  61  36   3.7   3.6  -2.1 328.3   3.1  -1.7   0.0  -1.7  -0.1   0.1   0.4   0.2   0.2   0.1   41736.   7.7  319.   0.2   0.3 0.024  1.39    4093.   0.3    1.   0.3   0.3 0.003   0.50   3.80  13.4  3  55    -2   14 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -9    11  6.2
2024   1  9 2596 51 52  60  37   3.3   3.2 -11.5 327.8   2.7  -1.4  -0.5  -1.4  -0.7   0.1   0.6   0.2   0.2   0.3   35991.   7.3  321.  -0.1   0.6 0.022  1.24    4174.   0.1    1.   0.2   0.4 0.001   0.39   4.52  15.3  3  55     2   18 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -6    12  6.6
2024   1 10 2596 51 52  57  37   3.1   3.0 -10.1 340.7   2.6  -1.2  -0.4  -1.1  -0.6   0.1   0.5   0.2   0.2   0.3   38127.   5.4  326.  -0.3   1.3 0.020  0.91    2556.   0.4    3.   0.1   0.2 0.001   0.19   4.13  15.3  3  55     2   25 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -5    17  5.8
2024   1 11 2596 51 52  60  36   3.4   3.4  19.1 352.0   2.9   0.9   0.9   1.2   0.6   0.2   1.1   0.1   0.4   0.5   46826.   3.8  352.   0.1   1.9 0.017  0.69    7282.   0.6    6.   0.2   0.3 0.005   0.78   1.87  12.7  3  55     3   23 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -5    16  5.3
2024   1 12 2596 51 52  60  37   3.1   2.7 -16.8 315.3   1.9  -1.3  -0.4  -1.3  -0.5   0.3   1.3   0.4   0.3   0.7   45474.   3.4  362.  -0.5   1.9 0.017  0.58    4997.   0.1    2.   0.1   0.3 0.001   0.52   1.55  14.0  3  55     2   29 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -3    20  5.3
2024   1 13 2596 51 52  62  37   3.2   2.7 -10.3 302.2   1.9  -1.3  -0.3  -1.3  -0.5   0.4   1.6   0.5   0.5   0.3   53736.   3.2  361.  -0.2   2.9 0.017  0.57    5316.   0.1    3.   0.1   0.4 0.002   0.55   1.40  14.0  3  55     3   30 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -3    19  5.4
2024   1 14 2596 51 52  61  37   3.7   3.2 -34.2 302.2   1.7  -1.2  -1.9  -0.8  -2.1   0.6   1.6   0.5   0.5   0.6   53929.   3.2  360.   0.2   1.5 0.019  0.58    3704.   0.2    3.   0.1   0.5 0.004   0.57   1.30  13.0  3  55     1   18 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -6    14  5.1
2024   1 15 2596 51 52  60  38   4.7   3.3  -3.5 314.1  -0.2  -2.3  -0.3  -2.3  -0.1   1.4   3.3   1.2   1.1   2.2   54755.   3.2  360.   0.2   1.7 0.019  0.59    5775.   0.2    3.   0.2   0.3 0.002   0.63   0.80  11.3  7  55     1   25 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -2    14  5.2
2024   1 16 2596 51 52  53  37   5.7   3.3  13.2 299.2  -0.3  -2.3   0.7  -2.4   0.4   2.1   4.1   1.9   1.8   1.5   66756.   3.5  362.  -0.5   1.9 0.020  0.66    8218.   0.3    6.   0.4   0.5 0.005   0.93   0.68  10.0  7  55    -1   36 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   3 131.2   0.3    -3    22  5.0
2024   1 17 2596 51 52  57  38   5.3   3.5 -13.1 274.3   0.2  -2.5  -0.8  -2.5  -0.6   1.7   3.8   1.7   1.3   1.4   60133.   3.2  369.  -1.8   1.7 0.018  0.55    4636.   0.1    3.   0.2   0.5 0.001   0.04   0.57  10.5  7  55     1   32 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -4    20  5.0
2024   1 18 2596 51 52  60  38   5.1   3.1   3.1 291.1  -0.3  -2.0   0.3  -2.0   0.1   1.9   3.9   2.0   0.7   0.5   63261.   3.7  369.  -0.9   2.1 0.019  0.63    7050.   0.3    3.   0.4   0.5 0.005   0.53   0.79  10.9  3  55     2   37 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -1    24  5.3
2024   1 19 2596 51 52  60  37   4.6   3.2  20.3 296.9   0.7  -2.0   1.1  -2.1   0.7   1.3   3.4   1.1   0.9   1.4   72474.   3.4  372.  -1.0   2.6 0.020  0.60    8651.   0.1    3.   0.2   0.5 0.003   0.34   0.84  11.6  3  55     2   34 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3    -1    22  5.3
2024   1 20 2596 51 52  60  38   4.3   3.7  22.2 305.6   2.0  -1.3   1.6  -1.5   1.3   0.5   2.4   0.6   0.4   2.2   71747.   3.5  370.  -1.1   1.7 0.018  0.60    9479.   0.2    2.   0.2   0.4 0.002   0.30   1.04  12.1  3  55     2   33 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0    22  5.4
2024   1 21 2596 51 52  62  38   4.5   3.3  -8.8 284.2   1.0  -2.0  -0.5  -2.0  -0.3   1.0   3.2   1.0   0.8   1.3   68137.   3.6  372.  -1.1   2.3 0.019  0.62    4805.   0.1    3.   0.3   0.5 0.002   0.27   0.87  11.7  7  55     1   31 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0    21  5.3
2024   1 22 2596 51 52  59  38   3.7   2.9  12.3 285.3   0.8  -1.5   0.6  -1.5   0.4   0.7   2.3   0.5   0.8   0.7   63717.   3.5  374.  -0.7   2.8 0.017  0.59    6696.   0.1    2.   0.1   0.5 0.001   0.23   1.13  12.7  3  55     0   23 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0    15  5.6
2024   1 23 2596 51 52  59  37   3.9   3.1 -27.5 273.3   1.0  -1.4  -1.4  -1.2  -1.5   0.7   2.4   0.6   0.7   0.5   65009.   3.5  380.  -2.0   2.1 0.017  0.57    5012.   0.1    3.   0.3   0.2 0.001   0.23   0.92  13.2  3  55    -2   19 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    13  5.7
2024   2  0 2596 51 52  58  37   4.0   3.4  -9.3 280.1   1.7  -1.7  -0.6  -1.7  -0.5   0.5   2.0   0.4   0.7   0.8   71028.   3.1  388.  -1.1   2.3 0.019  0.55    6429.   0.3    6.   0.2   0.6 0.001   0.40   0.72  13.3  3  55     1   18 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    12  5.5
2024   2  1 2596 51 52  60  37   4.2   3.1 -17.7 284.5   0.7  -1.6  -0.9  -1.6  -0.8   0.9   2.9   0.9   0.4   0.8   80135.   3.2  387.  -0.1   2.0 0.018  0.57    7414.   0.2    3.   0.2   0.7 0.001   0.32   0.72  12.8  3  55    -1   21 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0    14  5.5
2024   2  2 2596 51 52  59  37   3.1   2.5  17.8 263.9   1.2  -1.1   0.7  -1.1   0.5   0.6   2.0   0.4   0.6   0.6   81379.   3.2  395.  -0.5   2.1 0.016  0.56    4694.   0.1    2.   0.1   0.7 0.001   0.22   1.01  14.5  3  55     0   20 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0    13  5.9
2024   2  3 2596 51 52  59  38   3.5   2.6  17.2 257.3   0.8  -1.2   0.7  -1.2   0.5   0.9   2.5   0.9   0.4   0.6   85100.   3.2  395.  -0.3   1.4 0.018  0.56    5810.   0.2    5.   0.1   0.6 0.001   0.22   0.75  13.5  3  55    -1   12 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    10  5.7
2024   2  4 2596 51 52  60  38   3.5   2.1  18.3 254.3   0.3  -1.0   0.6  -1.0   0.5   1.3   2.8   1.0   0.6   0.7   81668.   3.2  397.  -0.1   1.1 0.015  0.55    5478.   0.2    3.   0.0   0.3 0.001   0.20   0.70  13.7  3  55    -1   15 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    10  5.8
2024   2  5 2596 51 52  61  38   3.5   2.7 -22.4 263.3   1.2  -1.1  -0.9  -1.0  -1.0   0.7   2.1   0.6   0.5   0.6   77662.   3.0  397.   0.0   1.5 0.015  0.52    8124.   0.1    3.   0.1   0.4 0.001   0.37   0.69  13.8  3  55    -2   14 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    10  5.7
2024   2  6 2596 51 52  60  38   3.3   2.6  -5.0 252.0   1.3  -1.1  -0.2  -1.2  -0.1   0.6   2.2   0.5   0.5   0.5   90820.   2.8  400.   0.2   1.3 0.012  0.47    5854.   0.1    3.   0.1   0.3 0.001   0.08   0.60  14.3  3  55    -2   13 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1     8  5.8
2024   2  7 2596 51 52  63  38   3.4   2.3  -7.1 248.2   0.4  -1.2  -0.3  -1.2  -0.2   1.1   2.5   1.0   0.5   0.5   89484.   3.0  392.   0.7   1.2 0.013  0.50    3999.   0.1    3.   0.1   0.4 0.001   0.07   0.72  14.0  3  55    -2   12 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1     9  5.8
2024   2  8 2596 51 52  55  37   3.8   2.2  -7.7 238.3   0.1  -1.3  -0.3  -1.3  -0.2   1.4   3.1   1.4   0.4   0.4  101696.   3.3  395.   0.1   1.2 0.014  0.56    4891.   0.2    2.   0.2   0.2 0.001   0.08   0.73  13.6  3  55    -1   18 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    12  5.7
2024   2  9 2596 51 52  61  38   3.5   2.4  16.8 249.2   0.7  -0.9   0.7  -0.9   0.6   1.1   2.6   1.1   0.4   0.4   87898.   3.8  392.   0.5   0.4 0.015  0.64    3625.   0.2    2.   0.1   0.4 0.000   0.13   1.05  14.1  3  55     0   11 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0     8  5.8
2024   2 10 2596 51 52  62  37   3.4   2.6 -38.2 249.3   0.8  -0.6  -1.5  -0.3  -1.6   0.8   2.5   0.5   0.8   0.8   93419.   3.7  391.   0.0   0.3 0.016  0.63    4791.   0.2    2.   0.2   0.3 0.001   0.09   1.04  14.0  3  55     0   13 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0    11  5.9
2024   2 11 2596 51 52  60  38   2.7   2.0 -25.7 237.3   0.6  -0.6  -0.8  -0.5  -0.9   0.6   1.9   0.5   0.5   0.7   89277.   3.5  394.   0.2   0.9 0.015  0.59    8004.   0.1    1.   0.1   0.3 0.001   0.22   1.20  15.1  3  55     0   11 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     0     8  6.0
2024   2 12 2596 51 52  63  38   3.9   3.0  -0.2 252.3   1.3  -1.4   0.0  -1.4   0.2   0.8   2.6   0.7   0.5   0.3  112697.   3.3  398.   0.3   1.6 0.018  0.59    2822.   0.2    3.   0.1   0.3 0.001   0.00   0.61  13.2  3  55     0   15 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    11  5.5
2024   2 13 2596 51 52  63  38   4.0   3.0 -13.5 257.5   1.3  -1.4  -0.9  -1.3  -1.0   0.9   2.6   0.8   0.4   0.3  108803.   3.1  402.   0.7   1.2 0.017  0.55    2899.   0.1    3.   0.1   0.3 0.001   0.13   0.54  13.2  3  55    -1   17 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    13  5.5
2024   2 14 2596 51 52  58  38   3.2   2.5  -3.2 262.3   1.0  -1.0  -0.1  -1.0  -0.1   0.7   2.1   0.5   0.5   0.3  100750.   3.1  398.   0.9   0.1 0.018  0.54    5432.   0.2    4.   0.4   0.5 0.002   0.44   0.74  14.3  3  55    -1   12 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1     8  5.9
2024   2 15 2596 51 52  62  39   3.3   2.7  -3.7 248.1   1.4  -1.1  -0.2  -1.1  -0.1   0.5   1.9   0.4   0.4   0.3   93655.   3.6  393.   0.2   0.3 0.016  0.61    5267.   0.2    2.   0.2   0.4 0.001   0.06   1.01  14.5  7  55    -1   10 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1     8  5.9
2024   2 16 2596 51 52  59  38   3.3   2.4  21.2 259.0   0.4  -0.9   0.8  -0.9   0.7   1.0   2.3   0.7   0.5   0.9   83261.   3.1  401.   0.3   0.8 0.015  0.54    3655.   0.1    3.   0.1   0.3 0.001   0.18   0.71  14.2  3  55    -1   16 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1    12  5.7
2024   2 17 2596 51 52  62  38   3.7   2.5  -4.2 247.5   0.3  -1.2  -0.2  -1.2  -0.1   1.1   2.6   1.0   0.4   0.2   80479.   3.3  397.   0.7   0.3 0.015  0.57    2994.   0.2    3.   0.3   0.2 0.001   0.03   0.71  13.6  3  55     0   13 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1     9  5.7
2024   2 18 2596 51 52  58  38   3.5   2.2  30.5 248.9  -0.2  -0.6   1.1  -0.8   0.9   1.3   2.8   1.2   0.4   0.5   78832.   3.7  393.   0.5  -0.3 0.017  0.63    2804.   0.2    3.   0.2   0.3 0.001   0.04   0.94  13.8  3  55     0   11 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     1     8  5.8
2024   2 19 2596 51 52  57  38   3.5   2.4 -16.2 249.9   0.6  -1.1  -0.6  -1.0  -0.7   1.0   2.5   0.8   0.4   0.6   93120.   3.2  405.  -0.3   1.4 0.015  0.56    5736.   0.2    3.   0.1   0.3 0.001   0.23   0.62  14.4  3  55    -1   17 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     2    11  5.6
2024   2 20 2596 51 52  60  38   3.1   2.3 -25.1 244.5   0.5  -0.6  -0.9  -0.4  -1.0   0.7   2.3   0.6   0.5   0.4   95620.   3.0  407.   0.5   1.4 0.017  0.53    4327.   0.1    2.   0.1   0.4 0.001   0.05   0.70  15.0  3  55    -1   13 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     2     8  5.9
2024   2 21 2596 51 52  60  38   3.3   2.4  27.0 268.1   0.3  -0.6   1.1  -0.8   0.9   0.8   2.4   0.7   0.6   0.8  102076.   2.6  418.  -0.3   1.2 0.012  0.45    5159.   0.1    3.   0.1   0.4 0.001   0.20   0.39  15.0  3  55    -1   14 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     2     9  5.7
2024   2 22 2596 51 52  59  38   2.8   2.3 -30.6 250.7   0.8  -0.4  -1.1  -0.2  -1.2   0.4   1.7   0.4   0.3   0.6  116879.   2.3  427.  -0.9   0.9 0.010  0.38    5665.   0.1    3.   0.1   0.3 0.001   0.12   0.34  15.8  3  55     0   10 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     2     8  6.0
2024   2 23 2596 51 52  61  38   3.1   2.4  27.2 272.2   0.3  -0.3   1.1  -0.4   1.0   0.7   2.3   0.5   0.5   0.7  119261.   2.4  425.  -1.3   0.5 0.012  0.41    6893.   0.1    2.   0.2   0.3 0.000   0.03   0.38  15.5  3  55    -1   14 999999.99 99999.99 99999.99 99999.99 99999.99 99999.99  0   2 131.2   0.3     2    10  5.9
";

fn select_soho_cadence(cli: &Cli) -> anyhow::Result<&'static str> {
    match cli.soho_celias_cadence.as_str() {
        "auto" => {
            if cli.time_resolution <= 900 {
                Ok("native")
            } else {
                Ok("hourly")
            }
        }
        "hourly" => Ok("hourly"),
        "native" => Ok("native"),
        other => anyhow::bail!(
            "invalid --soho-celias-cadence '{}'; expected auto, hourly, or native",
            other
        ),
    }
}

/// Physical-to-LBM unit conversion parameters.
struct UnitConversion {
    /// Reference proton density (cm^-3). Median of the selected window.
    n_ref: f64,
    /// Reference bulk speed (km/s). Median of the selected window.
    v_ref: f64,
    /// LBM velocity scale. u_lbm = (v / v_ref) * u_scale.
    u_scale: f64,
    /// Density clamp range in LBM units: [min, max].
    density_clamp: [f64; 2],
    /// Speed clamp range in LBM units: [min, max].
    speed_clamp: [f64; 2],
}

impl UnitConversion {
    fn from_omni(records: &[OmniRecord]) -> Self {
        let mut densities: Vec<f64> = records
            .iter()
            .map(|r| r.proton_density)
            .filter(|d| !d.is_nan())
            .collect();
        let mut speeds: Vec<f64> = records
            .iter()
            .map(|r| r.bulk_speed)
            .filter(|s| !s.is_nan())
            .collect();

        densities.sort_by(|a, b| a.partial_cmp(b).unwrap());
        speeds.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n_ref = if densities.is_empty() {
            5.0
        } else {
            densities[densities.len() / 2]
        };
        let v_ref = if speeds.is_empty() {
            400.0
        } else {
            speeds[speeds.len() / 2]
        };

        Self {
            n_ref,
            v_ref,
            u_scale: 0.05,
            density_clamp: [0.1, 10.0],
            speed_clamp: [0.001, 0.15],
        }
    }

    fn density_to_lbm(&self, n_p: f64) -> f64 {
        if n_p.is_nan() {
            1.0
        } else {
            (n_p / self.n_ref).clamp(self.density_clamp[0], self.density_clamp[1])
        }
    }

    fn speed_to_lbm(&self, v_sw: f64) -> f64 {
        if v_sw.is_nan() {
            self.u_scale
        } else {
            (v_sw / self.v_ref * self.u_scale).clamp(self.speed_clamp[0], self.speed_clamp[1])
        }
    }
}

/// Parse a "min,max" string into [f64; 2].
fn parse_clamp_range(s: &str) -> [f64; 2] {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() == 2 {
        let lo = parts[0].trim().parse::<f64>().unwrap_or(0.1);
        let hi = parts[1].trim().parse::<f64>().unwrap_or(10.0);
        [lo, hi]
    } else {
        [0.1, 10.0]
    }
}

/// A single grid cell's initial condition data.
struct CellIc {
    x: usize,
    y: usize,
    z: usize,
    rho: f64,
    u: [f64; 3],
    b: [f64; 3],
}

/// Parker spiral B-field fallback (used when no real B data available).
fn parker_spiral_b(
    x: usize,
    y: usize,
    nx: usize,
    ny: usize,
    b0: f64,
    omega: f64,
    v_sw_local: f64,
) -> [f64; 3] {
    let r0 = nx as f64 / 2.0;
    let y_center = ny as f64 / 2.0;
    let dx = (x as f64).max(0.5);
    let dy = y as f64 - y_center;
    let r_cyl = (dx * dx + dy * dy).sqrt().max(0.5);
    let b_r = b0 * (r0 / r_cyl).powi(2);
    let b_phi = -b0 * omega * r0 * r0 / (v_sw_local.max(1e-30) * r_cyl);
    let cos_phi = dx / r_cyl;
    let sin_phi = dy / r_cyl;
    [
        b_r * cos_phi - b_phi * sin_phi,
        b_r * sin_phi + b_phi * cos_phi,
        0.0,
    ]
}

/// Generate 3D IC from OMNI records with real B-field data.
///
/// Maps hourly OMNI measurements to x-slices via Taylor's hypothesis.
/// Uses real Bx/By/Bz (GSE) directly when available, falling back to
/// Parker spiral when individual B components have fill values.
/// If `lat_profile` is Some, applies latitude modulation along z-axis.
fn generate_ic_from_omni(
    records: &[OmniRecord],
    cli: &Cli,
    units: &UnitConversion,
    lat_profile: Option<&LatitudinalProfile>,
) -> Vec<CellIc> {
    let nx = cli.nx;
    let ny = cli.ny;
    let nz = cli.nz;
    let n_hours = records.len().max(1);

    let mut data = Vec::with_capacity(nx * ny * nz);

    for z in 0..nz {
        // Latitude modulation: adjust density and speed based on z-position
        let (lat_n_factor, lat_v_factor) = lat_profile
            .map(|p| latitude_modulation(z, nz, cli.lat_max_deg, p))
            .unwrap_or((1.0, 1.0));

        for y in 0..ny {
            for x in 0..nx {
                // Proportional mapping: distribute hours uniformly across x-cells.
                // Avoids tail bias from integer division (old: cells_per_hour = nx / n_hours).
                let hour_idx = (x * n_hours / nx).min(n_hours - 1);

                let rec = &records[hour_idx];
                let n_phys = rec.proton_density * lat_n_factor;
                let v_phys = rec.bulk_speed * lat_v_factor;
                let rho = units.density_to_lbm(n_phys);
                let v_lbm = units.speed_to_lbm(v_phys);
                let u = [v_lbm, 0.0, 0.0];

                // Use real B-field from OMNI (GSE coordinates map directly
                // to LBM axes: Bx=radial, By=ecliptic transverse, Bz=north).
                // Only fall back to Parker spiral when ALL three B
                // components are NaN. Preserve partial measurements
                // (e.g., Bn-only from a MAG-only record) by zeroing
                // only the missing components.
                let b = if !rec.bx_gse.is_nan() || !rec.by_gse.is_nan() || !rec.bz_gse.is_nan() {
                    let bx = if rec.bx_gse.is_nan() { 0.0 } else { rec.bx_gse };
                    let by = if rec.by_gse.is_nan() { 0.0 } else { rec.by_gse };
                    let bz = if rec.bz_gse.is_nan() { 0.0 } else { rec.bz_gse };
                    [bx * cli.b_scale, by * cli.b_scale, bz * cli.b_scale]
                } else {
                    parker_spiral_b(x, y, nx, ny, cli.b_scale * 5.0, cli.omega, v_lbm)
                };

                data.push(CellIc { x, y, z, rho, u, b });
            }
        }
    }

    data
}

/// Write the IC as a single 3D volume CSV.
fn write_volume(
    path: &std::path::Path,
    data: &[CellIc],
    units: &UnitConversion,
    kn_regime: KnudsenRegime,
    kn_max: f64,
) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
    // Metadata header: unit conversion parameters for downstream consumers.
    // Lines starting with '#' are skipped by load_ic_file() in solar-wind-dm-mhd.
    writeln!(file, "# n_ref_cm3={:.6}", units.n_ref)?;
    writeln!(file, "# v_ref_kms={:.6}", units.v_ref)?;
    writeln!(file, "# u_scale={:.6}", units.u_scale)?;
    writeln!(file, "# kn_regime={kn_regime:?}")?;
    writeln!(file, "# kn_max={kn_max:.2}")?;
    writeln!(file, "x,y,z,rho,ux,uy,uz,bx,by,bz")?;
    for c in data {
        writeln!(
            file,
            "{},{},{},{:.8},{:.8},{:.8},{:.8},{:.8e},{:.8e},{:.8e}",
            c.x, c.y, c.z, c.rho, c.u[0], c.u[1], c.u[2], c.b[0], c.b[1], c.b[2],
        )?;
    }
    Ok(())
}

/// Write the IC as one CSV per z-slice.
fn write_slices(dir: &std::path::Path, data: &[CellIc], nz: usize) -> std::io::Result<()> {
    fs::create_dir_all(dir)?;
    for z in 0..nz {
        let filename = dir.join(format!("slice_z{z:04}.csv"));
        let mut file = fs::File::create(&filename)?;
        writeln!(file, "x,y,rho,ux,uy,uz,bx,by,bz")?;
        for c in data {
            if c.z != z {
                continue;
            }
            writeln!(
                file,
                "{},{},{:.8},{:.8},{:.8},{:.8},{:.8e},{:.8e},{:.8e}",
                c.x, c.y, c.rho, c.u[0], c.u[1], c.u[2], c.b[0], c.b[1], c.b[2],
            )?;
        }
    }
    Ok(())
}

/// Filter OMNI records: keep rows where at least density or speed is valid.
fn filter_valid_omni(records: &[OmniRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .filter(|r| !r.proton_density.is_nan() || !r.bulk_speed.is_nan())
        .cloned()
        .collect()
}

/// Filter OMNI records for MAG-only near-Earth sources (ACE MAG, WIND MFI,
/// STEREO MAG). Accepts rows where at least one B-field component is valid,
/// even when plasma fields (density, speed) are NaN. This is a legacy path
/// for the OmniRecord adapter; new spacecraft integrations should implement
/// PlasmaBoundaryProvider where Option<f64> naturally expresses partial data.
fn filter_valid_omni_mag(records: &[OmniRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .filter(|r| {
            !r.proton_density.is_nan()
                || !r.bulk_speed.is_nan()
                || !r.bx_gse.is_nan()
                || !r.by_gse.is_nan()
                || !r.bz_gse.is_nan()
        })
        .cloned()
        .collect()
}

/// SWEPAM-to-OMNI adapter: convert SWEPAM records to OmniRecord with
/// NaN for B-field (triggers Parker spiral fallback).
fn swepam_to_omni(records: &[data_core::catalogs::solar_wind::SwepamRecord]) -> Vec<OmniRecord> {
    records
        .iter()
        .filter(|r| !r.proton_density.is_nan() || !r.bulk_speed.is_nan())
        .map(|r| OmniRecord {
            year: r.decimal_year as u16,
            doy: r.doy,
            hour: r.hour,
            b_magnitude: f64::NAN,
            bx_gse: f64::NAN,
            by_gse: f64::NAN,
            bz_gse: f64::NAN,
            proton_temperature: r.ion_temperature,
            proton_density: r.proton_density,
            bulk_speed: r.bulk_speed,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: 1.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
        })
        .collect()
}

/// Load WIND SWE (plasma) and/or MFI (B-field) data, merge if both present.
fn load_wind_data(cli: &Cli) -> anyhow::Result<Vec<OmniRecord>> {
    let mfi_records = if let Some(ref path) = cli.wind_mfi_file {
        eprintln!("loading WIND MFI from: {}", path.display());
        let raw = parse_wind_mfi_file(path)?;
        eprintln!("  {} MFI hourly records", raw.len());
        Some(raw)
    } else {
        None
    };

    let swe_records = if let Some(ref path) = cli.wind_swe_file {
        eprintln!("loading WIND SWE from: {}", path.display());
        let raw = parse_wind_swe_file(path)?;
        eprintln!("  {} SWE records (~92-sec cadence)", raw.len());
        Some(raw)
    } else {
        None
    };

    let omni = match (swe_records, mfi_records) {
        (Some(swe), Some(mfi)) => {
            eprintln!("merging WIND SWE + MFI (time-aligned hourly)");
            merge_wind_swe_mfi(&swe, &mfi)
        }
        (None, Some(mfi)) => {
            eprintln!("WIND MFI only (B-field, no plasma data)");
            wind_mfi_to_omni(&mfi)
        }
        (Some(swe), None) => {
            eprintln!("WIND SWE only (plasma, no B-field -> Parker spiral fallback)");
            // Convert SWE to OmniRecord with NaN B-field
            swe.iter()
                .filter(|r| !r.proton_density.is_nan() || !r.flow_speed.is_nan())
                .map(|r| OmniRecord {
                    year: r.year,
                    doy: r.decimal_doy as u16,
                    hour: ((r.decimal_doy.fract() * 24.0) as u8).min(23),
                    b_magnitude: f64::NAN,
                    bx_gse: f64::NAN,
                    by_gse: f64::NAN,
                    bz_gse: f64::NAN,
                    proton_temperature: r.temperature,
                    proton_density: r.proton_density,
                    bulk_speed: r.flow_speed,
                    flow_pressure: f64::NAN,
                    plasma_beta: f64::NAN,
                    alfven_mach: f64::NAN,
                    dst_index: f64::NAN,
                    ae_index: f64::NAN,
                    kp_times_10: 0,
                    r_au: 1.0,
                    lat_deg: f64::NAN,
                    lon_deg: f64::NAN,
                })
                .collect()
        }
        (None, None) => unreachable!("caller ensures at least one WIND file"),
    };

    Ok(filter_valid_omni(&omni))
}

/// Load STEREO-A PLASTIC (plasma) and/or IMPACT/MAG (B-field) data with
/// RTN -> GSE coordinate transform.
fn load_stereo_data(cli: &Cli) -> anyhow::Result<Vec<OmniRecord>> {
    let plastic = if let Some(ref path) = cli.stereo_file {
        eprintln!("loading STEREO-A PLASTIC from: {}", path.display());
        let raw = parse_stereo_plastic_file(path)?;
        eprintln!("  {} PLASTIC records", raw.len());
        Some(raw)
    } else {
        None
    };

    let mag = if let Some(ref path) = cli.stereo_mag_file {
        eprintln!("loading STEREO-A IMPACT/MAG from: {}", path.display());
        let raw = parse_stereo_magplasma_file(path)?;
        let hourly = average_stereo_mag_hourly(&raw);
        eprintln!(
            "  {} MAG records -> {} hourly averages",
            raw.len(),
            hourly.len()
        );
        Some(hourly)
    } else {
        None
    };

    eprintln!("STEREO-A separation angle: {:.1} deg", cli.stereo_sep_deg);

    let omni = stereo_to_omni(
        plastic.as_deref().unwrap_or(&[]),
        mag.as_deref().unwrap_or(&[]),
        cli.stereo_sep_deg,
    );

    Ok(filter_valid_omni(&omni))
}

/// Compute an ordinal time key for temporal alignment.
///
/// Returns (year * 366 + doy) * 24 + hour, giving a monotonic integer
/// suitable for set intersection and binary search.
fn time_key(r: &OmniRecord) -> u32 {
    (r.year as u32 * 366 + r.doy as u32) * 24 + r.hour as u32
}

/// Compute the temporal intersection of two OmniRecord slices.
///
/// Returns `(l1_aligned, stereo_aligned)` where both slices cover the
/// same time window. Records are matched by (year, doy, hour) keys.
/// Records present in one dataset but not the other are dropped.
///
/// Panics if the intersection is empty.
fn time_aligned_intersection<'a>(
    l1: &'a [OmniRecord],
    stereo: &'a [OmniRecord],
) -> (Vec<&'a OmniRecord>, Vec<&'a OmniRecord>) {
    use std::collections::BTreeMap;

    // Index both datasets by time key
    let mut l1_by_key: BTreeMap<u32, &OmniRecord> = BTreeMap::new();
    for r in l1 {
        l1_by_key.insert(time_key(r), r);
    }

    let mut stereo_by_key: BTreeMap<u32, &OmniRecord> = BTreeMap::new();
    for r in stereo {
        stereo_by_key.insert(time_key(r), r);
    }

    // Intersect: only keep keys present in both
    let mut l1_aligned = Vec::new();
    let mut st_aligned = Vec::new();
    for (&key, &l1_rec) in &l1_by_key {
        if let Some(&st_rec) = stereo_by_key.get(&key) {
            l1_aligned.push(l1_rec);
            st_aligned.push(st_rec);
        }
    }

    assert!(
        !l1_aligned.is_empty(),
        "No overlapping timestamps between L1 and STEREO datasets. \
         L1 range: {}-{}, STEREO range: {}-{}",
        l1.first().map_or(0, time_key),
        l1.last().map_or(0, time_key),
        stereo.first().map_or(0, time_key),
        stereo.last().map_or(0, time_key),
    );

    eprintln!(
        "  time-aligned intersection: {} hours (L1: {}, STEREO: {})",
        l1_aligned.len(),
        l1.len(),
        stereo.len(),
    );

    (l1_aligned, st_aligned)
}

/// Generate 3D IC with L1+STEREO-A longitudinal triangulation.
///
/// Y-axis maps to heliocentric longitude offset between L1 and STEREO-A:
///   y=0: pure L1 data, y=ny-1: pure STEREO-A data, intermediate: lerp.
/// Z-axis remains uniform (no out-of-ecliptic spacecraft available).
///
/// Time-aligned: only the temporal intersection of L1 and STEREO is used,
/// ensuring both datasets map the same physical time window to x-slices.
fn triangulate_ic_from_multi_spacecraft(
    l1_records: &[OmniRecord],
    stereo_records: &[OmniRecord],
    cli: &Cli,
    units: &UnitConversion,
) -> Vec<CellIc> {
    let nx = cli.nx;
    let ny = cli.ny;
    let nz = cli.nz;

    // Align datasets to their temporal intersection
    let (l1_aligned, st_aligned) = time_aligned_intersection(l1_records, stereo_records);
    let n_records = l1_aligned.len();

    // Parker spiral longitude weighting.
    //
    // At a fixed heliocentric radius (1 AU), the Parker spiral arc
    // length between two longitudes is proportional to the angular
    // separation: s(phi) = phi * r * sqrt(1 + (omega*r/v_sw)^2).
    // The sqrt factor is constant at fixed r, so it cancels in the
    // ratio s(phi)/s(sep), giving alpha = phi/sep = y/(ny-1).
    //
    // This means linear interpolation in y IS the physically correct
    // Parker spiral weighting at 1 AU. The spiral angle affects the
    // B-field direction (handled by RTN->GSE in stereo_plastic.rs),
    // not the interpolation weight between spacecraft.
    //
    // The separation angle determines the physical longitude span:
    //   phi(y) = (y / (ny-1)) * sep_deg
    // This is used for diagnostic output only; the weight is y/(ny-1).
    let sep_rad = cli.stereo_sep_deg.to_radians();
    let v_sw_phys = units.v_ref * 1.0e3; // km/s -> m/s
    let psi_spiral = cli.omega * 1.496e11 / v_sw_phys; // spiral angle at 1 AU (rad)
    eprintln!(
        "  Parker spiral angle at 1 AU: {:.1} deg (omega={:.3e} rad/s, v_sw={:.0} km/s)",
        psi_spiral.to_degrees(),
        cli.omega,
        units.v_ref,
    );
    eprintln!(
        "  longitude span: {:.1} deg, {ny} y-slices ({:.2} deg/slice)",
        cli.stereo_sep_deg,
        cli.stereo_sep_deg / ny.max(1) as f64,
    );
    // sep_rad is used in the B-field rotation below

    let mut data = Vec::with_capacity(nx * ny * nz);

    for z in 0..nz {
        for y in 0..ny {
            // Parker-weighted longitude interpolation.
            //
            // At fixed heliocentric radius (1 AU), the Parker spiral
            // winding angle psi = atan(omega * R / v_sw) is constant
            // across the grid, so the weight simplifies to y/(ny-1).
            //
            // For radial mode (Phase 10), each x-slice maps to a
            // different heliocentric distance R(x), making psi vary
            // with x. The generalized weight becomes:
            //   alpha(y,x) = y/(ny-1) * psi(R(x)) / psi_max
            // This hook is ready for Phase 10 integration.
            let alpha_base = if ny > 1 {
                y as f64 / (ny - 1) as f64
            } else {
                0.0
            };
            // At 1 AU fixed-r, psi_ratio = 1.0 (no radial variation).
            // Phase 10 will compute psi_ratio per x-slice.
            let alpha = alpha_base;

            for x in 0..nx {
                // Map x-slice to aligned record index (same for both datasets)
                let rec_idx = (x * n_records / nx).min(n_records - 1);
                let l1 = l1_aligned[rec_idx];
                let st = st_aligned[rec_idx];

                // Interpolate density (physical, then convert)
                let n_l1 = if l1.proton_density.is_nan() {
                    units.n_ref
                } else {
                    l1.proton_density
                };
                let n_st = if st.proton_density.is_nan() {
                    units.n_ref
                } else {
                    st.proton_density
                };
                let n_interp = n_l1 * (1.0 - alpha) + n_st * alpha;
                let rho = units.density_to_lbm(n_interp);

                // Interpolate speed
                let v_l1 = if l1.bulk_speed.is_nan() {
                    units.v_ref
                } else {
                    l1.bulk_speed
                };
                let v_st = if st.bulk_speed.is_nan() {
                    units.v_ref
                } else {
                    st.bulk_speed
                };
                let v_interp = v_l1 * (1.0 - alpha) + v_st * alpha;
                let v_lbm = units.speed_to_lbm(v_interp);
                let u = [v_lbm, 0.0, 0.0];

                // Interpolate B-field with Parker spiral direction rotation.
                //
                // Instead of LERP'ing raw GSE components (which mixes
                // radial and tangential directions incorrectly), we:
                // 1. Compute Parker spiral angle at each spacecraft longitude
                // 2. Rotate STEREO B-field by the longitude difference
                // 3. LERP the rotated components
                //
                // At 1 AU: psi = atan(omega * R / v_sw) is the same for both
                // spacecraft (same R). The rotation accounts for the
                // longitude offset between L1 (phi=0) and STEREO (phi=sep).
                //
                // When only one side has B data, use it directly (no
                // interpolation against zero which would dilute the field).
                let l1_has_b = !l1.bx_gse.is_nan() || !l1.by_gse.is_nan() || !l1.bz_gse.is_nan();
                let st_has_b = !st.bx_gse.is_nan() || !st.by_gse.is_nan() || !st.bz_gse.is_nan();

                let b = if l1_has_b && st_has_b {
                    // Both sides have B: interpolate with rotation
                    let bx_l1 = if l1.bx_gse.is_nan() { 0.0 } else { l1.bx_gse };
                    let by_l1 = if l1.by_gse.is_nan() { 0.0 } else { l1.by_gse };
                    let bz_l1 = if l1.bz_gse.is_nan() { 0.0 } else { l1.bz_gse };

                    let bx_st = if st.bx_gse.is_nan() { 0.0 } else { st.bx_gse };
                    let by_st = if st.by_gse.is_nan() { 0.0 } else { st.by_gse };
                    let bz_st = if st.bz_gse.is_nan() { 0.0 } else { st.bz_gse };

                    let delta_phi = sep_rad * alpha;
                    let cos_dp = delta_phi.cos();
                    let sin_dp = delta_phi.sin();

                    let bx_st_rot = bx_st * cos_dp - by_st * sin_dp;
                    let by_st_rot = bx_st * sin_dp + by_st * cos_dp;

                    [
                        (bx_l1 * (1.0 - alpha) + bx_st_rot * alpha) * cli.b_scale,
                        (by_l1 * (1.0 - alpha) + by_st_rot * alpha) * cli.b_scale,
                        (bz_l1 * (1.0 - alpha) + bz_st * alpha) * cli.b_scale,
                    ]
                } else if l1_has_b {
                    // Only L1 has B: use directly without dilution
                    let bx = if l1.bx_gse.is_nan() { 0.0 } else { l1.bx_gse };
                    let by = if l1.by_gse.is_nan() { 0.0 } else { l1.by_gse };
                    let bz = if l1.bz_gse.is_nan() { 0.0 } else { l1.bz_gse };
                    [bx * cli.b_scale, by * cli.b_scale, bz * cli.b_scale]
                } else if st_has_b {
                    // Only STEREO has B: rotate to interpolated frame, use directly
                    let bx_st = if st.bx_gse.is_nan() { 0.0 } else { st.bx_gse };
                    let by_st = if st.by_gse.is_nan() { 0.0 } else { st.by_gse };
                    let bz_st = if st.bz_gse.is_nan() { 0.0 } else { st.bz_gse };

                    let delta_phi = sep_rad * alpha;
                    let cos_dp = delta_phi.cos();
                    let sin_dp = delta_phi.sin();

                    let bx_rot = bx_st * cos_dp - by_st * sin_dp;
                    let by_rot = bx_st * sin_dp + by_st * cos_dp;
                    [
                        bx_rot * cli.b_scale,
                        by_rot * cli.b_scale,
                        bz_st * cli.b_scale,
                    ]
                } else {
                    parker_spiral_b(x, y, nx, ny, cli.b_scale * 5.0, cli.omega, v_lbm)
                };

                data.push(CellIc { x, y, z, rho, u, b });
            }
        }
    }

    data
}

/// A single spacecraft measurement at a known heliocentric distance,
/// used for building radial profiles across the heliosphere.
#[derive(Clone, Debug)]
struct RadialProfilePoint {
    /// Heliocentric distance (AU).
    r_au: f64,
    /// Proton number density (cm^-3).
    density_cm3: f64,
    /// Bulk flow speed (km/s).
    speed_kms: f64,
    /// Proton temperature (K).
    temp_k: f64,
    /// Radial B-field component (nT, GSE X).
    br_nt: f64,
    /// Tangential B-field component (nT, GSE Y).
    bt_nt: f64,
    /// Normal B-field component (nT, GSE Z).
    bn_nt: f64,
    /// Total B-field magnitude (nT).
    b_mag_nt: f64,
}

struct RadialFitRow {
    quantity: &'static str,
    expected_slope: f64,
    fitted_slope: f64,
    sample_count: usize,
}

/// Build a radial profile from multi-spacecraft OmniRecords.
///
/// Groups records by heliocentric distance (r_au), computes the median
/// plasma parameters at each distance bin, and returns sorted points.
/// Records without a valid r_au (NaN) are skipped.
fn build_radial_profile(records: &[OmniRecord]) -> Vec<RadialProfilePoint> {
    use std::collections::BTreeMap;

    // Bin records by distance: round to 0.1 AU granularity to avoid
    // excessive fragmentation from orbital variation within a dataset.
    let mut bins: BTreeMap<i32, Vec<&OmniRecord>> = BTreeMap::new();
    for r in records {
        if r.r_au.is_nan() || r.r_au <= 0.0 {
            continue;
        }
        // Key: distance in units of 0.1 AU (e.g., 1.0 AU -> 10, 84.5 AU -> 845)
        let key = (r.r_au * 10.0).round() as i32;
        bins.entry(key).or_default().push(r);
    }

    bins.into_iter()
        .map(|(key, recs)| {
            let r_au = key as f64 / 10.0;
            RadialProfilePoint {
                r_au,
                density_cm3: median_finite(recs.iter().map(|r| r.proton_density)),
                speed_kms: median_finite(recs.iter().map(|r| r.bulk_speed)),
                temp_k: median_finite(recs.iter().map(|r| r.proton_temperature)),
                br_nt: median_finite(recs.iter().map(|r| r.bx_gse)),
                bt_nt: median_finite(recs.iter().map(|r| r.by_gse)),
                bn_nt: median_finite(recs.iter().map(|r| r.bz_gse)),
                b_mag_nt: median_finite(recs.iter().map(|r| r.b_magnitude)),
            }
        })
        .collect()
}

fn sample_radial_profile(
    profile: &[RadialProfilePoint],
    r_min: f64,
    r_max: f64,
    sample_count: usize,
) -> Vec<RadialProfilePoint> {
    if sample_count <= 1 {
        return vec![interpolate_radial(profile, r_min)];
    }

    let ln_ratio = (r_max / r_min).ln();
    (0..sample_count)
        .map(|i| {
            let t = i as f64 / (sample_count - 1) as f64;
            let r_au = r_min * (t * ln_ratio).exp();
            interpolate_radial(profile, r_au)
        })
        .collect()
}

fn fit_power_law(
    rows: &[RadialProfilePoint],
    quantity: impl Fn(&RadialProfilePoint) -> f64,
) -> Option<(f64, usize)> {
    let samples: Vec<(f64, f64)> = rows
        .iter()
        .filter_map(|row| {
            let value = quantity(row);
            if row.r_au.is_finite() && row.r_au > 0.0 && value.is_finite() && value.abs() > 1e-30 {
                Some((row.r_au.ln(), value.abs().ln()))
            } else {
                None
            }
        })
        .collect();
    let n = samples.len();
    if n < 2 {
        return None;
    }
    let mean_x = samples.iter().map(|(x, _)| x).sum::<f64>() / n as f64;
    let mean_y = samples.iter().map(|(_, y)| y).sum::<f64>() / n as f64;
    let cov_xy = samples
        .iter()
        .map(|(x, y)| (x - mean_x) * (y - mean_y))
        .sum::<f64>();
    let var_x = samples
        .iter()
        .map(|(x, _)| (x - mean_x).powi(2))
        .sum::<f64>();
    if var_x.abs() < 1e-30 {
        return None;
    }
    Some((cov_xy / var_x, n))
}

fn write_radial_profile_csv(path: &PathBuf, rows: &[RadialProfilePoint]) -> anyhow::Result<()> {
    let mut out = String::from("r_au,density_cm3,speed_kms,temp_k,br_nt,bt_nt,bn_nt,b_mag_nt\n");
    for row in rows {
        out.push_str(&format!(
            "{:.6},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e},{:.6e}\n",
            row.r_au,
            row.density_cm3,
            row.speed_kms,
            row.temp_k,
            row.br_nt,
            row.bt_nt,
            row.bn_nt,
            row.b_mag_nt
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_radial_fit_csv(path: &PathBuf, rows: &[RadialFitRow]) -> anyhow::Result<()> {
    let mut out = String::from("quantity,expected_slope,fitted_slope,sample_count\n");
    for row in rows {
        out.push_str(&format!(
            "{},{:.6},{:.6},{}\n",
            row.quantity, row.expected_slope, row.fitted_slope, row.sample_count
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

fn write_radial_artifacts(
    cli: &Cli,
    measured: &[RadialProfilePoint],
    sampled: &[RadialProfilePoint],
) -> anyhow::Result<()> {
    if let Some(ref path) = cli.radial_profile_out {
        write_radial_profile_csv(path, measured)?;
        eprintln!("wrote radial profile bins: {}", path.display());
    }
    if let Some(ref path) = cli.radial_sample_out {
        write_radial_profile_csv(path, sampled)?;
        eprintln!("wrote radial profile samples: {}", path.display());
    }
    if let Some(ref path) = cli.radial_fit_out {
        let mut fits = Vec::new();
        if let Some((slope, n)) = fit_power_law(sampled, |row| row.density_cm3) {
            fits.push(RadialFitRow {
                quantity: "density_cm3",
                expected_slope: -2.0,
                fitted_slope: slope,
                sample_count: n,
            });
        }
        if let Some((slope, n)) = fit_power_law(sampled, |row| row.br_nt) {
            fits.push(RadialFitRow {
                quantity: "br_nt",
                expected_slope: -2.0,
                fitted_slope: slope,
                sample_count: n,
            });
        }
        if let Some((slope, n)) = fit_power_law(sampled, |row| row.bt_nt) {
            fits.push(RadialFitRow {
                quantity: "bt_nt",
                expected_slope: -1.0,
                fitted_slope: slope,
                sample_count: n,
            });
        }
        if let Some((slope, n)) = fit_power_law(sampled, |row| row.speed_kms) {
            fits.push(RadialFitRow {
                quantity: "speed_kms",
                expected_slope: 0.0,
                fitted_slope: slope,
                sample_count: n,
            });
        }
        write_radial_fit_csv(path, &fits)?;
        eprintln!("wrote radial fit diagnostics: {}", path.display());
    }
    Ok(())
}

/// Compute median of finite (non-NaN) values. Returns NaN if empty.
fn median_finite(iter: impl Iterator<Item = f64>) -> f64 {
    let mut vals: Vec<f64> = iter.filter(|v| v.is_finite()).collect();
    if vals.is_empty() {
        return f64::NAN;
    }
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
    vals[vals.len() / 2]
}

/// Interpolate radial profile to an arbitrary heliocentric distance.
///
/// Uses log-linear interpolation for density and B-field (which scale as
/// power laws in r), and linear interpolation for speed (approximately
/// constant). Extrapolation beyond the data range uses scaling laws:
///   n(r) ~ r^-2, B_r ~ r^-2, B_phi ~ r^-1, T ~ r^(-2/3), V ~ const.
fn interpolate_radial(profile: &[RadialProfilePoint], r_au: f64) -> RadialProfilePoint {
    if profile.is_empty() {
        return RadialProfilePoint {
            r_au,
            density_cm3: 5.0 / (r_au * r_au),
            speed_kms: 400.0,
            temp_k: 1e5 / r_au.powf(2.0 / 3.0),
            br_nt: 5.0 / (r_au * r_au),
            bt_nt: -3.0 / r_au,
            bn_nt: 0.0,
            b_mag_nt: (25.0 / (r_au * r_au * r_au * r_au) + 9.0 / (r_au * r_au)).sqrt(),
        };
    }

    if profile.len() == 1 {
        let p = &profile[0];
        let ratio = p.r_au / r_au;
        return RadialProfilePoint {
            r_au,
            density_cm3: scale_r2(p.density_cm3, ratio),
            speed_kms: p.speed_kms,
            temp_k: scale_temp(p.temp_k, ratio),
            br_nt: scale_r2(p.br_nt, ratio),
            bt_nt: scale_r1(p.bt_nt, ratio),
            bn_nt: scale_r1(p.bn_nt, ratio),
            b_mag_nt: (scale_r2(p.br_nt, ratio).powi(2)
                + scale_r1(p.bt_nt, ratio).powi(2)
                + scale_r1(p.bn_nt, ratio).powi(2))
            .sqrt(),
        };
    }

    // Find bracketing points for log-linear interpolation
    let ln_r = r_au.ln();

    // Extrapolate below minimum distance
    if r_au <= profile[0].r_au {
        let p = &profile[0];
        let ratio = p.r_au / r_au;
        return RadialProfilePoint {
            r_au,
            density_cm3: scale_r2(p.density_cm3, ratio),
            speed_kms: p.speed_kms,
            temp_k: scale_temp(p.temp_k, ratio),
            br_nt: scale_r2(p.br_nt, ratio),
            bt_nt: scale_r1(p.bt_nt, ratio),
            bn_nt: scale_r1(p.bn_nt, ratio),
            b_mag_nt: (scale_r2(p.br_nt, ratio).powi(2)
                + scale_r1(p.bt_nt, ratio).powi(2)
                + scale_r1(p.bn_nt, ratio).powi(2))
            .sqrt(),
        };
    }

    // Extrapolate above maximum distance
    if r_au >= profile[profile.len() - 1].r_au {
        let p = &profile[profile.len() - 1];
        let ratio = p.r_au / r_au;
        return RadialProfilePoint {
            r_au,
            density_cm3: scale_r2(p.density_cm3, ratio),
            speed_kms: p.speed_kms,
            temp_k: scale_temp(p.temp_k, ratio),
            br_nt: scale_r2(p.br_nt, ratio),
            bt_nt: scale_r1(p.bt_nt, ratio),
            bn_nt: scale_r1(p.bn_nt, ratio),
            b_mag_nt: (scale_r2(p.br_nt, ratio).powi(2)
                + scale_r1(p.bt_nt, ratio).powi(2)
                + scale_r1(p.bn_nt, ratio).powi(2))
            .sqrt(),
        };
    }

    // Find bracket: profile[i].r_au <= r_au < profile[i+1].r_au
    let i = profile
        .iter()
        .position(|p| p.r_au > r_au)
        .unwrap_or(profile.len() - 1)
        .saturating_sub(1);
    let p0 = &profile[i];
    let p1 = &profile[i + 1];

    // Log-linear weight: t in [0, 1] based on log(r)
    let ln_r0 = p0.r_au.ln();
    let ln_r1 = p1.r_au.ln();
    let t = if (ln_r1 - ln_r0).abs() < 1e-30 {
        0.5
    } else {
        (ln_r - ln_r0) / (ln_r1 - ln_r0)
    };

    // Density: interpolate log(n) vs log(r) (power law n ~ r^alpha)
    let density = interp_log(p0.density_cm3, p1.density_cm3, t);
    // Speed: linear interpolation (approximately constant)
    let speed = lerp(p0.speed_kms, p1.speed_kms, t);
    // Temperature: interpolate log(T) vs log(r) (power law T ~ r^beta)
    let temp = interp_log(p0.temp_k, p1.temp_k, t);
    // B_r: interpolate log(|B_r|) vs log(r) preserving sign
    let br = interp_log_signed(p0.br_nt, p1.br_nt, t);
    // B_t, B_n: log-linear with sign preservation
    let bt = interp_log_signed(p0.bt_nt, p1.bt_nt, t);
    let bn = interp_log_signed(p0.bn_nt, p1.bn_nt, t);

    RadialProfilePoint {
        r_au,
        density_cm3: density,
        speed_kms: speed,
        temp_k: temp,
        br_nt: br,
        bt_nt: bt,
        bn_nt: bn,
        b_mag_nt: (br * br + bt * bt + bn * bn).sqrt(),
    }
}

/// Scale a quantity that follows r^-2 law (density, B_r).
/// ratio = r_ref / r_target.
fn scale_r2(val: f64, ratio: f64) -> f64 {
    if val.is_nan() {
        return f64::NAN;
    }
    val * ratio * ratio
}

/// Scale a quantity that follows r^-1 law (B_phi, B_n).
fn scale_r1(val: f64, ratio: f64) -> f64 {
    if val.is_nan() {
        return f64::NAN;
    }
    val * ratio
}

/// Scale temperature: T ~ r^(-2/3) for adiabatic expansion (gamma=5/3).
fn scale_temp(val: f64, ratio: f64) -> f64 {
    if val.is_nan() {
        return f64::NAN;
    }
    val * ratio.powf(2.0 / 3.0)
}

/// Linear interpolation.
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    if a.is_nan() {
        return b;
    }
    if b.is_nan() {
        return a;
    }
    a * (1.0 - t) + b * t
}

/// Interpolate in log space (for power-law quantities like density).
fn interp_log(a: f64, b: f64, t: f64) -> f64 {
    if a.is_nan() || a <= 0.0 {
        return b;
    }
    if b.is_nan() || b <= 0.0 {
        return a;
    }
    (a.ln() * (1.0 - t) + b.ln() * t).exp()
}

/// Interpolate in log space with sign preservation (for signed B-field).
fn interp_log_signed(a: f64, b: f64, t: f64) -> f64 {
    if a.is_nan() {
        return b;
    }
    if b.is_nan() {
        return a;
    }
    // If both have the same sign, interpolate magnitudes in log space
    if a.signum() == b.signum() && a.abs() > 1e-30 && b.abs() > 1e-30 {
        a.signum() * interp_log(a.abs(), b.abs(), t)
    } else {
        // Mixed signs or near-zero: fall back to linear
        lerp(a, b, t)
    }
}

/// Distance-adaptive UnitConversion for radial IC generation.
///
/// At large heliocentric distances, n_ref and v_ref must adapt to avoid
/// LBM Mach number violations (Ma must stay < 0.3). Each x-slice gets
/// its own effective n_ref based on the local density from the profile.
fn radial_unit_conversion(
    profile: &[RadialProfilePoint],
    r_min: f64,
    r_max: f64,
) -> UnitConversion {
    // Use median density and speed across the full radial range.
    // The per-cell density mapping (density_to_lbm) normalizes by n_ref,
    // so n_ref should represent the "typical" density in the domain.
    // For a radial profile spanning 1-100 AU, the geometric mean
    // (log-midpoint) gives a balanced normalization.
    let r_mid = (r_min * r_max).sqrt();
    let mid_point = interpolate_radial(profile, r_mid);

    let n_ref = if mid_point.density_cm3.is_finite() && mid_point.density_cm3 > 0.0 {
        mid_point.density_cm3
    } else {
        5.0 / (r_mid * r_mid) // fallback: 5 cm^-3 at 1 AU, scaled
    };

    let v_ref = if mid_point.speed_kms.is_finite() && mid_point.speed_kms > 0.0 {
        mid_point.speed_kms
    } else {
        400.0
    };

    // Widen density clamps for the large dynamic range across the heliosphere.
    // At 100 AU, density is ~10000x smaller than at 1 AU.
    let density_ratio = (r_max / r_min).powi(2);
    let clamp_hi = (density_ratio * 2.0).min(1e6);

    UnitConversion {
        n_ref,
        v_ref,
        u_scale: 0.05,
        density_clamp: [1.0 / clamp_hi, clamp_hi],
        speed_clamp: [0.001, 0.25],
    }
}

/// Generate 3D IC from a radial profile spanning r_min to r_max AU.
///
/// X-axis maps to log-spaced heliocentric distance:
///   x=0 -> r_min AU, x=nx-1 -> r_max AU.
/// Y and Z axes are uniform (no longitudinal or latitudinal variation
/// in this mode; Phase 11 adds latitude via --latitudinal).
///
/// Each x-slice gets plasma parameters interpolated from the radial
/// profile, with proper Parker spiral B-field scaling.
fn generate_radial_ic(
    profile: &[RadialProfilePoint],
    cli: &Cli,
    units: &UnitConversion,
    lat_profile: Option<&LatitudinalProfile>,
) -> Vec<CellIc> {
    let nx = cli.nx;
    let ny = cli.ny;
    let nz = cli.nz;
    let r_min = cli.r_min_au;
    let r_max = cli.r_max_au;

    // Log-spaced distance grid: r(x) = r_min * (r_max/r_min)^(x/(nx-1))
    let ln_ratio = (r_max / r_min).ln();

    let mut data = Vec::with_capacity(nx * ny * nz);

    for z in 0..nz {
        let (lat_n_factor, lat_v_factor) = lat_profile
            .map(|p| latitude_modulation(z, nz, cli.lat_max_deg, p))
            .unwrap_or((1.0, 1.0));

        for y in 0..ny {
            for x in 0..nx {
                let frac = if nx > 1 {
                    x as f64 / (nx - 1) as f64
                } else {
                    0.5
                };
                let r_au = r_min * (frac * ln_ratio).exp();

                let point = interpolate_radial(profile, r_au);

                let rho = units.density_to_lbm(point.density_cm3 * lat_n_factor);
                let v_lbm = units.speed_to_lbm(point.speed_kms * lat_v_factor);
                let u = [v_lbm, 0.0, 0.0];

                // B-field from interpolated profile (already scaled by
                // distance). If all B components are NaN, fall back to
                // Parker spiral model.
                let b = if point.br_nt.is_finite()
                    || point.bt_nt.is_finite()
                    || point.bn_nt.is_finite()
                {
                    let br = if point.br_nt.is_finite() {
                        point.br_nt
                    } else {
                        0.0
                    };
                    let bt = if point.bt_nt.is_finite() {
                        point.bt_nt
                    } else {
                        0.0
                    };
                    let bn = if point.bn_nt.is_finite() {
                        point.bn_nt
                    } else {
                        0.0
                    };
                    [br * cli.b_scale, bt * cli.b_scale, bn * cli.b_scale]
                } else {
                    // Parker spiral fallback at this heliocentric distance
                    let b0 = 5.0 / (r_au * r_au); // B_r ~ r^-2
                    parker_spiral_b(x, y, nx, ny, b0 * cli.b_scale, cli.omega, v_lbm)
                };

                data.push(CellIc { x, y, z, rho, u, b });
            }
        }
    }

    data
}

/// Load outer heliosphere spacecraft data and merge into a single
/// Vec<OmniRecord> for radial profile construction.
fn load_outer_heliosphere(cli: &Cli) -> anyhow::Result<Vec<OmniRecord>> {
    let mut all_records: Vec<OmniRecord> = Vec::new();

    if let Some(ref path) = cli.voyager1_file {
        eprintln!("loading Voyager 1 from: {}", path.display());
        let raw = parse_voyager_file(path, VoyagerSpacecraft::V1)?;
        let omni = voyager_to_omni(&raw);
        eprintln!("  {} Voyager 1 records", omni.len());
        all_records.extend(omni);
    }

    if let Some(ref path) = cli.voyager2_file {
        eprintln!("loading Voyager 2 from: {}", path.display());
        let raw = parse_voyager_file(path, VoyagerSpacecraft::V2)?;
        let omni = voyager_to_omni(&raw);
        eprintln!("  {} Voyager 2 records", omni.len());
        all_records.extend(omni);
    }

    if let Some(ref path) = cli.pioneer10_file {
        eprintln!("loading Pioneer 10 from: {}", path.display());
        let raw = parse_pioneer_file(path, PioneerSpacecraft::P10)?;
        let omni = pioneer_to_omni(&raw);
        eprintln!("  {} Pioneer 10 records", omni.len());
        all_records.extend(omni);
    }

    if let Some(ref path) = cli.pioneer11_file {
        eprintln!("loading Pioneer 11 from: {}", path.display());
        let raw = parse_pioneer_file(path, PioneerSpacecraft::P11)?;
        let omni = pioneer_to_omni(&raw);
        eprintln!("  {} Pioneer 11 records", omni.len());
        all_records.extend(omni);
    }

    if let Some(ref path) = cli.nh_swap_file {
        eprintln!("loading NH SWAP from: {} (no magnetometer)", path.display());
        let raw = parse_nh_swap_file(path)?;
        let omni = nh_swap_to_omni(&raw);
        eprintln!("  {} NH SWAP records (B-field = NaN)", omni.len());
        all_records.extend(omni);
    }

    if let Some(ref path) = cli.juno_file {
        eprintln!("loading Juno cruise from: {}", path.display());
        let raw = parse_juno_cruise_file(path)?;
        let omni = juno_to_omni(&raw);
        eprintln!("  {} Juno cruise records", omni.len());
        all_records.extend(omni);
    }

    if let Some(ref path) = cli.cassini_file {
        eprintln!("loading Cassini cruise from: {}", path.display());
        let raw = parse_cassini_cruise_file(path)?;
        let omni = cassini_to_omni(&raw);
        eprintln!("  {} Cassini cruise records", omni.len());
        all_records.extend(omni);
    }

    // Ulysses: merged format (single file with plasma + MAG)
    if let Some(ref path) = cli.ulysses_swoops_file {
        eprintln!("loading Ulysses from: {}", path.display());
        let raw = parse_ulysses_file(path)?;
        // If separate MAG file provided, use merge; otherwise treat as merged
        if let Some(ref _mag_path) = cli.ulysses_mag_file {
            // The merged SPDF file already contains both plasma and MAG.
            // Separate SWOOPS+MAG files would need the raw parsers, but
            // SPDF provides merged hourly. Use merged data directly.
            eprintln!("  (MAG file ignored: SPDF merged already contains B-field)");
        }
        let omni = ulysses_to_omni(&raw);
        eprintln!("  {} Ulysses records (lat range available)", omni.len());
        all_records.extend(omni);
    }

    Ok(all_records)
}

/// Latitudinal profile parameters for fast/slow solar wind transition.
///
/// Ulysses uniquely constrains the heliographic latitude dependence:
/// fast polar wind (~750 km/s, ~3 cm^-3) transitions to slow equatorial
/// wind (~400 km/s, ~7 cm^-3) over a ~10-20 degree latitude band.
/// The transition is modeled by a tanh profile.
struct LatitudinalProfile {
    /// Fast polar wind speed (km/s). Default: 750.
    fast_speed_kms: f64,
    /// Slow equatorial wind speed (km/s). Default: 400.
    slow_speed_kms: f64,
    /// Fast polar wind density (cm^-3). Default: 3.0.
    fast_density_cm3: f64,
    /// Slow equatorial wind density (cm^-3). Default: 7.0.
    slow_density_cm3: f64,
    /// Transition latitude (degrees from equator). Default: 30.
    transition_lat_deg: f64,
    /// Width of the transition band (degrees). Default: 10.
    transition_width_deg: f64,
}

impl Default for LatitudinalProfile {
    fn default() -> Self {
        Self {
            fast_speed_kms: 750.0,
            slow_speed_kms: 400.0,
            fast_density_cm3: 3.0,
            slow_density_cm3: 7.0,
            transition_lat_deg: 30.0,
            transition_width_deg: 10.0,
        }
    }
}

/// Fit a latitudinal profile from Ulysses data (OmniRecords with lat_deg).
///
/// Groups records by latitude bin, computes median speed and density at
/// each latitude, and fits tanh transition parameters. If insufficient
/// data, returns defaults from McComas et al. (2000).
fn ulysses_latitudinal_fit(records: &[OmniRecord]) -> LatitudinalProfile {
    // Filter records that have valid latitude data
    let lat_records: Vec<&OmniRecord> = records
        .iter()
        .filter(|r| r.lat_deg.is_finite() && r.bulk_speed.is_finite())
        .collect();

    if lat_records.len() < 10 {
        eprintln!(
            "  insufficient Ulysses latitude data ({} records), using defaults",
            lat_records.len(),
        );
        return LatitudinalProfile::default();
    }

    // Separate polar (|lat| > 40) and equatorial (|lat| < 20) records
    let polar: Vec<&&OmniRecord> = lat_records
        .iter()
        .filter(|r| r.lat_deg.abs() > 40.0)
        .collect();
    let equatorial: Vec<&&OmniRecord> = lat_records
        .iter()
        .filter(|r| r.lat_deg.abs() < 20.0)
        .collect();

    let fast_speed = if polar.len() >= 3 {
        median_finite(polar.iter().map(|r| r.bulk_speed))
    } else {
        750.0
    };

    let slow_speed = if equatorial.len() >= 3 {
        median_finite(equatorial.iter().map(|r| r.bulk_speed))
    } else {
        400.0
    };

    let fast_density = if polar.len() >= 3 {
        median_finite(polar.iter().map(|r| r.proton_density))
    } else {
        3.0
    };

    let slow_density = if equatorial.len() >= 3 {
        median_finite(equatorial.iter().map(|r| r.proton_density))
    } else {
        7.0
    };

    eprintln!(
        "  Ulysses fit: fast={:.0} km/s ({:.1} cm^-3), slow={:.0} km/s ({:.1} cm^-3)",
        fast_speed, fast_density, slow_speed, slow_density,
    );
    eprintln!(
        "  latitude data: {} polar (|lat|>40), {} equatorial (|lat|<20)",
        polar.len(),
        equatorial.len(),
    );

    LatitudinalProfile {
        fast_speed_kms: fast_speed,
        slow_speed_kms: slow_speed,
        fast_density_cm3: fast_density,
        slow_density_cm3: slow_density,
        transition_lat_deg: 30.0,
        transition_width_deg: 10.0,
    }
}

/// Compute latitude modulation factors for density and speed at a given z-cell.
///
/// Returns `(density_factor, speed_factor)` as multiplicative modulations
/// relative to the equatorial values. The z-axis maps to heliographic
/// latitude: z=0 -> -lat_max, z=nz/2 -> equator, z=nz-1 -> +lat_max.
///
/// Uses a tanh transition profile:
///   f(lat) = 0.5 * (1 + tanh((|lat| - lat_transition) / width))
/// At the equator (lat=0): f -> 0 (slow wind)
/// At poles (|lat|>transition): f -> 1 (fast wind)
fn latitude_modulation(
    z: usize,
    nz: usize,
    lat_max_deg: f64,
    profile: &LatitudinalProfile,
) -> (f64, f64) {
    // Map z to latitude: z=0 -> -lat_max, z=nz/2 -> 0, z=nz-1 -> +lat_max
    let lat_deg = if nz > 1 {
        -lat_max_deg + 2.0 * lat_max_deg * z as f64 / (nz - 1) as f64
    } else {
        0.0
    };

    // Tanh transition: smooth step from equatorial to polar
    let arg = (lat_deg.abs() - profile.transition_lat_deg) / profile.transition_width_deg;
    let f_polar = 0.5 * (1.0 + arg.tanh());

    // Density: equatorial is DENSER, polar is LESS dense
    // density_factor = slow_n * (1-f) + fast_n * f, normalized to equatorial
    let density_factor =
        (1.0 - f_polar) + f_polar * (profile.fast_density_cm3 / profile.slow_density_cm3);

    // Speed: polar is FASTER, equatorial is SLOWER
    // speed_factor = slow_v * (1-f) + fast_v * f, normalized to equatorial
    let speed_factor =
        (1.0 - f_polar) + f_polar * (profile.fast_speed_kms / profile.slow_speed_kms);

    (density_factor, speed_factor)
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Radial mode: multi-distance heliospheric profile
    if cli.radial_mode {
        return run_radial_mode(&cli);
    }

    // Standard mode: L1 time-series IC generation
    // Load data via adapter pattern.
    // Priority: OMNI > WIND SWE+MFI > ACE MAG > STEREO > SOHO CELIAS > SWEPAM > Cassini > builtin
    let records: Vec<OmniRecord> = if let Some(ref path) = cli.omni_file {
        eprintln!("loading NASA OMNI2 from: {}", path.display());
        let raw = parse_omni_file(path)?;
        filter_valid_omni(&raw)
    } else if cli.wind_swe_file.is_some() || cli.wind_mfi_file.is_some() {
        load_wind_data(&cli)?
    } else if let Some(ref path) = cli.ace_mag_file {
        eprintln!("loading ACE MAG L2 from: {} (B-field only)", path.display());
        let raw = parse_ace_mag_file(path)?;
        let hourly = average_to_hourly(&raw);
        eprintln!(
            "  {} 16-sec samples -> {} hourly averages",
            raw.len(),
            hourly.len()
        );
        let omni = ace_mag_to_omni(&hourly);
        filter_valid_omni_mag(&omni)
    } else if cli.stereo_file.is_some() || cli.stereo_mag_file.is_some() {
        load_stereo_data(&cli)?
    } else if let Some(ref path) = cli.soho_celias_file {
        let cadence = select_soho_cadence(&cli)?;
        eprintln!("loading SOHO CELIAS PM bundle from: {}", path.display());
        let raw = parse_soho_celias_bundle_file(path)?;
        let omni = if cadence == "native" {
            eprintln!(
                "  cadence route: native CELIAS samples (time_resolution={} s <= 900 s, no real B-field)",
                cli.time_resolution
            );
            soho_to_native_omni(&raw)
        } else {
            eprintln!(
                "  cadence route: hourly median-normalized CELIAS boundary (time_resolution={} s, no real B-field)",
                cli.time_resolution
            );
            soho_to_hourly_omni(&raw)
        };
        eprintln!(
            "  {} raw CELIAS records -> {} boundary records",
            raw.len(),
            omni.len()
        );
        filter_valid_omni(&omni)
    } else if let Some(ref path) = cli.swepam_file {
        eprintln!(
            "loading ACE SWEPAM from: {} (no real B-field)",
            path.display()
        );
        let raw = parse_swepam_file(path)?;
        swepam_to_omni(&raw)
    } else if let Some(ref path) = cli.cassini_file {
        eprintln!("loading Cassini cruise hourly from: {}", path.display());
        let raw = parse_cassini_cruise_file(path)?;
        let omni = cassini_to_omni(&raw);
        eprintln!("  {} Cassini cruise records", omni.len());
        filter_valid_omni(&omni)
    } else {
        eprintln!("using built-in OMNI2 sample (2024 DOY 1-2, 48 hours, real B-field)");
        let raw = parse_omni_hourly(BUILTIN_OMNI_SAMPLE);
        filter_valid_omni(&raw)
    };

    eprintln!("{} valid records after filtering", records.len());
    if records.is_empty() {
        anyhow::bail!("no valid records after filtering");
    }

    // Select time window
    let num_hours = if cli.num_hours == 0 {
        cli.nx.min(records.len())
    } else {
        cli.num_hours.min(records.len())
    };
    let end = (cli.start_hour + num_hours).min(records.len());
    let start = end.saturating_sub(num_hours);
    let window = &records[start..end];

    eprintln!(
        "using {} hours (indices {}..{}) for {}x{}x{} grid",
        window.len(),
        start,
        end,
        cli.nx,
        cli.ny,
        cli.nz,
    );

    // Unit conversion from the selected window
    let mut units = UnitConversion::from_omni(window);
    units.density_clamp = parse_clamp_range(&cli.clamp_density_range);
    units.speed_clamp = parse_clamp_range(&cli.clamp_speed_range);

    if cli.time_resolution != 3600 {
        eprintln!(
            "time resolution: {} sec/x-slice (nx={} covers {} sec = {:.1} min)",
            cli.time_resolution,
            cli.nx,
            cli.nx as u64 * cli.time_resolution as u64,
            cli.nx as f64 * cli.time_resolution as f64 / 60.0,
        );
    }
    eprintln!(
        "units: n_ref={:.2} cm^-3, v_ref={:.1} km/s, u_scale={:.4}",
        units.n_ref, units.v_ref, units.u_scale,
    );
    eprintln!(
        "clamp: density=[{:.4}, {:.4}], speed=[{:.5}, {:.4}]",
        units.density_clamp[0], units.density_clamp[1], units.speed_clamp[0], units.speed_clamp[1],
    );

    // Report physical ranges
    let b_real_count = window.iter().filter(|r| !r.bx_gse.is_nan()).count();
    eprintln!(
        "B-field: {}/{} hours have real data (rest use Parker spiral fallback)",
        b_real_count,
        window.len(),
    );

    // Latitudinal profile (if --latitudinal enabled)
    let lat_profile = if cli.latitudinal {
        eprintln!(
            "latitudinal mode: lat_max={:.0} deg, tanh transition",
            cli.lat_max_deg,
        );
        // Use Ulysses data if loaded, otherwise defaults from McComas et al.
        if let Some(ref path) = cli.ulysses_swoops_file {
            eprintln!("  fitting latitudinal profile from: {}", path.display());
            let uly_records = parse_ulysses_file(path)?;
            let uly_omni = ulysses_to_omni(&uly_records);
            Some(ulysses_latitudinal_fit(&uly_omni))
        } else {
            eprintln!("  no Ulysses data; using McComas et al. (2000) defaults");
            Some(LatitudinalProfile::default())
        }
    } else {
        None
    };

    // Generate IC
    let data = if cli.triangulate {
        // Triangulation mode: interpolate between L1 and STEREO along Y.
        // Requires an L1 anchor (OMNI, WIND, or ACE) as the primary source.
        // STEREO alone cannot serve as both primary and secondary -- you
        // cannot triangulate a point against itself.
        let has_l1_primary = cli.omni_file.is_some()
            || cli.wind_swe_file.is_some()
            || cli.wind_mfi_file.is_some()
            || cli.ace_mag_file.is_some()
            || cli.soho_celias_file.is_some()
            || cli.swepam_file.is_some();
        if !has_l1_primary {
            anyhow::bail!(
                "--triangulate requires an L1 anchor (--omni-file, --wind-swe-file, \
                 --wind-mfi-file, --ace-mag-file, --soho-celias-file, or --swepam-file). \
                 STEREO cannot be both the primary and secondary source."
            );
        }
        if cli.stereo_file.is_none() && cli.stereo_mag_file.is_none() {
            anyhow::bail!("--triangulate requires --stereo-file and/or --stereo-mag-file");
        }
        let stereo_records = load_stereo_data(&cli)?;
        if stereo_records.is_empty() {
            anyhow::bail!("no valid STEREO records for triangulation");
        }
        let stereo_end = (cli.start_hour + num_hours).min(stereo_records.len());
        let stereo_start = stereo_end.saturating_sub(num_hours);
        let stereo_window = &stereo_records[stereo_start..stereo_end];
        eprintln!(
            "triangulate: {} L1 hours + {} STEREO hours, sep={:.1} deg",
            window.len(),
            stereo_window.len(),
            cli.stereo_sep_deg,
        );
        triangulate_ic_from_multi_spacecraft(window, stereo_window, &cli, &units)
    } else {
        generate_ic_from_omni(window, &cli, &units, lat_profile.as_ref())
    };

    // Knudsen diagnostic from L1 records
    let (kn_regime, kn_max) = compute_knudsen_diagnostic(window);

    output_diagnostics_and_write(&data, &units, kn_regime, kn_max, &cli)
}

/// Radial mode: build heliospheric profile from multi-spacecraft data
/// and generate IC with x-axis spanning r_min to r_max AU.
fn run_radial_mode(cli: &Cli) -> anyhow::Result<()> {
    eprintln!(
        "radial mode: {:.1}-{:.1} AU, {}x{}x{} grid",
        cli.r_min_au, cli.r_max_au, cli.nx, cli.ny, cli.nz,
    );

    if cli.r_min_au >= cli.r_max_au {
        anyhow::bail!(
            "--r-min-au ({}) must be less than --r-max-au ({})",
            cli.r_min_au,
            cli.r_max_au
        );
    }

    // Load L1 data (1 AU anchor) if available
    let mut all_records: Vec<OmniRecord> = Vec::new();

    if let Some(ref path) = cli.omni_file {
        eprintln!("loading L1 anchor (OMNI) from: {}", path.display());
        let raw = parse_omni_file(path)?;
        let valid = filter_valid_omni(&raw);
        eprintln!("  {} L1 records at 1 AU", valid.len());
        all_records.extend(valid);
    }

    // Load outer heliosphere spacecraft
    let outer = load_outer_heliosphere(cli)?;
    eprintln!("{} outer heliosphere records loaded", outer.len());
    all_records.extend(outer);

    if all_records.is_empty() {
        anyhow::bail!(
            "--radial-mode requires spacecraft data. Provide at least 2 files \
             at different heliocentric distances (e.g., --omni-file + --voyager1-file)"
        );
    }

    // Build radial profile from all spacecraft
    let profile = build_radial_profile(&all_records);
    eprintln!("radial profile: {} distance bins", profile.len());
    if profile.len() < 2 {
        eprintln!(
            "WARNING: only {} distance bin(s). Radial IC will rely heavily \
             on scaling laws rather than measured data.",
            profile.len(),
        );
    }

    // Report distance coverage
    if let (Some(first), Some(last)) = (profile.first(), profile.last()) {
        eprintln!(
            "  data coverage: {:.1}-{:.1} AU ({} bins)",
            first.r_au,
            last.r_au,
            profile.len(),
        );
        // Report a few representative points
        for p in &profile {
            if p.density_cm3.is_finite() {
                eprintln!(
                    "  r={:.1} AU: n={:.4} cm^-3, v={:.0} km/s, |B|={:.2} nT",
                    p.r_au, p.density_cm3, p.speed_kms, p.b_mag_nt,
                );
            }
        }
    }

    let sampled_profile =
        sample_radial_profile(&profile, cli.r_min_au, cli.r_max_au, cli.nx.max(2));
    write_radial_artifacts(cli, &profile, &sampled_profile)?;

    // Compute distance-adaptive unit conversion
    let mut units = radial_unit_conversion(&profile, cli.r_min_au, cli.r_max_au);
    units.density_clamp = parse_clamp_range(&cli.clamp_density_range);
    units.speed_clamp = parse_clamp_range(&cli.clamp_speed_range);
    eprintln!(
        "units (radial): n_ref={:.4} cm^-3, v_ref={:.1} km/s, u_scale={:.4}",
        units.n_ref, units.v_ref, units.u_scale,
    );
    eprintln!(
        "clamp: density=[{:.6}, {:.2}], speed=[{:.5}, {:.4}]",
        units.density_clamp[0], units.density_clamp[1], units.speed_clamp[0], units.speed_clamp[1],
    );

    // Latitudinal profile for radial mode (if --latitudinal enabled)
    let lat_profile = if cli.latitudinal {
        eprintln!(
            "latitudinal mode: lat_max={:.0} deg, tanh transition",
            cli.lat_max_deg,
        );
        // Ulysses data is already in all_records via load_outer_heliosphere
        let profile_fit = ulysses_latitudinal_fit(&all_records);
        Some(profile_fit)
    } else {
        None
    };

    // Generate radial IC
    let data = generate_radial_ic(&profile, cli, &units, lat_profile.as_ref());

    // Knudsen diagnostic from all records
    let (kn_regime, kn_max) = compute_knudsen_diagnostic(&all_records);

    output_diagnostics_and_write(&data, &units, kn_regime, kn_max, cli)
}

/// Compute Knudsen number statistics from a set of OmniRecords.
fn compute_knudsen_diagnostic(records: &[OmniRecord]) -> (KnudsenRegime, f64) {
    let mut kinetic_count = 0usize;
    let mut transitional_count = 0usize;
    let mut kn_max = 0.0_f64;
    for r in records {
        if !r.proton_density.is_nan() && !r.proton_temperature.is_nan() {
            let kn = knudsen_number(r.proton_density, r.proton_temperature);
            if kn.is_finite() {
                kn_max = kn_max.max(kn);
                match classify_knudsen(kn) {
                    KnudsenRegime::Kinetic => kinetic_count += 1,
                    KnudsenRegime::Transitional => transitional_count += 1,
                    KnudsenRegime::Fluid => {}
                }
            }
        }
    }
    let kn_regime = if kinetic_count > 0 {
        KnudsenRegime::Kinetic
    } else if transitional_count > 0 {
        KnudsenRegime::Transitional
    } else {
        KnudsenRegime::Fluid
    };
    eprintln!(
        "Kn diagnostic: max={kn_max:.1}, regime={kn_regime:?}, \
         kinetic={kinetic_count}, transitional={transitional_count}/{} records",
        records.len(),
    );
    if kinetic_count > 0 {
        eprintln!(
            "WARNING: {kinetic_count} records in kinetic regime (Kn > 100). \
             LBM fluid approximation may not be valid. Results should be \
             interpreted as effective-viscosity averages, not kinetic physics."
        );
    }
    (kn_regime, kn_max)
}

/// Print IC diagnostics and write output file(s).
fn output_diagnostics_and_write(
    data: &[CellIc],
    units: &UnitConversion,
    kn_regime: KnudsenRegime,
    kn_max: f64,
    cli: &Cli,
) -> anyhow::Result<()> {
    let (rho_min, rho_max) = data.iter().fold((f64::MAX, f64::MIN), |(lo, hi), c| {
        (lo.min(c.rho), hi.max(c.rho))
    });
    let b_max = data
        .iter()
        .map(|c| (c.b[0] * c.b[0] + c.b[1] * c.b[1] + c.b[2] * c.b[2]).sqrt())
        .fold(0.0_f64, f64::max);
    let u_max = data.iter().map(|c| c.u[0]).fold(0.0_f64, f64::max);

    eprintln!("LBM rho range: [{rho_min:.4}, {rho_max:.4}]");
    eprintln!("max |B| (LBM): {b_max:.6e}");
    eprintln!("max u_x (LBM): {u_max:.6e}");
    eprintln!("total cells: {}", data.len());

    // Write output
    match cli.format.as_str() {
        "volume" => {
            write_volume(&cli.out, data, units, kn_regime, kn_max)?;
            eprintln!("wrote volume: {}", cli.out.display());
        }
        "slices" => {
            write_slices(&cli.out, data, cli.nz)?;
            eprintln!("wrote {} slices to: {}", cli.nz, cli.out.display());
        }
        other => {
            anyhow::bail!("unknown format '{}': use 'volume' or 'slices'", other);
        }
    }

    eprintln!("done.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create an OmniRecord with given time and plasma values.
    fn make_record(year: u16, doy: u16, hour: u8, density: f64, speed: f64) -> OmniRecord {
        OmniRecord {
            year,
            doy,
            hour,
            b_magnitude: 5.0,
            bx_gse: 3.0,
            by_gse: -2.0,
            bz_gse: 1.0,
            proton_temperature: 1e5,
            proton_density: density,
            bulk_speed: speed,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au: 1.0,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
        }
    }

    #[test]
    fn test_time_aligned_intersection_partial_overlap() {
        // L1: hours 0-4, STEREO: hours 2-6 -> intersection = hours 2-4
        let l1: Vec<OmniRecord> = (0..5)
            .map(|h| make_record(2024, 1, h, 5.0 + h as f64, 400.0))
            .collect();
        let stereo: Vec<OmniRecord> = (2..7)
            .map(|h| make_record(2024, 1, h, 7.0 + h as f64, 350.0))
            .collect();

        let (l1_a, st_a) = time_aligned_intersection(&l1, &stereo);

        // Intersection should have 3 records (hours 2, 3, 4)
        assert_eq!(l1_a.len(), 3, "intersection should have 3 records");
        assert_eq!(st_a.len(), 3, "stereo should match L1 length");

        // Verify the correct records were selected
        assert_eq!(l1_a[0].hour, 2);
        assert_eq!(l1_a[1].hour, 3);
        assert_eq!(l1_a[2].hour, 4);

        // Verify L1 and STEREO records differ (different density)
        assert!((l1_a[0].proton_density - 7.0).abs() < 0.01); // 5.0 + 2
        assert!((st_a[0].proton_density - 9.0).abs() < 0.01); // 7.0 + 2
    }

    #[test]
    fn test_time_aligned_intersection_different_telemetry_rates() {
        // Simulate different drop rates: L1 has all 10 hours,
        // STEREO has only odd hours (5 records vs 10)
        let l1: Vec<OmniRecord> = (0..10)
            .map(|h| make_record(2024, 1, h, 5.0, 400.0))
            .collect();
        let stereo: Vec<OmniRecord> = (0..10)
            .filter(|h| h % 2 == 1)
            .map(|h| make_record(2024, 1, h as u8, 7.0, 350.0))
            .collect();

        let (l1_a, st_a) = time_aligned_intersection(&l1, &stereo);

        // Only odd hours overlap: 1, 3, 5, 7, 9
        assert_eq!(l1_a.len(), 5, "intersection with different rates");
        assert_eq!(st_a.len(), 5);
        assert_eq!(l1_a[0].hour, 1);
        assert_eq!(l1_a[4].hour, 9);
    }

    #[test]
    #[should_panic(expected = "No overlapping timestamps")]
    fn test_time_aligned_intersection_no_overlap_panics() {
        // L1: DOY 1, STEREO: DOY 2 -> no overlap
        let l1 = vec![make_record(2024, 1, 0, 5.0, 400.0)];
        let stereo = vec![make_record(2024, 2, 0, 7.0, 350.0)];
        let _ = time_aligned_intersection(&l1, &stereo);
    }

    /// Helper: create an OmniRecord at a specific heliocentric distance.
    fn make_record_at_r(
        r_au: f64,
        density: f64,
        speed: f64,
        temp: f64,
        bx: f64,
        by: f64,
        bz: f64,
    ) -> OmniRecord {
        OmniRecord {
            year: 2024,
            doy: 1,
            hour: 0,
            b_magnitude: (bx * bx + by * by + bz * bz).sqrt(),
            bx_gse: bx,
            by_gse: by,
            bz_gse: bz,
            proton_temperature: temp,
            proton_density: density,
            bulk_speed: speed,
            flow_pressure: f64::NAN,
            plasma_beta: f64::NAN,
            alfven_mach: f64::NAN,
            dst_index: f64::NAN,
            ae_index: f64::NAN,
            kp_times_10: 0,
            r_au,
            lat_deg: f64::NAN,
            lon_deg: f64::NAN,
        }
    }

    #[test]
    fn test_build_radial_profile_groups_by_distance() {
        // Create records at 1, 5, and 100 AU
        let records = vec![
            make_record_at_r(1.0, 5.0, 400.0, 1e5, 3.0, -2.0, 1.0),
            make_record_at_r(1.0, 6.0, 380.0, 1.1e5, 2.5, -1.5, 0.8),
            make_record_at_r(5.0, 0.2, 420.0, 2e4, 0.12, -0.08, 0.03),
            make_record_at_r(100.0, 0.0005, 410.0, 1500.0, 0.0003, -0.00015, 0.0),
        ];

        let profile = build_radial_profile(&records);
        assert_eq!(profile.len(), 3, "3 distinct distance bins");
        assert!((profile[0].r_au - 1.0).abs() < 0.1);
        assert!((profile[1].r_au - 5.0).abs() < 0.1);
        assert!((profile[2].r_au - 100.0).abs() < 0.1);

        // Median of 5.0 and 6.0 at 1 AU
        let n_1au = profile[0].density_cm3;
        assert!(
            (5.0..=6.0).contains(&n_1au),
            "median density at 1 AU: {n_1au}"
        );
    }

    #[test]
    fn test_build_radial_profile_skips_nan_distance() {
        let records = vec![
            make_record_at_r(f64::NAN, 5.0, 400.0, 1e5, 3.0, -2.0, 1.0),
            make_record_at_r(1.0, 5.0, 400.0, 1e5, 3.0, -2.0, 1.0),
        ];
        let profile = build_radial_profile(&records);
        assert_eq!(profile.len(), 1, "NaN distance record should be skipped");
    }

    #[test]
    fn test_fit_power_law_recovers_inverse_square_density() {
        let sampled = vec![
            RadialProfilePoint {
                r_au: 1.0,
                density_cm3: 5.0,
                speed_kms: 400.0,
                temp_k: 1.0e5,
                br_nt: 5.0,
                bt_nt: -3.0,
                bn_nt: 0.0,
                b_mag_nt: 5.83,
            },
            RadialProfilePoint {
                r_au: 10.0,
                density_cm3: 0.05,
                speed_kms: 400.0,
                temp_k: 2.15e4,
                br_nt: 0.05,
                bt_nt: -0.3,
                bn_nt: 0.0,
                b_mag_nt: 0.304,
            },
            RadialProfilePoint {
                r_au: 100.0,
                density_cm3: 0.0005,
                speed_kms: 400.0,
                temp_k: 4.64e3,
                br_nt: 0.0005,
                bt_nt: -0.03,
                bn_nt: 0.0,
                b_mag_nt: 0.03,
            },
        ];
        let (slope, n) = fit_power_law(&sampled, |row| row.density_cm3).unwrap();
        assert_eq!(n, 3);
        assert!(
            (slope + 2.0).abs() < 0.05,
            "unexpected fitted slope {slope}"
        );
    }

    #[test]
    fn test_interpolate_radial_r_squared_density() {
        // n(r) should scale as r^-2 for a single data point at 1 AU
        let profile = vec![RadialProfilePoint {
            r_au: 1.0,
            density_cm3: 5.0,
            speed_kms: 400.0,
            temp_k: 1e5,
            br_nt: 5.0,
            bt_nt: -3.0,
            bn_nt: 0.0,
            b_mag_nt: 5.83,
        }];

        // At 10 AU, density should be 5.0 / 100 = 0.05
        let p10 = interpolate_radial(&profile, 10.0);
        let expected = 5.0 / 100.0;
        let rel_err = (p10.density_cm3 - expected).abs() / expected;
        assert!(
            rel_err < 0.01,
            "density at 10 AU: {:.6} (expected {expected:.6}, err={rel_err:.4})",
            p10.density_cm3,
        );

        // Speed should be constant
        assert!(
            (p10.speed_kms - 400.0).abs() < 1.0,
            "speed should be ~constant: {}",
            p10.speed_kms,
        );

        // B_r should scale as r^-2
        let br_10 = p10.br_nt;
        let br_expected = 5.0 / 100.0;
        assert!(
            (br_10 - br_expected).abs() / br_expected < 0.01,
            "B_r at 10 AU: {br_10:.6} (expected {br_expected:.6})",
        );
    }

    #[test]
    fn test_interpolate_radial_between_two_points() {
        let profile = vec![
            RadialProfilePoint {
                r_au: 1.0,
                density_cm3: 5.0,
                speed_kms: 400.0,
                temp_k: 1e5,
                br_nt: 5.0,
                bt_nt: -3.0,
                bn_nt: 0.0,
                b_mag_nt: 5.83,
            },
            RadialProfilePoint {
                r_au: 100.0,
                density_cm3: 0.0005,
                speed_kms: 410.0,
                temp_k: 1500.0,
                br_nt: 0.0005,
                bt_nt: -0.03,
                bn_nt: 0.0,
                b_mag_nt: 0.03,
            },
        ];

        // Interpolate at 10 AU (log-midpoint between 1 and 100)
        let p10 = interpolate_radial(&profile, 10.0);

        // Density should be between the two values (log-linear interpolation)
        assert!(p10.density_cm3 > 0.0005, "density above 100 AU value");
        assert!(p10.density_cm3 < 5.0, "density below 1 AU value");
        // Speed should interpolate linearly
        assert!(
            p10.speed_kms > 399.0 && p10.speed_kms < 411.0,
            "speed: {}",
            p10.speed_kms
        );
    }

    #[test]
    fn test_interpolate_radial_empty_profile() {
        let profile: Vec<RadialProfilePoint> = vec![];
        let p = interpolate_radial(&profile, 10.0);
        // Should use scaling laws from defaults
        let expected_n = 5.0 / 100.0; // 5.0 / (10^2)
        assert!(
            (p.density_cm3 - expected_n).abs() / expected_n < 0.01,
            "fallback density: {}",
            p.density_cm3,
        );
    }

    #[test]
    fn test_radial_unit_conversion_adapts_to_range() {
        let profile = vec![RadialProfilePoint {
            r_au: 1.0,
            density_cm3: 5.0,
            speed_kms: 400.0,
            temp_k: 1e5,
            br_nt: 5.0,
            bt_nt: -3.0,
            bn_nt: 0.0,
            b_mag_nt: 5.83,
        }];

        let units_narrow = radial_unit_conversion(&profile, 0.5, 1.5);
        let units_wide = radial_unit_conversion(&profile, 1.0, 100.0);

        // Wide range should have wider density clamps
        assert!(
            units_wide.density_clamp[1] > units_narrow.density_clamp[1],
            "wider range needs wider clamps: narrow={}, wide={}",
            units_narrow.density_clamp[1],
            units_wide.density_clamp[1],
        );

        // Both should have valid u_scale
        assert_eq!(units_narrow.u_scale, 0.05);
        assert_eq!(units_wide.u_scale, 0.05);
    }

    #[test]
    fn test_generate_radial_ic_cell_count() {
        let profile = vec![RadialProfilePoint {
            r_au: 1.0,
            density_cm3: 5.0,
            speed_kms: 400.0,
            temp_k: 1e5,
            br_nt: 5.0,
            bt_nt: -3.0,
            bn_nt: 0.0,
            b_mag_nt: 5.83,
        }];

        let cli = Cli::parse_from([
            "solar-wind-ic",
            "--radial-mode",
            "--nx",
            "16",
            "--ny",
            "8",
            "--nz",
            "8",
            "--r-min-au",
            "1.0",
            "--r-max-au",
            "100.0",
        ]);

        let units = radial_unit_conversion(&profile, cli.r_min_au, cli.r_max_au);
        let data = generate_radial_ic(&profile, &cli, &units, None);

        assert_eq!(data.len(), 16 * 8 * 8, "total cell count");
    }

    #[test]
    fn test_generate_radial_ic_density_decreases_with_x() {
        // Density should decrease along x-axis (increasing distance)
        let profile = vec![RadialProfilePoint {
            r_au: 1.0,
            density_cm3: 5.0,
            speed_kms: 400.0,
            temp_k: 1e5,
            br_nt: 5.0,
            bt_nt: -3.0,
            bn_nt: 0.0,
            b_mag_nt: 5.83,
        }];

        let cli = Cli::parse_from([
            "solar-wind-ic",
            "--radial-mode",
            "--nx",
            "32",
            "--ny",
            "4",
            "--nz",
            "4",
            "--r-min-au",
            "1.0",
            "--r-max-au",
            "100.0",
        ]);

        let units = radial_unit_conversion(&profile, cli.r_min_au, cli.r_max_au);
        let data = generate_radial_ic(&profile, &cli, &units, None);

        // Compare density at x=0 (1 AU) vs x=31 (100 AU), y=0, z=0
        let rho_inner = data
            .iter()
            .find(|c| c.x == 0 && c.y == 0 && c.z == 0)
            .unwrap()
            .rho;
        let rho_outer = data
            .iter()
            .find(|c| c.x == 31 && c.y == 0 && c.z == 0)
            .unwrap()
            .rho;

        assert!(
            rho_inner > rho_outer,
            "density should decrease with distance: inner={rho_inner}, outer={rho_outer}",
        );

        // The ratio should be ~10000x (100^2), subject to clamping
        let ratio = rho_inner / rho_outer;
        assert!(ratio > 10.0, "density ratio should be large: {ratio}");
    }

    #[test]
    fn test_median_finite() {
        assert!((median_finite([1.0, 2.0, 3.0].iter().copied()) - 2.0).abs() < 1e-10);
        assert!((median_finite([f64::NAN, 5.0, 3.0].iter().copied()) - 5.0).abs() < 1e-10);
        assert!(median_finite([f64::NAN, f64::NAN].iter().copied()).is_nan());
        assert!((median_finite([7.0].iter().copied()) - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_interp_log() {
        // interp_log(1.0, 100.0, 0.5) = exp(0.5 * ln(1) + 0.5 * ln(100)) = 10.0
        let val = interp_log(1.0, 100.0, 0.5);
        assert!((val - 10.0).abs() < 0.01, "interp_log midpoint: {val}");

        // Endpoints
        let v0 = interp_log(1.0, 100.0, 0.0);
        assert!((v0 - 1.0).abs() < 0.01, "interp_log t=0: {v0}");
        let v1 = interp_log(1.0, 100.0, 1.0);
        assert!((v1 - 100.0).abs() < 0.01, "interp_log t=1: {v1}");
    }

    #[test]
    fn test_scale_r2() {
        // ratio = r_ref/r_target = 1/10 = 0.1 => val * 0.01
        assert!((scale_r2(5.0, 0.1) - 0.05).abs() < 1e-10);
        // ratio = 2.0 => val * 4.0
        assert!((scale_r2(5.0, 2.0) - 20.0).abs() < 1e-10);
        assert!(scale_r2(f64::NAN, 2.0).is_nan());
    }

    #[test]
    fn test_latitude_modulation_equator() {
        let profile = LatitudinalProfile::default();
        let nz = 64;
        let z_equator = nz / 2;
        let (n_factor, v_factor) = latitude_modulation(z_equator, nz, 60.0, &profile);

        // At equator (lat=0): tanh((-30)/10) ~ -0.995 => f_polar ~ 0.0025
        // density_factor ~ 1.0 (equatorial reference)
        // speed_factor ~ 1.0 (equatorial reference)
        assert!(
            (n_factor - 1.0).abs() < 0.05,
            "equatorial density factor should be ~1.0: {n_factor}",
        );
        assert!(
            (v_factor - 1.0).abs() < 0.05,
            "equatorial speed factor should be ~1.0: {v_factor}",
        );
    }

    #[test]
    fn test_latitude_modulation_pole() {
        let profile = LatitudinalProfile::default();
        let nz = 64;
        // z=63 maps to lat_max = +60 deg (well into polar regime)
        let z_pole = nz - 1;
        let (n_factor, v_factor) = latitude_modulation(z_pole, nz, 60.0, &profile);

        // At |lat|=60 > transition(30): f_polar ~ 1.0
        // density_factor ~ fast_n/slow_n = 3/7 ~ 0.43
        assert!(
            n_factor < 0.8,
            "polar density factor should be < 0.8 (fast wind less dense): {n_factor}",
        );

        // speed_factor ~ fast_v/slow_v = 750/400 = 1.875
        assert!(
            v_factor > 1.5,
            "polar speed factor should be > 1.5 (fast wind faster): {v_factor}",
        );
    }

    #[test]
    fn test_latitude_modulation_symmetric() {
        let profile = LatitudinalProfile::default();
        let nz = 64;
        // South pole (z=0) and north pole (z=63) should have the same factors
        let (n_south, v_south) = latitude_modulation(0, nz, 60.0, &profile);
        let (n_north, v_north) = latitude_modulation(nz - 1, nz, 60.0, &profile);

        assert!(
            (n_south - n_north).abs() < 1e-10,
            "density modulation should be symmetric: south={n_south}, north={n_north}",
        );
        assert!(
            (v_south - v_north).abs() < 1e-10,
            "speed modulation should be symmetric: south={v_south}, north={v_north}",
        );
    }

    #[test]
    fn test_latitude_modulation_smooth_transition() {
        let profile = LatitudinalProfile::default();
        let nz = 128;
        // Modulation factors should change smoothly (no jumps)
        let mut prev_n = f64::NAN;
        let mut max_jump = 0.0_f64;
        for z in 0..nz {
            let (n_factor, _) = latitude_modulation(z, nz, 60.0, &profile);
            if prev_n.is_finite() {
                max_jump = max_jump.max((n_factor - prev_n).abs());
            }
            prev_n = n_factor;
        }
        // Smooth tanh: max jump between adjacent cells should be small
        assert!(
            max_jump < 0.05,
            "transition should be smooth (max jump < 0.05): {max_jump}",
        );
    }

    #[test]
    fn test_ulysses_latitudinal_fit_defaults() {
        // With no data, should return defaults
        let profile = ulysses_latitudinal_fit(&[]);
        assert!((profile.fast_speed_kms - 750.0).abs() < 1e-10);
        assert!((profile.slow_speed_kms - 400.0).abs() < 1e-10);
        assert!((profile.fast_density_cm3 - 3.0).abs() < 1e-10);
        assert!((profile.slow_density_cm3 - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_generate_radial_ic_with_latitude() {
        // Radial IC with latitudinal modulation: poles should be faster/less dense
        let profile = vec![RadialProfilePoint {
            r_au: 1.0,
            density_cm3: 5.0,
            speed_kms: 400.0,
            temp_k: 1e5,
            br_nt: 5.0,
            bt_nt: -3.0,
            bn_nt: 0.0,
            b_mag_nt: 5.83,
        }];

        let cli = Cli::parse_from([
            "solar-wind-ic",
            "--radial-mode",
            "--latitudinal",
            "--lat-max-deg",
            "60",
            "--nx",
            "8",
            "--ny",
            "4",
            "--nz",
            "16",
            "--r-min-au",
            "1.0",
            "--r-max-au",
            "10.0",
        ]);

        let units = radial_unit_conversion(&profile, cli.r_min_au, cli.r_max_au);
        let lat_profile = LatitudinalProfile::default();
        let data = generate_radial_ic(&profile, &cli, &units, Some(&lat_profile));

        assert_eq!(data.len(), 8 * 4 * 16, "total cell count");

        // At x=0 (1 AU), compare equatorial z=8 vs polar z=0/z=15
        let rho_equator = data
            .iter()
            .find(|c| c.x == 0 && c.y == 0 && c.z == 8)
            .unwrap()
            .rho;
        let rho_pole = data
            .iter()
            .find(|c| c.x == 0 && c.y == 0 && c.z == 0)
            .unwrap()
            .rho;

        // Polar wind is less dense than equatorial (fast_n < slow_n)
        assert!(
            rho_pole < rho_equator,
            "polar density should be < equatorial: pole={rho_pole}, equator={rho_equator}",
        );
    }
}
