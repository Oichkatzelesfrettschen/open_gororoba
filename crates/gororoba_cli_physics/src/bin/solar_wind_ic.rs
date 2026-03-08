//! Real-data solar wind initial condition generator.
//!
//! Loads NASA OMNI2 hourly data (proton density, bulk speed, ion temperature,
//! Bx/By/Bz in GSE coordinates) and maps them to 3D LBM grid initial
//! conditions using Taylor's frozen-in hypothesis.
//!
//! Also supports ACE SWEPAM data (plasma only, no B-field -- falls back to
//! Parker spiral for magnetic field).
//!
//! GSE coordinate convention:
//!   X -> Sun-Earth line (radial, our LBM x-axis)
//!   Y -> ecliptic plane, dawn side (our LBM y-axis)
//!   Z -> ecliptic north (our LBM z-axis)
//!
//! Output: CSV files with columns x,y,z,rho,ux,uy,uz,bx,by,bz matching the
//! snapshot format used by solar-wind-mhd-sim and solar-wind-dm-mhd.

use clap::Parser;
use data_core::catalogs::omni::{OmniRecord, parse_omni_file, parse_omni_hourly};
use data_core::catalogs::solar_wind::parse_swepam_file;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

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

/// Physical-to-LBM unit conversion parameters.
struct UnitConversion {
    /// Reference proton density (cm^-3). Median of the selected window.
    n_ref: f64,
    /// Reference bulk speed (km/s). Median of the selected window.
    v_ref: f64,
    /// LBM velocity scale. u_lbm = (v / v_ref) * u_scale.
    u_scale: f64,
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
        }
    }

    fn density_to_lbm(&self, n_p: f64) -> f64 {
        if n_p.is_nan() {
            1.0
        } else {
            (n_p / self.n_ref).clamp(0.1, 10.0)
        }
    }

    fn speed_to_lbm(&self, v_sw: f64) -> f64 {
        if v_sw.is_nan() {
            self.u_scale
        } else {
            (v_sw / self.v_ref * self.u_scale).clamp(0.001, 0.15)
        }
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
fn generate_ic_from_omni(
    records: &[OmniRecord],
    cli: &Cli,
    units: &UnitConversion,
) -> Vec<CellIc> {
    let nx = cli.nx;
    let ny = cli.ny;
    let nz = cli.nz;
    let n_hours = records.len().max(1);
    let cells_per_hour = nx / n_hours;

    let mut data = Vec::with_capacity(nx * ny * nz);

    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let hour_idx = x
                    .checked_div(cells_per_hour)
                    .map_or_else(|| x * n_hours / nx, |q| q.min(n_hours - 1));

                let rec = &records[hour_idx];
                let rho = units.density_to_lbm(rec.proton_density);
                let v_lbm = units.speed_to_lbm(rec.bulk_speed);
                let u = [v_lbm, 0.0, 0.0];

                // Use real B-field from OMNI (GSE coordinates map directly
                // to LBM axes: Bx=radial, By=ecliptic transverse, Bz=north).
                // If any component is NaN, fall back to Parker spiral.
                let b = if !rec.bx_gse.is_nan()
                    && !rec.by_gse.is_nan()
                    && !rec.bz_gse.is_nan()
                {
                    [
                        rec.bx_gse * cli.b_scale,
                        rec.by_gse * cli.b_scale,
                        rec.bz_gse * cli.b_scale,
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

/// Write the IC as a single 3D volume CSV.
fn write_volume(path: &std::path::Path, data: &[CellIc]) -> std::io::Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::File::create(path)?;
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

/// SWEPAM-to-OMNI adapter: convert SWEPAM records to OmniRecord with
/// NaN for B-field (triggers Parker spiral fallback).
fn swepam_to_omni(
    records: &[data_core::catalogs::solar_wind::SwepamRecord],
) -> Vec<OmniRecord> {
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
        })
        .collect()
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    // Load data: prefer OMNI (has B-field), fall back to SWEPAM
    let records: Vec<OmniRecord> = if let Some(ref path) = cli.omni_file {
        eprintln!("loading NASA OMNI2 from: {}", path.display());
        let raw = parse_omni_file(path)?;
        filter_valid_omni(&raw)
    } else if let Some(ref path) = cli.swepam_file {
        eprintln!("loading ACE SWEPAM from: {} (no real B-field)", path.display());
        let raw = parse_swepam_file(path)?;
        swepam_to_omni(&raw)
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
        window.len(), start, end, cli.nx, cli.ny, cli.nz,
    );

    // Unit conversion from the selected window
    let units = UnitConversion::from_omni(window);
    eprintln!(
        "units: n_ref={:.2} cm^-3, v_ref={:.1} km/s, u_scale={:.4}",
        units.n_ref, units.v_ref, units.u_scale,
    );

    // Report physical ranges
    let b_real_count = window.iter().filter(|r| !r.bx_gse.is_nan()).count();
    eprintln!(
        "B-field: {}/{} hours have real data (rest use Parker spiral fallback)",
        b_real_count,
        window.len(),
    );

    // Generate IC
    let data = generate_ic_from_omni(window, &cli, &units);

    // Diagnostics
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
            write_volume(&cli.out, &data)?;
            eprintln!("wrote volume: {}", cli.out.display());
        }
        "slices" => {
            write_slices(&cli.out, &data, cli.nz)?;
            eprintln!("wrote {} slices to: {}", cli.nz, cli.out.display());
        }
        other => {
            anyhow::bail!("unknown format '{}': use 'volume' or 'slices'", other);
        }
    }

    eprintln!("done.");
    Ok(())
}
