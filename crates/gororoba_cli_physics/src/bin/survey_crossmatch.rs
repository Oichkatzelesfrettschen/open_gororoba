//! Survey crossmatch fabric for Euclid, MaNGA, LoTSS, and adjacent catalogs.

#[cfg(feature = "euclid-catalog")]
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "euclid-catalog")]
use clap::{Parser, Subcommand, ValueEnum};
#[cfg(feature = "euclid-catalog")]
use cosmology_core::euclid_morphology::{EuclidMorphologyRecord, read_euclid_visual_morphology};
#[cfg(feature = "euclid-catalog")]
use data_core::{
    CatalogModality, PreparedPointGrid, SkyGridIndex, SkyPoint,
    catalogs::{
        atnf::parse_atnf_csv,
        chime::parse_chime_csv,
        desi_bao::desi_dr2_bao,
        gaia::parse_gaia_csv,
        hi_cube::parse_hi_rotcurves,
        hst::parse_hst_public_metadata_csv,
        jwst::parse_jwst_public_metadata_csv,
        lotss::{LoTSSRelease, crossmatch_points_against_fits_catalog, lotss_fits_row_count},
        mcgill::parse_mcgill_csv,
        sdss::parse_sdss_quasar_csv,
        things::{build_things_hi_metadata, parse_things_galaxies, parse_things_hi_spectra},
    },
    for_each_point_grid_match,
    formats::fits_table::{FitsValue, read_fits_table},
    prepare_point_grid,
};
#[cfg(feature = "euclid-catalog")]
use fitsio::{FitsFile, hdu::HduInfo};
#[cfg(feature = "euclid-catalog")]
use rayon::prelude::*;
#[cfg(feature = "euclid-catalog")]
use serde::{Deserialize, Serialize};
#[cfg(feature = "euclid-catalog")]
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
    time::Instant,
};
#[cfg(feature = "euclid-catalog")]
use verified_core::topology::HardwareTopology;

#[cfg(feature = "euclid-catalog")]
const DRPALL_COLUMNS: &[&str] = &["plateifu", "mangaid", "objra", "objdec"];
#[cfg(feature = "euclid-catalog")]
const MATRIX_NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW: usize = 24;
#[cfg(feature = "euclid-catalog")]
const MATRIX_MIN_PARALLEL_FITS_CHUNK_ROWS: usize = 65_536;
#[cfg(feature = "euclid-catalog")]
const MATRIX_MAX_PARALLEL_FITS_CHUNK_ROWS: usize = 1_048_576;

#[cfg(feature = "euclid-catalog")]
#[derive(Parser, Debug)]
#[command(name = "survey-crossmatch")]
#[command(about = "Pure Rust survey crossmatch and overlap reporting")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Subcommand, Debug)]
enum Command {
    /// Crossmatch Euclid morphology objects against a LoTSS FITS release.
    EuclidLotss {
        #[arg(
            long,
            default_value = "data/external/euclid/zenodo/15106473/morphology_catalogue.parquet"
        )]
        euclid_morphology: PathBuf,

        #[arg(long, value_enum, default_value_t = ReleaseArg::Dr3)]
        release: ReleaseArg,

        #[arg(long)]
        lotss: Option<PathBuf>,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 3.0)]
        match_radius_arcsec: f64,

        #[arg(long)]
        chunk_rows: Option<usize>,

        #[arg(long)]
        workers: Option<usize>,

        #[arg(long, default_value_t = 0.5)]
        featured_min: f32,

        #[arg(long, default_value_t = 0.5)]
        spiral_min: f32,

        #[arg(long, default_value_t = 0.5)]
        face_on_min: f32,

        #[arg(long, default_value_t = 0.7)]
        non_merge_min: f32,

        #[arg(long, default_value_t = false)]
        no_morphology_cuts: bool,
    },
    /// Join Euclid-MaNGA and MaNGA-LoTSS outputs by plateifu.
    EuclidMangaLotss {
        #[arg(
            long,
            default_value = "data/external/euclid/euclid_manga_crossmatch.csv"
        )]
        euclid_manga: PathBuf,

        #[arg(long, default_value = "data/external/manga/manga_lotss_xmatch_dr3.csv")]
        manga_lotss: PathBuf,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Crossmatch ATNF pulsars against a LoTSS FITS release.
    AtnfLotss {
        #[arg(long, default_value = "data/external/atnf_pulsars.csv")]
        atnf: PathBuf,

        #[arg(long, value_enum, default_value_t = ReleaseArg::Dr3)]
        release: ReleaseArg,

        #[arg(long)]
        lotss: Option<PathBuf>,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 3.0)]
        match_radius_arcsec: f64,

        #[arg(long)]
        chunk_rows: Option<usize>,

        #[arg(long)]
        workers: Option<usize>,
    },
    /// Crossmatch McGill magnetars against a LoTSS FITS release.
    McgillLotss {
        #[arg(long, default_value = "data/external/mcgill_magnetars.csv")]
        mcgill: PathBuf,

        #[arg(long, value_enum, default_value_t = ReleaseArg::Dr3)]
        release: ReleaseArg,

        #[arg(long)]
        lotss: Option<PathBuf>,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 3.0)]
        match_radius_arcsec: f64,

        #[arg(long)]
        chunk_rows: Option<usize>,

        #[arg(long)]
        workers: Option<usize>,
    },
    /// Crossmatch THINGS table galaxies against MaNGA selected targets.
    ThingsManga {
        #[arg(long, default_value = "data/external/things/table1.dat")]
        things_table1: PathBuf,

        #[arg(long)]
        things_rotcurves: Option<PathBuf>,

        #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
        manga_selection: PathBuf,

        #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
        manga_drpall: PathBuf,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 15.0)]
        match_radius_arcsec: f64,
    },
    /// Crossmatch THINGS table galaxies against a LoTSS release.
    ThingsLotss {
        #[arg(long, default_value = "data/external/things/table1.dat")]
        things_table1: PathBuf,

        #[arg(long)]
        things_rotcurves: Option<PathBuf>,

        #[arg(long, value_enum, default_value_t = ReleaseArg::Dr3)]
        release: ReleaseArg,

        #[arg(long)]
        lotss: Option<PathBuf>,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 15.0)]
        match_radius_arcsec: f64,

        #[arg(long)]
        chunk_rows: Option<usize>,

        #[arg(long)]
        workers: Option<usize>,
    },
    /// Join THINGS galaxies that overlap both MaNGA and LoTSS.
    ThingsMangaLotss {
        #[arg(long, default_value = "data/external/things/table1.dat")]
        things_table1: PathBuf,

        #[arg(long)]
        things_rotcurves: Option<PathBuf>,

        #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
        manga_selection: PathBuf,

        #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
        manga_drpall: PathBuf,

        #[arg(long, value_enum, default_value_t = ReleaseArg::Dr3)]
        release: ReleaseArg,

        #[arg(long)]
        lotss: Option<PathBuf>,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 15.0)]
        match_radius_arcsec: f64,

        #[arg(long)]
        chunk_rows: Option<usize>,

        #[arg(long)]
        workers: Option<usize>,
    },
    /// Build THINGS HI metadata CSV from VizieR table1/table4 products.
    ThingsMetadata {
        #[arg(long, default_value = "data/external/things/table1.dat")]
        things_table1: PathBuf,

        #[arg(long, default_value = "data/external/things/table4.dat")]
        things_table4: PathBuf,

        #[arg(long)]
        output: Option<PathBuf>,

        #[arg(long)]
        report: Option<PathBuf>,

        #[arg(long, default_value_t = 6.0)]
        default_beam_fwhm_arcsec: f64,
    },
    /// Summarize point/non-spatial dataset overlap readiness against LoTSS.
    CatalogMatrix {
        #[arg(long, value_enum, default_value_t = ReleaseArg::Dr3)]
        release: ReleaseArg,

        #[arg(long)]
        lotss: Option<PathBuf>,

        #[arg(long, default_value_t = 3.0)]
        match_radius_arcsec: f64,

        #[arg(long)]
        chunk_rows: Option<usize>,

        #[arg(long)]
        workers: Option<usize>,

        #[arg(long, default_value = "data/external/atnf_pulsars.csv")]
        atnf: PathBuf,

        #[arg(long, default_value = "data/external/mcgill_magnetars.csv")]
        mcgill: PathBuf,

        #[arg(long, default_value = "data/external/chime_frb_cat2.csv")]
        chime: PathBuf,

        #[arg(long, default_value = "data/external/sdss_dr18_quasars.csv")]
        sdss: PathBuf,

        #[arg(long, default_value = "data/external/gaia_dr3_nearby.csv")]
        gaia: PathBuf,

        #[arg(long, default_value = "data/external/jwst_public_observations.csv")]
        jwst: PathBuf,

        #[arg(long, default_value = "data/external/hst_public_observations.csv")]
        hst: PathBuf,

        #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
        manga_selection: PathBuf,

        #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
        manga_drpall: PathBuf,

        #[arg(long, default_value = "data/external/manga/manga_lotss_xmatch_dr3.csv")]
        manga_lotss: PathBuf,

        #[arg(long, default_value = "data/external/things/table1.dat")]
        things_table1: PathBuf,

        #[arg(long)]
        things_rotcurves: Option<PathBuf>,

        #[arg(long, default_value = "reports/euclid_lotss_xmatch_dr3.toml")]
        euclid_lotss_report: PathBuf,

        #[arg(long)]
        report: Option<PathBuf>,
    },
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ReleaseArg {
    Dr1,
    Dr2,
    Dr3,
}

#[cfg(feature = "euclid-catalog")]
impl ReleaseArg {
    fn label(self) -> &'static str {
        match self {
            Self::Dr1 => "dr1",
            Self::Dr2 => "dr2",
            Self::Dr3 => "dr3",
        }
    }

    fn lotss_release(self) -> LoTSSRelease {
        match self {
            Self::Dr1 => LoTSSRelease::DR1,
            Self::Dr2 => LoTSSRelease::DR2,
            Self::Dr3 => LoTSSRelease::DR3,
        }
    }
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Clone)]
struct MangaSkyTarget {
    plateifu: String,
    mangaid: String,
    ra_deg: f64,
    dec_deg: f64,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Clone)]
struct SelectedMangaSample {
    targets: Vec<MangaSkyTarget>,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Clone)]
struct EuclidSelection {
    records: Vec<EuclidMorphologyRecord>,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Clone)]
struct LotssBestMatch {
    separation_arcsec: f64,
    source_name: String,
    lotss_ra_deg: f64,
    lotss_dec_deg: f64,
    lotss_flux_mjy: f32,
    lotss_spectral_index: Option<f32>,
    lotss_structure_code: char,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Clone)]
struct MangaBestMatch {
    separation_arcsec: f64,
    plateifu: String,
    mangaid: String,
    manga_ra_deg: f64,
    manga_dec_deg: f64,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct EuclidLotssRow {
    euclid_object_id: String,
    euclid_ra_deg: String,
    euclid_dec_deg: String,
    euclid_featured_fraction: String,
    euclid_spiral_fraction: String,
    euclid_face_on_fraction: String,
    euclid_non_merging_fraction: String,
    lotss_release: String,
    lotss_source_name: String,
    lotss_ra_deg: String,
    lotss_dec_deg: String,
    lotss_separation_arcsec: String,
    lotss_flux_mjy: String,
    lotss_spectral_index: String,
    lotss_structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsMangaRow {
    galaxy_name: String,
    galaxy_ra_deg: String,
    galaxy_dec_deg: String,
    manga_plateifu: String,
    mangaid: String,
    manga_ra_deg: String,
    manga_dec_deg: String,
    separation_arcsec: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsMangaReport {
    generated_at_utc: String,
    things_table1_path: String,
    things_rotcurves_path: Option<String>,
    manga_selection_path: String,
    manga_drpall_path: String,
    output_csv_path: String,
    input_things_count: usize,
    matched_things_count: usize,
    match_radius_arcsec: f64,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsLotssRow {
    galaxy_name: String,
    galaxy_ra_deg: String,
    galaxy_dec_deg: String,
    lotss_release: String,
    lotss_source_name: String,
    lotss_ra_deg: String,
    lotss_dec_deg: String,
    separation_arcsec: String,
    flux_mjy: String,
    spectral_index: String,
    structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsLotssReport {
    generated_at_utc: String,
    things_table1_path: String,
    things_rotcurves_path: Option<String>,
    lotss_path: String,
    output_csv_path: String,
    release: String,
    input_things_count: usize,
    matched_things_count: usize,
    detection_fraction: f64,
    match_radius_arcsec: f64,
    chunk_rows: usize,
    execution: LotssExecutionReport,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsMangaLotssRow {
    galaxy_name: String,
    galaxy_ra_deg: String,
    galaxy_dec_deg: String,
    manga_plateifu: String,
    mangaid: String,
    manga_separation_arcsec: String,
    lotss_release: String,
    lotss_source_name: String,
    lotss_separation_arcsec: String,
    lotss_flux_mjy: String,
    lotss_structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsMangaLotssReport {
    generated_at_utc: String,
    things_table1_path: String,
    things_rotcurves_path: Option<String>,
    manga_selection_path: String,
    manga_drpall_path: String,
    lotss_path: String,
    output_csv_path: String,
    release: String,
    input_things_count: usize,
    manga_overlap_count: usize,
    lotss_overlap_count: usize,
    triple_overlap_count: usize,
    match_radius_arcsec: f64,
    chunk_rows: usize,
    execution: LotssExecutionReport,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsMetadataRow {
    name: String,
    ra_deg: String,
    dec_deg: String,
    distance_mpc: String,
    inclination_deg: String,
    pa_deg: String,
    beam_fwhm_arcsec: String,
    channel_width_km_s: String,
    v_sys_km_s: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct ThingsMetadataReport {
    generated_at_utc: String,
    things_table1_path: String,
    things_table4_path: String,
    output_csv_path: String,
    default_beam_fwhm_arcsec: f64,
    galaxy_count: usize,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct AtnfLotssRow {
    pulsar_name: String,
    atnf_ra_deg: String,
    atnf_dec_deg: String,
    gl_deg: String,
    gb_deg: String,
    dm_pc_cm3: String,
    lotss_release: String,
    lotss_source_name: String,
    lotss_ra_deg: String,
    lotss_dec_deg: String,
    separation_arcsec: String,
    flux_mjy: String,
    spectral_index: String,
    structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct McgillLotssRow {
    magnetar_name: String,
    mcgill_ra_deg: String,
    mcgill_dec_deg: String,
    period_s: String,
    b_dipole_1e14g: String,
    lotss_release: String,
    lotss_source_name: String,
    lotss_ra_deg: String,
    lotss_dec_deg: String,
    separation_arcsec: String,
    flux_mjy: String,
    spectral_index: String,
    structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct PointLotssReport {
    generated_at_utc: String,
    catalog: String,
    input_path: String,
    lotss_path: String,
    output_csv_path: String,
    release: String,
    input_count: usize,
    matched_count: usize,
    detection_fraction: f64,
    match_radius_arcsec: f64,
    chunk_rows: usize,
    execution: LotssExecutionReport,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct EuclidLotssReport {
    generated_at_utc: String,
    release: String,
    euclid_morphology_path: String,
    lotss_path: String,
    output_csv_path: String,
    selected_euclid_count: usize,
    matched_euclid_count: usize,
    detection_fraction: f64,
    lotss_source_count_scanned: usize,
    chunk_rows: usize,
    match_radius_arcsec: f64,
    no_morphology_cuts: bool,
    featured_min: f32,
    spiral_min: f32,
    face_on_min: f32,
    non_merge_min: f32,
    execution: LotssExecutionReport,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize, Clone)]
struct LotssExecutionReport {
    mode: String,
    worker_count: usize,
    chunk_rows: usize,
    pinned_core_ids: Vec<usize>,
    l3_cache_bytes: usize,
    l3_safe_working_set_bytes: usize,
    avx2_detected: bool,
    fma_detected: bool,
    simd_lane_f64: usize,
    x87_confirmation_used: bool,
    scan_wall_seconds: f64,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Deserialize)]
struct EuclidMangaCsvRow {
    plateifu: String,
    mangaid: String,
    euclid_object_id: String,
    sep_arcsec: String,
    euclid_featured_fraction: String,
    euclid_spiral_fraction: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Deserialize)]
struct MangaLotssCsvRow {
    plateifu: String,
    lotss_release: String,
    lotss_detected: String,
    lotss_separation_arcsec: String,
    lotss_source_name: String,
    lotss_flux_mjy: String,
    lotss_structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct EuclidMangaLotssRow {
    plateifu: String,
    mangaid: String,
    euclid_object_id: String,
    euclid_sep_arcsec: String,
    euclid_featured_fraction: String,
    euclid_spiral_fraction: String,
    lotss_release: String,
    lotss_detected: String,
    lotss_source_name: String,
    lotss_separation_arcsec: String,
    lotss_flux_mjy: String,
    lotss_structure_code: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct EuclidMangaLotssReport {
    generated_at_utc: String,
    euclid_manga_path: String,
    manga_lotss_path: String,
    output_csv_path: String,
    euclid_manga_row_count: usize,
    manga_lotss_row_count: usize,
    shared_plateifu_count: usize,
    radio_loud_triple_count: usize,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct CatalogMatrixEntry {
    catalog: String,
    modality: String,
    path: String,
    row_count: Option<usize>,
    exact_overlap_with_lotss: Option<usize>,
    status: String,
    note: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct CatalogMatrixReport {
    generated_at_utc: String,
    release: String,
    lotss_path: String,
    match_radius_arcsec: f64,
    execution: CatalogMatrixExecutionReport,
    entries: Vec<CatalogMatrixEntry>,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct CatalogMatrixExecutionReport {
    mode: String,
    worker_count: usize,
    chunk_rows: usize,
    pinned_core_ids: Vec<usize>,
    l3_cache_bytes: usize,
    l3_safe_working_set_bytes: usize,
    avx2_detected: bool,
    fma_detected: bool,
    simd_lane_f64: usize,
    single_pass_catalog_count: usize,
    scan_wall_seconds: f64,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct DesiDr2ReferenceMeasurement {
    z_or_label: String,
    measurement: String,
    uncertainty: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct DesiDr2ReferenceReport {
    generated_at_utc: String,
    provenance_note: String,
    source: String,
    measurement_count: usize,
    measurements: Vec<DesiDr2ReferenceMeasurement>,
}

#[cfg(feature = "euclid-catalog")]
fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::EuclidLotss {
            euclid_morphology,
            release,
            lotss,
            output,
            report,
            match_radius_arcsec,
            chunk_rows,
            workers,
            featured_min,
            spiral_min,
            face_on_min,
            non_merge_min,
            no_morphology_cuts,
        } => cmd_euclid_lotss(EuclidLotssArgs {
            euclid_morphology,
            release,
            lotss: lotss.unwrap_or_else(|| default_lotss_path(release)),
            output: output
                .unwrap_or_else(|| default_euclid_lotss_csv_for_mode(release, no_morphology_cuts)),
            report: report.unwrap_or_else(|| {
                default_euclid_lotss_report_for_mode(release, no_morphology_cuts)
            }),
            match_radius_arcsec,
            chunk_rows,
            workers,
            featured_min,
            spiral_min,
            face_on_min,
            non_merge_min,
            no_morphology_cuts,
        }),
        Command::EuclidMangaLotss {
            euclid_manga,
            manga_lotss,
            output,
            report,
        } => cmd_euclid_manga_lotss(
            euclid_manga,
            manga_lotss,
            output.unwrap_or_else(default_euclid_manga_lotss_csv),
            report.unwrap_or_else(default_euclid_manga_lotss_report),
        ),
        Command::AtnfLotss {
            atnf,
            release,
            lotss,
            output,
            report,
            match_radius_arcsec,
            chunk_rows,
            workers,
        } => cmd_atnf_lotss(PointCatalogLotssArgs {
            input: atnf,
            release,
            lotss: lotss.unwrap_or_else(|| default_lotss_path(release)),
            output: output.unwrap_or_else(|| default_atnf_lotss_csv(release)),
            report: report.unwrap_or_else(|| default_atnf_lotss_report(release)),
            match_radius_arcsec,
            chunk_rows,
            workers,
        }),
        Command::McgillLotss {
            mcgill,
            release,
            lotss,
            output,
            report,
            match_radius_arcsec,
            chunk_rows,
            workers,
        } => cmd_mcgill_lotss(PointCatalogLotssArgs {
            input: mcgill,
            release,
            lotss: lotss.unwrap_or_else(|| default_lotss_path(release)),
            output: output.unwrap_or_else(|| default_mcgill_lotss_csv(release)),
            report: report.unwrap_or_else(|| default_mcgill_lotss_report(release)),
            match_radius_arcsec,
            chunk_rows,
            workers,
        }),
        Command::ThingsManga {
            things_table1,
            things_rotcurves,
            manga_selection,
            manga_drpall,
            output,
            report,
            match_radius_arcsec,
        } => cmd_things_manga(
            things_table1,
            things_rotcurves,
            manga_selection,
            manga_drpall,
            output.unwrap_or_else(default_things_manga_csv),
            report.unwrap_or_else(default_things_manga_report),
            match_radius_arcsec,
        ),
        Command::ThingsLotss {
            things_table1,
            things_rotcurves,
            release,
            lotss,
            output,
            report,
            match_radius_arcsec,
            chunk_rows,
            workers,
        } => cmd_things_lotss(ThingsLotssArgs {
            things_table1,
            things_rotcurves,
            release,
            lotss: lotss.unwrap_or_else(|| default_lotss_path(release)),
            output: output.unwrap_or_else(|| default_things_lotss_csv(release)),
            report: report.unwrap_or_else(|| default_things_lotss_report(release)),
            match_radius_arcsec,
            chunk_rows,
            workers,
        }),
        Command::ThingsMangaLotss {
            things_table1,
            things_rotcurves,
            manga_selection,
            manga_drpall,
            release,
            lotss,
            output,
            report,
            match_radius_arcsec,
            chunk_rows,
            workers,
        } => cmd_things_manga_lotss(ThingsMangaLotssArgs {
            things_table1,
            things_rotcurves,
            manga_selection,
            manga_drpall,
            release,
            lotss: lotss.unwrap_or_else(|| default_lotss_path(release)),
            output: output.unwrap_or_else(|| default_things_manga_lotss_csv(release)),
            report: report.unwrap_or_else(|| default_things_manga_lotss_report(release)),
            match_radius_arcsec,
            chunk_rows,
            workers,
        }),
        Command::ThingsMetadata {
            things_table1,
            things_table4,
            output,
            report,
            default_beam_fwhm_arcsec,
        } => cmd_things_metadata(
            things_table1,
            things_table4,
            output.unwrap_or_else(default_things_metadata_csv),
            report.unwrap_or_else(default_things_metadata_report),
            default_beam_fwhm_arcsec,
        ),
        Command::CatalogMatrix {
            release,
            lotss,
            match_radius_arcsec,
            chunk_rows,
            workers,
            atnf,
            mcgill,
            chime,
            sdss,
            gaia,
            jwst,
            hst,
            manga_selection,
            manga_drpall,
            manga_lotss,
            things_table1,
            things_rotcurves,
            euclid_lotss_report,
            report,
        } => cmd_catalog_matrix(CatalogMatrixArgs {
            release,
            lotss: lotss.unwrap_or_else(|| default_lotss_path(release)),
            match_radius_arcsec,
            chunk_rows,
            workers,
            atnf,
            mcgill,
            chime,
            sdss,
            gaia,
            jwst,
            hst,
            manga_selection,
            manga_drpall,
            manga_lotss,
            things_table1,
            things_rotcurves,
            euclid_lotss_report,
            report: report.unwrap_or_else(|| default_catalog_matrix_report(release)),
        }),
    }
}

#[cfg(not(feature = "euclid-catalog"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "survey-crossmatch requires the 'euclid-catalog' feature.\n\
         Run: cargo run -p gororoba_cli_physics --features euclid-catalog --bin survey-crossmatch -- --help"
    )
}

#[cfg(feature = "euclid-catalog")]
struct EuclidLotssArgs {
    euclid_morphology: PathBuf,
    release: ReleaseArg,
    lotss: PathBuf,
    output: PathBuf,
    report: PathBuf,
    match_radius_arcsec: f64,
    chunk_rows: Option<usize>,
    workers: Option<usize>,
    featured_min: f32,
    spiral_min: f32,
    face_on_min: f32,
    non_merge_min: f32,
    no_morphology_cuts: bool,
}

#[cfg(feature = "euclid-catalog")]
struct CatalogMatrixArgs {
    release: ReleaseArg,
    lotss: PathBuf,
    match_radius_arcsec: f64,
    chunk_rows: Option<usize>,
    workers: Option<usize>,
    atnf: PathBuf,
    mcgill: PathBuf,
    chime: PathBuf,
    sdss: PathBuf,
    gaia: PathBuf,
    jwst: PathBuf,
    hst: PathBuf,
    manga_selection: PathBuf,
    manga_drpall: PathBuf,
    manga_lotss: PathBuf,
    things_table1: PathBuf,
    things_rotcurves: Option<PathBuf>,
    euclid_lotss_report: PathBuf,
    report: PathBuf,
}

#[cfg(feature = "euclid-catalog")]
struct PointCatalogLotssArgs {
    input: PathBuf,
    release: ReleaseArg,
    lotss: PathBuf,
    output: PathBuf,
    report: PathBuf,
    match_radius_arcsec: f64,
    chunk_rows: Option<usize>,
    workers: Option<usize>,
}

#[cfg(feature = "euclid-catalog")]
struct ThingsLotssArgs {
    things_table1: PathBuf,
    things_rotcurves: Option<PathBuf>,
    release: ReleaseArg,
    lotss: PathBuf,
    output: PathBuf,
    report: PathBuf,
    match_radius_arcsec: f64,
    chunk_rows: Option<usize>,
    workers: Option<usize>,
}

#[cfg(feature = "euclid-catalog")]
struct ThingsMangaLotssArgs {
    things_table1: PathBuf,
    things_rotcurves: Option<PathBuf>,
    manga_selection: PathBuf,
    manga_drpall: PathBuf,
    release: ReleaseArg,
    lotss: PathBuf,
    output: PathBuf,
    report: PathBuf,
    match_radius_arcsec: f64,
    chunk_rows: Option<usize>,
    workers: Option<usize>,
}

#[cfg(feature = "euclid-catalog")]
fn cmd_euclid_lotss(args: EuclidLotssArgs) -> Result<()> {
    if args.match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }
    let euclid = select_euclid_records(
        &args.euclid_morphology,
        args.featured_min,
        args.spiral_min,
        args.face_on_min,
        args.non_merge_min,
        args.no_morphology_cuts,
        args.match_radius_arcsec,
    )?;
    if euclid.records.is_empty() {
        bail!("No Euclid records remain after selection");
    }

    let best_match_result = find_best_lotss_matches(
        &args.lotss,
        args.release,
        args.match_radius_arcsec,
        args.chunk_rows,
        args.workers,
        &euclid
            .records
            .iter()
            .map(|row| SkyPoint {
                id: row.object_id.clone(),
                ra_deg: row.ra_deg,
                dec_deg: row.dec_deg,
            })
            .collect::<Vec<_>>(),
    )?;
    let best_matches = best_match_result.matches;
    let scanned_sources = best_match_result.scanned_source_count;

    let mut rows = Vec::new();
    for (index, record) in euclid.records.iter().enumerate() {
        let Some(matched) = &best_matches[index] else {
            continue;
        };
        rows.push(EuclidLotssRow {
            euclid_object_id: record.object_id.clone(),
            euclid_ra_deg: format!("{:.6}", record.ra_deg),
            euclid_dec_deg: format!("{:.6}", record.dec_deg),
            euclid_featured_fraction: format!("{:.6}", record.featured_fraction),
            euclid_spiral_fraction: format!("{:.6}", record.spiral_fraction),
            euclid_face_on_fraction: format!("{:.6}", record.face_on_fraction),
            euclid_non_merging_fraction: format!("{:.6}", record.non_merging_fraction),
            lotss_release: args.release.label().to_ascii_uppercase(),
            lotss_source_name: matched.source_name.clone(),
            lotss_ra_deg: format!("{:.6}", matched.lotss_ra_deg),
            lotss_dec_deg: format!("{:.6}", matched.lotss_dec_deg),
            lotss_separation_arcsec: format!("{:.4}", matched.separation_arcsec),
            lotss_flux_mjy: format!("{:.4}", matched.lotss_flux_mjy),
            lotss_spectral_index: matched
                .lotss_spectral_index
                .map(|value| format!("{:.4}", value))
                .unwrap_or_default(),
            lotss_structure_code: matched.lotss_structure_code.to_string(),
        });
    }
    rows.sort_by(|left, right| left.euclid_object_id.cmp(&right.euclid_object_id));
    write_csv(&args.output, &rows)?;

    let report = EuclidLotssReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        release: args.release.label().to_ascii_uppercase(),
        euclid_morphology_path: args.euclid_morphology.display().to_string(),
        lotss_path: args.lotss.display().to_string(),
        output_csv_path: args.output.display().to_string(),
        selected_euclid_count: euclid.records.len(),
        matched_euclid_count: rows.len(),
        detection_fraction: fraction(rows.len(), euclid.records.len()),
        lotss_source_count_scanned: scanned_sources,
        chunk_rows: best_match_result.execution.chunk_rows,
        match_radius_arcsec: args.match_radius_arcsec,
        no_morphology_cuts: args.no_morphology_cuts,
        featured_min: args.featured_min,
        spiral_min: args.spiral_min,
        face_on_min: args.face_on_min,
        non_merge_min: args.non_merge_min,
        execution: best_match_result.execution.clone(),
    };
    write_toml_report(&args.report, &report)?;

    println!("Selected Euclid rows: {}", report.selected_euclid_count);
    println!("Matched Euclid rows:  {}", report.matched_euclid_count);
    println!(
        "LoTSS scanned rows:   {}",
        report.lotss_source_count_scanned
    );
    println!(
        "Execution:            mode={} workers={} chunk_rows={} pinned={:?} scan_wall={:.2}s",
        report.execution.mode,
        report.execution.worker_count,
        report.execution.chunk_rows,
        report.execution.pinned_core_ids,
        report.execution.scan_wall_seconds
    );
    println!("CSV:                  {}", args.output.display());
    println!("Report:               {}", args.report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_euclid_manga_lotss(
    euclid_manga: PathBuf,
    manga_lotss: PathBuf,
    output: PathBuf,
    report: PathBuf,
) -> Result<()> {
    let euclid_rows = load_csv_rows::<EuclidMangaCsvRow>(&euclid_manga)?;
    let lotss_rows = load_csv_rows::<MangaLotssCsvRow>(&manga_lotss)?;
    let mut lotss_by_plateifu = HashMap::with_capacity(lotss_rows.len());
    for row in lotss_rows {
        lotss_by_plateifu.insert(row.plateifu.clone(), row);
    }

    let mut joined = Vec::new();
    let mut radio_loud = 0usize;
    for row in &euclid_rows {
        let Some(lotss_row) = lotss_by_plateifu.get(&row.plateifu) else {
            continue;
        };
        if lotss_row.lotss_detected.trim() == "1" {
            radio_loud += 1;
        }
        joined.push(EuclidMangaLotssRow {
            plateifu: row.plateifu.clone(),
            mangaid: row.mangaid.clone(),
            euclid_object_id: row.euclid_object_id.clone(),
            euclid_sep_arcsec: row.sep_arcsec.clone(),
            euclid_featured_fraction: row.euclid_featured_fraction.clone(),
            euclid_spiral_fraction: row.euclid_spiral_fraction.clone(),
            lotss_release: lotss_row.lotss_release.clone(),
            lotss_detected: lotss_row.lotss_detected.clone(),
            lotss_source_name: lotss_row.lotss_source_name.clone(),
            lotss_separation_arcsec: lotss_row.lotss_separation_arcsec.clone(),
            lotss_flux_mjy: lotss_row.lotss_flux_mjy.clone(),
            lotss_structure_code: lotss_row.lotss_structure_code.clone(),
        });
    }
    joined.sort_by(|left, right| left.plateifu.cmp(&right.plateifu));
    write_csv(&output, &joined)?;

    let report_model = EuclidMangaLotssReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        euclid_manga_path: euclid_manga.display().to_string(),
        manga_lotss_path: manga_lotss.display().to_string(),
        output_csv_path: output.display().to_string(),
        euclid_manga_row_count: euclid_rows.len(),
        manga_lotss_row_count: lotss_by_plateifu.len(),
        shared_plateifu_count: joined.len(),
        radio_loud_triple_count: radio_loud,
    };
    write_toml_report(&report, &report_model)?;

    println!("Shared plateifus: {}", report_model.shared_plateifu_count);
    println!("Radio-loud rows:  {}", report_model.radio_loud_triple_count);
    println!("CSV:              {}", output.display());
    println!("Report:           {}", report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_atnf_lotss(args: PointCatalogLotssArgs) -> Result<()> {
    if args.match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }
    let pulsars = parse_atnf_csv(&args.input).map_err(anyhow::Error::msg)?;
    let points = pulsars
        .iter()
        .filter(|row| row.ra.is_finite() && row.dec.is_finite())
        .map(|row| SkyPoint {
            id: row.name.clone(),
            ra_deg: row.ra,
            dec_deg: row.dec,
        })
        .collect::<Vec<_>>();
    let best_match_result = find_best_lotss_matches(
        &args.lotss,
        args.release,
        args.match_radius_arcsec,
        args.chunk_rows,
        args.workers,
        &points,
    )?;
    let lotss_matches = best_match_result.matches;

    let mut rows = Vec::new();
    let mut point_index = 0usize;
    for pulsar in &pulsars {
        if !pulsar.ra.is_finite() || !pulsar.dec.is_finite() {
            continue;
        }
        let matched = lotss_matches
            .get(point_index)
            .and_then(|entry| entry.as_ref())
            .cloned();
        point_index += 1;
        let Some(matched) = matched else {
            continue;
        };
        rows.push(AtnfLotssRow {
            pulsar_name: pulsar.name.clone(),
            atnf_ra_deg: format!("{:.6}", pulsar.ra),
            atnf_dec_deg: format!("{:.6}", pulsar.dec),
            gl_deg: format!("{:.6}", pulsar.gl),
            gb_deg: format!("{:.6}", pulsar.gb),
            dm_pc_cm3: format!("{:.6}", pulsar.dm),
            lotss_release: args.release.label().to_ascii_uppercase(),
            lotss_source_name: matched.source_name,
            lotss_ra_deg: format!("{:.6}", matched.lotss_ra_deg),
            lotss_dec_deg: format!("{:.6}", matched.lotss_dec_deg),
            separation_arcsec: format!("{:.4}", matched.separation_arcsec),
            flux_mjy: format!("{:.4}", matched.lotss_flux_mjy),
            spectral_index: matched
                .lotss_spectral_index
                .map(|value| format!("{:.4}", value))
                .unwrap_or_default(),
            structure_code: matched.lotss_structure_code.to_string(),
        });
    }
    rows.sort_by(|left, right| left.pulsar_name.cmp(&right.pulsar_name));
    write_csv(&args.output, &rows)?;

    let input_count = points.len();
    let report_model = PointLotssReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        catalog: "ATNF Pulsars".to_string(),
        input_path: args.input.display().to_string(),
        lotss_path: args.lotss.display().to_string(),
        output_csv_path: args.output.display().to_string(),
        release: args.release.label().to_ascii_uppercase(),
        input_count,
        matched_count: rows.len(),
        detection_fraction: fraction(rows.len(), input_count),
        match_radius_arcsec: args.match_radius_arcsec,
        chunk_rows: best_match_result.execution.chunk_rows,
        execution: best_match_result.execution.clone(),
    };
    write_toml_report(&args.report, &report_model)?;
    println!("ATNF pulsars: {}", report_model.input_count);
    println!("LoTSS matches: {}", report_model.matched_count);
    println!(
        "Execution:     mode={} workers={} chunk_rows={} pinned={:?} scan_wall={:.2}s",
        report_model.execution.mode,
        report_model.execution.worker_count,
        report_model.execution.chunk_rows,
        report_model.execution.pinned_core_ids,
        report_model.execution.scan_wall_seconds
    );
    println!("CSV:           {}", args.output.display());
    println!("Report:        {}", args.report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_mcgill_lotss(args: PointCatalogLotssArgs) -> Result<()> {
    if args.match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }
    let magnetars = parse_mcgill_csv(&args.input).map_err(anyhow::Error::msg)?;
    let points = magnetars
        .iter()
        .filter(|row| row.ra.is_finite() && row.dec.is_finite())
        .map(|row| SkyPoint {
            id: row.name.clone(),
            ra_deg: row.ra,
            dec_deg: row.dec,
        })
        .collect::<Vec<_>>();
    let best_match_result = find_best_lotss_matches(
        &args.lotss,
        args.release,
        args.match_radius_arcsec,
        args.chunk_rows,
        args.workers,
        &points,
    )?;
    let lotss_matches = best_match_result.matches;

    let mut rows = Vec::new();
    let mut point_index = 0usize;
    for magnetar in &magnetars {
        if !magnetar.ra.is_finite() || !magnetar.dec.is_finite() {
            continue;
        }
        let matched = lotss_matches
            .get(point_index)
            .and_then(|entry| entry.as_ref())
            .cloned();
        point_index += 1;
        let Some(matched) = matched else {
            continue;
        };
        rows.push(McgillLotssRow {
            magnetar_name: magnetar.name.clone(),
            mcgill_ra_deg: format!("{:.6}", magnetar.ra),
            mcgill_dec_deg: format!("{:.6}", magnetar.dec),
            period_s: format!("{:.9}", magnetar.period),
            b_dipole_1e14g: format!("{:.6}", magnetar.b_dipole),
            lotss_release: args.release.label().to_ascii_uppercase(),
            lotss_source_name: matched.source_name,
            lotss_ra_deg: format!("{:.6}", matched.lotss_ra_deg),
            lotss_dec_deg: format!("{:.6}", matched.lotss_dec_deg),
            separation_arcsec: format!("{:.4}", matched.separation_arcsec),
            flux_mjy: format!("{:.4}", matched.lotss_flux_mjy),
            spectral_index: matched
                .lotss_spectral_index
                .map(|value| format!("{:.4}", value))
                .unwrap_or_default(),
            structure_code: matched.lotss_structure_code.to_string(),
        });
    }
    rows.sort_by(|left, right| left.magnetar_name.cmp(&right.magnetar_name));
    write_csv(&args.output, &rows)?;

    let input_count = points.len();
    let report_model = PointLotssReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        catalog: "McGill Magnetars".to_string(),
        input_path: args.input.display().to_string(),
        lotss_path: args.lotss.display().to_string(),
        output_csv_path: args.output.display().to_string(),
        release: args.release.label().to_ascii_uppercase(),
        input_count,
        matched_count: rows.len(),
        detection_fraction: fraction(rows.len(), input_count),
        match_radius_arcsec: args.match_radius_arcsec,
        chunk_rows: best_match_result.execution.chunk_rows,
        execution: best_match_result.execution.clone(),
    };
    write_toml_report(&args.report, &report_model)?;
    println!("McGill magnetars: {}", report_model.input_count);
    println!("LoTSS matches:    {}", report_model.matched_count);
    println!(
        "Execution:        mode={} workers={} chunk_rows={} pinned={:?} scan_wall={:.2}s",
        report_model.execution.mode,
        report_model.execution.worker_count,
        report_model.execution.chunk_rows,
        report_model.execution.pinned_core_ids,
        report_model.execution.scan_wall_seconds
    );
    println!("CSV:              {}", args.output.display());
    println!("Report:           {}", args.report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_things_manga(
    things_table1: PathBuf,
    things_rotcurves: Option<PathBuf>,
    manga_selection: PathBuf,
    manga_drpall: PathBuf,
    output: PathBuf,
    report: PathBuf,
    match_radius_arcsec: f64,
) -> Result<()> {
    if match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }
    let things = load_things_galaxies_filtered(&things_table1, things_rotcurves.as_deref())?;
    let thing_points = things_to_points(&things);
    let manga_sample = load_selected_manga_targets(&manga_selection, &manga_drpall)?;
    let manga_matches = match_points_to_manga(&thing_points, &manga_sample, match_radius_arcsec);

    let mut rows = Vec::new();
    for (index, galaxy) in things.iter().enumerate() {
        let Some(matched) = &manga_matches[index] else {
            continue;
        };
        rows.push(ThingsMangaRow {
            galaxy_name: galaxy.name.clone(),
            galaxy_ra_deg: format!("{:.6}", galaxy.ra_hours * 15.0),
            galaxy_dec_deg: format!("{:.6}", galaxy.dec_deg),
            manga_plateifu: matched.plateifu.clone(),
            mangaid: matched.mangaid.clone(),
            manga_ra_deg: format!("{:.6}", matched.manga_ra_deg),
            manga_dec_deg: format!("{:.6}", matched.manga_dec_deg),
            separation_arcsec: format!("{:.4}", matched.separation_arcsec),
        });
    }
    rows.sort_by(|left, right| left.galaxy_name.cmp(&right.galaxy_name));
    write_csv(&output, &rows)?;

    let report_model = ThingsMangaReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        things_table1_path: things_table1.display().to_string(),
        things_rotcurves_path: things_rotcurves.map(|path| path.display().to_string()),
        manga_selection_path: manga_selection.display().to_string(),
        manga_drpall_path: manga_drpall.display().to_string(),
        output_csv_path: output.display().to_string(),
        input_things_count: things.len(),
        matched_things_count: rows.len(),
        match_radius_arcsec,
    };
    write_toml_report(&report, &report_model)?;
    println!("THINGS galaxies: {}", report_model.input_things_count);
    println!("MaNGA matches:   {}", report_model.matched_things_count);
    println!("CSV:             {}", output.display());
    println!("Report:          {}", report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_things_lotss(args: ThingsLotssArgs) -> Result<()> {
    if args.match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }
    let things =
        load_things_galaxies_filtered(&args.things_table1, args.things_rotcurves.as_deref())?;
    let thing_points = things_to_points(&things);
    let best_match_result = find_best_lotss_matches(
        &args.lotss,
        args.release,
        args.match_radius_arcsec,
        args.chunk_rows,
        args.workers,
        &thing_points,
    )?;
    let lotss_matches = best_match_result.matches;

    let mut rows = Vec::new();
    for (index, galaxy) in things.iter().enumerate() {
        let Some(matched) = &lotss_matches[index] else {
            continue;
        };
        rows.push(ThingsLotssRow {
            galaxy_name: galaxy.name.clone(),
            galaxy_ra_deg: format!("{:.6}", galaxy.ra_hours * 15.0),
            galaxy_dec_deg: format!("{:.6}", galaxy.dec_deg),
            lotss_release: args.release.label().to_ascii_uppercase(),
            lotss_source_name: matched.source_name.clone(),
            lotss_ra_deg: format!("{:.6}", matched.lotss_ra_deg),
            lotss_dec_deg: format!("{:.6}", matched.lotss_dec_deg),
            separation_arcsec: format!("{:.4}", matched.separation_arcsec),
            flux_mjy: format!("{:.4}", matched.lotss_flux_mjy),
            spectral_index: matched
                .lotss_spectral_index
                .map(|value| format!("{:.4}", value))
                .unwrap_or_default(),
            structure_code: matched.lotss_structure_code.to_string(),
        });
    }
    rows.sort_by(|left, right| left.galaxy_name.cmp(&right.galaxy_name));
    write_csv(&args.output, &rows)?;

    let report_model = ThingsLotssReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        things_table1_path: args.things_table1.display().to_string(),
        things_rotcurves_path: args
            .things_rotcurves
            .as_ref()
            .map(|path| path.display().to_string()),
        lotss_path: args.lotss.display().to_string(),
        output_csv_path: args.output.display().to_string(),
        release: args.release.label().to_ascii_uppercase(),
        input_things_count: things.len(),
        matched_things_count: rows.len(),
        detection_fraction: fraction(rows.len(), things.len()),
        match_radius_arcsec: args.match_radius_arcsec,
        chunk_rows: best_match_result.execution.chunk_rows,
        execution: best_match_result.execution.clone(),
    };
    write_toml_report(&args.report, &report_model)?;
    println!("THINGS galaxies: {}", report_model.input_things_count);
    println!("LoTSS matches:   {}", report_model.matched_things_count);
    println!(
        "Execution:       mode={} workers={} chunk_rows={} pinned={:?} scan_wall={:.2}s",
        report_model.execution.mode,
        report_model.execution.worker_count,
        report_model.execution.chunk_rows,
        report_model.execution.pinned_core_ids,
        report_model.execution.scan_wall_seconds
    );
    println!("CSV:             {}", args.output.display());
    println!("Report:          {}", args.report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_things_manga_lotss(args: ThingsMangaLotssArgs) -> Result<()> {
    if args.match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }
    let things =
        load_things_galaxies_filtered(&args.things_table1, args.things_rotcurves.as_deref())?;
    let thing_points = things_to_points(&things);
    let manga_sample = load_selected_manga_targets(&args.manga_selection, &args.manga_drpall)?;
    let manga_matches =
        match_points_to_manga(&thing_points, &manga_sample, args.match_radius_arcsec);
    let best_match_result = find_best_lotss_matches(
        &args.lotss,
        args.release,
        args.match_radius_arcsec,
        args.chunk_rows,
        args.workers,
        &thing_points,
    )?;
    let lotss_matches = best_match_result.matches;

    let manga_overlap_count = manga_matches.iter().filter(|entry| entry.is_some()).count();
    let lotss_overlap_count = lotss_matches.iter().filter(|entry| entry.is_some()).count();

    let mut rows = Vec::new();
    for (index, galaxy) in things.iter().enumerate() {
        let (Some(manga_matched), Some(lotss_matched)) =
            (&manga_matches[index], &lotss_matches[index])
        else {
            continue;
        };
        rows.push(ThingsMangaLotssRow {
            galaxy_name: galaxy.name.clone(),
            galaxy_ra_deg: format!("{:.6}", galaxy.ra_hours * 15.0),
            galaxy_dec_deg: format!("{:.6}", galaxy.dec_deg),
            manga_plateifu: manga_matched.plateifu.clone(),
            mangaid: manga_matched.mangaid.clone(),
            manga_separation_arcsec: format!("{:.4}", manga_matched.separation_arcsec),
            lotss_release: args.release.label().to_ascii_uppercase(),
            lotss_source_name: lotss_matched.source_name.clone(),
            lotss_separation_arcsec: format!("{:.4}", lotss_matched.separation_arcsec),
            lotss_flux_mjy: format!("{:.4}", lotss_matched.lotss_flux_mjy),
            lotss_structure_code: lotss_matched.lotss_structure_code.to_string(),
        });
    }
    rows.sort_by(|left, right| left.galaxy_name.cmp(&right.galaxy_name));
    write_csv(&args.output, &rows)?;

    let report_model = ThingsMangaLotssReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        things_table1_path: args.things_table1.display().to_string(),
        things_rotcurves_path: args
            .things_rotcurves
            .as_ref()
            .map(|path| path.display().to_string()),
        manga_selection_path: args.manga_selection.display().to_string(),
        manga_drpall_path: args.manga_drpall.display().to_string(),
        lotss_path: args.lotss.display().to_string(),
        output_csv_path: args.output.display().to_string(),
        release: args.release.label().to_ascii_uppercase(),
        input_things_count: things.len(),
        manga_overlap_count,
        lotss_overlap_count,
        triple_overlap_count: rows.len(),
        match_radius_arcsec: args.match_radius_arcsec,
        chunk_rows: best_match_result.execution.chunk_rows,
        execution: best_match_result.execution.clone(),
    };
    write_toml_report(&args.report, &report_model)?;
    println!("THINGS galaxies: {}", report_model.input_things_count);
    println!("MaNGA overlap:   {}", report_model.manga_overlap_count);
    println!("LoTSS overlap:   {}", report_model.lotss_overlap_count);
    println!("Triple overlap:  {}", report_model.triple_overlap_count);
    println!(
        "Execution:       mode={} workers={} chunk_rows={} pinned={:?} scan_wall={:.2}s",
        report_model.execution.mode,
        report_model.execution.worker_count,
        report_model.execution.chunk_rows,
        report_model.execution.pinned_core_ids,
        report_model.execution.scan_wall_seconds
    );
    println!("CSV:             {}", args.output.display());
    println!("Report:          {}", args.report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_things_metadata(
    things_table1: PathBuf,
    things_table4: PathBuf,
    output: PathBuf,
    report: PathBuf,
    default_beam_fwhm_arcsec: f64,
) -> Result<()> {
    if default_beam_fwhm_arcsec <= 0.0 {
        bail!("default_beam_fwhm_arcsec must be positive");
    }
    let galaxies = parse_things_galaxies(&things_table1).map_err(anyhow::Error::msg)?;
    let spectra = parse_things_hi_spectra(&things_table4).map_err(anyhow::Error::msg)?;
    let rows = build_things_hi_metadata(&galaxies, &spectra, default_beam_fwhm_arcsec)
        .into_iter()
        .map(|row| ThingsMetadataRow {
            name: row.name,
            ra_deg: format!("{:.6}", row.ra_deg),
            dec_deg: format!("{:.6}", row.dec_deg),
            distance_mpc: format!("{:.6}", row.distance_mpc),
            inclination_deg: format!("{:.3}", row.inclination_deg),
            pa_deg: format!("{:.3}", row.pa_deg),
            beam_fwhm_arcsec: format!("{:.3}", row.beam_fwhm_arcsec),
            channel_width_km_s: format!("{:.4}", row.channel_width_km_s),
            v_sys_km_s: format!("{:.4}", row.v_sys_km_s),
        })
        .collect::<Vec<_>>();
    write_csv(&output, &rows)?;

    let report_model = ThingsMetadataReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        things_table1_path: things_table1.display().to_string(),
        things_table4_path: things_table4.display().to_string(),
        output_csv_path: output.display().to_string(),
        default_beam_fwhm_arcsec,
        galaxy_count: rows.len(),
    };
    write_toml_report(&report, &report_model)?;
    println!("Metadata rows: {}", report_model.galaxy_count);
    println!("CSV:           {}", output.display());
    println!("Report:        {}", report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn cmd_catalog_matrix(args: CatalogMatrixArgs) -> Result<()> {
    if args.match_radius_arcsec <= 0.0 {
        bail!("match_radius_arcsec must be positive");
    }

    let execution_plan = build_catalog_matrix_execution_plan(args.workers, args.chunk_rows);
    let euclid_report_path =
        resolve_euclid_lotss_report_path(args.release, &args.euclid_lotss_report);
    let mut entries = Vec::new();
    entries.push(CatalogMatrixEntry {
        catalog: format!("LoTSS {}", args.release.label().to_ascii_uppercase()),
        modality: modality_label(CatalogModality::SkyPoint),
        path: args.lotss.display().to_string(),
        row_count: Some(lotss_fits_row_count(&args.lotss).map_err(anyhow::Error::msg)?),
        exact_overlap_with_lotss: None,
        status: "reference".to_string(),
        note: "Reference radio survey for overlap counts.".to_string(),
    });

    let atnf_points = load_point_catalog(&args.atnf, "ATNF Pulsars", parse_atnf_csv, |row| {
        Some(SkyPoint {
            id: row.name.clone(),
            ra_deg: row.ra,
            dec_deg: row.dec,
        })
    })?;

    let mcgill_points =
        load_point_catalog(&args.mcgill, "McGill Magnetars", parse_mcgill_csv, |row| {
            Some(SkyPoint {
                id: row.name.clone(),
                ra_deg: row.ra,
                dec_deg: row.dec,
            })
        })?;

    let chime_points = load_point_catalog(&args.chime, "CHIME FRB", parse_chime_csv, |row| {
        Some(SkyPoint {
            id: row.tns_name.clone(),
            ra_deg: row.ra,
            dec_deg: row.dec,
        })
    })?;

    let sdss_points = load_point_catalog(
        &args.sdss,
        "SDSS DR18 Quasars",
        parse_sdss_quasar_csv,
        |row| {
            Some(SkyPoint {
                id: row.objid.clone(),
                ra_deg: row.ra,
                dec_deg: row.dec,
            })
        },
    )?;

    let gaia_points = load_point_catalog(&args.gaia, "Gaia DR3 Nearby", parse_gaia_csv, |row| {
        Some(SkyPoint {
            id: row.source_id.clone(),
            ra_deg: row.ra,
            dec_deg: row.dec,
        })
    })?;
    let jwst_points = load_point_catalog(
        &args.jwst,
        "JWST Public Metadata",
        parse_jwst_public_metadata_csv,
        |row| {
            if row.s_ra.is_finite() && row.s_dec.is_finite() {
                Some(SkyPoint {
                    id: row.obsid.clone(),
                    ra_deg: row.s_ra,
                    dec_deg: row.s_dec,
                })
            } else {
                None
            }
        },
    )?;
    let hst_points = load_point_catalog(
        &args.hst,
        "HST Public Metadata",
        parse_hst_public_metadata_csv,
        |row| {
            if row.s_ra.is_finite() && row.s_dec.is_finite() {
                Some(SkyPoint {
                    id: row.obsid.clone(),
                    ra_deg: row.s_ra,
                    dec_deg: row.s_dec,
                })
            } else {
                None
            }
        },
    )?;

    let manga_sample = load_selected_manga_targets(&args.manga_selection, &args.manga_drpall)?;
    let manga_overlap = count_detected_rows(&args.manga_lotss, "lotss_detected")?;
    entries.push(CatalogMatrixEntry {
        catalog: "MaNGA selected sample".to_string(),
        modality: modality_label(CatalogModality::SkyPoint),
        path: args.manga_selection.display().to_string(),
        row_count: Some(manga_sample.targets.len()),
        exact_overlap_with_lotss: Some(manga_overlap),
        status: "implemented".to_string(),
        note: format!("Exact overlap sourced from {}", args.manga_lotss.display()),
    });

    let things_points = load_things_points(&args.things_table1)?;
    let mut point_catalogs = vec![
        PreparedPointMatrixCatalog {
            catalog: "ATNF Pulsars".to_string(),
            path: args.atnf.clone(),
            points: atnf_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "McGill Magnetars".to_string(),
            path: args.mcgill.clone(),
            points: mcgill_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "CHIME FRB".to_string(),
            path: args.chime.clone(),
            points: chime_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "SDSS DR18 Quasars".to_string(),
            path: args.sdss.clone(),
            points: sdss_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "Gaia DR3 Nearby".to_string(),
            path: args.gaia.clone(),
            points: gaia_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "JWST Public Metadata".to_string(),
            path: args.jwst.clone(),
            points: jwst_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "HST Public Metadata".to_string(),
            path: args.hst.clone(),
            points: hst_points,
        },
        PreparedPointMatrixCatalog {
            catalog: "THINGS galaxies".to_string(),
            path: args.things_table1.clone(),
            points: things_points,
        },
    ];
    if let Some(rotcurves_path) = args.things_rotcurves.as_ref().filter(|path| path.exists()) {
        point_catalogs.push(PreparedPointMatrixCatalog {
            catalog: "THINGS HI rotcurves".to_string(),
            path: rotcurves_path.clone(),
            points: load_things_points_filtered(
                &args.things_table1,
                Some(rotcurves_path.as_path()),
            )?,
        });
    }
    let scan_started = Instant::now();
    let overlap_counts = count_multi_point_overlaps_with_lotss(
        &args.lotss,
        args.match_radius_arcsec,
        &execution_plan,
        &point_catalogs,
    )?;
    let scan_wall_seconds = scan_started.elapsed().as_secs_f64();
    for (catalog, overlap_count) in point_catalogs.iter().zip(overlap_counts) {
        entries.push(CatalogMatrixEntry {
            catalog: catalog.catalog.clone(),
            modality: modality_label(CatalogModality::SkyPoint),
            path: catalog.path.display().to_string(),
            row_count: Some(catalog.points.len()),
            exact_overlap_with_lotss: Some(overlap_count),
            status: "implemented".to_string(),
            note: format!(
                "Exact sky-point overlap computed in a single shared LoTSS scan (workers={}, chunk_rows={}).",
                execution_plan.worker_count, execution_plan.chunk_rows
            ),
        });
    }

    entries.push(CatalogMatrixEntry {
        catalog: "Euclid morphology".to_string(),
        modality: modality_label(CatalogModality::SkyPoint),
        path: euclid_report_path.display().to_string(),
        row_count: None,
        exact_overlap_with_lotss: read_report_usize(&euclid_report_path, "matched_euclid_count"),
        status: if euclid_report_path.exists() {
            "implemented".to_string()
        } else {
            "pending".to_string()
        },
        note: if euclid_report_path.exists() {
            "Exact overlap sourced from prior Euclid-LoTSS report.".to_string()
        } else {
            "Run `survey-crossmatch euclid-lotss` to materialize exact overlap.".to_string()
        },
    });

    entries.push(CatalogMatrixEntry {
        catalog: "Pantheon+ SH0ES".to_string(),
        modality: modality_label(CatalogModality::NonSpatial),
        path: "data/external/PantheonPlusSH0ES.dat".to_string(),
        row_count: None,
        exact_overlap_with_lotss: None,
        status: "parallel".to_string(),
        note: "No sky-position columns in the operational Pantheon+ distance table; keep in cosmology lane.".to_string(),
    });
    let desi_reference_path = ensure_desi_dr2_reference_report()?;
    entries.push(CatalogMatrixEntry {
        catalog: "DESI BAO".to_string(),
        modality: modality_label(CatalogModality::NonSpatial),
        path: desi_reference_path.display().to_string(),
        row_count: None,
        exact_overlap_with_lotss: None,
        status: "parallel".to_string(),
        note: "Paper-summary input refreshed to DESI DR2 BAO Table 1 provenance; keep in the cosmology lane rather than forcing a sky join.".to_string(),
    });
    entries.push(CatalogMatrixEntry {
        catalog: "GWOSC/GWTC".to_string(),
        modality: modality_label(CatalogModality::SkyLocalization),
        path: "data/external/gwosc_all_events.csv".to_string(),
        row_count: None,
        exact_overlap_with_lotss: None,
        status: "deferred".to_string(),
        note: "Operational catalog lacks per-event sky-map geometry here; probabilistic localization join deferred.".to_string(),
    });

    let report = CatalogMatrixReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        release: args.release.label().to_ascii_uppercase(),
        lotss_path: args.lotss.display().to_string(),
        match_radius_arcsec: args.match_radius_arcsec,
        execution: CatalogMatrixExecutionReport {
            mode: execution_plan.mode.to_string(),
            worker_count: execution_plan.worker_count,
            chunk_rows: execution_plan.chunk_rows,
            pinned_core_ids: execution_plan.physical_core_ids.clone(),
            l3_cache_bytes: execution_plan.l3_cache_bytes,
            l3_safe_working_set_bytes: execution_plan.l3_safe_working_set_bytes,
            avx2_detected: execution_plan.avx2_detected,
            fma_detected: execution_plan.fma_detected,
            simd_lane_f64: execution_plan.simd_lane_f64,
            single_pass_catalog_count: point_catalogs.len(),
            scan_wall_seconds,
        },
        entries,
    };
    write_toml_report(&args.report, &report)?;
    println!("Matrix entries: {}", report.entries.len());
    println!(
        "Execution:      mode={} workers={} chunk_rows={} pinned={:?} scan_wall={:.2}s",
        report.execution.mode,
        report.execution.worker_count,
        report.execution.chunk_rows,
        report.execution.pinned_core_ids,
        report.execution.scan_wall_seconds
    );
    println!("Report:         {}", args.report.display());
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn select_euclid_records(
    morphology_path: &Path,
    featured_min: f32,
    spiral_min: f32,
    face_on_min: f32,
    non_merge_min: f32,
    no_morphology_cuts: bool,
    _match_radius_arcsec: f64,
) -> Result<EuclidSelection> {
    let mut records = read_euclid_visual_morphology(
        morphology_path
            .to_str()
            .ok_or_else(|| anyhow!("Non-UTF8 Euclid morphology path"))?,
    )
    .map_err(anyhow::Error::msg)?;
    if !no_morphology_cuts {
        records.retain(|row| {
            row.featured_fraction > featured_min
                && row.spiral_fraction > spiral_min
                && row.face_on_fraction > face_on_min
                && row.non_merging_fraction > non_merge_min
        });
    }
    Ok(EuclidSelection { records })
}

#[cfg(feature = "euclid-catalog")]
fn load_selected_manga_targets(
    selection_path: &Path,
    drpall_path: &Path,
) -> Result<SelectedMangaSample> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(selection_path)?;
    let headers = reader.headers()?.clone();
    let plateifu_idx = headers
        .iter()
        .position(|header| header.eq_ignore_ascii_case("plateifu"))
        .ok_or_else(|| anyhow!("No plateifu column in {}", selection_path.display()))?;

    let rows = read_fits_table(drpall_path, DRPALL_COLUMNS)
        .map_err(|e| anyhow!("read DRPall {}: {}", drpall_path.display(), e))?;
    let mut by_plateifu = HashMap::with_capacity(rows.len());
    for row in rows {
        let plateifu = fits_string(&row, "PLATEIFU");
        if plateifu.is_empty() {
            continue;
        }
        let ra_deg = fits_f64(&row, "OBJRA").unwrap_or(f64::NAN);
        let dec_deg = fits_f64(&row, "OBJDEC").unwrap_or(f64::NAN);
        if !ra_deg.is_finite() || !dec_deg.is_finite() {
            continue;
        }
        by_plateifu.insert(
            plateifu.clone(),
            MangaSkyTarget {
                plateifu,
                mangaid: fits_string(&row, "MANGAID"),
                ra_deg,
                dec_deg,
            },
        );
    }

    let mut targets = Vec::new();
    let mut seen = HashSet::new();
    for record in reader.records() {
        let record = record?;
        let plateifu = record.get(plateifu_idx).unwrap_or("").trim();
        if plateifu.is_empty() || !seen.insert(plateifu.to_string()) {
            continue;
        }
        if let Some(target) = by_plateifu.get(plateifu) {
            targets.push(target.clone());
        }
    }
    Ok(SelectedMangaSample { targets })
}

#[cfg(feature = "euclid-catalog")]
fn load_things_points(path: &Path) -> Result<Vec<SkyPoint>> {
    let galaxies = parse_things_galaxies(path).map_err(anyhow::Error::msg)?;
    Ok(things_to_points(&galaxies))
}

#[cfg(feature = "euclid-catalog")]
fn things_name_key(raw: &str) -> String {
    raw.chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .map(|ch| ch.to_ascii_uppercase())
        .collect()
}

#[cfg(feature = "euclid-catalog")]
fn load_things_rotcurve_name_set(path: &Path) -> Result<HashSet<String>> {
    let curves = parse_hi_rotcurves(path).map_err(anyhow::Error::msg)?;
    Ok(curves
        .into_iter()
        .map(|curve| things_name_key(&curve.name))
        .collect())
}

#[cfg(feature = "euclid-catalog")]
fn load_things_galaxies_filtered(
    table1_path: &Path,
    things_rotcurves: Option<&Path>,
) -> Result<Vec<data_core::catalogs::things::ThingsGalaxy>> {
    let mut galaxies = parse_things_galaxies(table1_path).map_err(anyhow::Error::msg)?;
    if let Some(rotcurves_path) = things_rotcurves.filter(|path| path.exists()) {
        let allowed = load_things_rotcurve_name_set(rotcurves_path)?;
        galaxies.retain(|galaxy| allowed.contains(&things_name_key(&galaxy.name)));
    }
    Ok(galaxies)
}

#[cfg(feature = "euclid-catalog")]
fn load_things_points_filtered(
    table1_path: &Path,
    things_rotcurves: Option<&Path>,
) -> Result<Vec<SkyPoint>> {
    let galaxies = load_things_galaxies_filtered(table1_path, things_rotcurves)?;
    Ok(things_to_points(&galaxies))
}

#[cfg(feature = "euclid-catalog")]
fn things_to_points(galaxies: &[data_core::catalogs::things::ThingsGalaxy]) -> Vec<SkyPoint> {
    galaxies
        .iter()
        .map(|galaxy| SkyPoint {
            id: galaxy.name.clone(),
            ra_deg: galaxy.ra_hours * 15.0,
            dec_deg: galaxy.dec_deg,
        })
        .collect()
}

#[cfg(feature = "euclid-catalog")]
fn match_points_to_manga(
    points: &[SkyPoint],
    sample: &SelectedMangaSample,
    match_radius_arcsec: f64,
) -> Vec<Option<MangaBestMatch>> {
    let manga_points = sample
        .targets
        .iter()
        .map(|target| SkyPoint {
            id: target.plateifu.clone(),
            ra_deg: target.ra_deg,
            dec_deg: target.dec_deg,
        })
        .collect::<Vec<_>>();
    let grid = SkyGridIndex::from_points(manga_points, (match_radius_arcsec / 3600.0).max(0.01));
    points
        .iter()
        .map(|point| {
            let matched = grid.nearest_within(point.ra_deg, point.dec_deg, match_radius_arcsec)?;
            let target = &sample.targets[matched.candidate_index];
            Some(MangaBestMatch {
                separation_arcsec: matched.separation_arcsec,
                plateifu: target.plateifu.clone(),
                mangaid: target.mangaid.clone(),
                manga_ra_deg: target.ra_deg,
                manga_dec_deg: target.dec_deg,
            })
        })
        .collect()
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug)]
struct PreparedPointMatrixCatalog {
    catalog: String,
    path: PathBuf,
    points: Vec<SkyPoint>,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug)]
struct CatalogMatrixExecutionPlan {
    mode: &'static str,
    worker_count: usize,
    chunk_rows: usize,
    physical_core_ids: Vec<usize>,
    l3_cache_bytes: usize,
    l3_safe_working_set_bytes: usize,
    avx2_detected: bool,
    fma_detected: bool,
    simd_lane_f64: usize,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug)]
struct LotssPositionLayout {
    table_idx: usize,
    row_count: usize,
    ra_column: String,
    dec_column: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug)]
struct MatrixScanSummary {
    matched_flags: Vec<Vec<bool>>,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug)]
struct LotssBestMatchResult {
    matches: Vec<Option<LotssBestMatch>>,
    scanned_source_count: usize,
    execution: LotssExecutionReport,
}

#[cfg(feature = "euclid-catalog")]
fn build_catalog_matrix_execution_plan(
    requested_workers: Option<usize>,
    chunk_rows_override: Option<usize>,
) -> CatalogMatrixExecutionPlan {
    let topo = HardwareTopology::current();
    let mut physical_core_ids = topo.physical_core_ids.clone();
    if physical_core_ids.is_empty() {
        physical_core_ids.push(0);
    }
    if let Some(limit) = requested_workers {
        physical_core_ids.truncate(limit.max(1).min(physical_core_ids.len()));
    }
    let worker_count = physical_core_ids.len().max(1);
    let avx2_detected = detect_avx2();
    let fma_detected = detect_fma();
    let simd_lane_f64 = preferred_f64_simd_lane(avx2_detected);
    let per_worker_bytes = topo
        .l3_safe_working_set_bytes
        .max(MATRIX_NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW)
        / worker_count.max(1);
    let auto_chunk_rows = (per_worker_bytes / MATRIX_NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW).clamp(
        MATRIX_MIN_PARALLEL_FITS_CHUNK_ROWS,
        MATRIX_MAX_PARALLEL_FITS_CHUNK_ROWS,
    );
    let alignment_rows = (simd_lane_f64 * 256).max(1);
    let chunk_rows = chunk_rows_override.unwrap_or(auto_chunk_rows).clamp(
        MATRIX_MIN_PARALLEL_FITS_CHUNK_ROWS,
        MATRIX_MAX_PARALLEL_FITS_CHUNK_ROWS,
    ) / alignment_rows
        * alignment_rows;
    let chunk_rows = chunk_rows.max(MATRIX_MIN_PARALLEL_FITS_CHUNK_ROWS);

    CatalogMatrixExecutionPlan {
        mode: if worker_count > 1 {
            "pinned_physical_single_pass"
        } else {
            "scalar_single_pass"
        },
        worker_count,
        chunk_rows,
        physical_core_ids,
        l3_cache_bytes: topo.l3_cache_bytes,
        l3_safe_working_set_bytes: topo.l3_safe_working_set_bytes,
        avx2_detected,
        fma_detected,
        simd_lane_f64,
    }
}

#[cfg(feature = "euclid-catalog")]
fn count_multi_point_overlaps_with_lotss(
    lotss_path: &Path,
    match_radius_arcsec: f64,
    execution_plan: &CatalogMatrixExecutionPlan,
    catalogs: &[PreparedPointMatrixCatalog],
) -> Result<Vec<usize>> {
    let layout = inspect_lotss_position_layout(lotss_path)?;
    let prepared = prepare_point_matrix_grids(catalogs, match_radius_arcsec);
    let bounds = split_matrix_work_bounds(layout.row_count, execution_plan.worker_count);
    let core_ids = execution_plan.physical_core_ids.clone();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(execution_plan.worker_count.max(1))
        .start_handler(move |idx| {
            if let Some(&core_id) = core_ids.get(idx) {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
            }
        })
        .build()
        .map_err(|e| anyhow!("build catalog-matrix rayon pool: {}", e))?;

    let worker_summaries = pool.install(|| {
        bounds
            .into_par_iter()
            .map(|(start, end)| {
                scan_lotss_point_overlap_range(
                    lotss_path,
                    &layout,
                    &prepared,
                    match_radius_arcsec,
                    start..end,
                    execution_plan.chunk_rows,
                )
            })
            .collect::<Vec<_>>()
    });

    let mut merged = prepared
        .iter()
        .map(|catalog| vec![false; catalog.point_count])
        .collect::<Vec<_>>();
    for summary in worker_summaries {
        let summary = summary?;
        for (global, local) in merged.iter_mut().zip(summary.matched_flags) {
            for (slot, hit) in global.iter_mut().zip(local) {
                *slot |= hit;
            }
        }
    }

    Ok(merged
        .into_iter()
        .map(|hits| hits.into_iter().filter(|hit| *hit).count())
        .collect())
}

#[cfg(feature = "euclid-catalog")]
fn prepare_point_matrix_grids(
    catalogs: &[PreparedPointMatrixCatalog],
    match_radius_arcsec: f64,
) -> Vec<PreparedPointGrid> {
    catalogs
        .iter()
        .map(|catalog| prepare_point_grid(&catalog.points, match_radius_arcsec))
        .collect()
}

#[cfg(feature = "euclid-catalog")]
fn inspect_lotss_position_layout(path: &Path) -> Result<LotssPositionLayout> {
    let mut fits = FitsFile::open(path).map_err(|e| anyhow!("open {}: {}", path.display(), e))?;
    let num_hdus = {
        let mut count = 0usize;
        for _ in fits.iter() {
            count += 1;
        }
        count
    };

    let mut table_idx = None;
    let mut row_count = 0usize;
    let mut column_names = Vec::new();
    for idx in 1..num_hdus {
        let hdu = fits.hdu(idx).map_err(|e| anyhow!("hdu {}: {}", idx, e))?;
        if let HduInfo::TableInfo {
            column_descriptions,
            num_rows,
            ..
        } = hdu.info
        {
            table_idx = Some(idx);
            row_count = num_rows;
            column_names = column_descriptions
                .into_iter()
                .map(|description| description.name)
                .collect();
            break;
        }
    }

    let table_idx =
        table_idx.ok_or_else(|| anyhow!("No BINTABLE HDU found in {}", path.display()))?;
    let ra_column = column_names
        .iter()
        .find(|name| name.eq_ignore_ascii_case("RA"))
        .cloned()
        .ok_or_else(|| anyhow!("No RA column found in {}", path.display()))?;
    let dec_column = column_names
        .iter()
        .find(|name| name.eq_ignore_ascii_case("DEC"))
        .cloned()
        .ok_or_else(|| anyhow!("No DEC column found in {}", path.display()))?;

    Ok(LotssPositionLayout {
        table_idx,
        row_count,
        ra_column,
        dec_column,
    })
}

#[cfg(feature = "euclid-catalog")]
fn split_matrix_work_bounds(len: usize, parts: usize) -> Vec<(usize, usize)> {
    if len == 0 {
        return vec![(0, 0)];
    }
    let parts = parts.max(1).min(len);
    let base = len / parts;
    let remainder = len % parts;
    let mut bounds = Vec::with_capacity(parts);
    let mut start = 0usize;
    for idx in 0..parts {
        let extra = usize::from(idx < remainder);
        let end = start + base + extra;
        bounds.push((start, end));
        start = end;
    }
    bounds
}

#[cfg(feature = "euclid-catalog")]
fn scan_lotss_point_overlap_range(
    lotss_path: &Path,
    layout: &LotssPositionLayout,
    catalogs: &[PreparedPointGrid],
    match_radius_arcsec: f64,
    worker_bounds: std::ops::Range<usize>,
    chunk_rows: usize,
) -> Result<MatrixScanSummary> {
    let mut fits =
        FitsFile::open(lotss_path).map_err(|e| anyhow!("open {}: {}", lotss_path.display(), e))?;
    let table_hdu = fits
        .hdu(layout.table_idx)
        .map_err(|e| anyhow!("hdu {}: {}", layout.table_idx, e))?;
    let mut matched_flags = catalogs
        .iter()
        .map(|catalog| vec![false; catalog.point_count])
        .collect::<Vec<_>>();

    for start in (worker_bounds.start..worker_bounds.end).step_by(chunk_rows.max(1)) {
        let end = (start + chunk_rows).min(worker_bounds.end);
        let row_range = start..end;
        let ras: Vec<f64> = table_hdu
            .read_col_range(&mut fits, &layout.ra_column, &row_range)
            .map_err(|e| {
                anyhow!(
                    "Load FITS {} rows {}..{}: {}",
                    layout.ra_column,
                    start,
                    end,
                    e
                )
            })?;
        let decs: Vec<f64> = table_hdu
            .read_col_range(&mut fits, &layout.dec_column, &row_range)
            .map_err(|e| {
                anyhow!(
                    "Load FITS {} rows {}..{}: {}",
                    layout.dec_column,
                    start,
                    end,
                    e
                )
            })?;

        for row_idx in 0..ras.len() {
            let ra_deg = *ras.get(row_idx).unwrap_or(&f64::NAN);
            let dec_deg = *decs.get(row_idx).unwrap_or(&f64::NAN);
            if !ra_deg.is_finite() || !dec_deg.is_finite() {
                continue;
            }
            for (catalog_idx, catalog) in catalogs.iter().enumerate() {
                for_each_point_grid_match(
                    catalog,
                    ra_deg,
                    dec_deg,
                    match_radius_arcsec,
                    |matched_index, _| matched_flags[catalog_idx][matched_index] = true,
                );
            }
        }
    }

    Ok(MatrixScanSummary { matched_flags })
}

#[cfg(feature = "euclid-catalog")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(feature = "euclid-catalog")]
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_avx2() -> bool {
    false
}

#[cfg(feature = "euclid-catalog")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_fma() -> bool {
    std::arch::is_x86_feature_detected!("fma")
}

#[cfg(feature = "euclid-catalog")]
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_fma() -> bool {
    false
}

#[cfg(feature = "euclid-catalog")]
fn preferred_f64_simd_lane(avx2_detected: bool) -> usize {
    if avx2_detected { 4 } else { 1 }
}

#[cfg(feature = "euclid-catalog")]
fn find_best_lotss_matches(
    lotss_path: &Path,
    release: ReleaseArg,
    match_radius_arcsec: f64,
    chunk_rows: Option<usize>,
    workers: Option<usize>,
    points: &[SkyPoint],
) -> Result<LotssBestMatchResult> {
    let shared = crossmatch_points_against_fits_catalog(
        lotss_path,
        release.lotss_release(),
        points,
        match_radius_arcsec,
        workers,
        chunk_rows,
    )
    .map_err(anyhow::Error::msg)?;
    Ok(LotssBestMatchResult {
        matches: shared
            .matches
            .into_iter()
            .map(|entry| entry.map(shared_lotss_match_to_local))
            .collect(),
        scanned_source_count: shared.scanned_source_count,
        execution: shared_execution_to_local(&shared.execution),
    })
}

#[cfg(feature = "euclid-catalog")]
fn shared_execution_to_local(
    execution: &data_core::LotssFitsExecutionReport,
) -> LotssExecutionReport {
    LotssExecutionReport {
        mode: execution.mode.clone(),
        worker_count: execution.worker_count,
        chunk_rows: execution.chunk_rows,
        pinned_core_ids: execution.pinned_core_ids.clone(),
        l3_cache_bytes: execution.l3_cache_bytes,
        l3_safe_working_set_bytes: execution.l3_safe_working_set_bytes,
        avx2_detected: execution.avx2_detected,
        fma_detected: execution.fma_detected,
        simd_lane_f64: execution.simd_lane_f64,
        x87_confirmation_used: execution.x87_confirmation_used,
        scan_wall_seconds: execution.scan_wall_seconds,
    }
}

#[cfg(feature = "euclid-catalog")]
fn shared_lotss_match_to_local(entry: data_core::LotssFitsBestMatch) -> LotssBestMatch {
    LotssBestMatch {
        separation_arcsec: entry.separation_arcsec,
        source_name: entry.source.source_name,
        lotss_ra_deg: entry.source.ra_deg,
        lotss_dec_deg: entry.source.dec_deg,
        lotss_flux_mjy: entry.source.flux_mjy,
        lotss_spectral_index: entry.source.spectral_index,
        lotss_structure_code: entry.source.structure_code,
    }
}

#[cfg(feature = "euclid-catalog")]
fn fits_string(row: &HashMap<String, FitsValue>, key: &str) -> String {
    row.get(key)
        .and_then(FitsValue::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string()
}

#[cfg(feature = "euclid-catalog")]
fn fits_f64(row: &HashMap<String, FitsValue>, key: &str) -> Option<f64> {
    row.get(key).and_then(FitsValue::as_f64)
}

#[cfg(feature = "euclid-catalog")]
fn load_point_catalog<T, F, G>(
    path: &Path,
    label: &str,
    parser: F,
    mut mapper: G,
) -> Result<Vec<SkyPoint>>
where
    F: Fn(&Path) -> std::result::Result<Vec<T>, data_core::fetcher::FetchError>,
    G: FnMut(&T) -> Option<SkyPoint>,
{
    if !path.exists() {
        bail!("Missing {} catalog at {}", label, path.display());
    }
    let rows = parser(path).map_err(anyhow::Error::msg)?;
    let mut points = Vec::new();
    for row in &rows {
        if let Some(point) = mapper(row)
            && point.ra_deg.is_finite()
            && point.dec_deg.is_finite()
        {
            points.push(point);
        }
    }
    Ok(points)
}

#[cfg(feature = "euclid-catalog")]
fn count_detected_rows(path: &Path, column: &str) -> Result<usize> {
    if !path.exists() {
        return Ok(0);
    }
    let mut reader = csv::Reader::from_path(path)?;
    let headers = reader.headers()?.clone();
    let Some(idx) = headers
        .iter()
        .position(|header| header.eq_ignore_ascii_case(column))
    else {
        return Ok(0);
    };
    let mut count = 0usize;
    for record in reader.records() {
        let record = record?;
        if record.get(idx).unwrap_or("").trim() == "1" {
            count += 1;
        }
    }
    Ok(count)
}

#[cfg(feature = "euclid-catalog")]
fn read_report_usize(path: &Path, key: &str) -> Option<usize> {
    let text = fs::read_to_string(path).ok()?;
    let value = text.parse::<toml::Value>().ok()?;
    value
        .get(key)
        .and_then(toml::Value::as_integer)
        .and_then(|value| usize::try_from(value).ok())
}

#[cfg(feature = "euclid-catalog")]
fn resolve_euclid_lotss_report_path(release: ReleaseArg, requested: &Path) -> PathBuf {
    if requested.exists() {
        return requested.to_path_buf();
    }
    let Some(parent) = requested.parent() else {
        return requested.to_path_buf();
    };
    let release_label = release.label();
    let mut candidates = Vec::new();
    let patterns = [
        format!("euclid_lotss_xmatch_{}_", release_label),
        format!("euclid_lotss_xmatch_{}_nocuts_", release_label),
    ];
    if let Ok(entries) = fs::read_dir(parent) {
        for entry in entries.flatten() {
            let path = entry.path();
            let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
                continue;
            };
            if !name.ends_with(".toml") {
                continue;
            }
            if patterns.iter().any(|prefix| name.starts_with(prefix)) {
                let modified = entry.metadata().and_then(|meta| meta.modified()).ok();
                candidates.push((modified, path));
            }
        }
    }
    candidates.sort_by_key(|left| left.0);
    candidates
        .pop()
        .map(|(_, path)| path)
        .unwrap_or_else(|| requested.to_path_buf())
}

#[cfg(feature = "euclid-catalog")]
fn write_csv<T: Serialize>(path: &Path, rows: &[T]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut writer = csv::Writer::from_path(path)?;
    for row in rows {
        writer.serialize(row)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, toml::to_string_pretty(value)?)?;
    Ok(())
}

#[cfg(feature = "euclid-catalog")]
fn load_csv_rows<T>(path: &Path) -> Result<Vec<T>>
where
    T: for<'de> Deserialize<'de>,
{
    if !path.exists() {
        return Ok(Vec::new());
    }
    if fs::metadata(path)?.len() == 0 {
        return Ok(Vec::new());
    }
    let mut reader = csv::Reader::from_path(path)?;
    let mut rows = Vec::new();
    for row in reader.deserialize() {
        rows.push(row?);
    }
    Ok(rows)
}

#[cfg(feature = "euclid-catalog")]
fn modality_label(modality: CatalogModality) -> String {
    match modality {
        CatalogModality::SkyPoint => "sky_point",
        CatalogModality::SkyFootprint => "sky_footprint",
        CatalogModality::SkyLocalization => "sky_localization",
        CatalogModality::NonSpatial => "non_spatial",
    }
    .to_string()
}

#[cfg(feature = "euclid-catalog")]
fn default_lotss_path(release: ReleaseArg) -> PathBuf {
    PathBuf::from(format!(
        "data/external/radio_surveys/lotss_{}.fits",
        release.label()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_euclid_lotss_csv(release: ReleaseArg) -> PathBuf {
    PathBuf::from(format!(
        "data/external/euclid/euclid_lotss_xmatch_{}.csv",
        release.label()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_euclid_lotss_csv_for_mode(release: ReleaseArg, no_morphology_cuts: bool) -> PathBuf {
    if no_morphology_cuts {
        PathBuf::from(format!(
            "data/external/euclid/euclid_lotss_xmatch_{}_nocuts.csv",
            release.label()
        ))
    } else {
        default_euclid_lotss_csv(release)
    }
}

#[cfg(feature = "euclid-catalog")]
fn default_euclid_lotss_report(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "euclid_lotss_xmatch_{}_{}.toml",
        release.label(),
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_euclid_lotss_report_for_mode(release: ReleaseArg, no_morphology_cuts: bool) -> PathBuf {
    if no_morphology_cuts {
        PathBuf::from("reports").join(format!(
            "euclid_lotss_xmatch_{}_nocuts_{}.toml",
            release.label(),
            chrono::Utc::now().date_naive()
        ))
    } else {
        default_euclid_lotss_report(release)
    }
}

#[cfg(feature = "euclid-catalog")]
fn default_euclid_manga_lotss_csv() -> PathBuf {
    PathBuf::from("data/external/euclid/euclid_manga_lotss_triple.csv")
}

#[cfg(feature = "euclid-catalog")]
fn default_euclid_manga_lotss_report() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "euclid_manga_lotss_triple_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_atnf_lotss_csv(release: ReleaseArg) -> PathBuf {
    PathBuf::from("data/external")
        .join(format!("atnf_pulsars_lotss_xmatch_{}.csv", release.label()))
}

#[cfg(feature = "euclid-catalog")]
fn default_atnf_lotss_report(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "atnf_lotss_xmatch_{}_{}.toml",
        release.label(),
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_mcgill_lotss_csv(release: ReleaseArg) -> PathBuf {
    PathBuf::from("data/external").join(format!(
        "mcgill_magnetars_lotss_xmatch_{}.csv",
        release.label()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_mcgill_lotss_report(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "mcgill_lotss_xmatch_{}_{}.toml",
        release.label(),
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_things_manga_csv() -> PathBuf {
    PathBuf::from("data/external/things/things_manga_xmatch.csv")
}

#[cfg(feature = "euclid-catalog")]
fn default_things_manga_report() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "things_manga_xmatch_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_things_lotss_csv(release: ReleaseArg) -> PathBuf {
    PathBuf::from(format!(
        "data/external/things/things_lotss_xmatch_{}.csv",
        release.label()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_things_lotss_report(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "things_lotss_xmatch_{}_{}.toml",
        release.label(),
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_things_manga_lotss_csv(release: ReleaseArg) -> PathBuf {
    PathBuf::from(format!(
        "data/external/things/things_manga_lotss_triple_{}.csv",
        release.label()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_things_manga_lotss_report(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "things_manga_lotss_triple_{}_{}.toml",
        release.label(),
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_things_metadata_csv() -> PathBuf {
    PathBuf::from("data/external/things/things_metadata.csv")
}

#[cfg(feature = "euclid-catalog")]
fn default_things_metadata_report() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "things_metadata_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn default_catalog_matrix_report(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "survey_catalog_matrix_{}_{}.toml",
        release.label(),
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(feature = "euclid-catalog")]
fn ensure_desi_dr2_reference_report() -> Result<PathBuf> {
    let path = PathBuf::from("reports").join(format!(
        "desi_dr2_bao_reference_{}.toml",
        chrono::Utc::now().date_naive()
    ));
    let measurements = desi_dr2_bao()
        .into_iter()
        .map(|row| DesiDr2ReferenceMeasurement {
            z_or_label: if row.z_eff > 0.0 {
                format!("{}@z={:.6}", row.tracer, row.z_eff)
            } else {
                row.tracer
            },
            measurement: if row.is_isotropic {
                format!("DV/rd={:.8}", row.dm_over_rd)
            } else {
                format!("DM/rd={:.8}, DH/rd={:.8}", row.dm_over_rd, row.dh_over_rd)
            },
            uncertainty: if row.is_isotropic {
                format!("{:.8}", row.dm_over_rd_err)
            } else {
                format!(
                    "sigma_dm={:.8}, sigma_dh={:.8}, rho={:.6}",
                    row.dm_over_rd_err, row.dh_over_rd_err, row.rho
                )
            },
        })
        .collect::<Vec<_>>();
    let report = DesiDr2ReferenceReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        provenance_note: "prime_distillation.md identifies DESI DR2 BAO Table 1 as the operative provenance surface for this matrix row.".to_string(),
        source: "DESI DR2 BAO cosmology paper Table 1 (arXiv:2503.14738) via data_core::catalogs::desi_bao::desi_dr2_bao()".to_string(),
        measurement_count: measurements.len(),
        measurements,
    };
    write_toml_report(&path, &report)?;
    Ok(path)
}

#[cfg(feature = "euclid-catalog")]
fn fraction(numer: usize, denom: usize) -> f64 {
    if denom == 0 {
        0.0
    } else {
        numer as f64 / denom as f64
    }
}

#[cfg(all(test, feature = "euclid-catalog"))]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn fraction_handles_zero_denominator() {
        assert_eq!(fraction(1, 0), 0.0);
        assert!((fraction(2, 4) - 0.5).abs() < 1.0e-12);
    }

    #[test]
    fn modality_labels_are_stable() {
        assert_eq!(modality_label(CatalogModality::SkyPoint), "sky_point");
        assert_eq!(modality_label(CatalogModality::NonSpatial), "non_spatial");
    }

    #[test]
    fn default_lotss_paths_match_release_labels() {
        assert!(default_lotss_path(ReleaseArg::Dr1).ends_with("lotss_dr1.fits"));
        assert!(default_lotss_path(ReleaseArg::Dr3).ends_with("lotss_dr3.fits"));
    }

    #[test]
    fn separation_stays_small_for_close_points() {
        let sep = data_core::angular_separation_arcsec(10.0, 10.0, 10.0001, 10.0001);
        assert!(sep < 1.0);
    }

    #[test]
    fn default_euclid_nocuts_paths_include_suffix() {
        assert!(
            default_euclid_lotss_csv_for_mode(ReleaseArg::Dr3, true)
                .ends_with("euclid_lotss_xmatch_dr3_nocuts.csv")
        );
        assert!(
            default_euclid_lotss_report_for_mode(ReleaseArg::Dr3, true)
                .to_string_lossy()
                .contains("euclid_lotss_xmatch_dr3_nocuts_")
        );
    }

    #[test]
    fn resolve_euclid_report_prefers_existing_nocuts_file() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_dir = std::env::temp_dir().join(format!("survey-crossmatch-test-{}", unique));
        fs::create_dir_all(&temp_dir).unwrap();
        let requested = temp_dir.join("euclid_lotss_xmatch_dr3.toml");
        let nocuts = temp_dir.join("euclid_lotss_xmatch_dr3_nocuts_2026-03-13.toml");
        fs::write(&nocuts, "matched_euclid_count = 0\n").unwrap();

        let resolved = resolve_euclid_lotss_report_path(ReleaseArg::Dr3, &requested);
        assert_eq!(resolved, nocuts);

        let _ = fs::remove_file(&nocuts);
        let _ = fs::remove_dir(&temp_dir);
    }
}
