//! LoTSS catalog download, footprint caching, and MaNGA crossmatch tool.
//!
//! Downloads LoTSS DR1/DR2 bulk FITS catalogs and performs VO Cone Search
//! queries for DR3 over the MaNGA sky footprint. Also joins the selected MaNGA
//! sample to DRPall coordinates and measures LoTSS detection fractions.
//!
//! Registered experiment: E-198 (LoTSS-MaNGA Kinematic Bisection)
//!
//! Usage:
//!   lotss-fetch download --release dr1|dr2|dr3
//!   lotss-fetch cone-search --release dr3 --ra-center <deg> --dec-center <deg> --radius <deg>
//!   lotss-fetch verify --release dr1|dr2|dr3
//!   lotss-fetch summary --input <fits-path>
//!   lotss-fetch manga-footprint [--tile-dir path] [--summary-out path]
//!       [--manga-selection path] [--manga-drpall path] [--full-bounding-box]
//!   lotss-fetch manga-preflight [--manga-selection path] [--manga-drpall path] [--report path]
//!   lotss-fetch crossmatch-manga --release dr1|dr2|dr3 [--input-format fits|dr3-tiles] [--input path]
//!       [--manga-selection path] [--manga-drpall path] [--radius-arcsec 3.0]
//!       [--output path] [--report path] [--summary path] [--allow-partial]

use clap::Parser;
use data_core::{
    SkyPoint,
    catalogs::lotss::{
        LoTSSRelease, LoTSSSource, LotssFitsExecutionReport,
        crossmatch_points_against_fits_catalog, load_from_votable,
    },
    download_stack::{DownloadStack, TransferRequest, TransferResult},
    fetcher::{compute_sha256, validate_not_html},
    formats::fits_table::{FitsValue, read_fits_table},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    path::{Path, PathBuf},
    thread,
    time::Duration,
};
use verified_core::topology::HardwareTopology;
use walkdir::WalkDir;

// CLI surface (Cli + Cmd subcommand + ReleaseArg/InputFormatArg
// ValueEnums + label() impls) lives in the `types` submodule.
// #[path] indirection because this binary has explicit Cargo.toml path.
#[path = "lotss_fetch/types.rs"]
mod types;
use types::*;

// ---- Constants ---------------------------------------------------------------

const DR1_URL: &str = "https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits";
const DR2_URL: &str =
    "https://lofar-surveys.org/public/DR2/catalogues/LoTSS_DR2_v110_masked.srl.fits";
const DR3_URL: &str = "https://lofar-surveys.org/public/DR3/catalogues/LoTSS_DR3_v1.0.srl.fits";
const DR3_CONE_BASE: &str = "https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml";

/// Expected source counts (approximate, used for validation sanity checks).
const DR1_EXPECTED_SOURCES: u64 = 325_694;
const DR2_EXPECTED_SOURCES: u64 = 4_396_228;
const MIN_PARALLEL_FITS_CHUNK_ROWS: usize = 10_000;
const MAX_PARALLEL_FITS_CHUNK_ROWS: usize = 250_000;
const NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW: usize = 128;

/// MaNGA sky footprint bounding box.
const MANGA_RA_MIN: f64 = 100.0;
const MANGA_RA_MAX: f64 = 260.0;
const MANGA_DEC_MIN: f64 = 0.0;
const MANGA_DEC_MAX: f64 = 70.0;

const DRPALL_COLUMNS: &[&str] = &["plateifu", "mangaid", "objra", "objdec"];

// ---- Main --------------------------------------------------------------------

fn main() {
    env_logger::init();
    let cli = Cli::parse();
    if let Err(e) = run(cli) {
        eprintln!("Error: {}", e);
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.cmd {
        Cmd::Download { release, output } => cmd_download(release, output),
        Cmd::ConeSearch {
            release,
            ra_center,
            dec_center,
            radius,
            output,
        } => cmd_cone_search(release, ra_center, dec_center, radius, output),
        Cmd::Verify { release, input } => cmd_verify(release, input),
        Cmd::Summary { input } => cmd_summary(input),
        Cmd::MangaFootprint {
            tile_dir,
            summary_out,
            manga_selection,
            manga_drpall,
            full_bounding_box,
            tile_radius,
            request_delay_ms,
        } => cmd_manga_footprint(
            tile_dir,
            summary_out,
            manga_selection,
            manga_drpall,
            full_bounding_box,
            tile_radius,
            request_delay_ms,
        ),
        Cmd::MangaPreflight {
            manga_selection,
            manga_drpall,
            report,
        } => cmd_manga_preflight(manga_selection, manga_drpall, report),
        Cmd::CrossmatchManga {
            release,
            input_format,
            input,
            manga_selection,
            manga_drpall,
            radius_arcsec,
            output,
            report,
            summary,
            allow_partial,
            workers,
            chunk_rows,
        } => cmd_crossmatch_manga(CrossmatchMangaArgs {
            release,
            input_format,
            input,
            manga_selection,
            manga_drpall,
            radius_arcsec,
            output,
            report,
            summary,
            allow_partial,
            workers,
            chunk_rows,
        }),
    }
}

// ---- Domain types ------------------------------------------------------------

#[derive(Debug, Clone)]
struct DrpallTarget {
    plateifu: String,
    mangaid: String,
    ra_deg: f64,
    dec_deg: f64,
}

struct CrossmatchMangaArgs {
    release: ReleaseArg,
    input_format: Option<InputFormatArg>,
    input: Option<PathBuf>,
    manga_selection: PathBuf,
    manga_drpall: PathBuf,
    radius_arcsec: f64,
    output: Option<PathBuf>,
    report: Option<PathBuf>,
    summary: Option<PathBuf>,
    allow_partial: bool,
    workers: Option<usize>,
    chunk_rows: Option<usize>,
}

#[derive(Debug, Clone)]
struct MangaTarget {
    plateifu: String,
    mangaid: String,
    ra_deg: f64,
    dec_deg: f64,
}

#[derive(Debug)]
struct SelectedMangaSample {
    targets: Vec<MangaTarget>,
    selection_row_count: usize,
    unique_plateifu_count: usize,
    duplicate_plateifu_count: usize,
    missing_plateifus: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Ord, PartialOrd)]
enum DeclinationBand {
    Below27,
    Deg27To30,
    Deg30To40,
    Deg40Plus,
}

impl DeclinationBand {
    fn label(self) -> &'static str {
        match self {
            Self::Below27 => "<27",
            Self::Deg27To30 => "27-30",
            Self::Deg30To40 => "30-40",
            Self::Deg40Plus => ">=40",
        }
    }

    fn ordered() -> [Self; 4] {
        [
            Self::Below27,
            Self::Deg27To30,
            Self::Deg30To40,
            Self::Deg40Plus,
        ]
    }
}

#[derive(Debug, Clone)]
struct MatchRecord {
    separation_arcsec: f64,
    source_name: String,
    source_ra_deg: f64,
    source_dec_deg: f64,
    flux_mjy: f32,
    spectral_index: Option<f32>,
    structure_code: char,
}

#[derive(Debug, Clone)]
struct TileLoadSummary {
    tile_file_count: usize,
    raw_source_count: usize,
    deduped_source_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct FootprintSweepReport {
    generated_at_utc: String,
    tile_dir: String,
    planning_mode: Option<String>,
    planning_target_count: Option<usize>,
    tile_count: u64,
    existing_tile_count: u64,
    downloaded_tile_count: u64,
    fail_count: u64,
    total_bytes_downloaded: u64,
    tile_radius_deg: f64,
    ra_min_deg: f64,
    ra_max_deg: f64,
    dec_min_deg: f64,
    dec_max_deg: f64,
}

#[derive(Debug, Serialize)]
struct BandFraction {
    band: String,
    count: usize,
    fraction: f64,
}

#[derive(Debug, Serialize)]
struct ThresholdFraction {
    threshold: String,
    count: usize,
    fraction: f64,
}

#[derive(Debug, Serialize)]
struct MangaPreflightReport {
    generated_at_utc: String,
    manga_selection_path: String,
    manga_drpall_path: String,
    selection_row_count: usize,
    unique_plateifu_count: usize,
    duplicate_plateifu_count: usize,
    matched_target_count: usize,
    missing_plateifu_count: usize,
    missing_plateifus: Vec<String>,
    ra_min_deg: f64,
    ra_max_deg: f64,
    dec_min_deg: f64,
    dec_max_deg: f64,
    declination_bands: Vec<BandFraction>,
    coverage_ceiling: Vec<ThresholdFraction>,
}

#[derive(Debug, Serialize)]
struct BandDetectionStats {
    band: String,
    total: usize,
    detected: usize,
    quiet: usize,
    detection_fraction: f64,
}

#[derive(Debug, Clone)]
struct CrossmatchExecutionPlan {
    mode: &'static str,
    worker_count: usize,
    chunk_rows: usize,
    physical_core_ids: Vec<usize>,
    l3_cache_bytes: usize,
    l3_safe_working_set_bytes: usize,
    pin_threads: bool,
    simd_lane_f64: usize,
    avx2_detected: bool,
    fma_detected: bool,
    x87_extended_precision_used: bool,
    precision_strategy: &'static str,
}

#[derive(Debug, Serialize)]
struct CrossmatchReport {
    generated_at_utc: String,
    release: String,
    input_format: String,
    input_path: String,
    manga_selection_path: String,
    manga_drpall_path: String,
    output_path: String,
    report_path: String,
    radius_arcsec: f64,
    allow_partial: bool,
    execution_mode: String,
    execution_worker_count: usize,
    execution_chunk_rows: usize,
    execution_pinned_core_ids: Vec<usize>,
    execution_l3_cache_bytes: usize,
    execution_l3_safe_working_set_bytes: usize,
    execution_thread_pinning_enabled: bool,
    execution_simd_lane_f64: usize,
    execution_avx2_detected: bool,
    execution_fma_detected: bool,
    execution_x87_extended_precision_used: bool,
    execution_precision_strategy: String,
    footprint_summary_path: Option<String>,
    footprint_fail_count: Option<u64>,
    footprint_tile_count: Option<u64>,
    footprint_downloaded_tile_count: Option<u64>,
    lotss_source_count_raw: usize,
    lotss_source_count_effective: usize,
    dr3_tile_file_count: Option<usize>,
    manga_target_count: usize,
    detected_target_count: usize,
    quiet_target_count: usize,
    detection_fraction: f64,
    flux_min_mjy: Option<f32>,
    flux_median_mjy: Option<f32>,
    flux_max_mjy: Option<f32>,
    declination_bands: Vec<BandDetectionStats>,
}

// ---- Download ----------------------------------------------------------------

fn default_bulk_catalog_path(release: ReleaseArg) -> Result<PathBuf, String> {
    let name = match release {
        ReleaseArg::Dr1 => "lotss_dr1.fits",
        ReleaseArg::Dr2 => "lotss_dr2.fits",
        ReleaseArg::Dr3 => "lotss_dr3.fits",
    };
    Ok(PathBuf::from("data/external/radio_surveys").join(name))
}

fn default_dr3_tile_dir() -> PathBuf {
    PathBuf::from("data/external/radio_surveys/dr3_tiles")
}

fn default_dr3_summary_path() -> PathBuf {
    PathBuf::from("reports/lotss_dr3_manga_footprint_summary.toml")
}

fn build_download_stack() -> DownloadStack {
    DownloadStack::new().with_user_agent("gororoba-lotss-fetch/0.1 (research)")
}

fn recover_with_capabilities(
    url: &str,
    output_path: &Path,
    note: impl Into<String>,
) -> Result<TransferResult, String> {
    let stack = build_download_stack();
    let mut request = TransferRequest::download(url.to_string(), output_path.to_path_buf());
    request.note = Some(note.into());
    let trace = stack.recover_with_trace(&request);
    let capabilities = trace.capabilities.clone();
    let result = trace.into_result(url).map_err(|err| err.to_string())?;
    if let Some(capabilities) = capabilities {
        println!(
            "Detected surface={} ranges={} rsync_reachable={} content_type={} backend={}",
            capabilities.surface,
            capabilities.supports_ranges,
            capabilities.rsync_reachable,
            capabilities.content_type.unwrap_or_default(),
            result.backend
        );
    }
    Ok(result)
}

fn cmd_download(release: ReleaseArg, output: Option<PathBuf>) -> Result<(), String> {
    let out_path = match output {
        Some(path) => path,
        None => default_bulk_catalog_path(release)?,
    };

    if out_path.exists() {
        println!("Already exists: {}", out_path.display());
        println!("Use --output to specify a different path, or delete the file to re-download.");
        return Ok(());
    }

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }

    let url = match release {
        ReleaseArg::Dr1 => DR1_URL,
        ReleaseArg::Dr2 => DR2_URL,
        ReleaseArg::Dr3 => DR3_URL,
    };

    println!("Downloading LoTSS {} from {}...", release.label(), url);

    let result = recover_with_capabilities(
        url,
        &out_path,
        format!("LoTSS {} bulk catalog", release.label()),
    )?;
    let bytes = result.bytes;

    println!("Saved {} bytes -> {}", bytes, out_path.display());

    let data = fs::read(&out_path).map_err(|e| format!("Read back failed: {}", e))?;
    validate_not_html(&data).map_err(|e| format!("Validation failed: {}", e))?;

    if data.len() >= 6 && &data[0..6] != b"SIMPLE" {
        return Err(format!(
            "File does not start with FITS magic 'SIMPLE': {}",
            out_path.display()
        ));
    }

    let sha = compute_sha256(&out_path).map_err(|e| format!("SHA256: {}", e))?;
    println!("SHA-256: {}", sha);

    let expected = match release {
        ReleaseArg::Dr1 => Some(DR1_EXPECTED_SOURCES),
        ReleaseArg::Dr2 => Some(DR2_EXPECTED_SOURCES),
        ReleaseArg::Dr3 => None,
    };
    if let Some(n) = expected {
        println!(
            "Expected ~{} sources. Run `verify` after download to confirm.",
            n
        );
    }

    println!("Done.");
    Ok(())
}

// ---- Cone search -------------------------------------------------------------

/// Build a VO SCS query URL with RA, Dec, SR parameters.
fn scs_url(base: &str, ra: f64, dec: f64, radius: f64) -> String {
    format!(
        "{}?RA={}&DEC={}&SR={}&FORMAT=votable/td",
        base, ra, dec, radius
    )
}

fn cmd_cone_search(
    release: ReleaseArg,
    ra_center: f64,
    dec_center: f64,
    radius: f64,
    output: Option<PathBuf>,
) -> Result<(), String> {
    if release != ReleaseArg::Dr3 {
        return Err(
            "Cone search is only available for DR3. Download DR1/DR2 with `download`.".to_string(),
        );
    }

    let url = scs_url(DR3_CONE_BASE, ra_center, dec_center, radius);
    let out_path = output.unwrap_or_else(|| {
        PathBuf::from("data/external/radio_surveys").join(format!(
            "lotss_dr3_tile_{:.2}_{:.2}.xml",
            ra_center, dec_center
        ))
    });

    if let Some(parent) = out_path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }

    println!(
        "LoTSS DR3 cone search: RA={} Dec={} r={} deg",
        ra_center, dec_center, radius
    );
    println!("URL: {}", url);

    let bytes = recover_with_capabilities(
        &url,
        &out_path,
        format!("LoTSS DR3 cone search RA={ra_center:.3} Dec={dec_center:.3}"),
    )?
    .bytes;
    let data = fs::read(&out_path).map_err(|e| format!("Read back failed: {}", e))?;
    validate_not_html(&data).map_err(|e| format!("Validation failed: {}", e))?;

    println!("Saved {} bytes -> {}", bytes, out_path.display());
    Ok(())
}

// ---- MaNGA footprint tile sweep ----------------------------------------------

fn cmd_manga_footprint(
    tile_dir: PathBuf,
    summary_out: PathBuf,
    manga_selection: PathBuf,
    manga_drpall: PathBuf,
    full_bounding_box: bool,
    tile_radius: f64,
    request_delay_ms: u64,
) -> Result<(), String> {
    if tile_radius <= 0.0 {
        return Err("tile_radius must be positive".to_string());
    }

    fs::create_dir_all(&tile_dir).map_err(|e| format!("mkdir {}: {}", tile_dir.display(), e))?;
    if let Some(parent) = summary_out.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }

    let (tile_centers, planning_mode, planning_target_count, bounds) = if full_bounding_box {
        (
            bounding_box_tile_centers(tile_radius),
            "bounding_box".to_string(),
            None,
            SkyBounds {
                ra_min_deg: MANGA_RA_MIN,
                ra_max_deg: MANGA_RA_MAX,
                dec_min_deg: MANGA_DEC_MIN,
                dec_max_deg: MANGA_DEC_MAX,
            },
        )
    } else {
        let sample = load_selected_manga_targets(&manga_selection, &manga_drpall)?;
        let bounds = compute_sky_bounds(&sample.targets)
            .ok_or_else(|| "Selected MaNGA sample is empty after DRPall join".to_string())?;
        (
            sample_covering_tile_centers(&sample.targets, tile_radius),
            "selected_sample_cover".to_string(),
            Some(sample.targets.len()),
            bounds,
        )
    };

    let mut tile_count: u64 = 0;
    let mut existing_tile_count: u64 = 0;
    let mut downloaded_tile_count: u64 = 0;
    let mut fail_count: u64 = 0;
    let mut total_bytes_downloaded: u64 = 0;

    for (ra, dec) in tile_centers {
        tile_count += 1;

        let tile_path = tile_dir.join(format!("tile_{:.2}_{:.2}.xml", ra, dec));
        if tile_path.exists() {
            existing_tile_count += 1;
            continue;
        }

        let url = scs_url(DR3_CONE_BASE, ra, dec, tile_radius);
        match recover_with_capabilities(
            &url,
            &tile_path,
            format!("LoTSS DR3 footprint tile RA={ra:.3} Dec={dec:.3}"),
        ) {
            Ok(result) => match fs::read(&tile_path) {
                Ok(data) => {
                    if let Err(e) = validate_not_html(&data) {
                        eprintln!(
                            "Tile RA={:.2} Dec={:.2} looked like HTML and will be counted as a failure: {}",
                            ra, dec, e
                        );
                        fail_count += 1;
                    } else {
                        downloaded_tile_count += 1;
                        total_bytes_downloaded += result.bytes;
                    }
                }
                Err(e) => {
                    eprintln!("Tile RA={:.2} Dec={:.2} readback failed: {}", ra, dec, e);
                    fail_count += 1;
                }
            },
            Err(e) => {
                eprintln!("Tile RA={:.2} Dec={:.2} failed: {}", ra, dec, e);
                fail_count += 1;
            }
        }
        if request_delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(request_delay_ms));
        }
    }

    let report = FootprintSweepReport {
        generated_at_utc: timestamp_utc(),
        tile_dir: tile_dir.display().to_string(),
        planning_mode: Some(planning_mode.clone()),
        planning_target_count,
        tile_count,
        existing_tile_count,
        downloaded_tile_count,
        fail_count,
        total_bytes_downloaded,
        tile_radius_deg: tile_radius,
        ra_min_deg: bounds.ra_min_deg,
        ra_max_deg: bounds.ra_max_deg,
        dec_min_deg: bounds.dec_min_deg,
        dec_max_deg: bounds.dec_max_deg,
    };
    write_toml_report(&summary_out, &report)?;

    println!(
        "MaNGA footprint sweep complete: {} tiles, {} downloaded, {} already present, {} failures",
        report.tile_count,
        report.downloaded_tile_count,
        report.existing_tile_count,
        report.fail_count
    );
    println!("Planning mode: {}", planning_mode);
    if let Some(target_count) = planning_target_count {
        println!("Selected targets covered: {}", target_count);
    }
    println!("Tile VOTable XML files stored in {}", tile_dir.display());
    println!("Summary written to {}", summary_out.display());

    Ok(())
}

// ---- MaNGA preflight ---------------------------------------------------------

fn cmd_manga_preflight(
    manga_selection: PathBuf,
    manga_drpall: PathBuf,
    report: Option<PathBuf>,
) -> Result<(), String> {
    let sample = load_selected_manga_targets(&manga_selection, &manga_drpall)?;
    let bounds = compute_sky_bounds(&sample.targets)
        .ok_or_else(|| "Selected MaNGA sample is empty after DRPall join".to_string())?;
    let declination_bands = band_fractions(sample.targets.len(), &band_counts(&sample.targets));
    let coverage_ceiling = threshold_fractions(&sample.targets);

    let report_path = report.unwrap_or_else(default_preflight_report_path);
    let preflight_report = MangaPreflightReport {
        generated_at_utc: timestamp_utc(),
        manga_selection_path: manga_selection.display().to_string(),
        manga_drpall_path: manga_drpall.display().to_string(),
        selection_row_count: sample.selection_row_count,
        unique_plateifu_count: sample.unique_plateifu_count,
        duplicate_plateifu_count: sample.duplicate_plateifu_count,
        matched_target_count: sample.targets.len(),
        missing_plateifu_count: sample.missing_plateifus.len(),
        missing_plateifus: sample.missing_plateifus.clone(),
        ra_min_deg: bounds.ra_min_deg,
        ra_max_deg: bounds.ra_max_deg,
        dec_min_deg: bounds.dec_min_deg,
        dec_max_deg: bounds.dec_max_deg,
        declination_bands,
        coverage_ceiling,
    };

    write_toml_report(&report_path, &preflight_report)?;

    println!(
        "Selected MaNGA rows: {}",
        preflight_report.selection_row_count
    );
    println!(
        "Unique plateifus:    {}",
        preflight_report.unique_plateifu_count
    );
    println!(
        "Matched targets:     {}",
        preflight_report.matched_target_count
    );
    println!(
        "Missing plateifus:   {}",
        preflight_report.missing_plateifu_count
    );
    for band in &preflight_report.declination_bands {
        println!(
            "Dec band {:>5}: {:>5} ({:.2}%)",
            band.band,
            band.count,
            band.fraction * 100.0
        );
    }
    for threshold in &preflight_report.coverage_ceiling {
        println!(
            "Dec {}: {:>5} ({:.2}%)",
            threshold.threshold,
            threshold.count,
            threshold.fraction * 100.0
        );
    }
    println!("Report written to {}", report_path.display());

    Ok(())
}

// ---- MaNGA crossmatch --------------------------------------------------------

fn cmd_crossmatch_manga(args: CrossmatchMangaArgs) -> Result<(), String> {
    let CrossmatchMangaArgs {
        release,
        input_format,
        input,
        manga_selection,
        manga_drpall,
        radius_arcsec,
        output,
        report,
        summary,
        allow_partial,
        workers,
        chunk_rows,
    } = args;
    if radius_arcsec <= 0.0 {
        return Err("radius_arcsec must be positive".to_string());
    }

    let resolved_format = resolve_input_format(release, input_format)?;
    let sample = load_selected_manga_targets(&manga_selection, &manga_drpall)?;
    if sample.targets.is_empty() {
        return Err("Selected MaNGA sample is empty after DRPall join".to_string());
    }
    let execution_plan = build_crossmatch_execution_plan(workers, chunk_rows);

    let output_path = output.unwrap_or_else(|| default_crossmatch_output_path(release));
    let report_path = report.unwrap_or_else(|| default_crossmatch_report_path(release));
    let summary_path = if matches!(resolved_format, InputFormatArg::Dr3Tiles) {
        Some(summary.unwrap_or_else(default_dr3_summary_path))
    } else {
        None
    };

    println!(
        "Crossmatch execution: mode={} workers={} chunk_rows={} pinned={} l3_safe_mb={:.1} simd_f64_lane={} avx2={} fma={} precision={}",
        execution_plan.mode,
        execution_plan.worker_count,
        execution_plan.chunk_rows,
        execution_plan.pin_threads,
        execution_plan.l3_safe_working_set_bytes as f64 / 1_048_576.0,
        execution_plan.simd_lane_f64,
        execution_plan.avx2_detected,
        execution_plan.fma_detected,
        execution_plan.precision_strategy
    );

    let mut footprint_report: Option<FootprintSweepReport> = None;
    let (
        matches,
        raw_source_count,
        effective_source_count,
        dr3_tile_count,
        input_path,
        shared_execution,
    ) = match resolved_format {
        InputFormatArg::Fits => {
            let input_path = match input {
                Some(path) => path,
                None => default_bulk_catalog_path(release)?,
            };
            let summary = crossmatch_targets_with_fits_catalog(
                &sample.targets,
                &input_path,
                release.as_catalog_release(),
                radius_arcsec,
                &execution_plan,
            )?;
            (
                summary.matches,
                summary.raw_source_count,
                summary.effective_source_count,
                None,
                input_path,
                summary.shared_execution,
            )
        }
        InputFormatArg::Dr3Tiles => {
            let input_path = input.unwrap_or_else(default_dr3_tile_dir);
            footprint_report = Some(load_or_validate_footprint_summary(
                summary_path
                    .as_deref()
                    .ok_or_else(|| "missing DR3 summary path".to_string())?,
                allow_partial,
            )?);
            let loaded = load_dr3_sources_from_tiles(&input_path)?;
            let matches = crossmatch_targets_with_sources_parallel(
                &sample.targets,
                &loaded.sources,
                radius_arcsec,
                &execution_plan,
            );
            (
                matches,
                loaded.summary.raw_source_count,
                loaded.summary.deduped_source_count,
                Some(loaded.summary.tile_file_count),
                input_path,
                None,
            )
        }
    };

    write_crossmatch_csv(&output_path, &sample.targets, &matches, release)?;

    let detected = matches.iter().filter(|m| m.is_some()).count();
    let quiet = sample.targets.len().saturating_sub(detected);
    let detection_fraction = fraction(detected, sample.targets.len());
    let flux_stats = matched_flux_stats(&matches);
    let shared_execution = shared_execution.as_ref();
    let report_model = CrossmatchReport {
        generated_at_utc: timestamp_utc(),
        release: release.label().to_string(),
        input_format: resolved_format.label().to_string(),
        input_path: input_path.display().to_string(),
        manga_selection_path: manga_selection.display().to_string(),
        manga_drpall_path: manga_drpall.display().to_string(),
        output_path: output_path.display().to_string(),
        report_path: report_path.display().to_string(),
        radius_arcsec,
        allow_partial,
        execution_mode: shared_execution
            .map(|report| report.mode.clone())
            .unwrap_or_else(|| execution_plan.mode.to_string()),
        execution_worker_count: shared_execution
            .map(|report| report.worker_count)
            .unwrap_or(execution_plan.worker_count),
        execution_chunk_rows: shared_execution
            .map(|report| report.chunk_rows)
            .unwrap_or(execution_plan.chunk_rows),
        execution_pinned_core_ids: shared_execution
            .map(|report| report.pinned_core_ids.clone())
            .unwrap_or_else(|| execution_plan.physical_core_ids.clone()),
        execution_l3_cache_bytes: shared_execution
            .map(|report| report.l3_cache_bytes)
            .unwrap_or(execution_plan.l3_cache_bytes),
        execution_l3_safe_working_set_bytes: shared_execution
            .map(|report| report.l3_safe_working_set_bytes)
            .unwrap_or(execution_plan.l3_safe_working_set_bytes),
        execution_thread_pinning_enabled: execution_plan.pin_threads,
        execution_simd_lane_f64: shared_execution
            .map(|report| report.simd_lane_f64)
            .unwrap_or(execution_plan.simd_lane_f64),
        execution_avx2_detected: shared_execution
            .map(|report| report.avx2_detected)
            .unwrap_or(execution_plan.avx2_detected),
        execution_fma_detected: shared_execution
            .map(|report| report.fma_detected)
            .unwrap_or(execution_plan.fma_detected),
        execution_x87_extended_precision_used: shared_execution
            .map(|report| report.x87_confirmation_used)
            .unwrap_or(execution_plan.x87_extended_precision_used),
        execution_precision_strategy: if shared_execution
            .map(|report| report.x87_confirmation_used)
            .unwrap_or(false)
        {
            "simd-prefilter+x87-confirmation".to_string()
        } else {
            execution_plan.precision_strategy.to_string()
        },
        footprint_summary_path: summary_path.as_ref().map(|path| path.display().to_string()),
        footprint_fail_count: footprint_report.as_ref().map(|r| r.fail_count),
        footprint_tile_count: footprint_report.as_ref().map(|r| r.tile_count),
        footprint_downloaded_tile_count: footprint_report.as_ref().map(|r| r.downloaded_tile_count),
        lotss_source_count_raw: raw_source_count,
        lotss_source_count_effective: effective_source_count,
        dr3_tile_file_count: dr3_tile_count,
        manga_target_count: sample.targets.len(),
        detected_target_count: detected,
        quiet_target_count: quiet,
        detection_fraction,
        flux_min_mjy: flux_stats.map(|stats| stats.0),
        flux_median_mjy: flux_stats.map(|stats| stats.1),
        flux_max_mjy: flux_stats.map(|stats| stats.2),
        declination_bands: band_detection_stats(&sample.targets, &matches),
    };
    write_toml_report(&report_path, &report_model)?;

    println!("MaNGA targets:   {}", report_model.manga_target_count);
    println!(
        "LoTSS matches:   {} ({:.2}%)",
        report_model.detected_target_count,
        report_model.detection_fraction * 100.0
    );
    println!(
        "Execution:       {} workers on cores {:?} | AVX2={} FMA={} | precision={}",
        report_model.execution_worker_count,
        report_model.execution_pinned_core_ids,
        report_model.execution_avx2_detected,
        report_model.execution_fma_detected,
        report_model.execution_precision_strategy
    );
    println!("Radio-quiet:     {}", report_model.quiet_target_count);
    println!("Crossmatch CSV:  {}", output_path.display());
    println!("Report written:  {}", report_path.display());

    Ok(())
}

// ---- Verify ------------------------------------------------------------------

fn cmd_verify(release: ReleaseArg, input: Option<PathBuf>) -> Result<(), String> {
    if release == ReleaseArg::Dr3 {
        return verify_dr3_cache(input.unwrap_or_else(default_dr3_tile_dir));
    }

    let path = match input {
        Some(path) => path,
        None => default_bulk_catalog_path(release)?,
    };

    if !path.exists() {
        return Err(format!(
            "File not found: {}\nRun `lotss-fetch download --release {}` first.",
            path.display(),
            release.label().to_ascii_lowercase()
        ));
    }

    let sha = compute_sha256(&path).map_err(|e| format!("SHA256: {}", e))?;
    let meta = fs::metadata(&path).map_err(|e| format!("metadata: {}", e))?;

    let mut header_bytes = vec![0u8; 80];
    let mut file = fs::File::open(&path).map_err(|e| format!("open: {}", e))?;
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("read header: {}", e))?;
    if &header_bytes[0..6] != b"SIMPLE" {
        return Err(format!(
            "Not a FITS file (magic mismatch): {}",
            path.display()
        ));
    }

    println!("Release:    {}", release.label());
    println!("Path:       {}", path.display());
    println!("Size:       {} bytes", meta.len());
    println!("SHA-256:    {}", sha);
    println!("FITS magic: OK (SIMPLE)");

    let expected = match release {
        ReleaseArg::Dr1 => Some(DR1_EXPECTED_SOURCES),
        ReleaseArg::Dr2 => Some(DR2_EXPECTED_SOURCES),
        ReleaseArg::Dr3 => None,
    };
    if let Some(n) = expected {
        println!("Expected sources: ~{}", n);
        println!(
            "Run `lotss-fetch summary --input {}` for exact count.",
            path.display()
        );
    }

    Ok(())
}

fn verify_dr3_cache(path: PathBuf) -> Result<(), String> {
    if !path.exists() {
        return Err(format!(
            "DR3 tile cache not found at {}. Run `lotss-fetch manga-footprint` first.",
            path.display()
        ));
    }

    if path.is_file() {
        let sha = compute_sha256(&path).map_err(|e| format!("SHA256: {}", e))?;
        let mut file = File::open(&path).map_err(|e| format!("open {}: {}", path.display(), e))?;
        let mut prefix = vec![0u8; 512];
        let bytes = file
            .read(&mut prefix)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        prefix.truncate(bytes);
        validate_not_html(&prefix).map_err(|e| format!("Validation failed: {}", e))?;
        let size = fs::metadata(&path)
            .map_err(|e| format!("metadata {}: {}", path.display(), e))?
            .len();
        println!("DR3 catalog file: {}", path.display());
        println!("Size:             {} bytes", size);
        println!("SHA-256:     {}", sha);
        return Ok(());
    }

    let mut tile_count = 0usize;
    let mut total_bytes = 0u64;
    for entry in WalkDir::new(&path).into_iter().filter_map(Result::ok) {
        let entry_path = entry.path();
        if entry.file_type().is_file() && is_votable_path(entry_path) {
            tile_count += 1;
            total_bytes += entry
                .metadata()
                .map_err(|e| format!("metadata {}: {}", entry_path.display(), e))?
                .len();
        }
    }

    println!("DR3 tile directory: {}", path.display());
    println!("Tile files:         {}", tile_count);
    println!("Total size:         {} bytes", total_bytes);

    Ok(())
}

// ---- Summary -----------------------------------------------------------------

fn cmd_summary(input: PathBuf) -> Result<(), String> {
    let meta = fs::metadata(&input).map_err(|e| format!("metadata: {}", e))?;
    println!("File: {}", input.display());
    println!(
        "Size: {} bytes ({:.1} MB)",
        meta.len(),
        meta.len() as f64 / 1_048_576.0
    );

    let naxis2 = first_bintable_naxis2(&input)?;
    match naxis2 {
        Some(n) => println!("NAXIS2 (row count): {}", n),
        None => println!("NAXIS2: could not parse from FITS header"),
    }

    let summary = format!(
        "input = {:?}\nsize_bytes = {}\nnaxis2 = {}\n",
        input.display(),
        meta.len(),
        naxis2.unwrap_or(0)
    );
    let summary_path_stem = input
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .into_owned();
    let summary_path =
        PathBuf::from("reports").join(format!("lotss_{}_summary.toml", summary_path_stem));
    if let Some(p) = summary_path.parent() {
        let _ = fs::create_dir_all(p);
    }
    fs::write(&summary_path, &summary).map_err(|e| format!("write summary: {}", e))?;
    println!("Summary written to {}", summary_path.display());

    Ok(())
}

fn first_bintable_naxis2(path: &Path) -> Result<Option<u64>, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("metadata {}: {}", path.display(), e))?
        .len();
    let mut hdu_offset = 0u64;

    while hdu_offset < file_len {
        let (cards, header_bytes) = read_fits_header_cards(&mut file, hdu_offset)?;
        let xtension = fits_header_value(&cards, "XTENSION").unwrap_or_default();
        if xtension.eq_ignore_ascii_case("BINTABLE") {
            return Ok(
                fits_header_value(&cards, "NAXIS2").and_then(|value| value.parse::<u64>().ok())
            );
        }
        let data_bytes = padded_hdu_data_bytes(&cards)?;
        let next_offset = hdu_offset + header_bytes + data_bytes;
        if next_offset <= hdu_offset {
            break;
        }
        hdu_offset = next_offset;
    }

    Ok(None)
}

// ---- MaNGA helpers -----------------------------------------------------------

fn load_selected_manga_targets(
    selection_path: &Path,
    drpall_path: &Path,
) -> Result<SelectedMangaSample, String> {
    let selection = load_selection_plateifus(selection_path)?;
    let drpall_by_plateifu = load_drpall_targets(drpall_path)?;
    let (targets, missing_plateifus) =
        join_selected_targets(&selection.plateifus, &drpall_by_plateifu);

    Ok(SelectedMangaSample {
        targets,
        selection_row_count: selection.row_count,
        unique_plateifu_count: selection.plateifus.len(),
        duplicate_plateifu_count: selection.duplicate_count,
        missing_plateifus,
    })
}

#[derive(Debug)]
struct SelectionPlateifus {
    row_count: usize,
    duplicate_count: usize,
    plateifus: Vec<String>,
}

fn load_selection_plateifus(path: &Path) -> Result<SelectionPlateifus, String> {
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let headers = reader
        .headers()
        .map_err(|e| format!("read headers {}: {}", path.display(), e))?
        .clone();
    let plateifu_idx = headers
        .iter()
        .position(|header| header.eq_ignore_ascii_case("plateifu"))
        .ok_or_else(|| format!("CSV {} does not contain a plateifu column", path.display()))?;

    let mut seen = HashSet::new();
    let mut plateifus = Vec::new();
    let mut row_count = 0usize;
    let mut duplicate_count = 0usize;

    for record in reader.records() {
        let record = record.map_err(|e| format!("read record {}: {}", path.display(), e))?;
        row_count += 1;
        let plateifu = record.get(plateifu_idx).unwrap_or("").trim();
        if plateifu.is_empty() {
            continue;
        }
        if seen.insert(plateifu.to_string()) {
            plateifus.push(plateifu.to_string());
        } else {
            duplicate_count += 1;
        }
    }

    Ok(SelectionPlateifus {
        row_count,
        duplicate_count,
        plateifus,
    })
}

fn load_drpall_targets(path: &Path) -> Result<HashMap<String, DrpallTarget>, String> {
    let rows = read_fits_table(path, DRPALL_COLUMNS)
        .map_err(|e| format!("read DRPall {}: {}", path.display(), e))?;
    let mut targets = HashMap::with_capacity(rows.len());

    for row in rows {
        let plateifu = row_string(&row, "PLATEIFU");
        if plateifu.is_empty() {
            continue;
        }
        let mangaid = row_string(&row, "MANGAID");
        let ra_deg = row_f64(&row, "OBJRA").unwrap_or(f64::NAN);
        let dec_deg = row_f64(&row, "OBJDEC").unwrap_or(f64::NAN);
        if !ra_deg.is_finite() || !dec_deg.is_finite() {
            continue;
        }
        targets.insert(
            plateifu.clone(),
            DrpallTarget {
                plateifu,
                mangaid,
                ra_deg,
                dec_deg,
            },
        );
    }

    Ok(targets)
}

fn join_selected_targets(
    selected_plateifus: &[String],
    drpall_by_plateifu: &HashMap<String, DrpallTarget>,
) -> (Vec<MangaTarget>, Vec<String>) {
    let mut targets = Vec::with_capacity(selected_plateifus.len());
    let mut missing = Vec::new();

    for plateifu in selected_plateifus {
        match drpall_by_plateifu.get(plateifu) {
            Some(drpall) => targets.push(MangaTarget {
                plateifu: drpall.plateifu.clone(),
                mangaid: drpall.mangaid.clone(),
                ra_deg: drpall.ra_deg,
                dec_deg: drpall.dec_deg,
            }),
            None => missing.push(plateifu.clone()),
        }
    }

    (targets, missing)
}

#[derive(Debug, Clone, Copy)]
struct SkyBounds {
    ra_min_deg: f64,
    ra_max_deg: f64,
    dec_min_deg: f64,
    dec_max_deg: f64,
}

fn compute_sky_bounds(targets: &[MangaTarget]) -> Option<SkyBounds> {
    let first = targets.first()?;
    let mut bounds = SkyBounds {
        ra_min_deg: first.ra_deg,
        ra_max_deg: first.ra_deg,
        dec_min_deg: first.dec_deg,
        dec_max_deg: first.dec_deg,
    };

    for target in targets.iter().skip(1) {
        bounds.ra_min_deg = bounds.ra_min_deg.min(target.ra_deg);
        bounds.ra_max_deg = bounds.ra_max_deg.max(target.ra_deg);
        bounds.dec_min_deg = bounds.dec_min_deg.min(target.dec_deg);
        bounds.dec_max_deg = bounds.dec_max_deg.max(target.dec_deg);
    }

    Some(bounds)
}

fn bounding_box_tile_centers(tile_radius: f64) -> Vec<(f64, f64)> {
    let tile_step = tile_radius * 2.0;
    let mut centers = Vec::new();

    let mut dec = MANGA_DEC_MIN + tile_radius;
    while dec <= MANGA_DEC_MAX {
        let cos_dec = dec.to_radians().cos().max(0.01);
        let ra_step = tile_step / cos_dec;
        let mut ra = MANGA_RA_MIN + tile_radius;
        while ra <= MANGA_RA_MAX {
            centers.push((ra, dec));
            ra += ra_step;
        }
        dec += tile_step;
    }

    centers
}

fn sample_covering_tile_centers(targets: &[MangaTarget], tile_radius: f64) -> Vec<(f64, f64)> {
    let mut ordered_targets: Vec<&MangaTarget> = targets.iter().collect();
    ordered_targets.sort_by(|left, right| {
        left.dec_deg
            .partial_cmp(&right.dec_deg)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                left.ra_deg
                    .partial_cmp(&right.ra_deg)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mut covered = vec![false; ordered_targets.len()];
    let mut centers = Vec::new();
    let tile_radius_arcsec = tile_radius * 3600.0;

    for idx in 0..ordered_targets.len() {
        if covered[idx] {
            continue;
        }

        let center = ordered_targets[idx];
        centers.push((center.ra_deg, center.dec_deg));

        for probe_idx in idx..ordered_targets.len() {
            if covered[probe_idx] {
                continue;
            }
            let target = ordered_targets[probe_idx];
            let separation_arcsec = angular_separation_arcsec(
                center.ra_deg,
                center.dec_deg,
                target.ra_deg,
                target.dec_deg,
            );
            if separation_arcsec <= tile_radius_arcsec {
                covered[probe_idx] = true;
            }
        }
    }

    centers
}

fn band_counts(targets: &[MangaTarget]) -> BTreeMap<DeclinationBand, usize> {
    let mut counts = BTreeMap::new();
    for band in DeclinationBand::ordered() {
        counts.insert(band, 0usize);
    }
    for target in targets {
        let band = declination_band(target.dec_deg);
        if let Some(count) = counts.get_mut(&band) {
            *count += 1;
        }
    }
    counts
}

fn band_fractions(total: usize, counts: &BTreeMap<DeclinationBand, usize>) -> Vec<BandFraction> {
    DeclinationBand::ordered()
        .into_iter()
        .map(|band| {
            let count = counts.get(&band).copied().unwrap_or(0);
            BandFraction {
                band: band.label().to_string(),
                count,
                fraction: fraction(count, total),
            }
        })
        .collect()
}

fn threshold_fractions(targets: &[MangaTarget]) -> Vec<ThresholdFraction> {
    [27.0, 30.0, 40.0]
        .into_iter()
        .map(|threshold| {
            let count = targets
                .iter()
                .filter(|target| target.dec_deg >= threshold)
                .count();
            ThresholdFraction {
                threshold: format!(">= {:.0}", threshold),
                count,
                fraction: fraction(count, targets.len()),
            }
        })
        .collect()
}

fn declination_band(dec_deg: f64) -> DeclinationBand {
    if dec_deg < 27.0 {
        DeclinationBand::Below27
    } else if dec_deg < 30.0 {
        DeclinationBand::Deg27To30
    } else if dec_deg < 40.0 {
        DeclinationBand::Deg30To40
    } else {
        DeclinationBand::Deg40Plus
    }
}

// ---- LoTSS loading helpers ---------------------------------------------------

struct LoadedSources {
    sources: Vec<LoTSSSource>,
    summary: TileLoadSummary,
}

fn load_dr3_sources_from_tiles(tile_dir: &Path) -> Result<LoadedSources, String> {
    if !tile_dir.exists() {
        return Err(format!(
            "DR3 tile directory does not exist: {}",
            tile_dir.display()
        ));
    }
    if !tile_dir.is_dir() {
        return Err(format!(
            "DR3 tile input is not a directory: {}",
            tile_dir.display()
        ));
    }

    let mut tile_paths: Vec<PathBuf> = WalkDir::new(tile_dir)
        .into_iter()
        .filter_map(Result::ok)
        .filter(|entry| entry.file_type().is_file() && is_votable_path(entry.path()))
        .map(|entry| entry.path().to_path_buf())
        .collect();
    tile_paths.sort();

    if tile_paths.is_empty() {
        return Err(format!(
            "No DR3 VOTable tiles found under {}",
            tile_dir.display()
        ));
    }

    let mut raw_sources = Vec::new();
    let mut raw_source_count = 0usize;

    for tile_path in &tile_paths {
        let data =
            fs::read(tile_path).map_err(|e| format!("read {}: {}", tile_path.display(), e))?;
        validate_not_html(&data).map_err(|e| {
            format!(
                "DR3 tile {} looks like HTML, not a VOTable: {}",
                tile_path.display(),
                e
            )
        })?;
        let xml =
            String::from_utf8(data).map_err(|e| format!("utf8 {}: {}", tile_path.display(), e))?;
        let mut tile_sources = load_from_votable(&xml, LoTSSRelease::DR3)
            .map_err(|e| format!("parse {}: {}", tile_path.display(), e))?;
        raw_source_count += tile_sources.len();
        raw_sources.append(&mut tile_sources);
    }

    let deduped_sources = dedupe_dr3_sources(raw_sources);
    let summary = TileLoadSummary {
        tile_file_count: tile_paths.len(),
        raw_source_count,
        deduped_source_count: deduped_sources.len(),
    };

    Ok(LoadedSources {
        sources: deduped_sources,
        summary,
    })
}

fn dedupe_dr3_sources(sources: Vec<LoTSSSource>) -> Vec<LoTSSSource> {
    let mut seen_names = HashSet::new();
    let mut seen_positions = HashSet::new();
    let mut deduped = Vec::with_capacity(sources.len());

    for source in sources {
        let name_key = source.source_name.trim().to_ascii_uppercase();
        let position_key = rounded_position_key(source.ra_deg, source.dec_deg);

        let duplicate_by_name = !name_key.is_empty() && seen_names.contains(&name_key);
        let duplicate_by_position = seen_positions.contains(&position_key);
        if duplicate_by_name || duplicate_by_position {
            continue;
        }

        if !name_key.is_empty() {
            seen_names.insert(name_key);
        }
        seen_positions.insert(position_key);
        deduped.push(source);
    }

    deduped
}

fn rounded_position_key(ra_deg: f64, dec_deg: f64) -> String {
    format!("{:.6}:{:.6}", ra_deg, dec_deg)
}

fn load_or_validate_footprint_summary(
    path: &Path,
    allow_partial: bool,
) -> Result<FootprintSweepReport, String> {
    if !path.exists() {
        if allow_partial {
            return Ok(FootprintSweepReport {
                generated_at_utc: timestamp_utc(),
                tile_dir: default_dr3_tile_dir().display().to_string(),
                planning_mode: None,
                planning_target_count: None,
                tile_count: 0,
                existing_tile_count: 0,
                downloaded_tile_count: 0,
                fail_count: 0,
                total_bytes_downloaded: 0,
                tile_radius_deg: 0.0,
                ra_min_deg: MANGA_RA_MIN,
                ra_max_deg: MANGA_RA_MAX,
                dec_min_deg: MANGA_DEC_MIN,
                dec_max_deg: MANGA_DEC_MAX,
            });
        }
        return Err(format!(
            "DR3 tile analysis requires a footprint summary at {}. Run `lotss-fetch manga-footprint` or pass --allow-partial.",
            path.display()
        ));
    }

    let text = fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;
    let report: FootprintSweepReport =
        toml::from_str(&text).map_err(|e| format!("parse {}: {}", path.display(), e))?;
    if report.fail_count > 0 && !allow_partial {
        return Err(format!(
            "DR3 footprint summary reports {} failed tiles in {}. Re-run `manga-footprint` or pass --allow-partial.",
            report.fail_count,
            path.display()
        ));
    }
    Ok(report)
}

fn is_votable_path(path: &Path) -> bool {
    matches!(
        path.extension()
            .and_then(|ext| ext.to_str())
            .map(|ext| ext.to_ascii_lowercase())
            .as_deref(),
        Some("xml") | Some("vot") | Some("votable")
    )
}

fn build_crossmatch_execution_plan(
    requested_workers: Option<usize>,
    chunk_rows_override: Option<usize>,
) -> CrossmatchExecutionPlan {
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
        .max(NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW)
        / worker_count.max(1);
    let auto_chunk_rows = (per_worker_bytes / NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW)
        .clamp(MIN_PARALLEL_FITS_CHUNK_ROWS, MAX_PARALLEL_FITS_CHUNK_ROWS);
    let alignment_rows = (simd_lane_f64 * 256).max(1);
    let chunk_rows = chunk_rows_override
        .unwrap_or(auto_chunk_rows)
        .clamp(MIN_PARALLEL_FITS_CHUNK_ROWS, MAX_PARALLEL_FITS_CHUNK_ROWS)
        / alignment_rows
        * alignment_rows;
    let chunk_rows = chunk_rows.max(MIN_PARALLEL_FITS_CHUNK_ROWS);

    CrossmatchExecutionPlan {
        mode: if worker_count > 1 {
            "pinned_physical"
        } else {
            "scalar"
        },
        worker_count,
        chunk_rows,
        physical_core_ids,
        l3_cache_bytes: topo.l3_cache_bytes,
        l3_safe_working_set_bytes: topo.l3_safe_working_set_bytes,
        pin_threads: worker_count > 1,
        simd_lane_f64,
        avx2_detected,
        fma_detected,
        x87_extended_precision_used: false,
        precision_strategy: "native-f64",
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_avx2() -> bool {
    false
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_fma() -> bool {
    std::arch::is_x86_feature_detected!("fma")
}

#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_fma() -> bool {
    false
}

fn preferred_f64_simd_lane(avx2_detected: bool) -> usize {
    if avx2_detected { 4 } else { 1 }
}

// ---- Crossmatch helpers ------------------------------------------------------

#[derive(Debug)]
struct TargetGrid {
    cell_size_deg: f64,
    cells: HashMap<(i32, i32), Vec<usize>>,
}

struct StreamingCrossmatchSummary {
    matches: Vec<Option<MatchRecord>>,
    raw_source_count: usize,
    effective_source_count: usize,
    shared_execution: Option<LotssFitsExecutionReport>,
}

fn crossmatch_targets_with_sources(
    targets: &[MangaTarget],
    sources: &[LoTSSSource],
    radius_arcsec: f64,
) -> Vec<Option<MatchRecord>> {
    let radius_deg = radius_arcsec / 3600.0;
    let grid = build_target_grid(targets, radius_deg);
    let mut best_matches: Vec<Option<MatchRecord>> = vec![None; targets.len()];

    update_matches_from_sources(targets, &grid, &mut best_matches, sources, radius_arcsec);
    best_matches
}

fn crossmatch_targets_with_sources_parallel(
    targets: &[MangaTarget],
    sources: &[LoTSSSource],
    radius_arcsec: f64,
    execution_plan: &CrossmatchExecutionPlan,
) -> Vec<Option<MatchRecord>> {
    if execution_plan.worker_count <= 1 || sources.len() < execution_plan.worker_count * 4_096 {
        return crossmatch_targets_with_sources(targets, sources, radius_arcsec);
    }

    let radius_deg = radius_arcsec / 3600.0;
    let grid = build_target_grid(targets, radius_deg);
    let bounds = split_work_bounds(sources.len(), execution_plan.worker_count);
    let pin_threads = execution_plan.pin_threads;
    let core_ids = execution_plan.physical_core_ids.clone();

    let local_results = thread::scope(|scope| {
        let mut handles = Vec::with_capacity(bounds.len());
        for (worker_idx, (start, end)) in bounds.iter().copied().enumerate() {
            let worker_sources = &sources[start..end];
            let grid_ref = &grid;
            let targets_ref = targets;
            let core_id = core_ids
                .get(worker_idx)
                .copied()
                .unwrap_or_else(|| *core_ids.first().unwrap_or(&0));
            handles.push(scope.spawn(move || {
                if pin_threads {
                    pin_current_thread_to_core(core_id);
                }
                let mut local = vec![None; targets_ref.len()];
                update_matches_from_sources(
                    targets_ref,
                    grid_ref,
                    &mut local,
                    worker_sources,
                    radius_arcsec,
                );
                local
            }));
        }

        handles
            .into_iter()
            .map(|handle| {
                handle
                    .join()
                    .expect("parallel source crossmatch worker panicked")
            })
            .collect::<Vec<_>>()
    });

    let mut merged = vec![None; targets.len()];
    for local in local_results {
        merge_match_vectors(&mut merged, &local);
    }
    merged
}

fn crossmatch_targets_with_fits_catalog(
    targets: &[MangaTarget],
    path: &Path,
    release: LoTSSRelease,
    radius_arcsec: f64,
    execution_plan: &CrossmatchExecutionPlan,
) -> Result<StreamingCrossmatchSummary, String> {
    let points = targets
        .iter()
        .map(|target| SkyPoint {
            id: target.plateifu.clone(),
            ra_deg: target.ra_deg,
            dec_deg: target.dec_deg,
        })
        .collect::<Vec<_>>();
    let shared = crossmatch_points_against_fits_catalog(
        path,
        release,
        &points,
        radius_arcsec,
        Some(execution_plan.worker_count),
        Some(execution_plan.chunk_rows),
    )
    .map_err(|e| e.to_string())?;
    let matches = shared
        .matches
        .into_iter()
        .map(|entry| {
            entry.map(|entry| MatchRecord {
                separation_arcsec: entry.separation_arcsec,
                source_name: entry.source.source_name,
                source_ra_deg: entry.source.ra_deg,
                source_dec_deg: entry.source.dec_deg,
                flux_mjy: entry.source.flux_mjy,
                spectral_index: entry.source.spectral_index,
                structure_code: entry.source.structure_code,
            })
        })
        .collect();

    Ok(StreamingCrossmatchSummary {
        matches,
        raw_source_count: shared.scanned_source_count,
        effective_source_count: shared.scanned_source_count,
        shared_execution: Some(shared.execution),
    })
}

fn split_work_bounds(len: usize, parts: usize) -> Vec<(usize, usize)> {
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

fn pin_current_thread_to_core(core_id: usize) {
    let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
}

fn merge_match_vectors(merged: &mut [Option<MatchRecord>], local: &[Option<MatchRecord>]) {
    for (slot, candidate) in merged.iter_mut().zip(local.iter().cloned()) {
        if let Some(candidate) = candidate {
            update_best_match(slot, candidate);
        }
    }
}

fn update_matches_from_sources(
    targets: &[MangaTarget],
    grid: &TargetGrid,
    best_matches: &mut [Option<MatchRecord>],
    sources: &[LoTSSSource],
    radius_arcsec: f64,
) {
    for source in sources {
        let (cell_x, cell_y) = grid_cell(
            projected_x(source.ra_deg, source.dec_deg),
            source.dec_deg,
            grid.cell_size_deg,
        );
        for dx in -1..=1 {
            for dy in -1..=1 {
                if let Some(target_indices) = grid.cells.get(&(cell_x + dx, cell_y + dy)) {
                    for &target_idx in target_indices {
                        let target = &targets[target_idx];
                        let separation_arcsec = angular_separation_arcsec(
                            target.ra_deg,
                            target.dec_deg,
                            source.ra_deg,
                            source.dec_deg,
                        );
                        if separation_arcsec <= radius_arcsec {
                            let candidate = MatchRecord {
                                separation_arcsec,
                                source_name: source.source_name.clone(),
                                source_ra_deg: source.ra_deg,
                                source_dec_deg: source.dec_deg,
                                flux_mjy: source.flux_mjy,
                                spectral_index: source.spectral_index,
                                structure_code: source.structure_code,
                            };
                            update_best_match(&mut best_matches[target_idx], candidate);
                        }
                    }
                }
            }
        }
    }
}

fn build_target_grid(targets: &[MangaTarget], cell_size_deg: f64) -> TargetGrid {
    let mut cells: HashMap<(i32, i32), Vec<usize>> = HashMap::new();
    for (idx, target) in targets.iter().enumerate() {
        let key = grid_cell(
            projected_x(target.ra_deg, target.dec_deg),
            target.dec_deg,
            cell_size_deg,
        );
        cells.entry(key).or_default().push(idx);
    }
    TargetGrid {
        cell_size_deg,
        cells,
    }
}

fn projected_x(ra_deg: f64, dec_deg: f64) -> f64 {
    ra_deg * dec_deg.to_radians().cos()
}

fn grid_cell(x_deg: f64, y_deg: f64, cell_size_deg: f64) -> (i32, i32) {
    (
        (x_deg / cell_size_deg).floor() as i32,
        (y_deg / cell_size_deg).floor() as i32,
    )
}

fn update_best_match(slot: &mut Option<MatchRecord>, candidate: MatchRecord) {
    match slot {
        Some(existing) if existing.separation_arcsec <= candidate.separation_arcsec => {}
        _ => *slot = Some(candidate),
    }
}

fn angular_separation_arcsec(ra1_deg: f64, dec1_deg: f64, ra2_deg: f64, dec2_deg: f64) -> f64 {
    let ra1 = ra1_deg.to_radians();
    let dec1 = dec1_deg.to_radians();
    let ra2 = ra2_deg.to_radians();
    let dec2 = dec2_deg.to_radians();
    let delta_ra = ra2 - ra1;
    let delta_dec = dec2 - dec1;
    let sin_ddec = (delta_dec / 2.0).sin();
    let sin_dra = (delta_ra / 2.0).sin();
    let sin_ddec_sq = sin_ddec * sin_ddec;
    let sin_dra_sq = sin_dra * sin_dra;
    let a = sin_dra_sq.mul_add(dec1.cos() * dec2.cos(), sin_ddec_sq);
    let c = 2.0 * a.sqrt().min(1.0).asin();
    c.to_degrees() * 3600.0
}

fn write_crossmatch_csv(
    path: &Path,
    targets: &[MangaTarget],
    matches: &[Option<MatchRecord>],
    release: ReleaseArg,
) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }
    let mut writer = csv::WriterBuilder::new()
        .has_headers(true)
        .from_path(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;

    for (target, matched) in targets.iter().zip(matches.iter()) {
        let row = CrossmatchCsvRow::from_parts(target, matched.as_ref(), release);
        writer
            .serialize(row)
            .map_err(|e| format!("write {}: {}", path.display(), e))?;
    }
    writer
        .flush()
        .map_err(|e| format!("flush {}: {}", path.display(), e))?;
    Ok(())
}

#[derive(Debug, Serialize)]
struct CrossmatchCsvRow {
    plateifu: String,
    mangaid: String,
    objra_deg: String,
    objdec_deg: String,
    declination_band: String,
    lotss_release: String,
    lotss_detected: u8,
    lotss_separation_arcsec: String,
    lotss_source_name: String,
    lotss_ra_deg: String,
    lotss_dec_deg: String,
    lotss_flux_mjy: String,
    lotss_spectral_index: String,
    lotss_structure_code: String,
}

impl CrossmatchCsvRow {
    fn from_parts(
        target: &MangaTarget,
        matched: Option<&MatchRecord>,
        release: ReleaseArg,
    ) -> Self {
        Self {
            plateifu: target.plateifu.clone(),
            mangaid: target.mangaid.clone(),
            objra_deg: format!("{:.6}", target.ra_deg),
            objdec_deg: format!("{:.6}", target.dec_deg),
            declination_band: declination_band(target.dec_deg).label().to_string(),
            lotss_release: release.label().to_string(),
            lotss_detected: u8::from(matched.is_some()),
            lotss_separation_arcsec: matched
                .map(|m| format!("{:.3}", m.separation_arcsec))
                .unwrap_or_default(),
            lotss_source_name: matched.map(|m| m.source_name.clone()).unwrap_or_default(),
            lotss_ra_deg: matched
                .map(|m| format!("{:.6}", m.source_ra_deg))
                .unwrap_or_default(),
            lotss_dec_deg: matched
                .map(|m| format!("{:.6}", m.source_dec_deg))
                .unwrap_or_default(),
            lotss_flux_mjy: matched
                .map(|m| format!("{:.4}", m.flux_mjy))
                .unwrap_or_default(),
            lotss_spectral_index: matched
                .and_then(|m| m.spectral_index)
                .map(|value| format!("{:.3}", value))
                .unwrap_or_default(),
            lotss_structure_code: matched
                .map(|m| m.structure_code.to_string())
                .unwrap_or_default(),
        }
    }
}

fn band_detection_stats(
    targets: &[MangaTarget],
    matches: &[Option<MatchRecord>],
) -> Vec<BandDetectionStats> {
    let mut counts: BTreeMap<DeclinationBand, (usize, usize)> = BTreeMap::new();
    for band in DeclinationBand::ordered() {
        counts.insert(band, (0usize, 0usize));
    }

    for (target, matched) in targets.iter().zip(matches.iter()) {
        let band = declination_band(target.dec_deg);
        let entry = counts.entry(band).or_default();
        entry.0 += 1;
        if matched.is_some() {
            entry.1 += 1;
        }
    }

    DeclinationBand::ordered()
        .into_iter()
        .map(|band| {
            let (total, detected) = counts.get(&band).copied().unwrap_or_default();
            BandDetectionStats {
                band: band.label().to_string(),
                total,
                detected,
                quiet: total.saturating_sub(detected),
                detection_fraction: fraction(detected, total),
            }
        })
        .collect()
}

fn matched_flux_stats(matches: &[Option<MatchRecord>]) -> Option<(f32, f32, f32)> {
    let mut fluxes: Vec<f32> = matches
        .iter()
        .filter_map(|matched| matched.as_ref().map(|m| m.flux_mjy))
        .filter(|flux| flux.is_finite())
        .collect();
    if fluxes.is_empty() {
        return None;
    }

    fluxes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let min_flux = *fluxes.first()?;
    let max_flux = *fluxes.last()?;
    let median_flux = median_sorted_f32(&fluxes)?;

    Some((min_flux, median_flux, max_flux))
}

fn median_sorted_f32(values: &[f32]) -> Option<f32> {
    if values.is_empty() {
        return None;
    }
    let mid = values.len() / 2;
    if values.len() % 2 == 1 {
        Some(values[mid])
    } else {
        Some((values[mid - 1] + values[mid]) / 2.0)
    }
}

// ---- Formatting helpers ------------------------------------------------------

fn resolve_input_format(
    release: ReleaseArg,
    input_format: Option<InputFormatArg>,
) -> Result<InputFormatArg, String> {
    match (
        release,
        input_format.unwrap_or(default_input_format(release)),
    ) {
        (ReleaseArg::Dr1, InputFormatArg::Fits) | (ReleaseArg::Dr2, InputFormatArg::Fits) => {
            Ok(InputFormatArg::Fits)
        }
        (ReleaseArg::Dr3, InputFormatArg::Fits) | (ReleaseArg::Dr3, InputFormatArg::Dr3Tiles) => {
            Ok(input_format.unwrap_or(InputFormatArg::Dr3Tiles))
        }
        (_, InputFormatArg::Dr3Tiles) => {
            Err("`--input-format dr3-tiles` is only valid with `--release dr3`.".to_string())
        }
    }
}

fn default_input_format(release: ReleaseArg) -> InputFormatArg {
    match release {
        ReleaseArg::Dr1 | ReleaseArg::Dr2 => InputFormatArg::Fits,
        ReleaseArg::Dr3 => InputFormatArg::Dr3Tiles,
    }
}

fn default_preflight_report_path() -> PathBuf {
    PathBuf::from("reports").join(format!("lotss_manga_preflight_{}.toml", today_stamp()))
}

fn default_crossmatch_output_path(release: ReleaseArg) -> PathBuf {
    PathBuf::from("data/external/manga").join(format!(
        "manga_lotss_xmatch_{}.csv",
        release.label().to_ascii_lowercase()
    ))
}

fn default_crossmatch_report_path(release: ReleaseArg) -> PathBuf {
    PathBuf::from("reports").join(format!(
        "lotss_manga_crossmatch_{}_{}.toml",
        release.label().to_ascii_lowercase(),
        today_stamp()
    ))
}

fn write_toml_report<T: Serialize>(path: &Path, value: &T) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }
    let rendered = toml::to_string_pretty(value)
        .map_err(|e| format!("serialize {}: {}", path.display(), e))?;
    fs::write(path, rendered).map_err(|e| format!("write {}: {}", path.display(), e))
}

fn row_string(row: &HashMap<String, FitsValue>, key: &str) -> String {
    row.get(key)
        .and_then(FitsValue::as_str)
        .map(str::trim)
        .unwrap_or_default()
        .to_string()
}

fn row_f64(row: &HashMap<String, FitsValue>, key: &str) -> Option<f64> {
    row.get(key).and_then(FitsValue::as_f64)
}

fn today_stamp() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}

fn timestamp_utc() -> String {
    chrono::Utc::now().to_rfc3339()
}

fn read_fits_header_cards(file: &mut File, offset: u64) -> Result<(Vec<String>, u64), String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|e| format!("seek header {}: {}", offset, e))?;

    let mut cards = Vec::new();
    let mut header_bytes = 0u64;
    loop {
        let mut block = [0u8; 2880];
        file.read_exact(&mut block)
            .map_err(|e| format!("read FITS header block at {}: {}", offset + header_bytes, e))?;
        header_bytes += block.len() as u64;
        for card in block.chunks(80) {
            let text = String::from_utf8_lossy(card).to_string();
            let is_end = text.get(0..8).unwrap_or("").trim() == "END";
            cards.push(text);
            if is_end {
                return Ok((cards, header_bytes));
            }
        }
    }
}

fn fits_header_value(cards: &[String], key: &str) -> Option<String> {
    cards.iter().find_map(|card| {
        let card_key = card.get(0..8).unwrap_or("").trim();
        if card_key != key {
            return None;
        }
        let (_, raw_value) = card.split_once('=')?;
        let value = raw_value.split('/').next().unwrap_or("").trim();
        Some(value.trim_matches('\'').trim().to_string())
    })
}

fn fits_header_usize(cards: &[String], key: &str) -> Result<usize, String> {
    fits_header_value(cards, key)
        .ok_or_else(|| format!("Missing FITS header key {}", key))?
        .parse::<usize>()
        .map_err(|e| format!("Parse FITS header key {}: {}", key, e))
}

fn padded_hdu_data_bytes(cards: &[String]) -> Result<u64, String> {
    let xtension = fits_header_value(cards, "XTENSION");
    let raw_bytes = if xtension
        .as_deref()
        .map(|value| value.eq_ignore_ascii_case("BINTABLE"))
        .unwrap_or(false)
    {
        let naxis1 = fits_header_usize(cards, "NAXIS1")? as u64;
        let naxis2 = fits_header_usize(cards, "NAXIS2")? as u64;
        let pcount = fits_header_value(cards, "PCOUNT")
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        naxis1 * naxis2 + pcount
    } else {
        let bitpix = fits_header_value(cards, "BITPIX")
            .ok_or_else(|| "Missing FITS header key BITPIX".to_string())?
            .parse::<i64>()
            .map_err(|e| format!("Parse FITS header key BITPIX: {}", e))?;
        let naxis = fits_header_value(cards, "NAXIS")
            .ok_or_else(|| "Missing FITS header key NAXIS".to_string())?
            .parse::<usize>()
            .map_err(|e| format!("Parse FITS header key NAXIS: {}", e))?;
        if naxis == 0 {
            0
        } else {
            let mut element_count = 1u64;
            for axis in 1..=naxis {
                let key = format!("NAXIS{axis}");
                let axis_len = fits_header_value(cards, &key)
                    .ok_or_else(|| format!("Missing FITS header key {}", key))?
                    .parse::<u64>()
                    .map_err(|e| format!("Parse FITS header key {}: {}", key, e))?;
                element_count = element_count.saturating_mul(axis_len);
            }
            let gcount = fits_header_value(cards, "GCOUNT")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(1);
            let pcount = fits_header_value(cards, "PCOUNT")
                .and_then(|value| value.parse::<u64>().ok())
                .unwrap_or(0);
            element_count
                .saturating_mul((bitpix.unsigned_abs() / 8).max(1))
                .saturating_mul(gcount)
                .saturating_add(pcount)
        }
    };
    Ok(raw_bytes.div_ceil(2880) * 2880)
}

fn fraction(numerator: usize, denominator: usize) -> f64 {
    if denominator == 0 {
        0.0
    } else {
        numerator as f64 / denominator as f64
    }
}

// ---- Tests -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use fitsio::FitsFile;

    fn sample_source(name: &str, ra_deg: f64, dec_deg: f64) -> LoTSSSource {
        LoTSSSource {
            source_name: name.to_string(),
            ra_deg,
            dec_deg,
            flux_mjy: 5.0,
            flux_err_mjy: 0.2,
            peak_flux: 4.8,
            local_rms: 0.1,
            spectral_index: Some(-0.7),
            spectral_index_err: Some(0.1),
            resolved: false,
            structure_code: 'S',
            maj_arcsec: 6.0,
            min_arcsec: 5.0,
            pa_deg: 30.0,
            release: LoTSSRelease::DR3,
        }
    }

    fn sample_target(plateifu: &str, ra_deg: f64, dec_deg: f64) -> MangaTarget {
        MangaTarget {
            plateifu: plateifu.to_string(),
            mangaid: format!("manga-{}", plateifu),
            ra_deg,
            dec_deg,
        }
    }

    #[test]
    fn declination_band_boundaries_are_stable() {
        assert_eq!(declination_band(0.0), DeclinationBand::Below27);
        assert_eq!(declination_band(26.999), DeclinationBand::Below27);
        assert_eq!(declination_band(27.0), DeclinationBand::Deg27To30);
        assert_eq!(declination_band(29.999), DeclinationBand::Deg27To30);
        assert_eq!(declination_band(30.0), DeclinationBand::Deg30To40);
        assert_eq!(declination_band(39.999), DeclinationBand::Deg30To40);
        assert_eq!(declination_band(40.0), DeclinationBand::Deg40Plus);
    }

    #[test]
    fn dr3_dedupe_prefers_unique_name_or_position() {
        let sources = vec![
            sample_source("ILTJ0001+0001", 150.0000001, 30.0000001),
            sample_source("ILTJ0001+0001", 150.0000002, 30.0000002),
            sample_source("ILTJ0002+0002", 150.5000001, 31.5000001),
            sample_source("ILTJ0003+0003", 150.5000002, 31.5000002),
        ];

        let deduped = dedupe_dr3_sources(sources);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].source_name, "ILTJ0001+0001");
        assert_eq!(deduped[1].source_name, "ILTJ0002+0002");
    }

    #[test]
    fn crossmatch_finds_nearest_source_within_radius() {
        let targets = vec![
            sample_target("10001-1901", 150.0, 32.0),
            sample_target("10002-1902", 151.0, 35.0),
        ];
        let sources = vec![
            sample_source("ILTJ1500+3200", 150.0002, 32.0001),
            sample_source("ILTJ1500+3200-far", 150.01, 32.01),
        ];

        let matches = crossmatch_targets_with_sources(&targets, &sources, 3.0);
        assert!(matches[0].is_some());
        assert!(matches[1].is_none());
        assert_eq!(
            matches[0].as_ref().map(|m| m.source_name.as_str()),
            Some("ILTJ1500+3200")
        );
    }

    #[test]
    fn incremental_source_updates_match_batch_crossmatch() {
        let targets = vec![
            sample_target("10001-1901", 150.0, 32.0),
            sample_target("10002-1902", 151.0, 35.0),
        ];
        let chunk_a = vec![sample_source("ILTJ1500+3200-far", 150.01, 32.01)];
        let chunk_b = vec![
            sample_source("ILTJ1500+3200", 150.0002, 32.0001),
            sample_source("ILTJ1510+3500", 151.0001, 35.0001),
        ];

        let mut all_sources = chunk_a.clone();
        all_sources.extend(chunk_b.clone());
        let batch = crossmatch_targets_with_sources(&targets, &all_sources, 3.0);

        let radius_deg = 3.0 / 3600.0;
        let grid = build_target_grid(&targets, radius_deg);
        let mut incremental = vec![None; targets.len()];
        update_matches_from_sources(&targets, &grid, &mut incremental, &chunk_a, 3.0);
        update_matches_from_sources(&targets, &grid, &mut incremental, &chunk_b, 3.0);

        assert_eq!(incremental.len(), batch.len());
        for (left, right) in incremental.iter().zip(batch.iter()) {
            assert_eq!(
                left.as_ref().map(|m| m.source_name.as_str()),
                right.as_ref().map(|m| m.source_name.as_str())
            );
        }
    }

    #[test]
    fn fits_crossmatch_reads_required_columns_without_spectral_index() {
        use fitsio::tables::{ColumnDataType, ColumnDescription};

        let tempdir = tempfile::tempdir().unwrap();
        let path = tempdir.path().join("lotss_small.fits");

        let mut fits = FitsFile::create(&path).open().unwrap();
        let columns = vec![
            ColumnDescription::new("Source_Name")
                .with_type(ColumnDataType::String)
                .that_repeats(32)
                .create()
                .unwrap(),
            ColumnDescription::new("RA")
                .with_type(ColumnDataType::Double)
                .create()
                .unwrap(),
            ColumnDescription::new("DEC")
                .with_type(ColumnDataType::Double)
                .create()
                .unwrap(),
            ColumnDescription::new("Total_flux")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
            ColumnDescription::new("E_Total_flux")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
            ColumnDescription::new("Peak_flux")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
            ColumnDescription::new("Isl_rms")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
            ColumnDescription::new("S_Code")
                .with_type(ColumnDataType::String)
                .that_repeats(1)
                .create()
                .unwrap(),
            ColumnDescription::new("Maj")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
            ColumnDescription::new("Min")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
            ColumnDescription::new("PA")
                .with_type(ColumnDataType::Float)
                .create()
                .unwrap(),
        ];
        let table_hdu = fits.create_table("LOTSS", &columns).unwrap();
        table_hdu
            .write_col(
                &mut fits,
                "Source_Name",
                &["ILTJ1500+3200".to_string(), "ILTJ1510+3500".to_string()],
            )
            .unwrap();
        table_hdu
            .write_col(&mut fits, "RA", &[150.0002_f64, 151.0002_f64])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "DEC", &[32.0001_f64, 35.0001_f64])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "Total_flux", &[3.5_f32, 7.2_f32])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "E_Total_flux", &[0.2_f32, 0.3_f32])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "Peak_flux", &[3.2_f32, 6.8_f32])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "Isl_rms", &[0.05_f32, 0.07_f32])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "S_Code", &["S".to_string(), "M".to_string()])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "Maj", &[6.0_f32, 9.0_f32])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "Min", &[5.0_f32, 7.0_f32])
            .unwrap();
        table_hdu
            .write_col(&mut fits, "PA", &[30.0_f32, 45.0_f32])
            .unwrap();
        drop(fits);

        let targets = vec![
            sample_target("10001-1901", 150.0, 32.0),
            sample_target("10002-1902", 151.0, 35.0),
        ];
        let execution_plan = build_crossmatch_execution_plan(Some(1), Some(10_000));
        let summary = crossmatch_targets_with_fits_catalog(
            &targets,
            &path,
            LoTSSRelease::DR2,
            3.0,
            &execution_plan,
        )
        .unwrap();

        assert_eq!(summary.raw_source_count, 2);
        assert_eq!(summary.effective_source_count, 2);
        assert_eq!(
            summary.matches[0].as_ref().map(|m| m.source_name.as_str()),
            Some("ILTJ1500+3200")
        );
        assert_eq!(
            summary.matches[1].as_ref().map(|m| m.structure_code),
            Some('M')
        );
        assert_eq!(
            summary.matches[0].as_ref().and_then(|m| m.spectral_index),
            None
        );
    }

    #[test]
    fn sample_covering_tiles_cover_every_target() {
        let targets = vec![
            sample_target("10001-1901", 10.0, 20.0),
            sample_target("10002-1902", 10.3, 20.2),
            sample_target("10003-1903", 30.0, 40.0),
        ];

        let tile_centers = sample_covering_tile_centers(&targets, 1.0);
        assert_eq!(tile_centers.len(), 2);

        for target in &targets {
            assert!(tile_centers.iter().any(|(ra, dec)| {
                angular_separation_arcsec(*ra, *dec, target.ra_deg, target.dec_deg) <= 3600.0
            }));
        }
    }

    #[test]
    fn join_selected_targets_reports_missing_plateifus() {
        let selected = vec!["8485-1901".to_string(), "8485-1902".to_string()];
        let mut drpall = HashMap::new();
        drpall.insert(
            "8485-1901".to_string(),
            DrpallTarget {
                plateifu: "8485-1901".to_string(),
                mangaid: "1-1".to_string(),
                ra_deg: 150.0,
                dec_deg: 32.0,
            },
        );

        let (targets, missing) = join_selected_targets(&selected, &drpall);
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].plateifu, "8485-1901");
        assert_eq!(missing, vec!["8485-1902".to_string()]);
    }

    #[test]
    fn strict_dr3_summary_rejects_failures() {
        let tempdir = tempfile::tempdir().unwrap();
        let summary_path = tempdir.path().join("summary.toml");
        let report = FootprintSweepReport {
            generated_at_utc: timestamp_utc(),
            tile_dir: tempdir.path().display().to_string(),
            planning_mode: Some("selected_sample_cover".to_string()),
            planning_target_count: Some(10),
            tile_count: 10,
            existing_tile_count: 0,
            downloaded_tile_count: 9,
            fail_count: 1,
            total_bytes_downloaded: 100,
            tile_radius_deg: 1.0,
            ra_min_deg: MANGA_RA_MIN,
            ra_max_deg: MANGA_RA_MAX,
            dec_min_deg: MANGA_DEC_MIN,
            dec_max_deg: MANGA_DEC_MAX,
        };
        write_toml_report(&summary_path, &report).unwrap();

        let error = load_or_validate_footprint_summary(&summary_path, false).unwrap_err();
        assert!(error.contains("failed tiles"));
        assert!(load_or_validate_footprint_summary(&summary_path, true).is_ok());
    }
}
