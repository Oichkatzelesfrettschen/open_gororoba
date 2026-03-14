//! LoTSS (LOFAR Two-metre Sky Survey) source catalog parser.
//!
//! Supports three releases:
//! - DR1: ~325K sources, Shimwell et al. 2019, A&A 622 A1
//!   https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits
//! - DR2: ~4.4M sources, Shimwell et al. 2022, A&A 659 A1
//!   https://lofar-surveys.org/public/LoTSS_DR2_v110.srl.fits
//! - DR3: ~13.7M sources (Feb 2026), 88% northern sky at 144 MHz, 6" resolution
//!   VO Cone Search: https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml
//!
//! DR1/DR2 bulk catalogs are FITS BINTABLE; DR3 is accessed via VO Cone Search
//! returning VOTable XML or FITS depending on the requested format.
//!
//! Column correspondence (DR1/DR2 and DR3 share the same names in the
//! source-catalogue extensions):
//!   Source_Name, RA, DEC, Total_flux, E_Total_flux, Peak_flux, Isl_rms,
//!   Spectral_index, E_spectral_index, Resolved, S_Code, Maj, Min, PA
//!
//! Feature-gated behind `data_core/fits`.

use crate::{
    fetcher::FetchError,
    spatial::{SkyGridIndex, SkyPoint, angular_separation_arcsec},
};
#[cfg(feature = "fits")]
use rayon::prelude::*;
use std::{collections::HashMap, ops::Range, path::Path, time::Instant};
#[cfg(feature = "fits")]
use wide::f64x4;
#[cfg(feature = "fits")]
use verified_core::topology::HardwareTopology;
#[cfg(all(feature = "fits", target_arch = "x86_64"))]
use cd_kernel::angular_separation_arcsec_ext80_deg;

/// LoTSS catalog release identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoTSSRelease {
    DR1,
    DR2,
    DR3,
}

impl LoTSSRelease {
    /// Human-readable label.
    pub fn label(self) -> &'static str {
        match self {
            LoTSSRelease::DR1 => "DR1",
            LoTSSRelease::DR2 => "DR2",
            LoTSSRelease::DR3 => "DR3",
        }
    }
}

/// A single radio source from a LoTSS catalog.
#[derive(Debug, Clone)]
pub struct LoTSSSource {
    /// IAU source name (e.g. "ILTJ000001.1+123456").
    pub source_name: String,
    /// Right ascension (degrees, J2000).
    pub ra_deg: f64,
    /// Declination (degrees, J2000).
    pub dec_deg: f64,
    /// Integrated 144 MHz flux density (mJy).
    pub flux_mjy: f32,
    /// Uncertainty on integrated flux (mJy).
    pub flux_err_mjy: f32,
    /// Peak flux density (mJy/beam).
    pub peak_flux: f32,
    /// Local RMS noise (mJy/beam).
    pub local_rms: f32,
    /// In-band spectral index (None if not measured).
    pub spectral_index: Option<f32>,
    /// Uncertainty on spectral index.
    pub spectral_index_err: Option<f32>,
    /// True if the source is spatially resolved.
    pub resolved: bool,
    /// Morphology code: 'S' single, 'C' overlapping Gaussians, 'M' multi-component.
    pub structure_code: char,
    /// Major axis FWHM (arcsec).
    pub maj_arcsec: f32,
    /// Minor axis FWHM (arcsec).
    pub min_arcsec: f32,
    /// Position angle of major axis (degrees, N through E).
    pub pa_deg: f32,
    /// Catalog release this source comes from.
    pub release: LoTSSRelease,
}

/// Execution metadata for pinned multicore LoTSS FITS point crossmatches.
#[cfg(feature = "fits")]
#[derive(Debug, Clone)]
pub struct LotssFitsExecutionReport {
    pub mode: String,
    pub worker_count: usize,
    pub chunk_rows: usize,
    pub pinned_core_ids: Vec<usize>,
    pub l3_cache_bytes: usize,
    pub l3_safe_working_set_bytes: usize,
    pub avx2_detected: bool,
    pub fma_detected: bool,
    pub simd_lane_f64: usize,
    pub x87_confirmation_used: bool,
    pub scan_wall_seconds: f64,
}

/// Best-match record for one sky point against a LoTSS FITS catalog.
#[cfg(feature = "fits")]
#[derive(Debug, Clone)]
pub struct LotssFitsBestMatch {
    pub source_row_index: usize,
    pub separation_arcsec: f64,
    pub source: LoTSSSource,
}

/// Summary for a point-to-LoTSS FITS best-match scan.
#[cfg(feature = "fits")]
#[derive(Debug, Clone)]
pub struct LotssFitsBestMatchSummary {
    pub matches: Vec<Option<LotssFitsBestMatch>>,
    pub scanned_source_count: usize,
    pub execution: LotssFitsExecutionReport,
}

/// Columns commonly present across LoTSS releases.
///
/// DR2 masked source catalogs omit some value-added columns like
/// `Spectral_index` and `Resolved`, so those are treated as optional by the
/// FITS loader even though DR3 cone-search VOTables usually include them.
const LOTSS_FITS_COLUMNS: &[&str] = &[
    "Source_Name",
    "RA",
    "DEC",
    "Total_flux",
    "E_Total_flux",
    "Peak_flux",
    "Isl_rms",
    "S_Code",
    "Maj",
    "Min",
    "PA",
    "Spectral_index",
    "E_spectral_index",
    "Resolved",
];

#[cfg(feature = "fits")]
const LOTSS_NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW: usize = 16;
#[cfg(feature = "fits")]
const LOTSS_MIN_PARALLEL_FITS_CHUNK_ROWS: usize = 65_536;
#[cfg(feature = "fits")]
const LOTSS_MAX_PARALLEL_FITS_CHUNK_ROWS: usize = 1_048_576;

/// Load LoTSS sources from a FITS BINTABLE file (DR1 or DR2 bulk download).
///
/// The FITS file should be the `*.srl.fits` source-list file distributed by
/// lofar-surveys.org.
#[cfg(feature = "fits")]
pub fn load_from_fits(path: &Path, release: LoTSSRelease) -> Result<Vec<LoTSSSource>, FetchError> {
    let row_count = lotss_fits_row_count(path)?;
    load_from_fits_range(path, release, 0..row_count)
}

/// Count rows in the first BINTABLE HDU of a LoTSS FITS catalog.
#[cfg(feature = "fits")]
pub fn lotss_fits_row_count(path: &Path) -> Result<usize, FetchError> {
    use fitsio::{FitsFile, hdu::HduInfo};

    let mut fptr = FitsFile::open(path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", path.display(), e)))?;

    let num_hdus = {
        let mut count = 0usize;
        for _ in fptr.iter() {
            count += 1;
        }
        count
    };

    let mut table_idx: Option<usize> = None;
    let mut column_names: Vec<String> = Vec::new();
    let mut row_count: Option<usize> = None;

    for idx in 1..num_hdus {
        let hdu = fptr
            .hdu(idx)
            .map_err(|e| FetchError::Validation(format!("hdu {}: {}", idx, e)))?;
        if let HduInfo::TableInfo {
            column_descriptions,
            num_rows,
            ..
        } = hdu.info
        {
            table_idx = Some(idx);
            column_names = column_descriptions.iter().map(|d| d.name.clone()).collect();
            row_count = Some(num_rows);
            break;
        }
    }

    let table_idx = table_idx
        .ok_or_else(|| FetchError::Validation("No BINTABLE HDU found in FITS file".to_string()))?;

    let _ = table_idx;
    let _ = column_names;
    row_count.ok_or_else(|| FetchError::Validation("No row count found in FITS table".to_string()))
}

/// Load a range of LoTSS sources from a FITS BINTABLE file.
#[cfg(feature = "fits")]
pub fn load_from_fits_range(
    path: &Path,
    release: LoTSSRelease,
    row_range: Range<usize>,
) -> Result<Vec<LoTSSSource>, FetchError> {
    use fitsio::{FitsFile, hdu::HduInfo};

    let mut fptr = FitsFile::open(path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", path.display(), e)))?;

    let num_hdus = {
        let mut count = 0usize;
        for _ in fptr.iter() {
            count += 1;
        }
        count
    };

    let mut table_idx: Option<usize> = None;
    let mut column_names: Vec<String> = Vec::new();
    let mut table_row_count: Option<usize> = None;

    for idx in 1..num_hdus {
        let hdu = fptr
            .hdu(idx)
            .map_err(|e| FetchError::Validation(format!("hdu {}: {}", idx, e)))?;
        if let HduInfo::TableInfo {
            column_descriptions,
            num_rows,
            ..
        } = hdu.info
        {
            table_idx = Some(idx);
            column_names = column_descriptions.iter().map(|d| d.name.clone()).collect();
            table_row_count = Some(num_rows);
            break;
        }
    }

    let table_idx = table_idx
        .ok_or_else(|| FetchError::Validation("No BINTABLE HDU found in FITS file".to_string()))?;
    let table_row_count = table_row_count.unwrap_or(0);
    if row_range.start > row_range.end {
        return Err(FetchError::Validation(format!(
            "Invalid FITS row range {}..{}",
            row_range.start, row_range.end
        )));
    }
    let bounded_range = row_range.start.min(table_row_count)..row_range.end.min(table_row_count);
    if bounded_range.is_empty() {
        return Ok(Vec::new());
    }

    let resolve = |want: &str| -> Result<String, FetchError> {
        column_names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(want))
            .cloned()
            .ok_or_else(|| {
                FetchError::Validation(format!(
                    "Required FITS column '{}' not found in {}",
                    want,
                    path.display()
                ))
            })
    };
    let resolve_optional = |want: &str| -> Option<String> {
        column_names
            .iter()
            .find(|name| name.eq_ignore_ascii_case(want))
            .cloned()
    };

    let source_name_col = resolve("Source_Name")?;
    let ra_col = resolve("RA")?;
    let dec_col = resolve("DEC")?;
    let total_flux_col = resolve("Total_flux")?;
    let e_total_flux_col = resolve("E_Total_flux")?;
    let peak_flux_col = resolve("Peak_flux")?;
    let isl_rms_col = resolve("Isl_rms")?;
    let s_code_col = resolve("S_Code")?;
    let maj_col = resolve("Maj")?;
    let min_col = resolve("Min")?;
    let pa_col = resolve("PA")?;
    let spectral_index_col = resolve_optional("Spectral_index");
    let e_spectral_index_col = resolve_optional("E_spectral_index");
    let resolved_col = resolve_optional("Resolved");

    let hdu = fptr
        .hdu(table_idx)
        .map_err(|e| FetchError::Validation(format!("hdu {}: {}", table_idx, e)))?;

    let source_name: Vec<String> = hdu
        .read_col_range(&mut fptr, &source_name_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", source_name_col, e)))?;
    let ra: Vec<f64> = hdu
        .read_col_range(&mut fptr, &ra_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", ra_col, e)))?;
    let dec: Vec<f64> = hdu
        .read_col_range(&mut fptr, &dec_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", dec_col, e)))?;
    let flux_mjy: Vec<f32> = hdu
        .read_col_range(&mut fptr, &total_flux_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", total_flux_col, e)))?;
    let flux_err_mjy: Vec<f32> = hdu
        .read_col_range(&mut fptr, &e_total_flux_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", e_total_flux_col, e)))?;
    let peak_flux: Vec<f32> = hdu
        .read_col_range(&mut fptr, &peak_flux_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", peak_flux_col, e)))?;
    let local_rms: Vec<f32> = hdu
        .read_col_range(&mut fptr, &isl_rms_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", isl_rms_col, e)))?;
    let s_code: Vec<String> = hdu
        .read_col_range(&mut fptr, &s_code_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", s_code_col, e)))?;
    let maj_arcsec: Vec<f32> = hdu
        .read_col_range(&mut fptr, &maj_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", maj_col, e)))?;
    let min_arcsec: Vec<f32> = hdu
        .read_col_range(&mut fptr, &min_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", min_col, e)))?;
    let pa_deg: Vec<f32> = hdu
        .read_col_range(&mut fptr, &pa_col, &bounded_range)
        .map_err(|e| FetchError::Validation(format!("col {}: {}", pa_col, e)))?;

    let len = ra.len();
    let spectral_index: Vec<f32> = if let Some(col) = spectral_index_col.as_ref() {
        hdu.read_col_range(&mut fptr, col, &bounded_range)
            .map_err(|e| FetchError::Validation(format!("col {}: {}", col, e)))?
    } else {
        vec![f32::NAN; len]
    };
    let spectral_index_err: Vec<f32> = if let Some(col) = e_spectral_index_col.as_ref() {
        hdu.read_col_range(&mut fptr, col, &bounded_range)
            .map_err(|e| FetchError::Validation(format!("col {}: {}", col, e)))?
    } else {
        vec![f32::NAN; len]
    };
    let resolved_raw: Vec<i32> = if let Some(col) = resolved_col.as_ref() {
        hdu.read_col_range(&mut fptr, col, &bounded_range)
            .map_err(|e| FetchError::Validation(format!("col {}: {}", col, e)))?
    } else {
        vec![0; len]
    };
    let mut sources = Vec::with_capacity(len);

    for idx in 0..len {
        let source = source_name
            .get(idx)
            .map(|s| s.trim().to_string())
            .unwrap_or_default();
        if source.is_empty() {
            continue;
        }
        let structure_code = s_code
            .get(idx)
            .and_then(|s| s.trim().chars().next())
            .unwrap_or('S');
        let spectral = spectral_index.get(idx).copied().filter(|v| v.is_finite());
        let spectral_err = spectral_index_err
            .get(idx)
            .copied()
            .filter(|v| v.is_finite());

        sources.push(LoTSSSource {
            source_name: source,
            ra_deg: *ra.get(idx).unwrap_or(&f64::NAN),
            dec_deg: *dec.get(idx).unwrap_or(&f64::NAN),
            flux_mjy: *flux_mjy.get(idx).unwrap_or(&f32::NAN),
            flux_err_mjy: *flux_err_mjy.get(idx).unwrap_or(&f32::NAN),
            peak_flux: *peak_flux.get(idx).unwrap_or(&f32::NAN),
            local_rms: *local_rms.get(idx).unwrap_or(&f32::NAN),
            spectral_index: spectral,
            spectral_index_err: spectral_err,
            resolved: resolved_raw.get(idx).copied().unwrap_or_default() != 0,
            structure_code,
            maj_arcsec: *maj_arcsec.get(idx).unwrap_or(&f32::NAN),
            min_arcsec: *min_arcsec.get(idx).unwrap_or(&f32::NAN),
            pa_deg: *pa_deg.get(idx).unwrap_or(&f32::NAN),
            release,
        });
    }

    Ok(sources)
}

/// Load LoTSS sources from a VOTable XML string (DR3 VO Cone Search response).
///
/// The `xml` argument should be the raw response body from a VO Cone Search
/// request to `https://vo.astron.nl/lotss_dr3/q/src_cone/scs.xml`.
#[cfg(feature = "fits")]
pub fn load_from_votable(xml: &str, release: LoTSSRelease) -> Result<Vec<LoTSSSource>, FetchError> {
    use crate::formats::votable::parse_votable;

    let rows = parse_votable(xml, LOTSS_FITS_COLUMNS)?;
    let mut sources = Vec::with_capacity(rows.len());

    for row in rows {
        if let Some(src) = parse_row_from_votable(&row, release) {
            sources.push(src);
        }
    }

    Ok(sources)
}

/// Crossmatch sky points against a LoTSS FITS catalog using a pinned multicore
/// broad-phase scan plus precise angular confirmation.
#[cfg(feature = "fits")]
pub fn crossmatch_points_against_fits_catalog(
    path: &Path,
    release: LoTSSRelease,
    points: &[SkyPoint],
    match_radius_arcsec: f64,
    requested_workers: Option<usize>,
    chunk_rows_override: Option<usize>,
) -> Result<LotssFitsBestMatchSummary, FetchError> {
    if match_radius_arcsec <= 0.0 {
        return Err(FetchError::Validation(
            "match_radius_arcsec must be positive".to_string(),
        ));
    }
    if points.is_empty() {
        return Ok(LotssFitsBestMatchSummary {
            matches: Vec::new(),
            scanned_source_count: 0,
            execution: LotssFitsExecutionReport {
                mode: "empty".to_string(),
                worker_count: 0,
                chunk_rows: 0,
                pinned_core_ids: Vec::new(),
                l3_cache_bytes: 0,
                l3_safe_working_set_bytes: 0,
                avx2_detected: detect_avx2(),
                fma_detected: detect_fma(),
                simd_lane_f64: preferred_f64_simd_lane(detect_avx2()),
                x87_confirmation_used: cfg!(target_arch = "x86_64"),
                scan_wall_seconds: 0.0,
            },
        });
    }

    let execution_plan = build_lotss_execution_plan(requested_workers, chunk_rows_override);
    let prepared = prepare_point_grid(points, match_radius_arcsec);
    let layout = inspect_lotss_position_layout(path)?;
    let bounds = split_lotss_work_bounds(layout.row_count, execution_plan.worker_count);
    let core_ids = execution_plan.physical_core_ids.clone();
    let scan_started = Instant::now();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(execution_plan.worker_count.max(1))
        .start_handler(move |idx| {
            if let Some(&core_id) = core_ids.get(idx) {
                let _ = core_affinity::set_for_current(core_affinity::CoreId { id: core_id });
            }
        })
        .build()
        .map_err(|e| FetchError::Validation(format!("build LoTSS rayon pool: {}", e)))?;

    let worker_summaries = pool.install(|| {
        bounds
            .into_par_iter()
            .map(|(start, end)| {
                scan_lotss_best_match_range(
                    path,
                    &layout,
                    &prepared,
                    match_radius_arcsec,
                    start..end,
                    execution_plan.chunk_rows,
                )
            })
            .collect::<Vec<_>>()
    });

    let mut merged = vec![None; prepared.point_count];
    for summary in worker_summaries {
        let summary = summary?;
        merge_pending_best_match_vectors(&mut merged, &summary.matches);
    }
    let matches = finalize_pending_lotss_best_matches(path, release, merged)?;
    let scan_wall_seconds = scan_started.elapsed().as_secs_f64();
    Ok(LotssFitsBestMatchSummary {
        matches,
        scanned_source_count: layout.row_count,
        execution: LotssFitsExecutionReport {
            mode: execution_plan.mode.to_string(),
            worker_count: execution_plan.worker_count,
            chunk_rows: execution_plan.chunk_rows,
            pinned_core_ids: execution_plan.physical_core_ids,
            l3_cache_bytes: execution_plan.l3_cache_bytes,
            l3_safe_working_set_bytes: execution_plan.l3_safe_working_set_bytes,
            avx2_detected: execution_plan.avx2_detected,
            fma_detected: execution_plan.fma_detected,
            simd_lane_f64: execution_plan.simd_lane_f64,
            x87_confirmation_used: cfg!(target_arch = "x86_64"),
            scan_wall_seconds,
        },
    })
}

#[cfg(feature = "fits")]
#[derive(Debug)]
struct LotssExecutionPlan {
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

#[cfg(feature = "fits")]
#[derive(Debug)]
struct LotssPositionLayout {
    table_idx: usize,
    row_count: usize,
    ra_column: String,
    dec_column: String,
}

#[cfg(feature = "fits")]
#[derive(Debug)]
struct PreparedLotssPointGrid {
    point_count: usize,
    grid: SkyGridIndex,
    unit_x: Vec<f64>,
    unit_y: Vec<f64>,
    unit_z: Vec<f64>,
}

#[cfg(feature = "fits")]
#[derive(Debug, Clone)]
struct PendingLotssBestMatch {
    row_index: usize,
    separation_arcsec: f64,
}

#[cfg(feature = "fits")]
#[derive(Debug)]
struct PointBestMatchScanSummary {
    matches: Vec<Option<PendingLotssBestMatch>>,
}

#[cfg(feature = "fits")]
fn build_lotss_execution_plan(
    requested_workers: Option<usize>,
    chunk_rows_override: Option<usize>,
) -> LotssExecutionPlan {
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
        .max(LOTSS_NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW)
        / worker_count.max(1);
    let auto_chunk_rows = (per_worker_bytes / LOTSS_NUMERIC_FITS_WORKING_SET_BYTES_PER_ROW)
        .clamp(
            LOTSS_MIN_PARALLEL_FITS_CHUNK_ROWS,
            LOTSS_MAX_PARALLEL_FITS_CHUNK_ROWS,
        );
    let alignment_rows = (simd_lane_f64 * 256).max(1);
    let chunk_rows = (chunk_rows_override.unwrap_or(auto_chunk_rows))
        .clamp(
            LOTSS_MIN_PARALLEL_FITS_CHUNK_ROWS,
            LOTSS_MAX_PARALLEL_FITS_CHUNK_ROWS,
        )
        / alignment_rows
        * alignment_rows;

    LotssExecutionPlan {
        mode: if worker_count > 1 {
            "pinned_physical_single_pass"
        } else {
            "scalar_single_pass"
        },
        worker_count,
        chunk_rows: chunk_rows.max(LOTSS_MIN_PARALLEL_FITS_CHUNK_ROWS),
        physical_core_ids,
        l3_cache_bytes: topo.l3_cache_bytes,
        l3_safe_working_set_bytes: topo.l3_safe_working_set_bytes,
        avx2_detected,
        fma_detected,
        simd_lane_f64,
    }
}

#[cfg(feature = "fits")]
fn prepare_point_grid(points: &[SkyPoint], match_radius_arcsec: f64) -> PreparedLotssPointGrid {
    let mut unit_x = Vec::with_capacity(points.len());
    let mut unit_y = Vec::with_capacity(points.len());
    let mut unit_z = Vec::with_capacity(points.len());
    for point in points {
        let [x, y, z] = sky_point_unit_vector(point.ra_deg, point.dec_deg);
        unit_x.push(x);
        unit_y.push(y);
        unit_z.push(z);
    }
    PreparedLotssPointGrid {
        point_count: points.len(),
        grid: SkyGridIndex::from_points(points.to_vec(), (match_radius_arcsec / 3600.0).max(0.01)),
        unit_x,
        unit_y,
        unit_z,
    }
}

#[cfg(feature = "fits")]
fn inspect_lotss_position_layout(path: &Path) -> Result<LotssPositionLayout, FetchError> {
    use fitsio::{FitsFile, hdu::HduInfo};

    let mut fits = FitsFile::open(path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", path.display(), e)))?;
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
        let hdu = fits
            .hdu(idx)
            .map_err(|e| FetchError::Validation(format!("hdu {}: {}", idx, e)))?;
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

    let table_idx = table_idx.ok_or_else(|| {
        FetchError::Validation(format!("No BINTABLE HDU found in {}", path.display()))
    })?;
    let ra_column = column_names
        .iter()
        .find(|name| name.eq_ignore_ascii_case("RA"))
        .cloned()
        .ok_or_else(|| FetchError::Validation(format!("No RA column found in {}", path.display())))?;
    let dec_column = column_names
        .iter()
        .find(|name| name.eq_ignore_ascii_case("DEC"))
        .cloned()
        .ok_or_else(|| FetchError::Validation(format!("No DEC column found in {}", path.display())))?;

    Ok(LotssPositionLayout {
        table_idx,
        row_count,
        ra_column,
        dec_column,
    })
}

#[cfg(feature = "fits")]
fn split_lotss_work_bounds(len: usize, parts: usize) -> Vec<(usize, usize)> {
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

#[cfg(feature = "fits")]
fn scan_lotss_best_match_range(
    lotss_path: &Path,
    layout: &LotssPositionLayout,
    prepared: &PreparedLotssPointGrid,
    match_radius_arcsec: f64,
    worker_bounds: Range<usize>,
    chunk_rows: usize,
) -> Result<PointBestMatchScanSummary, FetchError> {
    use fitsio::FitsFile;

    let mut fits = FitsFile::open(lotss_path)
        .map_err(|e| FetchError::Validation(format!("FITS open {}: {}", lotss_path.display(), e)))?;
    let table_hdu = fits
        .hdu(layout.table_idx)
        .map_err(|e| FetchError::Validation(format!("hdu {}: {}", layout.table_idx, e)))?;
    let mut matches: Vec<Option<PendingLotssBestMatch>> = vec![None; prepared.point_count];

    for start in (worker_bounds.start..worker_bounds.end).step_by(chunk_rows.max(1)) {
        let end = (start + chunk_rows).min(worker_bounds.end);
        let row_range = start..end;
        let ras: Vec<f64> = table_hdu
            .read_col_range(&mut fits, &layout.ra_column, &row_range)
            .map_err(|e| {
                FetchError::Validation(format!(
                    "Load FITS {} rows {}..{}: {}",
                    layout.ra_column, start, end, e
                ))
            })?;
        let decs: Vec<f64> = table_hdu
            .read_col_range(&mut fits, &layout.dec_column, &row_range)
            .map_err(|e| {
                FetchError::Validation(format!(
                    "Load FITS {} rows {}..{}: {}",
                    layout.dec_column, start, end, e
                ))
            })?;

        for row_idx in 0..ras.len() {
            let ra_deg = *ras.get(row_idx).unwrap_or(&f64::NAN);
            let dec_deg = *decs.get(row_idx).unwrap_or(&f64::NAN);
            if !ra_deg.is_finite() || !dec_deg.is_finite() {
                continue;
            }
            for_each_prepared_point_match(
                prepared,
                ra_deg,
                dec_deg,
                match_radius_arcsec,
                |candidate_index, separation_arcsec| {
                    update_pending_best_match(
                        &mut matches[candidate_index],
                        PendingLotssBestMatch {
                            row_index: start + row_idx,
                            separation_arcsec,
                        },
                    );
                },
            );
        }
    }

    Ok(PointBestMatchScanSummary { matches })
}

#[cfg(feature = "fits")]
fn merge_pending_best_match_vectors(
    merged: &mut [Option<PendingLotssBestMatch>],
    local: &[Option<PendingLotssBestMatch>],
) {
    for (slot, candidate) in merged.iter_mut().zip(local.iter().cloned()) {
        if let Some(candidate) = candidate {
            update_pending_best_match(slot, candidate);
        }
    }
}

#[cfg(feature = "fits")]
fn update_pending_best_match(
    slot: &mut Option<PendingLotssBestMatch>,
    candidate: PendingLotssBestMatch,
) {
    match slot {
        Some(existing) if existing.separation_arcsec <= candidate.separation_arcsec => {}
        _ => *slot = Some(candidate),
    }
}

#[cfg(feature = "fits")]
fn finalize_pending_lotss_best_matches(
    path: &Path,
    release: LoTSSRelease,
    pending: Vec<Option<PendingLotssBestMatch>>,
) -> Result<Vec<Option<LotssFitsBestMatch>>, FetchError> {
    let row_indices = pending
        .iter()
        .filter_map(|entry| entry.as_ref().map(|entry| entry.row_index))
        .collect::<Vec<_>>();
    let rows_by_index = load_fits_rows_by_index(path, release, &row_indices)?;
    let mut finalized = Vec::with_capacity(pending.len());
    for entry in pending {
        match entry {
            Some(entry) => {
                let source = rows_by_index.get(&entry.row_index).cloned().ok_or_else(|| {
                    FetchError::Validation(format!(
                        "Missing LoTSS source row {} during batch finalization for {}",
                        entry.row_index,
                        path.display()
                    ))
                })?;
                finalized.push(Some(LotssFitsBestMatch {
                    source_row_index: entry.row_index,
                    separation_arcsec: entry.separation_arcsec,
                    source,
                }));
            }
            None => finalized.push(None),
        }
    }
    Ok(finalized)
}

#[cfg(feature = "fits")]
fn load_fits_rows_by_index(
    path: &Path,
    release: LoTSSRelease,
    row_indices: &[usize],
) -> Result<HashMap<usize, LoTSSSource>, FetchError> {
    let mut sorted = row_indices.to_vec();
    sorted.sort_unstable();
    sorted.dedup();
    let mut rows_by_index = HashMap::with_capacity(sorted.len());
    for (start, end) in group_consecutive_row_runs(&sorted) {
        for (row_index, source) in load_from_fits_indexed_range(path, release, start..end)? {
            rows_by_index.insert(row_index, source);
        }
    }
    Ok(rows_by_index)
}

#[cfg(feature = "fits")]
fn load_from_fits_indexed_range(
    path: &Path,
    release: LoTSSRelease,
    row_range: Range<usize>,
) -> Result<Vec<(usize, LoTSSSource)>, FetchError> {
    let start = row_range.start;
    let sources = load_from_fits_range(path, release, row_range.clone())?;
    let expected = row_range.end.saturating_sub(row_range.start);
    if sources.len() != expected {
        return Err(FetchError::Validation(format!(
            "Expected {} LoTSS rows from {}..{} in {}, got {}",
            expected,
            row_range.start,
            row_range.end,
            path.display(),
            sources.len()
        )));
    }
    Ok(sources
        .into_iter()
        .enumerate()
        .map(|(offset, source)| (start + offset, source))
        .collect())
}

#[cfg(feature = "fits")]
fn group_consecutive_row_runs(sorted_row_indices: &[usize]) -> Vec<(usize, usize)> {
    if sorted_row_indices.is_empty() {
        return Vec::new();
    }
    let mut runs = Vec::new();
    let mut start = sorted_row_indices[0];
    let mut prev = start;
    for &row_index in &sorted_row_indices[1..] {
        if row_index == prev + 1 {
            prev = row_index;
            continue;
        }
        runs.push((start, prev + 1));
        start = row_index;
        prev = row_index;
    }
    runs.push((start, prev + 1));
    runs
}

#[cfg(feature = "fits")]
fn for_each_prepared_point_match<F>(
    prepared: &PreparedLotssPointGrid,
    query_ra_deg: f64,
    query_dec_deg: f64,
    match_radius_arcsec: f64,
    mut visitor: F,
) where
    F: FnMut(usize, f64),
{
    let mut candidate_indices = Vec::new();
    prepared
        .grid
        .for_each_search_candidate(query_ra_deg, query_dec_deg, match_radius_arcsec, |idx| {
            candidate_indices.push(idx);
        });
    if candidate_indices.is_empty() {
        return;
    }

    let [query_x, query_y, query_z] = sky_point_unit_vector(query_ra_deg, query_dec_deg);
    let cos_threshold = (match_radius_arcsec / 3600.0).to_radians().cos();
    let cos_threshold_slack = 1.0e-12;
    let points = prepared.grid.points();

    let mut offset = 0usize;
    while offset + 4 <= candidate_indices.len() {
        let chunk = &candidate_indices[offset..offset + 4];
        let dots = candidate_dot_chunk(prepared, query_x, query_y, query_z, chunk);
        let dot_arr = dots.to_array();
        for lane in 0..4 {
            if dot_arr[lane] + cos_threshold_slack < cos_threshold {
                continue;
            }
            let candidate_index = chunk[lane];
            let point = &points[candidate_index];
            let separation_arcsec = precise_angular_separation_arcsec(
                query_ra_deg,
                query_dec_deg,
                point.ra_deg,
                point.dec_deg,
            );
            if separation_arcsec <= match_radius_arcsec {
                visitor(candidate_index, separation_arcsec);
            }
        }
        offset += 4;
    }

    while offset < candidate_indices.len() {
        let candidate_index = candidate_indices[offset];
        let point = &points[candidate_index];
        let separation_arcsec = precise_angular_separation_arcsec(
            query_ra_deg,
            query_dec_deg,
            point.ra_deg,
            point.dec_deg,
        );
        if separation_arcsec <= match_radius_arcsec {
            visitor(candidate_index, separation_arcsec);
        }
        offset += 1;
    }
}

#[cfg(feature = "fits")]
fn candidate_dot_chunk(
    prepared: &PreparedLotssPointGrid,
    query_x: f64,
    query_y: f64,
    query_z: f64,
    candidate_indices: &[usize],
) -> f64x4 {
    let idx0 = candidate_indices[0];
    let idx1 = candidate_indices[1];
    let idx2 = candidate_indices[2];
    let idx3 = candidate_indices[3];
    let xs = f64x4::from([
        prepared.unit_x[idx0],
        prepared.unit_x[idx1],
        prepared.unit_x[idx2],
        prepared.unit_x[idx3],
    ]);
    let ys = f64x4::from([
        prepared.unit_y[idx0],
        prepared.unit_y[idx1],
        prepared.unit_y[idx2],
        prepared.unit_y[idx3],
    ]);
    let zs = f64x4::from([
        prepared.unit_z[idx0],
        prepared.unit_z[idx1],
        prepared.unit_z[idx2],
        prepared.unit_z[idx3],
    ]);
    xs * f64x4::splat(query_x) + ys * f64x4::splat(query_y) + zs * f64x4::splat(query_z)
}

#[cfg(feature = "fits")]
fn sky_point_unit_vector(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

#[cfg(feature = "fits")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_avx2() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(feature = "fits")]
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_avx2() -> bool {
    false
}

#[cfg(feature = "fits")]
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
fn detect_fma() -> bool {
    std::arch::is_x86_feature_detected!("fma")
}

#[cfg(feature = "fits")]
#[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
fn detect_fma() -> bool {
    false
}

#[cfg(feature = "fits")]
fn preferred_f64_simd_lane(avx2_detected: bool) -> usize {
    if avx2_detected { 4 } else { 1 }
}

#[cfg(feature = "fits")]
fn precise_angular_separation_arcsec(
    ra1_deg: f64,
    dec1_deg: f64,
    ra2_deg: f64,
    dec2_deg: f64,
) -> f64 {
    #[cfg(target_arch = "x86_64")]
    {
        let separation = angular_separation_arcsec_ext80_deg(ra1_deg, dec1_deg, ra2_deg, dec2_deg);
        if separation.is_finite() {
            separation
        } else {
            angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg)
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        angular_separation_arcsec(ra1_deg, dec1_deg, ra2_deg, dec2_deg)
    }
}

// ---- Internal helpers --------------------------------------------------------

#[cfg(feature = "fits")]
fn parse_row_from_votable(
    row: &HashMap<String, String>,
    release: LoTSSRelease,
) -> Option<LoTSSSource> {
    let get = |key: &str| -> &str { row.get(key).map(|s| s.as_str()).unwrap_or("") };
    let parse_f32 = |key: &str| -> f32 { get(key).parse::<f32>().unwrap_or(f32::NAN) };
    let parse_f64 = |key: &str| -> f64 { get(key).parse::<f64>().unwrap_or(f64::NAN) };
    let parse_opt_f32 = |key: &str| -> Option<f32> {
        let v: f32 = get(key).parse().ok()?;
        if v.is_finite() { Some(v) } else { None }
    };

    let source_name = get("SOURCE_NAME").to_string();
    if source_name.is_empty() {
        return None;
    }

    let structure_code = get("S_CODE").chars().next().unwrap_or('S');
    let resolved_str = get("RESOLVED");
    let resolved =
        resolved_str.eq_ignore_ascii_case("R") || resolved_str == "1" || resolved_str == "true";

    Some(LoTSSSource {
        source_name,
        ra_deg: parse_f64("RA"),
        dec_deg: parse_f64("DEC"),
        flux_mjy: parse_f32("TOTAL_FLUX"),
        flux_err_mjy: parse_f32("E_TOTAL_FLUX"),
        peak_flux: parse_f32("PEAK_FLUX"),
        local_rms: parse_f32("ISL_RMS"),
        spectral_index: parse_opt_f32("SPECTRAL_INDEX"),
        spectral_index_err: parse_opt_f32("E_SPECTRAL_INDEX"),
        resolved,
        structure_code,
        maj_arcsec: parse_f32("MAJ"),
        min_arcsec: parse_f32("MIN"),
        pa_deg: parse_f32("PA"),
        release,
    })
}

#[cfg(all(test, feature = "fits"))]
mod tests {
    use super::*;

    #[test]
    fn release_labels() {
        assert_eq!(LoTSSRelease::DR1.label(), "DR1");
        assert_eq!(LoTSSRelease::DR2.label(), "DR2");
        assert_eq!(LoTSSRelease::DR3.label(), "DR3");
    }

    #[test]
    fn parse_votable_minimal() {
        let xml = r#"<?xml version="1.0"?>
<VOTABLE version="1.4">
  <RESOURCE>
    <TABLE>
      <FIELD name="Source_Name" datatype="char"/>
      <FIELD name="RA" datatype="double"/>
      <FIELD name="DEC" datatype="double"/>
      <FIELD name="Total_flux" datatype="float"/>
      <FIELD name="E_Total_flux" datatype="float"/>
      <FIELD name="Peak_flux" datatype="float"/>
      <FIELD name="Isl_rms" datatype="float"/>
      <FIELD name="Spectral_index" datatype="float"/>
      <FIELD name="E_spectral_index" datatype="float"/>
      <FIELD name="Resolved" datatype="char"/>
      <FIELD name="S_Code" datatype="char"/>
      <FIELD name="Maj" datatype="float"/>
      <FIELD name="Min" datatype="float"/>
      <FIELD name="PA" datatype="float"/>
      <DATA>
        <TABLEDATA>
          <TR>
            <TD>ILTJ000001.1+123456</TD><TD>0.0042</TD><TD>12.5820</TD>
            <TD>3.4</TD><TD>0.1</TD><TD>3.1</TD><TD>0.08</TD>
            <TD>-0.8</TD><TD>0.1</TD><TD>R</TD><TD>S</TD>
            <TD>6.1</TD><TD>5.2</TD><TD>45.0</TD>
          </TR>
        </TABLEDATA>
      </DATA>
    </TABLE>
  </RESOURCE>
</VOTABLE>"#;

        let sources = load_from_votable(xml, LoTSSRelease::DR3).unwrap();
        assert_eq!(sources.len(), 1);
        let s = &sources[0];
        assert_eq!(s.source_name, "ILTJ000001.1+123456");
        assert!((s.ra_deg - 0.0042).abs() < 1e-6);
        assert!(s.resolved);
        assert_eq!(s.structure_code, 'S');
        assert!((s.flux_mjy - 3.4).abs() < 0.01);
        assert_eq!(s.release, LoTSSRelease::DR3);
    }

    #[test]
    fn consecutive_row_runs_are_grouped() {
        let runs = group_consecutive_row_runs(&[2, 3, 4, 9, 12, 13]);
        assert_eq!(runs, vec![(2, 5), (9, 10), (12, 14)]);
    }
}
