//! Rust-native Euclid Q1 x MaNGA selected-sample crossmatch.
//!
//! Uses the governed MaNGA selected sample (`dapall_selection.csv` + DRPall
//! coordinates) together with the Euclid Q1 morphology parquet to produce a
//! plateifu selection CSV for morphology-filtered stacking.

#[cfg(feature = "euclid-catalog")]
use anyhow::{Result, anyhow, bail};
#[cfg(feature = "euclid-catalog")]
use clap::Parser;
#[cfg(feature = "euclid-catalog")]
use cosmology_core::euclid_morphology::{EuclidMorphologyRecord, read_euclid_visual_morphology};
#[cfg(feature = "euclid-catalog")]
use data_core::{
    angular_separation_arcsec,
    formats::fits_table::{FitsValue, read_fits_table},
};
#[cfg(feature = "euclid-catalog")]
use kiddo::{ImmutableKdTree, SquaredEuclidean};
#[cfg(feature = "euclid-catalog")]
use serde::Serialize;
#[cfg(feature = "euclid-catalog")]
use std::{
    collections::{HashMap, HashSet},
    fs,
    path::{Path, PathBuf},
};

#[cfg(feature = "euclid-catalog")]
const DRPALL_COLUMNS: &[&str] = &["plateifu", "mangaid", "objra", "objdec"];

#[cfg(feature = "euclid-catalog")]
#[derive(Parser, Debug)]
#[command(name = "euclid-manga-crossmatch")]
#[command(about = "Cross-match Euclid morphology with the selected MaNGA sample")]
struct Cli {
    #[arg(
        long,
        default_value = "data/external/euclid/zenodo/15106473/morphology_catalogue.parquet"
    )]
    euclid_morphology: PathBuf,

    #[arg(long, default_value = "data/external/manga/dapall_selection.csv")]
    manga_selection: PathBuf,

    #[arg(long, default_value = "data/external/manga/drpall-v3_1_1.fits")]
    manga_drpall: PathBuf,

    #[arg(
        long,
        default_value = "data/external/euclid/euclid_manga_crossmatch.csv"
    )]
    output: PathBuf,

    #[arg(long)]
    report: Option<PathBuf>,

    #[arg(long, default_value_t = 2.0)]
    match_radius_arcsec: f64,

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
#[derive(Debug, Serialize)]
struct EuclidMangaCrossmatchReport {
    generated_at_utc: String,
    euclid_morphology_path: String,
    manga_selection_path: String,
    manga_drpall_path: String,
    output_csv_path: String,
    selection_row_count: usize,
    unique_plateifu_count: usize,
    duplicate_plateifu_count: usize,
    missing_plateifu_count: usize,
    match_radius_arcsec: f64,
    matched_manga_count: usize,
    selected_output_count: usize,
    no_morphology_cuts: bool,
    featured_min: f32,
    spiral_min: f32,
    face_on_min: f32,
    non_merge_min: f32,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug, Serialize)]
struct EuclidMangaCrossmatchRow {
    plateifu: String,
    mangaid: String,
    objra: String,
    objdec: String,
    euclid_object_id: String,
    euclid_ra: String,
    euclid_dec: String,
    sep_arcsec: String,
    euclid_ellipticity: String,
    euclid_featured_fraction: String,
    euclid_spiral_fraction: String,
    euclid_face_on_fraction: String,
    euclid_non_merging_fraction: String,
    euclid_bar_no_fraction: String,
    euclid_clumps_yes_fraction: String,
    morphology_selected: String,
}

#[cfg(feature = "euclid-catalog")]
#[derive(Debug)]
struct SelectedMangaSample {
    row_count: usize,
    duplicate_count: usize,
    missing_plateifus: Vec<String>,
    targets: Vec<MangaSkyTarget>,
}

#[cfg(feature = "euclid-catalog")]
fn main() -> Result<()> {
    let cli = Cli::parse();

    let sample = load_selected_manga_targets(&cli.manga_selection, &cli.manga_drpall)?;
    let morphology = read_euclid_visual_morphology(
        cli.euclid_morphology
            .to_str()
            .ok_or_else(|| anyhow!("Non-UTF8 Euclid path"))?,
    )
    .map_err(anyhow::Error::msg)?;
    if morphology.is_empty() {
        bail!(
            "No Euclid morphology rows loaded from {}",
            cli.euclid_morphology.display()
        );
    }

    let euclid_points: Vec<[f64; 3]> = morphology
        .iter()
        .map(|row| unit_vector(row.ra_deg, row.dec_deg))
        .collect();
    let tree: ImmutableKdTree<f64, 3> = ImmutableKdTree::new_from_slice(&euclid_points);

    let mut matched_rows = Vec::new();
    for target in &sample.targets {
        let nearest =
            tree.nearest_one::<SquaredEuclidean>(&unit_vector(target.ra_deg, target.dec_deg));
        let candidate = &morphology[nearest.item as usize];
        let separation_arcsec = angular_separation_arcsec(
            target.ra_deg,
            target.dec_deg,
            candidate.ra_deg,
            candidate.dec_deg,
        );
        if separation_arcsec > cli.match_radius_arcsec {
            continue;
        }
        let morphology_selected = cli.no_morphology_cuts
            || passes_morphology_cuts(
                candidate,
                cli.featured_min,
                cli.spiral_min,
                cli.face_on_min,
                cli.non_merge_min,
            );
        matched_rows.push(EuclidMangaCrossmatchRow {
            plateifu: target.plateifu.clone(),
            mangaid: target.mangaid.clone(),
            objra: format!("{:.6}", target.ra_deg),
            objdec: format!("{:.6}", target.dec_deg),
            euclid_object_id: candidate.object_id.clone(),
            euclid_ra: format!("{:.6}", candidate.ra_deg),
            euclid_dec: format!("{:.6}", candidate.dec_deg),
            sep_arcsec: format!("{:.4}", separation_arcsec),
            euclid_ellipticity: format!("{:.6}", candidate.ellipticity),
            euclid_featured_fraction: format!("{:.6}", candidate.featured_fraction),
            euclid_spiral_fraction: format!("{:.6}", candidate.spiral_fraction),
            euclid_face_on_fraction: format!("{:.6}", candidate.face_on_fraction),
            euclid_non_merging_fraction: format!("{:.6}", candidate.non_merging_fraction),
            euclid_bar_no_fraction: format!("{:.6}", candidate.bar_no_fraction),
            euclid_clumps_yes_fraction: format!("{:.6}", candidate.clumps_yes_fraction),
            morphology_selected: if morphology_selected {
                "1".to_string()
            } else {
                "0".to_string()
            },
        });
    }

    matched_rows.sort_by(|a, b| a.plateifu.cmp(&b.plateifu));
    let selected_rows: Vec<&EuclidMangaCrossmatchRow> = if cli.no_morphology_cuts {
        matched_rows.iter().collect()
    } else {
        matched_rows
            .iter()
            .filter(|row| row.morphology_selected == "1")
            .collect()
    };

    write_crossmatch_csv(&cli.output, &selected_rows)?;
    let report_path = cli.report.unwrap_or_else(default_report_path);
    let report = EuclidMangaCrossmatchReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        euclid_morphology_path: cli.euclid_morphology.display().to_string(),
        manga_selection_path: cli.manga_selection.display().to_string(),
        manga_drpall_path: cli.manga_drpall.display().to_string(),
        output_csv_path: cli.output.display().to_string(),
        selection_row_count: sample.row_count,
        unique_plateifu_count: sample.targets.len() + sample.missing_plateifus.len(),
        duplicate_plateifu_count: sample.duplicate_count,
        missing_plateifu_count: sample.missing_plateifus.len(),
        match_radius_arcsec: cli.match_radius_arcsec,
        matched_manga_count: matched_rows.len(),
        selected_output_count: selected_rows.len(),
        no_morphology_cuts: cli.no_morphology_cuts,
        featured_min: cli.featured_min,
        spiral_min: cli.spiral_min,
        face_on_min: cli.face_on_min,
        non_merge_min: cli.non_merge_min,
    };
    write_toml_report(&report_path, &report)?;

    println!("Matched MaNGA galaxies: {}", matched_rows.len());
    println!("Rows written:           {}", selected_rows.len());
    println!("CSV:                    {}", cli.output.display());
    println!("Report:                 {}", report_path.display());
    Ok(())
}

#[cfg(not(feature = "euclid-catalog"))]
fn main() -> anyhow::Result<()> {
    anyhow::bail!(
        "euclid-manga-crossmatch requires the 'euclid-catalog' feature.\n\
         Run: cargo run -p gororoba_cli_physics --features euclid-catalog --bin euclid-manga-crossmatch -- --help"
    )
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

    let mut row_count = 0usize;
    let mut duplicate_count = 0usize;
    let mut selected = Vec::new();
    let mut seen = HashSet::new();
    for record in reader.records() {
        let record = record?;
        row_count += 1;
        let plateifu = record.get(plateifu_idx).unwrap_or("").trim();
        if plateifu.is_empty() {
            continue;
        }
        if seen.insert(plateifu.to_string()) {
            selected.push(plateifu.to_string());
        } else {
            duplicate_count += 1;
        }
    }

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

    let mut targets = Vec::with_capacity(selected.len());
    let mut missing_plateifus = Vec::new();
    for plateifu in selected {
        if let Some(target) = by_plateifu.get(&plateifu) {
            targets.push(target.clone());
        } else {
            missing_plateifus.push(plateifu);
        }
    }

    Ok(SelectedMangaSample {
        row_count,
        duplicate_count,
        missing_plateifus,
        targets,
    })
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
fn passes_morphology_cuts(
    row: &EuclidMorphologyRecord,
    featured_min: f32,
    spiral_min: f32,
    face_on_min: f32,
    non_merge_min: f32,
) -> bool {
    row.featured_fraction > featured_min
        && row.spiral_fraction > spiral_min
        && row.face_on_fraction > face_on_min
        && row.non_merging_fraction > non_merge_min
}

#[cfg(feature = "euclid-catalog")]
fn unit_vector(ra_deg: f64, dec_deg: f64) -> [f64; 3] {
    let ra = ra_deg.to_radians();
    let dec = dec_deg.to_radians();
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

#[cfg(feature = "euclid-catalog")]
fn write_crossmatch_csv(path: &Path, rows: &[&EuclidMangaCrossmatchRow]) -> Result<()> {
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
fn default_report_path() -> PathBuf {
    PathBuf::from("reports").join(format!(
        "euclid_manga_crossmatch_{}.toml",
        chrono::Utc::now().date_naive()
    ))
}

#[cfg(all(test, feature = "euclid-catalog"))]
mod tests {
    use super::*;

    #[test]
    fn morphology_cut_logic_matches_thresholds() {
        let row = EuclidMorphologyRecord {
            object_id: "1".to_string(),
            ra_deg: 0.0,
            dec_deg: 0.0,
            ellipticity: 0.2,
            featured_fraction: 0.8,
            spiral_fraction: 0.9,
            face_on_fraction: 0.7,
            non_merging_fraction: 0.95,
            bar_no_fraction: 0.4,
            clumps_yes_fraction: 0.1,
        };
        assert!(passes_morphology_cuts(&row, 0.5, 0.5, 0.5, 0.7));
        assert!(!passes_morphology_cuts(&row, 0.9, 0.5, 0.5, 0.7));
    }

    #[test]
    fn unit_vector_is_normalized() {
        let v = unit_vector(270.0, 66.0);
        let norm = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        assert!((norm - 1.0).abs() < 1.0e-12);
    }
}
