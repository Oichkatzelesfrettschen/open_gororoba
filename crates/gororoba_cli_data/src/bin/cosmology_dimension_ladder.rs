use algebra_analysis::{
    associator_entropy::decompose_entropy_adaptive,
    boxkite_alignment::compute_alignment,
    graph_projections::{
        compute_budgeted_invariants, generate_pathion_matching, generate_zd_parity_cliques,
    },
    phase_transition::PhaseTransitionAnalyzer,
};
use anyhow::{Context, Result, anyhow, bail};
use cd_kernel::cayley_dickson::{cd_associator_norm, cd_basis_mul_sign, cd_norm_sq};
use chrono::{DateTime, Duration, TimeZone, Utc};
use clap::Parser;
use cosmology_core::euclid_morphology::{
    read_euclid_physical_measurements, read_euclid_visual_morphology,
};
use csv::{ReaderBuilder, Writer};
use data_core::catalogs::jwst::{JwstPublicObservation, parse_jwst_public_metadata_csv};
use fitsio::{FitsFile, hdu::HduInfo};
use petgraph::graph::UnGraph;
use serde::{Deserialize, Serialize};
use stats_core::ultrametric::local::local_ultrametricity_test_nd;
use std::{
    cmp::Ordering,
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
};

const DIMENSION_LADDER: &[usize] = &[4, 8, 16, 32, 64, 128, 256, 512, 1024];
const MAX_SOURCE_POINTS: usize = 2048;
const WINDOW_POINTS: usize = 64;
const KNN_K: usize = 4;

#[derive(Parser, Debug)]
#[command(
    name = "cosmology-dimension-ladder",
    about = "Project cached cosmology sources into a Cayley-Dickson dimension ladder with explicit measured-vs-derived bookkeeping"
)]
struct Args {
    #[arg(long, default_value = "data/csv/cosmology_dimension_ladder.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "docs/COSMOLOGY_DIMENSION_LADDER.md")]
    markdown_out: PathBuf,
}

#[derive(Clone)]
struct DatasetCloud {
    dataset_id: String,
    family: String,
    measured_surface: String,
    measured_vs_derived_note: String,
    source_paths: Vec<String>,
    feature_labels: Vec<String>,
    points: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct AlgebraReference {
    cd_dim: usize,
    negative_sign_density: f64,
    defect_density_mc: f64,
    order_parameter: f64,
    zd_edge_density: f64,
    entropy_reduction: f64,
    near_zd_fraction: f64,
}

#[derive(Clone)]
struct GraphMetrics {
    node_count: usize,
    edge_density: f64,
    triangle_density: f64,
    component_ratio: f64,
    mean_degree: f64,
    degree_std: f64,
}

#[derive(Clone)]
struct RowView {
    csv: CsvRow,
    graph_signature: GraphMetrics,
}

#[derive(Serialize, Clone)]
struct CsvRow {
    dataset_id: String,
    family: String,
    measured_surface: String,
    measured_vs_derived_note: String,
    source_paths: String,
    raw_point_count: usize,
    base_feature_dim: usize,
    base_feature_labels: String,
    cd_dim: usize,
    embedding_kind: String,
    window_points: usize,
    ultrametric_epsilon: f64,
    ultrametric_mean: f64,
    ultrametric_median: f64,
    ultrametric_null_mean: f64,
    ultrametric_p_value: f64,
    mean_associator_norm: f64,
    median_associator_norm: f64,
    dataset_edge_density: f64,
    dataset_triangle_density: f64,
    dataset_component_ratio: f64,
    dataset_mean_degree: f64,
    dataset_degree_std: f64,
    parity_graph_distance: f64,
    pathion_graph_distance: Option<f64>,
    mean_boxkite_capture: Option<f64>,
    dominant_boxkite_share: Option<f64>,
    cd_negative_sign_density: f64,
    cd_defect_density_mc: f64,
    cd_order_parameter: f64,
    cd_zd_edge_density: f64,
    cd_entropy_reduction: f64,
    cd_near_zd_fraction: f64,
}

#[derive(Deserialize)]
struct EuclidLensRow {
    right_ascension: f64,
    declination: f64,
    segmentation_area: f64,
    flux_vis_1fwhm_aper: f64,
    expert_score: f64,
    grade: String,
    expert_total_votes: Option<f64>,
}

#[derive(Deserialize)]
struct EuclidMergerRow {
    right_ascension: f64,
    declination: f64,
    #[serde(rename = "CNN pred")]
    cnn_pred: f64,
    #[serde(rename = "CNN classification")]
    cnn_classification: i32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let datasets = load_datasets()?;
    let algebra_refs = build_algebra_references();
    let rows = build_rows(&datasets, &algebra_refs);

    write_csv(&args.csv_out, &rows)?;
    write_markdown(&args.markdown_out, &datasets, &algebra_refs, &rows)?;

    println!("WROTE {}", args.csv_out.display());
    println!("WROTE {}", args.markdown_out.display());
    Ok(())
}

fn load_datasets() -> Result<Vec<DatasetCloud>> {
    let mut datasets = vec![
        load_euclid_physical_dataset()?,
        load_euclid_morphology_dataset()?,
        load_euclid_lens_dataset()?,
        load_euclid_merger_dataset()?,
        load_jwst_metadata_dataset()?,
        load_jwst_massmap_dataset(
            "jwst_massmap_fits_1",
            "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/supplementary_data_1_m2.fits",
        )?,
        load_jwst_massmap_dataset(
            "jwst_massmap_fits_2",
            "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/supplementary_data_3_m4.fits",
        )?,
        load_jwst_massmap_dataset(
            "jwst_massmap_fits_3",
            "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/supplementary_data_5_m6.fits",
        )?,
        load_basin_text_dataset()?,
    ];

    for dataset in &mut datasets {
        normalize_columns(&mut dataset.points);
        dataset.points = evenly_subsample(&dataset.points, MAX_SOURCE_POINTS);
    }

    Ok(datasets)
}

fn build_algebra_references() -> Vec<AlgebraReference> {
    DIMENSION_LADDER
        .iter()
        .map(|&cd_dim| {
            let analyzer = PhaseTransitionAnalyzer::new(cd_dim);
            let defect_density_mc =
                analyzer.calculate_defect_density(4096, 0xC0A1_0000 + cd_dim as u64);
            let entropy = decompose_entropy_adaptive(
                cd_dim,
                entropy_sample_budget(cd_dim),
                32,
                0xC0A1_A550 + cd_dim as u64,
            );

            AlgebraReference {
                cd_dim,
                negative_sign_density: negative_sign_density(cd_dim),
                defect_density_mc,
                order_parameter: analyzer.calculate_order_parameter(defect_density_mc),
                zd_edge_density: analyzer.edge_density(),
                entropy_reduction: entropy.zd_entropy_reduction,
                near_zd_fraction: entropy.near_zd_fraction,
            }
        })
        .collect()
}

fn build_rows(datasets: &[DatasetCloud], algebra_refs: &[AlgebraReference]) -> Vec<RowView> {
    let mut rows = Vec::new();

    for dataset in datasets {
        for reference in algebra_refs {
            let window_points = lifted_window(&dataset.points, reference.cd_dim, WINDOW_POINTS);
            if window_points.len() < 3 {
                continue;
            }

            let epsilon = epsilon_from_knn(&window_points, 4).max(1e-6);
            let ultrametric = local_ultrametricity_test_nd(
                &window_points,
                epsilon,
                48,
                64,
                0x5EED_1000 + reference.cd_dim as u64,
            );
            let associators = sampled_associator_norms(&window_points);
            let graph = knn_graph(&window_points, KNN_K);
            let observed_graph_metrics = graph_metrics(&graph);
            let parity_ref = graph_metrics(&generate_zd_parity_cliques(window_points.len()));
            let pathion_ref = (window_points.len() >= 64)
                .then(|| graph_metrics(&generate_pathion_matching(window_points.len())));
            let (mean_boxkite_capture, dominant_boxkite_share) =
                boxkite_summary(&window_points, reference.cd_dim);

            let csv = CsvRow {
                dataset_id: dataset.dataset_id.clone(),
                family: dataset.family.clone(),
                measured_surface: dataset.measured_surface.clone(),
                measured_vs_derived_note: dataset.measured_vs_derived_note.clone(),
                source_paths: dataset.source_paths.join("; "),
                raw_point_count: dataset.points.len(),
                base_feature_dim: dataset.feature_labels.len(),
                base_feature_labels: dataset.feature_labels.join(", "),
                cd_dim: reference.cd_dim,
                embedding_kind: "deterministic harmonic feature lift into unit-normalized Cayley-Dickson coordinates".to_string(),
                window_points: window_points.len(),
                ultrametric_epsilon: epsilon,
                ultrametric_mean: ultrametric.mean_local_index,
                ultrametric_median: ultrametric.median_local_index,
                ultrametric_null_mean: ultrametric.null_mean_index,
                ultrametric_p_value: ultrametric.p_value,
                mean_associator_norm: mean(&associators),
                median_associator_norm: median(&associators),
                dataset_edge_density: observed_graph_metrics.edge_density,
                dataset_triangle_density: observed_graph_metrics.triangle_density,
                dataset_component_ratio: observed_graph_metrics.component_ratio,
                dataset_mean_degree: observed_graph_metrics.mean_degree,
                dataset_degree_std: observed_graph_metrics.degree_std,
                parity_graph_distance: graph_distance(&observed_graph_metrics, &parity_ref),
                pathion_graph_distance: pathion_ref
                    .as_ref()
                    .map(|metrics| graph_distance(&observed_graph_metrics, metrics)),
                mean_boxkite_capture,
                dominant_boxkite_share,
                cd_negative_sign_density: reference.negative_sign_density,
                cd_defect_density_mc: reference.defect_density_mc,
                cd_order_parameter: reference.order_parameter,
                cd_zd_edge_density: reference.zd_edge_density,
                cd_entropy_reduction: reference.entropy_reduction,
                cd_near_zd_fraction: reference.near_zd_fraction,
            };

            rows.push(RowView {
                csv,
                graph_signature: observed_graph_metrics,
            });
        }
    }

    rows
}

fn write_csv(path: &Path, rows: &[RowView]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }
    let mut writer =
        Writer::from_path(path).with_context(|| format!("create CSV {}", path.display()))?;
    for row in rows {
        writer.serialize(&row.csv)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_markdown(
    path: &Path,
    datasets: &[DatasetCloud],
    algebra_refs: &[AlgebraReference],
    rows: &[RowView],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create directory {}", parent.display()))?;
    }

    let mut lines = vec![
        "# Cosmology Dimension Ladder".to_string(),
        "".to_string(),
        "This report separates measured source surfaces from deterministic algebraic embeddings. The observational catalogs and FITS payloads are real cached inputs; the 4D..1024D coordinates are derived lifts used for comparison against Cayley-Dickson reference metrics, not claims that the sky is natively measured in those dimensions.".to_string(),
        "".to_string(),
        "## Source Surfaces".to_string(),
        "".to_string(),
        "| Dataset | Family | Raw points | Measured surface | Note |".to_string(),
        "|---|---:|---:|---|---|".to_string(),
    ];

    for dataset in datasets {
        lines.push(format!(
            "| {} | {} | {} | {} | {} |",
            dataset.dataset_id,
            dataset.family,
            dataset.points.len(),
            dataset.measured_surface,
            dataset.measured_vs_derived_note
        ));
    }

    lines.push("".to_string());
    lines.push("## Algebra Reference Ladder".to_string());
    lines.push("".to_string());
    lines.push("| D | Negative sign density | Defect density | Order parameter | ZD edge density | Entropy reduction | Near-ZD fraction |".to_string());
    lines.push("|---:|---:|---:|---:|---:|---:|---:|".to_string());
    for reference in algebra_refs {
        lines.push(format!(
            "| {} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} | {:.6} |",
            reference.cd_dim,
            reference.negative_sign_density,
            reference.defect_density_mc,
            reference.order_parameter,
            reference.zd_edge_density,
            reference.entropy_reduction,
            reference.near_zd_fraction
        ));
    }

    let mut grouped: BTreeMap<&str, Vec<&RowView>> = BTreeMap::new();
    for row in rows {
        grouped.entry(&row.csv.dataset_id).or_default().push(row);
    }

    lines.push("".to_string());
    lines.push("## Dataset Highlights".to_string());
    lines.push("".to_string());

    for (dataset_id, dataset_rows) in grouped {
        let mut ultrametric_sorted = dataset_rows.clone();
        ultrametric_sorted.sort_by(|a, b| {
            cmp_f64_desc(a.csv.ultrametric_mean, b.csv.ultrametric_mean)
                .then_with(|| cmp_f64_asc(a.csv.parity_graph_distance, b.csv.parity_graph_distance))
        });

        let mut parity_sorted = dataset_rows.clone();
        parity_sorted.sort_by(|a, b| {
            cmp_f64_asc(a.csv.parity_graph_distance, b.csv.parity_graph_distance)
                .then_with(|| cmp_f64_desc(a.csv.ultrametric_mean, b.csv.ultrametric_mean))
        });

        let best_ultra = ultrametric_sorted[0];
        let best_parity = parity_sorted[0];
        lines.push(format!("### {}", dataset_id));
        lines.push("".to_string());
        lines.push(format!(
            "- Best local ultrametricity: D={} with mean={:.4}, p={:.4}, associator={:.4}.",
            best_ultra.csv.cd_dim,
            best_ultra.csv.ultrametric_mean,
            best_ultra.csv.ultrametric_p_value,
            best_ultra.csv.mean_associator_norm
        ));
        lines.push(format!(
            "- Closest graph motif to the parity-clique baseline: D={} with distance={:.4}, vertices={}, edge density={:.4}, triangle density={:.4}, components={:.4}.",
            best_parity.csv.cd_dim,
            best_parity.csv.parity_graph_distance,
            best_parity.graph_signature.node_count,
            best_parity.graph_signature.edge_density,
            best_parity.graph_signature.triangle_density,
            best_parity.graph_signature.component_ratio
        ));
        if let Some(pathion_row) = dataset_rows
            .iter()
            .filter(|row| row.csv.pathion_graph_distance.is_some())
            .min_by(|a, b| {
                cmp_f64_asc(
                    a.csv.pathion_graph_distance.unwrap_or(f64::INFINITY),
                    b.csv.pathion_graph_distance.unwrap_or(f64::INFINITY),
                )
            })
        {
            lines.push(format!(
                "- Closest graph motif to the pathion matching baseline: D={} with distance={:.4}.",
                pathion_row.csv.cd_dim,
                pathion_row.csv.pathion_graph_distance.unwrap_or(f64::NAN)
            ));
        }
        if let Some(capture) = best_ultra.csv.mean_boxkite_capture {
            lines.push(format!(
                "- 16D box-kite capture summary at the best-ultrametric slice: mean capture={:.4}, dominant share={:.4}.",
                capture,
                best_ultra.csv.dominant_boxkite_share.unwrap_or(f64::NAN)
            ));
        }
        lines.push("".to_string());
    }

    lines.push("## Limits".to_string());
    lines.push("".to_string());
    lines.push("- Euclid catalogs, JWST metadata, and JWST FITS maps are measured inputs. Their higher-dimensional coordinates are deterministic embeddings used to compare algebraic metrics, not direct physical measurements of 16D..1024D state spaces.".to_string());
    lines.push("- The basin-of-attraction lane is represented here by the cached paper text sidecar because the public supplement available in-repo is a movie, not a machine-readable scalar field. That makes it useful for term-structure alignment, but not yet for basin geometry extraction.".to_string());
    lines.push("- Graph distances are motif-comparison heuristics between dataset kNN graphs and algebraic reference graphs. They rank follow-up targets; they do not by themselves prove physical equivalence.".to_string());

    fs::write(path, lines.join("\n") + "\n")
        .with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn load_euclid_physical_dataset() -> Result<DatasetCloud> {
    let path = "data/external/euclid/zenodo/15106473/useful_physical_measurements.parquet";
    let rows = read_euclid_physical_measurements(path).map_err(|err| anyhow!(err))?;
    if rows.is_empty() {
        bail!("Euclid physical measurements yielded zero rows");
    }

    let points = rows
        .into_iter()
        .map(|row| {
            vec![
                wrap_ra(row.ra_deg),
                row.dec_deg / 90.0,
                row.photo_z,
                row.log_stellar_mass,
                row.log_luminosity,
                row.log_sfr,
                row.n,
                row.r_e_kpc(),
                row.pa_deg / 180.0,
            ]
        })
        .collect();

    Ok(DatasetCloud {
        dataset_id: "euclid_q1_physical".to_string(),
        family: "euclid".to_string(),
        measured_surface: "catalog rows from useful_physical_measurements.parquet".to_string(),
        measured_vs_derived_note: "Measured catalog observables; high-dimensional coordinates are a deterministic lift from physical row features.".to_string(),
        source_paths: vec![path.to_string()],
        feature_labels: vec![
            "ra_wrap".to_string(),
            "dec_deg".to_string(),
            "photo_z".to_string(),
            "log_stellar_mass".to_string(),
            "log_luminosity".to_string(),
            "log_sfr".to_string(),
            "sersic_n".to_string(),
            "r_e_kpc".to_string(),
            "position_angle".to_string(),
        ],
        points,
    })
}

fn load_euclid_morphology_dataset() -> Result<DatasetCloud> {
    let path = "data/external/euclid/zenodo/15106473/morphology_catalogue.parquet";
    let rows = read_euclid_visual_morphology(path).map_err(|err| anyhow!(err))?;
    if rows.is_empty() {
        bail!("Euclid morphology catalog yielded zero rows");
    }

    let points = rows
        .into_iter()
        .map(|row| {
            vec![
                wrap_ra(row.ra_deg),
                row.dec_deg / 90.0,
                row.ellipticity,
                row.featured_fraction as f64,
                row.spiral_fraction as f64,
                row.face_on_fraction as f64,
                row.non_merging_fraction as f64,
                row.bar_no_fraction as f64,
                row.clumps_yes_fraction as f64,
            ]
        })
        .collect();

    Ok(DatasetCloud {
        dataset_id: "euclid_q1_morphology".to_string(),
        family: "euclid".to_string(),
        measured_surface: "visual morphology probability rows from morphology_catalogue.parquet".to_string(),
        measured_vs_derived_note: "Measured morphology probabilities; the ladder compares their structure after deterministic feature lifting.".to_string(),
        source_paths: vec![path.to_string()],
        feature_labels: vec![
            "ra_wrap".to_string(),
            "dec_deg".to_string(),
            "ellipticity".to_string(),
            "featured_fraction".to_string(),
            "spiral_fraction".to_string(),
            "face_on_fraction".to_string(),
            "non_merging_fraction".to_string(),
            "bar_no_fraction".to_string(),
            "clumps_yes_fraction".to_string(),
        ],
        points,
    })
}

fn load_euclid_lens_dataset() -> Result<DatasetCloud> {
    let path = "data/external/euclid/zenodo/15025832/q1_discovery_engine_lens_catalog.csv";
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("open {path}"))?;

    let mut points = Vec::new();
    for row in reader.deserialize::<EuclidLensRow>() {
        let row = row.with_context(|| format!("parse lens row from {path}"))?;
        points.push(vec![
            wrap_ra(row.right_ascension),
            row.declination / 90.0,
            safe_log10(row.segmentation_area),
            safe_log10(row.flux_vis_1fwhm_aper),
            row.expert_score,
            grade_scalar(&row.grade),
            row.expert_total_votes.unwrap_or(0.0),
        ]);
    }
    if points.is_empty() {
        bail!("Euclid lens catalog yielded zero rows");
    }

    Ok(DatasetCloud {
        dataset_id: "euclid_q1_strong_lensing".to_string(),
        family: "euclid".to_string(),
        measured_surface: "strong-lensing candidate table rows".to_string(),
        measured_vs_derived_note: "Measured candidate attributes; the ladder uses a deterministic lift of sky position plus expert/flux metadata.".to_string(),
        source_paths: vec![path.to_string()],
        feature_labels: vec![
            "ra_wrap".to_string(),
            "dec_deg".to_string(),
            "log_segmentation_area".to_string(),
            "log_flux_vis".to_string(),
            "expert_score".to_string(),
            "grade_scalar".to_string(),
            "expert_total_votes".to_string(),
        ],
        points,
    })
}

fn load_euclid_merger_dataset() -> Result<DatasetCloud> {
    let path = "data/external/euclid/zenodo/17087034/Q1_merger_classification.csv";
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .with_context(|| format!("open {path}"))?;

    let mut points = Vec::new();
    for row in reader.deserialize::<EuclidMergerRow>() {
        let row = row.with_context(|| format!("parse merger row from {path}"))?;
        points.push(vec![
            wrap_ra(row.right_ascension),
            row.declination / 90.0,
            row.cnn_pred,
            row.cnn_classification as f64,
        ]);
    }
    if points.is_empty() {
        bail!("Euclid merger catalog yielded zero rows");
    }

    Ok(DatasetCloud {
        dataset_id: "euclid_q1_mergers".to_string(),
        family: "euclid".to_string(),
        measured_surface: "CNN merger classification rows".to_string(),
        measured_vs_derived_note: "Measured classifier output plus sky coordinates; high-dimensional comparisons are exploratory and deterministic.".to_string(),
        source_paths: vec![path.to_string()],
        feature_labels: vec![
            "ra_wrap".to_string(),
            "dec_deg".to_string(),
            "cnn_pred".to_string(),
            "cnn_classification".to_string(),
        ],
        points,
    })
}

fn load_jwst_metadata_dataset() -> Result<DatasetCloud> {
    let path = Path::new("data/external/jwst_public_observations.csv");
    let rows = parse_jwst_public_metadata_csv(path)
        .map_err(|err| anyhow!("parse JWST metadata {}: {err}", path.display()))?;
    if rows.is_empty() {
        bail!("JWST metadata yielded zero rows");
    }

    let points = rows.into_iter().map(jwst_point).collect();

    Ok(DatasetCloud {
        dataset_id: "jwst_public_metadata".to_string(),
        family: "jwst".to_string(),
        measured_surface: "public observation metadata rows from MAST".to_string(),
        measured_vs_derived_note: "Measured archive metadata; release/instrument/filter fields are encoded deterministically for comparison only.".to_string(),
        source_paths: vec![path.display().to_string()],
        feature_labels: vec![
            "ra_wrap".to_string(),
            "dec_deg".to_string(),
            "release_scalar".to_string(),
            "proposal_scalar".to_string(),
            "filter_count".to_string(),
            "instrument_scalar".to_string(),
            "dataproduct_scalar".to_string(),
            "calib_level".to_string(),
        ],
        points,
    })
}

fn load_jwst_massmap_dataset(dataset_id: &str, path: &str) -> Result<DatasetCloud> {
    let points = read_fits_massmap_points(Path::new(path))?;
    if points.is_empty() {
        bail!("{dataset_id} yielded zero point samples");
    }

    Ok(DatasetCloud {
        dataset_id: dataset_id.to_string(),
        family: "jwst".to_string(),
        measured_surface: "supplementary weak-lensing FITS intensity field".to_string(),
        measured_vs_derived_note: "Measured FITS pixel intensities; the ladder uses top-contrast pixels with local gradients and deterministic high-dimensional lifting.".to_string(),
        source_paths: vec![path.to_string()],
        feature_labels: vec![
            "x_norm".to_string(),
            "y_norm".to_string(),
            "intensity".to_string(),
            "abs_intensity".to_string(),
            "grad_x".to_string(),
            "grad_y".to_string(),
            "radius".to_string(),
            "angle".to_string(),
        ],
        points,
    })
}

fn load_basin_text_dataset() -> Result<DatasetCloud> {
    let path = "data/external/cosmology_maps/basin_of_attraction/arxiv_2409.17261_identification_of_basins_of_attraction.txt";
    let text = fs::read_to_string(path).with_context(|| format!("read {path}"))?;
    let paragraphs: Vec<&str> = text
        .split("\n\n")
        .map(str::trim)
        .filter(|chunk| !chunk.is_empty())
        .collect();
    if paragraphs.is_empty() {
        bail!("Basin text sidecar yielded zero paragraphs");
    }

    let mut points = Vec::new();
    for (index, paragraph) in paragraphs.iter().enumerate() {
        let lower = paragraph.to_ascii_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let total_words = words.len() as f64;
        points.push(vec![
            index as f64 / paragraphs.len() as f64,
            total_words,
            keyword_ratio(&words, &["basin", "basins"]),
            keyword_ratio(&words, &["attraction", "attractor"]),
            keyword_ratio(&words, &["void", "voids"]),
            keyword_ratio(&words, &["cluster", "clusters"]),
            keyword_ratio(&words, &["filament", "filaments"]),
            keyword_ratio(&words, &["potential", "potentials"]),
            keyword_ratio(&words, &["minimum", "minima"]),
            keyword_ratio(&words, &["streamline", "streamlines", "flow"]),
        ]);
    }

    if points.is_empty() {
        bail!("Basin text sidecar yielded zero paragraph features");
    }

    Ok(DatasetCloud {
        dataset_id: "basin_of_attraction_text".to_string(),
        family: "basin".to_string(),
        measured_surface: "paper text sidecar windows (not the movie field)".to_string(),
        measured_vs_derived_note: "Public in-repo basin source is a paper PDF plus movie. This lane measures term-structure from the text sidecar, not the basin scalar field itself.".to_string(),
        source_paths: vec![path.to_string()],
        feature_labels: vec![
            "paragraph_index".to_string(),
            "word_count".to_string(),
            "basin_ratio".to_string(),
            "attraction_ratio".to_string(),
            "void_ratio".to_string(),
            "cluster_ratio".to_string(),
            "filament_ratio".to_string(),
            "potential_ratio".to_string(),
            "minimum_ratio".to_string(),
            "streamline_ratio".to_string(),
        ],
        points,
    })
}

fn read_fits_massmap_points(path: &Path) -> Result<Vec<Vec<f64>>> {
    let path_str = path
        .to_str()
        .ok_or_else(|| anyhow!("non-UTF8 FITS path {}", path.display()))?;
    let mut fits = FitsFile::open(path_str).with_context(|| format!("open {}", path.display()))?;
    let hdu = fits
        .primary_hdu()
        .with_context(|| format!("primary HDU for {}", path.display()))?;

    let (ny, nx) = match &hdu.info {
        HduInfo::ImageInfo { shape, .. } if shape.len() == 2 => (shape[0], shape[1]),
        HduInfo::ImageInfo { shape, .. } => {
            bail!(
                "expected 2D FITS image at {}, found shape {:?}",
                path.display(),
                shape
            )
        }
        _ => bail!("expected image HDU at {}", path.display()),
    };

    let n_total = ny * nx;
    let data: Vec<f32> = hdu
        .read_section(&mut fits, 0, n_total)
        .with_context(|| format!("read FITS pixels from {}", path.display()))?;
    if data.is_empty() {
        bail!("empty FITS payload at {}", path.display());
    }

    let mean = data.iter().map(|&value| value as f64).sum::<f64>() / data.len() as f64;
    let variance = data
        .iter()
        .map(|&value| {
            let centered = value as f64 - mean;
            centered * centered
        })
        .sum::<f64>()
        / data.len() as f64;
    let std_dev = variance.sqrt().max(1e-12);

    let mut ranked_pixels = Vec::with_capacity(data.len());
    for y in 0..ny {
        for x in 0..nx {
            let idx = y * nx + x;
            let intensity = (data[idx] as f64 - mean) / std_dev;
            ranked_pixels.push((idx, intensity.abs(), intensity));
        }
    }

    ranked_pixels.sort_by(|a, b| cmp_f64_desc(a.1, b.1));
    let keep = ranked_pixels.len().min(MAX_SOURCE_POINTS);

    let mut points = Vec::with_capacity(keep);
    for &(idx, _rank, intensity) in ranked_pixels.iter().take(keep) {
        let x = idx % nx;
        let y = idx / nx;
        let grad_x = local_gradient(&data, nx, ny, x, y, true);
        let grad_y = local_gradient(&data, nx, ny, x, y, false);
        let x_norm = if nx > 1 {
            2.0 * x as f64 / (nx - 1) as f64 - 1.0
        } else {
            0.0
        };
        let y_norm = if ny > 1 {
            2.0 * y as f64 / (ny - 1) as f64 - 1.0
        } else {
            0.0
        };
        let radius = (x_norm * x_norm + y_norm * y_norm).sqrt();
        let angle = y_norm.atan2(x_norm);
        points.push(vec![
            x_norm,
            y_norm,
            intensity,
            intensity.abs(),
            grad_x,
            grad_y,
            radius,
            angle / std::f64::consts::PI,
        ]);
    }

    Ok(points)
}

fn local_gradient(data: &[f32], nx: usize, ny: usize, x: usize, y: usize, x_axis: bool) -> f64 {
    let (prev_x, prev_y, next_x, next_y) = if x_axis {
        let prev = x.saturating_sub(1);
        let next = (x + 1).min(nx.saturating_sub(1));
        (prev, y, next, y)
    } else {
        let prev = y.saturating_sub(1);
        let next = (y + 1).min(ny.saturating_sub(1));
        (x, prev, x, next)
    };

    let prev_idx = prev_y * nx + prev_x;
    let next_idx = next_y * nx + next_x;
    (data[next_idx] as f64 - data[prev_idx] as f64) * 0.5
}

fn jwst_point(row: JwstPublicObservation) -> Vec<f64> {
    vec![
        wrap_ra(row.s_ra),
        row.s_dec / 90.0,
        release_scalar(&row.t_obs_release),
        stable_token_value(&row.proposal_id),
        filter_count(&row.filters),
        stable_token_value(&row.instrument_name),
        stable_token_value(&row.dataproduct_type),
        row.calib_level.parse::<f64>().unwrap_or(0.0),
    ]
}

fn release_scalar(text: &str) -> f64 {
    parse_release_time(text)
        .map(|dt| dt.timestamp_millis() as f64 / 86_400_000.0 / 20_000.0)
        .unwrap_or(0.0)
}

fn parse_release_time(value: &str) -> Option<DateTime<Utc>> {
    if value.trim().is_empty() {
        return None;
    }
    if let Ok(dt) = DateTime::parse_from_rfc3339(value) {
        return Some(dt.with_timezone(&Utc));
    }
    let mjd = value.trim().parse::<f64>().ok()?;
    let epoch = Utc.with_ymd_and_hms(1858, 11, 17, 0, 0, 0).single()?;
    let millis = (mjd * 86_400_000.0).round() as i64;
    Some(epoch + Duration::milliseconds(millis))
}

fn filter_count(filters: &str) -> f64 {
    filters
        .split('|')
        .filter(|token| !token.trim().is_empty())
        .count() as f64
}

fn stable_token_value(text: &str) -> f64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in text.as_bytes() {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
    }
    (hash % 20_000) as f64 / 10_000.0 - 1.0
}

fn keyword_ratio(words: &[&str], vocabulary: &[&str]) -> f64 {
    let hits = words
        .iter()
        .filter(|word| vocabulary.iter().any(|needle| word.contains(needle)))
        .count();
    hits as f64 / words.len() as f64
}

fn grade_scalar(text: &str) -> f64 {
    match text.trim() {
        "A" => 1.0,
        "B" => 0.5,
        "C" => 0.0,
        "D" => -0.5,
        "E" => -1.0,
        _ => 0.0,
    }
}

fn lifted_window(points: &[Vec<f64>], cd_dim: usize, max_points: usize) -> Vec<Vec<f64>> {
    let count = points.len().min(max_points.max(3));
    evenly_subsample(points, count)
        .into_iter()
        .map(|point| harmonic_lift(&point, cd_dim))
        .collect()
}

fn harmonic_lift(point: &[f64], cd_dim: usize) -> Vec<f64> {
    let mut lifted = vec![0.0; cd_dim];
    let span = point.len().max(1);

    for index in 0..cd_dim {
        let phase = std::f64::consts::TAU * (index as f64 + 1.0) / cd_dim as f64;
        let a = point[index % span];
        let b = point[(index * 3 + 1) % span];
        let c = point[(index * 5 + 2) % span];
        lifted[index] = 0.60 * a + 0.25 * b * phase.sin() + 0.15 * c * phase.cos();
    }

    let norm = cd_norm_sq(&lifted).sqrt();
    if norm < 1e-12 {
        lifted[0] = 1.0;
        return lifted;
    }
    for value in &mut lifted {
        *value /= norm;
    }
    lifted
}

fn knn_graph(points: &[Vec<f64>], k: usize) -> UnGraph<(), ()> {
    let mut graph = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..points.len()).map(|_| graph.add_node(())).collect();
    let mut edges = std::collections::BTreeSet::new();

    for i in 0..points.len() {
        let mut distances = Vec::with_capacity(points.len().saturating_sub(1));
        for j in 0..points.len() {
            if i == j {
                continue;
            }
            distances.push((j, euclidean_distance(&points[i], &points[j])));
        }
        distances.sort_by(|a, b| cmp_f64_asc(a.1, b.1));
        for &(neighbor, _) in distances.iter().take(k.min(distances.len())) {
            let edge = if i < neighbor {
                (i, neighbor)
            } else {
                (neighbor, i)
            };
            edges.insert(edge);
        }
    }

    for (source, target) in edges {
        graph.add_edge(nodes[source], nodes[target], ());
    }

    graph
}

fn graph_metrics(graph: &UnGraph<(), ()>) -> GraphMetrics {
    let invariants = compute_budgeted_invariants(graph);
    let node_count = invariants.n_nodes().max(1);
    let edge_density = if invariants.n_nodes() > 1 {
        2.0 * invariants.n_edges() as f64
            / (invariants.n_nodes() * (invariants.n_nodes() - 1)) as f64
    } else {
        0.0
    };
    let triangle_density = if invariants.n_nodes() >= 3 {
        invariants.triangle_count() as f64 / combinations_3(invariants.n_nodes()) as f64
    } else {
        0.0
    };
    let component_ratio = invariants.n_components() as f64 / node_count as f64;
    let mean_degree = invariants.degrees().iter().sum::<usize>() as f64 / node_count as f64;
    let degree_std = degree_std(invariants.degrees(), mean_degree);

    GraphMetrics {
        node_count: invariants.n_nodes(),
        edge_density,
        triangle_density,
        component_ratio,
        mean_degree,
        degree_std,
    }
}

fn graph_distance(left: &GraphMetrics, right: &GraphMetrics) -> f64 {
    let left_signature = [
        left.edge_density,
        left.triangle_density,
        left.component_ratio,
        left.mean_degree,
        left.degree_std,
    ];
    let right_signature = [
        right.edge_density,
        right.triangle_density,
        right.component_ratio,
        right.mean_degree,
        right.degree_std,
    ];
    left_signature
        .iter()
        .zip(right_signature.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn boxkite_summary(points: &[Vec<f64>], cd_dim: usize) -> (Option<f64>, Option<f64>) {
    if cd_dim < 16 {
        return (None, None);
    }

    let mut captures = Vec::new();
    let mut dominant = Vec::new();
    for point in points {
        let spectrum = compute_alignment(&point[..16]);
        captures.push(spectrum.total_captured);
        let dominant_weight = spectrum.weights.iter().copied().fold(0.0_f64, f64::max);
        dominant.push(dominant_weight);
    }

    (Some(mean(&captures)), Some(mean(&dominant)))
}

fn sampled_associator_norms(points: &[Vec<f64>]) -> Vec<f64> {
    let n_triples = (points.len() / 3).clamp(1, 16);
    let mut values = Vec::with_capacity(n_triples);
    for triple in 0..n_triples {
        let base = triple * 3;
        let a = &points[base % points.len()];
        let b = &points[(base + 1) % points.len()];
        let c = &points[(base + 2) % points.len()];
        values.push(cd_associator_norm(a, b, c));
    }
    values
}

fn epsilon_from_knn(points: &[Vec<f64>], k: usize) -> f64 {
    if points.len() < 3 {
        return 1.0;
    }
    let mut kth_distances = Vec::with_capacity(points.len());
    for i in 0..points.len() {
        let mut distances = Vec::with_capacity(points.len().saturating_sub(1));
        for j in 0..points.len() {
            if i == j {
                continue;
            }
            distances.push(euclidean_distance(&points[i], &points[j]));
        }
        distances.sort_by(|a, b| cmp_f64_asc(*a, *b));
        let idx = k.min(distances.len().saturating_sub(1));
        kth_distances.push(distances[idx]);
    }
    1.2 * median(&kth_distances)
}

fn euclidean_distance(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right.iter())
        .map(|(a, b)| {
            let diff = a - b;
            diff * diff
        })
        .sum::<f64>()
        .sqrt()
}

fn normalize_columns(points: &mut [Vec<f64>]) {
    if points.is_empty() {
        return;
    }
    let dims = points[0].len();
    for dim in 0..dims {
        let values: Vec<f64> = points
            .iter()
            .map(|point| point[dim])
            .filter(|value| value.is_finite())
            .collect();
        let mean_value = mean(&values);
        let variance = if values.is_empty() {
            1.0
        } else {
            values
                .iter()
                .map(|value| {
                    let diff = value - mean_value;
                    diff * diff
                })
                .sum::<f64>()
                / values.len() as f64
        };
        let std_dev = variance.sqrt().max(1e-9);
        for point in points.iter_mut() {
            let value = point[dim];
            point[dim] = if value.is_finite() {
                (value - mean_value) / std_dev
            } else {
                0.0
            };
        }
    }
}

fn evenly_subsample(points: &[Vec<f64>], max_points: usize) -> Vec<Vec<f64>> {
    if points.len() <= max_points {
        return points.to_vec();
    }
    let mut subset = Vec::with_capacity(max_points);
    let last = points.len() - 1;
    for step in 0..max_points {
        let idx = (step * last) / (max_points - 1);
        subset.push(points[idx].clone());
    }
    subset
}

fn entropy_sample_budget(cd_dim: usize) -> usize {
    let divisor = (cd_dim / 16).max(1);
    (2048 / divisor).max(256)
}

fn negative_sign_density(cd_dim: usize) -> f64 {
    if cd_dim < 4 || !cd_dim.is_power_of_two() {
        return 0.0;
    }
    let mut negatives = 0usize;
    let mut total = 0usize;
    for i in 1..cd_dim {
        for j in (i + 1)..cd_dim {
            total += 1;
            if cd_basis_mul_sign(cd_dim, i, j) < 0 {
                negatives += 1;
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        negatives as f64 / total as f64
    }
}

fn safe_log10(value: f64) -> f64 {
    if value.is_finite() && value.abs() > 1e-12 {
        value.abs().log10()
    } else {
        0.0
    }
}

fn wrap_ra(ra_deg: f64) -> f64 {
    ((ra_deg / 180.0) - 1.0).clamp(-1.0, 1.0)
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

fn median(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| cmp_f64_asc(*a, *b));
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    }
}

fn degree_std(degrees: &[usize], mean_degree: f64) -> f64 {
    if degrees.is_empty() {
        return 0.0;
    }
    let variance = degrees
        .iter()
        .map(|degree| {
            let diff = *degree as f64 - mean_degree;
            diff * diff
        })
        .sum::<f64>()
        / degrees.len() as f64;
    variance.sqrt()
}

fn combinations_3(n: usize) -> usize {
    n.saturating_mul(n.saturating_sub(1))
        .saturating_mul(n.saturating_sub(2))
        / 6
}

fn cmp_f64_asc(left: f64, right: f64) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn cmp_f64_desc(left: f64, right: f64) -> Ordering {
    cmp_f64_asc(right, left)
}
