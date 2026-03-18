use anyhow::{Context, Result};
use arrow_array::RecordBatch;
use clap::Parser;
use csv::Writer;
use data_core::catalogs::jwst::parse_jwst_public_metadata_csv;
use fitsio::{FitsFile, hdu::HduInfo};
use gororoba_cli::data_governance::sha256_file;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

const DIMENSION_LADDER: &str = "4,8,16,32,64,128,256,512,1024";
const DIMENSION_GUIDE: &[(&str, &str)] = &[
    (
        "4",
        "Quaternion baseline: associative rotation and smooth geometric control.",
    ),
    (
        "8",
        "Octonion baseline: first non-associative comparison with still-rigid low-dimensional structure.",
    ),
    (
        "16",
        "Sedenion transition: zero-divisor onset, box-kites, annihilators, and basin splitting tests.",
    ),
    (
        "32",
        "Pathion extension: denser interaction-graph and higher-order zero-divisor web probes.",
    ),
    (
        "64",
        "Chingon scale-up: large graph-projection and alternativity-violation stress lane.",
    ),
    (
        "128",
        "128D extension: intermediate high-dimensional optimization and sparsity stress regime.",
    ),
    (
        "256",
        "256D Voudon lane: global imbalance-density and cosmology-facing smoothing hypotheses.",
    ),
    (
        "512",
        "512D filtration lane: large-scale closure and affine/coset obstruction tests.",
    ),
    (
        "1024",
        "1024D filtration lane: highest bounded prefix-chain and stress-test tier currently surfaced in-repo.",
    ),
];

const ANALYSIS_LANES: &[(&str, &str)] = &[
    (
        "Euclid morphology / physical measurements",
        "`euclid-fetch`, `euclid-dm-coupling`, `euclid-df-sweep`, `survey-crossmatch`, `harmonic-halo-stacking-manga`",
    ),
    (
        "JWST public metadata / COSMOS-Web context",
        "`fetch-datasets --dataset \"JWST Public Observation Metadata\"`, `mast-program-clustering`, `catalog-feature-cube`, `multi-dataset-ultrametric`",
    ),
    (
        "Basin / void / graph analogues",
        "`cosmic-dendrogram`, `generate-topological-voids`, `repo-visuals`, `survey-crossmatch`",
    ),
    (
        "Algebraic structure lanes",
        "`boxkite_alignment`, `subalgebra`, `graph_projections`, `projective_geometry`, `codebook`, `sedenion_lifting`",
    ),
];

#[derive(Parser, Debug)]
#[command(
    name = "cosmology-map-audit",
    about = "Audit weak-lensing and basin-of-attraction source caches and align them to Cayley-Dickson analysis lanes"
)]
struct Args {
    #[arg(long, default_value = "data/csv/cosmology_map_algebra_alignment.csv")]
    csv_out: PathBuf,

    #[arg(long, default_value = "docs/COSMOLOGY_MAP_ALGEBRA_ALIGNMENT.md")]
    markdown_out: PathBuf,
}

#[derive(Clone)]
struct SourceSpec {
    family: &'static str,
    dataset_id: &'static str,
    kind: &'static str,
    local_path: &'static str,
    repo_entrypoints: &'static str,
    alignment_note: &'static str,
}

#[derive(Serialize)]
struct CsvRow {
    family: String,
    dataset_id: String,
    kind: String,
    local_path: String,
    exists: bool,
    size_bytes: u64,
    sha256_16: String,
    payload_summary: String,
    repo_entrypoints: String,
    dimension_ladder: String,
    alignment_note: String,
}

struct MaterializedRow {
    csv: CsvRow,
    bytes_human: String,
}

const SOURCE_SPECS: &[SourceSpec] = &[
    SourceSpec {
        family: "euclid",
        dataset_id: "euclid_q1_morphology_catalogue",
        kind: "parquet",
        local_path: "data/external/euclid/zenodo/15106473/morphology_catalogue.parquet",
        repo_entrypoints: "`euclid-fetch -- zenodo-download --catalog morphology`; `survey-crossmatch -- euclid-lotss`",
        alignment_note: "Morphology probabilities and sky positions are the cleanest low-noise geometric lift into quaternion/octonion baselines before zero-divisor structure is introduced.",
    },
    SourceSpec {
        family: "euclid",
        dataset_id: "euclid_q1_useful_physical_measurements",
        kind: "parquet",
        local_path: "data/external/euclid/zenodo/15106473/useful_physical_measurements.parquet",
        repo_entrypoints: "`euclid-dm-coupling`; `euclid-df-sweep`; `harmonic-halo-stacking-manga`",
        alignment_note: "Sersic, redshift, stellar-mass, and luminosity fields are the current best in-repo bridge from weak-lensing-adjacent survey data to mass-profile construction.",
    },
    SourceSpec {
        family: "euclid",
        dataset_id: "euclid_q1_strong_lensing_candidates",
        kind: "csv",
        local_path: "data/external/euclid/zenodo/15025832/q1_discovery_engine_lens_catalog.csv",
        repo_entrypoints: "`euclid-fetch -- zenodo-download --catalog strong_lensing`",
        alignment_note: "Strong-lensing candidates are natural anchors for comparing local overdensity graphs against higher-dimensional interaction subgraphs.",
    },
    SourceSpec {
        family: "euclid",
        dataset_id: "euclid_q1_merger_classification",
        kind: "csv",
        local_path: "data/external/euclid/zenodo/17087034/Q1_merger_classification.csv",
        repo_entrypoints: "`euclid-fetch -- zenodo-download --catalog mergers`",
        alignment_note: "Merger labels provide a disturbance prior that can be separated from algebraic forcing when comparing basin-like morphology against non-associative lifts.",
    },
    SourceSpec {
        family: "euclid",
        dataset_id: "euclid_q1_tap_manifest",
        kind: "json",
        local_path: "data/external/euclid/tap/euclid_tap_manifest.json",
        repo_entrypoints: "`euclid-fetch -- tap-query ...`; `survey-crossmatch`",
        alignment_note: "This is the governed pointer to the bounded primary-survey query lane. It represents the 35 TB-scale Euclid catalog surface without attempting to mirror it wholesale.",
    },
    SourceSpec {
        family: "euclid",
        dataset_id: "euclid_official_release_pages",
        kind: "html_bundle",
        local_path: "data/external/cosmology_maps/euclid_cosmic_web",
        repo_entrypoints: "`cosmology-map-audit`",
        alignment_note: "Official ESA release context for interpreting the local Euclid supplementary catalogs as cosmic-web and weak-lensing inputs, not just generic tables.",
    },
    SourceSpec {
        family: "jwst",
        dataset_id: "jwst_public_observation_metadata",
        kind: "csv",
        local_path: "data/external/jwst_public_observations.csv",
        repo_entrypoints: "`fetch-datasets -- --dataset \"JWST Public Observation Metadata\"`; `mast-program-clustering`; `catalog-feature-cube`",
        alignment_note: "MAST metadata is the bounded repo-native bridge from general JWST archive state into field- and program-level selection for cosmology overlays.",
    },
    SourceSpec {
        family: "jwst",
        dataset_id: "jwst_cosmosweb_dark_matter_paper",
        kind: "pdf",
        local_path: "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/arxiv_2601.17239_ultra_high_resolution_dark_matter_map.pdf",
        repo_entrypoints: "`cosmology-map-audit`",
        alignment_note: "Authoritative paper for the weak-lensing mass-map analogue. Use as the descriptive baseline for filaments, clusters, and under-densities.",
    },
    SourceSpec {
        family: "jwst",
        dataset_id: "jwst_cosmosweb_massmap_fits_1",
        kind: "fits",
        local_path: "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/supplementary_data_1_m2.fits",
        repo_entrypoints: "`cosmology-map-audit`",
        alignment_note: "First directly cached mass-map-style FITS payload from the JWST weak-lensing paper; suitable for future image-statistics and graph-extraction lanes.",
    },
    SourceSpec {
        family: "jwst",
        dataset_id: "jwst_cosmosweb_massmap_fits_2",
        kind: "fits",
        local_path: "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/supplementary_data_3_m4.fits",
        repo_entrypoints: "`cosmology-map-audit`",
        alignment_note: "Second directly cached FITS payload for comparing projected mass structure against higher-dimensional zero-divisor graph projections.",
    },
    SourceSpec {
        family: "jwst",
        dataset_id: "jwst_cosmosweb_massmap_fits_3",
        kind: "fits",
        local_path: "data/external/cosmology_maps/jwst_cosmosweb_dark_matter/supplementary_data_5_m6.fits",
        repo_entrypoints: "`cosmology-map-audit`",
        alignment_note: "Third directly cached FITS payload; together the trio provides a bounded image lane without mirroring the full COSMOS-Web raw imaging release.",
    },
    SourceSpec {
        family: "basin",
        dataset_id: "boa_local_universe_paper",
        kind: "pdf",
        local_path: "data/external/cosmology_maps/basin_of_attraction/arxiv_2409.17261_identification_of_basins_of_attraction.pdf",
        repo_entrypoints: "`cosmology-map-audit`; `cosmic-dendrogram`; `generate-topological-voids`",
        alignment_note: "Authoritative basin-of-attraction formulation for streamlines, gravitational-potential minima, and void/basin segmentation analogies.",
    },
    SourceSpec {
        family: "basin",
        dataset_id: "boa_supplementary_movie",
        kind: "mp4",
        local_path: "data/external/cosmology_maps/basin_of_attraction/supplementary_movie_m1.mp4",
        repo_entrypoints: "`cosmology-map-audit`",
        alignment_note: "Dynamic basin segmentation reference for future streamline extraction and topological basin persistence studies.",
    },
    SourceSpec {
        family: "algebra_reference",
        dataset_id: "reggiani_sedenion_geometry",
        kind: "pdf",
        local_path: "data/external/arxiv_2411.18881_sedenion_geometry.pdf",
        repo_entrypoints: "`algebra-theory-crosswalk`; `boxkite_alignment`; `grassmannian`",
        alignment_note: "Core differential-geometric reference for mapping sedenion zero divisors onto recognized manifold language before cosmology analogies are made.",
    },
    SourceSpec {
        family: "algebra_reference",
        dataset_id: "koebisu_zd_holonomy",
        kind: "pdf",
        local_path: "data/external/arxiv_2512.13002_zd_holonomy.pdf",
        repo_entrypoints: "`algebra-theory-crosswalk`; `subalgebra`; `projective_geometry`",
        alignment_note: "Holonomy and singular-structure reference for making the sedenion side less ad hoc and more compatible with geometric field language.",
    },
    SourceSpec {
        family: "algebra_reference",
        dataset_id: "de_marrais_boxkites_iii",
        kind: "pdf",
        local_path: "papers/pdf/de_marrais_2004_math0403113_boxkites_III.pdf",
        repo_entrypoints: "`boxkite_alignment`; `graph_projections`",
        alignment_note: "Historical zero-divisor and box-kite source grounding the graph-like central substructure that visually resembles cosmological overdensity scaffolds.",
    },
    SourceSpec {
        family: "algebra_reference",
        dataset_id: "tang_sedenion_su5",
        kind: "pdf",
        local_path: "papers/pdf/tang_2023_230814768_sedenion_su5.pdf",
        repo_entrypoints: "`algebra-theory-crosswalk`; `projective_geometry`",
        alignment_note: "Example of a field-theory-facing sedenion construction, useful when linking the cosmology source family to higher-dimensional physical interpretations.",
    },
];

fn main() -> Result<()> {
    let args = Args::parse();
    let rows = SOURCE_SPECS
        .iter()
        .map(materialize_row)
        .collect::<Result<Vec<_>>>()?;
    write_csv(&args.csv_out, &rows)?;
    write_markdown(&args.markdown_out, &rows)?;
    println!("WROTE {}", args.csv_out.display());
    println!("WROTE {}", args.markdown_out.display());
    Ok(())
}

fn materialize_row(spec: &SourceSpec) -> Result<MaterializedRow> {
    let path = Path::new(spec.local_path);
    let exists = path.exists();
    let size_bytes = if exists {
        path.metadata()
            .with_context(|| format!("metadata {}", path.display()))?
            .len()
    } else {
        0
    };
    let sha256 = if exists && path.is_file() {
        sha256_file(path)?
    } else {
        String::new()
    };
    let payload_summary = payload_summary(path, spec.kind)?;
    Ok(MaterializedRow {
        bytes_human: human_bytes(size_bytes),
        csv: CsvRow {
            family: spec.family.to_string(),
            dataset_id: spec.dataset_id.to_string(),
            kind: spec.kind.to_string(),
            local_path: spec.local_path.to_string(),
            exists,
            size_bytes,
            sha256_16: sha256.chars().take(16).collect(),
            payload_summary,
            repo_entrypoints: spec.repo_entrypoints.to_string(),
            dimension_ladder: DIMENSION_LADDER.to_string(),
            alignment_note: spec.alignment_note.to_string(),
        },
    })
}

fn payload_summary(path: &Path, kind: &str) -> Result<String> {
    if !path.exists() {
        return Ok("missing".to_string());
    }
    match kind {
        "parquet" => parquet_summary(path),
        "csv" => csv_summary(path),
        "fits" => fits_summary(path),
        "pdf" => pdf_summary(path),
        "mp4" => Ok("supplementary video".to_string()),
        "json" => Ok("manifest / metadata".to_string()),
        "html_bundle" => directory_summary(path),
        _ => {
            if path.is_dir() {
                directory_summary(path)
            } else {
                Ok("file present".to_string())
            }
        }
    }
}

fn parquet_summary(path: &Path) -> Result<String> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata {}", path.display()))?;
    let num_rows = builder.metadata().file_metadata().num_rows();
    let num_cols = builder.schema().fields().len();
    Ok(format!("{num_rows} rows, {num_cols} columns"))
}

fn csv_summary(path: &Path) -> Result<String> {
    if path.ends_with("jwst_public_observations.csv") {
        let rows = parse_jwst_public_metadata_csv(path)
            .with_context(|| format!("parse JWST CSV {}", path.display()))?;
        let proposal_count = rows
            .iter()
            .map(|row| row.proposal_id.as_str())
            .collect::<std::collections::BTreeSet<_>>()
            .len();
        return Ok(format!(
            "{} JWST rows, {} unique proposals",
            rows.len(),
            proposal_count
        ));
    }
    let mut reader =
        csv::Reader::from_path(path).with_context(|| format!("open CSV {}", path.display()))?;
    let mut count = 0usize;
    for row in reader.records() {
        row.with_context(|| format!("read CSV row {}", path.display()))?;
        count += 1;
    }
    Ok(format!("{count} data rows"))
}

fn fits_summary(path: &Path) -> Result<String> {
    let path_str = path
        .to_str()
        .with_context(|| format!("non-UTF8 path {}", path.display()))?;
    let mut fits =
        FitsFile::open(path_str).with_context(|| format!("open FITS {}", path.display()))?;
    let hdu = fits.hdu(0usize).context("primary HDU")?;
    match &hdu.info {
        HduInfo::ImageInfo { shape, image_type } => {
            Ok(format!("FITS image {:?}, shape {:?}", image_type, shape))
        }
        other => Ok(format!("non-image HDU {:?}", other)),
    }
}

fn pdf_summary(path: &Path) -> Result<String> {
    let txt_path = path.with_extension("txt");
    if txt_path.exists() {
        let text = fs::read_to_string(&txt_path)
            .with_context(|| format!("read {}", txt_path.display()))?;
        let line_count = text.lines().count();
        let word_count = text.split_whitespace().count();
        Ok(format!(
            "PDF with text sidecar: {line_count} lines, {word_count} words"
        ))
    } else {
        Ok("PDF present".to_string())
    }
}

fn directory_summary(path: &Path) -> Result<String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(path).with_context(|| format!("read_dir {}", path.display()))? {
        let entry = entry?;
        if entry.file_type()?.is_file() {
            files.push(entry.file_name().to_string_lossy().to_string());
        }
    }
    files.sort();
    Ok(format!("{} files: {}", files.len(), files.join(", ")))
}

fn write_csv(path: &Path, rows: &[MaterializedRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut writer =
        Writer::from_path(path).with_context(|| format!("create {}", path.display()))?;
    for row in rows {
        writer.serialize(&row.csv)?;
    }
    writer.flush()?;
    Ok(())
}

fn write_markdown(path: &Path, rows: &[MaterializedRow]) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut out = String::new();
    out.push_str("# Cosmology Map Algebra Alignment\n\n");
    out.push_str(
        "This audit ties three cosmology-map families to the repository's Cayley-Dickson and survey-analysis infrastructure: Euclid weak-lensing / deep-field catalogs, JWST COSMOS-Web weak-lensing mass-map artifacts, and basin-of-attraction work on the local universe.\n\n",
    );
    out.push_str("## Acquisition status\n\n");
    out.push_str("- Euclid supplementary Q1 catalogs are already local and governed through the Rust `euclid-fetch` lane.\n");
    out.push_str("- JWST COSMOS-Web source material is cached locally as the paper, official pages, and three supplementary FITS mass-map payloads.\n");
    out.push_str("- Basin-of-attraction material is cached locally as the paper, official page, and supplementary movie.\n");
    out.push_str("- The full Euclid primary survey and full COSMOS-Web raw-imaging backends are not mirrored here; the repo holds bounded subsets and authoritative pointers instead.\n");
    out.push_str("- The algebra side is grounded by in-repo sedenion / box-kite / holonomy papers already cached locally.\n\n");
    out.push_str("## Inventory\n\n");
    out.push_str("| Family | Dataset | Kind | Local path | Size | Payload |\n");
    out.push_str("|---|---|---|---|---:|---|\n");
    for row in rows {
        out.push_str(&format!(
            "| {} | {} | {} | `{}` | {} | {} |\n",
            row.csv.family,
            row.csv.dataset_id,
            row.csv.kind,
            row.csv.local_path,
            row.bytes_human,
            escape_pipes(&row.csv.payload_summary)
        ));
    }
    out.push_str("\n## Repo analysis lanes\n\n");
    for (label, lanes) in ANALYSIS_LANES {
        out.push_str(&format!("- **{}**: {}\n", label, lanes));
    }
    out.push_str("\n## Dimension ladder\n\n");
    for (dim, note) in DIMENSION_GUIDE {
        out.push_str(&format!("- **{}D**: {}\n", dim, note));
    }
    out.push_str("\n## Alignment guidance\n\n");
    out.push_str("- Use **4D and 8D** as sanity baselines where geometry should remain interpretable without zero divisors.\n");
    out.push_str("- Use **16D and 32D** when testing whether basin boundaries, lensing ridges, or overdensity cores align with zero-divisor graphs or box-kite-like decompositions.\n");
    out.push_str("- Use **64D and 128D** for interaction-web and alternativity-violation stress tests once a lower-dimensional relation is stable.\n");
    out.push_str("- Use **256D, 512D, and 1024D** only as extrapolation and optimization lanes unless a lower-dimensional physical analogue is already established.\n");
    out.push_str("- Treat the cosmology sources as **observational fields and labels**, and the Cayley-Dickson tower as a **family of analysis projections**; do not treat visual resemblance alone as a physical equivalence claim.\n");
    fs::write(path, out).with_context(|| format!("write {}", path.display()))?;
    Ok(())
}

fn escape_pipes(text: &str) -> String {
    text.replace('|', "\\|")
}

fn human_bytes(size: u64) -> String {
    const UNITS: [&str; 5] = ["B", "KB", "MB", "GB", "TB"];
    let mut value = size as f64;
    let mut unit = 0usize;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    if unit == 0 {
        format!("{size} {}", UNITS[unit])
    } else {
        format!("{value:.1} {}", UNITS[unit])
    }
}

#[allow(dead_code)]
fn parquet_rows(path: &Path) -> Result<usize> {
    let file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)
        .with_context(|| format!("read parquet metadata {}", path.display()))?;
    let mut reader = builder.build().context("build parquet reader")?;
    let mut rows = 0usize;
    for batch in &mut reader {
        let batch: RecordBatch = batch.context("read parquet batch")?;
        rows += batch.num_rows();
    }
    Ok(rows)
}
