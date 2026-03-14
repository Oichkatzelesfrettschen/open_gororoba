//! Rust-native Euclid catalog fetcher.
//!
//! Replaces the Python `bin/fetch_euclid.py` and `bin/fetch_euclid_zenodo.py`
//! surfaces with a single clap-based Rust binary for:
//! - Zenodo supplementary catalog discovery/download.
//! - TAP table inspection.
//! - TAP cone or tile-limited ADQL queries against ESA / IRSA mirrors.

use anyhow::{Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use data_core::download_stack::{DownloadStack, TransferRequest};
use md5::Digest as _;
use reqwest::blocking::Client;
use serde::Serialize;
use serde_json::Value;
use std::{
    collections::BTreeMap,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};

const ZENODO_API: &str = "https://zenodo.org/api/records";
const BASE_OUT_DIR: &str = "data/external/euclid/zenodo";
const USER_AGENT: &str = "gororoba-euclid-fetch/0.1 (research)";
const SYNC_MAX_ROWS: usize = 500_000;

#[derive(Parser)]
#[command(name = "euclid-fetch")]
#[command(about = "Rust-native Euclid supplementary and TAP fetcher")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Discover Euclid Q1 supplementary science records on Zenodo.
    ZenodoDiscover {
        #[arg(long, default_value_t = 5)]
        max_pages: usize,
        #[arg(long, default_value_t = 100)]
        page_size: usize,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Emit the current governed Zenodo target state (verified vs provisional).
    ZenodoState {
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Download one or more governed Euclid Zenodo catalogs.
    ZenodoDownload {
        #[arg(long, default_value = "all")]
        catalog: String,
        #[arg(long, default_value_t = true)]
        skip_existing: bool,
        #[arg(long)]
        manifest: Option<PathBuf>,
    },
    /// Print the TAP /tables document for an endpoint.
    TapTables {
        #[arg(long, value_enum, default_value_t = TapEndpointArg::Esa)]
        endpoint: TapEndpointArg,
        #[arg(long)]
        output: Option<PathBuf>,
    },
    /// Execute a bounded TAP ADQL query in CSV/VOTable/TSV.
    TapQuery {
        #[arg(long, value_enum, default_value_t = TapEndpointArg::Esa)]
        endpoint: TapEndpointArg,
        #[arg(long, value_enum)]
        catalog: TapCatalogArg,
        #[arg(long)]
        ra_center: Option<f64>,
        #[arg(long)]
        dec_center: Option<f64>,
        #[arg(long)]
        radius_deg: Option<f64>,
        #[arg(long)]
        tile_id: Option<String>,
        #[arg(long, default_value = "csv")]
        format: String,
        #[arg(long, default_value_t = SYNC_MAX_ROWS)]
        maxrec: usize,
        #[arg(long)]
        output: Option<PathBuf>,
    },
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TapEndpointArg {
    Esa,
    Irsa,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TapCatalogArg {
    Mer,
    Phz,
    SpeRedshift,
    PhzClassification,
}

#[derive(Debug, Clone, Serialize)]
struct ZenodoDiscoveryManifest {
    generated_at_utc: String,
    records: Vec<ZenodoDiscoveryRecord>,
}

#[derive(Debug, Clone, Serialize)]
struct ZenodoDiscoveryRecord {
    record_id: u64,
    title: String,
    doi: String,
    file_count: usize,
    total_size_bytes: u64,
    extensions: Vec<String>,
    has_catalog: bool,
    image_only: bool,
    known: bool,
    large: bool,
    license: String,
}

#[derive(Debug, Clone, Serialize)]
struct ZenodoDownloadManifest {
    generated_at_utc: String,
    requested_catalog: String,
    downloads: Vec<ZenodoDownloadEntry>,
}

#[derive(Debug, Clone, Serialize)]
struct ZenodoDownloadEntry {
    catalog: String,
    record_id: u64,
    filename: String,
    path: String,
    status: String,
    bytes: u64,
    sha256: String,
    md5_verified: bool,
    url: String,
    note: String,
}

#[derive(Debug, Clone, Serialize)]
struct ZenodoStateReport {
    generated_at_utc: String,
    verified_targets: Vec<ZenodoStateEntry>,
    provisional_targets: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
struct ZenodoStateEntry {
    catalog: String,
    record_id: u64,
    description: String,
    target_files: Vec<String>,
}

#[derive(Debug, Clone)]
struct EuclidZenodoTarget {
    name: &'static str,
    record_id: u64,
    description: &'static str,
    target_files: &'static [&'static str],
}

const ZENODO_TARGETS: &[EuclidZenodoTarget] = &[
    EuclidZenodoTarget {
        name: "morphology",
        record_id: 15_106_473,
        description: "Euclid Q1 Visual Morphology Classification Catalogue",
        target_files: &[
            "morphology_catalogue.parquet",
            "useful_physical_measurements.parquet",
        ],
    },
    EuclidZenodoTarget {
        name: "strong_lensing",
        record_id: 15_025_832,
        description: "Euclid Q1 Strong Lensing Discovery Engine Candidates",
        target_files: &["q1_discovery_engine_lens_catalog.csv"],
    },
    EuclidZenodoTarget {
        name: "mergers",
        record_id: 17_087_034,
        description: "Euclid Q1 Galaxy Merger Classification",
        target_files: &["Q1_merger_classification.csv"],
    },
    EuclidZenodoTarget {
        name: "morphology_supplementary",
        record_id: 15_027_787,
        description: "Euclid Q1 Morphology latent-space representations",
        target_files: &[
            "representations_pca_40.parquet",
            "representations_pca_100.parquet",
        ],
    },
];

const PROVISIONAL_ZENODO_TARGETS: &[&str] = &[
    "photo_z",
    "galaxy_clustering",
    "compact_groups",
    "globular_clusters",
];

const ESA_TAP: &str = "https://eas.esac.esa.int/tap-server/tap";
const IRSA_TAP: &str = "https://irsa.ipac.caltech.edu/TAP";

const MER_COLUMNS: &[&str] = &[
    "object_id",
    "ra",
    "dec",
    "flux_vis_sersic",
    "fluxerr_vis_sersic",
    "flux_y_sersic",
    "flux_j_sersic",
    "flux_h_sersic",
    "flux_detection_total",
    "fluxerr_detection_total",
    "kron_radius",
    "ellipticity",
    "fwhm",
    "extended_flag",
    "point_like_flag",
    "flag_vis",
];

const PHZ_COLUMNS: &[&str] = &[
    "object_id",
    "phz_median",
    "phz_mode_1",
    "phz_mode_1_area",
    "phz_mode_2",
    "phz_70_int1",
    "phz_70_int2",
    "phz_90_int1",
    "phz_90_int2",
    "best_chi2",
    "phz_classification",
    "phz_flags",
    "phz_weight",
    "tom_bin_id",
];

const SPE_COLUMNS: &[&str] = &["object_id", "phz_classification", "phz_flags"];

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Command::ZenodoDiscover {
            max_pages,
            page_size,
            manifest,
        } => cmd_zenodo_discover(max_pages, page_size, manifest.as_deref()),
        Command::ZenodoDownload {
            catalog,
            skip_existing,
            manifest,
        } => cmd_zenodo_download(&catalog, skip_existing, manifest.as_deref()),
        Command::ZenodoState { report } => cmd_zenodo_state(report.as_deref()),
        Command::TapTables { endpoint, output } => cmd_tap_tables(endpoint, output.as_deref()),
        Command::TapQuery {
            endpoint,
            catalog,
            ra_center,
            dec_center,
            radius_deg,
            tile_id,
            format,
            maxrec,
            output,
        } => cmd_tap_query(TapQueryArgs {
            endpoint,
            catalog,
            ra_center,
            dec_center,
            radius_deg,
            tile_id: tile_id.as_deref(),
            format: &format,
            maxrec,
            output: output.as_deref(),
        }),
    }
}

fn cmd_zenodo_state(report: Option<&Path>) -> Result<()> {
    let state = ZenodoStateReport {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        verified_targets: ZENODO_TARGETS
            .iter()
            .map(|target| ZenodoStateEntry {
                catalog: target.name.to_string(),
                record_id: target.record_id,
                description: target.description.to_string(),
                target_files: target.target_files.iter().map(|item| item.to_string()).collect(),
            })
            .collect(),
        provisional_targets: PROVISIONAL_ZENODO_TARGETS
            .iter()
            .map(|item| item.to_string())
            .collect(),
    };
    let output = report
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("reports/euclid_zenodo_state_2026-03-13.toml"));
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, toml::to_string_pretty(&state)?)?;
    println!("Verified targets:   {}", state.verified_targets.len());
    println!("Provisional targets: {}", state.provisional_targets.len());
    println!("Report:             {}", output.display());
    Ok(())
}

struct TapQueryArgs<'a> {
    endpoint: TapEndpointArg,
    catalog: TapCatalogArg,
    ra_center: Option<f64>,
    dec_center: Option<f64>,
    radius_deg: Option<f64>,
    tile_id: Option<&'a str>,
    format: &'a str,
    maxrec: usize,
    output: Option<&'a Path>,
}

fn cmd_zenodo_discover(max_pages: usize, page_size: usize, manifest: Option<&Path>) -> Result<()> {
    let stack = build_stack();
    let mut records = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    for page in 1..=max_pages {
        let url = format!(
            "{ZENODO_API}?q=title:euclid+q1&size={page_size}&sort=bestmatch&page={page}"
        );
        let response = stack
            .fetch_text(&TransferRequest::probe(url))
            .map_err(|e| anyhow!(e.to_string()))?;
        let json: Value = serde_json::from_str(&response)?;
        let Some(hits) = json
            .get("hits")
            .and_then(|hits| hits.get("hits"))
            .and_then(Value::as_array)
        else {
            break;
        };
        if hits.is_empty() {
            break;
        }
        for hit in hits {
            let record_id = hit.get("id").and_then(Value::as_u64).unwrap_or_default();
            if record_id == 0 || !seen.insert(record_id) {
                continue;
            }
            let metadata = hit.get("metadata").unwrap_or(&Value::Null);
            let resource_type = metadata
                .get("resource_type")
                .and_then(|value| value.get("type"))
                .and_then(Value::as_str)
                .unwrap_or_default();
            if resource_type != "dataset" {
                continue;
            }
            let title = metadata
                .get("title")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let creators = metadata
                .get("creators")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.get("name").and_then(Value::as_str))
                        .collect::<Vec<_>>()
                        .join(" ")
                })
                .unwrap_or_default()
                .to_ascii_lowercase();
            let title_lower = title.to_ascii_lowercase();
            if !title_lower.contains("euclid") && !creators.contains("euclid") {
                continue;
            }
            let files = hit
                .get("files")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let total_size_bytes = files
                .iter()
                .filter_map(|file| file.get("size").and_then(Value::as_u64))
                .sum::<u64>();
            let mut extensions = std::collections::BTreeSet::new();
            for file in &files {
                if let Some(name) = file.get("key").and_then(Value::as_str)
                    && let Some((_, ext)) = name.rsplit_once('.')
                {
                    extensions.insert(ext.to_ascii_lowercase());
                }
            }
            let extensions_vec: Vec<String> = extensions.into_iter().collect();
            let has_catalog = extensions_vec
                .iter()
                .any(|ext| matches!(ext.as_str(), "parquet" | "csv" | "fits" | "ecsv" | "hdf5"));
            let image_only = !has_catalog
                && extensions_vec
                    .iter()
                    .all(|ext| matches!(ext.as_str(), "fits" | "png" | "jpg" | "jpeg" | "tiff"));
            records.push(ZenodoDiscoveryRecord {
                record_id,
                title,
                doi: hit
                    .get("doi")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                file_count: files.len(),
                total_size_bytes,
                extensions: extensions_vec,
                has_catalog,
                image_only,
                known: ZENODO_TARGETS
                    .iter()
                    .any(|target| target.record_id == record_id),
                large: total_size_bytes > 50 * 1024 * 1024 * 1024_u64,
                license: metadata
                    .get("license")
                    .and_then(|value| value.get("id"))
                    .and_then(Value::as_str)
                    .unwrap_or("unknown")
                    .to_string(),
            });
        }
    }
    let manifest_model = ZenodoDiscoveryManifest {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        records,
    };
    let out_path = manifest
        .map(PathBuf::from)
        .unwrap_or_else(default_zenodo_discovery_manifest_path);
    write_json(&out_path, &manifest_model)?;
    println!("Discovery manifest written to {}", out_path.display());
    Ok(())
}

fn cmd_zenodo_download(catalog: &str, skip_existing: bool, manifest: Option<&Path>) -> Result<()> {
    let targets = resolve_targets(catalog)?;
    let stack = build_stack();
    let mut entries = Vec::new();
    for target in targets {
        println!(
            "Resolving Zenodo record {} ({})...",
            target.record_id, target.description
        );
        let url = format!("{ZENODO_API}/{}", target.record_id);
        let metadata_text = stack
            .fetch_text(&TransferRequest::probe(url))
            .map_err(|e| anyhow!(e.to_string()))?;
        let metadata: Value = serde_json::from_str(&metadata_text)?;
        let title = metadata
            .get("metadata")
            .and_then(|value| value.get("title"))
            .and_then(Value::as_str)
            .unwrap_or_default();
        if !title.to_ascii_lowercase().contains("euclid") {
            bail!(
                "Zenodo record {} resolved to a non-Euclid title: {}",
                target.record_id,
                title
            );
        }
        let files = metadata
            .get("files")
            .and_then(Value::as_array)
            .ok_or_else(|| anyhow!("No files array for record {}", target.record_id))?;
        let files_by_name = files
            .iter()
            .filter_map(|file| {
                let name = file.get("key").and_then(Value::as_str)?;
                Some((name.to_string(), file))
            })
            .collect::<BTreeMap<_, _>>();
        for &filename in target.target_files {
            let file = files_by_name.get(filename).ok_or_else(|| {
                anyhow!(
                    "Target file {} missing from record {}",
                    filename,
                    target.record_id
                )
            })?;
            let checksum = file
                .get("checksum")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .strip_prefix("md5:")
                .unwrap_or_default()
                .to_string();
            let download_url = file
                .get("links")
                .and_then(|links| links.get("self"))
                .and_then(Value::as_str)
                .or_else(|| {
                    file.get("links")
                        .and_then(|links| links.get("content"))
                        .and_then(Value::as_str)
                })
                .ok_or_else(|| anyhow!("No download link for {}", filename))?;
            let dest = PathBuf::from(BASE_OUT_DIR)
                .join(target.record_id.to_string())
                .join(filename);
            let expected_size = file.get("size").and_then(Value::as_u64).unwrap_or_default();
            if skip_existing && dest.exists() && verify_md5(&dest, &checksum)? {
                entries.push(ZenodoDownloadEntry {
                    catalog: target.name.to_string(),
                    record_id: target.record_id,
                    filename: filename.to_string(),
                    path: dest.display().to_string(),
                    status: "skipped_existing".to_string(),
                    bytes: dest.metadata()?.len(),
                    sha256: data_core::compute_sha256(&dest)?,
                    md5_verified: true,
                    url: download_url.to_string(),
                    note: "Existing file matched Zenodo MD5".to_string(),
                });
                continue;
            }
            let mut request = TransferRequest::download(download_url, &dest);
            request.note = Some(format!("euclid_zenodo:{}:{}", target.name, filename));
            let result = stack
                .recover(&request)
                .map_err(|e| anyhow!("{} -> {}: {}", download_url, dest.display(), e))?;
            let md5_verified = checksum.is_empty() || verify_md5(&dest, &checksum)?;
            if !md5_verified {
                bail!("MD5 mismatch for {}", dest.display());
            }
            if expected_size > 0 && result.bytes != expected_size {
                println!(
                    "WARNING: {} expected {} bytes, got {}",
                    filename, expected_size, result.bytes
                );
            }
            entries.push(ZenodoDownloadEntry {
                catalog: target.name.to_string(),
                record_id: target.record_id,
                filename: filename.to_string(),
                path: dest.display().to_string(),
                status: "downloaded".to_string(),
                bytes: result.bytes,
                sha256: result.sha256.unwrap_or_default(),
                md5_verified,
                url: download_url.to_string(),
                note: result.note,
            });
        }
    }
    let manifest_model = ZenodoDownloadManifest {
        generated_at_utc: chrono::Utc::now().to_rfc3339(),
        requested_catalog: catalog.to_string(),
        downloads: entries,
    };
    let out_path = manifest
        .map(PathBuf::from)
        .unwrap_or_else(default_zenodo_download_manifest_path);
    write_json(&out_path, &manifest_model)?;
    println!("Download manifest written to {}", out_path.display());
    Ok(())
}

fn cmd_tap_tables(endpoint: TapEndpointArg, output: Option<&Path>) -> Result<()> {
    let client = tap_client()?;
    let url = format!("{}/tables", tap_endpoint_url(endpoint));
    let body = client
        .get(&url)
        .header(reqwest::header::USER_AGENT, USER_AGENT)
        .send()?
        .error_for_status()?
        .text()?;
    if let Some(path) = output {
        write_text(path, &body)?;
        println!("TAP tables written to {}", path.display());
    } else {
        println!("{body}");
    }
    Ok(())
}

fn cmd_tap_query(args: TapQueryArgs<'_>) -> Result<()> {
    let TapQueryArgs {
        endpoint,
        catalog,
        ra_center,
        dec_center,
        radius_deg,
        tile_id,
        format,
        maxrec,
        output,
    } = args;
    let adql = if let Some(tile_id) = tile_id {
        build_tile_adql(endpoint, catalog, tile_id, maxrec)?
    } else {
        let ra_center =
            ra_center.ok_or_else(|| anyhow!("--ra-center is required without --tile-id"))?;
        let dec_center =
            dec_center.ok_or_else(|| anyhow!("--dec-center is required without --tile-id"))?;
        let radius_deg =
            radius_deg.ok_or_else(|| anyhow!("--radius-deg is required without --tile-id"))?;
        build_cone_adql(endpoint, catalog, ra_center, dec_center, radius_deg, maxrec)?
    };
    let body = tap_sync_query(endpoint, &adql, format, maxrec)?;
    let out_path = output
        .map(PathBuf::from)
        .unwrap_or_else(|| default_tap_query_path(endpoint, catalog, format));
    write_text(&out_path, &body)?;
    println!("Query output written to {}", out_path.display());
    Ok(())
}

fn build_stack() -> DownloadStack {
    DownloadStack::new().with_user_agent(USER_AGENT)
}

fn resolve_targets(catalog: &str) -> Result<Vec<&'static EuclidZenodoTarget>> {
    if catalog == "all" {
        return Ok(ZENODO_TARGETS.iter().collect());
    }
    if PROVISIONAL_ZENODO_TARGETS.contains(&catalog) {
        bail!(
            "Catalog {} is currently disabled until its official Euclid Zenodo record is verified",
            catalog
        );
    }
    let target = ZENODO_TARGETS
        .iter()
        .find(|target| target.name == catalog)
        .ok_or_else(|| anyhow!("Unknown catalog {catalog}"))?;
    Ok(vec![target])
}

fn tap_endpoint_url(endpoint: TapEndpointArg) -> &'static str {
    match endpoint {
        TapEndpointArg::Esa => ESA_TAP,
        TapEndpointArg::Irsa => IRSA_TAP,
    }
}

fn tap_table_name(endpoint: TapEndpointArg, catalog: TapCatalogArg) -> &'static str {
    match (endpoint, catalog) {
        (TapEndpointArg::Esa, TapCatalogArg::Mer) => "catalogue.mer_catalogue",
        (TapEndpointArg::Esa, TapCatalogArg::Phz) => "catalogue.phz_photo_z",
        (TapEndpointArg::Esa, TapCatalogArg::SpeRedshift) => "catalogue.phz_classification",
        (TapEndpointArg::Esa, TapCatalogArg::PhzClassification) => "catalogue.phz_classification",
        (TapEndpointArg::Irsa, TapCatalogArg::Mer) => "euclid_q1_mer_catalogue",
        (TapEndpointArg::Irsa, TapCatalogArg::Phz) => "euclid_q1_phz_photo_z",
        (TapEndpointArg::Irsa, TapCatalogArg::SpeRedshift) => {
            "euclid_q1_spectro_zcatalog_spe_quality"
        }
        (TapEndpointArg::Irsa, TapCatalogArg::PhzClassification) => "euclid_q1_phz_classification",
    }
}

fn build_cone_adql(
    endpoint: TapEndpointArg,
    catalog: TapCatalogArg,
    ra_center: f64,
    dec_center: f64,
    radius_deg: f64,
    maxrec: usize,
) -> Result<String> {
    let cos_dec = dec_center.to_radians().cos().max(0.01);
    let ra_half = radius_deg / cos_dec;
    let dec_half = radius_deg;
    let ra_min = ra_center - ra_half;
    let ra_max = ra_center + ra_half;
    let dec_min = dec_center - dec_half;
    let dec_max = dec_center + dec_half;
    let mer_table = tap_table_name(endpoint, TapCatalogArg::Mer);
    let table = tap_table_name(endpoint, catalog);
    let columns = match catalog {
        TapCatalogArg::Mer => MER_COLUMNS,
        TapCatalogArg::Phz => PHZ_COLUMNS,
        TapCatalogArg::SpeRedshift | TapCatalogArg::PhzClassification => SPE_COLUMNS,
    };
    let select = columns.join(", ");
    if matches!(catalog, TapCatalogArg::Mer) {
        Ok(format!(
            "SELECT TOP {maxrec} {select} FROM {table} \
             WHERE ra BETWEEN {ra_min} AND {ra_max} \
             AND dec BETWEEN {dec_min} AND {dec_max}"
        ))
    } else {
        Ok(format!(
            "SELECT TOP {maxrec} p.{select} FROM {table} AS p \
             JOIN {mer_table} AS m ON p.object_id = m.object_id \
             WHERE m.ra BETWEEN {ra_min} AND {ra_max} \
             AND m.dec BETWEEN {dec_min} AND {dec_max}"
        ))
    }
}

fn build_tile_adql(
    endpoint: TapEndpointArg,
    catalog: TapCatalogArg,
    tile_id: &str,
    maxrec: usize,
) -> Result<String> {
    let table = tap_table_name(endpoint, catalog);
    let columns = match catalog {
        TapCatalogArg::Mer => MER_COLUMNS,
        TapCatalogArg::Phz => PHZ_COLUMNS,
        TapCatalogArg::SpeRedshift | TapCatalogArg::PhzClassification => SPE_COLUMNS,
    };
    let select = columns.join(", ");
    Ok(format!(
        "SELECT TOP {maxrec} {select} FROM {table} WHERE tile_id = '{}'",
        escape_adql_string(tile_id)
    ))
}

fn tap_sync_query(
    endpoint: TapEndpointArg,
    adql: &str,
    format: &str,
    maxrec: usize,
) -> Result<String> {
    let client = tap_client()?;
    let url = format!("{}/sync", tap_endpoint_url(endpoint));
    let response = client
        .post(&url)
        .header(reqwest::header::USER_AGENT, USER_AGENT)
        .form(&[
            ("REQUEST", "doQuery"),
            ("LANG", "ADQL"),
            ("FORMAT", format),
            ("MAXREC", &maxrec.to_string()),
            ("QUERY", adql),
        ])
        .send()?
        .error_for_status()?
        .text()?;
    Ok(response)
}

fn tap_client() -> Result<Client> {
    Ok(Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?)
}

fn verify_md5(path: &Path, expected_md5: &str) -> Result<bool> {
    if expected_md5.is_empty() {
        return Ok(true);
    }
    let mut file = fs::File::open(path)?;
    let mut digest = md5::Md5::new();
    let mut buffer = [0u8; 1 << 20];
    loop {
        let read = std::io::Read::read(&mut file, &mut buffer)?;
        if read == 0 {
            break;
        }
        md5::Digest::update(&mut digest, &buffer[..read]);
    }
    Ok(format!("{:x}", md5::Digest::finalize(digest)) == expected_md5)
}

fn write_json<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let body = serde_json::to_string_pretty(value)?;
    fs::write(path, body)?;
    Ok(())
}

fn write_text(path: &Path, body: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body)?;
    Ok(())
}

fn default_zenodo_discovery_manifest_path() -> PathBuf {
    PathBuf::from(BASE_OUT_DIR).join("euclid_zenodo_discovery.json")
}

fn default_zenodo_download_manifest_path() -> PathBuf {
    PathBuf::from(BASE_OUT_DIR).join("euclid_zenodo_manifest.json")
}

fn default_tap_query_path(
    endpoint: TapEndpointArg,
    catalog: TapCatalogArg,
    format: &str,
) -> PathBuf {
    let endpoint_name = match endpoint {
        TapEndpointArg::Esa => "esa",
        TapEndpointArg::Irsa => "irsa",
    };
    let catalog_name = match catalog {
        TapCatalogArg::Mer => "mer",
        TapCatalogArg::Phz => "phz",
        TapCatalogArg::SpeRedshift => "spe_redshift",
        TapCatalogArg::PhzClassification => "phz_classification",
    };
    PathBuf::from("data/external/euclid/tap")
        .join(format!("{}_{}.{}", endpoint_name, catalog_name, format))
}

fn escape_adql_string(value: &str) -> String {
    value.replace('\'', "''")
}
