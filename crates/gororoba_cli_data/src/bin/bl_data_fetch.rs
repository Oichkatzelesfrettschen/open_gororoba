//! Breakthrough Listen 6EQUJ5 data download orchestrator.
//!
//! Manages the download lifecycle for BL filterbank HDF5 files from the
//! Berkeley data archive. Generates a TOML manifest for reproducibility,
//! supports resume via HTTP Range, verifies SHA-256 hashes, and provides
//! a clean subcommand to delete raw data after analysis.
//!
//! Subcommands:
//! - `manifest`: Generate download manifest TOML from observation list
//! - `download`: Download files (with resume, progress, SHA-256)
//! - `verify`:   Verify downloaded files against manifest checksums
//! - `status`:   Show download progress
//! - `clean`:    Delete raw HDF5 files after analysis

use clap::{Parser, Subcommand};
use data_core::catalogs::bl_filterbank::{
    BlObservation, bl_6equj5_observations, observation_file_path,
};
use data_core::fetcher::{FetchError, compute_sha256};
use std::fs;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

#[derive(Parser)]
#[command(name = "bl-data-fetch", about = "BL 6EQUJ5 data download orchestrator")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate download manifest TOML from the observation list.
    Manifest {
        /// Output manifest file path.
        #[arg(long, default_value = "registry/bl_6equj5_manifest.toml")]
        output: PathBuf,

        /// Resolution level(s) to include (comma-separated: 0000,0001,0002).
        #[arg(long, default_value = "0002")]
        res: String,

        /// Nodes to include (comma-separated: blc40,blc41,...,blc47).
        #[arg(long, default_value = "blc40")]
        nodes: String,
    },
    /// Download files from the BL data archive.
    Download {
        /// Data directory for downloaded files.
        #[arg(long, default_value = "data/bl_6equj5_gbt")]
        data_dir: PathBuf,

        /// Resolution level to download.
        #[arg(long, default_value = "0002")]
        res: String,

        /// Node(s) to download (comma-separated).
        #[arg(long, default_value = "blc40")]
        nodes: String,

        /// Maximum concurrent downloads (respect server limits).
        #[arg(long, default_value_t = 1)]
        concurrency: usize,

        /// Skip files that already exist on disk.
        #[arg(long, default_value_t = true)]
        skip_existing: bool,
    },
    /// Verify downloaded files against SHA-256 checksums.
    Verify {
        /// Data directory containing downloaded files.
        #[arg(long, default_value = "data/bl_6equj5_gbt")]
        data_dir: PathBuf,

        /// Manifest TOML file with checksums.
        #[arg(long, default_value = "registry/bl_6equj5_manifest.toml")]
        manifest: PathBuf,
    },
    /// Show download status.
    Status {
        /// Data directory containing downloaded files.
        #[arg(long, default_value = "data/bl_6equj5_gbt")]
        data_dir: PathBuf,

        /// Resolution level to check.
        #[arg(long, default_value = "0002")]
        res: String,

        /// Node(s) to check (comma-separated).
        #[arg(long, default_value = "blc40")]
        nodes: String,
    },
    /// Delete raw HDF5 files after analysis (keeps manifest for reproducibility).
    Clean {
        /// Data directory containing downloaded files.
        #[arg(long, default_value = "data/bl_6equj5_gbt")]
        data_dir: PathBuf,

        /// Resolution level to clean.
        #[arg(long, default_value = "0000")]
        res: String,

        /// Require confirmation before deletion.
        #[arg(long, default_value_t = true)]
        confirm: bool,
    },
}

/// Build the full observation list for specified nodes and resolution levels.
fn build_observation_list(nodes: &[&str], res_levels: &[&str]) -> Vec<BlObservation> {
    let base_obs = bl_6equj5_observations();
    let mut result = Vec::new();

    for node in nodes {
        for obs in &base_obs {
            for res in res_levels {
                let mut o = BlObservation {
                    node: node.to_string(),
                    mjd: obs.mjd,
                    seconds: obs.seconds,
                    source: obs.source.clone(),
                    obs_id: obs.obs_id.clone(),
                    res_level: res.to_string(),
                    pointing_type: obs.pointing_type.clone(),
                    cadence: obs.cadence,
                };
                // Multi-node: only blc40 base obs are defined.
                // For other nodes, the seconds and source may differ.
                // For now, use the same observation list (blc40 template).
                if *node != "blc40" {
                    o.node = node.to_string();
                }
                result.push(o);
            }
        }
    }

    result
}

fn run_manifest(output: &Path, res: &str, nodes: &str) {
    let res_levels: Vec<&str> = res.split(',').map(|s| s.trim()).collect();
    let node_list: Vec<&str> = nodes.split(',').map(|s| s.trim()).collect();
    let obs_list = build_observation_list(&node_list, &res_levels);

    println!("=== BL 6EQUJ5 Download Manifest Generator ===");
    println!();
    println!("Nodes: {:?}", node_list);
    println!("Resolution levels: {:?}", res_levels);
    println!("Total files: {}", obs_list.len());
    println!();

    let mut toml_out = String::new();
    toml_out.push_str("# BL 6EQUJ5 GBT download manifest\n");
    toml_out.push_str("# Generated by bl-data-fetch manifest\n");
    toml_out.push_str("# This file is the reproducibility trail for the 6EQUJ5 dataset.\n\n");

    toml_out.push_str("[manifest]\n");
    toml_out.push_str("dataset = \"BL_6EQUJ5_GBT\"\n");
    toml_out.push_str(&format!(
        "created = \"{}\"\n",
        chrono::Local::now().format("%Y-%m-%d")
    ));
    toml_out.push_str("base_url = \"https://bldata.berkeley.edu/6EQUJ5/6EQUJ5_GBT/\"\n");
    toml_out.push_str(&format!("total_files = {}\n", obs_list.len()));
    toml_out.push_str(&format!("nodes = {:?}\n", node_list));
    toml_out.push_str(&format!("res_levels = {:?}\n", res_levels));
    toml_out.push('\n');

    for obs in &obs_list {
        toml_out.push_str("[[file]]\n");
        toml_out.push_str(&format!("filename = \"{}\"\n", obs.filename()));
        toml_out.push_str(&format!("url = \"{}\"\n", obs.download_url()));
        toml_out.push_str(&format!("node = \"{}\"\n", obs.node));
        toml_out.push_str(&format!("obs_id = \"{}\"\n", obs.obs_id));
        toml_out.push_str(&format!("source = \"{}\"\n", obs.source));
        toml_out.push_str(&format!("pointing_type = \"{}\"\n", obs.pointing_type));
        toml_out.push_str(&format!("cadence = {}\n", obs.cadence));
        toml_out.push_str(&format!("res_level = \"{}\"\n", obs.res_level));
        toml_out.push_str("sha256 = \"\"  # populated after download\n");
        toml_out.push_str("downloaded = false\n");
        toml_out.push_str("analyzed = false\n");
        toml_out.push('\n');
    }

    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).ok();
    }
    fs::write(output, &toml_out).expect("Failed to write manifest");
    println!("Manifest written to {}", output.display());
}

fn run_download(data_dir: &Path, res: &str, nodes: &str, _concurrency: usize, skip_existing: bool) {
    let res_levels: Vec<&str> = res.split(',').map(|s| s.trim()).collect();
    let node_list: Vec<&str> = nodes.split(',').map(|s| s.trim()).collect();
    let obs_list = build_observation_list(&node_list, &res_levels);

    fs::create_dir_all(data_dir).expect("Failed to create data directory");

    println!("=== BL 6EQUJ5 Data Download ===");
    println!();
    println!("Data directory: {}", data_dir.display());
    println!("Files to download: {}", obs_list.len());
    println!();

    let mut downloaded = 0u32;
    let mut skipped = 0u32;
    let mut failed = 0u32;
    let total = obs_list.len();

    for (i, obs) in obs_list.iter().enumerate() {
        let dest = observation_file_path(obs, data_dir);
        let url = obs.download_url();

        eprint!(
            "[{}/{}] {} ({})... ",
            i + 1,
            total,
            obs.filename(),
            obs.pointing_type
        );

        if skip_existing && dest.exists() {
            let size = fs::metadata(&dest).map(|m| m.len()).unwrap_or(0);
            if size > 0 {
                eprintln!("EXISTS ({:.1} MB)", size as f64 / 1e6);
                skipped += 1;
                continue;
            }
        }

        match download_file_with_resume(&url, &dest) {
            Ok(bytes) => {
                eprintln!("OK ({:.1} MB)", bytes as f64 / 1e6);

                // Compute SHA-256
                match compute_sha256(&dest) {
                    Ok(hash) => {
                        eprintln!("  SHA-256: {}", hash);
                    }
                    Err(e) => {
                        eprintln!("  SHA-256 failed: {}", e);
                    }
                }
                downloaded += 1;
            }
            Err(e) => {
                eprintln!("FAILED: {}", e);
                failed += 1;
            }
        }
    }

    println!();
    println!(
        "Summary: {} downloaded, {} skipped, {} failed",
        downloaded, skipped, failed
    );
}

/// Download a file with resume support via HTTP Range header.
fn download_file_with_resume(url: &str, dest: &Path) -> Result<u64, FetchError> {
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    // Check for partial download
    let existing_size = if dest.exists() {
        fs::metadata(dest).map(|m| m.len()).unwrap_or(0)
    } else {
        0
    };

    let mut request = ureq::get(url).header("User-Agent", "gororoba-fetch/0.1 (research)");

    if existing_size > 0 {
        request = request.header("Range", &format!("bytes={}-", existing_size));
        eprintln!("resuming from {} bytes... ", existing_size);
    }

    let response = request.call().map_err(|e| FetchError::HttpError {
        url: url.to_string(),
        source: Box::new(e),
    })?;

    let status: u16 = response.status().into();

    // 206 = partial content (resume), 200 = full file
    if status != 200 && status != 206 {
        return Err(FetchError::HttpStatus {
            url: url.to_string(),
            status,
        });
    }

    let mut file = if status == 206 {
        fs::OpenOptions::new().append(true).open(dest)?
    } else {
        fs::File::create(dest)?
    };

    let mut reader = response.into_body().into_reader();
    let mut buf = [0u8; 65536];
    let mut total_new = 0u64;

    loop {
        let n = reader.read(&mut buf).map_err(FetchError::Io)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        total_new += n as u64;
    }

    Ok(existing_size + total_new)
}

fn run_verify(data_dir: &Path, manifest: &Path) {
    println!("=== BL 6EQUJ5 Integrity Verification ===");
    println!();

    if !manifest.exists() {
        eprintln!("ERROR: Manifest not found: {}", manifest.display());
        eprintln!("Run `bl-data-fetch manifest` first.");
        std::process::exit(1);
    }

    // Parse manifest to extract expected hashes
    let content = fs::read_to_string(manifest).expect("Failed to read manifest");
    let table: toml::Table = content.parse().expect("Invalid TOML manifest");

    let files = match table.get("file") {
        Some(toml::Value::Array(arr)) => arr,
        _ => {
            println!("No files in manifest.");
            return;
        }
    };

    let mut verified = 0u32;
    let mut missing = 0u32;
    let mut mismatched = 0u32;
    let mut no_hash = 0u32;

    for entry in files {
        let filename = entry
            .get("filename")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");
        let expected_hash = entry.get("sha256").and_then(|v| v.as_str()).unwrap_or("");

        let filepath = data_dir.join(filename);

        if !filepath.exists() {
            println!("  [MISSING] {}", filename);
            missing += 1;
            continue;
        }

        if expected_hash.is_empty() {
            // Compute and report hash for filling in
            match compute_sha256(&filepath) {
                Ok(hash) => {
                    println!("  [NO_HASH] {} -> sha256={}", filename, hash);
                    no_hash += 1;
                }
                Err(e) => {
                    println!("  [ERROR] {} -> {}", filename, e);
                    mismatched += 1;
                }
            }
            continue;
        }

        match compute_sha256(&filepath) {
            Ok(hash) => {
                if hash == expected_hash {
                    println!("  [OK] {}", filename);
                    verified += 1;
                } else {
                    println!(
                        "  [MISMATCH] {} expected={} got={}",
                        filename, expected_hash, hash
                    );
                    mismatched += 1;
                }
            }
            Err(e) => {
                println!("  [ERROR] {} -> {}", filename, e);
                mismatched += 1;
            }
        }
    }

    println!();
    println!(
        "Summary: {} verified, {} missing, {} mismatched, {} no hash",
        verified, missing, mismatched, no_hash
    );
}

fn run_status(data_dir: &Path, res: &str, nodes: &str) {
    let res_levels: Vec<&str> = res.split(',').map(|s| s.trim()).collect();
    let node_list: Vec<&str> = nodes.split(',').map(|s| s.trim()).collect();
    let obs_list = build_observation_list(&node_list, &res_levels);

    println!("=== BL 6EQUJ5 Download Status ===");
    println!();
    println!("Data directory: {}", data_dir.display());
    println!();

    let mut present = 0u32;
    let mut absent = 0u32;
    let mut total_bytes = 0u64;

    for obs in &obs_list {
        let path = observation_file_path(obs, data_dir);
        if path.exists() {
            let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
            total_bytes += size;
            present += 1;
            println!(
                "  [FOUND] {} ({:.1} MB, {})",
                obs.filename(),
                size as f64 / 1e6,
                obs.pointing_type
            );
        } else {
            absent += 1;
            println!("  [ABSENT] {} ({})", obs.filename(), obs.pointing_type);
        }
    }

    println!();
    println!(
        "Summary: {}/{} present ({:.1} GB total), {} absent",
        present,
        present + absent,
        total_bytes as f64 / 1e9,
        absent
    );
}

fn run_clean(data_dir: &Path, res: &str, confirm: bool) {
    println!("=== BL 6EQUJ5 Data Cleanup ===");
    println!();

    let pattern = format!("*.rawspec.{}.h5", res);
    println!("Deleting: {} in {}", pattern, data_dir.display());

    if !data_dir.exists() {
        println!("Data directory does not exist. Nothing to clean.");
        return;
    }

    let mut files_to_delete: Vec<PathBuf> = Vec::new();
    let mut total_bytes = 0u64;

    if let Ok(entries) = fs::read_dir(data_dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if let Some(name) = path.file_name().and_then(|n| n.to_str())
                && name.ends_with(&format!(".rawspec.{}.h5", res))
            {
                let size = fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
                total_bytes += size;
                files_to_delete.push(path);
            }
        }
    }

    if files_to_delete.is_empty() {
        println!("No matching files found.");
        return;
    }

    println!(
        "Found {} files ({:.1} GB)",
        files_to_delete.len(),
        total_bytes as f64 / 1e9
    );

    if confirm {
        eprint!("Delete these files? [y/N] ");
        io::stdout().flush().ok();
        let mut input = String::new();
        io::stdin().read_line(&mut input).ok();
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Aborted.");
            return;
        }
    }

    let mut deleted = 0u32;
    for path in &files_to_delete {
        match fs::remove_file(path) {
            Ok(()) => {
                deleted += 1;
            }
            Err(e) => {
                eprintln!("  Failed to delete {}: {}", path.display(), e);
            }
        }
    }

    println!(
        "Deleted {} files ({:.1} GB freed)",
        deleted,
        total_bytes as f64 / 1e9
    );
}

fn main() {
    let args = Args::parse();
    match args.command {
        Command::Manifest { output, res, nodes } => run_manifest(&output, &res, &nodes),
        Command::Download {
            data_dir,
            res,
            nodes,
            concurrency,
            skip_existing,
        } => run_download(&data_dir, &res, &nodes, concurrency, skip_existing),
        Command::Verify { data_dir, manifest } => run_verify(&data_dir, &manifest),
        Command::Status {
            data_dir,
            res,
            nodes,
        } => run_status(&data_dir, &res, &nodes),
        Command::Clean {
            data_dir,
            res,
            confirm,
        } => run_clean(&data_dir, &res, confirm),
    }
}
