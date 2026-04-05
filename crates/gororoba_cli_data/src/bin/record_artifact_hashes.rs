//! Record local hashes for manifest-tracked artifacts.
//!
//! Migrated from bin/record_artifact_hashes.py.

use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{fs, io::Read, path::Path};

#[derive(Serialize, Deserialize, Debug)]
struct ArtifactProvenance {
    generated_at_utc: DateTime<Utc>,
    root: String,
    hashes: Vec<FileHash>,
    manifest: String,
    missing_manifest_paths: Vec<String>,
}

#[derive(Serialize, Deserialize, Debug)]
struct FileHash {
    path: String,
    size_bytes: u64,
    mtime_utc: DateTime<Utc>,
    sha256: String,
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path)?;
    let mut hasher = Sha256::new();
    let mut buffer = [0u8; 8192];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect())
}

fn main() -> Result<()> {
    let manifest_path = Path::new("data/artifacts/ARTIFACTS_MANIFEST.csv");
    let _artifacts_root = Path::new("data/artifacts");
    let output_path = Path::new("data/artifacts/PROVENANCE.local.json");

    if !manifest_path.exists() {
        return Ok(()); // Skip if no manifest
    }

    let mut reader = csv::Reader::from_path(manifest_path)?;
    let mut artifact_paths = Vec::new();
    for result in reader.records() {
        let record = result?;
        if let Some(p) = record.get(0)
            && !p.is_empty()
        {
            artifact_paths.push(p.to_string());
        }
    }

    let mut entries = Vec::new();
    let mut missing = Vec::new();

    for ap in artifact_paths {
        let path = Path::new(&ap);
        if !path.exists() {
            missing.push(ap);
            continue;
        }
        if !path.is_file() {
            continue;
        }

        let st = fs::metadata(path)?;
        entries.append(&mut vec![FileHash {
            path: ap.clone(),
            size_bytes: st.len(),
            mtime_utc: st.modified()?.into(),
            sha256: sha256_file(path)?,
        }]);
    }

    let payload = ArtifactProvenance {
        generated_at_utc: Utc::now(),
        root: "data/artifacts".to_string(),
        hashes: entries,
        manifest: manifest_path.to_string_lossy().to_string(),
        missing_manifest_paths: missing,
    };

    let json = serde_json::to_string_pretty(&payload)?;
    fs::write(output_path, json)?;

    println!("Wrote: {}", output_path.display());
    Ok(())
}
