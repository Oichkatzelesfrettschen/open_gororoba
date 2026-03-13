//! Provenance recording for gororoba-engine.
//!
//! Automatically records hashes of produced datasets into a compatibility
//! artifact-hash ledger. The canonical control plane can ingest this downstream.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{BufReader, Read, Write},
    path::Path,
};

pub struct ProvenanceRecorder;

impl ProvenanceRecorder {
    /// Compute SHA256 of a file and append it to the claims registry.
    pub fn record_artifact<P: AsRef<Path>>(path: P, claim_id: &str) -> Result<String> {
        let path = path.as_ref();
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 1024 * 1024];

        loop {
            let n = reader.read(&mut buffer)?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }

        let hash = format!("{:x}", hasher.finalize());

        // Append to the compatibility artifact hash ledger.
        // The canonical control plane ingests this downstream; this helper
        // intentionally avoids writing into registry/claims.toml directly.
        let mut registry = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("registry/artifact_hashes.toml")?;

        writeln!(
            registry,
            "[[artifact]]\nid = \"{}\"\npath = \"{}\"\nhash = \"{}\"\ntimestamp = \"{}\"",
            claim_id,
            path.display(),
            hash,
            chrono::Utc::now().to_rfc3339()
        )?;

        Ok(hash)
    }
}
