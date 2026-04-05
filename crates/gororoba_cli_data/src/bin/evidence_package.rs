//! Evidence artifact packaging and checksum verification (Sprint 51, NA-029).
//!
//! Walks `data/evidence/` to catalog all evidence artifacts, computes
//! SHA-256 checksums, and produces a MANIFEST.toml for reproducibility
//! and publication provenance.
//!
//! Subcommands:
//! - `check`:  List artifacts and flag any missing/empty files
//! - `bundle`: Compute SHA-256 for all artifacts, write MANIFEST.toml
//! - `verify`: Re-check existing MANIFEST.toml against current files

use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::{Path, PathBuf},
    time::SystemTime,
};
use walkdir::WalkDir;

#[derive(Parser)]
#[command(
    name = "evidence-package",
    about = "Evidence artifact packaging and checksum verification"
)]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// List evidence artifacts and flag missing/empty files.
    Check {
        /// Evidence directory to scan.
        #[arg(long, default_value = "data/evidence")]
        dir: PathBuf,
    },
    /// Compute SHA-256 checksums and write MANIFEST.toml.
    Bundle {
        /// Evidence directory to scan.
        #[arg(long, default_value = "data/evidence")]
        dir: PathBuf,

        /// Output manifest path.
        #[arg(long, default_value = "data/evidence/MANIFEST.toml")]
        output: PathBuf,

        /// Skip missing or unreadable files instead of failing.
        #[arg(long)]
        skip_missing: bool,
    },
    /// Verify existing MANIFEST.toml against current files.
    Verify {
        /// Path to manifest file.
        #[arg(long, default_value = "data/evidence/MANIFEST.toml")]
        manifest: PathBuf,
    },
}

/// Compute SHA-256 hex digest of a file.
fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let hash = Sha256::digest(&bytes);
    Ok(hash.iter().map(|byte| format!("{byte:02x}")).collect())
}

/// Get the git revision of HEAD.
fn git_rev() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .and_then(|out| {
            if out.status.success() {
                String::from_utf8(out.stdout)
                    .ok()
                    .map(|s| s.trim().to_string())
            } else {
                None
            }
        })
        .unwrap_or_else(|| "unknown".to_string())
}

/// Collect all evidence files (TOML, CSV, JSON) under a directory.
fn collect_evidence_files(dir: &Path) -> Vec<PathBuf> {
    let mut files: Vec<PathBuf> = WalkDir::new(dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter(|e| {
            let ext = e.path().extension().and_then(|s| s.to_str()).unwrap_or("");
            matches!(ext, "toml" | "csv" | "json" | "txt")
        })
        .filter(|e| {
            // Exclude the manifest itself
            e.path().file_name().is_none_or(|n| n != "MANIFEST.toml")
        })
        .map(|e| e.into_path())
        .collect();
    files.sort();
    files
}

fn main() {
    let args = Args::parse();

    match args.command {
        Command::Check { dir } => {
            if !dir.is_dir() {
                eprintln!("ERROR: {} is not a directory", dir.display());
                std::process::exit(1);
            }

            let files = collect_evidence_files(&dir);
            let mut present = 0usize;
            let mut missing = 0usize;
            let mut empty = 0usize;

            println!("=== Evidence Artifact Check ===");
            println!("Directory: {}", dir.display());
            println!();

            for path in &files {
                let rel = path.strip_prefix(&dir).unwrap_or(path);
                match fs::metadata(path) {
                    Ok(meta) if meta.len() == 0 => {
                        println!("  EMPTY   {}", rel.display());
                        empty += 1;
                    }
                    Ok(_) => {
                        println!("  OK      {}", rel.display());
                        present += 1;
                    }
                    Err(_) => {
                        println!("  MISSING {}", rel.display());
                        missing += 1;
                    }
                }
            }

            println!();
            println!(
                "Total: {}, Present: {}, Empty: {}, Missing: {}",
                files.len(),
                present,
                empty,
                missing
            );

            if missing > 0 || empty > 0 {
                std::process::exit(1);
            }
        }

        Command::Bundle {
            dir,
            output,
            skip_missing,
        } => {
            if !dir.is_dir() {
                eprintln!("ERROR: {} is not a directory", dir.display());
                std::process::exit(1);
            }

            let files = collect_evidence_files(&dir);
            let git_revision = git_rev();
            let now = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .map_or(0, |d| d.as_secs());

            let mut manifest_lines = Vec::new();
            manifest_lines.push("# Evidence Artifact Manifest".to_string());
            manifest_lines.push("# Auto-generated by evidence-package".to_string());
            manifest_lines.push(String::new());
            manifest_lines.push("[metadata]".to_string());
            manifest_lines.push(format!("generated_unix = {now}"));
            manifest_lines.push(format!("git_rev = \"{git_revision}\""));
            manifest_lines.push(format!("evidence_dir = \"{}\"", dir.display()));
            manifest_lines.push(String::new());

            let mut present = 0usize;
            let mut missing = 0usize;

            for path in &files {
                let rel = path.strip_prefix(&dir).unwrap_or(path);
                let rel_str = rel.display().to_string().replace('\\', "/");

                match sha256_file(path) {
                    Ok(hash) => {
                        let size = fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                        manifest_lines.push("[[artifact]]".to_string());
                        manifest_lines.push(format!("path = \"{rel_str}\""));
                        manifest_lines.push(format!("sha256 = \"{hash}\""));
                        manifest_lines.push(format!("size_bytes = {size}"));
                        manifest_lines.push("status = \"present\"".to_string());
                        manifest_lines.push(String::new());
                        present += 1;
                    }
                    Err(e) => {
                        if skip_missing {
                            manifest_lines.push("[[artifact]]".to_string());
                            manifest_lines.push(format!("path = \"{rel_str}\""));
                            manifest_lines.push("sha256 = \"\"".to_string());
                            manifest_lines.push("size_bytes = 0".to_string());
                            manifest_lines.push("status = \"missing\"".to_string());
                            manifest_lines.push(format!("note = \"{e}\""));
                            manifest_lines.push(String::new());
                            missing += 1;
                        } else {
                            eprintln!("ERROR: {e}");
                            std::process::exit(1);
                        }
                    }
                }
            }

            // Write summary at the end
            manifest_lines.push("[summary]".to_string());
            manifest_lines.push(format!("total = {}", files.len()));
            manifest_lines.push(format!("present = {present}"));
            manifest_lines.push(format!("missing = {missing}"));

            if let Some(parent) = output.parent() {
                fs::create_dir_all(parent).ok();
            }
            fs::write(&output, manifest_lines.join("\n") + "\n").unwrap_or_else(|e| {
                eprintln!("ERROR writing manifest: {e}");
                std::process::exit(1);
            });

            println!("Wrote manifest to {}", output.display());
            println!(
                "  Artifacts: {} total, {} present, {} missing",
                files.len(),
                present,
                missing
            );
        }

        Command::Verify { manifest } => {
            let content = match fs::read_to_string(&manifest) {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("ERROR reading manifest: {e}");
                    std::process::exit(1);
                }
            };

            // Simple TOML parsing for verification
            let parsed: toml::Value = match toml::from_str(&content) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("ERROR parsing manifest: {e}");
                    std::process::exit(1);
                }
            };

            let evidence_dir = parsed
                .get("metadata")
                .and_then(|m| m.get("evidence_dir"))
                .and_then(|v| v.as_str())
                .unwrap_or("data/evidence");

            let artifacts = parsed
                .get("artifact")
                .and_then(|a| a.as_array())
                .cloned()
                .unwrap_or_default();

            let mut ok = 0usize;
            let mut mismatch = 0usize;
            let mut missing = 0usize;

            println!("=== Evidence Manifest Verification ===");
            println!("Manifest: {}", manifest.display());
            println!();

            for artifact in &artifacts {
                let path_str = artifact.get("path").and_then(|v| v.as_str()).unwrap_or("");
                let expected_hash = artifact
                    .get("sha256")
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let status = artifact
                    .get("status")
                    .and_then(|v| v.as_str())
                    .unwrap_or("present");

                if status == "missing" {
                    println!("  SKIP    {} (recorded as missing)", path_str);
                    missing += 1;
                    continue;
                }

                let full_path = PathBuf::from(evidence_dir).join(path_str);
                match sha256_file(&full_path) {
                    Ok(actual_hash) => {
                        if actual_hash == expected_hash {
                            println!("  OK      {}", path_str);
                            ok += 1;
                        } else {
                            println!(
                                "  CHANGED {} (expected {}..., got {}...)",
                                path_str,
                                &expected_hash[..8.min(expected_hash.len())],
                                &actual_hash[..8.min(actual_hash.len())]
                            );
                            mismatch += 1;
                        }
                    }
                    Err(_) => {
                        println!("  MISSING {}", path_str);
                        missing += 1;
                    }
                }
            }

            println!();
            println!(
                "Total: {}, OK: {}, Changed: {}, Missing: {}",
                artifacts.len(),
                ok,
                mismatch,
                missing
            );

            if mismatch > 0 {
                eprintln!("VERIFICATION FAILED: {} artifact(s) changed", mismatch);
                std::process::exit(1);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_temp_dir(suffix: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "evidence_pkg_test_{}_{}",
            std::process::id(),
            suffix,
        ));
        let _ = fs::remove_dir_all(&dir); // clean up from prior runs
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn test_bundle_empty_dir() {
        let dir = make_temp_dir("empty");
        let files = collect_evidence_files(&dir);
        assert!(files.is_empty(), "empty dir should yield no evidence files");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_sha256_known_content() {
        let dir = make_temp_dir("sha256");
        let test_file = dir.join("test.toml");
        let mut f = fs::File::create(&test_file).unwrap();
        writeln!(f, "hello = \"world\"").unwrap();
        drop(f);

        let hash = sha256_file(&test_file).unwrap();
        assert_eq!(hash.len(), 64, "SHA-256 hex should be 64 chars");
        // Verify it's reproducible
        let hash2 = sha256_file(&test_file).unwrap();
        assert_eq!(hash, hash2, "same file should produce same hash");

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_verify_tampered_file() {
        let dir = make_temp_dir("tamper");

        // Create a test artifact
        let artifact = dir.join("data.toml");
        fs::write(&artifact, "value = 42\n").unwrap();

        let original_hash = sha256_file(&artifact).unwrap();

        // Tamper with the file
        fs::write(&artifact, "value = 999\n").unwrap();

        let tampered_hash = sha256_file(&artifact).unwrap();
        assert_ne!(
            original_hash, tampered_hash,
            "tampered file should have different hash"
        );

        let _ = fs::remove_dir_all(&dir);
    }
}
