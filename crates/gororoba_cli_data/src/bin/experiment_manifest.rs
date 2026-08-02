//! Validate one reproducibility manifest against the checked-out repository.
//!
//! A manifest is evidence only when its code, command, inputs, outputs, and
//! execution environment can be checked together. This binary performs those
//! checks without changing the registry or the referenced files.

use anyhow::{Context, Result, bail};
use chrono::DateTime;
use clap::{Parser, Subcommand};
use rusqlite::{Connection, OpenFlags};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::Command,
};
use toml::Value;

const REPRODUCIBILITY_CLASSES: &[&str] = &[
    "bit_exact",
    "numeric_close",
    "statistical",
    "inferential",
    "external_only",
];
const RANDOMNESS_MODES: &[&str] = &["none", "seeded", "external_entropy"];

#[derive(Parser, Debug)]
#[command(
    name = "experiment-manifest",
    about = "Validate reproducibility metadata and retained file hashes"
)]
struct Cli {
    #[command(subcommand)]
    command: CommandLine,
}

#[derive(Subcommand, Debug)]
enum CommandLine {
    /// Validate one [experiment] table or one [[experiment]] entry.
    Verify {
        /// TOML manifest containing one experiment record.
        manifest: PathBuf,
        /// Repository root used for code, input, and output resolution.
        #[arg(long, default_value = ".")]
        repo_root: PathBuf,
    },
    /// Hash a file using the byte-preserving SHA-256 contract.
    Hash {
        /// File to hash.
        path: PathBuf,
    },
}

fn main() -> Result<()> {
    match Cli::parse().command {
        CommandLine::Verify {
            manifest,
            repo_root,
        } => {
            let repo_root = repo_root
                .canonicalize()
                .context("resolve repository root")?;
            verify_manifest(&manifest, &repo_root)
        }
        CommandLine::Hash { path } => {
            println!("{}  {}", sha256_file(&path)?, path.display());
            Ok(())
        }
    }
}

fn verify_manifest(manifest_path: &Path, repo_root: &Path) -> Result<()> {
    let manifest_text = fs::read_to_string(manifest_path)
        .with_context(|| format!("read manifest {}", manifest_path.display()))?;
    if !manifest_text.is_ascii() {
        bail!(
            "manifest {} contains non-ASCII bytes",
            manifest_path.display()
        );
    }
    let document: Value = toml::from_str(&manifest_text)
        .with_context(|| format!("parse manifest {}", manifest_path.display()))?;
    let experiment = experiment_table(&document)?;
    let artifact_index =
        ArtifactIndex::open(&repo_root.join("registry/canonical/control_plane.sqlite3"))?;

    let experiment_id = required_string(experiment, "id")?;
    if !experiment_id.starts_with("E-") {
        bail!("experiment.id must use the E- prefix: {experiment_id}");
    }
    let title = required_string(experiment, "title")?;
    if title.trim().is_empty() {
        bail!("experiment {experiment_id} has an empty title");
    }

    let reproducibility_class = required_string(experiment, "reproducibility_class")?;
    if !REPRODUCIBILITY_CLASSES.contains(&reproducibility_class) {
        bail!(
            "experiment {experiment_id} uses unsupported reproducibility_class {reproducibility_class:?}"
        );
    }

    let code_commit_sha = required_string(experiment, "code_commit_sha")?;
    validate_lowercase_hex(code_commit_sha, 40, "experiment.code_commit_sha")?;
    let checked_out_commit = checked_out_commit(repo_root)?;
    if code_commit_sha != checked_out_commit {
        bail!(
            "experiment {experiment_id} targets commit {code_commit_sha}, but the checkout is {checked_out_commit}"
        );
    }

    validate_timestamp(experiment, experiment_id)?;
    required_string(experiment, "toolchain")?;
    required_string(experiment, "hardware")?;
    validate_features(experiment, experiment_id)?;
    validate_randomness(experiment, experiment_id)?;
    validate_command_hash(experiment, experiment_id)?;
    validate_inputs(experiment, experiment_id, repo_root)?;
    validate_outputs(experiment, experiment_id, repo_root, &artifact_index)?;
    validate_numeric_tolerances(experiment, experiment_id, reproducibility_class)?;

    println!("verified {experiment_id}: {title}");
    Ok(())
}

fn experiment_table(document: &Value) -> Result<&toml::map::Map<String, Value>> {
    let value = document
        .get("experiment")
        .context("manifest must contain an experiment table")?;
    match value {
        Value::Table(table) => Ok(table),
        Value::Array(rows) if rows.len() == 1 => rows[0]
            .as_table()
            .context("the single experiment array entry must be a table"),
        Value::Array(_) => bail!("manifest must contain exactly one experiment entry"),
        _ => bail!("manifest experiment must be a table or a one-entry array"),
    }
}

fn required_string<'a>(table: &'a toml::map::Map<String, Value>, key: &str) -> Result<&'a str> {
    table
        .get(key)
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("experiment.{key} must be a non-empty string"))
}

fn validate_timestamp(table: &toml::map::Map<String, Value>, experiment_id: &str) -> Result<()> {
    let actual = table.get("actual_timestamp");
    let inferred = table.get("inferred_timestamp");
    if actual.is_some() == inferred.is_some() {
        bail!(
            "experiment {experiment_id} must contain exactly one of actual_timestamp or inferred_timestamp"
        );
    }
    let timestamp = actual
        .or(inferred)
        .and_then(Value::as_str)
        .with_context(|| {
            format!("experiment {experiment_id} timestamp must be an RFC3339 string")
        })?;
    DateTime::parse_from_rfc3339(timestamp).with_context(|| {
        format!("experiment {experiment_id} timestamp is not RFC3339: {timestamp}")
    })?;
    Ok(())
}

fn validate_features(table: &toml::map::Map<String, Value>, experiment_id: &str) -> Result<()> {
    let features = table
        .get("features")
        .and_then(Value::as_array)
        .with_context(|| format!("experiment {experiment_id} requires a features array"))?;
    for feature in features {
        let feature_name = feature
            .as_str()
            .with_context(|| format!("experiment {experiment_id} features must contain strings"))?;
        if feature_name.trim().is_empty() {
            bail!("experiment {experiment_id} contains an empty feature name");
        }
    }
    Ok(())
}

fn validate_randomness(table: &toml::map::Map<String, Value>, experiment_id: &str) -> Result<()> {
    let mode = required_string(table, "randomness_mode")?;
    if !RANDOMNESS_MODES.contains(&mode) {
        bail!("experiment {experiment_id} uses unsupported randomness_mode {mode:?}");
    }
    let seed = table.get("random_seed");
    let generator = table.get("random_generator");
    match mode {
        "none" => {
            if seed.is_some() || generator.is_some() {
                bail!(
                    "experiment {experiment_id} cannot record a random seed or generator with randomness_mode=none"
                );
            }
        }
        "seeded" => {
            seed.context(format!("experiment {experiment_id} requires random_seed"))?;
            required_string(table, "random_generator")?;
        }
        "external_entropy" => {
            required_string(table, "random_generator")?;
            required_string(table, "randomness_source")?;
            if seed.is_some() {
                bail!(
                    "experiment {experiment_id} cannot claim external_entropy and record random_seed"
                );
            }
        }
        _ => unreachable!("randomness mode validated above"),
    }
    Ok(())
}

fn validate_command_hash(table: &toml::map::Map<String, Value>, experiment_id: &str) -> Result<()> {
    let command = required_string(table, "run_command")?;
    let declared_hash = required_string(table, "run_command_sha256")?;
    validate_lowercase_hex(declared_hash, 64, "experiment.run_command_sha256")?;
    let actual_hash = sha256_bytes(command.as_bytes());
    if declared_hash != actual_hash {
        bail!(
            "experiment {experiment_id} run_command_sha256 mismatch: declared {declared_hash}, computed {actual_hash}"
        );
    }
    Ok(())
}

fn validate_inputs(
    table: &toml::map::Map<String, Value>,
    experiment_id: &str,
    repo_root: &Path,
) -> Result<()> {
    let inputs = table
        .get("input_hashes")
        .and_then(Value::as_array)
        .with_context(|| format!("experiment {experiment_id} requires input_hashes"))?;
    if inputs.is_empty() {
        bail!("experiment {experiment_id} requires at least one input hash");
    }
    for (index, input) in inputs.iter().enumerate() {
        let row = input.as_table().with_context(|| {
            format!("experiment {experiment_id} input_hashes[{index}] must be a table")
        })?;
        let path = row.get("path").and_then(Value::as_str).with_context(|| {
            format!("experiment {experiment_id} input_hashes[{index}] needs path")
        })?;
        let declared_hash = row.get("sha256").and_then(Value::as_str).with_context(|| {
            format!("experiment {experiment_id} input_hashes[{index}] needs sha256")
        })?;
        verify_file_hash(repo_root, path, declared_hash, "input", experiment_id)?;
    }
    Ok(())
}

fn validate_outputs(
    table: &toml::map::Map<String, Value>,
    experiment_id: &str,
    repo_root: &Path,
    artifact_index: &ArtifactIndex,
) -> Result<()> {
    let outputs = table
        .get("output_hash_refs")
        .and_then(Value::as_array)
        .with_context(|| format!("experiment {experiment_id} requires output_hash_refs"))?;
    if outputs.is_empty() {
        bail!("experiment {experiment_id} requires at least one output hash");
    }
    for (index, output) in outputs.iter().enumerate() {
        let row = output.as_table().with_context(|| {
            format!("experiment {experiment_id} output_hash_refs[{index}] must be a table")
        })?;
        let artifact_id = row
            .get("artifact_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| {
                format!("experiment {experiment_id} output_hash_refs[{index}] needs artifact_id")
            })?;
        let path = row.get("path").and_then(Value::as_str).with_context(|| {
            format!("experiment {experiment_id} output_hash_refs[{index}] needs path")
        })?;
        let declared_hash = row.get("sha256").and_then(Value::as_str).with_context(|| {
            format!("experiment {experiment_id} output_hash_refs[{index}] needs sha256")
        })?;
        artifact_index.verify_registered_path(artifact_id, path, experiment_id)?;
        verify_file_hash(repo_root, path, declared_hash, "output", experiment_id)?;
    }
    Ok(())
}

fn validate_numeric_tolerances(
    table: &toml::map::Map<String, Value>,
    experiment_id: &str,
    reproducibility_class: &str,
) -> Result<()> {
    let tolerances = table.get("numeric_tolerance");
    if reproducibility_class != "numeric_close" {
        return Ok(());
    }
    let tolerances = tolerances.and_then(Value::as_array).with_context(|| {
        format!("experiment {experiment_id} numeric_close needs numeric_tolerance")
    })?;
    if tolerances.is_empty() {
        bail!("experiment {experiment_id} numeric_close needs at least one tolerance row");
    }
    for (index, tolerance) in tolerances.iter().enumerate() {
        let row = tolerance.as_table().with_context(|| {
            format!("experiment {experiment_id} numeric_tolerance[{index}] must be a table")
        })?;
        let output_id = row
            .get("output_id")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| {
                format!("experiment {experiment_id} numeric_tolerance[{index}] needs output_id")
            })?;
        let column = row
            .get("column")
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())
            .with_context(|| {
                format!("experiment {experiment_id} numeric_tolerance[{index}] needs column")
            })?;
        let relative = nonnegative_number(row, "rel_tol", experiment_id, index)?;
        let absolute = nonnegative_number(row, "abs_tol", experiment_id, index)?;
        if !relative.is_finite() || !absolute.is_finite() {
            bail!(
                "experiment {experiment_id} numeric_tolerance[{index}] has a non-finite tolerance for {output_id}.{column}"
            );
        }
    }
    Ok(())
}

fn nonnegative_number(
    table: &toml::map::Map<String, Value>,
    key: &str,
    experiment_id: &str,
    index: usize,
) -> Result<f64> {
    let value = table.get(key).with_context(|| {
        format!("experiment {experiment_id} numeric_tolerance[{index}] needs {key}")
    })?;
    let number = match value {
        Value::Integer(value) => *value as f64,
        Value::Float(value) => *value,
        _ => bail!("experiment {experiment_id} numeric_tolerance[{index}] {key} must be numeric"),
    };
    if number < 0.0 {
        bail!("experiment {experiment_id} numeric_tolerance[{index}] {key} is negative");
    }
    Ok(number)
}

fn verify_file_hash(
    repo_root: &Path,
    relative_path: &str,
    declared_hash: &str,
    kind: &str,
    experiment_id: &str,
) -> Result<()> {
    validate_lowercase_hex(declared_hash, 64, &format!("{kind} hash"))?;
    let path = resolve_repo_file(repo_root, relative_path)
        .with_context(|| format!("experiment {experiment_id} {kind} path {relative_path:?}"))?;
    let actual_hash = sha256_file(&path)?;
    if actual_hash != declared_hash {
        bail!(
            "experiment {experiment_id} {kind} hash mismatch for {relative_path}: declared {declared_hash}, computed {actual_hash}"
        );
    }
    Ok(())
}

fn resolve_repo_file(repo_root: &Path, relative_path: &str) -> Result<PathBuf> {
    let relative = Path::new(relative_path);
    if relative.is_absolute() {
        bail!("path must be repository-relative: {relative_path}");
    }
    let candidate = repo_root.join(relative);
    let canonical = candidate
        .canonicalize()
        .with_context(|| format!("resolve repository-relative path {relative_path}"))?;
    if !canonical.starts_with(repo_root) {
        bail!("path escapes repository root: {relative_path}");
    }
    if !canonical.is_file() {
        bail!("path is not a regular file: {relative_path}");
    }
    Ok(canonical)
}

struct ArtifactIndex {
    connection: Connection,
}

impl ArtifactIndex {
    fn open(database_path: &Path) -> Result<Self> {
        let connection =
            Connection::open_with_flags(database_path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                .with_context(|| {
                    format!(
                        "open artifact registry {} read-only",
                        database_path.display()
                    )
                })?;
        Ok(Self { connection })
    }

    fn verify_registered_path(
        &self,
        artifact_id: &str,
        path: &str,
        experiment_id: &str,
    ) -> Result<()> {
        let mut statement = self.connection.prepare(
            "SELECT path FROM artifact_paths WHERE artifact_id = ?1 AND relation = 'downloaded'
             UNION
             SELECT canonical_download_path FROM artifacts
             WHERE id = ?1 AND canonical_download_path IS NOT NULL
               AND canonical_download_path != ''",
        )?;
        let registered_paths = statement
            .query_map([artifact_id], |row| row.get::<_, String>(0))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if registered_paths.is_empty() {
            bail!(
                "experiment {experiment_id} output artifact {artifact_id} is absent from the canonical artifact registry"
            );
        }
        if !registered_paths
            .iter()
            .any(|registered_path| registered_path == path)
        {
            bail!(
                "experiment {experiment_id} output path {path} is not registered for artifact {artifact_id}"
            );
        }
        Ok(())
    }
}

fn checked_out_commit(repo_root: &Path) -> Result<String> {
    let repo_root_string = repo_root.to_string_lossy();
    let output = Command::new("git")
        .args(["-C", repo_root_string.as_ref(), "rev-parse", "HEAD"])
        .output()
        .context("run git rev-parse HEAD")?;
    if !output.status.success() {
        bail!("git rev-parse HEAD failed");
    }
    let commit = String::from_utf8(output.stdout).context("decode git commit")?;
    let commit = commit.trim();
    validate_lowercase_hex(commit, 40, "checked-out commit")?;
    Ok(commit.to_string())
}

fn validate_lowercase_hex(value: &str, length: usize, field: &str) -> Result<()> {
    if value.len() != length
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("{field} must be {length} lowercase hexadecimal characters");
    }
    Ok(())
}

fn sha256_file(path: &Path) -> Result<String> {
    let mut file = fs::File::open(path).with_context(|| format!("open {}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .with_context(|| format!("read {}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex_digest(hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hex_digest(hasher.finalize())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    bytes
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::{
        checked_out_commit, resolve_repo_file, sha256_bytes, sha256_file, validate_lowercase_hex,
        verify_manifest,
    };
    use std::{fs, io::Write, path::Path};
    use tempfile::tempdir;

    #[test]
    fn hashes_bytes_with_lowercase_hex() {
        assert_eq!(
            sha256_bytes(b"open_gororoba"),
            "d38cc65aee185e6f8a9330a1ae906503dc6eaa12b500581ca02e76d5c151ba1a"
        );
    }

    #[test]
    fn rejects_uppercase_or_short_hashes() {
        assert!(validate_lowercase_hex(&"a".repeat(64), 64, "hash").is_ok());
        assert!(validate_lowercase_hex(&"A".repeat(64), 64, "hash").is_err());
        assert!(validate_lowercase_hex("abc", 64, "hash").is_err());
    }

    #[test]
    fn resolves_only_files_under_the_repository_root() {
        let root = tempdir().expect("tempdir");
        let file_path = root.path().join("input.txt");
        fs::write(&file_path, b"data").expect("write input");
        assert!(resolve_repo_file(root.path(), "input.txt").is_ok());
        assert!(resolve_repo_file(root.path(), "missing.txt").is_err());
        assert!(resolve_repo_file(root.path(), "/etc/passwd").is_err());
    }

    #[test]
    fn verifies_complete_manifest_contract_and_rejects_missing_hardware() {
        let repo_root = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .canonicalize()
            .expect("repository root");
        let retained_path = "data/csv/apt_dimensional_census_summary.csv";
        let retained_hash = sha256_file(&repo_root.join(retained_path)).expect("retained hash");
        let commit = checked_out_commit(&repo_root).expect("checked out commit");
        let command = "cargo test -p materials_core";
        let manifest_text = format!(
            "[experiment]\n\
id = \"E-test-manifest\"\n\
title = \"Manifest verifier unit test\"\n\
reproducibility_class = \"bit_exact\"\n\
code_commit_sha = \"{commit}\"\n\
actual_timestamp = \"2026-08-01T00:00:00Z\"\n\
run_command = \"{command}\"\n\
run_command_sha256 = \"{}\"\n\
toolchain = \"rustc 1.97.0\"\n\
features = []\n\
hardware = \"unit-test host\"\n\
randomness_mode = \"none\"\n\n\
[[experiment.input_hashes]]\n\
path = \"{retained_path}\"\n\
sha256 = \"{retained_hash}\"\n\n\
[[experiment.output_hash_refs]]\n\
artifact_id = \"ASOT-0984\"\n\
path = \"{retained_path}\"\n\
sha256 = \"{retained_hash}\"\n",
            sha256_bytes(command.as_bytes())
        );
        let mut manifest = tempfile::NamedTempFile::new().expect("manifest temp file");
        manifest
            .write_all(manifest_text.as_bytes())
            .expect("write manifest");
        verify_manifest(manifest.path(), &repo_root).expect("complete manifest verifies");

        let incomplete = manifest_text.replace("hardware = \"unit-test host\"\n", "");
        let mut incomplete_manifest = tempfile::NamedTempFile::new().expect("incomplete temp file");
        incomplete_manifest
            .write_all(incomplete.as_bytes())
            .expect("write incomplete manifest");
        assert!(verify_manifest(incomplete_manifest.path(), &repo_root).is_err());
    }
}
