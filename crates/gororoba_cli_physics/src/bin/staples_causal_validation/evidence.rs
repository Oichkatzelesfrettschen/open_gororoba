//! Source-bound atomic records and explicit retained execution failures.

use anyhow::{Context, Result, ensure};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

const SOURCES: &[(&str, &[u8])] = &[
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/geometric_capacity.rs",
        include_bytes!("geometric_capacity.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/control_uncertainty.rs",
        include_bytes!("control_uncertainty.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation.rs",
        include_bytes!("../staples_causal_validation.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/admission.rs",
        include_bytes!("admission.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/features.rs",
        include_bytes!("features.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/splits.rs",
        include_bytes!("splits.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/fitting.rs",
        include_bytes!("fitting.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/metrics.rs",
        include_bytes!("metrics.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/bin/staples_causal_validation/evidence.rs",
        include_bytes!("evidence.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_associator.rs",
        include_bytes!("../../staple_associator.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_controls.rs",
        include_bytes!("../../staple_controls.rs"),
    ),
    (
        "crates/gororoba_cli_physics/src/staple_logistic.rs",
        include_bytes!("../../staple_logistic.rs"),
    ),
    (
        "crates/gororoba_cli_physics/Cargo.toml",
        include_bytes!("../../../Cargo.toml"),
    ),
    (
        "crates/cd_kernel/src/mult_table.rs",
        include_bytes!("../../../../cd_kernel/src/mult_table.rs"),
    ),
    (
        "crates/cd_kernel/src/cayley_dickson/mod.rs",
        include_bytes!("../../../../cd_kernel/src/cayley_dickson/mod.rs"),
    ),
    (
        "crates/cd_kernel/src/cayley_dickson/arith.rs",
        include_bytes!("../../../../cd_kernel/src/cayley_dickson/arith.rs"),
    ),
    (
        "crates/cd_kernel/src/lib.rs",
        include_bytes!("../../../../cd_kernel/src/lib.rs"),
    ),
    (
        "crates/cd_kernel/Cargo.toml",
        include_bytes!("../../../../cd_kernel/Cargo.toml"),
    ),
    ("Cargo.toml", include_bytes!("../../../../../Cargo.toml")),
    ("Cargo.lock", include_bytes!("../../../../../Cargo.lock")),
    (
        "rust-toolchain.toml",
        include_bytes!("../../../../../rust-toolchain.toml"),
    ),
    (
        ".cargo/config.toml",
        include_bytes!("../../../../../.cargo/config.toml"),
    ),
];

pub(super) fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

pub(super) fn digest(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

pub(super) fn hash_file(path: &Path) -> Result<String> {
    let mut file = File::open(path)?;
    let mut hash = Sha256::new();
    let mut buffer = [0_u8; 65536];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        hash.update(&buffer[..count]);
    }
    Ok(hex(&hash.finalize()))
}

pub(super) fn source_identity() -> Result<Value> {
    let current = std::env::current_dir()?;
    let root = current
        .ancestors()
        .find(|path| {
            path.join("crates/gororoba_cli_physics/src/bin/staples_causal_validation.rs")
                .is_file()
        })
        .context("launch from source checkout")?;
    let mut sources = serde_json::Map::new();
    for &(path, compiled) in SOURCES {
        let observed = hash_file(&root.join(path))?;
        ensure!(
            observed == digest(compiled),
            "binary source differs from {path}; rebuild before inference"
        );
        sources.insert(path.to_owned(), json!(observed));
    }
    Ok(Value::Object(sources))
}

pub(super) fn atomic_json(path: &Path, payload: &Value) -> Result<()> {
    ensure!(
        !path.exists(),
        "refusing output overwrite: {}",
        path.display()
    );
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> Result<()> {
        serde_json::to_writer(&mut file, payload)?;
        file.write_all(b"\n")?;
        file.sync_all()?;
        fs::hard_link(&temporary, path)?;
        File::open(path.parent().context("output has no parent")?)?.sync_all()?;
        Ok(())
    })();
    let cleanup = fs::remove_file(&temporary);
    result?;
    cleanup?;
    Ok(())
}

pub(super) fn preserve(path: &Path, payload: &Value) -> Result<()> {
    if path.exists() {
        let observed: Value = serde_json::from_reader(File::open(path)?)?;
        ensure!(
            observed == *payload,
            "existing evidence differs: {}",
            path.display()
        );
        Ok(())
    } else {
        atomic_json(path, payload)
    }
}

pub(super) fn read_record(path: &Path, identity: &str) -> Result<Value> {
    let record: Value = serde_json::from_reader(File::open(path)?)?;
    ensure!(
        record["identity"] == identity
            && record["payload_sha256"] == digest(&serde_json::to_vec(&record["payload"])?),
        "stale identity or corrupt payload: {}",
        path.display()
    );
    Ok(record["payload"].clone())
}

pub(super) fn record(
    path: &Path,
    identity: &str,
    sources: &Value,
    execute: impl FnOnce() -> Result<Value>,
) -> Result<Value> {
    ensure!(
        source_identity()? == *sources,
        "source changed before record"
    );
    if path.exists() {
        return read_record(path, identity);
    }
    let payload = execute()?;
    ensure!(
        source_identity()? == *sources,
        "source changed during record"
    );
    atomic_json(
        path,
        &json!({"identity":identity,"payload_sha256":digest(&serde_json::to_vec(&payload)?),"payload":payload}),
    )?;
    Ok(payload)
}

pub(super) fn exact_records(directory: &Path, expected: &BTreeSet<String>) -> Result<()> {
    let observed: BTreeSet<String> = fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name().to_string_lossy().into_owned()))
        .collect::<std::io::Result<Vec<_>>>()?
        .into_iter()
        .filter(|name| name.ends_with(".json") && name != "summary.json")
        .collect();
    ensure!(
        &observed == expected,
        "record set mismatch; missing={:?} extra={:?}",
        expected.difference(&observed).collect::<Vec<_>>(),
        observed.difference(expected).collect::<Vec<_>>()
    );
    Ok(())
}

pub(super) struct OutputLock(PathBuf);
impl OutputLock {
    pub(super) fn acquire(directory: &Path) -> Result<Self> {
        fs::create_dir_all(directory)?;
        let path = directory.join(".runner.lock");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .context("output is locked; inspect process before explicit recovery")?;
        writeln!(file, "{}", std::process::id())?;
        file.sync_all()?;
        Ok(Self(path))
    }
}
impl Drop for OutputLock {
    fn drop(&mut self) {
        if let Err(error) = fs::remove_file(&self.0) {
            eprintln!("output lock cleanup failed: {error}");
        }
    }
}

pub(super) fn retain_failure(
    directory: &Path,
    context: &Value,
    error: anyhow::Error,
) -> anyhow::Error {
    let value =
        json!({"status":"rejected_execution","context":context,"error":format!("{error:#}")});
    let hash = digest(&serde_json::to_vec(&value).unwrap_or_default());
    let path = directory.join(format!("failure-{hash}.json"));
    match preserve(&path, &value) {
        Ok(()) => error.context(format!("failure retained at {}", path.display())),
        Err(retention) => error.context(format!("failure retention also failed: {retention:#}")),
    }
}

pub(super) fn preserve_bytes(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.exists() {
        ensure!(
            fs::metadata(path)?.len() == bytes.len() as u64 && hash_file(path)? == digest(bytes),
            "retained support bytes differ: {}",
            path.display()
        );
        return Ok(());
    }
    let temporary = path.with_extension(format!("{}.tmp", std::process::id()));
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    let result = (|| -> Result<()> {
        file.write_all(bytes)?;
        file.sync_all()?;
        fs::hard_link(&temporary, path)?;
        File::open(path.parent().context("support path lacks parent")?)?.sync_all()?;
        Ok(())
    })();
    let cleanup = fs::remove_file(temporary);
    result?;
    cleanup?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn resume_rejects_identity_drift_payload_mutation_and_overwrite() {
        let directory =
            std::env::temp_dir().join(format!("causal-evidence-test-{}", std::process::id()));
        fs::create_dir(&directory).unwrap();
        let path = directory.join("record.json");
        let payload = json!({"value":2.4641099530828617e-9});
        let record = json!({"identity":"sealed","payload_sha256":digest(&serde_json::to_vec(&payload).unwrap()),"payload":payload});
        atomic_json(&path, &record).unwrap();
        assert_eq!(read_record(&path, "sealed").unwrap(), payload);
        assert!(read_record(&path, "other").is_err());
        assert!(atomic_json(&path, &record).is_err());
        let mut corrupt = record;
        corrupt["payload"]["value"] = json!(0.0);
        fs::write(&path, serde_json::to_vec(&corrupt).unwrap()).unwrap();
        assert!(read_record(&path, "sealed").is_err());
        fs::remove_file(path).unwrap();
        fs::remove_dir(directory).unwrap();
    }
}
