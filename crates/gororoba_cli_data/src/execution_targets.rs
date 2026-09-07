//! Shared execution-target identities for planning and registry validation.

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, fs, path::Path};
use toml::{Table, Value};

/// Resolve workspace tests and benches together with retained external receipts.
pub fn load_workspace_execution_targets(repo_root: &Path) -> Result<BTreeSet<String>> {
    let root_manifest = load_toml(&repo_root.join("Cargo.toml"))?;
    let members = root_manifest
        .get("workspace")
        .and_then(Value::as_table)
        .and_then(|workspace| workspace.get("members"))
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut targets = BTreeSet::new();
    for member in members {
        let Some(member_rel) = member.as_str() else {
            continue;
        };
        let manifest_path = if member_rel.ends_with("Cargo.toml") {
            repo_root.join(member_rel)
        } else {
            repo_root.join(member_rel).join("Cargo.toml")
        };
        if !manifest_path.exists() {
            continue;
        }
        let manifest = load_toml(&manifest_path)?;
        let package = table_value(&manifest, "package");
        let package_name = string_field(&package, "name");
        let package_root = manifest_path.parent().context("package manifest parent")?;
        let explicit_tests = table_array(&manifest, "test")?;
        let mut test_names = BTreeSet::new();
        let mut explicit_names = BTreeSet::new();
        let mut explicit_paths = BTreeSet::new();
        for target in &explicit_tests {
            let name = string_field(target, "name");
            let declared_path = string_field(target, "path");
            let relative_path = if declared_path.is_empty() {
                format!("tests/{name}.rs")
            } else {
                declared_path
            };
            explicit_names.insert(name.clone());
            let target_path = package_root.join(relative_path);
            if let Ok(canonical_path) = target_path.canonicalize() {
                explicit_paths.insert(canonical_path);
            }
            if !name.is_empty() && target_path.is_file() {
                test_names.insert(name);
            }
        }
        if package.get("autotests").and_then(Value::as_bool) != Some(false) {
            let tests_dir = package_root.join("tests");
            if tests_dir.is_dir() {
                for entry in fs::read_dir(tests_dir)? {
                    let path = entry?.path();
                    let name = if path.is_file() && path.extension().is_some_and(|ext| ext == "rs")
                    {
                        path.file_stem()
                    } else if path.is_dir() && path.join("main.rs").is_file() {
                        path.file_name()
                    } else {
                        None
                    };
                    if let Some(name) = name.and_then(|name| name.to_str()) {
                        let source_path = if path.is_dir() {
                            path.join("main.rs")
                        } else {
                            path.clone()
                        };
                        if explicit_names.contains(name)
                            || explicit_paths.contains(&source_path.canonicalize()?)
                        {
                            continue;
                        }
                        test_names.insert(name.to_string());
                    }
                }
            }
        }
        for name in test_names {
            targets.insert(format!("cargo-test:{package_name}:{name}"));
        }
        for bench in table_array(&manifest, "bench")? {
            let name = string_field(&bench, "name");
            if !name.is_empty() {
                targets.insert(name);
            }
        }
    }
    targets.extend(load_external_execution_targets(repo_root)?);
    Ok(targets)
}

/// Validate external target identities and both layers of receipt hashes.
pub fn load_external_execution_targets(repo_root: &Path) -> Result<BTreeSet<String>> {
    let declaration = repo_root.join("plans/external-execution-targets.toml");
    if !declaration.exists() {
        return Ok(BTreeSet::new());
    }
    let mut targets = BTreeSet::new();
    for row in table_array(&load_toml(&declaration)?, "external_target")? {
        let name = string_field(&row, "name");
        let version = string_field(&row, "version");
        let executable = string_field(&row, "executable");
        if name.is_empty()
            || version.is_empty()
            || !Path::new(&executable).is_absolute()
            || name
                .chars()
                .chain(version.chars())
                .any(|character| character.is_whitespace() || character == ':')
        {
            bail!("invalid external execution target identity: {name}");
        }
        let receipt_path = string_field(&row, "receipt");
        let relative = Path::new(&receipt_path);
        if relative.is_absolute()
            || relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!("external target receipt must be repository relative: {name}");
        }
        let bytes = fs::read(repo_root.join(relative))?;
        if sha256_hex(&bytes) != string_field(&row, "receipt_sha256") {
            bail!("external target receipt hash mismatch: {name}");
        }
        let receipt: Table = toml::from_str(std::str::from_utf8(&bytes)?)?;
        for (field, expected) in [
            ("name", &name),
            ("version", &version),
            ("executable", &executable),
        ] {
            if string_field(&receipt, field) != *expected {
                bail!("external target receipt {field} mismatch: {name}");
            }
        }
        let executable_hash = string_field(&receipt, "executable_sha256");
        if executable_hash.len() != 64
            || !executable_hash.bytes().all(|byte| byte.is_ascii_hexdigit())
        {
            bail!("external target receipt lacks executable digest: {name}");
        }
        let source_receipt = string_field(&receipt, "source_receipt");
        let source_relative = Path::new(&source_receipt);
        if source_receipt.is_empty()
            || source_relative.is_absolute()
            || source_relative
                .components()
                .any(|component| !matches!(component, std::path::Component::Normal(_)))
        {
            bail!("external source receipt must be repository relative: {name}");
        }
        if sha256_hex(&fs::read(repo_root.join(source_relative))?)
            != string_field(&receipt, "source_receipt_sha256")
        {
            bail!("external source receipt hash mismatch: {name}");
        }
        if !targets.insert(format!("external:{name}:{version}")) {
            bail!("duplicate external execution target: {name}");
        }
    }
    Ok(targets)
}

/// Return the buildable binary or bench name before dispatcher arguments.
pub fn execution_target_head(target: &str) -> &str {
    target.split_whitespace().next().unwrap_or(target)
}

/// Require exact typed identities while allowing arguments on registered binaries.
pub fn execution_target_registered(target: &str, registered: &BTreeSet<String>) -> bool {
    if target.starts_with("cargo-test:") || target.starts_with("external:") {
        return registered.contains(target);
    }
    registered.contains(target) || registered.contains(execution_target_head(target))
}

fn load_toml(path: &Path) -> Result<Table> {
    let text = fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    // Use toml::from_str rather than .parse::<Value>(): in toml 1.1 the FromStr
    // implementation uses a stricter parser that rejects valid [table] headers.
    let value =
        toml::from_str::<Value>(&text).with_context(|| format!("parse TOML {}", path.display()))?;
    let table = value
        .as_table()
        .cloned()
        .ok_or_else(|| anyhow::anyhow!("root TOML document is not a table: {}", path.display()))?;
    Ok(table)
}

fn table_array(root: &Table, key: &str) -> Result<Vec<Table>> {
    let Some(value) = root.get(key) else {
        return Ok(Vec::new());
    };
    let Some(values) = value.as_array() else {
        bail!("expected array for key {key}");
    };
    Ok(values
        .iter()
        .filter_map(|value| value.as_table().cloned())
        .collect())
}

fn table_value(root: &Table, key: &str) -> Table {
    root.get(key)
        .and_then(Value::as_table)
        .cloned()
        .unwrap_or_default()
}

fn string_field(table: &Table, key: &str) -> String {
    table
        .get(key)
        .map(value_to_string)
        .map(|value| collapse(&value))
        .unwrap_or_default()
}

fn value_to_string(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Integer(v) => v.to_string(),
        Value::Float(v) => v.to_string(),
        Value::Boolean(v) => v.to_string(),
        Value::Datetime(v) => v.to_string(),
        other => other.to_string(),
    }
}

fn ascii_clean(text: &str) -> String {
    let mut out = String::new();
    for ch in text.chars() {
        let code = ch as u32;
        if ch == '\n' || ch == '\r' || ch == '\t' {
            out.push(ch);
        } else if code < 32 {
            out.push(' ');
        } else if code <= 127 {
            out.push(ch);
        } else {
            out.push_str(&format!("\\u{code:04X}"));
        }
    }
    out
}

fn collapse(text: &str) -> String {
    ascii_clean(text)
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    hasher
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}
