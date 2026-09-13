// SPDX-License-Identifier: GPL-2.0-or-later

use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;
use std::process::{Command, Output};

fn write_workspace(root: &Path, members: &[(&str, &str)]) {
    let member_lines = members
        .iter()
        .map(|(member, _)| format!("    \"{member}\","))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(
        root.join("Cargo.toml"),
        format!("[workspace]\nmembers = [\n{member_lines}\n]\n"),
    )
    .unwrap();
    for (member, package_name) in members {
        let member_dir = root.join(member);
        std::fs::create_dir_all(&member_dir).unwrap();
        std::fs::write(
            member_dir.join("Cargo.toml"),
            format!("[package]\nname = \"{package_name}\"\nversion = \"0.1.0\"\n"),
        )
        .unwrap();
    }
}

fn run_sharder(root: &Path, rust_scope: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("ci-rust-shard-matrix")
        .arg("--workspace-manifest")
        .arg(root.join("Cargo.toml"))
        .arg(format!("--rust-scope={rust_scope}"))
        .arg("--clippy-scope=-p alpha")
        .arg("--light-shard-count=3")
        .arg("--heavy-package=heavy")
        .output()
        .unwrap()
}

fn run_target_sharder(root: &Path, package_names: &[&str]) -> Output {
    let mut sorted_package_names = package_names.to_vec();
    sorted_package_names.sort_unstable();
    let package_scope = sorted_package_names
        .iter()
        .map(|package_name| format!("-p {package_name}"))
        .collect::<Vec<_>>()
        .join(" ");
    let mut command = Command::new(env!("CARGO_BIN_EXE_xtask"));
    command
        .arg("ci-rust-shard-matrix")
        .arg("--workspace-manifest")
        .arg(root.join("Cargo.toml"))
        .arg(format!("--rust-scope={package_scope}"))
        .arg(format!("--clippy-scope={package_scope}"))
        .arg("--light-shard-count=3")
        .arg("--target-shard-count=3");
    for package_name in package_names {
        command.arg(format!("--target-shard-package={package_name}"));
    }
    command.output().unwrap()
}

fn packages_from_scope(scope: &str) -> Vec<&str> {
    let tokens: Vec<&str> = scope.split_whitespace().collect();
    assert_eq!(tokens.len() % 2, 0, "scope has incomplete package pairs");
    tokens
        .chunks_exact(2)
        .map(|pair| {
            assert_eq!(pair[0], "-p");
            pair[1]
        })
        .collect()
}

#[test]
fn workspace_scope_partitions_into_one_exact_package_set() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(
        temp.path(),
        &[
            ("crates/alpha", "alpha"),
            ("crates/beta", "beta"),
            ("crates/gamma", "gamma"),
            ("crates/delta", "delta"),
            ("crates/heavy", "heavy"),
        ],
    );
    let first = run_sharder(temp.path(), "--workspace");
    let second = run_sharder(temp.path(), "--workspace");
    assert!(first.status.success(), "{}", String::from_utf8_lossy(&first.stderr));
    assert_eq!(first.stdout, second.stdout, "shard assignment must be deterministic");

    let matrix: Value = serde_json::from_slice(&first.stdout).unwrap();
    let entries = matrix["include"].as_array().unwrap();
    let mut observed = BTreeSet::new();
    for entry in entries {
        if entry["target"] == "clippy" {
            continue;
        }
        for package_name in packages_from_scope(entry["rust_scope"].as_str().unwrap()) {
            assert!(
                observed.insert(package_name.to_string()),
                "duplicate package across shards: {package_name}"
            );
        }
    }
    assert_eq!(
        observed,
        ["alpha", "beta", "delta", "gamma", "heavy"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
}

#[test]
fn target_shards_cover_every_declared_binary_exactly_once() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(temp.path(), &[("crates/physics", "physics")]);
    std::fs::write(
        temp.path().join("crates/physics/Cargo.toml"),
        "[package]\nname = \"physics\"\nversion = \"0.1.0\"\n\n[[bin]]\nname = \"alpha-bin\"\npath = \"src/bin/alpha.rs\"\n\n[[bin]]\nname = \"beta-bin\"\npath = \"src/bin/beta.rs\"\nrequired-features = [\"gpu\"]\n\n[[bin]]\nname = \"gamma-bin\"\npath = \"src/bin/gamma.rs\"\n",
    )
    .unwrap();
    let output = run_target_sharder(temp.path(), &["physics"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let matrix: Value = serde_json::from_slice(&output.stdout).unwrap();
    let entries = matrix["include"].as_array().unwrap();
    let mut observed = BTreeSet::new();
    let mut non_binary_shards = 0;
    let mut gpu_shards = 0;
    for entry in entries {
        let target_args = entry["cargo_target_args"].as_str().unwrap();
        if target_args == "--lib --tests" {
            non_binary_shards += 1;
        } else if target_args.starts_with("--bin ") {
            let features = entry["cargo_features"].as_str().unwrap();
            let tokens: Vec<&str> = target_args.split_whitespace().collect();
            for pair in tokens.chunks_exact(2) {
                assert_eq!(pair[0], "--bin");
                if pair[1] == "beta-bin" {
                    assert_eq!(features, "gpu");
                    gpu_shards += 1;
                } else {
                    assert_eq!(features, "");
                }
                assert!(observed.insert(pair[1].to_string()));
            }
        }
    }
    assert_eq!(non_binary_shards, 1);
    assert_eq!(gpu_shards, 1);
    assert_eq!(
        observed,
        ["alpha-bin", "beta-bin", "gamma-bin"]
            .into_iter()
            .map(str::to_string)
            .collect()
    );
}

#[test]
fn multiple_target_shard_packages_cover_each_package_and_binary_once() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(
        temp.path(),
        &[("crates/physics", "physics"), ("crates/data", "data")],
    );
    for (member, package_name) in [("physics", "physics"), ("data", "data")] {
        std::fs::write(
            temp.path().join(format!("crates/{member}/Cargo.toml")),
            format!(
                "[package]\nname = \"{package_name}\"\nversion = \"0.1.0\"\n\n[[bin]]\nname = \"{package_name}-alpha\"\npath = \"src/bin/alpha.rs\"\n\n[[bin]]\nname = \"{package_name}-beta\"\npath = \"src/bin/beta.rs\"\n"
            ),
        )
        .unwrap();
    }

    let output = run_target_sharder(temp.path(), &["physics", "data"]);
    let reversed_output = run_target_sharder(temp.path(), &["data", "physics"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert_eq!(
        output.stdout, reversed_output.stdout,
        "target package option order must not affect the matrix"
    );
    let matrix: Value = serde_json::from_slice(&output.stdout).unwrap();
    let entries = matrix["include"].as_array().unwrap();
    let mut observed_non_binary = BTreeSet::new();
    let mut observed_binaries = BTreeSet::new();
    for entry in entries {
        let package_name = packages_from_scope(entry["rust_scope"].as_str().unwrap());
        if entry["target"] == "clippy" {
            continue;
        }
        assert_eq!(package_name.len(), 1);
        let package_name = package_name[0];
        let target_args = entry["cargo_target_args"].as_str().unwrap();
        if target_args == "--lib --tests" {
            assert!(observed_non_binary.insert(package_name.to_string()));
            continue;
        }
        let target_tokens: Vec<&str> = target_args.split_whitespace().collect();
        for pair in target_tokens.chunks_exact(2) {
            assert_eq!(pair[0], "--bin");
            assert!(observed_binaries.insert((package_name.to_string(), pair[1].to_string())));
        }
    }
    assert_eq!(
        observed_non_binary,
        ["data", "physics"].into_iter().map(str::to_string).collect()
    );
    assert_eq!(
        observed_binaries,
        [
            ("data", "data-alpha"),
            ("data", "data-beta"),
            ("physics", "physics-alpha"),
            ("physics", "physics-beta"),
        ]
        .into_iter()
        .map(|(package, binary)| (package.to_string(), binary.to_string()))
        .collect()
    );
}

#[test]
fn duplicate_target_shard_package_mutation_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(temp.path(), &[("crates/physics", "physics")]);
    std::fs::write(
        temp.path().join("crates/physics/Cargo.toml"),
        "[package]\nname = \"physics\"\nversion = \"0.1.0\"\n\n[[bin]]\nname = \"physics-bin\"\npath = \"src/bin/physics.rs\"\n",
    )
    .unwrap();

    let output = run_target_sharder(temp.path(), &["physics", "physics"]);
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("target-shard package is named more than once"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn heavy_target_shard_overlap_mutation_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(temp.path(), &[("crates/physics", "physics")]);
    let output = Command::new(env!("CARGO_BIN_EXE_xtask"))
        .arg("ci-rust-shard-matrix")
        .arg("--workspace-manifest")
        .arg(temp.path().join("Cargo.toml"))
        .arg("--rust-scope=-p physics")
        .arg("--clippy-scope=-p physics")
        .arg("--heavy-package=physics")
        .arg("--target-shard-package=physics")
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("package cannot be both heavy and target-sharded"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn duplicate_package_name_mutation_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(
        temp.path(),
        &[("crates/alpha", "same"), ("crates/beta", "same")],
    );
    let output = run_sharder(temp.path(), "--workspace");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("duplicate workspace package name"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}

#[test]
fn missing_member_manifest_mutation_is_rejected() {
    let temp = tempfile::tempdir().unwrap();
    write_workspace(temp.path(), &[("crates/alpha", "alpha")]);
    std::fs::write(
        temp.path().join("Cargo.toml"),
        "[workspace]\nmembers = [\"crates/alpha\", \"crates/missing\"]\n",
    )
    .unwrap();
    let output = run_sharder(temp.path(), "--workspace");
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr).contains("has no readable manifest"),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
}
