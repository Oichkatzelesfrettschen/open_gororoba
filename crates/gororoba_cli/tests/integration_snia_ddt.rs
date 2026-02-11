use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn snia_binary_path() -> PathBuf {
    let keys = ["CARGO_BIN_EXE_snia-ddt", "CARGO_BIN_EXE_snia_ddt"];
    for key in keys {
        if let Ok(path) = std::env::var(key) {
            return PathBuf::from(path);
        }
    }

    for (key, value) in std::env::vars() {
        if key.starts_with("CARGO_BIN_EXE_")
            && (key.ends_with("snia-ddt") || key.ends_with("snia_ddt"))
        {
            return PathBuf::from(value);
        }
    }

    panic!("CARGO_BIN_EXE_snia-ddt is not available for integration test");
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "gororoba_cli_{}_{}_{}",
        label,
        std::process::id(),
        timestamp
    ));
    fs::create_dir_all(&dir).expect("create temp output dir");
    dir
}

fn run_snia(args: &[&str]) -> Output {
    Command::new(snia_binary_path())
        .args(args)
        .output()
        .expect("failed to execute snia-ddt")
}

fn assert_success(output: &Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "command failed: status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
}

#[test]
fn snia_help_lists_quick_full_scan_and_examples() {
    let output = run_snia(&["--help"]);
    assert_success(&output);

    let stdout = String::from_utf8(output.stdout).expect("help output should be valid utf-8");
    assert!(
        stdout.contains("--mode <MODE>"),
        "help output must describe mode argument"
    );
    assert!(
        stdout.contains("quick") && stdout.contains("full"),
        "help output must include quick/full modes"
    );
    assert!(
        stdout.contains("--scan"),
        "help output must describe optional scan mode"
    );
    assert!(
        stdout.contains("Examples:") && stdout.contains("snia-ddt --mode quick"),
        "help output must include executable examples"
    );
}

#[test]
fn snia_quick_writes_snapshot_and_summary() {
    let out_dir = unique_temp_dir("snia_quick");
    let out_dir_arg = out_dir.to_string_lossy().into_owned();

    let output = run_snia(&["--mode", "quick", "--output-dir", &out_dir_arg]);
    assert_success(&output);

    let mode_dir = out_dir.join("quick");
    let snapshot_path = mode_dir.join("snia_snapshot.toml");
    let summary_path = mode_dir.join("snia_summary.csv");

    assert!(
        snapshot_path.is_file(),
        "missing snapshot: {:?}",
        snapshot_path
    );
    assert!(
        summary_path.is_file(),
        "missing summary: {:?}",
        summary_path
    );

    let snapshot = fs::read_to_string(&snapshot_path).expect("read snapshot");
    assert!(
        snapshot.contains("nickel56_mass_msun"),
        "snapshot should include nickel56 summary field"
    );

    let summary = fs::read_to_string(&summary_path).expect("read summary");
    assert!(
        summary.contains("mode,scan_enabled"),
        "summary header should include mode and scan columns"
    );
    assert!(
        summary.contains("quick,false"),
        "summary row should include quick mode and scan=false"
    );

    fs::remove_dir_all(&out_dir).expect("remove temp output dir");
}

#[test]
fn snia_full_scan_writes_snapshot_summary_and_scan_csv() {
    let out_dir = unique_temp_dir("snia_full_scan");
    let out_dir_arg = out_dir.to_string_lossy().into_owned();

    let output = run_snia(&["--mode", "full", "--scan", "--output-dir", &out_dir_arg]);
    assert_success(&output);

    let mode_dir = out_dir.join("full");
    let snapshot_path = mode_dir.join("snia_snapshot.toml");
    let summary_path = mode_dir.join("snia_summary.csv");
    let scan_path = mode_dir.join("snia_scan.csv");

    assert!(
        snapshot_path.is_file(),
        "missing snapshot: {:?}",
        snapshot_path
    );
    assert!(
        summary_path.is_file(),
        "missing summary: {:?}",
        summary_path
    );
    assert!(scan_path.is_file(), "missing scan CSV: {:?}", scan_path);

    let summary = fs::read_to_string(&summary_path).expect("read summary");
    assert!(
        summary.contains("full,true"),
        "summary row should include full mode and scan=true"
    );

    let scan = fs::read_to_string(&scan_path).expect("read scan CSV");
    assert!(
        scan.contains("transition_density,turbulence_intensity,ignition_offset_km"),
        "scan CSV should contain expected header"
    );
    assert!(
        scan.contains("objective"),
        "scan CSV should include objective column"
    );

    fs::remove_dir_all(&out_dir).expect("remove temp output dir");
}
