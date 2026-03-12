use materials_core::{PathionToyLayer, default_c053_layers, write_c053_summary};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

const COMMITTED_CSV: &str = "data/csv/c053_pathion_tmm_summary.csv";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn c053_binary_path() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_c053-pathion-metamaterial-mapping") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_c053-pathion-metamaterial-mapping") {
        return PathBuf::from(path);
    }
    panic!("CARGO_BIN_EXE_c053-pathion-metamaterial-mapping is not available");
}

fn unique_temp_path(label: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock is before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "gororoba_cli_physics_{}_{}_{}",
        label,
        std::process::id(),
        timestamp
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir.join("c053_pathion_toy.csv")
}

fn read_rows(path: &std::path::Path) -> Vec<PathionToyLayer> {
    let mut reader = csv::Reader::from_path(path).expect("open CSV");
    reader
        .deserialize()
        .collect::<Result<Vec<PathionToyLayer>, _>>()
        .expect("deserialize CSV rows")
}

fn run_c053(args: &[&str]) -> Output {
    Command::new(c053_binary_path())
        .args(args)
        .output()
        .expect("failed to execute c053-pathion-metamaterial-mapping")
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
fn rust_generator_writes_expected_uniform_csv() {
    let out_path = unique_temp_path("c053_write");
    write_c053_summary(&out_path).expect("write summary");
    let rows = read_rows(&out_path);
    assert_eq!(rows.len(), 8);
    assert!(rows.iter().all(|row| row.diag_abs_mean == "1.000000"));
    assert!(rows.iter().all(|row| row.n_real == "1.750000"));
    assert!(rows.iter().all(|row| row.thickness_nm == "40.000000"));
    assert!(
        rows.iter()
            .all(|row| row.degeneracy_note == "diagonal-only toy map: uniform layer response")
    );
    fs::remove_file(&out_path).expect("remove temp CSV");
    fs::remove_dir_all(out_path.parent().expect("temp parent")).expect("remove temp dir");
}

#[test]
fn committed_csv_matches_rust_generator() {
    let committed = read_rows(&repo_root().join(COMMITTED_CSV));
    let regenerated = default_c053_layers().expect("default rows");
    assert_eq!(committed, regenerated);
}

#[test]
fn rust_cli_exits_zero_and_reports_output_path() {
    let out_path = unique_temp_path("c053_cli");
    let out_arg = out_path.to_string_lossy().into_owned();
    let output = run_c053(&["--output", &out_arg]);
    assert_success(&output);
    let stdout = String::from_utf8(output.stdout).expect("stdout is utf-8");
    assert!(
        stdout.contains("c053_pathion_toy.csv"),
        "stdout should report the output path: {}",
        stdout
    );
    assert!(out_path.is_file(), "CLI should create {:?}", out_path);
    fs::remove_file(&out_path).expect("remove temp CSV");
    fs::remove_dir_all(out_path.parent().expect("temp parent")).expect("remove temp dir");
}
