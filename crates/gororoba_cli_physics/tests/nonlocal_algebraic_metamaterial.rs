use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

const CALIBRATION_ID: &str = "yig_magnonic_kaman_2026";

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn nonlocal_binary_path() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_nonlocal-algebraic-metamaterial") {
        return PathBuf::from(path);
    }
    if let Ok(path) = std::env::var("CARGO_BIN_EXE_nonlocal-algebraic-metamaterial") {
        return PathBuf::from(path);
    }
    panic!("CARGO_BIN_EXE_nonlocal-algebraic-metamaterial is not available");
}

fn unique_temp_dir(label: &str) -> PathBuf {
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
    dir
}

fn run_nonlocal(args: &[&str]) -> Output {
    Command::new(nonlocal_binary_path())
        .current_dir(repo_root())
        .args(args)
        .output()
        .expect("failed to execute nonlocal-algebraic-metamaterial")
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
fn benchmark_writes_sidecar_rich_summary() {
    let out_dir = unique_temp_dir("c010_benchmark");
    let out_arg = out_dir.to_string_lossy().into_owned();
    let output = run_nonlocal(&[
        "benchmark",
        "--calibration-id",
        CALIBRATION_ID,
        "--output-dir",
        &out_arg,
    ]);
    assert_success(&output);

    let summary_path = out_dir.join("benchmark_summary.toml");
    assert!(summary_path.is_file(), "missing {:?}", summary_path);
    let summary = fs::read_to_string(&summary_path).expect("read benchmark summary");
    assert!(summary.contains("[spectral_crosscheck]"));
    assert!(summary.contains("zd_flat_band_fraction"));
    assert!(summary.contains("graphene_valley_chern_k"));
    assert!(summary.contains("[[topologies]]"));
    assert!(summary.contains("projection_gate_verdict"));

    fs::remove_dir_all(&out_dir).expect("remove temp dir");
}
