use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

fn binary_path() -> PathBuf {
    if let Some(path) = option_env!("CARGO_BIN_EXE_thesis-42-support") {
        return PathBuf::from(path);
    }
    if let Some(path) = option_env!("CARGO_BIN_EXE_thesis_42_support") {
        return PathBuf::from(path);
    }

    let keys = [
        "CARGO_BIN_EXE_thesis-42-support",
        "CARGO_BIN_EXE_thesis_42_support",
    ];
    for key in keys {
        if let Ok(path) = std::env::var(key) {
            return PathBuf::from(path);
        }
    }

    panic!("thesis-42-support binary not available for integration test");
}

fn unique_temp_dir(label: &str) -> PathBuf {
    let timestamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!(
        "gororoba_cli_{}_{}_{}",
        label,
        std::process::id(),
        timestamp
    ));
    fs::create_dir_all(&dir).expect("create temp dir");
    dir
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("crate parent")
        .parent()
        .expect("workspace root")
        .to_path_buf()
}

fn run_binary(output_dir: &str) -> Output {
    Command::new(binary_path())
        .current_dir(repo_root())
        .args(["--output-dir", output_dir, "--theta-steps", "8"])
        .output()
        .expect("failed to run thesis-42-support")
}

fn assert_success(output: &Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "command failed: {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        stdout,
        stderr
    );
}

#[test]
fn thesis_42_support_writes_expected_bundle() {
    let out_dir = unique_temp_dir("thesis_42_support");
    let out_arg = out_dir.to_string_lossy().into_owned();
    let output = run_binary(&out_arg);
    assert_success(&output);

    let summary = out_dir.join("summary.toml");
    let majorana_csv = out_dir.join("majorana_friction_sweep.csv");
    let harmonic_csv = out_dir.join("harmonic_halo_reference.csv");
    let nonlocal_csv = out_dir.join("nonlocal_topologies.csv");
    let gravastar_csv = out_dir.join("gravastar_bridge_model.csv");

    for path in [
        &summary,
        &majorana_csv,
        &harmonic_csv,
        &nonlocal_csv,
        &gravastar_csv,
    ] {
        assert!(path.is_file(), "missing artifact: {}", path.display());
    }

    let summary_text = fs::read_to_string(&summary).expect("read summary");
    assert!(summary_text.contains("design_stage_only"));
    assert!(summary_text.contains("algebraic_not_physical"));
    assert!(summary_text.contains("falsifiable_observable_lane"));
    assert!(summary_text.contains("gravastar_bridge_unsupported"));
    assert!(summary_text.contains("calibration_id = \"nonlocal_cable_chen_2023\""));
    assert!(summary_text.contains("projection_gate_pass = true"));
    assert!(summary_text.contains("physical_claim_status = \"refuted_or_unsupported\""));
    assert!(summary_text.contains("physical_claim_status = \"closed_obstructed\""));
    assert!(summary_text.contains("model_claim_status = \"verified_model_under_assumptions\""));
    assert!(summary_text.contains("law_label = \"linear_homotopy_lambda\""));
    assert!(summary_text.contains("exact_nfw_recovery = true"));
    assert!(summary_text.contains("bound_supported = false"));

    fs::remove_dir_all(out_dir).expect("remove temp output dir");
}
