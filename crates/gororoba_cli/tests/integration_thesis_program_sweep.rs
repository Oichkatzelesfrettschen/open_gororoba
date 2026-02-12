use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

fn binary_path() -> PathBuf {
    let keys = [
        "CARGO_BIN_EXE_thesis-program-sweep",
        "CARGO_BIN_EXE_thesis_program_sweep",
    ];
    for key in keys {
        if let Ok(path) = std::env::var(key) {
            return PathBuf::from(path);
        }
    }

    for (key, value) in std::env::vars() {
        if key.starts_with("CARGO_BIN_EXE_")
            && (key.ends_with("thesis-program-sweep") || key.ends_with("thesis_program_sweep"))
        {
            return PathBuf::from(value);
        }
    }

    panic!("thesis-program-sweep binary not available for integration test");
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

fn run_sweep(output_dir: &str) -> Output {
    Command::new(binary_path())
        .args(["--output-dir", output_dir])
        .output()
        .expect("failed to run thesis-program-sweep")
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
fn thesis_program_sweep_writes_all_artifacts() {
    let out_dir = unique_temp_dir("thesis_sweep");
    let out_dir_arg = out_dir.to_string_lossy().into_owned();

    let output = run_sweep(&out_dir_arg);
    assert_success(&output);

    let paths = [
        out_dir.join("thesis1_scalar_tov_sweep.toml"),
        out_dir.join("thesis2_thickening_threshold.toml"),
        out_dir.join("thesis3_epoch_alignment.toml"),
        out_dir.join("thesis4_latency_law_suite.toml"),
    ];

    for path in &paths {
        assert!(path.is_file(), "missing artifact: {}", path.display());
    }

    let t1 = fs::read_to_string(&paths[0]).expect("read thesis1 artifact");
    assert!(t1.contains("cassini_lower_bound"));
    assert!(t1.contains("[summary]"));

    let t2 = fs::read_to_string(&paths[1]).expect("read thesis2 artifact");
    assert!(t2.contains("threshold_radians"));
    assert!(t2.contains("verdict ="));

    let t3 = fs::read_to_string(&paths[2]).expect("read thesis3 artifact");
    assert!(t3.contains("hubble_alignment"));
    assert!(t3.contains("plateau_label"));

    let t4 = fs::read_to_string(&paths[3]).expect("read thesis4 artifact");
    assert!(t4.contains("inverse_square_r2"));
    assert!(t4.contains("latency_law"));

    fs::remove_dir_all(out_dir).expect("remove temp output dir");
}
