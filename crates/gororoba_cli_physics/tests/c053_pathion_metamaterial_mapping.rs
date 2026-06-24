use materials_core::{PathionToyLayer, default_c053_layers, write_c053_summary};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

const CANONICAL_TOML: &str =
    "registry/data/project_csv/canonical/PC-0007_c053_pathion_tmm_summary.toml";

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

fn read_canonical_rows(path: &std::path::Path) -> Vec<PathionToyLayer> {
    let raw = fs::read_to_string(path).expect("read canonical TOML");
    let mut rows = Vec::new();
    let mut in_rows = false;
    let mut current_row: Option<Vec<String>> = None;
    for line in raw.lines() {
        let trimmed = line.trim();
        if !in_rows {
            in_rows = trimmed == "rows = [";
            continue;
        }
        match trimmed {
            "]" => break,
            "[" => current_row = Some(Vec::new()),
            "]," => rows.push(current_row.take().expect("row opened")),
            _ => {
                if let Some(value) = parse_project_csv_string_cell(trimmed) {
                    current_row.as_mut().expect("row opened").push(value);
                }
            }
        }
    }
    rows.into_iter()
        .map(|cells| PathionToyLayer {
            layer_id: cells[0].parse().expect("usize layer_id"),
            pathion_indices: cells[1].clone(),
            diag_abs_mean: cells[2].clone(),
            n_real: cells[3].clone(),
            n_imag: cells[4].clone(),
            thickness_nm: cells[5].clone(),
            tmm_absorptance: cells[6].clone(),
            degeneracy_note: cells[7].clone(),
        })
        .collect()
}

fn parse_project_csv_string_cell(line: &str) -> Option<String> {
    let trimmed = line.trim_end_matches(',');
    let mut chars = trimmed.chars();
    if chars.next()? != '"' {
        return None;
    }
    let mut value = String::new();
    let mut escaped = false;
    for ch in chars {
        if escaped {
            value.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
        } else if ch == '"' {
            return Some(value);
        } else {
            value.push(ch);
        }
    }
    None
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
    let committed = read_canonical_rows(&repo_root().join(CANONICAL_TOML));
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
