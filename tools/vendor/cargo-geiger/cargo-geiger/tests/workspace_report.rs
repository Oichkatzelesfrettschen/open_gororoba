use std::{
    fs,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

#[test]
fn workspace_report_covers_all_roots_and_propagates_compile_failures() {
    let unique = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = std::env::temp_dir().join(format!(
        "geiger-workspace-report-{}-{unique}",
        std::process::id()
    ));
    fs::create_dir_all(root.join("first/src")).unwrap();
    fs::create_dir_all(root.join("second/src")).unwrap();
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = [\"first\", \"second\"]\nresolver = \"2\"\n",
    )
    .unwrap();
    for name in ["first", "second"] {
        fs::write(root.join(name).join("Cargo.toml"), format!("[package]\nname = \"{name}\"\nversion = \"0.1.0\"\nedition = \"2021\"\n")).unwrap();
        fs::write(
            root.join(name).join("src/lib.rs"),
            "pub fn value() -> i32 { 1 }\n",
        )
        .unwrap();
    }
    let second_manifest = root.join("second/Cargo.toml");
    let manifest = fs::read_to_string(&second_manifest).unwrap();
    fs::write(&second_manifest, format!("{manifest}\n[features]\ngpu = []\n[[bin]]\nname = \"gpu-only\"\npath = \"src/gpu.rs\"\nrequired-features = [\"gpu\"]\n")).unwrap();
    fs::write(
        root.join("second/src/gpu.rs"),
        "compile_error!(\"requires GPU hardware\"); fn main() {}\n",
    )
    .unwrap();
    fs::write(root.join("second/src/unused_invalid.rs"), "pub fn (").unwrap();
    fs::write(
        root.join("second/src/unused_deep.rs"),
        format!(
            "fn nested() -> i32 {{ {} 0 {} }}",
            "unsafe {".repeat(768),
            "}".repeat(768)
        ),
    )
    .unwrap();
    let run = || {
        Command::new(env!("CARGO_BIN_EXE_cargo-geiger"))
            .current_dir(&root)
            .env("RUSTC_WRAPPER", "")
            .env("CARGO_TARGET_DIR", root.join("target"))
            .args([
                "--workspace-report",
                "--offline",
                "--include-tests",
                "--manifest-path",
            ])
            .arg(root.join("Cargo.toml"))
            .output()
            .unwrap()
    };
    let output = run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: serde_json::Value =
        serde_json::from_slice(&output.stdout).unwrap();
    let roots = report["root_reports"].as_array().unwrap();
    assert_eq!(report["compilation_count"], 1);
    assert!(report["excluded_graph_packages_with_compiler_sources"]
        .as_array()
        .unwrap()
        .is_empty());
    assert!(report["metadata_packages_outside_selected_graph"].is_array());
    assert_eq!(report["syntax_scan_complete"], false);
    assert_eq!(report["syntax_files_failed"], 1);
    assert!(report["parse_errors"][0]["path"]
        .as_str()
        .unwrap()
        .ends_with("unused_invalid.rs"));
    assert_eq!(report["parser_worker_stack_bytes"], 16 * 1024 * 1024);
    for row in roots {
        for package in row["report"]["packages"].as_array().unwrap() {
            let name = package["package"]["id"]["name"].as_str().unwrap();
            let expected_url =
                url::Url::from_directory_path(root.join(name)).unwrap();
            assert_eq!(
                package["package"]["id"]["source"]["Path"],
                expected_url.as_str()
            );
        }
    }

    assert_eq!(roots.len(), 2);
    let mut reported = roots
        .iter()
        .map(|row| row["root"].as_str().unwrap())
        .collect::<Vec<_>>();
    reported.sort();
    let mut expected = report["workspace_members"]
        .as_array()
        .unwrap()
        .iter()
        .map(|root| root.as_str().unwrap())
        .collect::<Vec<_>>();
    expected.sort();
    assert_eq!(reported, expected);
    assert!(roots.iter().all(|row| row["used_source_scope"]
        .as_str()
        .unwrap()
        .contains("not isolated per-root usage")));
    fs::write(
        root.join("second/src/lib.rs"),
        "compile_error!(\"member failure\");\n",
    )
    .unwrap();
    assert!(!run().status.success());
    fs::remove_dir_all(root).unwrap();
}
