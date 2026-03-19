//! proof_metrics: Pure Rust port of `proofs/scripts/collect_metrics.sh`.
//! Gathers proof compilation metrics for PGFPlots.

use std::fs;
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = Path::new(".");
    let proofs_dir = repo_root.join("proofs");
    let out_dir = proofs_dir.join("metrics");
    let out_file = out_dir.join("compile_times.csv");
    let summary_file = out_dir.join("summary.csv");

    fs::create_dir_all(&out_dir)?;

    let mut csv_content = String::from("file,lines,vo_size_bytes,compile_seconds\n");

    let mut v_files = Vec::new();
    let theories_dir = proofs_dir.join("theories");
    let verified_dir = proofs_dir.join("verified");

    if theories_dir.exists() {
        for entry in fs::read_dir(&theories_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("v") {
                v_files.push(path);
            }
        }
    }
    if verified_dir.exists() {
        for entry in fs::read_dir(&verified_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("v") {
                v_files.push(path);
            }
        }
    }

    let mut total_lines = 0;
    let mut total_vo = 0;

    for v_file in &v_files {
        let base = v_file.file_name().and_then(|s| s.to_str()).unwrap_or_default();
        let content = fs::read_to_string(v_file)?;
        let lines = content.lines().count();
        total_lines += lines;

        let vo_file = v_file.with_extension("vo");
        let vo_size = if vo_file.exists() {
            total_vo += 1;
            fs::metadata(&vo_file)?.len()
        } else {
            0
        };

        let mut compile_seconds = 0.0;
        if Command::new("rocq").arg("--version").output().is_ok() {
            let start = Instant::now();
            let status = Command::new("rocq")
                .arg("compile")
                .arg("-R")
                .arg(theories_dir.to_str().unwrap())
                .arg("OpenGororoba")
                .arg("-R")
                .arg(verified_dir.to_str().unwrap())
                .arg("OpenGororobaVerified")
                .arg(v_file.to_str().unwrap())
                .output()?;
            if status.status.success() {
                compile_seconds = start.elapsed().as_secs_f64();
            }
        }

        let label = base.replace(".v", "").replace("_", "\\_");
        csv_content.push_str(&format!("{},{},{},{:.2}\n", label, lines, vo_size, compile_seconds));
    }

    fs::write(&out_file, csv_content)?;

    let theory_count = if theories_dir.exists() {
        fs::read_dir(&theories_dir)?.filter_map(|e| e.ok()).filter(|e| e.path().extension().map(|s| s == "v").unwrap_or(false)).count()
    } else {
        0
    };
    let verified_count = if verified_dir.exists() {
        fs::read_dir(&verified_dir)?.filter_map(|e| e.ok()).filter(|e| e.path().extension().map(|s| s == "v").unwrap_or(false)).count()
    } else {
        0
    };
    let total_v = theory_count + verified_count;

    let summary_content = format!(
        "metric,value\ntheory_files,{}\nverified_files,{}\ntotal_v_files,{}\ntotal_vo_files,{}\ntotal_lines,{}\nkernel_checked_claims,{}\n",
        theory_count, verified_count, total_v, total_vo, total_lines, verified_count
    );
    fs::write(&summary_file, summary_content)?;

    println!("\nSummary: {} .v files, {} .vo compiled, {} total lines", total_v, total_vo, total_lines);
    println!("Output: {}", out_file.display());
    println!("Summary: {}", summary_file.display());

    Ok(())
}
