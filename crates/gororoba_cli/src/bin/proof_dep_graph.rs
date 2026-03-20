//! proof_dep_graph: Pure Rust port of `proofs/scripts/dep_graph.sh`.
//! Generates a proof dependency graph in DOT format by parsing .v files.

use std::fs;
use std::path::Path;
use regex::Regex;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = Path::new(".");
    let proofs_dir = repo_root.join("proofs");
    let out_dir = proofs_dir.join("metrics");
    let dot_file = out_dir.join("dep_graph.dot");

    fs::create_dir_all(&out_dir)?;

    let mut dot_content = String::from(
        "digraph ProofDependencies {\n  rankdir=BT;\n  node [shape=box, fontsize=10, fontname=\"monospace\"];\n  edge [color=\"#666666\"];\n\n  // Theory nodes (blue)\n  node [style=filled, fillcolor=\"#d4e6f1\"];\n"
    );

    // Theory nodes
    let theories_dir = proofs_dir.join("theories");
    if theories_dir.exists() {
        for entry in fs::read_dir(&theories_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("v")
                && let Some(base) = path.file_stem().and_then(|s| s.to_str()) {
                    dot_content.push_str(&format!("  \"{}\";\n", base));
                }
        }
    }

    // Verified nodes
    dot_content.push_str("\n  // Verified claim nodes (green)\n  node [style=filled, fillcolor=\"#d5f5e3\"];\n");
    let verified_dir = proofs_dir.join("verified");
    if verified_dir.exists() {
        for entry in fs::read_dir(&verified_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("v")
                && let Some(base) = path.file_stem().and_then(|s| s.to_str()) {
                    dot_content.push_str(&format!("  \"{}\";\n", base));
                }
        }
    }

    // Dependencies
    dot_content.push_str("\n  // Dependencies\n");
    let re_open = Regex::new(r"From OpenGororoba Require Import ([^.]+)\.")?;
    let re_verified = Regex::new(r"From OpenGororobaVerified Require Import ([^.]+)\.")?;

    let mut v_files = Vec::new();
    if theories_dir.exists() {
        for entry in fs::read_dir(theories_dir)? {
            v_files.push(entry?.path());
        }
    }
    if verified_dir.exists() {
        for entry in fs::read_dir(verified_dir)? {
            v_files.push(entry?.path());
        }
    }

    for v_file in v_files {
        if v_file.extension().and_then(|s| s.to_str()) != Some("v") {
            continue;
        }
        let src = v_file.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        let content = fs::read_to_string(&v_file)?;

        for cap in re_open.captures_iter(&content) {
            for dep in cap[1].split_whitespace() {
                dot_content.push_str(&format!("  \"{}\" -> \"{}\";\n", src, dep));
            }
        }
        for cap in re_verified.captures_iter(&content) {
            for dep in cap[1].split_whitespace() {
                dot_content.push_str(&format!("  \"{}\" -> \"{}\";\n", src, dep));
            }
        }
    }

    dot_content.push_str("}\n");

    fs::write(&dot_file, dot_content)?;
    println!("Generated: {}", dot_file.display());

    // Render if graphviz is available
    if Command::new("dot").arg("-V").output().is_ok() {
        if Command::new("dot")
            .args(["-Tpdf", dot_file.to_str().unwrap(), "-o", out_dir.join("dep_graph.pdf").to_str().unwrap()])
            .output()
            .is_ok()
        {
            println!("Generated: {}", out_dir.join("dep_graph.pdf").display());
        }
        if Command::new("dot")
            .args(["-Tsvg", dot_file.to_str().unwrap(), "-o", out_dir.join("dep_graph.svg").to_str().unwrap()])
            .output()
            .is_ok()
        {
            println!("Generated: {}", out_dir.join("dep_graph.svg").display());
        }
    }

    // Convert DOT to TikZ for paper inclusion (requires dot2tex)
    if Command::new("dot2tex").arg("--version").output().is_ok() {
        let tikz_output = out_dir.join("dep_graph.tikz");
        let output = Command::new("dot2tex")
            .args(["-f", "tikz", "--figonly", dot_file.to_str().unwrap()])
            .output()?;
        fs::write(&tikz_output, output.stdout)?;
        println!("Generated: {}", tikz_output.display());
    }

    Ok(())
}
