//! proof_doc: Pure Rust port of `proofs/scripts/generate_rocqdoc.sh`.
//! Produces browsable HTML documentation via rocq doc.

use std::fs;
use std::path::Path;
use std::process::Command;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo_root = Path::new(".");
    let proofs_dir = repo_root.join("proofs");
    let out_dir = proofs_dir.join("html");

    if !Command::new("rocq").arg("--version").output().is_ok() {
        println!("SKIP: rocq not found");
        return Ok(());
    }

    fs::create_dir_all(&out_dir)?;

    println!("Generating rocq doc HTML...");

    let theories_dir = proofs_dir.join("theories");
    let verified_dir = proofs_dir.join("verified");

    let mut theory_files = Vec::new();
    if theories_dir.exists() {
        for entry in fs::read_dir(&theories_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("v") {
                theory_files.push(path);
            }
        }
    }

    let mut verified_files = Vec::new();
    if verified_dir.exists() {
        for entry in fs::read_dir(&verified_dir)? {
            let path = entry?.path();
            if path.extension().and_then(|s| s.to_str()) == Some("v") {
                verified_files.push(path);
            }
        }
    }

    for v_file in theory_files.iter().chain(verified_files.iter()) {
        let base = v_file.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        println!("  rocq doc: {}", base);
        let output_file = out_dir.join(format!("{}.html", base));
        
        let _ = Command::new("rocq")
            .arg("doc")
            .arg("--html")
            .arg("-R")
            .arg(theories_dir.to_str().unwrap())
            .arg("OpenGororoba")
            .arg("-R")
            .arg(verified_dir.to_str().unwrap())
            .arg("OpenGororobaVerified")
            .arg(v_file.to_str().unwrap())
            .arg("-o")
            .arg(output_file.to_str().unwrap())
            .output();
    }

    // Generate index.html
    let mut index_content = String::from(
        "<!DOCTYPE html>\n<html><head><title>open_gororoba Proof Documentation</title>\n\
        <style>body{font-family:monospace;max-width:800px;margin:2em auto}\n\
        a{color:#2200cc}h2{border-bottom:1px solid #ccc;padding-bottom:0.3em}</style>\n\
        </head><body>\n<h1>open_gororoba: Formal Proofs</h1>\n<h2>Theories</h2><ul>\n"
    );

    for v_file in &theory_files {
        let base = v_file.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        index_content.push_str(&format!("<li><a href=\"{}.html\">{}</a></li>\n", base, base));
    }

    index_content.push_str("</ul>\n<h2>Verified Claims</h2><ul>\n");

    for v_file in &verified_files {
        let base = v_file.file_stem().and_then(|s| s.to_str()).unwrap_or_default();
        index_content.push_str(&format!("<li><a href=\"{}.html\">{}</a></li>\n", base, base));
    }

    index_content.push_str("</ul></body></html>\n");

    fs::write(out_dir.join("index.html"), index_content)?;

    println!("\nHTML documentation in: {}", out_dir.display());
    println!("Open: {}/index.html", out_dir.display());

    Ok(())
}
