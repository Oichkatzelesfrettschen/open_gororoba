// paper_build.rs -- Rust-native paper builder for arXiv submission
//
// Uses tectonic CLI for LaTeX compilation. Install via:
//   cargo install tectonic
//   -- or --
//   paru -S tectonic
//
// Usage:
//   cargo run --bin paper-build -- build     # compile PDF
//   cargo run --bin paper-build -- arxiv     # create submission tarball

use std::{
    path::{Path, PathBuf},
    process::Command,
};

fn project_root() -> PathBuf {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    Path::new(&manifest)
        .parent() // crates/
        .and_then(|p| p.parent()) // root
        .unwrap_or(Path::new("."))
        .to_path_buf()
}

fn check_tectonic() -> Result<(), String> {
    Command::new("tectonic")
        .arg("--version")
        .output()
        .map_err(|_| {
            "tectonic not found. Install via:\n  \
             cargo install tectonic\n  \
             -- or --\n  \
             paru -S tectonic"
                .to_string()
        })?;
    Ok(())
}

fn build_pdf(root: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let tex_dir = root.join("docs/latex");
    let tex_file = tex_dir.join("llm_scaffold_paper.tex");
    let out_dir = tex_dir.join("out");
    let bib_dir = root.join("papers/bib");

    if !tex_file.exists() {
        return Err(format!("TeX source not found: {}", tex_file.display()).into());
    }

    std::fs::create_dir_all(&out_dir)?;

    println!("Building: {}", tex_file.display());
    println!("Output:   {}", out_dir.display());
    println!("Bib dir:  {}", bib_dir.display());

    // tectonic -X compile handles multi-pass (latex + bibtex + latex + latex)
    // automatically. -Z search-path adds the bib directory.
    let status = Command::new("tectonic")
        .arg("-X")
        .arg("compile")
        .arg(&tex_file)
        .arg("--outdir")
        .arg(&out_dir)
        .arg("-Z")
        .arg(format!("search-path={}", bib_dir.display()))
        .arg("--keep-intermediates")
        .status()?;

    if !status.success() {
        return Err(format!("tectonic exited with status {}", status).into());
    }

    let pdf = out_dir.join("llm_scaffold_paper.pdf");
    if pdf.exists() {
        println!(
            "PDF written: {} ({:.1} MB)",
            pdf.display(),
            std::fs::metadata(&pdf)?.len() as f64 / 1_048_576.0
        );
    }

    Ok(pdf)
}

fn build_arxiv_tarball(root: &Path) -> Result<PathBuf, Box<dyn std::error::Error>> {
    // First build to generate .bbl (kept via --keep-intermediates)
    build_pdf(root)?;

    let tex_dir = root.join("docs/latex");
    let out_dir = tex_dir.join("out");
    let bib_dir = root.join("papers/bib");
    let tarball = out_dir.join("arxiv_submission.tar.gz");

    // Collect files for arXiv submission:
    // 1. Main .tex file
    // 2. Generated .bbl file (arXiv does not run bibtex)
    // 3. Any \input appendix .tex files
    // 4. Bib style file if custom
    let mut files: Vec<(PathBuf, String)> = Vec::new();

    // Main tex
    let main_tex = tex_dir.join("llm_scaffold_paper.tex");
    files.push((main_tex, "llm_scaffold_paper.tex".to_string()));

    // Generated .bbl
    let bbl = out_dir.join("llm_scaffold_paper.bbl");
    if bbl.exists() {
        files.push((bbl, "llm_scaffold_paper.bbl".to_string()));
    } else {
        eprintln!("WARNING: .bbl file not found; arXiv needs this for references");
    }

    // Appendix tex files
    for entry in std::fs::read_dir(&tex_dir)? {
        let entry = entry?;
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str())
            && name.ends_with("_appendix.tex")
        {
            files.push((path.clone(), name.to_string()));
        }
    }

    // Bibliography (.bib) -- include in case arXiv needs it
    let bib_file = bib_dir.join("cayley_dickson.bib");
    if bib_file.exists() {
        files.push((bib_file, "cayley_dickson.bib".to_string()));
    }

    // Create tarball using tar command
    let file_args: Vec<String> = files
        .iter()
        .map(|(src, name)| format!("{}={}", name, src.display()))
        .collect();

    let mut cmd = Command::new("tar");
    cmd.arg("czf").arg(&tarball);
    for arg in &file_args {
        cmd.arg("--transform")
            .arg(format!("s|.*|{}|", arg.split('=').next().unwrap()));
    }
    // Simpler: just add files with full paths, arXiv accepts flat archives
    let mut cmd = Command::new("tar");
    cmd.arg("czf").arg(&tarball).arg("-C").arg("/");
    for (src, _name) in &files {
        cmd.arg(src.to_str().unwrap_or_default());
    }

    // Actually, the simplest correct approach: copy files to a temp dir, tar it
    let staging = out_dir.join("arxiv_staging");
    if staging.exists() {
        std::fs::remove_dir_all(&staging)?;
    }
    std::fs::create_dir_all(&staging)?;

    for (src, name) in &files {
        let dest = staging.join(name);
        std::fs::copy(src, &dest)?;
        println!("  Staged: {}", name);
    }

    let status = Command::new("tar")
        .arg("czf")
        .arg(&tarball)
        .arg("-C")
        .arg(&staging)
        .arg(".")
        .status()?;

    // Clean up staging
    std::fs::remove_dir_all(&staging)?;

    if !status.success() {
        return Err("tar failed".into());
    }

    println!(
        "arXiv tarball: {} ({:.1} KB)",
        tarball.display(),
        std::fs::metadata(&tarball)?.len() as f64 / 1024.0
    );

    Ok(tarball)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let subcmd = args.get(1).map(|s| s.as_str()).unwrap_or("build");

    check_tectonic()?;
    let root = project_root();

    match subcmd {
        "build" => {
            build_pdf(&root)?;
        }
        "arxiv" => {
            build_arxiv_tarball(&root)?;
        }
        other => {
            eprintln!("Unknown subcommand: {other}");
            eprintln!("Usage: paper-build [build|arxiv]");
            std::process::exit(1);
        }
    }

    Ok(())
}
