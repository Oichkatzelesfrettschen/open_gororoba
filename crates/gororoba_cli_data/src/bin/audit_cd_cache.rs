use anyhow::Result;
use clap::Parser;
use regex::Regex;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(name = "audit-cd-cache", about = "Audit the off-site Cayley-Dickson document cache (Rust replacement for audit_cayley_dickson_cache.py)")]
struct Cli {
    #[arg(long, default_value = "/home/eirikr/Documents/Projects/CayleyDickson")]
    cache_root: PathBuf,
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long)]
    output: Option<PathBuf>,
}

#[derive(Debug, Default)]
struct AuditData {
    total_files: usize,
    pdf_count: usize,
    markdown_count: usize,
    html_count: usize,
    text_count: usize,
    top_level_counts: BTreeMap<String, usize>,
    tier1_pdf_counts: BTreeMap<String, usize>,
    flagged_paths: Vec<String>,
    manifest_total_paths: usize,
    manifest_missing_paths: Vec<String>,
    manifest_empty_paths: usize,
    chronology_entry_count: usize,
    chronology_on_disk_like: usize,
    chronology_missing: usize,
    chronology_mislabeled: usize,
    proofs_theories: usize,
    proofs_verified: usize,
    crate_count: usize,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut data = AuditData::default();

    if !cli.cache_root.exists() {
        eprintln!("Cache root {} not found", cli.cache_root.display());
        return Ok(());
    }

    let files: Vec<PathBuf> = WalkDir::new(&cli.cache_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .map(|e| e.path().to_path_buf())
        .collect();

    data.total_files = files.len();
    for path in &files {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();
        match ext.as_str() {
            "pdf" => data.pdf_count += 1,
            "md" => data.markdown_count += 1,
            "html" => data.html_count += 1,
            "txt" => data.text_count += 1,
            _ => {}
        }

        if let Ok(rel) = path.strip_prefix(&cli.cache_root) {
            let parts: Vec<_> = rel.components().collect();
            if !parts.is_empty() {
                let top = parts[0].as_os_str().to_string_lossy().to_string();
                *data.top_level_counts.entry(top.clone()).or_default() += 1;

                if top == "tier1_core_cd_algebra" && parts.len() >= 2 && ext == "pdf" {
                    let sub = parts[1].as_os_str().to_string_lossy().to_string();
                    *data.tier1_pdf_counts.entry(sub).or_default() += 1;
                }
            }

            let rel_str = rel.to_string_lossy().to_string();
            let rel_lower = rel_str.to_lowercase();
            if ["_dup.", "_mirror.", "blocked.html", "captcha.html", "placeholder", "_full.pdf", "_full.tex"]
                .iter().any(|token| rel_lower.contains(token)) {
                data.flagged_paths.push(rel_str);
            }
        }
    }
    data.flagged_paths.sort();

    // Manifest audit
    let manifest_path = cli.cache_root.join("metadata/repo_extracted_metadata/MANIFEST.toml");
    if manifest_path.exists() {
        let content = fs::read_to_string(&manifest_path)?;
        let manifest_re = Regex::new(r#"(?m)^local_pdf\s*=\s*"([^"]*)""#)?;
        for cap in manifest_re.captures_iter(&content) {
            let local_pdf = cap[1].trim();
            data.manifest_total_paths += 1;
            if local_pdf.is_empty() {
                data.manifest_empty_paths += 1;
            } else if !cli.cache_root.join(local_pdf).exists() {
                data.manifest_missing_paths.push(local_pdf.to_string());
            }
        }
    }
    data.manifest_missing_paths.sort();

    // Chronology audit
    let chrono_path = cli.cache_root.join("CHRONOLOGICAL_REFERENCE_MATRIX.md");
    if chrono_path.exists() {
        let content = fs::read_to_string(&chrono_path)?;
        for line in content.lines() {
            if !line.trim().starts_with('|') { continue; }
            let cells: Vec<_> = line.trim_matches('|').split('|').map(|s| s.trim()).collect();
            if cells.len() < 7 { continue; }
            if ["#", "ID", "Family"].contains(&cells[0]) { continue; }
            if cells.iter().all(|c| c.chars().all(|ch| ch == '-')) { continue; }

            data.chronology_entry_count += 1;
            let status = cells[cells.len() - 1];
            if status.starts_with("[ON DISK]") || status.starts_with("[FORMALIZED]") {
                data.chronology_on_disk_like += 1;
            } else if status.starts_with("[MISSING]") || status.starts_with("[AUDIT]") {
                data.chronology_missing += 1;
            } else if status.starts_with("[MISLABELED]") {
                data.chronology_mislabeled += 1;
            }
        }
    }

    // Repo audit
    data.proofs_theories = WalkDir::new(cli.repo_root.join("proofs/theories"))
        .into_iter().filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|ext| ext == "v")).count();
    data.proofs_verified = WalkDir::new(cli.repo_root.join("proofs/verified"))
        .into_iter().filter_map(|e| e.ok()).filter(|e| e.path().extension().is_some_and(|ext| ext == "v")).count();
    data.crate_count = WalkDir::new(cli.repo_root.join("crates"))
        .into_iter().filter_map(|e| e.ok()).filter(|e| e.file_name() == "Cargo.toml").count();

    let mut report = String::new();
    report.push_str("# Cayley-Dickson Cache Audit\n\n");
    report.push_str(&format!("- Cache root: `{}`\n", cli.cache_root.display()));
    report.push_str(&format!("- Repo root: `{}`\n\n", cli.repo_root.display()));
    
    report.push_str("## Corpus Snapshot\n\n");
    report.push_str(&format!("- Total files: {}\n", data.total_files));
    report.push_str(&format!("- PDFs: {}\n", data.pdf_count));
    report.push_str(&format!("- Markdown notes: {}\n", data.markdown_count));
    report.push_str(&format!("- HTML traces: {}\n", data.html_count));
    report.push_str(&format!("- Plain-text notes: {}\n\n", data.text_count));

    report.push_str("### Top-Level Layout\n\n");
    let mut top_counts: Vec<_> = data.top_level_counts.iter().collect();
    top_counts.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in top_counts {
        report.push_str(&format!("- `{}`: {}\n", name, count));
    }

    report.push_str("\n### Tier 1 PDF Density\n\n");
    let mut tier1_counts: Vec<_> = data.tier1_pdf_counts.iter().collect();
    tier1_counts.sort_by(|a, b| b.1.cmp(a.1));
    for (name, count) in tier1_counts {
        report.push_str(&format!("- `{}`: {} PDFs\n", name, count));
    }

    report.push_str("\n## Drift Findings\n\n");
    report.push_str(&format!("- `CHRONOLOGICAL_REFERENCE_MATRIX.md` currently has {} table entries.\n", data.chronology_entry_count));
    report.push_str(&format!("- Chronology row statuses normalize to {} on-disk/formalized, {} missing/audit-needed, and {} mislabeled.\n",
        data.chronology_on_disk_like, data.chronology_missing, data.chronology_mislabeled));
    report.push_str(&format!("- `metadata/repo_extracted_metadata/MANIFEST.toml` tracks {} `local_pdf` entries; {} currently fail to resolve under the cache root and {} are blank placeholders.\n",
        data.manifest_total_paths, data.manifest_missing_paths.len(), data.manifest_empty_paths));
    
    report.push_str("\n### Flagged Files\n\n");
    for rel in &data.flagged_paths {
        report.push_str(&format!("- `{}`\n", rel));
    }

    report.push_str("\n### Missing Manifest Paths\n\n");
    if data.manifest_missing_paths.is_empty() {
        report.push_str("- None\n");
    } else {
        for rel in data.manifest_missing_paths.iter().take(40) {
            report.push_str(&format!("- `{}`\n", rel));
        }
        if data.manifest_missing_paths.len() > 40 {
            report.push_str(&format!("- ... and {} more\n", data.manifest_missing_paths.len() - 40));
        }
    }

    report.push_str("\n## Repo Surface Crosswalk\n\n");
    report.push_str(&format!("- Rocq theory files: {}\n", data.proofs_theories));
    report.push_str(&format!("- Rocq verified files: {}\n", data.proofs_verified));
    report.push_str(&format!("- Rust crates: {}\n", data.crate_count));

    if let Some(out_path) = cli.output {
        fs::write(out_path, report)?;
    } else {
        print!("{}", report);
    }

    Ok(())
}
