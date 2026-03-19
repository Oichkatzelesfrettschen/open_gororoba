//! markdown-to-rust: Convert Markdown files to Rust module-level documentation.
//!
//! Iterates over defined pairs of (source_md, destination_rs), converts
//! the markdown content into //! prefixed lines, and writes them to the
//! destination. If requested, it can prepend the content to an existing file.
//!
//! Migrated from convert_all_md.py, convert_remaining.py, and convert_git_md.py.

use anyhow::{Context, Result};
use std::fs;
use std::path::{Path, PathBuf};

struct ConversionJob {
    src: &'static str,
    dst: &'static str,
    prepend: bool,
}

const JOBS: &[ConversionJob] = &[
    ConversionJob {
        src: "crates/lbm_3d_cuda/README.md",
        dst: "crates/lbm_3d_cuda/src/lib.rs",
        prepend: true,
    },
    ConversionJob {
        src: "apps/gororoba_studio/README.md",
        dst: "crates/data_core/src/registry_mirrors/gororoba_studio_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "README.md",
        dst: "crates/data_core/src/registry_mirrors/project_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "curated/README.md",
        dst: "crates/data_core/src/registry_mirrors/curated_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "data/artifacts/README.md",
        dst: "crates/data_core/src/registry_mirrors/data_artifacts_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "experiments/manga_zd_null/charts/README.md",
        dst: "crates/data_core/src/registry_mirrors/manga_zd_null_charts_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "CLAUDE.md",
        dst: "crates/data_core/src/registry_mirrors/claude_instructions.rs",
        prepend: false,
    },
    ConversionJob {
        src: "data/external/README.md",
        dst: "crates/data_core/src/registry_mirrors/data_external_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "GEMINI.md",
        dst: "crates/data_core/src/registry_mirrors/gemini_instructions.rs",
        prepend: false,
    },
    // From convert_remaining.py
    ConversionJob {
        src: "docs/external_sources/PIONEER_FLYBY_ANOMALY_SOURCES.md",
        dst: "crates/data_core/src/registry_mirrors/pioneer_flyby_anomaly_sources.rs",
        prepend: false,
    },
    ConversionJob {
        src: "docs/book/src/proofs/index.md",
        dst: "crates/data_core/src/registry_mirrors/proofs_index.rs",
        prepend: false,
    },
    ConversionJob {
        src: "docs/convos/README.md",
        dst: "crates/data_core/src/registry_mirrors/convos_readme.rs",
        prepend: false,
    },
    ConversionJob {
        src: "docs/research/high_dimensional_algebra_unification_2026.md",
        dst: "crates/data_core/src/registry_mirrors/high_dimensional_algebra_unification_2026.rs",
        prepend: false,
    },
    ConversionJob {
        src: "docs/tickets/TICKET_PIONEER_FLYBY_FRACTAL_BRIDGE.md",
        dst: "crates/data_core/src/registry_mirrors/ticket_pioneer_flyby_fractal_bridge.rs",
        prepend: false,
    },
];

fn main() -> Result<()> {
    for job in JOBS {
        let src_path = Path::new(job.src);
        let dst_path = Path::new(job.dst);

        if src_path.exists() {
            let content = fs::read_to_string(src_path)
                .with_context(|| format!("read {}", src_path.display()))?;
            let mut rustdoc = String::new();
            for line in content.lines() {
                if line.is_empty() {
                    rustdoc.push_str("//!\n");
                } else {
                    rustdoc.push_str(&format!("//! {}\n", line));
                }
            }

            if job.prepend && dst_path.exists() {
                let existing = fs::read_to_string(dst_path)
                    .with_context(|| format!("read existing {}", dst_path.display()))?;
                // Remove old include_str if present
                let existing = existing.replace("#![doc = include_str!(\"../README.md\")]\n", "");
                let new_content = format!("{}\n{}", rustdoc, existing);
                fs::write(dst_path, new_content)
                    .with_context(|| format!("write {}", dst_path.display()))?;
            } else {
                if let Some(parent) = dst_path.parent() {
                    fs::create_dir_all(parent)
                        .with_context(|| format!("create dir {}", parent.display()))?;
                }
                fs::write(dst_path, rustdoc)
                    .with_context(|| format!("write {}", dst_path.display()))?;
            }

            fs::remove_file(src_path).with_context(|| format!("remove {}", src_path.display()))?;
            println!("Converted {} -> {}", job.src, job.dst);
        }
    }

    // Update crates/data_core/src/registry_mirrors/mod.rs
    let mirrors_dir = Path::new("crates/data_core/src/registry_mirrors");
    if mirrors_dir.exists() {
        let mut mods = Vec::new();
        for entry in fs::read_dir(mirrors_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("rs") {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap();
                if stem != "mod" {
                    mods.push(format!("pub mod {};", stem));
                }
            }
        }
        mods.sort();
        let mod_content = mods.join("\n") + "\n";
        fs::write(mirrors_dir.join("mod.rs"), mod_content).context("write mirrors/mod.rs")?;
    }

    println!("Done converting markdown files.");
    Ok(())
}
