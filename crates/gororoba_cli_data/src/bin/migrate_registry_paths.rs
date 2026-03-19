//! migrate-registry-paths: Update file paths in Makefile, xtask, and freshness checkers.
//!
//! Replaces legacy Markdown generated paths with new Rust mirror paths.
//!
//! Migrated from migrate_xtask.py, migrate_makefile.py, and migrate_freshness.py.

use anyhow::{Context, Result};
use regex::Regex;
use std::fs;
use std::path::Path;

fn main() -> Result<()> {
    // 1. migrate_xtask logic
    let xtask_path = Path::new("xtask/src/main.rs");
    if xtask_path.exists() {
        let mut content = fs::read_to_string(xtask_path)?;
        content = content.replace("docs/db/catalog.md", "crates/data_core/src/registry_mirrors/db_catalog.rs");
        
        let wrapper_re = Regex::new(r"fn render_catalog_markdown\(snapshot: &SchemaSnapshot\) -> String \{(?s:.*?)\}")?;
        let wrapper = r#"fn render_catalog_markdown(snapshot: &SchemaSnapshot) -> String {
    let mut rustdoc = String::new();
    rustdoc.push_str("//! # Database Catalog Snapshot\n//!\n");
    rustdoc.push_str(&format!("//! Generated at: {}\n", Local::now().to_rfc3339()));
    rustdoc.push_str("//!\n");
    for table in &snapshot.tables {
        rustdoc.push_str(&format!("//! ## Table: {}\n", table.name));
        rustdoc.push_str("//! | Column | Type | Null | Key | Default | Extra |\n");
        rustdoc.push_str("//! | --- | --- | --- | --- | --- | --- |\n");
        for col in &table.columns {
            rustdoc.push_str(&format!(
                "//! | {} | {} | {} | {} | {} | {} |\n",
                col.name,
                col.data_type,
                if col.is_nullable { "YES" } else { "NO" },
                col.key.as_deref().unwrap_or(""),
                col.default_value.as_deref().unwrap_or("NULL"),
                col.extra.as_deref().unwrap_or("")
            ));
        }
        rustdoc.push_str("//!\n");
    }
    rustdoc
}

fn render_catalog_markdown_raw(snapshot: &SchemaSnapshot) -> String {"#;
        
        content = wrapper_re.replace(&content, wrapper).to_string();
        fs::write(xtask_path, content)?;
        println!("Migrated xtask paths.");
    }

    // 2. migrate_makefile logic
    let makefile_path = Path::new("Makefile");
    if makefile_path.exists() {
        let content = fs::read_to_string(makefile_path)?;
        let makefile_re = Regex::new(r#""\$\(MARKDOWN_EXPORT_OUT_DIR\)/([A-Z_a-z0-9]+)\.md""#)?;
        let new_content = makefile_re.replace_all(&content, |caps: &regex::Captures| {
            let stem = caps.get(1).unwrap().as_str().to_lowercase();
            format!(r#""crates/data_core/src/registry_mirrors/{stem}.rs""#)
        });
        fs::write(makefile_path, new_content)?;
        println!("Migrated Makefile paths.");
    }

    // 3. migrate_freshness logic
    let freshness_path = Path::new("crates/gororoba_cli_data/src/bin/verify_registry_mirror_freshness.rs");
    if freshness_path.exists() {
        let mut content = fs::read_to_string(freshness_path)?;
        let freshness_re = Regex::new(r#"(docs/generated/|docs/)([A-Z_a-z0-9]+)\.md"#)?;
        content = freshness_re.replace_all(&content, |caps: &regex::Captures| {
            let stem = caps.get(2).unwrap().as_str().to_lowercase();
            format!("crates/data_core/src/registry_mirrors/{stem}.rs")
        }).to_string();
        
        content = content.replace("\"NAVIGATOR.md\"", "\"crates/data_core/src/registry_mirrors/navigator.rs\"");
        content = content.replace("NAVIGATOR.md", "crates/data_core/src/registry_mirrors/navigator.rs");
        
        fs::write(freshness_path, content)?;
        println!("Migrated freshness paths.");
    }

    Ok(())
}
