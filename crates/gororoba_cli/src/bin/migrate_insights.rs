//! Migrate insights from INSIGHTS.md (markdown sections) to TOML.
//!
//! Parses `## I-NNN: Title` sections from docs/INSIGHTS.md and generates
//! `registry/insights.toml`.  Designed for one-shot conversion with future
//! re-runs producing identical output.

use std::path::PathBuf;

use clap::Parser;
use regex::Regex;

/// Migrate insights from markdown INSIGHTS.md to TOML registry format.
#[derive(Parser)]
#[command(name = "migrate-insights")]
struct Args {
    /// Path to INSIGHTS.md
    #[arg(long, default_value = "docs/INSIGHTS.md")]
    input: PathBuf,

    /// Output TOML file
    #[arg(long, default_value = "registry/insights.toml")]
    output: PathBuf,
}

#[derive(serde::Serialize)]
struct InsightsRegistry {
    insight: Vec<Insight>,
}

#[derive(serde::Serialize)]
struct Insight {
    id: String,
    title: String,
    date: String,
    status: String,
    summary: String,
    claims: Vec<String>,
    sprint: u32,
}

fn main() {
    let args = Args::parse();

    let content = match std::fs::read_to_string(&args.input) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("ERROR: cannot read {}: {e}", args.input.display());
            std::process::exit(1);
        }
    };

    // Match section headers: ## I-NNN: Title
    let header_re = Regex::new(r"^## (I-\d+): (.+)$").unwrap();
    // Match **Date**: ... or **Date:** ...
    let date_re = Regex::new(r"^\*\*Date\*\*:?\s*(.+)$").unwrap();
    // Match **Status**: ...
    let status_re = Regex::new(r"^\*\*Status\*\*:?\s*(.+)$").unwrap();
    // Match **Related claims**: ... or **Claims:** ...
    let claims_re = Regex::new(r"^\*\*(Related )?[Cc]laims\*\*:?\s*(.+)$").unwrap();
    // Match claim IDs in text
    let claim_id_re = Regex::new(r"C-\d+").unwrap();

    let mut insights = Vec::new();
    let mut current_id: Option<String> = None;
    let mut current_title = String::new();
    let mut current_date = String::new();
    let mut current_status = String::new();
    let mut current_claims: Vec<String> = Vec::new();
    let mut body_lines: Vec<String> = Vec::new();
    let mut in_body = false;

    let flush = |id: &str,
                 title: &str,
                 date: &str,
                 status: &str,
                 claims: &[String],
                 body: &[String],
                 insights: &mut Vec<Insight>| {
        if id.is_empty() {
            return;
        }
        // Build summary from first non-empty paragraph of body
        let summary: String = body
            .iter()
            .filter(|l| !l.is_empty())
            .take(3)
            .cloned()
            .collect::<Vec<_>>()
            .join(" ");
        let summary = if summary.len() > 500 {
            format!("{}...", &summary[..497])
        } else {
            summary
        };

        // Guess sprint from date
        let sprint = if date.contains("2026-02-08") {
            9 // or 10, approximate
        } else if date.contains("2026-02-07") {
            7
        } else {
            6
        };

        let status_clean = if status.is_empty() {
            "open".to_string()
        } else {
            status.to_lowercase()
        };

        insights.push(Insight {
            id: id.to_string(),
            title: title.to_string(),
            date: date.to_string(),
            status: status_clean,
            summary,
            claims: claims.to_vec(),
            sprint,
        });
    };

    for line in content.lines() {
        if let Some(caps) = header_re.captures(line) {
            // Flush previous insight
            if let Some(ref id) = current_id {
                flush(
                    id,
                    &current_title,
                    &current_date,
                    &current_status,
                    &current_claims,
                    &body_lines,
                    &mut insights,
                );
            }
            current_id = Some(caps[1].to_string());
            current_title = caps[2].to_string();
            current_date.clear();
            current_status.clear();
            current_claims.clear();
            body_lines.clear();
            in_body = false;
            continue;
        }

        if current_id.is_none() {
            continue;
        }

        if let Some(caps) = date_re.captures(line) {
            current_date = caps[1].trim().to_string();
            continue;
        }
        if let Some(caps) = status_re.captures(line) {
            current_status = caps[1].trim().to_string();
            continue;
        }
        if let Some(caps) = claims_re.captures(line) {
            let claims_text = caps[2].trim();
            current_claims = claim_id_re
                .find_iter(claims_text)
                .map(|m| m.as_str().to_string())
                .collect();
            continue;
        }

        // Skip context/related lines
        if line.starts_with("**Context**") {
            continue;
        }

        // After the metadata lines, collect body for summary
        if line == "---" {
            in_body = false;
            continue;
        }
        if !line.starts_with("**") && current_id.is_some() {
            in_body = true;
        }
        if in_body {
            body_lines.push(line.to_string());
        }
    }

    // Flush last insight
    if let Some(ref id) = current_id {
        flush(
            id,
            &current_title,
            &current_date,
            &current_status,
            &current_claims,
            &body_lines,
            &mut insights,
        );
    }

    // Sort by ID
    insights.sort_by(|a, b| {
        let num_a: u32 =
            a.id.strip_prefix("I-")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        let num_b: u32 =
            b.id.strip_prefix("I-")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
        num_a.cmp(&num_b)
    });

    println!("Parsed {} insights", insights.len());
    for ins in &insights {
        println!("  {} - {} [{}]", ins.id, ins.title, ins.status);
    }

    let registry = InsightsRegistry { insight: insights };

    let toml_str = match toml::to_string_pretty(&registry) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("ERROR: TOML serialization failed: {e}");
            std::process::exit(1);
        }
    };

    // Prepend header comment
    let output = format!(
        "# Insights registry -- discoveries and interpretations from the computational census.\n\
         # Each insight cross-references the claims it produced.\n\
         # Auto-generated by migrate-insights from docs/INSIGHTS.md\n\n{toml_str}"
    );

    if let Some(parent) = args.output.parent() {
        std::fs::create_dir_all(parent).ok();
    }

    match std::fs::write(&args.output, &output) {
        Ok(()) => println!("Wrote {} bytes to {}", output.len(), args.output.display()),
        Err(e) => {
            eprintln!("ERROR: cannot write {}: {e}", args.output.display());
            std::process::exit(1);
        }
    }
}
