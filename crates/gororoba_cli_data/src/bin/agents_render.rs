use anyhow::{Context, Result, bail};
use clap::Parser;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Parser, Debug)]
#[command(
    name = "agents-render",
    about = "Render AGENTS.md deterministically from agents.toml"
)]
struct Cli {
    /// Path to the machine-readable source-of-truth file.
    #[arg(long, default_value = "agents.toml")]
    agents: PathBuf,

    /// Path to the rendered AGENTS markdown file.
    #[arg(long, default_value = "AGENTS.md")]
    output: PathBuf,

    /// Check-only mode: fail if AGENTS.md differs from the rendered content.
    #[arg(long)]
    check: bool,
}

#[derive(Debug, Deserialize)]
struct AgentsRoot {
    #[serde(default)]
    agents_md_render: AgentsMdRender,
}

#[derive(Debug, Default, Deserialize)]
struct AgentsMdRender {
    title: String,
    source_comment: String,
    rust_first_heading: String,
    rust_first_intro: String,
    #[serde(default)]
    rust_first_rules: Vec<String>,
    policy_link_heading: String,
    policy_link_body: String,
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("crate must be nested under repo/crates")
        .to_path_buf()
}

fn load_agents(path: &Path) -> Result<AgentsRoot> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("read agents policy {}", path.display()))?;
    toml::from_str(&raw).with_context(|| format!("parse agents policy {}", path.display()))
}

fn render_markdown(config: &AgentsMdRender) -> Result<String> {
    if config.title.trim().is_empty() {
        bail!("agents_md_render.title must not be empty");
    }
    if config.source_comment.trim().is_empty() {
        bail!("agents_md_render.source_comment must not be empty");
    }
    if config.rust_first_heading.trim().is_empty() {
        bail!("agents_md_render.rust_first_heading must not be empty");
    }
    if config.rust_first_intro.trim().is_empty() {
        bail!("agents_md_render.rust_first_intro must not be empty");
    }
    if config.policy_link_heading.trim().is_empty() {
        bail!("agents_md_render.policy_link_heading must not be empty");
    }
    if config.policy_link_body.trim().is_empty() {
        bail!("agents_md_render.policy_link_body must not be empty");
    }

    let mut lines = vec![
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        format!("<!-- Source of truth: {} -->", config.source_comment.trim()),
        String::new(),
        format!("# {}", config.title.trim()),
        String::new(),
        format!("## {}", config.rust_first_heading.trim()),
        String::new(),
        config.rust_first_intro.trim().to_string(),
        String::new(),
    ];

    for rule in &config.rust_first_rules {
        lines.push(rule.trim().to_string());
        lines.push(String::new());
    }

    lines.push(format!("## {}", config.policy_link_heading.trim()));
    lines.push(String::new());
    lines.extend(
        config
            .policy_link_body
            .lines()
            .map(|line| line.trim_end().to_string()),
    );
    lines.push(String::new());

    let rendered = lines.join("\n");
    if !rendered.is_ascii() {
        bail!("rendered AGENTS.md must remain ASCII-only");
    }
    Ok(rendered)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let root = repo_root();
    let agents_path = root.join(&cli.agents);
    let output_path = root.join(&cli.output);

    let config = load_agents(&agents_path)?;
    let rendered = render_markdown(&config.agents_md_render)?;

    if cli.check {
        let current = fs::read_to_string(&output_path)
            .with_context(|| format!("read rendered file {}", output_path.display()))?;
        if current != rendered {
            bail!(
                "AGENTS.md is out of date; run `cargo run -p gororoba_cli_data --bin agents-render --`"
            );
        }
        println!("agents-render: status=ok path={}", output_path.display());
        return Ok(());
    }

    fs::write(&output_path, rendered)
        .with_context(|| format!("write rendered file {}", output_path.display()))?;
    println!("agents-render: wrote {}", output_path.display());
    Ok(())
}
