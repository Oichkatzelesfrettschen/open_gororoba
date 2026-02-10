use clap::{Parser, Subcommand};
use regex::Regex;
use serde::Deserialize;
use std::fs;
use std::path::{Path, PathBuf};

/// Emit non-canonical views from TOML registries (markdown, bibtex, tex, pgfplots, svg, mermaid).
#[derive(Debug, Parser)]
#[command(name = "registry-emit")]
#[command(about = "TOML-first multi-format emitter frontend")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Emit markdown files from artifact scroll TOML registry.
    ArtifactMarkdown(ArtifactMarkdownArgs),
    /// Emit BibTeX from bibliography TOML registry.
    BibliographyBibtex(BibliographyBibtexArgs),
    /// Emit TeX equation report from equation atoms TOML registry.
    EquationsTex(EquationsTexArgs),
    /// Emit PGFPlots from a dataset scroll TOML file.
    DatasetPgfplots(DatasetPgfplotsArgs),
    /// Emit Mermaid text from a TOML graph registry.
    Mermaid(MermaidArgs),
    /// Emit SVG text from a TOML vector registry.
    Svg(SvgArgs),
}

#[derive(Debug, Parser)]
struct ArtifactMarkdownArgs {
    /// Repository root for resolving relative paths.
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    /// Artifact scroll index TOML path.
    #[arg(long, default_value = "registry/artifact_scrolls.toml")]
    index: PathBuf,
    /// Output directory where markdown files are emitted.
    #[arg(long)]
    out_dir: PathBuf,
    /// Optional scroll id filter (e.g., ART-001).
    #[arg(long)]
    id: Option<String>,
    /// Include generated/source-of-truth header.
    #[arg(long, default_value_t = true)]
    with_header: bool,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct BibliographyBibtexArgs {
    /// Bibliography TOML registry path.
    #[arg(long, default_value = "registry/bibliography.toml")]
    input: PathBuf,
    /// Output .bib file path.
    #[arg(long)]
    output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct EquationsTexArgs {
    /// Equation atoms TOML registry path.
    #[arg(long, default_value = "registry/knowledge/equation_atoms.toml")]
    input: PathBuf,
    /// Output .tex path.
    #[arg(long)]
    output: PathBuf,
    /// Optional domain filter, e.g. algebra or cosmology.
    #[arg(long)]
    domain: Option<String>,
    /// Maximum equation count to emit.
    #[arg(long, default_value_t = 500usize)]
    max_equations: usize,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DatasetPgfplotsArgs {
    /// Dataset scroll TOML path.
    #[arg(long)]
    input: PathBuf,
    /// Output .tex path.
    #[arg(long)]
    output: PathBuf,
    /// X column name (defaults to first numeric-rich column).
    #[arg(long)]
    x_col: Option<String>,
    /// Y column name (defaults to second numeric-rich column).
    #[arg(long)]
    y_col: Option<String>,
    /// Maximum points to emit.
    #[arg(long, default_value_t = 2000usize)]
    max_points: usize,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct MermaidArgs {
    /// Mermaid graph TOML path.
    #[arg(long)]
    input: PathBuf,
    /// Output .mmd path.
    #[arg(long)]
    output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct SvgArgs {
    /// SVG vector TOML path.
    #[arg(long)]
    input: PathBuf,
    /// Output .svg path.
    #[arg(long)]
    output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Deserialize)]
struct ArtifactScrollIndex {
    scroll: Vec<ArtifactScrollIndexRow>,
}

#[derive(Debug, Deserialize)]
struct ArtifactScrollIndexRow {
    id: String,
    source_markdown: String,
    scroll_path: String,
}

#[derive(Debug, Deserialize)]
struct ArtifactScrollDoc {
    section: Option<Vec<ArtifactSection>>,
}

#[derive(Debug, Deserialize)]
struct ArtifactSection {
    title: Option<String>,
    level: Option<i64>,
    body_text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct BibliographyRegistry {
    entry: Vec<BibEntry>,
}

#[derive(Debug, Deserialize)]
struct BibEntry {
    id: String,
    citation_markdown: String,
    section: Option<String>,
    urls: Option<Vec<String>>,
    dois: Option<Vec<String>>,
    notes: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct EquationRegistry {
    atom: Vec<EquationAtom>,
}

#[derive(Debug, Deserialize)]
struct EquationAtom {
    id: String,
    expression: String,
    source_path: Option<String>,
    source_line: Option<usize>,
    domain_hint: Option<String>,
}

#[derive(Debug, Deserialize)]
struct DatasetToml {
    dataset: DatasetRecord,
}

#[derive(Debug, Deserialize)]
struct DatasetRecord {
    id: String,
    source_csv: String,
    header: Vec<String>,
    rows: Option<Vec<Vec<String>>>,
}

#[derive(Debug, Deserialize)]
struct MermaidRegistry {
    diagram: MermaidMeta,
    node: Option<Vec<MermaidNode>>,
    edge: Option<Vec<MermaidEdge>>,
}

#[derive(Debug, Deserialize)]
struct MermaidMeta {
    kind: Option<String>,
    direction: Option<String>,
    title: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MermaidNode {
    id: String,
    label: Option<String>,
}

#[derive(Debug, Deserialize)]
struct MermaidEdge {
    from: String,
    to: String,
    label: Option<String>,
    style: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SvgRegistry {
    svg: SvgMeta,
    rect: Option<Vec<SvgRect>>,
    line: Option<Vec<SvgLine>>,
    circle: Option<Vec<SvgCircle>>,
    path: Option<Vec<SvgPath>>,
    text: Option<Vec<SvgText>>,
}

#[derive(Debug, Deserialize)]
struct SvgMeta {
    width: u32,
    height: u32,
    view_box: Option<String>,
    background: Option<String>,
}

#[derive(Debug, Deserialize)]
struct SvgRect {
    x: f64,
    y: f64,
    width: f64,
    height: f64,
    fill: Option<String>,
    stroke: Option<String>,
    stroke_width: Option<f64>,
    rx: Option<f64>,
    ry: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SvgLine {
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    stroke: Option<String>,
    stroke_width: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SvgCircle {
    cx: f64,
    cy: f64,
    r: f64,
    fill: Option<String>,
    stroke: Option<String>,
    stroke_width: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SvgPath {
    d: String,
    fill: Option<String>,
    stroke: Option<String>,
    stroke_width: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct SvgText {
    x: f64,
    y: f64,
    value: String,
    fill: Option<String>,
    font_size: Option<f64>,
    font_family: Option<String>,
}

fn read_toml<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<T, String> {
    let text =
        fs::read_to_string(path).map_err(|err| format!("read {}: {}", path.display(), err))?;
    toml::from_str(&text).map_err(|err| format!("parse {}: {}", path.display(), err))
}

fn ensure_ascii(text: &str, context: &str) -> Result<(), String> {
    let bad: Vec<char> = text.chars().filter(|ch| (*ch as u32) > 127).collect();
    if bad.is_empty() {
        return Ok(());
    }
    let sample: String = bad.into_iter().take(20).collect();
    Err(format!("non-ASCII output in {}: {:?}", context, sample))
}

fn write_output(path: &Path, text: &str, allow_unicode: bool) -> Result<(), String> {
    if !allow_unicode {
        ensure_ascii(text, &path.display().to_string())?;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {}", parent.display(), err))?;
    }
    fs::write(path, text).map_err(|err| format!("write {}: {}", path.display(), err))
}

fn level_prefix(level: i64) -> &'static str {
    match level {
        i64::MIN..=1 => "#",
        2 => "##",
        3 => "###",
        4 => "####",
        5 => "#####",
        _ => "######",
    }
}

fn emit_artifact_markdown(args: ArtifactMarkdownArgs) -> Result<(), String> {
    let index_path = args.repo_root.join(&args.index);
    let index: ArtifactScrollIndex = read_toml(&index_path)?;
    let mut written = 0usize;

    for row in index.scroll {
        if let Some(filter_id) = &args.id {
            if row.id != *filter_id {
                continue;
            }
        }
        let scroll_path = args.repo_root.join(&row.scroll_path);
        let scroll: ArtifactScrollDoc = read_toml(&scroll_path)?;
        let out_path = args.out_dir.join(&row.source_markdown);

        let mut lines: Vec<String> = Vec::new();
        if args.with_header {
            lines.push("<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string());
            lines.push("<!-- Source of truth: registry/artifact_scrolls.toml -->".to_string());
            lines.push(String::new());
        }

        let mut has_body = false;
        if let Some(sections) = scroll.section {
            for section in sections {
                let title = section.title.unwrap_or_else(|| "(root)".to_string());
                let body = section.body_text.unwrap_or_default();
                if title != "(root)" {
                    let level = section.level.unwrap_or(2);
                    lines.push(format!("{} {}", level_prefix(level), title));
                    lines.push(String::new());
                }
                if !body.trim().is_empty() {
                    lines.extend(body.lines().map(str::to_string));
                    lines.push(String::new());
                    has_body = true;
                }
            }
        }

        if !has_body {
            lines.push("# Empty Artifact Scroll".to_string());
            lines.push(String::new());
            lines.push(format!(
                "No section body_text is present in `{}` for `{}`.",
                row.scroll_path, row.id
            ));
            lines.push(String::new());
        }

        let rendered = lines.join("\n");
        write_output(&out_path, &rendered, args.allow_unicode)?;
        written += 1;
    }

    if written == 0 {
        return Err("no artifact markdown files were emitted (check --id filter)".to_string());
    }
    println!(
        "Emitted {} markdown file(s) from artifact scroll registry into {}.",
        written,
        args.out_dir.display()
    );
    Ok(())
}

fn strip_markdown_marks(input: &str) -> String {
    let mut out = input.replace("**", "");
    out = out.replace('*', "");
    out = out.replace('`', "");
    out = out.replace('[', "");
    out = out.replace(']', "");
    out = out.replace('(', "");
    out = out.replace(')', "");
    out
}

fn bibtex_escape(input: &str) -> String {
    input
        .replace('\\', "\\\\")
        .replace('{', "\\{")
        .replace('}', "\\}")
        .replace('"', "\\\"")
}

fn extract_year(text: &str) -> Option<String> {
    let re = Regex::new(r"\b(19|20)\d{2}\b").ok()?;
    re.find(text).map(|m| m.as_str().to_string())
}

fn extract_author(text: &str) -> Option<String> {
    let re = Regex::new(r"\*\*([^*]+)\*\*").ok()?;
    re.captures(text)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().trim().to_string())
}

fn extract_title(text: &str) -> Option<String> {
    let re = Regex::new(r"\*([^*]+)\*").ok()?;
    re.captures(text)
        .and_then(|cap| cap.get(1))
        .map(|m| m.as_str().trim().to_string())
}

fn emit_bibliography_bibtex(args: BibliographyBibtexArgs) -> Result<(), String> {
    let registry: BibliographyRegistry = read_toml(&args.input)?;
    let mut out = String::new();
    out.push_str("% Auto-generated by registry-emit bibliography-bibtex\n");
    out.push_str("% Source of truth: registry/bibliography.toml\n\n");

    for entry in registry.entry {
        let key = entry.id.to_lowercase().replace('-', "_");
        let author =
            extract_author(&entry.citation_markdown).unwrap_or_else(|| "Unknown".to_string());
        let title = extract_title(&entry.citation_markdown)
            .unwrap_or_else(|| strip_markdown_marks(&entry.citation_markdown));
        let year = extract_year(&entry.citation_markdown).unwrap_or_else(|| "unknown".to_string());
        let doi = entry
            .dois
            .as_deref()
            .and_then(|values| values.first())
            .cloned()
            .unwrap_or_default();
        let url = entry
            .urls
            .as_deref()
            .and_then(|values| values.first())
            .cloned()
            .unwrap_or_default();
        let section = entry.section.unwrap_or_else(|| "Unscoped".to_string());
        let notes = entry.notes.unwrap_or_default().join(" | ");

        out.push_str(&format!("@misc{{{},\n", key));
        out.push_str(&format!("  author = {{{}}},\n", bibtex_escape(&author)));
        out.push_str(&format!("  title = {{{}}},\n", bibtex_escape(&title)));
        out.push_str(&format!("  year = {{{}}},\n", bibtex_escape(&year)));
        out.push_str(&format!("  keywords = {{{}}},\n", bibtex_escape(&section)));
        if !doi.is_empty() {
            out.push_str(&format!("  doi = {{{}}},\n", bibtex_escape(&doi)));
        }
        if !url.is_empty() {
            out.push_str(&format!("  url = {{{}}},\n", bibtex_escape(&url)));
        }
        if !notes.is_empty() {
            out.push_str(&format!("  note = {{{}}},\n", bibtex_escape(&notes)));
        }
        out.push_str("}\n\n");
    }

    write_output(&args.output, &out, args.allow_unicode)?;
    println!(
        "Emitted BibTeX bibliography from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn tex_escape(input: &str) -> String {
    input
        .replace('\\', "\\textbackslash{}")
        .replace('&', "\\&")
        .replace('%', "\\%")
        .replace('$', "\\$")
        .replace('#', "\\#")
        .replace('_', "\\_")
        .replace('{', "\\{")
        .replace('}', "\\}")
}

fn emit_equations_tex(args: EquationsTexArgs) -> Result<(), String> {
    let registry: EquationRegistry = read_toml(&args.input)?;
    let domain_filter = args.domain.as_deref().map(str::to_lowercase);
    let mut selected: Vec<EquationAtom> = registry
        .atom
        .into_iter()
        .filter(|row| {
            if let Some(filter) = &domain_filter {
                return row
                    .domain_hint
                    .as_deref()
                    .map(|value| value.to_lowercase() == *filter)
                    .unwrap_or(false);
            }
            true
        })
        .take(args.max_equations)
        .collect();
    selected.sort_by(|a, b| a.id.cmp(&b.id));

    if selected.is_empty() {
        return Err("no equations matched the selected filter".to_string());
    }

    let mut out = String::new();
    out.push_str("% Auto-generated by registry-emit equations-tex\n");
    out.push_str("% Source of truth: registry/knowledge/equation_atoms.toml\n\n");
    out.push_str("\\section*{Equation Atoms}\\label{sec:equation-atoms}\n\n");

    for eq in selected {
        out.push_str(&format!(
            "\\subsection*{{{}}}\n",
            tex_escape(&format!(
                "{} ({})",
                eq.id,
                eq.domain_hint.unwrap_or_else(|| "cross_domain".to_string())
            ))
        ));
        out.push_str("\\[\n");
        out.push_str(&eq.expression);
        out.push_str("\n\\]\n");
        let source_path = eq.source_path.unwrap_or_default();
        let source_line = eq.source_line.unwrap_or(0);
        out.push_str(&format!(
            "\\noindent\\texttt{{Source: {}:{}}}\n\n",
            tex_escape(&source_path),
            source_line
        ));
    }

    write_output(&args.output, &out, args.allow_unicode)?;
    println!(
        "Emitted TeX equation report from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn numeric_score(rows: &[Vec<String>], col_idx: usize, max_scan: usize) -> usize {
    rows.iter()
        .take(max_scan)
        .filter_map(|row| row.get(col_idx))
        .filter(|value| value.trim().parse::<f64>().is_ok())
        .count()
}

fn emit_dataset_pgfplots(args: DatasetPgfplotsArgs) -> Result<(), String> {
    let dataset: DatasetToml = read_toml(&args.input)?;
    let rows = dataset.dataset.rows.unwrap_or_default();
    if rows.is_empty() {
        return Err(format!(
            "dataset {} has no rows in {}",
            dataset.dataset.id,
            args.input.display()
        ));
    }
    let headers = dataset.dataset.header;
    if headers.len() < 2 {
        return Err("dataset header requires at least 2 columns".to_string());
    }

    let x_idx = if let Some(name) = &args.x_col {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("x_col '{}' not found in dataset header", name))?
    } else {
        (0..headers.len())
            .max_by_key(|idx| numeric_score(&rows, *idx, 2000))
            .ok_or_else(|| "cannot determine default x column".to_string())?
    };

    let y_idx = if let Some(name) = &args.y_col {
        headers
            .iter()
            .position(|h| h == name)
            .ok_or_else(|| format!("y_col '{}' not found in dataset header", name))?
    } else {
        let mut candidates: Vec<(usize, usize)> = (0..headers.len())
            .filter(|idx| *idx != x_idx)
            .map(|idx| (idx, numeric_score(&rows, idx, 2000)))
            .collect();
        candidates.sort_by_key(|item| std::cmp::Reverse(item.1));
        candidates
            .first()
            .map(|(idx, _)| *idx)
            .ok_or_else(|| "cannot determine default y column".to_string())?
    };

    if x_idx == y_idx {
        return Err("x and y columns must be distinct".to_string());
    }

    let mut points: Vec<(f64, f64)> = Vec::new();
    for row in rows {
        if let (Some(x), Some(y)) = (row.get(x_idx), row.get(y_idx)) {
            if let (Ok(xn), Ok(yn)) = (x.trim().parse::<f64>(), y.trim().parse::<f64>()) {
                points.push((xn, yn));
            }
        }
        if points.len() >= args.max_points {
            break;
        }
    }
    if points.is_empty() {
        return Err("no numeric points found for selected x/y columns".to_string());
    }

    let x_label = headers
        .get(x_idx)
        .cloned()
        .unwrap_or_else(|| "x".to_string());
    let y_label = headers
        .get(y_idx)
        .cloned()
        .unwrap_or_else(|| "y".to_string());

    let mut out = String::new();
    out.push_str("% Auto-generated by registry-emit dataset-pgfplots\n");
    out.push_str(&format!(
        "% Source dataset TOML: {}\n",
        args.input.display()
    ));
    out.push_str(&format!("% Source CSV: {}\n\n", dataset.dataset.source_csv));
    out.push_str("\\begin{tikzpicture}\n");
    out.push_str("\\begin{axis}[\n");
    out.push_str(&format!(
        "  title={{{}}},\n",
        tex_escape(&dataset.dataset.id)
    ));
    out.push_str(&format!("  xlabel={{{}}},\n", tex_escape(&x_label)));
    out.push_str(&format!("  ylabel={{{}}},\n", tex_escape(&y_label)));
    out.push_str("  grid=both,\n");
    out.push_str("]\n");
    out.push_str("\\addplot+[mark=none] coordinates {\n");
    for (x, y) in points {
        out.push_str(&format!("  ({:.12}, {:.12})\n", x, y));
    }
    out.push_str("};\n");
    out.push_str("\\end{axis}\n");
    out.push_str("\\end{tikzpicture}\n");

    write_output(&args.output, &out, args.allow_unicode)?;
    println!(
        "Emitted PGFPlots from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn mermaid_arrow(style: Option<&str>) -> &'static str {
    match style.unwrap_or_default().to_ascii_lowercase().as_str() {
        "dotted" => "-.->",
        "thick" => "==>",
        "open" => "---",
        _ => "-->",
    }
}

fn mermaid_escape(input: &str) -> String {
    input.replace('"', "\\\"")
}

fn emit_mermaid(args: MermaidArgs) -> Result<(), String> {
    let model: MermaidRegistry = read_toml(&args.input)?;
    let kind = model
        .diagram
        .kind
        .unwrap_or_else(|| "flowchart".to_string())
        .to_ascii_lowercase();
    let direction = model
        .diagram
        .direction
        .unwrap_or_else(|| "TD".to_string())
        .to_ascii_uppercase();

    let mut out = String::new();
    out.push_str("%% Auto-generated by registry-emit mermaid\n");
    out.push_str(&format!("%% Source TOML: {}\n", args.input.display()));
    if let Some(title) = model.diagram.title {
        out.push_str("---\n");
        out.push_str(&format!("title: {}\n", mermaid_escape(&title)));
        out.push_str("---\n");
    }
    out.push('\n');

    match kind.as_str() {
        "graph" | "flowchart" => {
            out.push_str(&format!("flowchart {}\n", direction));
            let nodes = model.node.unwrap_or_default();
            for node in nodes {
                let label = node.label.unwrap_or_else(|| node.id.clone());
                out.push_str(&format!("  {}[\"{}\"]\n", node.id, mermaid_escape(&label)));
            }
            let edges = model.edge.unwrap_or_default();
            for edge in edges {
                let arrow = mermaid_arrow(edge.style.as_deref());
                if let Some(label) = edge.label {
                    out.push_str(&format!(
                        "  {} {}|{}| {}\n",
                        edge.from,
                        arrow,
                        mermaid_escape(&label),
                        edge.to
                    ));
                } else {
                    out.push_str(&format!("  {} {} {}\n", edge.from, arrow, edge.to));
                }
            }
        }
        "sequencediagram" => {
            out.push_str("sequenceDiagram\n");
            let edges = model.edge.unwrap_or_default();
            for edge in edges {
                let label = edge.label.unwrap_or_default();
                out.push_str(&format!(
                    "  {}->>{}: {}\n",
                    edge.from,
                    edge.to,
                    mermaid_escape(&label)
                ));
            }
        }
        _ => {
            return Err(format!(
                "unsupported diagram.kind='{}' (supported: flowchart, graph, sequenceDiagram)",
                kind
            ));
        }
    }

    write_output(&args.output, &out, args.allow_unicode)?;
    println!(
        "Emitted Mermaid diagram from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn svg_attr(value: Option<&str>, fallback: &str) -> String {
    value.unwrap_or(fallback).to_string()
}

fn emit_svg(args: SvgArgs) -> Result<(), String> {
    let model: SvgRegistry = read_toml(&args.input)?;
    let view_box = model
        .svg
        .view_box
        .unwrap_or_else(|| format!("0 0 {} {}", model.svg.width, model.svg.height));
    let mut out = String::new();
    out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    out.push_str("<!-- Auto-generated by registry-emit svg -->\n");
    out.push_str(&format!("<!-- Source TOML: {} -->\n", args.input.display()));
    out.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{}\" height=\"{}\" viewBox=\"{}\">\n",
        model.svg.width, model.svg.height, view_box
    ));

    if let Some(bg) = model.svg.background {
        out.push_str(&format!(
            "  <rect x=\"0\" y=\"0\" width=\"{}\" height=\"{}\" fill=\"{}\" />\n",
            model.svg.width, model.svg.height, bg
        ));
    }

    if let Some(rects) = model.rect {
        for rect in rects {
            out.push_str(&format!(
                "  <rect x=\"{:.6}\" y=\"{:.6}\" width=\"{:.6}\" height=\"{:.6}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.6}\" rx=\"{:.6}\" ry=\"{:.6}\" />\n",
                rect.x,
                rect.y,
                rect.width,
                rect.height,
                svg_attr(rect.fill.as_deref(), "none"),
                svg_attr(rect.stroke.as_deref(), "none"),
                rect.stroke_width.unwrap_or(0.0),
                rect.rx.unwrap_or(0.0),
                rect.ry.unwrap_or(0.0)
            ));
        }
    }

    if let Some(lines) = model.line {
        for line in lines {
            out.push_str(&format!(
                "  <line x1=\"{:.6}\" y1=\"{:.6}\" x2=\"{:.6}\" y2=\"{:.6}\" stroke=\"{}\" stroke-width=\"{:.6}\" />\n",
                line.x1,
                line.y1,
                line.x2,
                line.y2,
                svg_attr(line.stroke.as_deref(), "#ffffff"),
                line.stroke_width.unwrap_or(1.0)
            ));
        }
    }

    if let Some(circles) = model.circle {
        for circle in circles {
            out.push_str(&format!(
                "  <circle cx=\"{:.6}\" cy=\"{:.6}\" r=\"{:.6}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.6}\" />\n",
                circle.cx,
                circle.cy,
                circle.r,
                svg_attr(circle.fill.as_deref(), "none"),
                svg_attr(circle.stroke.as_deref(), "none"),
                circle.stroke_width.unwrap_or(0.0)
            ));
        }
    }

    if let Some(paths) = model.path {
        for path in paths {
            out.push_str(&format!(
                "  <path d=\"{}\" fill=\"{}\" stroke=\"{}\" stroke-width=\"{:.6}\" />\n",
                path.d.replace('"', "&quot;"),
                svg_attr(path.fill.as_deref(), "none"),
                svg_attr(path.stroke.as_deref(), "none"),
                path.stroke_width.unwrap_or(1.0)
            ));
        }
    }

    if let Some(text_rows) = model.text {
        for row in text_rows {
            out.push_str(&format!(
                "  <text x=\"{:.6}\" y=\"{:.6}\" fill=\"{}\" font-size=\"{:.6}\" font-family=\"{}\">{}</text>\n",
                row.x,
                row.y,
                svg_attr(row.fill.as_deref(), "#ffffff"),
                row.font_size.unwrap_or(12.0),
                svg_attr(row.font_family.as_deref(), "sans-serif"),
                row.value
                    .replace('&', "&amp;")
                    .replace('<', "&lt;")
                    .replace('>', "&gt;")
            ));
        }
    }

    out.push_str("</svg>\n");
    write_output(&args.output, &out, args.allow_unicode)?;
    println!(
        "Emitted SVG from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::ArtifactMarkdown(args) => emit_artifact_markdown(args),
        Commands::BibliographyBibtex(args) => emit_bibliography_bibtex(args),
        Commands::EquationsTex(args) => emit_equations_tex(args),
        Commands::DatasetPgfplots(args) => emit_dataset_pgfplots(args),
        Commands::Mermaid(args) => emit_mermaid(args),
        Commands::Svg(args) => emit_svg(args),
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
}
