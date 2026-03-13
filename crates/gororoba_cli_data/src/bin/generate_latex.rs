//! Generate LaTeX appendices from the canonical SQLite control plane.
//!
//! Reads DB-rendered compatibility text for claims, insights, and experiments
//! from `registry/canonical/control_plane.sqlite3` and generates:
//! - `docs/latex/claims_appendix.tex`: All claims as a LaTeX longtable
//! - `docs/latex/insights_appendix.tex`: All insights as LaTeX sections
//! - `docs/latex/experiments_appendix.tex`: All experiments as a LaTeX longtable
//! - `docs/latex/thesis_results_table.tex`: Thesis summary + T2 detail (--thesis-tables)
//! - `docs/latex/reproducibility_appendix.tex`: Reproducibility appendix (--repro-appendix)

use anyhow::{Context, Result};
use std::path::PathBuf;

use clap::Parser;
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};

/// Generate LaTeX appendices from the canonical SQLite control plane.
#[derive(Parser)]
#[command(name = "generate-latex")]
struct Args {
    /// Canonical SQLite control-plane database.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    db: PathBuf,

    /// Compatibility registry directory used only as a legacy fallback.
    #[arg(long, default_value = "registry")]
    dir: PathBuf,

    /// Output directory for generated .tex files
    #[arg(long, default_value = "docs/latex")]
    output_dir: PathBuf,

    /// Generate thesis results tables from synthesis evidence
    #[arg(long)]
    thesis_tables: bool,

    /// Generate reproducibility appendix
    #[arg(long)]
    repro_appendix: bool,

    /// Path to evidence manifest (for reproducibility appendix)
    #[arg(long, default_value = "data/evidence/MANIFEST.toml")]
    manifest: PathBuf,

    /// Hardware description string
    #[arg(
        long,
        default_value = "AMD Ryzen 9 7950X / 64 GB DDR5 / NVIDIA RTX 4090"
    )]
    hardware: String,
}

#[derive(serde::Deserialize)]
struct ClaimsRegistry {
    claim: Vec<ClaimEntry>,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct ClaimEntry {
    id: String,
    #[serde(default)]
    statement: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    where_stated: Option<String>,
    #[serde(default)]
    confidence: Option<String>,
    #[serde(default)]
    phase: Option<String>,
    #[serde(default)]
    last_verified: Option<String>,
    #[serde(default)]
    what_would_verify_refute: Option<String>,
    #[serde(default)]
    supporting_evidence: Option<Vec<String>>,
    #[serde(default)]
    verification_method: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    sprint: Option<toml::Value>,
    #[serde(default)]
    dependencies: Option<Vec<String>>,
    #[serde(default)]
    claims: Option<Vec<String>>,
    #[serde(default)]
    insights: Option<Vec<String>>,
    #[serde(default)]
    status_note: Option<String>,
}

#[derive(serde::Deserialize)]
struct ExperimentsRegistry {
    experiment: Vec<ExperimentEntry>,
}

#[derive(serde::Deserialize)]
struct ExperimentEntry {
    id: String,
    title: String,
    #[serde(default)]
    binary: Option<String>,
    method: String,
    #[serde(default)]
    run: Option<String>,
    #[serde(default)]
    deterministic: Option<bool>,
    #[serde(default)]
    gpu: Option<bool>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    claims: Option<Vec<String>>,
}

#[derive(serde::Deserialize)]
struct InsightsRegistry {
    insight: Vec<InsightEntry>,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize)]
#[allow(dead_code)]
struct InsightEntry {
    id: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    date: Option<String>,
    #[serde(default)]
    status: Option<String>,
    #[serde(default)]
    summary: Option<String>,
    /// Newer insights use `insight` instead of `summary`.
    #[serde(default)]
    insight: Option<String>,
    #[serde(default)]
    claims: Vec<String>,
    #[serde(default)]
    sprint: Option<toml::Value>,
    #[serde(default)]
    supporting_evidence: Option<Vec<String>>,
    #[serde(default)]
    related_claims: Option<Vec<String>>,
    #[serde(default)]
    experimental_support: Option<Vec<String>>,
    #[serde(default)]
    confidence: Option<String>,
    #[serde(default)]
    verified_date: Option<String>,
    #[serde(default)]
    phase: Option<String>,
}

impl InsightEntry {
    fn body(&self) -> &str {
        self.summary
            .as_deref()
            .or(self.insight.as_deref())
            .unwrap_or("(no summary)")
    }
}

// ---------------------------------------------------------------------------
// Thesis synthesis summary structs
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct SynthesisSummary {
    thesis: Vec<ThesisSummaryEntry>,
}

#[derive(serde::Deserialize)]
struct ThesisSummaryEntry {
    id: u32,
    label: String,
    metric: f64,
    threshold: f64,
    pass: bool,
}

#[derive(serde::Deserialize)]
struct T2PowerLawData {
    #[serde(default)]
    series: Vec<T2SeriesEntry>,
}

#[derive(serde::Deserialize)]
struct T2SeriesEntry {
    alpha: f64,
    beta: f64,
    power_index: f64,
    #[serde(default)]
    non_newtonian: Option<bool>,
    #[serde(default)]
    slope_ratio: Option<f64>,
}

// ---------------------------------------------------------------------------
// Manifest structs for reproducibility appendix
// ---------------------------------------------------------------------------

#[derive(serde::Deserialize)]
struct ManifestFile {
    #[serde(default)]
    metadata: ManifestMetadata,
    #[serde(default)]
    artifact: Vec<ManifestArtifact>,
    #[serde(default)]
    summary: ManifestSummary,
}

// Constructed by serde::Deserialize; not all fields read by this binary.
#[derive(serde::Deserialize, Default)]
#[allow(dead_code)]
struct ManifestMetadata {
    #[serde(default)]
    generated_unix: u64,
    #[serde(default)]
    git_rev: String,
    #[serde(default)]
    evidence_dir: String,
}

#[derive(serde::Deserialize, Default)]
struct ManifestArtifact {
    #[serde(default)]
    path: String,
    #[serde(default)]
    sha256: String,
    #[serde(default)]
    status: String,
    #[serde(default)]
    size_bytes: u64,
}

#[derive(serde::Deserialize, Default)]
struct ManifestSummary {
    #[serde(default)]
    total: usize,
    #[serde(default)]
    present: usize,
    #[serde(default)]
    missing: usize,
}

fn escape_latex(s: &str) -> String {
    s.replace('\\', r"\textbackslash{}")
        .replace('&', r"\&")
        .replace('%', r"\%")
        .replace('$', r"\$")
        .replace('#', r"\#")
        .replace('_', r"\_")
        .replace('{', r"\{")
        .replace('}', r"\}")
        .replace('~', r"\textasciitilde{}")
        .replace('^', r"\textasciicircum{}")
}

fn truncate_for_table(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        escape_latex(s)
    } else {
        let truncated = &s[..max_len.min(s.len())];
        // Find last space to avoid cutting mid-word
        let cut = truncated.rfind(' ').unwrap_or(max_len);
        format!("{}...", escape_latex(&s[..cut]))
    }
}

fn generate_claims_appendix(claims: &[ClaimEntry]) -> String {
    let mut out = String::new();
    out.push_str("% Auto-generated by generate-latex from the canonical SQLite control plane\n");
    out.push_str("% DO NOT EDIT -- regenerate with: cargo run --release --bin generate-latex\n\n");
    out.push_str("\\section{Claims Evidence Matrix}\\label{sec:claims-appendix}\n\n");
    out.push_str(&format!("Total claims: {}.  Statuses: ", claims.len()));
    // Count by status
    let mut status_counts: std::collections::BTreeMap<&str, usize> =
        std::collections::BTreeMap::new();
    for claim in claims {
        let status = claim.status.as_deref().unwrap_or("unknown");
        *status_counts.entry(status).or_insert(0) += 1;
    }
    let counts: Vec<String> = status_counts
        .iter()
        .map(|(s, n)| format!("{} ({})", escape_latex(s), n))
        .collect();
    out.push_str(&counts.join(", "));
    out.push_str(".\n\n");
    out.push_str("\\begin{longtable}{|l|p{7cm}|l|l|l|}\n");
    out.push_str("\\hline\n");
    out.push_str("\\textbf{ID} & \\textbf{Statement} & \\textbf{Status} & \\textbf{Conf.} & \\textbf{Source} \\\\\n");
    out.push_str("\\hline\n");
    out.push_str("\\endfirsthead\n");
    out.push_str("\\hline\n");
    out.push_str("\\textbf{ID} & \\textbf{Statement} & \\textbf{Status} & \\textbf{Conf.} & \\textbf{Source} \\\\\n");
    out.push_str("\\hline\n");
    out.push_str("\\endhead\n");

    for claim in claims {
        let stmt = claim.statement.as_deref().unwrap_or("(no statement)");
        let status = claim.status.as_deref().unwrap_or("--");
        let conf = claim.confidence.as_deref().unwrap_or("--");
        let source = claim.where_stated.as_deref().unwrap_or("--");
        out.push_str(&format!(
            "{} & {} & {} & {} & {} \\\\\n\\hline\n",
            escape_latex(&claim.id),
            truncate_for_table(stmt, 100),
            escape_latex(status),
            escape_latex(conf),
            truncate_for_table(source, 35),
        ));
    }

    out.push_str("\\end{longtable}\n");
    out
}

fn generate_experiments_appendix(experiments: &[ExperimentEntry]) -> String {
    let mut out = String::new();
    out.push_str("% Auto-generated by generate-latex from the canonical SQLite control plane\n");
    out.push_str("% DO NOT EDIT -- regenerate with: cargo run --release --bin generate-latex\n\n");
    out.push_str("\\section{Experiments Portfolio}\\label{sec:experiments-appendix}\n\n");
    out.push_str(&format!(
        "Total experiments: {}.\\\\[0.5em]\n\n",
        experiments.len()
    ));

    for exp in experiments {
        let status = exp.status.as_deref().unwrap_or("unknown");
        let binary = exp.binary.as_deref().unwrap_or("--");
        let det = if exp.deterministic.unwrap_or(false) {
            "yes"
        } else {
            "no"
        };
        let gpu_flag = if exp.gpu.unwrap_or(false) {
            "yes"
        } else {
            "no"
        };
        let claims_str = match &exp.claims {
            Some(c) if !c.is_empty() => c.join(", "),
            _ => "none".to_string(),
        };

        out.push_str(&format!(
            "\\subsection{{{}: {}}}\\label{{sec:{}}}\n\n",
            escape_latex(&exp.id),
            escape_latex(&exp.title),
            exp.id.to_lowercase().replace('-', ""),
        ));
        out.push_str(&format!(
            "\\textbf{{Binary:}} \\texttt{{{}}} \\quad \
             \\textbf{{Status:}} {} \\quad \
             \\textbf{{Deterministic:}} {} \\quad \
             \\textbf{{GPU:}} {}\n\n",
            escape_latex(binary),
            escape_latex(status),
            det,
            gpu_flag,
        ));
        out.push_str(&format!(
            "\\textbf{{Method:}} {}\n\n",
            escape_latex(&exp.method)
        ));
        if let Some(run) = &exp.run {
            out.push_str(&format!(
                "\\textbf{{Run:}} \\texttt{{{}}}\n\n",
                escape_latex(run)
            ));
        }
        out.push_str(&format!(
            "\\textbf{{Claims:}} {}\n\n",
            escape_latex(&claims_str)
        ));
    }

    out
}

fn generate_insights_appendix(insights: &[InsightEntry]) -> String {
    let mut out = String::new();
    out.push_str("% Auto-generated by generate-latex from the canonical SQLite control plane\n");
    out.push_str("% DO NOT EDIT -- regenerate with: cargo run --release --bin generate-latex\n\n");
    out.push_str("\\section{Research Insights}\\label{sec:insights-appendix}\n\n");

    for insight in insights {
        let status = insight.status.as_deref().unwrap_or("open");
        let date = insight.date.as_deref().unwrap_or("unknown");
        let claims_str = if insight.claims.is_empty() {
            "none".to_string()
        } else {
            insight.claims.join(", ")
        };

        let title = insight.title.as_deref().unwrap_or("(untitled)");
        out.push_str(&format!(
            "\\subsection{{{}: {}}}\\label{{sec:{}}}\n\n",
            escape_latex(&insight.id),
            escape_latex(title),
            insight.id.to_lowercase().replace('-', ""),
        ));
        out.push_str(&format!(
            "\\textbf{{Date:}} {} \\quad \\textbf{{Status:}} {} \\quad \\textbf{{Claims:}} {}\n\n",
            escape_latex(date),
            escape_latex(status),
            escape_latex(&claims_str),
        ));
        out.push_str(&format!("{}\n\n", escape_latex(insight.body())));
    }

    out
}

// ---------------------------------------------------------------------------
// Thesis results table generation (T-016)
// ---------------------------------------------------------------------------

fn generate_thesis_results_table(
    summary: &SynthesisSummary,
    t2_data: Option<&T2PowerLawData>,
) -> String {
    let mut out = String::new();

    out.push_str("% Auto-generated by generate-latex --thesis-tables\n");
    out.push_str("\\section{Thesis Results Summary}\n\\label{sec:thesis-results}\n\n");

    // Main summary table
    out.push_str("\\begin{longtable}{c l r r c}\n");
    out.push_str("\\caption{Summary of thesis gate evaluations.}\\label{tab:thesis-summary}\\\\\n");
    out.push_str("\\toprule\n");
    out.push_str("\\textbf{Thesis} & \\textbf{Label} & \\textbf{Metric} & \\textbf{Threshold} & \\textbf{Result} \\\\\n");
    out.push_str("\\midrule\n");
    out.push_str("\\endfirsthead\n");
    out.push_str("\\toprule\n");
    out.push_str("\\textbf{Thesis} & \\textbf{Label} & \\textbf{Metric} & \\textbf{Threshold} & \\textbf{Result} \\\\\n");
    out.push_str("\\midrule\n");
    out.push_str("\\endhead\n");

    for t in &summary.thesis {
        let result_str = if t.pass { "\\textbf{PASS}" } else { "FAIL" };
        out.push_str(&format!(
            "T-{} & {} & {:.6} & {:.6} & {} \\\\\n",
            t.id,
            escape_latex(&t.label),
            t.metric,
            t.threshold,
            result_str,
        ));
    }

    out.push_str("\\bottomrule\n");
    out.push_str("\\end{longtable}\n\n");

    // T2 non-Newtonian sweep sub-table
    if let Some(data) = t2_data.filter(|d| !d.series.is_empty()) {
        out.push_str("\\subsection{T2 Non-Newtonian Viscosity Sweep}\n\\label{sec:t2-sweep}\n\n");
        out.push_str("\\begin{longtable}{r r r r r}\n");
        out.push_str(
            "\\caption{Power-law viscosity parameter sweep (T2).}\\label{tab:t2-sweep}\\\\\n",
        );
        out.push_str("\\toprule\n");
        out.push_str("$\\alpha$ & $\\beta$ & \\textbf{Power Index} & \\textbf{Slope Ratio} & \\textbf{Non-Newtonian} \\\\\n");
        out.push_str("\\midrule\n");
        out.push_str("\\endfirsthead\n");
        out.push_str("\\toprule\n");
        out.push_str("$\\alpha$ & $\\beta$ & \\textbf{Power Index} & \\textbf{Slope Ratio} & \\textbf{Non-Newtonian} \\\\\n");
        out.push_str("\\midrule\n");
        out.push_str("\\endhead\n");

        for s in &data.series {
            let nn = s
                .non_newtonian
                .map_or("--", |b| if b { "Yes" } else { "No" });
            let ratio = s
                .slope_ratio
                .map_or("--".to_string(), |r| format!("{r:.4}"));
            out.push_str(&format!(
                "{:.3} & {:.3} & {:.3} & {} & {} \\\\\n",
                s.alpha, s.beta, s.power_index, ratio, nn,
            ));
        }

        out.push_str("\\bottomrule\n");
        out.push_str("\\end{longtable}\n\n");
    }

    out
}

// ---------------------------------------------------------------------------
// Reproducibility appendix generation (T-017)
// ---------------------------------------------------------------------------

fn generate_reproducibility_appendix(
    experiments: &[ExperimentEntry],
    manifest: Option<&ManifestFile>,
    hardware: &str,
) -> String {
    let mut out = String::new();

    out.push_str("% Auto-generated by generate-latex --repro-appendix\n");
    out.push_str("\\section{Reproducibility Appendix}\n\\label{sec:reproducibility}\n\n");

    // Build environment
    out.push_str("\\subsection{Build Environment}\n\n");
    out.push_str("\\begin{itemize}\n");
    out.push_str(&format!(
        "  \\item \\textbf{{Hardware}}: {}\n",
        escape_latex(hardware)
    ));
    out.push_str("  \\item \\textbf{Toolchain}: Rust nightly (Edition 2024)\n");
    out.push_str("  \\item \\textbf{Build}: \\texttt{cargo build --release --workspace}\n");
    if let Some(m) = manifest.filter(|m| !m.metadata.git_rev.is_empty()) {
        out.push_str(&format!(
            "  \\item \\textbf{{Git revision}}: \\texttt{{{}}}\n",
            escape_latex(&m.metadata.git_rev)
        ));
    }
    out.push_str("\\end{itemize}\n\n");

    // Evidence regeneration commands
    out.push_str("\\subsection{Evidence Regeneration Commands}\n\n");
    out.push_str("\\begin{longtable}{l l l}\n");
    out.push_str("\\toprule\n");
    out.push_str("\\textbf{Experiment} & \\textbf{Binary} & \\textbf{Status} \\\\\n");
    out.push_str("\\midrule\n");
    out.push_str("\\endfirsthead\n");
    out.push_str("\\toprule\n");
    out.push_str("\\textbf{Experiment} & \\textbf{Binary} & \\textbf{Status} \\\\\n");
    out.push_str("\\midrule\n");
    out.push_str("\\endhead\n");

    for exp in experiments {
        let binary = exp.binary.as_deref().unwrap_or("--");
        let status = exp.status.as_deref().unwrap_or("--");
        out.push_str(&format!(
            "{} & \\texttt{{{}}} & {} \\\\\n",
            escape_latex(&exp.id),
            escape_latex(binary),
            escape_latex(status),
        ));
    }

    out.push_str("\\bottomrule\n");
    out.push_str("\\end{longtable}\n\n");

    // Artifact checksums
    if let Some(m) = manifest {
        out.push_str("\\subsection{Artifact Checksums}\n\n");
        out.push_str(&format!(
            "{} artifacts in \\texttt{{{}}}, {} present, {} missing.\n\n",
            m.summary.total,
            escape_latex(&m.metadata.evidence_dir),
            m.summary.present,
            m.summary.missing,
        ));

        if !m.artifact.is_empty() {
            out.push_str("\\begin{longtable}{l l r}\n");
            out.push_str("\\toprule\n");
            out.push_str(
                "\\textbf{Artifact} & \\textbf{SHA-256 (first 16)} & \\textbf{Size} \\\\\n",
            );
            out.push_str("\\midrule\n");
            out.push_str("\\endfirsthead\n");
            out.push_str("\\toprule\n");
            out.push_str(
                "\\textbf{Artifact} & \\textbf{SHA-256 (first 16)} & \\textbf{Size} \\\\\n",
            );
            out.push_str("\\midrule\n");
            out.push_str("\\endhead\n");

            for a in &m.artifact {
                let hash_short = if a.sha256.len() >= 16 {
                    &a.sha256[..16]
                } else {
                    &a.sha256
                };
                let size_str = if a.size_bytes > 0 {
                    format!("{}", a.size_bytes)
                } else {
                    "--".to_string()
                };
                out.push_str(&format!(
                    "\\texttt{{{}}} & \\texttt{{{}}} & {} \\\\\n",
                    escape_latex(&a.path),
                    hash_short,
                    size_str,
                ));
            }

            out.push_str("\\bottomrule\n");
            out.push_str("\\end{longtable}\n\n");
        }

        // Known gaps
        let missing_artifacts: Vec<&ManifestArtifact> = m
            .artifact
            .iter()
            .filter(|a| a.status == "missing")
            .collect();
        if !missing_artifacts.is_empty() {
            out.push_str("\\subsection{Known Gaps}\n\n");
            out.push_str("\\begin{itemize}\n");
            for a in &missing_artifacts {
                out.push_str(&format!(
                    "  \\item \\texttt{{{}}} -- recorded as missing\n",
                    escape_latex(&a.path)
                ));
            }
            out.push_str("\\end{itemize}\n\n");
        }
    }

    out
}

fn load_registry_compat<T>(args: &Args, kind: ControlPlaneCompatKind, fallback: &str) -> Result<T>
where
    T: serde::de::DeserializeOwned,
{
    if args.db.exists() {
        let mut store = ProvenanceStore::open(&args.db)
            .with_context(|| format!("open canonical control-plane DB {}", args.db.display()))?;
        let text = store.control_plane_compat_text(kind).with_context(|| {
            format!(
                "render {:?} compatibility text from {}",
                kind,
                args.db.display()
            )
        })?;
        return toml::from_str(&text).with_context(|| {
            format!(
                "parse {:?} compatibility TOML rendered from {}",
                kind,
                args.db.display()
            )
        });
    }

    let fallback_path = args.dir.join(fallback);
    let content = std::fs::read_to_string(&fallback_path).with_context(|| {
        format!(
            "read legacy compatibility registry fallback {}",
            fallback_path.display()
        )
    })?;
    toml::from_str(&content).with_context(|| {
        format!(
            "parse legacy compatibility registry {}",
            fallback_path.display()
        )
    })
}

fn main() -> Result<()> {
    let args = Args::parse();

    std::fs::create_dir_all(&args.output_dir)
        .with_context(|| format!("create output directory {}", args.output_dir.display()))?;

    // --- Claims ---
    let claims_registry: ClaimsRegistry =
        load_registry_compat(&args, ControlPlaneCompatKind::Claims, "claims.toml")?;
    let claims_tex = generate_claims_appendix(&claims_registry.claim);
    let claims_out_path = args.output_dir.join("claims_appendix.tex");
    std::fs::write(&claims_out_path, &claims_tex)
        .with_context(|| format!("write {}", claims_out_path.display()))?;
    println!(
        "Wrote {} claims to {}",
        claims_registry.claim.len(),
        claims_out_path.display()
    );

    // --- Insights ---
    let insights_registry: InsightsRegistry =
        load_registry_compat(&args, ControlPlaneCompatKind::Insights, "insights.toml")?;
    let insights_tex = generate_insights_appendix(&insights_registry.insight);
    let insights_out_path = args.output_dir.join("insights_appendix.tex");
    std::fs::write(&insights_out_path, &insights_tex)
        .with_context(|| format!("write {}", insights_out_path.display()))?;
    println!(
        "Wrote {} insights to {}",
        insights_registry.insight.len(),
        insights_out_path.display()
    );

    // --- Experiments ---
    let experiments_registry: ExperimentsRegistry = load_registry_compat(
        &args,
        ControlPlaneCompatKind::Experiments,
        "experiments.toml",
    )?;
    let experiments_tex = generate_experiments_appendix(&experiments_registry.experiment);
    let experiments_out_path = args.output_dir.join("experiments_appendix.tex");
    std::fs::write(&experiments_out_path, &experiments_tex)
        .with_context(|| format!("write {}", experiments_out_path.display()))?;
    println!(
        "Wrote {} experiments to {}",
        experiments_registry.experiment.len(),
        experiments_out_path.display()
    );

    // --- Thesis Results Table (T-016) ---
    if args.thesis_tables {
        let synthesis_path = PathBuf::from("data/evidence/synthesis_final/synthesis_summary.toml");
        if synthesis_path.exists() {
            let content = std::fs::read_to_string(&synthesis_path)
                .with_context(|| format!("read {}", synthesis_path.display()))?;
            let summary: SynthesisSummary = toml::from_str(&content)
                .with_context(|| format!("parse {}", synthesis_path.display()))?;

            // Try to load T2 power-law data
            let t2_path = PathBuf::from("data/evidence/thesis2_power_law_viscosity_v2.toml");
            let t2_data = if t2_path.exists() {
                let t2_content = std::fs::read_to_string(&t2_path)
                    .with_context(|| format!("read {}", t2_path.display()))?;
                toml::from_str::<T2PowerLawData>(&t2_content).ok()
            } else {
                None
            };

            let tex = generate_thesis_results_table(&summary, t2_data.as_ref());
            let out_path = args.output_dir.join("thesis_results_table.tex");
            std::fs::write(&out_path, &tex)
                .with_context(|| format!("write {}", out_path.display()))?;
            println!(
                "Wrote {} thesis entries to {}",
                summary.thesis.len(),
                out_path.display()
            );
        } else {
            eprintln!(
                "WARNING: {} not found, skipping thesis tables",
                synthesis_path.display()
            );
        }
    }

    // --- Reproducibility Appendix (T-017) ---
    if args.repro_appendix {
        let experiments = experiments_registry.experiment;

        // Load manifest if available
        let manifest = if args.manifest.exists() {
            let content = std::fs::read_to_string(&args.manifest)
                .with_context(|| format!("read {}", args.manifest.display()))?;
            toml::from_str::<ManifestFile>(&content).ok()
        } else {
            eprintln!(
                "WARNING: {} not found, checksums will be omitted",
                args.manifest.display()
            );
            None
        };

        let tex =
            generate_reproducibility_appendix(&experiments, manifest.as_ref(), &args.hardware);
        let out_path = args.output_dir.join("reproducibility_appendix.tex");
        std::fs::write(&out_path, &tex).with_context(|| format!("write {}", out_path.display()))?;
        println!("Wrote reproducibility appendix to {}", out_path.display());
    }

    Ok(())
}
