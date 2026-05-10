use clap::{Parser, Subcommand};
use csv::WriterBuilder;
use provenance_store::{ControlPlaneCompatKind, ProvenanceStore};
use regex::Regex;
use serde::Deserialize;
use std::{
    fs,
    path::{Path, PathBuf},
};
use toml::Value;

/// Emit non-canonical views from canonical registries and SQLite-backed compatibility exports
/// (markdown, bibtex, tex, pgfplots, svg, mermaid).
#[derive(Debug, Parser)]
#[command(name = "registry-emit")]
#[command(about = "Registry/control-plane multi-format emitter frontend")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

// Boxing the larger Subcommand variants would force callers through an
// extra deref and lose clap derive ergonomics; the enum is dropped after
// dispatch so the size cost is one-shot.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Subcommand)]
enum Commands {
    /// Emit markdown files from artifact scroll TOML registry.
    ArtifactMarkdown(ArtifactMarkdownArgs),
    /// Emit the structured TODO markdown mirror from registry/todo.toml.
    TodoMirror(TodoMirrorArgs),
    /// Emit the legacy crates/data_core/src/registry_mirrors/todo.rs from TODO TOML sources.
    TodoLegacy(TodoLegacyArgs),
    /// Emit the roadmap markdown mirror from registry/roadmap.toml.
    RoadmapMirror(RoadmapMirrorArgs),
    /// Emit the legacy crates/data_core/src/registry_mirrors/roadmap.rs from roadmap TOML sources.
    RoadmapLegacy(RoadmapLegacyArgs),
    /// Emit the next-actions markdown mirror from registry/next_actions.toml.
    NextActionsMirror(NextActionsMirrorArgs),
    /// Emit the legacy crates/data_core/src/registry_mirrors/next_actions.rs from next-actions TOML sources.
    NextActionsLegacy(NextActionsLegacyArgs),
    /// Emit the knowledge migration plan markdown mirror from registry/knowledge_migration_plan.toml.
    KnowledgeMigrationPlanMirror(KnowledgeMigrationPlanMirrorArgs),
    /// Emit the navigator markdown mirror from registry/navigator.toml.
    NavigatorMirror(NavigatorMirrorArgs),
    /// Emit the legacy crates/data_core/src/registry_mirrors/navigator.rs from registry/navigator.toml.
    NavigatorLegacy(NavigatorLegacyArgs),
    /// Emit the entrypoint-docs markdown mirror from registry/entrypoint_docs.toml.
    EntrypointDocsMirror(EntrypointDocsMirrorArgs),
    /// Emit the legacy entrypoint docs from registry/entrypoint_docs.toml.
    EntrypointDocsLegacy(EntrypointDocsLegacyArgs),
    /// Emit the requirements markdown mirror from registry/requirements.toml.
    RequirementsMirror(RequirementsMirrorArgs),
    /// Emit the legacy requirements docs from registry/requirements*.toml.
    RequirementsLegacy(RequirementsLegacyArgs),
    /// Emit the docs-root narratives markdown mirror from registry/docs_root_narratives.toml.
    DocsRootNarrativesMirror(DocsRootNarrativesMirrorArgs),
    /// Emit the legacy published docs/*.md files from registry/docs_root_narratives.toml.
    DocsRootNarrativesLegacy(DocsRootNarrativesLegacyArgs),
    /// Emit the research narratives markdown mirror from registry/research_narratives.toml.
    ResearchNarrativesMirror(ResearchNarrativesMirrorArgs),
    /// Emit the legacy published docs/theory|engineering|research markdown from registry/research_narratives.toml.
    ResearchNarrativesLegacy(ResearchNarrativesLegacyArgs),
    /// Emit the insights markdown mirror from the canonical control plane.
    InsightsMirror(InsightsMirrorArgs),
    /// Emit the legacy insights markdown from canonical control-plane data and the narrative overlay.
    InsightsLegacy(InsightsLegacyArgs),
    /// Emit the claims markdown mirror from the canonical control plane.
    ClaimsMirror(ClaimsMirrorArgs),
    /// Emit the legacy claims matrix markdown from canonical control-plane data.
    ClaimsMatrixLegacy(ClaimsMatrixLegacyArgs),
    /// Emit the bibliography markdown mirror from registry/bibliography.toml.
    BibliographyMirror(BibliographyMirrorArgs),
    /// Emit the legacy bibliography markdown from registry/bibliography.toml.
    BibliographyLegacy(BibliographyLegacyArgs),
    /// Emit the experiments markdown mirror from the canonical control plane.
    ExperimentsMirror(ExperimentsMirrorArgs),
    /// Emit the legacy experiments markdown from canonical control-plane data and the narrative overlay.
    ExperimentsLegacy(ExperimentsLegacyArgs),
    /// Emit the theorem markdown mirror from the canonical control plane.
    TheoremsMirror(TheoremsMirrorArgs),
    /// Emit the legacy theorem index markdown from the canonical control plane.
    TheoremsLegacy(TheoremsLegacyArgs),
    /// Emit the standard control-plane web docs bundle from the canonical SQLite source.
    ControlPlaneDocs(ControlPlaneDocsArgs),
    /// Emit the markdown governance mirror from registry/markdown_governance.toml.
    MarkdownGovernanceMirror(MarkdownGovernanceMirrorArgs),
    /// Emit the claims-tasks markdown mirror from registry/claims_tasks.toml.
    ClaimsTasksMirror(ClaimsTasksMirrorArgs),
    /// Emit the legacy claims-tasks markdown from registry/claims_tasks.toml.
    ClaimsTasksLegacy(ClaimsTasksLegacyArgs),
    /// Emit the claims-domains markdown mirror from registry/claims_domains.toml.
    ClaimsDomainsMirror(ClaimsDomainsMirrorArgs),
    /// Emit the legacy claims-domains markdown/csv from registry/claims_domains.toml.
    ClaimsDomainsLegacy(ClaimsDomainsLegacyArgs),
    /// Emit the claim-tickets markdown mirror from registry/claim_tickets.toml.
    ClaimTicketsMirror(ClaimTicketsMirrorArgs),
    /// Emit the legacy claim-ticket markdown from registry/claim_tickets.toml.
    ClaimTicketsLegacy(ClaimTicketsLegacyArgs),
    /// Emit the external-sources markdown mirror from registry/external_sources.toml.
    ExternalSourcesMirror(ExternalSourcesMirrorArgs),
    /// Emit the legacy external-sources markdown from registry/external_sources.toml.
    ExternalSourcesLegacy(ExternalSourcesLegacyArgs),
    /// Emit the book-docs markdown mirror from registry/book_docs.toml.
    BookDocsMirror(BookDocsMirrorArgs),
    /// Emit the legacy book-docs markdown from registry/book_docs.toml.
    BookDocsLegacy(BookDocsLegacyArgs),
    /// Emit the data-artifact narratives markdown mirror from registry/data_artifact_narratives.toml.
    DataArtifactNarrativesMirror(DataArtifactNarrativesMirrorArgs),
    /// Emit the legacy data-artifact narratives markdown from registry/data_artifact_narratives.toml.
    DataArtifactNarrativesLegacy(DataArtifactNarrativesLegacyArgs),
    /// Emit the reports narratives markdown mirror from registry/reports_narratives.toml.
    ReportsNarrativesMirror(ReportsNarrativesMirrorArgs),
    /// Emit the legacy reports narratives markdown from registry/reports_narratives.toml.
    ReportsNarrativesLegacy(ReportsNarrativesLegacyArgs),
    /// Emit the docs-convos markdown mirror from registry/docs_convos.toml.
    DocsConvosMirror(DocsConvosMirrorArgs),
    /// Emit the legacy docs-convos markdown from registry/docs_convos.toml.
    DocsConvosLegacy(DocsConvosLegacyArgs),
    /// Emit the legacy monograph markdown from registry/monograph.toml.
    MonographLegacy(MonographLegacyArgs),
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
    /// Patch static registry_mirrors .rs files that lack AUTO-GENERATED headers.
    ///
    /// Scans crates/data_core/src/registry_mirrors/ for .rs files (excluding mod.rs)
    /// whose first five lines do not contain "AUTO-GENERATED", and prepends the standard
    /// three-line markdown_to_rust header.  Safe to run repeatedly (idempotent).
    PatchStaticMirrorHeaders(PatchStaticMirrorHeadersArgs),
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
struct TodoMirrorArgs {
    /// TODO registry TOML path.
    #[arg(long, default_value = "registry/todo.toml")]
    input: PathBuf,
    /// Output markdown mirror path.
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/todo_registry_mirror.rs"
    )]
    output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct TodoLegacyArgs {
    #[arg(long, default_value = "registry/todo.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/todo_narrative.toml")]
    narrative: PathBuf,
    #[arg(long, default_value = "crates/data_core/src/registry_mirrors/todo.rs")]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct RoadmapMirrorArgs {
    #[arg(long, default_value = "registry/roadmap.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/roadmap_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct RoadmapLegacyArgs {
    #[arg(long, default_value = "registry/roadmap.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/roadmap_narrative.toml")]
    narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/roadmap.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct NextActionsMirrorArgs {
    #[arg(long, default_value = "registry/next_actions.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/next_actions_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct NextActionsLegacyArgs {
    #[arg(long, default_value = "registry/next_actions.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/next_actions_narrative.toml")]
    narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/next_actions.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct KnowledgeMigrationPlanMirrorArgs {
    /// Knowledge migration plan registry TOML path.
    #[arg(long, default_value = "registry/knowledge_migration_plan.toml")]
    input: PathBuf,
    /// Output markdown mirror path.
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/knowledge_migration_plan_registry_mirror.rs"
    )]
    output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct NavigatorMirrorArgs {
    #[arg(long, default_value = "registry/navigator.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/navigator_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct NavigatorLegacyArgs {
    #[arg(long, default_value = "registry/navigator.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/navigator.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct EntrypointDocsMirrorArgs {
    #[arg(long, default_value = "registry/entrypoint_docs.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/entrypoint_docs_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct EntrypointDocsLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/entrypoint_docs.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct RequirementsMirrorArgs {
    #[arg(long, default_value = "registry/requirements.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/requirements_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct RequirementsLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/requirements.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/requirements_narrative.toml")]
    narrative: PathBuf,
    /// Optional audit tools TOML; when provided, appends an "Audit Tools" section
    /// to the primary requirements markdown.
    #[arg(long)]
    audit_tools: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DocsRootNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/docs_root_narratives.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/docs_root_narratives_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DocsRootNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/docs_root_narratives.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ResearchNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/research_narratives.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/research_narratives_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ResearchNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/research_narratives.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct InsightsMirrorArgs {
    #[arg(long, default_value = "registry/insights.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct InsightsLegacyArgs {
    #[arg(long, default_value = "registry/insights.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "registry/insights_narrative.toml")]
    narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimsMirrorArgs {
    #[arg(long, default_value = "registry/claims.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimsMatrixLegacyArgs {
    #[arg(long, default_value = "registry/claims.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_evidence_matrix.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct BibliographyMirrorArgs {
    #[arg(long, default_value = "registry/bibliography.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/bibliography_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct BibliographyLegacyArgs {
    #[arg(long, default_value = "registry/bibliography.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/bibliography.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ExperimentsMirrorArgs {
    #[arg(long, default_value = "registry/experiments.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ExperimentsLegacyArgs {
    #[arg(long, default_value = "registry/experiments.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "registry/experiments_narrative.toml")]
    narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_portfolio_shortlist.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct TheoremsMirrorArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct TheoremsLegacyArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ControlPlaneDocsArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    canonical_db: PathBuf,
    #[arg(long, default_value = "registry/claims.toml")]
    claims_input: PathBuf,
    #[arg(long, default_value = "registry/insights.toml")]
    insights_input: PathBuf,
    #[arg(long, default_value = "registry/experiments.toml")]
    experiments_input: PathBuf,
    #[arg(long, default_value = "registry/insights_narrative.toml")]
    insights_narrative: PathBuf,
    #[arg(long, default_value = "registry/experiments_narrative.toml")]
    experiments_narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_registry_mirror.rs"
    )]
    claims_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_evidence_matrix.rs"
    )]
    claims_legacy_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights_registry_mirror.rs"
    )]
    insights_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights.rs"
    )]
    insights_legacy_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_registry_mirror.rs"
    )]
    experiments_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_portfolio_shortlist.rs"
    )]
    experiments_legacy_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems_registry_mirror.rs"
    )]
    theorems_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems.rs"
    )]
    theorems_legacy_output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct MarkdownGovernanceMirrorArgs {
    #[arg(long, default_value = "registry/markdown_governance.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/markdown_governance_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimsTasksMirrorArgs {
    #[arg(long, default_value = "registry/claims_tasks.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_tasks_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimsTasksLegacyArgs {
    #[arg(long, default_value = "registry/claims_tasks.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_tasks.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimsDomainsMirrorArgs {
    #[arg(long, default_value = "registry/claims_domains.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_domains_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimsDomainsLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/claims_domains.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimTicketsMirrorArgs {
    #[arg(long, default_value = "registry/claim_tickets.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claim_tickets_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ClaimTicketsLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/claim_tickets.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ExternalSourcesMirrorArgs {
    #[arg(long, default_value = "registry/external_sources.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/external_sources_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ExternalSourcesLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/external_sources.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct BookDocsMirrorArgs {
    #[arg(long, default_value = "registry/book_docs.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/book_docs_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct BookDocsLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/book_docs.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DataArtifactNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/data_artifact_narratives.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/data_artifact_narratives_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DataArtifactNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/data_artifact_narratives.toml")]
    input: PathBuf,
    #[arg(long, default_value = "registry/artifact_scrolls.toml")]
    artifact_index: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ReportsNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/reports_narratives.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/reports_narratives_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct ReportsNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/reports_narratives.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DocsConvosMirrorArgs {
    #[arg(long, default_value = "registry/docs_convos.toml")]
    input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/docs_convos_registry_mirror.rs"
    )]
    output: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct DocsConvosLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/docs_convos.toml")]
    input: PathBuf,
    #[arg(long, default_value_t = false)]
    allow_unicode: bool,
}

#[derive(Debug, Parser)]
struct MonographLegacyArgs {
    #[arg(long, default_value = ".")]
    repo_root: PathBuf,
    #[arg(long, default_value = "registry/monograph.toml")]
    input: PathBuf,
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

#[derive(Debug, Parser)]
struct PatchStaticMirrorHeadersArgs {
    /// Registry mirrors directory to scan.
    #[arg(long, default_value = "crates/data_core/src/registry_mirrors")]
    mirrors_dir: PathBuf,
    /// Dry-run: print which files would be patched without writing.
    #[arg(long, default_value_t = false)]
    dry_run: bool,
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
    let mut final_text = text.to_string();
    if path.extension().and_then(|e| e.to_str()) == Some("rs") {
        // Prepend the rustfmt @generated marker so cargo fmt --all skips
        // these auto-generated mirror files, preventing oscillation between
        // the SQLite -> Rust regen path and rustfmt's reformatting passes.
        // Reference: rustfmt 1.x recognizes both `// @generated` and the
        // file_lines `Generated` attribute; the comment form is portable.
        let mut rustdoc = String::from(
            "// @generated by registry-emit; do not edit. \
             Source: registry/canonical/control_plane.sqlite3\n",
        );
        // Insert module doc attributes
        for line in final_text.lines() {
            if line.is_empty() {
                rustdoc.push_str("//!\n");
            } else {
                rustdoc.push_str("//! ");
                rustdoc.push_str(line);
                rustdoc.push('\n');
            }
        }
        final_text = rustdoc;
    }

    if !allow_unicode {
        ensure_ascii(&final_text, &path.display().to_string())?;
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).map_err(|err| format!("mkdir {}: {}", parent.display(), err))?;
    }
    fs::write(path, final_text).map_err(|err| format!("write {}: {}", path.display(), err))
}

fn read_toml_value(path: &Path) -> Result<Value, String> {
    let text =
        fs::read_to_string(path).map_err(|err| format!("read {}: {}", path.display(), err))?;
    toml::from_str(&text).map_err(|err| format!("parse {}: {}", path.display(), err))
}

fn read_control_plane_compat_value(
    input: &Path,
    canonical_db: &Path,
    kind: ControlPlaneCompatKind,
) -> Result<(Value, String), String> {
    if canonical_db.exists() {
        let mut store = ProvenanceStore::open(canonical_db)
            .map_err(|err| format!("open canonical db {}: {err}", canonical_db.display()))?;
        let text = store.control_plane_compat_text(kind).map_err(|err| {
            format!(
                "render {:?} compatibility text from canonical db {}: {err}",
                kind,
                canonical_db.display()
            )
        })?;
        let value = toml::from_str(&text).map_err(|err| {
            format!(
                "parse {:?} compatibility text from canonical db {}: {err}",
                kind,
                canonical_db.display()
            )
        })?;
        return Ok((value, canonical_db.display().to_string()));
    }

    Ok((read_toml_value(input)?, input.display().to_string()))
}

fn read_control_plane_compat_text(
    canonical_db: &Path,
    kind: ControlPlaneCompatKind,
) -> Result<String, String> {
    let mut store = ProvenanceStore::open(canonical_db)
        .map_err(|err| format!("open canonical db {}: {err}", canonical_db.display()))?;
    store.control_plane_compat_text(kind).map_err(|err| {
        format!(
            "render {:?} compatibility text from canonical db {}: {err}",
            kind,
            canonical_db.display()
        )
    })
}

fn markdown_header(title: &str, source: &str) -> Vec<String> {
    vec![
        format!("# {}", title),
        String::new(),
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        "<!-- Source of truth: see authoritative source line below -->".to_string(),
        String::new(),
        format!("Authoritative source: `{}`.", source),
        String::new(),
    ]
}

fn control_plane_markdown_header(title: &str, source: &str) -> Vec<String> {
    vec![
        format!("# {}", title),
        String::new(),
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        "<!-- Source of truth: registry/canonical/control_plane.sqlite3 -->".to_string(),
        String::new(),
        format!("Authoritative source: `{}`.", source),
        String::new(),
    ]
}

fn generated_doc_header(source: &str) -> Vec<String> {
    vec![
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        format!("<!-- Source of truth: {} -->", source),
        String::new(),
    ]
}

fn table<'a>(value: &'a Value, key: &str) -> Result<&'a toml::map::Map<String, Value>, String> {
    value
        .get(key)
        .and_then(Value::as_table)
        .ok_or_else(|| format!("missing table `{}`", key))
}

fn rows<'a>(value: &'a Value, key: &str) -> Vec<&'a toml::map::Map<String, Value>> {
    value
        .get(key)
        .and_then(Value::as_array)
        .map(|items| items.iter().filter_map(Value::as_table).collect::<Vec<_>>())
        .unwrap_or_default()
}

fn str_field(table: &toml::map::Map<String, Value>, key: &str) -> String {
    table
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn raw_str_field(table: &toml::map::Map<String, Value>, key: &str) -> String {
    table
        .get(key)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn int_field(table: &toml::map::Map<String, Value>, key: &str) -> i64 {
    table.get(key).and_then(Value::as_integer).unwrap_or(0)
}

fn array_of_strings(table: &toml::map::Map<String, Value>, key: &str) -> Vec<String> {
    table
        .get(key)
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(|item| item.trim().to_string())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
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
        if let Some(filter_id) = &args.id
            && row.id != *filter_id
        {
            continue;
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

fn emit_todo_mirror(args: TodoMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "todo")?;
    let items = rows(&data, "item");
    let mut lines = markdown_header("TODO Registry Mirror", &args.input.display().to_string());
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!("- Status: `{}`", str_field(meta, "status")));
    lines.push(format!(
        "- Item count: `{}`",
        int_field(meta, "item_count").max(items.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Items".to_string());
    lines.push(String::new());
    for item in items {
        let item_id = str_field(item, "id");
        let title = str_field(item, "title");
        lines.push(format!("### {}: {}", item_id, title));
        lines.push(String::new());
        lines.push(format!("- Area: `{}`", str_field(item, "area")));
        lines.push(format!("- Priority: `{}`", str_field(item, "priority")));
        lines.push(format!("- Status: `{}`", str_field(item, "status")));
        lines.push(format!("- Description: {}", str_field(item, "description")));
        let dependencies = array_of_strings(item, "dependencies");
        let dep_text = if dependencies.is_empty() {
            "(none)".to_string()
        } else {
            dependencies
                .iter()
                .map(|dep| format!("`{}`", dep))
                .collect::<Vec<_>>()
                .join(", ")
        };
        lines.push(format!("- Dependencies: {}", dep_text));
        lines.push("- Acceptance criteria:".to_string());
        for criterion in array_of_strings(item, "acceptance_criteria") {
            lines.push(format!("  - {}", criterion));
        }
        lines.push("- Evidence refs:".to_string());
        let evidence_refs = array_of_strings(item, "evidence_refs");
        if evidence_refs.is_empty() {
            lines.push("  - (none)".to_string());
        } else {
            for item in evidence_refs {
                lines.push(format!("  - `{}`", item));
            }
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted TODO markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_todo_legacy(args: TodoLegacyArgs) -> Result<(), String> {
    let body = single_overlay_body(&args.narrative, "todo_narrative")?;
    let fallback = [
        "This file is generated from `registry/todo.toml` and `registry/todo_narrative.toml`.",
        "",
        "See the structured mirror at `crates/data_core/src/registry_mirrors/todo_registry_mirror.rs`.",
    ];
    let mut lines = generated_doc_header("registry/todo.toml; registry/todo_narrative.toml");
    lines.extend(legacy_lines_from_body(&body, "TODO", &fallback));
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted TODO legacy markdown from {} and {} to {}.",
        args.input.display(),
        args.narrative.display(),
        args.output.display()
    );
    Ok(())
}

fn single_overlay_body(path: &Path, table_name: &str) -> Result<String, String> {
    let data = read_toml_value(path)?;
    Ok(data
        .get(table_name)
        .and_then(Value::as_table)
        .map(|table| raw_str_field(table, "body_markdown").trim().to_string())
        .unwrap_or_default())
}

fn legacy_lines_from_body(
    body: &str,
    fallback_title: &str,
    fallback_lines: &[&str],
) -> Vec<String> {
    if !body.trim().is_empty() {
        let mut lines = body.lines().map(ToString::to_string).collect::<Vec<_>>();
        if lines.is_empty() || !lines.last().map(|line| line.is_empty()).unwrap_or(false) {
            lines.push(String::new());
        }
        return lines;
    }
    let mut lines = vec![format!("# {}", fallback_title), String::new()];
    lines.extend(fallback_lines.iter().map(|line| (*line).to_string()));
    if !lines.last().map(|line| line.is_empty()).unwrap_or(false) {
        lines.push(String::new());
    }
    lines
}

fn emit_roadmap_mirror(args: RoadmapMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let roadmap = table(&data, "roadmap")?;
    let workstreams = rows(&data, "workstream");
    let mut lines = markdown_header("Roadmap Registry Mirror", &args.input.display().to_string());
    lines.push(format!(
        "- Consolidated date: {}",
        str_field(roadmap, "consolidated_date")
    ));
    lines.push(format!(
        "- Source markdown: `{}`",
        str_field(roadmap, "source_markdown")
    ));
    lines.push(format!("- Status: `{}`", str_field(roadmap, "status")));
    lines.push(String::new());
    lines.push("## Companion Docs".to_string());
    lines.push(String::new());
    for item in array_of_strings(roadmap, "companion_docs") {
        lines.push(format!("- `{}`", item));
    }
    lines.push(String::new());
    lines.push("## Workstreams".to_string());
    lines.push(String::new());
    for ws in workstreams {
        lines.push(format!(
            "### {}: {}",
            str_field(ws, "id"),
            str_field(ws, "name")
        ));
        lines.push(String::new());
        lines.push(format!("- Priority: `{}`", str_field(ws, "priority")));
        lines.push(format!("- Status: `{}`", str_field(ws, "status")));
        lines.push(format!("- Description: {}", str_field(ws, "description")));
        lines.push("- Primary outputs:".to_string());
        for out in array_of_strings(ws, "primary_outputs") {
            lines.push(format!("  - `{}`", out));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted roadmap markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_roadmap_legacy(args: RoadmapLegacyArgs) -> Result<(), String> {
    let body = single_overlay_body(&args.narrative, "roadmap_narrative")?;
    let fallback = [
        "This file is generated from `registry/roadmap.toml` and `registry/roadmap_narrative.toml`.",
        "",
        "See the structured mirror at `crates/data_core/src/registry_mirrors/roadmap_registry_mirror.rs`.",
    ];
    let mut lines = generated_doc_header("registry/roadmap.toml; registry/roadmap_narrative.toml");
    lines.extend(legacy_lines_from_body(&body, "ROADMAP", &fallback));
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted roadmap legacy markdown from {} and {} to {}.",
        args.input.display(),
        args.narrative.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_next_actions_mirror(args: NextActionsMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "meta").or_else(|_| table(&data, "next_actions"))?;
    let actions = rows(&data, "action");
    let mut lines = markdown_header(
        "Next Actions Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!("- Status: `{}`", str_field(meta, "status")));
    lines.push(String::new());
    lines.push("## Priority Queue".to_string());
    lines.push(String::new());
    for action in actions {
        lines.push(format!(
            "### {} ({}): {}",
            str_field(action, "id"),
            str_field(action, "priority"),
            str_field(action, "title")
        ));
        lines.push(String::new());
        lines.push(format!("- Status: `{}`", str_field(action, "status")));
        lines.push(format!(
            "- Description: {}",
            str_field(action, "description")
        ));
        lines.push("- References:".to_string());
        for reference in array_of_strings(action, "references")
            .into_iter()
            .chain(array_of_strings(action, "evidence_refs"))
        {
            lines.push(format!("  - `{}`", reference));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted next-actions markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_next_actions_legacy(args: NextActionsLegacyArgs) -> Result<(), String> {
    let body = single_overlay_body(&args.narrative, "next_actions_narrative")?;
    let fallback = [
        "This file is generated from `registry/next_actions.toml` and `registry/next_actions_narrative.toml`.",
        "",
        "See the structured mirror at `crates/data_core/src/registry_mirrors/next_actions_registry_mirror.rs`.",
    ];
    let mut lines =
        generated_doc_header("registry/next_actions.toml; registry/next_actions_narrative.toml");
    lines.extend(legacy_lines_from_body(&body, "NEXT ACTIONS", &fallback));
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted next-actions legacy markdown from {} and {} to {}.",
        args.input.display(),
        args.narrative.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_navigator_mirror(args: NavigatorMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "navigator")?;
    let sections = rows(&data, "section");
    let mut lines = markdown_header(
        "Navigator Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!("- Epoch: {}", str_field(meta, "epoch")));
    lines.push(format!("- Mission: {}", str_field(meta, "mission")));
    lines.push(format!("- Section count: {}", sections.len()));
    lines.push(String::new());
    for section in sections {
        lines.push(format!(
            "## {}: {}",
            str_field(section, "id"),
            str_field(section, "title")
        ));
        lines.push(String::new());
        let summary = str_field(section, "summary");
        if !summary.is_empty() {
            lines.push(summary);
            lines.push(String::new());
        }
        let links = section
            .get("link")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for link in links.iter().filter_map(Value::as_table) {
            lines.push(format!(
                "- `{}` -> `{}`",
                str_field(link, "label"),
                str_field(link, "path")
            ));
            lines.push(format!(
                "  - Hypothesis narrative: `{}`",
                link.get("hypothesis")
                    .and_then(Value::as_bool)
                    .unwrap_or(false)
            ));
            let notes = str_field(link, "notes");
            if !notes.is_empty() {
                lines.push(format!("  - Notes: {}", notes));
            }
        }
        lines.push(String::new());
    }
    if let Some(disclaimer) = meta.get("disclaimer").and_then(Value::as_table) {
        let text = str_field(disclaimer, "text");
        if !text.is_empty() {
            lines.push("## Disclaimer".to_string());
            lines.push(String::new());
            lines.push(text);
            let claims_source = str_field(disclaimer, "claims_source");
            if !claims_source.is_empty() {
                lines.push(format!("- Claims source: `{}`", claims_source));
            }
            let legacy = str_field(disclaimer, "legacy_claims_mirror");
            if !legacy.is_empty() {
                lines.push(format!("- Legacy claims mirror: `{}`", legacy));
            }
            lines.push(String::new());
        }
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted navigator markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_navigator_legacy(args: NavigatorLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "navigator")?;
    let sections = rows(&data, "section");
    let mut lines = generated_doc_header("registry/navigator.toml");
    let title = str_field(meta, "title");
    lines.push(format!(
        "# {}",
        if title.is_empty() {
            "Navigator"
        } else {
            &title
        }
    ));
    lines.push(String::new());
    let epoch = str_field(meta, "epoch");
    let mission = str_field(meta, "mission");
    if !epoch.is_empty() {
        lines.push(format!("**Current Epoch:** {}", epoch));
    }
    if !mission.is_empty() {
        lines.push(format!("**Mission:** {}", mission));
    }
    if !epoch.is_empty() || !mission.is_empty() {
        lines.push(String::new());
    }
    lines.push(
        "**Important:** This file is generated from registry compatibility data and is not authoritative."
            .to_string(),
    );
    lines.push(String::new());
    for section in sections {
        lines.push(format!(
            "## {}: {}",
            str_field(section, "id"),
            str_field(section, "title")
        ));
        lines.push(String::new());
        let summary = str_field(section, "summary");
        if !summary.is_empty() {
            lines.push(summary);
            lines.push(String::new());
        }
        let links = section
            .get("link")
            .and_then(Value::as_array)
            .cloned()
            .unwrap_or_default();
        for link in links.iter().filter_map(Value::as_table) {
            lines.push(format!(
                "- **{}:** `{}`",
                str_field(link, "label"),
                str_field(link, "path")
            ));
            let notes = str_field(link, "notes");
            if !notes.is_empty() {
                lines.push(format!("  - {}", notes));
            }
            if link
                .get("hypothesis")
                .and_then(Value::as_bool)
                .unwrap_or(false)
            {
                lines.push("  - Status: hypothesis narrative".to_string());
            }
        }
        lines.push(String::new());
    }
    if let Some(disclaimer) = meta.get("disclaimer").and_then(Value::as_table) {
        let text = str_field(disclaimer, "text");
        if !text.is_empty() {
            lines.push("---".to_string());
            lines.push(String::new());
            lines.push(text);
            let claims_source = str_field(disclaimer, "claims_source");
            if !claims_source.is_empty() {
                lines.push(format!("Claims source: `{}`", claims_source));
            }
            lines.push(String::new());
        }
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted navigator legacy markdown from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_entrypoint_docs_mirror(args: EntrypointDocsMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "entrypoint_docs")?;
    let docs = rows(&data, "document");
    let mut lines = markdown_header(
        "Entrypoint Docs Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    for doc in docs {
        lines.push(format!("## `{}`", str_field(doc, "path")));
        lines.push(String::new());
        lines.push(format!("- Title: {}", str_field(doc, "title")));
        let body = raw_str_field(doc, "body_markdown");
        if !body.trim().is_empty() {
            lines.push(format!("- Body lines: {}", body.lines().count()));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted entrypoint-docs markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_entrypoint_docs_legacy(args: EntrypointDocsLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = rows(&data, "document");
    for doc in docs {
        let rel_path = str_field(doc, "path");
        if rel_path.is_empty()
            || matches!(
                rel_path.as_str(),
                "crates/data_core/src/registry_mirrors/agents.rs" | "CLAUDE.md" | "GEMINI.md"
            )
        {
            continue;
        }
        let mut lines = generated_doc_header("registry/entrypoint_docs.toml");
        let body = raw_str_field(doc, "body_markdown");
        if !body.trim().is_empty() {
            lines.extend(body.lines().map(ToString::to_string));
        } else {
            lines.push(format!(
                "# {}",
                fallback_title(&rel_path, &str_field(doc, "title"))
            ));
            lines.push(String::new());
            lines.push("(No body_markdown captured in registry/entrypoint_docs.toml.)".to_string());
        }
        lines.push(String::new());
        write_output(
            &args.repo_root.join(&rel_path),
            &lines.join("\n"),
            args.allow_unicode,
        )?;
    }
    println!(
        "Emitted entrypoint legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_requirements_mirror(args: RequirementsMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let req = table(&data, "requirements")?;
    let modules = rows(&data, "module");
    let gaps = rows(&data, "coverage_gap");
    let mut lines = markdown_header(
        "Requirements Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(req, "updated")));
    lines.push(format!(
        "- Python recommended: `{}`",
        str_field(req, "python_recommended")
    ));
    lines.push(format!(
        "- Python allowed: `{}`",
        str_field(req, "python_allowed")
    ));
    lines.push(format!(
        "- Primary markdown: `{}`",
        str_field(req, "primary_markdown")
    ));
    lines.push(String::new());
    lines.push("## Modules".to_string());
    lines.push(String::new());
    for module in modules {
        lines.push(format!(
            "### {}: {}",
            str_field(module, "id"),
            str_field(module, "name")
        ));
        lines.push(String::new());
        lines.push(format!("- Status: `{}`", str_field(module, "status")));
        lines.push(format!("- Markdown: `{}`", str_field(module, "markdown")));
        let targets = array_of_strings(module, "install_targets");
        if !targets.is_empty() {
            lines.push("- Install targets:".to_string());
            for target in targets {
                lines.push(format!("  - `{}`", target));
            }
        }
        lines.push(String::new());
    }
    lines.push("## Coverage Gaps".to_string());
    lines.push(String::new());
    for gap in gaps {
        lines.push(format!(
            "### {}: {}",
            str_field(gap, "id"),
            str_field(gap, "area")
        ));
        lines.push(String::new());
        lines.push(format!("- Status: `{}`", str_field(gap, "status")));
        lines.push(format!("- Description: {}", str_field(gap, "description")));
        lines.push(format!(
            "- Proposed resolution: {}",
            str_field(gap, "proposed_resolution")
        ));
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted requirements markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_requirements_legacy(args: RequirementsLegacyArgs) -> Result<(), String> {
    let req_data = read_toml_value(&args.input)?;
    let narrative_data = read_toml_value(&args.narrative)?;
    let req_meta = table(&req_data, "requirements")?;
    let primary_markdown = str_field(req_meta, "primary_markdown");
    let mut targets = vec!["REQUIREMENTS.md".to_string(), primary_markdown.clone()];
    for module in rows(&req_data, "module") {
        let markdown = str_field(module, "markdown");
        if !markdown.is_empty() {
            targets.push(markdown);
        }
    }
    targets.sort();
    targets.dedup();

    let mut body_by_path = std::collections::BTreeMap::<String, String>::new();
    let mut title_by_path = std::collections::BTreeMap::<String, String>::new();
    for doc in rows(&narrative_data, "document") {
        let path = str_field(doc, "path");
        if path.is_empty() {
            continue;
        }
        body_by_path.insert(
            path.clone(),
            raw_str_field(doc, "body_markdown").trim().to_string(),
        );
        title_by_path.insert(path, str_field(doc, "title"));
    }
    let module_rows = rows(&req_data, "module");

    // Load audit tools TOML once if provided. Keep the Value alive so rows()
    // can borrow from it.
    let audit_tools_data = args
        .audit_tools
        .as_ref()
        .map(|path| read_toml_value(path))
        .transpose()?;

    for rel_path in targets {
        let title = title_by_path
            .get(&rel_path)
            .cloned()
            .unwrap_or_else(|| fallback_title(&rel_path, ""));
        let body = body_by_path.get(&rel_path).cloned().unwrap_or_default();
        let mut lines = generated_doc_header(
            "registry/requirements.toml; registry/requirements_narrative.toml",
        );
        if !body.is_empty() {
            lines.extend(body.lines().map(ToString::to_string));
        } else {
            lines.push(format!("# {}", title));
            lines.push(String::new());
            lines.push("This file is generated from `registry/requirements.toml` and `registry/requirements_narrative.toml`.".to_string());
            lines.push(String::new());
            lines.push(
                "See the structured mirror at `crates/data_core/src/registry_mirrors/requirements_registry_mirror.rs`."
                    .to_string(),
            );
            if let Some(module) = module_rows
                .iter()
                .find(|row| str_field(row, "markdown") == rel_path)
            {
                lines.push(String::new());
                lines.push(format!("- Module ID: `{}`", str_field(module, "id")));
                lines.push(format!("- Module name: `{}`", str_field(module, "name")));
                lines.push(format!("- Status: `{}`", str_field(module, "status")));
                let targets = array_of_strings(module, "install_targets");
                if !targets.is_empty() {
                    lines.push("- Install targets:".to_string());
                    for target in targets {
                        lines.push(format!("  - `{}`", target));
                    }
                }
            }
        }

        // Append "Audit Tools" section to the primary requirements markdown only.
        if (rel_path == primary_markdown || rel_path == "REQUIREMENTS.md")
            && let Some(ref data) = audit_tools_data
        {
            let tool_rows = rows(data, "tool");
            if !tool_rows.is_empty() {
                lines.push(String::new());
                lines.push("## Audit Tools".to_string());
                lines.push(String::new());
                lines.push(
                    "Each tool listed below is available via a dedicated `make` target. \
                     Tools marked **audit-deep** are included in `make audit-deep`."
                        .to_string(),
                );
                lines.push(String::new());
                lines.push("| Tool | Make Target | Install | audit-deep | Status |".to_string());
                lines.push("| --- | --- | --- | ---: | --- |".to_string());
                for tool in &tool_rows {
                    let name = str_field(tool, "name");
                    let make_target = str_field(tool, "make_target");
                    let install = str_field(tool, "install");
                    let in_audit_deep = bool_field(tool, "audit_deep");
                    let status = str_field(tool, "status");
                    let blocked = str_field(tool, "blocked_reason");
                    let status_cell = if blocked.is_empty() {
                        status
                    } else {
                        format!("{} -- {}", status, blocked)
                    };
                    lines.push(format!(
                        "| {} | `make {}` | {} | {} | {} |",
                        name,
                        make_target,
                        if install.is_empty() {
                            "(built-in)".to_string()
                        } else {
                            format!("`{}`", install)
                        },
                        if in_audit_deep { "yes" } else { "no" },
                        status_cell,
                    ));
                }
            }
        }

        lines.push(String::new());
        write_output(
            &args.repo_root.join(&rel_path),
            &lines.join("\n"),
            args.allow_unicode,
        )?;
    }
    println!(
        "Emitted requirements legacy markdown from {} and {} into {}.",
        args.input.display(),
        args.narrative.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_knowledge_migration_plan_mirror(
    args: KnowledgeMigrationPlanMirrorArgs,
) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "migration")?;
    let domains = rows(&data, "domain");
    let phases = rows(&data, "phase");
    let policies = rows(&data, "policy");
    let mut lines = markdown_header(
        "Knowledge Migration Plan Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Status: `{}`", str_field(meta, "status")));
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!("- Scope: {}", str_field(meta, "scope")));
    lines.push(String::new());
    lines.push("## Domains".to_string());
    lines.push(String::new());
    for domain in domains {
        lines.push(format!(
            "### {}: {}",
            str_field(domain, "id"),
            str_field(domain, "name")
        ));
        lines.push(String::new());
        lines.push(format!("- Strategy: `{}`", str_field(domain, "strategy")));
        lines.push(format!("- Status: `{}`", str_field(domain, "status")));
        let source_markdown = array_of_strings(domain, "source_markdown");
        if !source_markdown.is_empty() {
            lines.push("- Source markdown:".to_string());
            for item in source_markdown {
                lines.push(format!("  - `{}`", item));
            }
        }
        let authoritative = array_of_strings(domain, "authoritative_toml");
        if !authoritative.is_empty() {
            lines.push("- Authoritative TOML:".to_string());
            for item in authoritative {
                lines.push(format!("  - `{}`", item));
            }
        }
        let mirrors = array_of_strings(domain, "generated_mirror");
        if !mirrors.is_empty() {
            lines.push("- Generated mirrors:".to_string());
            for item in mirrors {
                lines.push(format!("  - `{}`", item));
            }
        }
        let notes = str_field(domain, "notes");
        if !notes.is_empty() {
            lines.push(format!("- Notes: {}", notes));
        }
        lines.push(String::new());
    }
    lines.push("## Phases".to_string());
    lines.push(String::new());
    for phase in phases {
        lines.push(format!(
            "### {}: {}",
            str_field(phase, "id"),
            str_field(phase, "name")
        ));
        lines.push(String::new());
        lines.push(format!("- Status: `{}`", str_field(phase, "status")));
        let deliverables = array_of_strings(phase, "deliverables");
        if !deliverables.is_empty() {
            lines.push("- Deliverables:".to_string());
            for item in deliverables {
                lines.push(format!("  - {}", item));
            }
        }
        lines.push(String::new());
    }
    if !policies.is_empty() {
        lines.push("## Policies".to_string());
        lines.push(String::new());
        for policy in policies {
            lines.push(format!(
                "### {}: {}",
                str_field(policy, "id"),
                str_field(policy, "name")
            ));
            lines.push(String::new());
            lines.push(format!("- Status: `{}`", str_field(policy, "status")));
            lines.push(format!("- Statement: {}", str_field(policy, "statement")));
            let enforcement = array_of_strings(policy, "enforcement");
            if !enforcement.is_empty() {
                lines.push("- Enforcement:".to_string());
                for item in enforcement {
                    lines.push(format!("  - {}", item));
                }
            }
            lines.push(String::new());
        }
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted knowledge migration plan markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn narrative_docs(value: &Value) -> Vec<&toml::map::Map<String, Value>> {
    let mut docs = rows(value, "document");
    docs.sort_by_key(|row| str_field(row, "source_markdown"));
    docs
}

fn emit_docs_root_narratives_mirror(args: DocsRootNarrativesMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "docs_root_narratives")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "Docs Root Narratives Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Document count: `{}`",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for doc in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(doc, "id"),
            str_field(doc, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(doc, "source_markdown")
        ));
        lines.push(format!("- Slug: `{}`", str_field(doc, "slug")));
        lines.push(format!(
            "- Status token: `{}`",
            str_field(doc, "status_token")
        ));
        lines.push(format!(
            "- Content kind: `{}`",
            str_field(doc, "content_kind")
        ));
        lines.push(format!("- Line count: {}", int_field(doc, "line_count")));
        let claims = array_of_strings(doc, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted docs-root narratives markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_docs_root_narratives_legacy(args: DocsRootNarrativesLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    let source = args.input.display().to_string();
    for doc in docs {
        let rel_path = str_field(doc, "source_markdown");
        if rel_path.is_empty() {
            continue;
        }
        let path = args.repo_root.join(&rel_path);
        let mut lines = generated_doc_header(&source);
        let body = raw_str_field(doc, "body_markdown");
        if body.is_empty() {
            lines.push(format!(
                "# {}",
                fallback_title(&rel_path, &str_field(doc, "title"))
            ));
            lines.push(String::new());
            lines.push(
                "(No body_markdown captured in registry/docs_root_narratives.toml.)".to_string(),
            );
        } else {
            lines.extend(body.lines().map(ToString::to_string));
        }
        lines.push(String::new());
        write_output(&path, &lines.join("\n"), args.allow_unicode)?;
    }
    println!(
        "Emitted legacy docs-root narratives from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_research_narratives_mirror(args: ResearchNarrativesMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "research_narratives")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "Research Narratives Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    let globs = array_of_strings(meta, "source_markdown_globs");
    if !globs.is_empty() {
        lines.push(format!(
            "- Source markdown globs: {}",
            globs
                .iter()
                .map(|item| format!("`{}`", item))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    lines.push(format!(
        "- Document count: `{}`",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for doc in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(doc, "id"),
            str_field(doc, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(doc, "source_markdown")
        ));
        lines.push(format!("- Domain: `{}`", str_field(doc, "domain")));
        lines.push(format!(
            "- Status token: `{}`",
            str_field(doc, "status_token")
        ));
        lines.push(format!(
            "- Content kind: `{}`",
            str_field(doc, "content_kind")
        ));
        lines.push(format!(
            "- Verification level: `{}`",
            str_field(doc, "verification_level")
        ));
        lines.push(format!("- Line count: {}", int_field(doc, "line_count")));
        let claims = array_of_strings(doc, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted research narratives markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_research_narratives_legacy(args: ResearchNarrativesLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    let source = args.input.display().to_string();
    let mut theory_index = vec![
        "# Theory Narratives".to_string(),
        String::new(),
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        "<!-- Source of truth: registry/research_narratives.toml -->".to_string(),
        String::new(),
        "This index and all files under `docs/theory/*.md` are generated from TOML.".to_string(),
        String::new(),
    ];
    let mut engineering_index = vec![
        "# Engineering Narratives".to_string(),
        String::new(),
        "<!-- AUTO-GENERATED: DO NOT EDIT -->".to_string(),
        "<!-- Source of truth: registry/research_narratives.toml -->".to_string(),
        String::new(),
        "This index and all files under `docs/engineering/*.md` are generated from TOML."
            .to_string(),
        String::new(),
    ];
    for doc in docs {
        let rel_path = str_field(doc, "source_markdown");
        if rel_path.is_empty() {
            continue;
        }
        let path = args.repo_root.join(&rel_path);
        let mut lines = generated_doc_header(&source);
        let body = raw_str_field(doc, "body_markdown");
        if body.is_empty() {
            lines.push(format!(
                "# {}",
                fallback_title(&rel_path, &str_field(doc, "title"))
            ));
            lines.push(String::new());
            lines.push(
                "(No body_markdown captured in registry/research_narratives.toml.)".to_string(),
            );
        } else {
            lines.extend(body.lines().map(ToString::to_string));
        }
        lines.push(String::new());
        write_output(&path, &lines.join("\n"), args.allow_unicode)?;

        let entry = format!(
            "- `{}` `{}`: `{}`",
            str_field(doc, "id"),
            str_field(doc, "status_token"),
            rel_path
        );
        if rel_path.starts_with("docs/theory/") {
            theory_index.push(entry);
        } else if rel_path.starts_with("docs/engineering/") {
            engineering_index.push(entry);
        }
    }
    theory_index.push(String::new());
    engineering_index.push(String::new());
    write_output(
        &args.repo_root.join("docs/theory/INDEX.md"),
        &theory_index.join("\n"),
        args.allow_unicode,
    )?;
    write_output(
        &args.repo_root.join("docs/engineering/INDEX.md"),
        &engineering_index.join("\n"),
        args.allow_unicode,
    )?;
    println!(
        "Emitted legacy research narratives from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn narrative_overlay_map(
    path: &Path,
    section_key: &str,
) -> Result<(String, std::collections::BTreeMap<String, String>), String> {
    if !path.exists() {
        return Ok((String::new(), std::collections::BTreeMap::new()));
    }
    let data = read_toml_value(path)?;
    let preamble = data
        .get(section_key)
        .and_then(Value::as_table)
        .map(|section| {
            raw_str_field(section, "preamble_markdown")
                .trim()
                .to_string()
        })
        .unwrap_or_default();
    let mut body_by_id = std::collections::BTreeMap::new();
    for row in rows(&data, "entry") {
        let id = str_field(row, "id");
        if !id.is_empty() {
            body_by_id.insert(id, raw_str_field(row, "body_markdown").trim().to_string());
        }
    }
    Ok((preamble, body_by_id))
}

fn emit_insights_mirror(args: InsightsMirrorArgs) -> Result<(), String> {
    let (data, source_label) = read_control_plane_compat_value(
        &args.input,
        &args.canonical_db,
        ControlPlaneCompatKind::Insights,
    )?;
    let mut insights = rows(&data, "insight");
    insights.sort_by_key(|row| str_field(row, "id"));
    let mut lines = control_plane_markdown_header("Insights Registry Mirror", &source_label);
    lines.push(format!("Total insights: {}", insights.len()));
    lines.push(String::new());
    for row in insights {
        let claims = array_of_strings(row, "claims");
        lines.push(format!(
            "## {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!("- Date: {}", str_field(row, "date")));
        lines.push(format!("- Status: {}", str_field(row, "status")));
        lines.push(format!("- Sprint: {}", int_field(row, "sprint")));
        lines.push(format!(
            "- Claims: {}",
            if claims.is_empty() {
                "(none)".to_string()
            } else {
                claims.join(", ")
            }
        ));
        lines.push(String::new());
        lines.push(str_field(row, "summary"));
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted insights markdown mirror from {} to {}.",
        source_label,
        args.output.display()
    );
    Ok(())
}

fn emit_insights_legacy(args: InsightsLegacyArgs) -> Result<(), String> {
    let (data, source_label) = read_control_plane_compat_value(
        &args.input,
        &args.canonical_db,
        ControlPlaneCompatKind::Insights,
    )?;
    let (preamble, body_by_id) = narrative_overlay_map(&args.narrative, "insights_narrative")?;
    let mut insights = rows(&data, "insight");
    insights.sort_by_key(|row| str_field(row, "id"));
    let mut lines = generated_doc_header(
        "registry/canonical/control_plane.sqlite3, registry/insights_narrative.toml",
    );
    if !preamble.is_empty() {
        lines.extend(preamble.lines().map(ToString::to_string));
    } else {
        lines.extend([
            "# Insights".to_string(),
            String::new(),
            "Source-of-truth policy:".to_string(),
            "- Authoritative machine-readable source: `registry/canonical/control_plane.sqlite3`"
                .to_string(),
            "- SQLite-exported compatibility view: `registry/insights.toml`".to_string(),
            "- Narrative overlay registry: `registry/insights_narrative.toml`".to_string(),
            "- TOML-driven markdown mirror: `crates/data_core/src/registry_mirrors/insights_registry_mirror.rs`"
                .to_string(),
            "- This file is generated from canonical SQLite control-plane data plus the narrative overlay.".to_string(),
            String::new(),
        ]);
    }
    lines.push(String::new());
    for row in insights {
        let insight_id = str_field(row, "id");
        lines.push(format!("## {}: {}", insight_id, str_field(row, "title")));
        lines.push(String::new());
        if let Some(body) = body_by_id.get(&insight_id)
            && !body.is_empty()
        {
            lines.extend(body.lines().map(ToString::to_string));
        } else {
            let claims = array_of_strings(row, "claims");
            lines.push(format!("Date: {}", str_field(row, "date")));
            lines.push(format!("Status: {}", str_field(row, "status")));
            lines.push(format!(
                "Claims: {}",
                if claims.is_empty() {
                    "(none)".to_string()
                } else {
                    claims.join(", ")
                }
            ));
            lines.push(String::new());
            lines.push(str_field(row, "summary"));
        }
        lines.push(String::new());
        lines.push("---".to_string());
        lines.push(String::new());
    }
    while lines
        .last()
        .map(|line| line.trim().is_empty())
        .unwrap_or(false)
    {
        lines.pop();
    }
    lines.push(String::new());
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted insights legacy markdown from {} and {} to {}.",
        source_label,
        args.narrative.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_claims_mirror(args: ClaimsMirrorArgs) -> Result<(), String> {
    let (data, source_label) = read_control_plane_compat_value(
        &args.input,
        &args.canonical_db,
        ControlPlaneCompatKind::Claims,
    )?;
    let mut claims = rows(&data, "claim");
    claims.sort_by_key(|row| claim_sort_key(&str_field(row, "id")));
    let mut lines = control_plane_markdown_header("Claims Registry Mirror", &source_label);
    lines.push(format!("Total claims: {}", claims.len()));
    lines.push(String::new());
    for row in claims {
        lines.push(format!("## {}", str_field(row, "id")));
        lines.push(String::new());
        lines.push(format!("- Status: `{}`", str_field(row, "status")));
        lines.push(format!(
            "- Last verified: {}",
            str_field(row, "last_verified")
        ));
        lines.push(format!("- Statement: {}", str_field(row, "statement")));
        lines.push(format!(
            "- Where stated: {}",
            str_field(row, "where_stated")
        ));
        lines.push(format!(
            "- What would verify/refute it: {}",
            str_field(row, "what_would_verify_refute")
        ));
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted claims markdown mirror from {} to {}.",
        source_label,
        args.output.display()
    );
    Ok(())
}

fn emit_claims_matrix_legacy(args: ClaimsMatrixLegacyArgs) -> Result<(), String> {
    let (data, source_label) = read_control_plane_compat_value(
        &args.input,
        &args.canonical_db,
        ControlPlaneCompatKind::Claims,
    )?;
    let mut claims = rows(&data, "claim");
    claims.sort_by_key(|row| claim_sort_key(&str_field(row, "id")));
    let mut lines = generated_doc_header("registry/canonical/control_plane.sqlite3");
    lines.extend([
        "# Claims / Evidence Matrix (Markdown Mirror)".to_string(),
        String::new(),
        "This file is generated from the canonical SQLite control plane (`registry/canonical/control_plane.sqlite3`).".to_string(),
        String::new(),
        "| ID | Claim | Where stated | Status | Last verified | What would verify/refute it |"
            .to_string(),
        "|---:|---|---|---|---|---|".to_string(),
    ]);
    for row in claims {
        lines.push(format!(
            "| {} | {} | {} | **{}** | {} | {} |",
            str_field(row, "id"),
            pipe_escape(&str_field(row, "statement")),
            pipe_escape(&str_field(row, "where_stated")),
            pipe_escape(&str_field(row, "status")),
            pipe_escape(&str_field(row, "last_verified")),
            pipe_escape(&str_field(row, "what_would_verify_refute"))
        ));
    }
    lines.push(String::new());
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted claims matrix legacy markdown from {} to {}.",
        source_label,
        args.output.display()
    );
    Ok(())
}

fn emit_bibliography_mirror(args: BibliographyMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "bibliography")?;
    let groups = rows(&data, "group");
    let entries = rows(&data, "entry");
    let mut lines = markdown_header(
        "Bibliography Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown: `{}`",
        str_field(meta, "source_markdown")
    ));
    lines.push(format!(
        "- Group count: {}",
        int_field(meta, "group_count").max(groups.len() as i64)
    ));
    lines.push(format!(
        "- Entry count: {}",
        int_field(meta, "entry_count").max(entries.len() as i64)
    ));
    lines.push(String::new());
    for group in groups {
        let group_name = str_field(group, "name");
        lines.push(format!("## {}", group_name));
        lines.push(String::new());
        let mut group_entries = entries
            .iter()
            .copied()
            .filter(|entry| str_field(entry, "group") == group_name)
            .collect::<Vec<_>>();
        group_entries.sort_by_key(|entry| int_field(entry, "order_index"));
        for entry in group_entries {
            lines.push(format!("- {}", str_field(entry, "citation_markdown")));
            for note in array_of_strings(entry, "notes") {
                lines.push(format!("  - {}", note));
            }
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted bibliography markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_bibliography_legacy(args: BibliographyLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let groups = rows(&data, "group");
    let entries = rows(&data, "entry");
    let mut lines = generated_doc_header("registry/bibliography.toml");
    lines.extend([
        "# Unified Bibliography".to_string(),
        String::new(),
        "This file is generated from `registry/bibliography.toml`.".to_string(),
        String::new(),
    ]);
    for group in groups {
        let group_name = str_field(group, "name");
        lines.push(format!("## {}", group_name));
        lines.push(String::new());
        let mut section_names = Vec::<String>::new();
        for entry in &entries {
            if str_field(entry, "group") != group_name {
                continue;
            }
            let section_name = {
                let raw = str_field(entry, "section");
                if raw.is_empty() {
                    "Unscoped".to_string()
                } else {
                    raw
                }
            };
            if !section_names.contains(&section_name) {
                section_names.push(section_name);
            }
        }
        for section_name in section_names {
            lines.push(format!("### {}", section_name));
            let mut section_entries = entries
                .iter()
                .copied()
                .filter(|entry| {
                    str_field(entry, "group") == group_name && {
                        let raw = str_field(entry, "section");
                        if raw.is_empty() {
                            "Unscoped".to_string()
                        } else {
                            raw
                        }
                    } == section_name
                })
                .collect::<Vec<_>>();
            section_entries.sort_by_key(|entry| int_field(entry, "order_index"));
            for entry in section_entries {
                lines.push(format!("*   {}", str_field(entry, "citation_markdown")));
                for note in array_of_strings(entry, "notes") {
                    lines.push(format!("    *   {}", note));
                }
            }
            lines.push(String::new());
        }
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted bibliography legacy markdown from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_experiments_mirror(args: ExperimentsMirrorArgs) -> Result<(), String> {
    let (data, source_label) = read_control_plane_compat_value(
        &args.input,
        &args.canonical_db,
        ControlPlaneCompatKind::Experiments,
    )?;
    let mut experiments = rows(&data, "experiment");
    experiments.sort_by_key(|row| str_field(row, "id"));
    let mut lines = control_plane_markdown_header("Experiments Registry Mirror", &source_label);
    lines.push(format!("Total experiments: {}", experiments.len()));
    lines.push(String::new());
    for row in experiments {
        lines.push(format!(
            "## {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!("- Binary: `{}`", str_field(row, "binary")));
        lines.push(format!("- Input: {}", str_field(row, "input")));
        let outputs = array_of_strings(row, "output");
        lines.push(format!(
            "- Output: {}",
            if outputs.is_empty() {
                "(none)".to_string()
            } else {
                outputs.join(", ")
            }
        ));
        lines.push(format!(
            "- Deterministic: `{}`",
            bool_field(row, "deterministic")
        ));
        if row.contains_key("seed") {
            lines.push(format!("- Seed: `{}`", int_field(row, "seed")));
        }
        lines.push(format!("- GPU: `{}`", bool_field(row, "gpu")));
        let claims = array_of_strings(row, "claims");
        lines.push(format!(
            "- Claims: {}",
            if claims.is_empty() {
                "(none)".to_string()
            } else {
                claims.join(", ")
            }
        ));
        lines.push(String::new());
        lines.push("Method:".to_string());
        lines.push(str_field(row, "method"));
        lines.push(String::new());
        lines.push("Run command:".to_string());
        lines.push("```bash".to_string());
        lines.push(str_field(row, "run"));
        lines.push("```".to_string());
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted experiments markdown mirror from {} to {}.",
        source_label,
        args.output.display()
    );
    Ok(())
}

fn emit_experiments_legacy(args: ExperimentsLegacyArgs) -> Result<(), String> {
    let (data, source_label) = read_control_plane_compat_value(
        &args.input,
        &args.canonical_db,
        ControlPlaneCompatKind::Experiments,
    )?;
    let (preamble, body_by_id) = narrative_overlay_map(&args.narrative, "experiments_narrative")?;
    let mut experiments = rows(&data, "experiment");
    experiments.sort_by_key(|row| str_field(row, "id"));
    let mut lines = generated_doc_header(
        "registry/canonical/control_plane.sqlite3, registry/experiments_narrative.toml",
    );
    if !preamble.is_empty() {
        lines.extend(preamble.lines().map(ToString::to_string));
    } else {
        lines.extend([
            "# Experiments Portfolio Shortlist".to_string(),
            String::new(),
            "Source-of-truth policy:".to_string(),
            "- Authoritative machine-readable source: `registry/canonical/control_plane.sqlite3`"
                .to_string(),
            "- SQLite-exported compatibility view: `registry/experiments.toml`".to_string(),
            "- Narrative overlay registry: `registry/experiments_narrative.toml`".to_string(),
            "- TOML-driven markdown mirror: `crates/data_core/src/registry_mirrors/experiments_registry_mirror.rs`"
                .to_string(),
            "- This file is generated from canonical SQLite control-plane data plus the narrative overlay.".to_string(),
            String::new(),
        ]);
    }
    lines.push(String::new());
    for row in experiments {
        let experiment_id = str_field(row, "id");
        lines.push(format!("## {}: {}", experiment_id, str_field(row, "title")));
        lines.push(String::new());
        if let Some(body) = body_by_id.get(&experiment_id)
            && !body.is_empty()
        {
            lines.extend(body.lines().map(ToString::to_string));
        } else {
            lines.push(format!("Method: {}", str_field(row, "method")));
            lines.push(format!("Input: {}", str_field(row, "input")));
            let outputs = array_of_strings(row, "output");
            lines.push(format!(
                "Output: {}",
                if outputs.is_empty() {
                    "(none)".to_string()
                } else {
                    outputs.join(", ")
                }
            ));
            lines.push("Run:".to_string());
            lines.push("```bash".to_string());
            lines.push(str_field(row, "run"));
            lines.push("```".to_string());
        }
        lines.push(String::new());
        lines.push("---".to_string());
        lines.push(String::new());
    }
    while lines
        .last()
        .map(|line| line.trim().is_empty())
        .unwrap_or(false)
    {
        lines.pop();
    }
    lines.push(String::new());
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted experiments legacy markdown from {} and {} to {}.",
        source_label,
        args.narrative.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_theorems_mirror(args: TheoremsMirrorArgs) -> Result<(), String> {
    let text =
        read_control_plane_compat_text(&args.canonical_db, ControlPlaneCompatKind::TheoremsMirror)?;
    write_output(&args.output, &text, args.allow_unicode)?;
    println!(
        "Emitted theorem markdown mirror from {} to {}.",
        args.canonical_db.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_theorems_legacy(args: TheoremsLegacyArgs) -> Result<(), String> {
    let text =
        read_control_plane_compat_text(&args.canonical_db, ControlPlaneCompatKind::Theorems)?;
    write_output(&args.output, &text, args.allow_unicode)?;
    println!(
        "Emitted theorem legacy markdown from {} to {}.",
        args.canonical_db.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_control_plane_docs(args: ControlPlaneDocsArgs) -> Result<(), String> {
    emit_claims_mirror(ClaimsMirrorArgs {
        input: args.claims_input.clone(),
        canonical_db: args.canonical_db.clone(),
        output: args.claims_mirror_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_claims_matrix_legacy(ClaimsMatrixLegacyArgs {
        input: args.claims_input.clone(),
        canonical_db: args.canonical_db.clone(),
        output: args.claims_legacy_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_insights_mirror(InsightsMirrorArgs {
        input: args.insights_input.clone(),
        canonical_db: args.canonical_db.clone(),
        output: args.insights_mirror_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_insights_legacy(InsightsLegacyArgs {
        input: args.insights_input.clone(),
        canonical_db: args.canonical_db.clone(),
        narrative: args.insights_narrative.clone(),
        output: args.insights_legacy_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_experiments_mirror(ExperimentsMirrorArgs {
        input: args.experiments_input.clone(),
        canonical_db: args.canonical_db.clone(),
        output: args.experiments_mirror_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_experiments_legacy(ExperimentsLegacyArgs {
        input: args.experiments_input.clone(),
        canonical_db: args.canonical_db.clone(),
        narrative: args.experiments_narrative.clone(),
        output: args.experiments_legacy_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_theorems_mirror(TheoremsMirrorArgs {
        canonical_db: args.canonical_db.clone(),
        output: args.theorems_mirror_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    emit_theorems_legacy(TheoremsLegacyArgs {
        canonical_db: args.canonical_db.clone(),
        output: args.theorems_legacy_output.clone(),
        allow_unicode: args.allow_unicode,
    })?;
    println!(
        "Emitted standard control-plane docs bundle from {}.",
        args.canonical_db.display()
    );
    Ok(())
}

fn emit_markdown_governance_mirror(args: MarkdownGovernanceMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "markdown_governance")?;
    let mut docs = rows(&data, "document");
    docs.sort_by_key(|row| str_field(row, "path"));
    let mut lines = markdown_header(
        "Markdown Governance Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!(
        "- Generated at: {}",
        str_field(meta, "generated_at")
    ));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(format!(
        "- TOML generated mirrors: {}",
        int_field(meta, "toml_generated_mirror_count")
    ));
    lines.push(format!(
        "- TOML manual sources: {}",
        int_field(meta, "toml_manual_source_count")
    ));
    lines.push(format!(
        "- Generated artifacts: {}",
        int_field(meta, "generated_artifact_count")
    ));
    lines.push(format!(
        "- Manual narratives: {}",
        int_field(meta, "manual_narrative_count")
    ));
    lines.push(format!(
        "- Immutable transcripts: {}",
        int_field(meta, "immutable_transcript_count")
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for row in docs {
        lines.push(format!(
            "### {}: `{}`",
            str_field(row, "id"),
            str_field(row, "path")
        ));
        lines.push(String::new());
        lines.push(format!("- Kind: `{}`", str_field(row, "kind")));
        lines.push(format!("- Mode: `{}`", str_field(row, "mode")));
        lines.push(format!(
            "- Header required: `{}`",
            bool_field(row, "header_required")
        ));
        let refs = array_of_strings(row, "source_toml_refs");
        if !refs.is_empty() {
            lines.push("- Source TOML refs:".to_string());
            for reference in refs {
                lines.push(format!("  - `{}`", reference));
            }
        }
        let notes = str_field(row, "notes");
        if !notes.is_empty() {
            lines.push(format!("- Notes: {}", notes));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted markdown-governance markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn bool_field(table: &toml::map::Map<String, Value>, key: &str) -> bool {
    table.get(key).and_then(Value::as_bool).unwrap_or(false)
}

fn pipe_escape(text: &str) -> String {
    text.replace('|', "\\|")
}

fn claim_sort_key(claim_id: &str) -> i64 {
    claim_id
        .strip_prefix("C-")
        .and_then(|tail| tail.parse::<i64>().ok())
        .unwrap_or(999_999)
}

fn render_csv(rows: &[Vec<String>]) -> Result<String, String> {
    let mut writer = WriterBuilder::new()
        .has_headers(false)
        .from_writer(Vec::<u8>::new());
    for row in rows {
        writer
            .write_record(row)
            .map_err(|err| format!("write csv row: {}", err))?;
    }
    let bytes = writer
        .into_inner()
        .map_err(|err| format!("finalize csv writer: {}", err.error()))?;
    String::from_utf8(bytes).map_err(|err| format!("csv utf8: {}", err))
}

fn emit_body_markdown_docs(
    repo_root: &Path,
    docs: &[&toml::map::Map<String, Value>],
    source: &str,
    body_missing_message: &str,
    rel_key: &str,
    allow_unicode: bool,
) -> Result<(), String> {
    for doc in docs {
        let rel_path = str_field(doc, rel_key);
        if rel_path.is_empty() {
            continue;
        }
        let body = raw_str_field(doc, "body_markdown");
        let mut lines = if body.starts_with("<!-- AUTO-GENERATED: DO NOT EDIT -->") {
            Vec::new()
        } else {
            generated_doc_header(source)
        };
        if body.is_empty() {
            lines.push(format!(
                "# {}",
                fallback_title(&rel_path, &str_field(doc, "title"))
            ));
            lines.push(String::new());
            lines.push(body_missing_message.to_string());
        } else {
            lines.extend(body.lines().map(ToString::to_string));
        }
        lines.push(String::new());
        write_output(&repo_root.join(&rel_path), &lines.join("\n"), allow_unicode)?;
    }
    Ok(())
}

fn emit_claims_tasks_mirror(args: ClaimsTasksMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "claims_tasks")?;
    let sections = rows(&data, "section");
    let tasks = rows(&data, "task");
    let mut lines = markdown_header(
        "Claims Tasks Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown: `{}`",
        str_field(meta, "source_markdown")
    ));
    lines.push(format!(
        "- Task count: {}",
        int_field(meta, "task_count").max(tasks.len() as i64)
    ));
    lines.push(format!(
        "- Section count: {}",
        int_field(meta, "section_count").max(sections.len() as i64)
    ));
    lines.push(format!(
        "- Canonical status task count: {}",
        int_field(meta, "canonical_status_task_count")
    ));
    lines.push(format!(
        "- Noncanonical status task count: {}",
        int_field(meta, "noncanonical_status_task_count")
    ));
    lines.push(String::new());
    lines.push("## Sections".to_string());
    lines.push(String::new());
    for section in sections {
        lines.push(format!(
            "- {}: {} ({} tasks)",
            str_field(section, "id"),
            str_field(section, "name"),
            int_field(section, "task_count")
        ));
    }
    lines.push(String::new());
    lines.push("## Tasks".to_string());
    lines.push(String::new());
    let mut ordered_tasks = tasks;
    ordered_tasks.sort_by_key(|task| int_field(task, "order_index"));
    for task in ordered_tasks {
        lines.push(format!(
            "### {} ({}, {})",
            str_field(task, "id"),
            str_field(task, "claim_id"),
            str_field(task, "status_token")
        ));
        lines.push(String::new());
        lines.push(format!("- Section: {}", str_field(task, "section")));
        lines.push(format!("- Source line: {}", int_field(task, "source_line")));
        lines.push(format!("- Status raw: {}", str_field(task, "status_raw")));
        lines.push(format!(
            "- Canonical: `{}`",
            bool_field(task, "status_canonical")
        ));
        lines.push(String::new());
        lines.push(str_field(task, "task"));
        lines.push(String::new());
        let artifacts = array_of_strings(task, "output_artifacts");
        if !artifacts.is_empty() {
            lines.push("Output artifacts:".to_string());
            for artifact in artifacts {
                lines.push(format!("- `{}`", artifact));
            }
            lines.push(String::new());
        }
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted claims-tasks markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_claims_tasks_legacy(args: ClaimsTasksLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let sections = rows(&data, "section");
    let tasks = rows(&data, "task");
    let mut tasks_by_section =
        std::collections::BTreeMap::<String, Vec<&toml::map::Map<String, Value>>>::new();
    for task in &tasks {
        tasks_by_section
            .entry(str_field(task, "section"))
            .or_default()
            .push(*task);
    }
    let mut ordered_sections = sections;
    ordered_sections.sort_by_key(|section| str_field(section, "id"));

    let mut lines = generated_doc_header("registry/claims_tasks.toml");
    lines.push("# Claims -> Tasks Tracker (Generated Mirror)".to_string());
    lines.push(String::new());
    lines.push("This file is generated from `registry/claims_tasks.toml`.".to_string());
    lines.push(String::new());
    for section in ordered_sections {
        let name = str_field(section, "name");
        lines.push(format!("## {}", name));
        lines.push(String::new());
        let mut section_tasks = tasks_by_section.remove(&name).unwrap_or_default();
        section_tasks.sort_by_key(|task| int_field(task, "order_index"));
        if section_tasks.is_empty() {
            lines.push("No task rows currently in this section.".to_string());
            lines.push(String::new());
            continue;
        }
        lines.push("| Claim ID | Task | Output artifact(s) | Status |".to_string());
        lines.push("|---|---|---|---|".to_string());
        for task in section_tasks {
            let artifacts = array_of_strings(task, "output_artifacts");
            let artifact_cell = if artifacts.is_empty() {
                "(none)".to_string()
            } else {
                artifacts
                    .iter()
                    .map(|item| format!("`{}`", item))
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            lines.push(format!(
                "| {} | {} | {} | {} |",
                str_field(task, "claim_id"),
                pipe_escape(&str_field(task, "task")),
                artifact_cell,
                str_field(task, "status_token")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted claims-tasks legacy markdown from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_claims_domains_mirror(args: ClaimsDomainsMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "claims_domains")?;
    let domains = rows(&data, "domain");
    let claim_domains = rows(&data, "claim_domain");
    let mut lines = markdown_header(
        "Claims Domains Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!("- Source CSV: `{}`", str_field(meta, "source_csv")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Domain file count: {}",
        int_field(meta, "domain_file_count").max(domains.len() as i64)
    ));
    lines.push(format!(
        "- Claim count: {}",
        int_field(meta, "claim_count").max(claim_domains.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Domain Files".to_string());
    lines.push(String::new());
    let mut ordered_domains = domains;
    ordered_domains.sort_by_key(|row| str_field(row, "id"));
    for row in ordered_domains {
        lines.push(format!("### {}", str_field(row, "id")));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!(
            "- Declared count: {}",
            int_field(row, "declared_count")
        ));
        lines.push(format!(
            "- CSV claim count: {}",
            int_field(row, "csv_claim_count")
        ));
        lines.push(format!(
            "- Markdown claim count: {}",
            int_field(row, "markdown_claim_count")
        ));
        lines.push(format!(
            "- Count match: `{}`",
            bool_field(row, "count_match")
        ));
        lines.push(format!(
            "- Mapping match: `{}`",
            bool_field(row, "mapping_match")
        ));
        lines.push(String::new());
    }
    lines.push("## Claim Crosswalk".to_string());
    lines.push(String::new());
    let mut ordered_claim_domains = claim_domains;
    ordered_claim_domains.sort_by_key(|row| str_field(row, "claim_id"));
    for row in ordered_claim_domains {
        lines.push(format!(
            "- {}: csv={:?}, markdown={:?}, match={}",
            str_field(row, "claim_id"),
            array_of_strings(row, "domains_csv"),
            array_of_strings(row, "domains_markdown"),
            bool_field(row, "domain_sets_match")
        ));
    }
    lines.push(String::new());
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted claims-domains markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_claims_domains_legacy(args: ClaimsDomainsLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let mut domains = rows(&data, "domain");
    domains.sort_by_key(|row| str_field(row, "id"));
    let mut claim_domains = rows(&data, "claim_domain");
    claim_domains.sort_by_key(|row| str_field(row, "claim_id"));
    let entries = rows(&data, "domain_entry");

    let mut index_lines = generated_doc_header("registry/claims_domains.toml");
    index_lines.push("# Claims by domain".to_string());
    index_lines.push(String::new());
    index_lines.push(
        "See also: crates/data_core/src/registry_mirrors/claims_domain_taxonomy.rs".to_string(),
    );
    index_lines.push(String::new());
    for row in &domains {
        index_lines.push(format!(
            "- `{}` ({}): `{}`",
            str_field(row, "id"),
            int_field(row, "markdown_claim_count"),
            str_field(row, "source_markdown")
        ));
    }
    index_lines.push(String::new());
    write_output(
        &args.repo_root.join("docs/claims/INDEX.md"),
        &index_lines.join("\n"),
        args.allow_unicode,
    )?;

    let mut csv_rows = vec![vec!["claim_id".to_string(), "domains".to_string()]];
    for row in &claim_domains {
        csv_rows.push(vec![
            str_field(row, "claim_id"),
            array_of_strings(row, "domains_csv").join(";"),
        ]);
    }
    let csv_text = render_csv(&csv_rows)?;
    write_output(
        &args.repo_root.join("docs/claims/CLAIMS_DOMAIN_MAP.csv"),
        &csv_text,
        args.allow_unicode,
    )?;

    let mut entries_by_domain =
        std::collections::BTreeMap::<String, Vec<&toml::map::Map<String, Value>>>::new();
    for entry in &entries {
        entries_by_domain
            .entry(str_field(entry, "domain"))
            .or_default()
            .push(*entry);
    }
    for row in domains {
        let domain_id = str_field(row, "id");
        let rel_path = str_field(row, "source_markdown");
        let mut domain_entries = entries_by_domain.remove(&domain_id).unwrap_or_default();
        domain_entries.sort_by_key(|entry| claim_sort_key(&str_field(entry, "claim_id")));
        let mut lines = generated_doc_header("registry/claims_domains.toml");
        lines.push(format!("# Claims: {}", domain_id));
        lines.push(String::new());
        lines.push(format!("Count: {}", domain_entries.len()));
        lines.push(String::new());
        for entry in domain_entries {
            let status_text = str_field(entry, "status_text");
            let status_date = str_field(entry, "status_date");
            let status_blob = if status_date.is_empty() {
                status_text
            } else {
                format!("{}, {}", status_text, status_date)
            };
            lines.push(format!(
                "- Hypothesis {} ({}): {}",
                str_field(entry, "claim_id"),
                status_blob,
                str_field(entry, "summary")
            ));
            let where_stated = array_of_strings(entry, "where_stated");
            if !where_stated.is_empty() {
                lines.push(format!(
                    "  - Where stated: {}",
                    where_stated
                        .iter()
                        .map(|item| format!("`{}`", item))
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            lines.push(String::new());
        }
        write_output(
            &args.repo_root.join(rel_path),
            &lines.join("\n"),
            args.allow_unicode,
        )?;
    }
    println!(
        "Emitted claims-domains legacy markdown/csv from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_claim_tickets_mirror(args: ClaimTicketsMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "claim_tickets")?;
    let mut tickets = rows(&data, "ticket");
    tickets.sort_by_key(|row| str_field(row, "id"));
    let mut lines = markdown_header(
        "Claim Tickets Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Ticket count: {}",
        int_field(meta, "ticket_count").max(tickets.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Tickets".to_string());
    lines.push(String::new());
    for row in tickets {
        lines.push(format!(
            "### {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!("- Kind: `{}`", str_field(row, "ticket_kind")));
        lines.push(format!("- Owner: {}", str_field(row, "owner")));
        lines.push(format!("- Created: {}", str_field(row, "created")));
        lines.push(format!(
            "- Status: `{}` ({})",
            str_field(row, "status_token"),
            str_field(row, "status_raw")
        ));
        lines.push(format!(
            "- Claim range: {}..{}",
            int_field(row, "claim_range_start"),
            int_field(row, "claim_range_end")
        ));
        lines.push(format!(
            "- Checkbox progress: done={}, open={}",
            int_field(row, "done_checkboxes"),
            int_field(row, "open_checkboxes")
        ));
        let claims = array_of_strings(row, "claims_referenced");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claims referenced ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        let backlog = array_of_strings(row, "backlog_reports");
        if !backlog.is_empty() {
            lines.push("- Backlog reports:".to_string());
            for item in backlog {
                lines.push(format!("  - `{}`", item));
            }
        }
        let checks = array_of_strings(row, "acceptance_checks");
        if !checks.is_empty() {
            lines.push("- Acceptance checks:".to_string());
            for item in checks {
                lines.push(format!("  - `{}`", item));
            }
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted claim-tickets markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_claim_tickets_legacy(args: ClaimTicketsLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let mut tickets = rows(&data, "ticket");
    tickets.sort_by_key(|row| str_field(row, "source_markdown"));

    let mut index_lines = generated_doc_header("registry/claim_tickets.toml");
    index_lines.push("# Claim Audit Tickets".to_string());
    index_lines.push(String::new());
    index_lines.push(
        "This index and all files under `docs/tickets/*.md` are generated from TOML.".to_string(),
    );
    index_lines.push(String::new());

    for row in tickets {
        let rel_path = str_field(row, "source_markdown");
        if rel_path.is_empty() {
            continue;
        }
        let mut lines = generated_doc_header("registry/claim_tickets.toml");
        lines.push(format!("# {}", str_field(row, "title")));
        lines.push(String::new());
        lines.push(format!("Owner: {}", str_field(row, "owner")));
        lines.push(format!("Created: {}", str_field(row, "created")));
        lines.push(format!("Status: {}", str_field(row, "status_raw")));
        lines.push(String::new());
        lines.push("## Goal".to_string());
        lines.push(String::new());
        let goal = str_field(row, "goal_summary");
        lines.push(if goal.is_empty() {
            "(not specified)".to_string()
        } else {
            goal
        });
        lines.push(String::new());
        lines.push("## Scope".to_string());
        lines.push(String::new());
        lines.push(format!("- Ticket ID: `{}`", str_field(row, "id")));
        lines.push(format!("- Kind: `{}`", str_field(row, "ticket_kind")));
        lines.push(format!(
            "- Status token: `{}`",
            str_field(row, "status_token")
        ));
        let claim_start = int_field(row, "claim_range_start");
        let claim_end = int_field(row, "claim_range_end");
        if claim_start > 0 && claim_end > 0 {
            lines.push(format!(
                "- Claim range: C-{:03}..C-{:03}",
                claim_start, claim_end
            ));
        } else {
            lines.push("- Claim range: none (general ticket)".to_string());
        }
        let claims = array_of_strings(row, "claims_referenced");
        if claims.is_empty() {
            lines.push("- Claims referenced: none".to_string());
        } else {
            lines.push(format!(
                "- Claims referenced ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
        lines.push("## Deliverables".to_string());
        lines.push(String::new());
        let deliverables = array_of_strings(row, "deliverable_links");
        if deliverables.is_empty() {
            lines.push("- (none recorded)".to_string());
        } else {
            for item in deliverables {
                lines.push(format!("- `{}`", item));
            }
        }
        lines.push(String::new());
        lines.push("## Acceptance checks".to_string());
        lines.push(String::new());
        let checks = array_of_strings(row, "acceptance_checks");
        if checks.is_empty() {
            lines.push("- (none recorded)".to_string());
        } else {
            for item in checks {
                lines.push(format!("- `{}`", item));
            }
        }
        lines.push(String::new());
        lines.push("## Progress snapshot".to_string());
        lines.push(String::new());
        lines.push(format!(
            "- Completed checkboxes: {}",
            int_field(row, "done_checkboxes")
        ));
        lines.push(format!(
            "- Open checkboxes: {}",
            int_field(row, "open_checkboxes")
        ));
        let backlog = array_of_strings(row, "backlog_reports");
        if !backlog.is_empty() {
            lines.push("- Backlog reports:".to_string());
            for item in backlog {
                lines.push(format!("  - `{}`", item));
            }
        }
        lines.push(String::new());
        write_output(
            &args.repo_root.join(&rel_path),
            &lines.join("\n"),
            args.allow_unicode,
        )?;
        index_lines.push(format!("- `{}`: `{}`", str_field(row, "id"), rel_path));
    }

    index_lines.push(String::new());
    write_output(
        &args.repo_root.join("docs/tickets/INDEX.md"),
        &index_lines.join("\n"),
        args.allow_unicode,
    )?;
    println!(
        "Emitted claim-ticket legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_external_sources_mirror(args: ExternalSourcesMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "external_sources")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "External Sources Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for row in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!("- Slug: `{}`", str_field(row, "slug")));
        lines.push(format!(
            "- Status token: `{}`",
            str_field(row, "status_token")
        ));
        lines.push(format!(
            "- Content kind: `{}`",
            str_field(row, "content_kind")
        ));
        lines.push(format!(
            "- Authority level: `{}`",
            str_field(row, "authority_level")
        ));
        lines.push(format!(
            "- Verification level: `{}`",
            str_field(row, "verification_level")
        ));
        lines.push(format!(
            "- Operational role: `{}`",
            str_field(row, "operational_role")
        ));
        let lineage = str_field(row, "source_lineage_summary");
        if !lineage.is_empty() {
            lines.push(format!("- Source lineage summary: {}", lineage));
        }
        let truth_surfaces = array_of_strings(row, "truth_surfaces");
        if !truth_surfaces.is_empty() {
            lines.push(format!("- Truth surfaces: {}", truth_surfaces.join(", ")));
        }
        let contracts = array_of_strings(row, "artifact_contract_paths");
        if !contracts.is_empty() {
            lines.push(format!("- Artifact contract paths ({}):", contracts.len()));
            for path in contracts {
                lines.push(format!("  - `{}`", path));
            }
        }
        lines.push(format!(
            "- Has full transcript: `{}`",
            bool_field(row, "has_full_transcript")
        ));
        lines.push(format!("- Line count: {}", int_field(row, "line_count")));
        let claims = array_of_strings(row, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        let urls = array_of_strings(row, "url_refs");
        if !urls.is_empty() {
            lines.push(format!("- URL refs ({}):", urls.len()));
            for url in urls.iter().take(10) {
                lines.push(format!("  - `{}`", url));
            }
            if urls.len() > 10 {
                lines.push(format!("  - ... ({} more)", urls.len() - 10));
            }
        }
        let notes = str_field(row, "notes");
        if !notes.is_empty() {
            lines.push(format!("- Notes: {}", notes));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted external-sources markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_external_sources_legacy(args: ExternalSourcesLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    emit_body_markdown_docs(
        &args.repo_root,
        &docs,
        "registry/external_sources.toml",
        "(No body_markdown captured in registry/external_sources.toml.)",
        "source_markdown",
        args.allow_unicode,
    )?;
    let mut index_lines = generated_doc_header("registry/external_sources.toml");
    index_lines.push("# External Sources".to_string());
    linesep(&mut index_lines);
    index_lines.push(
        "This index and all files under `docs/external_sources/*.md` are generated from TOML."
            .to_string(),
    );
    linesep(&mut index_lines);
    for row in docs {
        index_lines.push(format!(
            "- `{}` `{}`: `{}`",
            str_field(row, "id"),
            str_field(row, "status_token"),
            str_field(row, "source_markdown")
        ));
    }
    index_lines.push(String::new());
    write_output(
        &args.repo_root.join("docs/external_sources/INDEX.md"),
        &index_lines.join("\n"),
        args.allow_unicode,
    )?;
    println!(
        "Emitted external-sources legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_book_docs_mirror(args: BookDocsMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "book_docs")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "Book Docs Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for row in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!("- Section: `{}`", str_field(row, "section")));
        lines.push(format!("- Slug: `{}`", str_field(row, "slug")));
        lines.push(format!("- Line count: {}", int_field(row, "line_count")));
        let claims = array_of_strings(row, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted book-docs markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_book_docs_legacy(args: BookDocsLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    emit_body_markdown_docs(
        &args.repo_root,
        &docs,
        "registry/book_docs.toml",
        "(No body_markdown captured in registry/book_docs.toml.)",
        "source_markdown",
        args.allow_unicode,
    )?;
    println!(
        "Emitted book-docs legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn render_artifact_sections_from_scroll(scroll: &ArtifactScrollDoc) -> Vec<String> {
    let mut out = Vec::new();
    if let Some(sections) = &scroll.section {
        for section in sections {
            let title = section.title.clone().unwrap_or_default();
            if !title.is_empty() && title != "(root)" {
                out.push(format!(
                    "{} {}",
                    level_prefix(section.level.unwrap_or(2)),
                    title
                ));
                out.push(String::new());
            }
            let body = section.body_text.clone().unwrap_or_default();
            if !body.trim().is_empty() {
                out.extend(body.lines().map(ToString::to_string));
                out.push(String::new());
            }
        }
    }
    while out.last().map(|line| line.is_empty()).unwrap_or(false) {
        out.pop();
    }
    out
}

fn emit_data_artifact_narratives_mirror(
    args: DataArtifactNarrativesMirrorArgs,
) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "data_artifact_narratives")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "Data Artifact Narratives Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(
        "Authoritative source: `registry/artifact_scrolls.toml` (with per-document scrolls under `registry/knowledge/artifacts/`).".to_string()
    );
    lines.push(String::new());
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown count: {}",
        int_field(meta, "source_markdown_count").max(docs.len() as i64)
    ));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for row in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!(
            "- Content kind: `{}`",
            str_field(row, "content_kind")
        ));
        lines.push(format!("- Line count: {}", int_field(row, "line_count")));
        let claims = array_of_strings(row, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted data-artifact narratives markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_data_artifact_narratives_legacy(
    args: DataArtifactNarrativesLegacyArgs,
) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    let artifact_index: ArtifactScrollIndex =
        read_toml(&args.repo_root.join(&args.artifact_index))?;
    let scroll_by_source = artifact_index
        .scroll
        .into_iter()
        .filter_map(|row| {
            if row.source_markdown.trim().is_empty() || row.scroll_path.trim().is_empty() {
                None
            } else {
                Some((row.source_markdown, row.scroll_path))
            }
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    for row in docs {
        let rel_path = str_field(row, "source_markdown");
        if rel_path.is_empty() {
            continue;
        }
        let mut lines = generated_doc_header("registry/artifact_scrolls.toml");
        let mut rendered_from_scroll = false;
        if let Some(scroll_rel) = scroll_by_source.get(&rel_path) {
            let scroll_path = args.repo_root.join(scroll_rel);
            if scroll_path.exists() {
                let scroll: ArtifactScrollDoc = read_toml(&scroll_path)?;
                let rendered = render_artifact_sections_from_scroll(&scroll);
                if !rendered.is_empty() {
                    lines.extend(rendered);
                    rendered_from_scroll = true;
                }
            }
        }
        if !rendered_from_scroll {
            let body = raw_str_field(row, "body_markdown");
            if body.is_empty() {
                lines.push(format!(
                    "# {}",
                    fallback_title(&rel_path, &str_field(row, "title"))
                ));
                lines.push(String::new());
                lines.push(
                    "(No section/body content captured in registry/artifact_scrolls.toml.)"
                        .to_string(),
                );
            } else {
                lines.extend(body.lines().map(ToString::to_string));
            }
        }
        lines.push(String::new());
        write_output(
            &args.repo_root.join(&rel_path),
            &lines.join("\n"),
            args.allow_unicode,
        )?;
    }
    println!(
        "Emitted data-artifact narratives legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_reports_narratives_mirror(args: ReportsNarrativesMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "reports_narratives")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "Reports Narratives Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for row in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!("- Category: `{}`", str_field(row, "category")));
        lines.push(format!("- Line count: {}", int_field(row, "line_count")));
        let claims = array_of_strings(row, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted reports narratives markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_reports_narratives_legacy(args: ReportsNarrativesLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    emit_body_markdown_docs(
        &args.repo_root,
        &docs,
        "registry/reports_narratives.toml",
        "(No body_markdown captured in registry/reports_narratives.toml.)",
        "source_markdown",
        args.allow_unicode,
    )?;
    println!(
        "Emitted reports narratives legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_docs_convos_mirror(args: DocsConvosMirrorArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let meta = table(&data, "docs_convos")?;
    let docs = narrative_docs(&data);
    let mut lines = markdown_header(
        "Docs Convos Registry Mirror",
        &args.input.display().to_string(),
    );
    lines.push(format!("- Updated: {}", str_field(meta, "updated")));
    lines.push(format!(
        "- Source markdown glob: `{}`",
        str_field(meta, "source_markdown_glob")
    ));
    lines.push(format!(
        "- Document count: {}",
        int_field(meta, "document_count").max(docs.len() as i64)
    ));
    lines.push(String::new());
    lines.push("## Documents".to_string());
    lines.push(String::new());
    for row in docs {
        lines.push(format!(
            "### {}: {}",
            str_field(row, "id"),
            str_field(row, "title")
        ));
        lines.push(String::new());
        lines.push(format!(
            "- Source markdown: `{}`",
            str_field(row, "source_markdown")
        ));
        lines.push(format!(
            "- Content kind: `{}`",
            str_field(row, "content_kind")
        ));
        lines.push(format!("- Line count: {}", int_field(row, "line_count")));
        let claims = array_of_strings(row, "claim_refs");
        if !claims.is_empty() {
            lines.push(format!(
                "- Claim refs ({}): {}",
                claims.len(),
                claims.join(", ")
            ));
        }
        lines.push(String::new());
    }
    write_output(&args.output, &lines.join("\n"), args.allow_unicode)?;
    println!(
        "Emitted docs-convos markdown mirror from {} to {}.",
        args.input.display(),
        args.output.display()
    );
    Ok(())
}

fn emit_docs_convos_legacy(args: DocsConvosLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = narrative_docs(&data);
    emit_body_markdown_docs(
        &args.repo_root,
        &docs,
        "registry/docs_convos.toml",
        "(No body_markdown captured in registry/docs_convos.toml.)",
        "source_markdown",
        args.allow_unicode,
    )?;
    println!(
        "Emitted docs-convos legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn emit_monograph_legacy(args: MonographLegacyArgs) -> Result<(), String> {
    let data = read_toml_value(&args.input)?;
    let docs = rows(&data, "document");
    for doc in docs {
        let rel_path = str_field(doc, "path");
        if rel_path.is_empty() {
            continue;
        }
        let mut lines = generated_doc_header("registry/monograph.toml");
        let body = raw_str_field(doc, "body_markdown");
        if body.is_empty() {
            lines.push(format!(
                "# {}",
                fallback_title(&rel_path, &str_field(doc, "title"))
            ));
            lines.push(String::new());
            lines.push("(No body_markdown captured in registry/monograph.toml.)".to_string());
        } else {
            lines.extend(body.lines().map(ToString::to_string));
        }
        lines.push(String::new());
        write_output(
            &args.repo_root.join(&rel_path),
            &lines.join("\n"),
            args.allow_unicode,
        )?;
    }
    println!(
        "Emitted monograph legacy markdown from {} into {}.",
        args.input.display(),
        args.repo_root.display()
    );
    Ok(())
}

fn linesep(lines: &mut Vec<String>) {
    lines.push(String::new());
}

fn fallback_title(path: &str, preferred: &str) -> String {
    if !preferred.trim().is_empty() {
        return preferred.trim().to_string();
    }
    Path::new(path)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or(path)
        .to_string()
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
    let bytes = text.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        if bytes[i] == b'*' {
            let prev_star = i > 0 && bytes[i - 1] == b'*';
            let next_star = i + 1 < bytes.len() && bytes[i + 1] == b'*';
            if !prev_star && !next_star {
                let start = i + 1;
                let mut j = start;
                while j < bytes.len() {
                    if bytes[j] == b'*' {
                        let prev_is_star = j > 0 && bytes[j - 1] == b'*';
                        let next_is_star = j + 1 < bytes.len() && bytes[j + 1] == b'*';
                        if !prev_is_star && !next_is_star && j > start {
                            return Some(text[start..j].trim().to_string());
                        }
                    }
                    j += 1;
                }
            }
        }
        i += 1;
    }
    None
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
        if let (Some(x), Some(y)) = (row.get(x_idx), row.get(y_idx))
            && let (Ok(xn), Ok(yn)) = (x.trim().parse::<f64>(), y.trim().parse::<f64>())
        {
            points.push((xn, yn));
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

/// Patch static registry_mirrors .rs files that are missing the AUTO-GENERATED header.
///
/// The 145 files produced by the one-time markdown_to_rust mass-conversion (commit 20d93a12)
/// predate the generated_doc_header() convention and were written without metadata.  This
/// function backfills the standard three-line header so that downstream tooling (grepping for
/// AUTO-GENERATED, governance checks, CPD exclusion annotations) can treat all .rs files in
/// registry_mirrors/ uniformly.
fn patch_static_mirror_headers(args: PatchStaticMirrorHeadersArgs) -> Result<(), String> {
    let mirrors_dir = &args.mirrors_dir;
    if !mirrors_dir.exists() {
        return Err(format!(
            "mirrors directory not found: {}",
            mirrors_dir.display()
        ));
    }

    let entries = fs::read_dir(mirrors_dir)
        .map_err(|e| format!("read_dir {}: {}", mirrors_dir.display(), e))?;

    let mut patched = 0usize;
    let mut skipped = 0usize;

    let mut paths: Vec<_> = entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("rs"))
        .filter(|p| p.file_name().and_then(|s| s.to_str()) != Some("mod.rs"))
        .collect();
    paths.sort();

    for path in &paths {
        let content =
            fs::read_to_string(path).map_err(|e| format!("read {}: {}", path.display(), e))?;

        // Check the first five lines for the sentinel.
        let already_tagged = content
            .lines()
            .take(5)
            .any(|l| l.contains("AUTO-GENERATED"));

        if already_tagged {
            skipped += 1;
            continue;
        }

        // Determine a short source label: the last two path components are enough.
        let src_label = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");
        let header = format!(
            "//! <!-- AUTO-GENERATED BY markdown_to_rust: DO NOT EDIT -->\n\
             //! <!-- Source: static mirror converted from docs/data markdown; see git log for {src_label} -->\n\
             //! <!-- Regenerate: cargo run --release -p gororoba_cli_data --bin markdown_to_rust -->\n\
             //!\n"
        );

        if args.dry_run {
            println!("DRY-RUN would patch: {}", path.display());
        } else {
            let new_content = format!("{}{}", header, content);
            fs::write(path, new_content).map_err(|e| format!("write {}: {}", path.display(), e))?;
        }
        patched += 1;
    }

    if args.dry_run {
        println!(
            "Dry-run complete: {} files would be patched, {} already tagged.",
            patched, skipped
        );
    } else {
        println!(
            "Patched {} files with AUTO-GENERATED header; {} already tagged, {} total scanned.",
            patched,
            skipped,
            patched + skipped
        );
    }
    Ok(())
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Commands::ArtifactMarkdown(args) => emit_artifact_markdown(args),
        Commands::TodoMirror(args) => emit_todo_mirror(args),
        Commands::TodoLegacy(args) => emit_todo_legacy(args),
        Commands::RoadmapMirror(args) => emit_roadmap_mirror(args),
        Commands::RoadmapLegacy(args) => emit_roadmap_legacy(args),
        Commands::NextActionsMirror(args) => emit_next_actions_mirror(args),
        Commands::NextActionsLegacy(args) => emit_next_actions_legacy(args),
        Commands::KnowledgeMigrationPlanMirror(args) => emit_knowledge_migration_plan_mirror(args),
        Commands::NavigatorMirror(args) => emit_navigator_mirror(args),
        Commands::NavigatorLegacy(args) => emit_navigator_legacy(args),
        Commands::EntrypointDocsMirror(args) => emit_entrypoint_docs_mirror(args),
        Commands::EntrypointDocsLegacy(args) => emit_entrypoint_docs_legacy(args),
        Commands::RequirementsMirror(args) => emit_requirements_mirror(args),
        Commands::RequirementsLegacy(args) => emit_requirements_legacy(args),
        Commands::DocsRootNarrativesMirror(args) => emit_docs_root_narratives_mirror(args),
        Commands::DocsRootNarrativesLegacy(args) => emit_docs_root_narratives_legacy(args),
        Commands::ResearchNarrativesMirror(args) => emit_research_narratives_mirror(args),
        Commands::ResearchNarrativesLegacy(args) => emit_research_narratives_legacy(args),
        Commands::InsightsMirror(args) => emit_insights_mirror(args),
        Commands::InsightsLegacy(args) => emit_insights_legacy(args),
        Commands::ClaimsMirror(args) => emit_claims_mirror(args),
        Commands::ClaimsMatrixLegacy(args) => emit_claims_matrix_legacy(args),
        Commands::BibliographyMirror(args) => emit_bibliography_mirror(args),
        Commands::BibliographyLegacy(args) => emit_bibliography_legacy(args),
        Commands::ExperimentsMirror(args) => emit_experiments_mirror(args),
        Commands::ExperimentsLegacy(args) => emit_experiments_legacy(args),
        Commands::TheoremsMirror(args) => emit_theorems_mirror(args),
        Commands::TheoremsLegacy(args) => emit_theorems_legacy(args),
        Commands::ControlPlaneDocs(args) => emit_control_plane_docs(args),
        Commands::MarkdownGovernanceMirror(args) => emit_markdown_governance_mirror(args),
        Commands::ClaimsTasksMirror(args) => emit_claims_tasks_mirror(args),
        Commands::ClaimsTasksLegacy(args) => emit_claims_tasks_legacy(args),
        Commands::ClaimsDomainsMirror(args) => emit_claims_domains_mirror(args),
        Commands::ClaimsDomainsLegacy(args) => emit_claims_domains_legacy(args),
        Commands::ClaimTicketsMirror(args) => emit_claim_tickets_mirror(args),
        Commands::ClaimTicketsLegacy(args) => emit_claim_tickets_legacy(args),
        Commands::ExternalSourcesMirror(args) => emit_external_sources_mirror(args),
        Commands::ExternalSourcesLegacy(args) => emit_external_sources_legacy(args),
        Commands::BookDocsMirror(args) => emit_book_docs_mirror(args),
        Commands::BookDocsLegacy(args) => emit_book_docs_legacy(args),
        Commands::DataArtifactNarrativesMirror(args) => emit_data_artifact_narratives_mirror(args),
        Commands::DataArtifactNarrativesLegacy(args) => emit_data_artifact_narratives_legacy(args),
        Commands::ReportsNarrativesMirror(args) => emit_reports_narratives_mirror(args),
        Commands::ReportsNarrativesLegacy(args) => emit_reports_narratives_legacy(args),
        Commands::DocsConvosMirror(args) => emit_docs_convos_mirror(args),
        Commands::DocsConvosLegacy(args) => emit_docs_convos_legacy(args),
        Commands::MonographLegacy(args) => emit_monograph_legacy(args),
        Commands::BibliographyBibtex(args) => emit_bibliography_bibtex(args),
        Commands::EquationsTex(args) => emit_equations_tex(args),
        Commands::DatasetPgfplots(args) => emit_dataset_pgfplots(args),
        Commands::Mermaid(args) => emit_mermaid(args),
        Commands::Svg(args) => emit_svg(args),
        Commands::PatchStaticMirrorHeaders(args) => patch_static_mirror_headers(args),
    }
}

fn main() {
    let cli = Cli::parse();
    if let Err(err) = run(cli) {
        eprintln!("ERROR: {}", err);
        std::process::exit(1);
    }
}
