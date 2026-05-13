//! Clap-derived argument types + SVG element deserialize types for
//! the `registry-emit` binary. Relocated from inline at the top of
//! registry_emit.rs to keep the bin root focused on dispatch +
//! emitter logic. Fields are `pub(crate)` so the bin root can match
//! variants and read field values across the module boundary.

use clap::{Parser, Subcommand};
use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "registry-emit")]
#[command(about = "Registry/control-plane multi-format emitter frontend")]
pub(crate) struct Cli {
    #[command(subcommand)]
    pub(crate) command: Commands,
}

// Boxing the larger Subcommand variants would force callers through an
// extra deref and lose clap derive ergonomics; the enum is dropped after
// dispatch so the size cost is one-shot.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Subcommand)]
pub(crate) enum Commands {
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
pub(crate) struct ArtifactMarkdownArgs {
    /// Repository root for resolving relative paths.
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    /// Artifact scroll index TOML path.
    #[arg(long, default_value = "registry/artifact_scrolls.toml")]
    pub(crate) index: PathBuf,
    /// Output directory where markdown files are emitted.
    #[arg(long)]
    pub(crate) out_dir: PathBuf,
    /// Optional scroll id filter (e.g., ART-001).
    #[arg(long)]
    pub(crate) id: Option<String>,
    /// Include generated/source-of-truth header.
    #[arg(long, default_value_t = true)]
    pub(crate) with_header: bool,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct BibliographyBibtexArgs {
    /// Bibliography TOML registry path.
    #[arg(long, default_value = "registry/bibliography.toml")]
    pub(crate) input: PathBuf,
    /// Output .bib file path.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct TodoMirrorArgs {
    /// TODO registry TOML path.
    #[arg(long, default_value = "registry/todo.toml")]
    pub(crate) input: PathBuf,
    /// Output markdown mirror path.
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/todo_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct TodoLegacyArgs {
    #[arg(long, default_value = "registry/todo.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/todo_narrative.toml")]
    pub(crate) narrative: PathBuf,
    #[arg(long, default_value = "crates/data_core/src/registry_mirrors/todo.rs")]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct RoadmapMirrorArgs {
    #[arg(long, default_value = "registry/roadmap.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/roadmap_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct RoadmapLegacyArgs {
    #[arg(long, default_value = "registry/roadmap.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/roadmap_narrative.toml")]
    pub(crate) narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/roadmap.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct NextActionsMirrorArgs {
    #[arg(long, default_value = "registry/next_actions.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/next_actions_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct NextActionsLegacyArgs {
    #[arg(long, default_value = "registry/next_actions.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/next_actions_narrative.toml")]
    pub(crate) narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/next_actions.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct KnowledgeMigrationPlanMirrorArgs {
    /// Knowledge migration plan registry TOML path.
    #[arg(long, default_value = "registry/knowledge_migration_plan.toml")]
    pub(crate) input: PathBuf,
    /// Output markdown mirror path.
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/knowledge_migration_plan_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct NavigatorMirrorArgs {
    #[arg(long, default_value = "registry/navigator.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/navigator_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct NavigatorLegacyArgs {
    #[arg(long, default_value = "registry/navigator.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/navigator.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct EntrypointDocsMirrorArgs {
    #[arg(long, default_value = "registry/entrypoint_docs.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/entrypoint_docs_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct EntrypointDocsLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/entrypoint_docs.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct RequirementsMirrorArgs {
    #[arg(long, default_value = "registry/requirements.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/requirements_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct RequirementsLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/requirements.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/requirements_narrative.toml")]
    pub(crate) narrative: PathBuf,
    /// Optional audit tools TOML; when provided, appends an "Audit Tools" section
    /// to the primary requirements markdown.
    #[arg(long)]
    pub(crate) audit_tools: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DocsRootNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/docs_root_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/docs_root_narratives_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DocsRootNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/docs_root_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ResearchNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/research_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/research_narratives_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ResearchNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/research_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct InsightsMirrorArgs {
    #[arg(long, default_value = "registry/insights.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct InsightsLegacyArgs {
    #[arg(long, default_value = "registry/insights.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(long, default_value = "registry/insights_narrative.toml")]
    pub(crate) narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimsMirrorArgs {
    #[arg(long, default_value = "registry/claims.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimsMatrixLegacyArgs {
    #[arg(long, default_value = "registry/claims.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_evidence_matrix.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct BibliographyMirrorArgs {
    #[arg(long, default_value = "registry/bibliography.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/bibliography_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct BibliographyLegacyArgs {
    #[arg(long, default_value = "registry/bibliography.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/bibliography.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ExperimentsMirrorArgs {
    #[arg(long, default_value = "registry/experiments.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ExperimentsLegacyArgs {
    #[arg(long, default_value = "registry/experiments.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(long, default_value = "registry/experiments_narrative.toml")]
    pub(crate) narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_portfolio_shortlist.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct TheoremsMirrorArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct TheoremsLegacyArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ControlPlaneDocsArgs {
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) canonical_db: PathBuf,
    #[arg(long, default_value = "registry/claims.toml")]
    pub(crate) claims_input: PathBuf,
    #[arg(long, default_value = "registry/insights.toml")]
    pub(crate) insights_input: PathBuf,
    #[arg(long, default_value = "registry/experiments.toml")]
    pub(crate) experiments_input: PathBuf,
    #[arg(long, default_value = "registry/insights_narrative.toml")]
    pub(crate) insights_narrative: PathBuf,
    #[arg(long, default_value = "registry/experiments_narrative.toml")]
    pub(crate) experiments_narrative: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_registry_mirror.rs"
    )]
    pub(crate) claims_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_evidence_matrix.rs"
    )]
    pub(crate) claims_legacy_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights_registry_mirror.rs"
    )]
    pub(crate) insights_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/insights.rs"
    )]
    pub(crate) insights_legacy_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_registry_mirror.rs"
    )]
    pub(crate) experiments_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/experiments_portfolio_shortlist.rs"
    )]
    pub(crate) experiments_legacy_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems_registry_mirror.rs"
    )]
    pub(crate) theorems_mirror_output: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/theorems.rs"
    )]
    pub(crate) theorems_legacy_output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct MarkdownGovernanceMirrorArgs {
    #[arg(long, default_value = "registry/markdown_governance.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/markdown_governance_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimsTasksMirrorArgs {
    #[arg(long, default_value = "registry/claims_tasks.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_tasks_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimsTasksLegacyArgs {
    #[arg(long, default_value = "registry/claims_tasks.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_tasks.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimsDomainsMirrorArgs {
    #[arg(long, default_value = "registry/claims_domains.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claims_domains_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimsDomainsLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/claims_domains.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimTicketsMirrorArgs {
    #[arg(long, default_value = "registry/claim_tickets.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/claim_tickets_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ClaimTicketsLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/claim_tickets.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ExternalSourcesMirrorArgs {
    #[arg(long, default_value = "registry/external_sources.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/external_sources_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ExternalSourcesLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/external_sources.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct BookDocsMirrorArgs {
    #[arg(long, default_value = "registry/book_docs.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/book_docs_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct BookDocsLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/book_docs.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DataArtifactNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/data_artifact_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/data_artifact_narratives_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DataArtifactNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/data_artifact_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value = "registry/artifact_scrolls.toml")]
    pub(crate) artifact_index: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ReportsNarrativesMirrorArgs {
    #[arg(long, default_value = "registry/reports_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/reports_narratives_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct ReportsNarrativesLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/reports_narratives.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DocsConvosMirrorArgs {
    #[arg(long, default_value = "registry/docs_convos.toml")]
    pub(crate) input: PathBuf,
    #[arg(
        long,
        default_value = "crates/data_core/src/registry_mirrors/docs_convos_registry_mirror.rs"
    )]
    pub(crate) output: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DocsConvosLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/docs_convos.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct MonographLegacyArgs {
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,
    #[arg(long, default_value = "registry/monograph.toml")]
    pub(crate) input: PathBuf,
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct EquationsTexArgs {
    /// Equation atoms TOML registry path.
    #[arg(long, default_value = "registry/knowledge/equation_atoms.toml")]
    pub(crate) input: PathBuf,
    /// Output .tex path.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Optional domain filter, e.g. algebra or cosmology.
    #[arg(long)]
    pub(crate) domain: Option<String>,
    /// Maximum equation count to emit.
    #[arg(long, default_value_t = 500usize)]
    pub(crate) max_equations: usize,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct DatasetPgfplotsArgs {
    /// Dataset scroll TOML path.
    #[arg(long)]
    pub(crate) input: PathBuf,
    /// Output .tex path.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// X column name (defaults to first numeric-rich column).
    #[arg(long)]
    pub(crate) x_col: Option<String>,
    /// Y column name (defaults to second numeric-rich column).
    #[arg(long)]
    pub(crate) y_col: Option<String>,
    /// Maximum points to emit.
    #[arg(long, default_value_t = 2000usize)]
    pub(crate) max_points: usize,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct MermaidArgs {
    /// Mermaid graph TOML path.
    #[arg(long)]
    pub(crate) input: PathBuf,
    /// Output .mmd path.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct SvgArgs {
    /// SVG vector TOML path.
    #[arg(long)]
    pub(crate) input: PathBuf,
    /// Output .svg path.
    #[arg(long)]
    pub(crate) output: PathBuf,
    /// Allow unicode output (default false to satisfy repository ASCII policy).
    #[arg(long, default_value_t = false)]
    pub(crate) allow_unicode: bool,
}

#[derive(Debug, Parser)]
pub(crate) struct PatchStaticMirrorHeadersArgs {
    /// Registry mirrors directory to scan.
    #[arg(long, default_value = "crates/data_core/src/registry_mirrors")]
    pub(crate) mirrors_dir: PathBuf,
    /// Dry-run: print which files would be patched without writing.
    #[arg(long, default_value_t = false)]
    pub(crate) dry_run: bool,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ArtifactScrollIndex {
    pub(crate) scroll: Vec<ArtifactScrollIndexRow>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ArtifactScrollIndexRow {
    pub(crate) id: String,
    pub(crate) source_markdown: String,
    pub(crate) scroll_path: String,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ArtifactScrollDoc {
    pub(crate) section: Option<Vec<ArtifactSection>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct ArtifactSection {
    pub(crate) title: Option<String>,
    pub(crate) level: Option<i64>,
    pub(crate) body_text: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BibliographyRegistry {
    pub(crate) entry: Vec<BibEntry>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct BibEntry {
    pub(crate) id: String,
    pub(crate) citation_markdown: String,
    pub(crate) section: Option<String>,
    pub(crate) urls: Option<Vec<String>>,
    pub(crate) dois: Option<Vec<String>>,
    pub(crate) notes: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EquationRegistry {
    pub(crate) atom: Vec<EquationAtom>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct EquationAtom {
    pub(crate) id: String,
    pub(crate) expression: String,
    pub(crate) source_path: Option<String>,
    pub(crate) source_line: Option<usize>,
    pub(crate) domain_hint: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DatasetToml {
    pub(crate) dataset: DatasetRecord,
}

#[derive(Debug, Deserialize)]
pub(crate) struct DatasetRecord {
    pub(crate) id: String,
    pub(crate) source_csv: String,
    pub(crate) header: Vec<String>,
    pub(crate) rows: Option<Vec<Vec<String>>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MermaidRegistry {
    pub(crate) diagram: MermaidMeta,
    pub(crate) node: Option<Vec<MermaidNode>>,
    pub(crate) edge: Option<Vec<MermaidEdge>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MermaidMeta {
    pub(crate) kind: Option<String>,
    pub(crate) direction: Option<String>,
    pub(crate) title: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MermaidNode {
    pub(crate) id: String,
    pub(crate) label: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct MermaidEdge {
    pub(crate) from: String,
    pub(crate) to: String,
    pub(crate) label: Option<String>,
    pub(crate) style: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgRegistry {
    pub(crate) svg: SvgMeta,
    pub(crate) rect: Option<Vec<SvgRect>>,
    pub(crate) line: Option<Vec<SvgLine>>,
    pub(crate) circle: Option<Vec<SvgCircle>>,
    pub(crate) path: Option<Vec<SvgPath>>,
    pub(crate) text: Option<Vec<SvgText>>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgMeta {
    pub(crate) width: u32,
    pub(crate) height: u32,
    pub(crate) view_box: Option<String>,
    pub(crate) background: Option<String>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgRect {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) width: f64,
    pub(crate) height: f64,
    pub(crate) fill: Option<String>,
    pub(crate) stroke: Option<String>,
    pub(crate) stroke_width: Option<f64>,
    pub(crate) rx: Option<f64>,
    pub(crate) ry: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgLine {
    pub(crate) x1: f64,
    pub(crate) y1: f64,
    pub(crate) x2: f64,
    pub(crate) y2: f64,
    pub(crate) stroke: Option<String>,
    pub(crate) stroke_width: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgCircle {
    pub(crate) cx: f64,
    pub(crate) cy: f64,
    pub(crate) r: f64,
    pub(crate) fill: Option<String>,
    pub(crate) stroke: Option<String>,
    pub(crate) stroke_width: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgPath {
    pub(crate) d: String,
    pub(crate) fill: Option<String>,
    pub(crate) stroke: Option<String>,
    pub(crate) stroke_width: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub(crate) struct SvgText {
    pub(crate) x: f64,
    pub(crate) y: f64,
    pub(crate) value: String,
    pub(crate) fill: Option<String>,
    pub(crate) font_size: Option<f64>,
    pub(crate) font_family: Option<String>,
}
