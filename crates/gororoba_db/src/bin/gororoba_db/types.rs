//! Type definitions for the `gororoba-db` binary: clap Cli +
//! all command/subcommand variants + per-subcommand Args structs.
//! ~650 lines of declarative clap definitions split out so the bin
//! root focuses on dispatch + cmd_* implementations.
//!
//! Fields and enum variants are pub(crate) so the bin root can match.
//! Uses `#[path]` because this binary has explicit Cargo.toml path.

use clap::{Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

// ─── CLI definition ────────────────────────────────────────────────

#[derive(Parser, Debug)]
#[command(
    name = "gororoba-db",
    about = "Three-layer registry CLI: SQLite source -> compatibility exports -> query",
    long_about = "Unified entrypoint for building, querying, and auditing the registry.\n\n\
                  Layer 1 (Canonical Source): registry/canonical/control_plane.sqlite3.\n\
                  Layer 2 (Compatibility): registry/*.toml (legacy, generated/validated).\n\
                  Layer 3 (Query):  This CLI."
)]
pub(crate) struct Cli {
    /// Repository root.
    #[arg(long, default_value = ".")]
    pub(crate) repo_root: PathBuf,

    /// SQLite database path.
    #[arg(long, default_value = "registry/canonical/control_plane.sqlite3")]
    pub(crate) db: PathBuf,

    #[command(subcommand)]
    pub(crate) command: Commands,
}

#[derive(Subcommand, Debug)]
pub(crate) enum Commands {
    /// Rebuild registry/canonical/control_plane.sqlite3 from compatibility inputs and refresh compatibility artifacts.
    Build(BuildArgs),

    /// Show database statistics: table row counts, migration status, and source-of-truth manifest.
    Stats,

    /// Print full schema introspection (tables, columns, row counts).
    Schema,

    /// List, show, or search claims.
    Claims(ClaimsArgs),

    /// List or search insights.
    Insights(InsightsArgs),

    /// List experiments.
    Experiments(ExperimentsArgs),

    /// Full-text search across claims, insights, narratives, and bibliography.
    Search(SearchArgs),

    /// Cross-reference queries (dangling refs, unlinked claims, coverage).
    Xref(XrefArgs),

    /// Audit: verify signatures, crossrefs, and labels.
    AuditCmd(AuditArgs),

    /// Import knowledge-base TOML files (equation atoms, proofs, derivations) into SQLite.
    ImportKnowledge(ImportKnowledgeArgs),

    /// Import planning TOML files (roadmap, todo, next-actions) into SQLite.
    ImportPlanning(ImportPlanningArgs),

    /// Import requirements TOML into SQLite.
    ImportRequirements(ImportRequirementsArgs),

    /// Import research narrative TOML into SQLite.
    ImportNarratives(ImportNarrativesArgs),

    /// Export planning tables to TOML-compatible output (stdout or file).
    ExportPlanning(ExportPlanningArgs),

    /// Export requirements tables to TOML-compatible output (stdout or file).
    ExportRequirements(ExportRequirementsArgs),

    /// Mutate canonical planning rows and refresh generated compatibility exports.
    Planning(PlanningMutationArgs),

    /// Mutate canonical requirements rows and refresh generated compatibility exports.
    Requirements(RequirementsMutationArgs),

    /// Mutate a single claim row in the canonical SQLite. Status_note edits land
    /// here. Run `make registry-export-markdown` afterward to regenerate the
    /// compat-export TOMLs and downstream markdown.
    Claim(ClaimMutationArgs),

    /// Mutate a single insight row in the canonical SQLite. Mirrors `claim`
    /// modulo the target table. Requires migration 0016 applied.
    Insight(InsightMutationArgs),

    /// Mutate a single experiment row in the canonical SQLite. Mirrors `claim`
    /// modulo the target table (experiments_cp). Requires migration 0016 applied.
    Experiment(ExperimentMutationArgs),

    /// Register retained local evidence paths in canonical SQLite.
    Artifact(ArtifactArgs),

    /// Query rows from any table by name with optional status filter.
    Query(QueryArgs),

    /// Show legacy TOML files that should be archived.
    ArchiveLegacy,

    /// Show evcxr/Jupyter notebook integration status and capabilities.
    NotebookInfo,

    /// List or manage notebook sessions stored in the database.
    Notebooks(NotebookArgs),
}

#[derive(Parser, Debug)]
pub(crate) struct BuildArgs {
    /// Verify crossrefs and signatures after building (exit non-zero on errors).
    #[arg(long)]
    pub(crate) verify: bool,
}

#[derive(Parser, Debug)]
pub(crate) struct ClaimsArgs {
    #[command(subcommand)]
    pub(crate) action: ClaimsAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ClaimsAction {
    /// List claims with optional status filter.
    List {
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Show a single claim by ID.
    Show { id: String },
    /// Full-text search claims.
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
    /// List claims with no linked experiments or insights.
    Unlinked,
}

#[derive(Parser, Debug)]
pub(crate) struct InsightsArgs {
    #[command(subcommand)]
    pub(crate) action: InsightsAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum InsightsAction {
    /// List insights.
    List {
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
    /// Full-text search insights.
    Search {
        query: String,
        #[arg(long, default_value_t = 20)]
        limit: usize,
    },
}

#[derive(Parser, Debug)]
pub(crate) struct ExperimentsArgs {
    #[command(subcommand)]
    pub(crate) action: ExperimentsAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ExperimentsAction {
    /// List experiments with optional status filter.
    List {
        #[arg(long)]
        status: Option<String>,
        #[arg(long, default_value_t = 50)]
        limit: usize,
    },
}

#[derive(Parser, Debug)]
pub(crate) struct XrefArgs {
    #[command(subcommand)]
    pub(crate) action: XrefAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum XrefAction {
    /// Find dangling crossrefs (references to non-existent claims).
    Dangling,
    /// Find claims with no linked experiments or insights.
    Unlinked,
    /// Show crossref coverage summary.
    Coverage,
}

#[derive(Parser, Debug)]
pub(crate) struct AuditArgs {
    #[command(subcommand)]
    pub(crate) action: AuditAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum AuditAction {
    /// Verify schema signatures match TOML content hashes.
    Signatures,
    /// Check for dangling crossrefs.
    Crossrefs,
}

#[derive(Parser, Debug)]
pub(crate) struct ImportKnowledgeArgs {
    /// Path to equation atoms TOML (v3 preferred).
    #[arg(long, default_value = "registry/knowledge/equation_atoms_v3.toml")]
    pub(crate) equation_atoms: PathBuf,

    /// Path to derivation steps TOML.
    #[arg(long, default_value = "registry/knowledge/derivation_steps.toml")]
    pub(crate) derivation_steps: PathBuf,

    /// Path to proof skeletons TOML.
    #[arg(long, default_value = "registry/knowledge/proof_skeletons.toml")]
    pub(crate) proof_skeletons: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct ImportPlanningArgs {
    /// Path to roadmap TOML.
    #[arg(long, default_value = "registry/roadmap.toml")]
    pub(crate) roadmap: PathBuf,

    /// Path to todo TOML.
    #[arg(long, default_value = "registry/todo.toml")]
    pub(crate) todo: PathBuf,

    /// Path to next-actions TOML.
    #[arg(long, default_value = "registry/next_actions.toml")]
    pub(crate) next_actions: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct ImportRequirementsArgs {
    /// Path to requirements TOML.
    #[arg(long, default_value = "registry/requirements.toml")]
    pub(crate) requirements: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct ImportNarrativesArgs {
    /// Path to research narratives TOML.
    #[arg(long, default_value = "registry/research_narratives.toml")]
    pub(crate) narratives: PathBuf,
}

#[derive(Parser, Debug)]
pub(crate) struct ExportPlanningArgs {
    /// Output format.
    #[arg(long, default_value = "toml")]
    pub(crate) format: OutputFormat,

    /// Optional output file (defaults to stdout).
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,

    /// Which planning table to export.
    #[arg(long, default_value = "roadmap")]
    pub(crate) table: PlanningTable,
}

#[derive(Clone, Debug, ValueEnum)]
pub(crate) enum RequirementsOutputFormat {
    Toml,
    Json,
    Text,
}

#[derive(Parser, Debug)]
pub(crate) struct ExportRequirementsArgs {
    /// Output format.
    #[arg(long, default_value = "toml")]
    pub(crate) format: RequirementsOutputFormat,

    /// Optional output file (defaults to stdout).
    #[arg(long)]
    pub(crate) out: Option<PathBuf>,
}

#[derive(Parser, Debug)]
pub(crate) struct PlanningMutationArgs {
    #[command(subcommand)]
    pub(crate) action: PlanningMutationAction,
}

#[derive(Parser, Debug)]
pub(crate) struct ClaimMutationArgs {
    #[command(subcommand)]
    pub(crate) action: ClaimMutationAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ClaimMutationAction {
    /// Replace the status_note on one claim row inside a single
    /// BEGIN IMMEDIATE transaction. Appends to claim_revisions.
    UpdateStatusNote {
        /// Claim id (e.g., C-441).
        #[arg(long)]
        id: String,
        /// New status_note text. Plain ASCII; no smart quotes.
        #[arg(long)]
        status_note: String,
        /// Reviewer name. Defaults to $USER.
        #[arg(long)]
        actor: Option<String>,
        /// Free-form reason recorded in claim_revisions.reason.
        #[arg(long)]
        reason: Option<String>,
        /// Run provenance export-control-plane after the SQLite update.
        /// Pass `--regen-toml false` to skip; useful for batch updates where
        /// you want to call the exporter once at the end.
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        regen_toml: bool,
    },
    /// Print the current status_note for one claim.
    ShowStatusNote {
        #[arg(long)]
        id: String,
    },
    /// Replace the formal_proof on one claim row inside a single
    /// BEGIN IMMEDIATE transaction. Appends to claim_revisions with
    /// field_name='formal_proof'.
    UpdateFormalProof {
        /// Claim id (e.g., C-441).
        #[arg(long)]
        id: String,
        /// New formal_proof value. Per docs/engineering/formal_proof_field_schema_2026_05_09.md
        /// this should be one of: `na_empirical[:rationale]`,
        /// `na_observational[:source]`, `na_methodology[:tool]`, `pending[:reason]`,
        /// `proofs/verified/<file>.v[#theorem]`, `proofs/theories/<file>.v[#theorem]`,
        /// or `external:<citation>`.
        #[arg(long)]
        formal_proof: String,
        /// Reviewer name. Defaults to $USER.
        #[arg(long)]
        actor: Option<String>,
        /// Free-form reason recorded in claim_revisions.reason.
        #[arg(long)]
        reason: Option<String>,
        /// Run provenance export-control-plane after the SQLite update.
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        regen_toml: bool,
    },
    /// Print the current formal_proof for one claim.
    ShowFormalProof {
        #[arg(long)]
        id: String,
    },
}

#[derive(Parser, Debug)]
pub(crate) struct InsightMutationArgs {
    #[command(subcommand)]
    pub(crate) action: InsightMutationAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum InsightMutationAction {
    UpdateStatusNote {
        #[arg(long)]
        id: String,
        #[arg(long)]
        status_note: String,
        #[arg(long)]
        actor: Option<String>,
        #[arg(long)]
        reason: Option<String>,
        /// Run provenance export-control-plane after the SQLite update.
        /// Pass `--regen-toml false` to skip; useful for batch updates where
        /// you want to call the exporter once at the end.
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        regen_toml: bool,
    },
    ShowStatusNote {
        #[arg(long)]
        id: String,
    },
}

#[derive(Parser, Debug)]
pub(crate) struct ExperimentMutationArgs {
    #[command(subcommand)]
    pub(crate) action: ExperimentMutationAction,
}

#[derive(Parser, Debug)]
pub(crate) struct ArtifactArgs {
    #[command(subcommand)]
    pub(crate) action: ArtifactAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum ArtifactAction {
    /// Register a local evidence bundle and its repository-relative paths.
    RegisterLocal {
        /// Stable artifact identity.
        #[arg(long)]
        id: String,
        /// Unique canonical artifact key.
        #[arg(long)]
        key: String,
        /// Human-readable artifact title.
        #[arg(long)]
        title: String,
        /// Citation or provenance description.
        #[arg(long)]
        citation: String,
        /// Repository-relative retained path. Repeat for every bundle member.
        #[arg(long = "path", required = true)]
        paths: Vec<String>,
        /// Existing canonical lane used for the registration.
        #[arg(long, default_value = "web_references")]
        lane: String,
        /// Source reference. Repeat for every source reference.
        #[arg(long = "source-ref")]
        source_refs: Vec<String>,
        /// Actor recorded in the export-run details.
        #[arg(long)]
        actor: Option<String>,
        /// Reason recorded in the export-run details.
        #[arg(long)]
        reason: Option<String>,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum ExperimentMutationAction {
    UpdateStatusNote {
        #[arg(long)]
        id: String,
        #[arg(long)]
        status_note: String,
        #[arg(long)]
        actor: Option<String>,
        #[arg(long)]
        reason: Option<String>,
        /// Run provenance export-control-plane after the SQLite update.
        /// Pass `--regen-toml false` to skip; useful for batch updates where
        /// you want to call the exporter once at the end.
        #[arg(long, action = clap::ArgAction::Set, default_value_t = true)]
        regen_toml: bool,
    },
    ShowStatusNote {
        #[arg(long)]
        id: String,
    },
}

#[derive(Subcommand, Debug)]
pub(crate) enum PlanningMutationAction {
    /// Upsert one roadmap workstream in the canonical DB.
    UpsertRoadmapItem {
        #[arg(long)]
        id: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        priority: String,
        #[arg(long)]
        status: String,
        #[arg(long, default_value = "")]
        status_token: String,
        #[arg(long)]
        description: String,
        #[arg(long, default_value = "")]
        sprint: String,
        #[arg(long, value_delimiter = ',')]
        dependencies: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        acceptance_criteria: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        primary_outputs: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        evidence_refs: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        lacunae: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        claims: Vec<String>,
        #[arg(long, default_value = "")]
        insight: String,
    },
    /// Delete one roadmap workstream from the canonical DB.
    DeleteRoadmapItem {
        #[arg(long)]
        id: String,
    },
    /// Upsert one todo item in the canonical DB.
    UpsertTodoItem {
        #[arg(long)]
        id: String,
        #[arg(long)]
        area: String,
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        priority: String,
        #[arg(long)]
        status: String,
        #[arg(long, default_value = "")]
        status_token: String,
        #[arg(long, value_delimiter = ',')]
        dependencies: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        acceptance_criteria: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        evidence_refs: Vec<String>,
    },
    /// Delete one todo item from the canonical DB.
    DeleteTodoItem {
        #[arg(long)]
        id: String,
    },
    /// Upsert one next action in the canonical DB.
    UpsertNextAction {
        #[arg(long)]
        id: String,
        #[arg(long)]
        area: String,
        #[arg(long)]
        title: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        priority: String,
        #[arg(long)]
        status: String,
        #[arg(long, default_value = "")]
        status_token: String,
        #[arg(long, value_delimiter = ',')]
        dependencies: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        acceptance_criteria: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        evidence_refs: Vec<String>,
    },
    /// Delete one next action from the canonical DB.
    DeleteNextAction {
        #[arg(long)]
        id: String,
    },
}

#[derive(Parser, Debug)]
pub(crate) struct RequirementsMutationArgs {
    #[command(subcommand)]
    pub(crate) action: RequirementsMutationAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum RequirementsMutationAction {
    /// Update canonical requirements registry metadata.
    SetMeta {
        #[arg(long)]
        authoritative: Option<bool>,
        #[arg(long)]
        status: Option<String>,
        #[arg(long)]
        status_token: Option<String>,
        #[arg(long)]
        updated: Option<String>,
        #[arg(long)]
        python_recommended: Option<String>,
        #[arg(long)]
        python_allowed: Option<String>,
        #[arg(long)]
        primary_markdown: Option<String>,
        #[arg(long, value_delimiter = ',')]
        status_allowlist: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        runtime_stack_allowlist: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        required_module_fields: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        required_gap_fields: Vec<String>,
    },
    /// Upsert one requirements module in the canonical DB.
    UpsertModule {
        #[arg(long)]
        id: String,
        #[arg(long)]
        name: String,
        #[arg(long)]
        markdown: String,
        #[arg(long)]
        status: String,
        #[arg(long, default_value = "")]
        status_token: String,
        #[arg(long)]
        runtime_stack: String,
        #[arg(long, value_delimiter = ',')]
        requires_modules: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        install_targets: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        verify_targets: Vec<String>,
        #[arg(long, value_delimiter = ',')]
        acceptance_criteria: Vec<String>,
    },
    /// Delete one requirements module from the canonical DB.
    DeleteModule {
        #[arg(long)]
        id: String,
    },
    /// Upsert one requirements coverage gap in the canonical DB.
    UpsertGap {
        #[arg(long)]
        id: String,
        #[arg(long)]
        area: String,
        #[arg(long)]
        status: String,
        #[arg(long, default_value = "")]
        status_token: String,
        #[arg(long)]
        description: String,
        #[arg(long)]
        proposed_resolution: String,
        #[arg(long, value_delimiter = ',')]
        related_module_ids: Vec<String>,
    },
    /// Delete one requirements coverage gap from the canonical DB.
    DeleteGap {
        #[arg(long)]
        id: String,
    },
}

#[derive(Clone, Debug, ValueEnum)]
pub(crate) enum OutputFormat {
    Toml,
    Json,
    Text,
}

#[derive(Clone, Debug, ValueEnum)]
pub(crate) enum PlanningTable {
    Roadmap,
    Todo,
    NextActions,
}

#[derive(Parser, Debug)]
pub(crate) struct SearchArgs {
    /// Search query (FTS5 syntax supported).
    pub(crate) query: String,

    /// Maximum results.
    #[arg(long, default_value_t = 20)]
    pub(crate) limit: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct QueryArgs {
    /// Table name to query.
    pub(crate) table: String,

    /// Optional status filter.
    #[arg(long)]
    pub(crate) status: Option<String>,

    /// Maximum rows.
    #[arg(long, default_value_t = 50)]
    pub(crate) limit: usize,
}

#[derive(Parser, Debug)]
pub(crate) struct NotebookArgs {
    #[command(subcommand)]
    pub(crate) action: NotebookAction,
}

#[derive(Subcommand, Debug)]
pub(crate) enum NotebookAction {
    /// List existing notebook sessions.
    List,
    /// Create a new notebook session.
    Create {
        /// Session title.
        #[arg(long)]
        title: String,
        /// Optional description.
        #[arg(long, default_value = "")]
        description: String,
    },
}
