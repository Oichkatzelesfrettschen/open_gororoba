//! Canonical status taxonomies, control-plane paths, and the rusqlite
//! migration script registry for the ProvenanceStore database.
//!
//! All embedded SQL migration files live under `db/migrations/` at the
//! repo root. The `migrations()` function returns the ordered list of
//! schema migrations applied by `ProvenanceStore::open`. New migrations
//! must be appended (never reordered) to preserve the recorded
//! migration version in existing SQLite files.

use rusqlite_migration::{M, Migrations};

pub(crate) const CANONICAL_CLAIM_STATUSES: &[&str] = &[
    "Verified",
    "Established",
    "Refuted",
    "Partial",
    "Provisional",
    "Theoretical",
    "Inconclusive",
    "Superseded",
    "Closed/Negative-Result",
    "Closed/Obstructed",
    "Closed/Research-Program",
    "Closed/Toy",
    "Closed/Analogy",
    "Closed/Source-Insufficient",
    "Closed/Methodology-Insufficient",
    "Closed/Refuted",
];

pub(crate) const CANONICAL_INSIGHT_STATUSES: &[&str] = &[
    "verified",
    "open",
    "superseded",
    "cross-validation-complete",
    "partial",
];

pub(crate) const JUSTIFIED_UNLINKED_THEOREM_IDS: &[&str] = &[
    "C1007_CDPropertyLoss",
    "C1635_SedenionDriverSemantics",
    "C1636_Cariow2013SedenionSchedule",
    "C1637_R300SedenionZeroDivisor",
    "C1638_OctonionDowncastNoZeroDivisors",
    "C958_ZDGraphTopology",
    "C958b_ZDAdjacencyAnalytical",
    "C959_CHSHClassicalBound",
    "C993_CarlsonBranchFree",
    "C999_PathionEntropyBound",
    "C_ConjugateInvolution",
    "C_NormConjugate",
    "C_OctConjInvolution",
    "C_OverImbalancedSign",
    "C_QIBoundNegative",
    "C_QITauScaling",
    "C_SedConjInvolution",
    "C_TraceTracefreeVanishes",
    "C_WECImpliesNEC",
    "C_WarpEnergyNonpositive",
];

pub(crate) const CONTROL_PLANE_DB_PATH: &str = "registry/canonical/control_plane.sqlite3";
pub(crate) const CONTROL_PLANE_EXPORT_COMMAND: &str =
    "cargo run -p gororoba_cli_data --bin provenance -- export-control-plane";
pub(crate) const EXTERNAL_SOURCES_EXPORT_COMMAND: &str =
    "cargo run -p gororoba_cli_data --bin provenance -- export-external-sources";

pub(crate) fn migrations() -> Migrations<'static> {
    Migrations::new(vec![
        M::up(include_str!(
            "../../../db/migrations/0001_provenance_index.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0002_control_plane.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0003_binaries_crate_source.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0004_control_plane_compat_text.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0005_download_jobs.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0006_download_attempt_outcomes.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0007_download_campaigns.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0008_download_attempt_failure_class.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0009_external_sources.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0010_knowledge_and_planning.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0011_fts5_crossrefs.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0012_literature_verification.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0013_planning_compat_exports.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0014_requirements_compat_exports.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0015_revisions_audit.sql"
        )),
        M::up(include_str!(
            "../../../db/migrations/0016_status_note_columns.sql"
        )),
    ])
}
