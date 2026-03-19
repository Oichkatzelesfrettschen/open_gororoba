//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/docs_root_narratives.toml -->
//!
//! # Artifact Source of Truth
//!
//! This repository now has a single canonical lane for citation/download/mirror reconciliation:
//!
//! - `registry/artifact_source_of_truth.toml`
//! - `reports/artifact_source_of_truth_reconciliation_2026_02_15.toml`
//!
//! And a deterministic modular source infrastructure projected from that master:
//!
//! - `registry/source_infrastructure.toml`
//! - `registry/source_lanes/papers_pdf.toml`
//! - `registry/source_lanes/datasets.toml`
//! - `registry/source_lanes/slides_artifacts.toml`
//! - `registry/source_lanes/web_references.toml`
//! - `reports/source_infrastructure_reconciliation_2026_02_15.toml`
//!
//! ## What this solves
//!
//! - Keeps cited artifacts and mirror status in one place.
//! - Preserves working mirrors and non-working mirrors together.
//! - Marks manual-intervention links without deleting provenance.
//! - Tracks downloaded paths vs downloadable-only links.
//! - Avoids drift between bibliography, intake retries, and reconciliation reports.
//!
//! ## Rebuild commands
//!
//! ```texttext
//! cargo run -p gororoba_cli_data --bin provenance -- export
//! cargo run -p gororoba_cli_data --bin provenance -- index
//! cargo run -p gororoba_cli_data --bin provenance -- verify
//! ```texttext
//!
//! ## Optional live link audit refresh
//!
//! ```texttext
//! cargo run -p gororoba_cli_data --bin provenance -- export
//! cargo run -p gororoba_cli_data --bin provenance -- index
//! cargo run -p gororoba_cli_data --bin provenance -- link-audit
//! cargo run -p gororoba_cli_data --bin provenance -- recover
//! cargo run -p gororoba_cli_data --bin provenance -- verify
//! ```texttext
//!
//! ## Generated audit views
//!
//! - `reports/artifact_blocked_links_2026_02_15.tsv`
//! - `reports/artifact_missing_minimum_2026_02_15.tsv`
//! - `reports/blocked_artifact_recovery_attempts_2026_02_15.tsv`
//! - `reports/blocked_artifact_retry_plan_2026_02_15.toml`
//!
//! ## Inputs used by the builder
//!
//! Primary operator surface:
//!
//! - `cargo run -p gororoba_cli_data --bin provenance -- index`
//! - `cargo run -p gororoba_cli_data --bin provenance -- export`
//! - `cargo run -p gororoba_cli_data --bin provenance -- verify`
//! - `cargo run -p gororoba_cli_data --bin provenance -- query artifact <needle>`
//! - `cargo run -p gororoba_cli_data --bin provenance -- query document <needle>`
//! - `cargo run -p gororoba_cli_data --bin provenance -- doctor`
//!
//! - Source files are discovered repo-wide under:
//!   - `registry/`
//!   - `reports/`
//!   - `docs/`
//!   - `papers/`
//!   - `data/papers/`
//!   - root `refs.bib`
//! - File types scanned:
//!   - `.toml`, `.bib`, `.bibtex`
//!   - `.md`, `.txt`, `.rst` (filename/path must include one of: `source`, `bibli`, `reconcil`, `artifact`, `intake`, `cayley`, `sedenion`, `octonion`, `quaternion`, `mirror`, `provenance`)
//! - Only citation-like links are promoted into artifact candidates (DOI/arXiv/scispace/PDF/reference-host matches).
//! - `data/external/intake/**/fetch_results*_normalized.tsv`
//! - `data/external/intake/**/mirror_retry_results*.tsv`
//! - `data/external/intake/**/link_audit_results*.tsv`
//! - `data/external/intake/**/pdf_success_added.tsv`
//!
//! ## Exclusions
//!
//! - Raw-capture registries under `registry/knowledge/` are non-authoritative and
//!   must not be used to seed artifact ownership or source-lane identity.
//! - Generated projections such as `registry/source_lanes/*.toml`,
//!   `registry/embedded_markdown_*.toml`, and `registry/markdown_payload*.toml`
//!   are downstream views, not candidate-source inputs for the artifact master.
//! - Repo-local contract/code paths are not promoted as artifacts unless they live
//!   under artifact-bearing prefixes such as `data/`, `papers/`, `archive/`, or
//!   `registry/knowledge/artifacts/`.
//!
