//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Claims Evidence Matrix
//!
//! Claims are authored canonically in the SQLite control plane at
//! `registry/canonical/control_plane.sqlite3`. The compatibility file
//! `registry/claims.toml` and the web-viewable `docs/CLAIMS_EVIDENCE_MATRIX.md`
//! are downstream exports.
//!
//! Each claim records a testable statement about the codebase's mathematical or
//! physical results.
//!
//! ## Claim structure
//!
//! Each claim entry contains:
//!
//! - **id**: C-nnn identifier
//! - **statement**: The testable assertion
//! - **status**: Canonical claim status token (Verified, Refuted, Open, Partial,
//!   Pending, Superseded, Established, Inconclusive, Closed/*, etc.)
//! - **where\_stated**: Source reference (paper, conversation, analysis)
//! - **test**: Rust test function that verifies the claim (when applicable)
//!
//! ## Status distribution
//!
//! The bulk of claims fall into these categories:
//!
//! - **Verified**: Deterministic test passes against known results
//! - **Refuted**: Test demonstrates the claim is false (e.g., C-455 E8 connection)
//! - **Open**: Not yet tested, or test is pending implementation
//! - **Partial**: Some aspects verified, others remain open
//!
//! ## Validation
//!
//! The `provenance` and `registry-check` binaries validate claims for:
//!
//! - Sequential ID gaps
//! - Valid status enum values
//! - Consistency with project.toml counts
//! - SQLite control-plane integrity and proof linkage
//!
//! ```sh
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   import-legacy-control-plane
//!
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   export-control-plane
//!
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   verify-control-plane
//!
//! cargo run --release -p gororoba_cli_data --bin registry-check -- \
//!   --canonical-db registry/canonical/control_plane.sqlite3
//! ```
//!
//! For direct inspection, prefer querying the DB instead of opening the exported
//! TOML:
//!
//! ```sh
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   query claim C-001
//! ```
//!
//! Use `project-counter-sync` if you need top-level counts refreshed from the
//! current canonical DB state.
//!
//! ## Auto-generated appendix
//!
//! The `generate-latex` binary produces `docs/latex/claims_appendix.tex`, a
//! publication appendix from `registry/claims.toml`. That appendix is an export,
//! not a source of truth.
//!
//! ## Markdown mirror
//!
//! `docs/CLAIMS_EVIDENCE_MATRIX.md` is a generated convenience mirror for web
//! browsing and review. `migrate-claims` is retained only for explicit
//! legacy/bootstrap import flows, not steady-state authoring.
//!
