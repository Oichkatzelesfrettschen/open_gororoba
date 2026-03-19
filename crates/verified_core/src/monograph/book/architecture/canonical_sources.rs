//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Canonical Sources
//!
//! For everyday development and review, this repository should be read through
//! the Rust and Markdown surfaces that are easy to browse on the web:
//!
//! - crate rustdoc from `cargo doc --workspace --no-deps`
//! - the mdBook under `docs/book/`
//! - the canonical SQLite control plane at `registry/canonical/control_plane.sqlite3`
//! - generated compatibility registries under `registry/`
//! - focused evidence and source docs under `docs/`
//!
//! ## Canonical order
//!
//! When the same concept appears in multiple places, use this priority order:
//!
//! 1. Rust source/tests in `crates/` and Rocq proof files in `proofs/`
//! 2. The canonical SQLite control plane in `registry/canonical/`
//! 3. Crate rustdoc on `lib.rs` and module docs
//! 4. mdBook pages in `docs/book/src/`
//! 5. Generated compatibility registries in `registry/`
//! 6. Focused Markdown narratives in `docs/`
//! 7. LaTeX under `docs/latex/`
//!
//! ## What the LaTeX tree is for
//!
//! `docs/latex/` is publication output and paper assembly material. It is useful
//! for thesis builds, appendices, and archival paper packaging, but it should not
//! be treated as the primary place to learn current repo truth.
//!
//! In particular, the generated files below are mirrors, not canonical sources:
//!
//! - `docs/latex/claims_appendix.tex`
//! - `docs/latex/experiments_appendix.tex`
//! - `docs/latex/insights_appendix.tex`
//! - `docs/latex/arxiv/*.tex`
//!
//! If a LaTeX appendix disagrees with a crate, test, registry row, or mdBook
//! page, the LaTeX copy should be updated or removed rather than treated as the
//! source of truth.
//!
//! ## Recommended doc workflow
//!
//! Use these commands first:
//!
//! ```sh
//! # Inspect the canonical control plane directly
//! cargo run --release -p gororoba_cli_provenance --bin provenance -- \
//!   --db registry/canonical/control_plane.sqlite3 \
//!   query claim C-001
//!
//! # Browse crate-level API and module docs
//! cargo doc --workspace --no-deps
//!
//! # Build the web book
//! mdbook build docs/book
//!
//! # Refresh the standard control-plane docs bundle, including docs/THEOREMS.md
//! cargo run --release -p gororoba_cli_data --bin registry-emit -- \
//!   control-plane-docs
//!
//! # Refresh the mdBook registry after editing docs/book/src
//! cargo run --release --bin markdown-registry -- normalize-book-docs --bootstrap-from-markdown
//! ```
//!
//! ## Scope for this repo
//!
//! This repository is Rust-first. That means:
//!
//! - new implementation detail belongs in crates and rustdoc
//! - explanatory architecture belongs in the mdBook
//! - claims, insights, experiments, binaries, theorem inventory, and source
//!   indexing belong in the canonical SQLite control plane
//! - generated TOML and Markdown views exist for compatibility and web browsing
//! - LaTeX should be reserved for publication packaging, not duplicated operational
//!   documentation
//!
//! ## Current boundary: `book_docs`
//!
//! `registry/book_docs.toml` is intentionally not part of the SQLite control-plane
//! migration. It indexes authored mdBook source pages under `docs/book/src/` and
//! drives generated browsing aids such as `docs/generated/BOOK_DOCS_REGISTRY_MIRROR.md`.
//!
//! For now, treat this lane as:
//!
//! - authored mdBook pages in `docs/book/src/` are the source material
//! - `registry/book_docs.toml` is the mdBook catalog/normalization registry
//! - the SQLite control plane remains canonical for claims, insights, experiments,
//!   binaries, and theorems, but not for authored mdBook prose inventory
//!
//! This boundary is deliberate: the control plane owns structured research and
//! verification state, while the mdBook still owns curated explanatory pages.
//!
