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
//! # Project Layout
//!
//! open\_gororoba is organized as a large Rust workspace rooted in `crates/`, with
//! canonical control-plane state under `registry/`, long-form docs under `docs/`,
//! generated artifacts under `data/`, and a research corpus under `papers/`.
//!
//! ![Repo operator matrix](../assets/repo_operator_matrix_3160x2820.png)
//!
//! The generated operator matrix above is the fastest way to answer "which surface
//! should I touch first?" before you drill into the directory-level view below.
//!
//! ## Representative directory structure
//!
//! ```text
//! crates/
//!   gororoba_algebra/   Algebra facade: Cayley-Dickson, Clifford, wheels, p-adic, groups, box-kites
//!   cd_kernel/          Pure Cayley-Dickson arithmetic kernel
//!   cosmology_core/     FLRW, bouncing models, dark energy
//!   gr_core/            Schwarzschild, Kerr, Novikov-Thorne, spectral bands
//!   materials_core/     Periodic table, metamaterial models
//!   optics_core/        GRIN lenses, ray tracing
//!   quantum_core/       Casimir, tensor networks, Grover
//!   stats_core/         Ultrametric, dip test, bootstrap, GPU (cudarc)
//!   spectral_core/      Fractional PDE, negative dimensions
//!   lbm_core/           Lattice Boltzmann
//!   control_core/       Control theory
//!   data_core/          Data loading, benchmarks, HDF5 export
//!   docpipe/            PDF extraction (pdfium-render)
//!   gororoba_py/        PyO3 bindings (thin wrappers)
//!   gororoba_cli*/      Domain-specific operator surfaces
//!
//! registry/
//!   canonical/          Canonical SQLite control plane and migration ledger
//!   claims.toml         SQLite-exported claims compatibility view
//!   insights.toml       SQLite-exported insights compatibility view
//!   experiments.toml    SQLite-exported experiments compatibility view
//!   binaries.toml       SQLite-exported CLI inventory compatibility view
//!   project.toml        Version, synchronized counts, sprint history
//!
//! papers/
//!   pdf/                Research PDFs
//!   bib/                Bibliography database
//!   extracted/          Structured TOML extractions per paper
//!   MANIFEST.toml       Paper metadata registry
//!
//! docs/
//!   book/               Canonical web-viewable architecture docs
//!   generated/          Generated mirrors and registry exports for browsing
//!   latex/              Publication output, appendices, paper assembly
//!   CLAIMS_EVIDENCE_MATRIX.md   Generated claims mirror from the canonical SQLite control plane
//!   INSIGHTS.md                 Generated insights mirror from the canonical SQLite control plane
//!   MATH_CONVENTIONS.md         9 mathematical conventions
//! ```text
//!
//! ## Build commands
//!
//! ```textsh
//! # Crate and module docs
//! cargo doc --workspace --no-deps
//!
//! # mdBook
//! mdbook build docs/book
//!
//! # Full test suite
//! cargo test --workspace -j$(nproc)
//!
//! # Clippy with warnings-as-errors
//! cargo clippy --workspace -j$(nproc) -- -D warnings
//!
//! # Registry validation
//! cargo run --release --bin registry-check
//!
//! # GPU tests (requires CUDA + RTX 4070 Ti or similar)
//! cargo test --workspace -j$(nproc) --features gpu
//! ```text
//!
//! `make latex` remains available for publication builds, but it is not the
//! recommended first stop for development-facing documentation.
//!
//! The `docs/*.md` mirrors are still useful for quick browsing, but canonical
//! authoring for claims, insights, experiments, binaries, and theorems now lives
//! in the Rust workspace plus `registry/canonical/control_plane.sqlite3`.
//!
//! ## Dependency management
//!
//! All external crates are declared at the workspace level in the root `Cargo.toml`
//! under `[workspace.dependencies]` and referenced by sub-crates with
//! `workspace = true`.  This prevents version conflicts and keeps the dependency
//! tree consistent.
//!
