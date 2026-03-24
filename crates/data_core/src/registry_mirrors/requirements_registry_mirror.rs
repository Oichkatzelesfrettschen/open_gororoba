//! # Requirements Registry Mirror
//!
//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: see authoritative source line below -->
//!
//! Authoritative source: `registry/requirements.toml`.
//!
//! - Updated: 2026-02-10
//! - Python recommended: `3.11-3.12`
//! - Python allowed: `3.13+ (with optional extras caveats)`
//! - Primary markdown: `docs/REQUIREMENTS.md`
//!
//! ## Modules
//!
//! ### REQ-ALGEBRA: algebra
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/algebra.md`
//!
//! ### REQ-ANALYSIS: analysis
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/analysis.md`
//! - Install targets:
//!   - `make install-analysis`
//!
//! ### REQ-ASTRO: astro
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/astro.md`
//! - Install targets:
//!   - `make install-astro`
//!
//! ### REQ-COQ: rocq
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/rocq.md`
//! - Install targets:
//!   - `make rocq`
//!
//! ### REQ-CORE: core
//!
//! - Status: `active`
//! - Markdown: `docs/REQUIREMENTS.md`
//! - Install targets:
//!   - `make install`
//!
//! ### REQ-CPP: cpp
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/cpp.md`
//! - Install targets:
//!   - `make cpp-build`
//!   - `make cpp-test`
//!
//! ### REQ-DATA-GOVERNANCE: data_governance
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/analysis.md`
//! - Install targets:
//!   - `cargo run -p gororoba_cli_data --bin fetch-datasets -- --all --skip-existing --output-dir data/external`
//!   - `cargo run -p gororoba_cli_data --bin record-external-hashes -- --root data/external --output data/external/PROVENANCE.local.json`
//!   - `cargo run -p gororoba_cli_data --bin external-redownload-audit -- --out reports/external_redownload_audit_YYYY-MM-DD.toml --backend-order wget,curl,fetch`
//!   - `cargo run -p gororoba_cli_data --bin data-origin-audit -- --fail-on-strict-unknown`
//!   - `cargo run -p gororoba_cli_data --bin data-governance-gate --`
//!   - `cargo run -p gororoba_cli_data --bin data-semantic-validate --`
//!
//! ### REQ-HELIOSPHERE: heliosphere
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/heliosphere.md`
//! - Install targets:
//!   - `cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Helios 1 Merged Hourly" --skip-existing`
//!   - `cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Helios 2 Merged Hourly" --skip-existing`
//!   - `cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Voyager 1 Merged Hourly (2016)" --skip-existing`
//!   - `cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "Voyager 1 CRS Daily Flux (2016)" --skip-existing`
//!   - `cargo run -p gororoba_cli_data --bin fetch-datasets -- --dataset "NASA OMNI2 Solar Wind + IMF (2016)" --skip-existing`
//!
//! ### REQ-LATEX: latex
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/latex.md`
//! - Install targets:
//!   - `make latex`
//!
//! ### REQ-LBM-CUDA: lbm_3d_cuda
//!
//! - Status: `active`
//! - Markdown: `crates/lbm_3d_cuda/README.md`
//! - Install targets:
//!   - `cargo check -p lbm_3d_cuda`
//!
//! ### REQ-MATERIALS: materials
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/materials.md`
//!
//! ### REQ-PARTICLE: particle
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/particle.md`
//! - Install targets:
//!   - `make install-particle`
//!
//! ### REQ-QUANTUM: quantum_docker
//!
//! - Status: `active`
//! - Markdown: `docs/requirements/quantum-docker.md`
//! - Install targets:
//!   - `make install-quantum`
//!   - `docker-quantum-build`
//!   - `docker-quantum-run`
//!
//! ### REQ-STUDIO: gororoba_studio
//!
//! - Status: `active`
//! - Markdown: `apps/gororoba_studio/README.md`
//! - Install targets:
//!   - `make studio-run`
//!
//! ## Coverage Gaps
//!
//! ### REQ-GAP-001: crate_specific_docs
//!
//! - Status: `open`
//! - Description: Per-crate requirements markdown is missing for several Rust crates.
//! - Proposed resolution: Define crate-level requirements entries in this TOML and generate markdown stubs as needed.
//!
//! ### REQ-GAP-002: operational_sync
//!
//! - Status: `open`
//! - Description: Requirements markdown export is in a mixed compatibility state: Rust registry emitters now own several mirror lanes, while the remaining legacy markdown compatibility export still flows through Makefile registry-export-markdown.
//! - Proposed resolution: Continue porting the remaining legacy markdown export and freshness-verification seams into Rust. Roadmap, next-actions, navigator, entrypoint-docs, requirements, docs-root narratives, and research narratives are already Rust-emitted; the remaining compatibility export should be reduced until Makefile registry-export-markdown no longer depends on Python.
