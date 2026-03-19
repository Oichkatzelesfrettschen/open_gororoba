//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->
//!
//! # Requirements for Gemini Experiments
//!
//! This project follows a "Maximal Synthesis" protocol. Installs are designed to be reproducible,
//! offline-testable by default, and compatible with warnings-as-errors.
//!
//! If you need module-specific details, see:
//! - `docs/requirements/algebra.md`
//! - `docs/requirements/analysis.md`
//! - `docs/requirements/astro.md`
//! - `docs/requirements/materials.md`
//! - `docs/requirements/quantum-docker.md`
//! - `docs/requirements/rocq.md`
//! - `docs/requirements/latex.md`
//! - `apps/gororoba_studio/README.md`
//!
//! ## Python version policy
//!
//! - **Recommended**: Python 3.11 or 3.12 (best wheel availability).
//! - **Allowed**: Python 3.13+ for the core engine, but some optional extras may be skipped by
//!   dependency markers or may not have wheels yet (use Docker or a Python 3.11/3.12 env).
//!
//!
//! ## Core Rust Workflow
//!
//! The repository is a pure Rust workspace. All mathematical and scientific logic is driven via `cargo`.
//!
//! ### Quality Gates
//! - `make lint`: Workspace-wide Clippy
//! - `make test`: Full regression via nextest
//! - `make smoke`: Fast verification lane
//!
