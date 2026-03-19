//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/requirements.toml; registry/requirements_narrative.toml -->
//!
//! # Requirements: Rocq Proof Checking
//!
//! This repo contains `.v` files under `curated/01_theory_frameworks/`.
//!
//! Install Rocq (example using `opam`):
//! ```bash
//! opam install rocq
//! ```
//!
//! Then run:
//! ```bash
//! make rocq
//! ```
//!
//! Notes:
//! - The Makefile checks for `coqc` on PATH (Rocq compiler).
//! - The current `confine_theorems_*.v` are statement inventories without proofs.
//! - `make rocq` generates `confine_theorems_*_axioms.v` (gitignored) to typecheck the interface.
//!
