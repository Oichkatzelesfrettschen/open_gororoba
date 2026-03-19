//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Execution Targets
//!
//! Experiments do not only point at CLI binaries. Some valid execution targets are
//! workspace benches.
//!
//! This matters because the control plane checks experiment lineage against the
//! workspace execution inventory.
//!
//! ## Current rule
//!
//! An execution target is valid if it is one of:
//!
//! - a workspace binary target
//! - a workspace bench target
//!
//! Examples:
//!
//! - `thesis-42-support` is a binary target
//! - `x87_bench` is a bench target
//!
//! ## Why this page exists
//!
//! We previously had a false mismatch because the repo treated benches as missing
//! binaries. The current contract is stricter and more accurate:
//!
//! - active experiments must resolve to a real execution target
//! - planned or blocked experiments may reference future targets without failing
//!   the active execution-target gate
//!
//! ## Validation
//!
//! The execution-target contract is checked by:
//!
//! - `execution-planning`
//! - `registry-check`
//!
//! Both now use Cargo target discovery rather than a narrower hand-maintained view
//! of the workspace.
//!
