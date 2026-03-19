//! <!-- AUTO-GENERATED: DO NOT EDIT -->
//! <!-- Source of truth: registry/book_docs.toml -->
//!
//! # Theorem Linking
//!
//! The theorem lane connects three things:
//!
//! - Rocq proof files in `proofs/`
//! - theorem inventory rows in the canonical SQLite control plane
//! - linked claim IDs in the claims lane
//!
//! ## Canonical sources
//!
//! - proof manifest: `proofs/_RocqProject`
//! - canonical structured state: `registry/canonical/control_plane.sqlite3`
//! - generated web views: `docs/THEOREMS.md` and
//!   `docs/generated/THEOREMS_REGISTRY_MIRROR.md`
//!
//! ## Linking rule
//!
//! Each theorem row should link to one or more claim IDs unless it is in the
//! small justified-unlinked allowlist used for historical or structural proof
//! artifacts.
//!
//! The control-plane verifier checks:
//!
//! - proof path exists on disk
//! - linked claim IDs exist
//! - unexpected unlinked theorem rows fail verification
//!
//! ## Why the link matters
//!
//! Proofs are not just a file inventory. They feed:
//!
//! - `kernel_checked_claims`
//! - theorem-facing documentation
//! - claim-level formal proof status
//!
//! This is why theorem-link propagation is part of the control-plane verification
//! contract rather than a documentation-only feature.
//!
