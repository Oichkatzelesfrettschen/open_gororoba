//! # Registry Check Timing Baseline (2026-03-13)
//!
//! This snapshot records the first clean post-cutover timing baseline for
//! `registry-check` after the SQLite-first control-plane migration and the
//! freshness-gate reconciliation work.
//!
//! Command set:
//!
//! ```sh
//! cargo run -p gororoba_cli_data --bin registry-check -- \
//!   --canonical-db registry/canonical/control_plane.sqlite3 \
//!   --timings --compat-export-policy off
//!
//! cargo run -p gororoba_cli_data --bin registry-check -- \
//!   --canonical-db registry/canonical/control_plane.sqlite3 \
//!   --timings
//! ```
//!
//! ## Results
//!
//! Source-invariants mode (`--compat-export-policy off`):
//!
//! - `parse_sweep = 0.243s`
//! - `canonical_setup = 0.007s`
//! - `claims = 0.140s`
//! - `insights = 0.022s`
//! - `experiments = 0.041s`
//! - `binaries = 0.031s`
//! - `project = 0.005s`
//! - result: `PASS with 141 warnings`
//!
//! Full freshness mode (default compat export policy):
//!
//! - `parse_sweep = 0.238s`
//! - `canonical_setup = 0.013s`
//! - `claims = 0.139s`
//! - `insights = 0.022s`
//! - `experiments = 0.041s`
//! - `binaries = 0.031s`
//! - `project = 0.004s`
//! - result: `PASS with 141 warnings`
//!
//! ## Observations
//!
//! - The SQLite control-plane setup cost is small relative to the claims lane.
//! - The difference between source-only mode and full freshness mode is currently
//!   concentrated in `canonical_setup` and is roughly `0.006s` in this snapshot.
//! - The dominant remaining noise is not runtime failure but policy warnings:
//!   identity-gap drift and legacy status vocabulary in claims/insights.
//!
//! ## Immediate follow-up
//!
//! - Expand this timing baseline to the other heavy governance binaries:
//!   `execution-planning`, `integrity-resolution`, and `governance-verify`.
//! - Separate warning-volume reduction from timing work so performance numbers stay
//!   interpretable.
//!
//! ## Heavy governance binaries
//!
//! Warm-run command set:
//!
//! ```sh
//! /usr/bin/time -p cargo run -p gororoba_cli_data --bin execution-planning -- --verify
//! /usr/bin/time -p cargo run -p gororoba_cli_data --bin integrity-resolution -- --verify
//! /usr/bin/time -p cargo run -p gororoba_cli_data --bin governance-verify -- crossrefs
//! ```
//!
//! Warm-run wall times:
//!
//! - `execution-planning --verify`: `real 7.98s`
//! - `integrity-resolution --verify`: `real 19.39s`
//! - `governance-verify crossrefs`: `real 28.38s`
//!
//! Notes:
//!
//! - `execution-planning --verify` initially failed until the planning lane was
//!   regenerated after the control-plane refresh. After regeneration it passed.
//! - The widened timing pass also exposed two missing dependency declarations that
//!   were fixed before the final timings were taken:
//!   - `verified_core` needed `rayon`
//!   - `cd_kernel` needed `verified_core`
//!
