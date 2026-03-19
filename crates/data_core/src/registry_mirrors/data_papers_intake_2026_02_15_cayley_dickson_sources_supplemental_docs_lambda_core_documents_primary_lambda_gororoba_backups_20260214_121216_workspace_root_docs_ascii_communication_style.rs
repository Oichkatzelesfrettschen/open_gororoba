//! # ASCII-First Communication Style
//!
//! ## Purpose
//! Keep user updates visual, low-text, and easy to scan during merge, dedupe, and reconciliation work.
//!
//! ## Default format
//! 1. One-line status.
//! 2. ASCII diagram first.
//! 3. Short facts block (counts, conflicts, evidence files).
//! 4. Optional next actions as numbered choices.
//!
//! ## Constraints
//! - Use ASCII characters only in diagrams.
//! - Default diagram depth is two levels unless the user asks for more.
//! - Prefer exact paths and explicit timestamps.
//! - Keep prose short and avoid dense paragraphs.
//!
//! ## Reusable template
//! ```ignore
//! Status: <one sentence>
//!
//! <ASCII diagram>
//!
//! Snapshot:
//! - Timestamp: <YYYY-MM-DD HH:MM:SS>
//! - Counts: <module facts>
//! - Conflicts: <short list>
//! - Evidence: <path1>, <path2>
//!
//! Next options:
//! 1. <option A>
//! 2. <option B>
//! ```ignore
//!
//! ## Singular project mapping reminder
//! - Active sources: `research/`, `experiments/`, `learner/`
//! - Retired intake snapshots: `archive/intake_lane_retirement/<TS>/merge_in/*`
//! - Targets: `research/`, `experiments/`, `learner/`
//! - Detailed sync scope map: `docs/SYNC_SCOPE_FROM_SOURCE_FOLDERS.md`
//!
