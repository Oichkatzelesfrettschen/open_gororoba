//! # License Matrix for the Lambda Workspace
//!
//! ## Scope and intention
//! This workspace combines repositories that are now normalized to a common licensing baseline.
//! The workspace itself is an organizational layer and does not replace per-module records.
//!
//! ## Module legal posture
//!
//! | Module | Workspace path | License file | SPDX family | Integration implication |
//! |---|---|---|---|---|
//! | lambda-research | `research/` (root authoritative module) | `LICENSE` | GPL-2.0-only | Requires copyleft propagation for distribution; derivative works remain GPL-2.0-only-compatible. |
//! | lambda-synthesis-experiments | `experiments/` (root authoritative module) | `LICENSE` | GPL-2.0-only | Same copyleft scope as workspace target after harmonization. |
//! | LambdaLearner | `learner/` (root authoritative module) | `LICENSE` | GPL-2.0-only | Added for unified workspace target consistency. |
//!
//! ## Workspace policy
//! - Keep repositories as separate legal units.
//! - Record provenance of each third-party dependency in each module's `requirements.md` or equivalent package manifest.
//! - Consolidated outputs may reference the workspace target as `GPL-2.0-only` only after this snapshot is current.
//!
//! ## Short-term legal plan
//! 1. Keep module license files synchronized to `GPL-2.0-only` text.
//! 2. Preserve module boundaries in `research/`, `experiments/`, and `learner/` while sharing licensing assumptions.
//! 3. For any new external artifact, ensure `LICENSE` is present and copied into module seed with `source` references preserved.
//!
