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
//! # Paper Collection
//!
//! The repo now has two paper layers that should be kept distinct:
//!
//! - `data/papers/corpus/` is the repo-owned executable corpus. It currently holds
//!   318 cached files, including PDFs, sidecar text, and a local `MANIFEST.toml`.
//! - `~/Documents/Projects/CayleyDickson/CHRONOLOGICAL_REFERENCE_MATRIX.md` is the
//!   broader off-repo chronology and audit surface. As of 2026-03-25 it tracks
//!   119 rows across precursor, quaternion, octonion, sedenion, and later CD work.
//!
//! The earlier "19 extracted papers" tranche is still useful, but it is no longer
//! the whole paper corpus. The active paper-to-code surfaces are:
//!
//! | Lane | Repo surface | Role |
//! |------|--------------|------|
//! | Rust paper API | `crates/cd_papers/` | Unified access to paper-scoped Rust crates |
//! | PDF and text intake | `crates/docpipe/` + `data/papers/corpus/` | Extraction, normalization, and equation intake |
//! | Rocq formalization | `proofs/theories/` | Exact theorem companions for paper results |
//! | Provenance registry | `registry/source_lanes/papers_pdf.toml` | Machine-readable paper inventory |
//!
//! ## Current paired paper lanes
//!
//! | Paper | Rust lane | Rocq lane |
//! |------|-----------|-----------|
//! | Hurwitz (1898) | `crates/hurwitz_1898/` | `proofs/theories/HurwitzTheorem.v` |
//! | Dickson (1919) | `crates/dickson_1919/` | `proofs/theories/DicksonCDProcess.v` |
//! | Schafer (1945) | `crates/schafer_1945/` | `proofs/theories/SchaferDivAlg16.v` |
//! | Brown (1967) | `crates/brown_1967/` | `proofs/theories/BrownGeneralizedCD.v` |
//! | Brown (1972) | `crates/brown_1972/` | `proofs/theories/Brown1972.v` |
//! | Moreno (1997) | `crates/moreno_1997/` | `proofs/theories/Moreno1997.v` |
//! | de Marrais (2000) | `crates/de_marrais_2000/` | `proofs/theories/DeMarraisAssessors.v` |
//! | Wilmot (2025) | `crates/wilmot_2025/` | `proofs/theories/WilmotCDStructure.v` and retraction companions |
//!
//! ## Extraction pipeline
//!
//! `docpipe` is already the right place for PDF and sidecar-text extraction, but
//! the equation lane needs a curated second stage. For OCR-heavy papers, raw PDF
//! text is often not enough to recover equations faithfully. The current strategy
//! is therefore:
//!
//! - use `docpipe` for text, sections, and structural extraction
//! - promote paper-specific equations into curated catalogs
//! - route each equation to Rust, Rocq, or a paired Rust+Rocq lane depending on
//!   whether the source result is computational, formal, or both
//!
//! ```sh
//! # Extract a specific cached paper
//! cargo run --release --bin extract-papers -- --only demarrais-2000-math0011260
//! ```
//!
