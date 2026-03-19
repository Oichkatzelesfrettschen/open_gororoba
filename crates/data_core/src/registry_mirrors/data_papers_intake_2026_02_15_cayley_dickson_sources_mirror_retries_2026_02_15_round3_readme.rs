//! # Round-3 Mirror Retry Summary (2026-02-15)
//!
//! This round re-tried CORE, ScienceDirect, DOI, and SciSpace mirrors for the stubborn Eakin-Sathaye paper with stricter provenance capture and hash dedupe.
//!
//! ## Outcome
//! - `total_attempts = 18`
//! - `pdf_ok = 1` (Moreno companion paper, duplicate hash; not copied again)
//! - `ok_nonpdf = 12` (functional landing/navigation pages)
//! - `http_503 = 3` (CORE direct PDF variants)
//! - `http_404 = 1` (ScienceDirect `S002...` article URL variant)
//! - `http_405 = 1` (SciSpace endpoint)
//!
//! ## Key point
//! Functional links exist for Eakin-Sathaye (DOI/landing pages), but direct PDF routes remained gated or non-PDF in this CLI environment.
//!
//! ## Files
//! - `mirror_retry_results_round3.tsv`
//! - `pdf_success_added.tsv`
//! - `meta/status_counts.txt`
//! - `raw/`, `meta/` request artifacts
//!
