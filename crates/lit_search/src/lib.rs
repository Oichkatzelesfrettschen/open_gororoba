//! Multi-source academic literature search with deduplication.
//!
//! Pure Rust port of AutoResearchClaw's Python literature search pipeline.
//! Queries 17 academic APIs in parallel, deduplicates by DOI / arXiv ID /
//! fuzzy title match, and returns merged results sorted by citation count.
//!
//! # Source Tiers
//!
//! - **Tier 0** (always-on): OpenAlex, Semantic Scholar, arXiv
//! - **Tier 1** (open, no key): Crossref, EuropePMC, HAL, DataCite,
//!   SciELO, InspireHEP, DBLP, J-STAGE
//! - **Tier 2** (requires API key or scraping): CORE, ADS, CiNii, Lens,
//!   Unpaywall, Google Scholar

pub mod dedup;
pub mod download;
pub mod models;
pub mod search;
pub mod sources;

pub use download::{DownloadResult, download_pdfs};
pub use models::{Author, Paper};
pub use search::SearchEngine;
