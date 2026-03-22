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

pub mod models;
pub mod sources;
pub mod dedup;
pub mod search;
pub mod download;

pub use models::{Paper, Author};
pub use search::SearchEngine;
pub use download::{download_pdfs, DownloadResult};
