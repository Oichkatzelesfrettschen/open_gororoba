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

pub mod cache;
pub mod crawler;
pub mod critique;
pub mod dedup;
pub mod domain_queries;
pub mod download;
pub mod evolution;
pub mod models;
pub mod novelty;
pub mod pdf;
pub mod pipeline;
pub mod query_adapter;
pub mod search;
pub mod sources;
pub mod verify;

pub use download::{DownloadResult, download_pdfs};
pub use models::{Author, Paper};
pub use novelty::{NoveltyReport, SimilarPaper, check_novelty};
pub use search::SearchEngine;
pub use verify::{CitationResult, VerificationReport, VerifyStatus, verify_citations};
