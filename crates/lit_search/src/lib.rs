//! Multi-source academic literature search with deduplication.
//!
//! Pure Rust implementation of the repository's legacy literature search baseline.
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
//!
//! # User-local credentials
//!
//! `lit_search` reads optional keyed-source credentials from environment
//! variables only. Keep them in your user-local shell environment or secret
//! manager, not in the repository.
//!
//! Supported variables:
//!
//! - `SEMANTIC_SCHOLAR_API_KEY` (alias: `S2_API_KEY`)
//! - `CORE_API_KEY`
//! - `CINII_APPID`
//! - `UNPAYWALL_EMAIL`
//! - `ADS_API_KEY` (alias: `NASA_ADS_TOKEN`)
//! - `LENS_API_KEY`
//!
//! See `docs/engineering/lit_search_env_vars.txt` for installation guidance
//! and placeholder export examples.

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
pub mod semantic_scholar;
pub mod sources;
pub mod verify;

pub use dedup::canonicalize_doi;
pub use download::{DownloadResult, download_pdfs};
pub use models::{Author, Paper};
pub use novelty::{NoveltyReport, SimilarPaper, check_novelty};
pub use search::{
    ALL_SOURCE_NAMES, AUGMENT_ONLY_SOURCE_NAMES, CORE_SOURCE_NAMES, KEYED_SOURCE_NAMES,
    MultiQueryExecutionOutcome, MultiQueryExecutionReport, OPEN_SOURCE_NAMES,
    SEARCHABLE_SOURCE_NAMES, SOURCE_FAMILY_NAMES, SearchEngine, SearchExecutionOutcome,
    SearchExecutionReport, SourceExecutionReport, SourceTier, normalize_source_family_name,
    normalize_source_name, source_names_for_family, source_names_for_tier,
};
pub use semantic_scholar::{
    DatasetDiff, DatasetDiffEntry, DatasetManifest, DatasetRelease, DatasetSummary,
    SemanticScholarDatasetsClient, SemanticScholarError,
};
pub use verify::{CitationResult, VerificationReport, VerifyStatus, verify_citations};
