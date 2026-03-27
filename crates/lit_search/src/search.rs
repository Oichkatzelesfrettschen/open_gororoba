//! Unified search engine combining multiple sources.

use crate::{
    cache::{get_cached_search, put_cached_search},
    dedup::{canonicalize_doi, deduplicate},
    domain_queries::get_domain_queries,
    models::Paper,
    query_adapter::adapt_query,
    sources::{self, ApiKeys, SourceError},
};
use futures::future::join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Which source tiers to enable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceTier {
    /// Tier 0 only: OpenAlex, S2, arXiv
    Core,
    /// Tier 0 + Tier 1 (open, no key)
    Open,
    /// All tiers including keyed sources
    All,
}

pub const CORE_SOURCE_NAMES: &[&str] = &["openalex", "semantic_scholar", "arxiv"];
pub const OPEN_SOURCE_NAMES: &[&str] = &[
    "crossref",
    "europepmc",
    "hal",
    "datacite",
    "scielo",
    "inspirehep",
    "dblp",
    "jstage",
    "open_library",
    "internet_archive",
    "google_books",
    "hathitrust",
    "worldcat",
];
pub const KEYED_SOURCE_NAMES: &[&str] = &["core", "cinii", "ads", "lens", "google_scholar"];
pub const AUGMENT_ONLY_SOURCE_NAMES: &[&str] = &["unpaywall"];
pub const MIRROR_HUNT_SOURCE_NAMES: &[&str] = &[
    "openalex",
    "semantic_scholar",
    "crossref",
    "datacite",
    "unpaywall",
    "hal",
    "europepmc",
    "scielo",
    "dblp",
    "jstage",
    "open_library",
    "internet_archive",
    "google_books",
    "hathitrust",
    "worldcat",
    "core",
    "cinii",
    "lens",
    "google_scholar",
];
pub const SEARCHABLE_SOURCE_NAMES: &[&str] = &[
    "openalex",
    "semantic_scholar",
    "arxiv",
    "crossref",
    "europepmc",
    "hal",
    "datacite",
    "scielo",
    "inspirehep",
    "dblp",
    "jstage",
    "open_library",
    "internet_archive",
    "google_books",
    "hathitrust",
    "worldcat",
    "core",
    "cinii",
    "ads",
    "lens",
    "google_scholar",
];
pub const ALL_SOURCE_NAMES: &[&str] = &[
    "openalex",
    "semantic_scholar",
    "arxiv",
    "crossref",
    "europepmc",
    "hal",
    "datacite",
    "scielo",
    "inspirehep",
    "dblp",
    "jstage",
    "open_library",
    "internet_archive",
    "google_books",
    "hathitrust",
    "worldcat",
    "core",
    "cinii",
    "ads",
    "lens",
    "google_scholar",
    "unpaywall",
];
pub const SOURCE_FAMILY_NAMES: &[&str] = &[
    "core",
    "open",
    "keyed",
    "augment",
    "citation_graph",
    "resolver",
    "archive",
    "catalog",
    "holder_catalog",
    "mirror_hunt",
];

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SourceExecutionReport {
    pub source: String,
    pub adapted_query: String,
    pub result_count: usize,
    pub cache_hit: bool,
    pub error: String,
    pub skipped_reason: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchExecutionReport {
    pub query: String,
    pub limit: usize,
    pub year_min: u32,
    pub requested_sources: Vec<String>,
    pub raw_result_count: usize,
    pub deduplicated_result_count: usize,
    pub cache_hit_count: usize,
    pub source_reports: Vec<SourceExecutionReport>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SearchExecutionOutcome {
    pub papers: Vec<Paper>,
    pub raw_papers: Vec<Paper>,
    pub report: SearchExecutionReport,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiQueryExecutionReport {
    pub queries: Vec<String>,
    pub limit: usize,
    pub year_min: u32,
    pub requested_sources: Vec<String>,
    pub raw_result_count: usize,
    pub deduplicated_result_count: usize,
    pub query_reports: Vec<SearchExecutionReport>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MultiQueryExecutionOutcome {
    pub papers: Vec<Paper>,
    pub raw_papers: Vec<Paper>,
    pub report: MultiQueryExecutionReport,
}

#[derive(Debug, Clone, Default)]
struct RequestedSourcePlan {
    requested_sources: Vec<String>,
    primary_sources: Vec<String>,
    augment_unpaywall: bool,
    invalid_sources: Vec<String>,
}

#[derive(Debug)]
struct SourceFetchOutcome {
    papers: Vec<Paper>,
    cache_hit: bool,
}

#[derive(Debug)]
struct SourceTaskOutcome {
    source: String,
    adapted_query: String,
    outcome: Result<SourceFetchOutcome, SourceError>,
}

pub fn source_names_for_tier(tier: SourceTier) -> &'static [&'static str] {
    match tier {
        SourceTier::Core => CORE_SOURCE_NAMES,
        SourceTier::Open => &[
            "openalex",
            "semantic_scholar",
            "arxiv",
            "crossref",
            "europepmc",
            "hal",
            "datacite",
            "scielo",
            "inspirehep",
            "dblp",
            "jstage",
            "open_library",
            "internet_archive",
            "google_books",
            "hathitrust",
            "worldcat",
        ],
        SourceTier::All => &[
            "openalex",
            "semantic_scholar",
            "arxiv",
            "crossref",
            "europepmc",
            "hal",
            "datacite",
            "scielo",
            "inspirehep",
            "dblp",
            "jstage",
            "open_library",
            "internet_archive",
            "google_books",
            "hathitrust",
            "worldcat",
            "core",
            "cinii",
            "ads",
            "lens",
            "google_scholar",
        ],
    }
}

pub fn normalize_source_name(source: &str) -> String {
    match source
        .trim()
        .to_ascii_lowercase()
        .replace(['-', ' '], "_")
        .as_str()
    {
        "s2" => "semantic_scholar".to_string(),
        normalized => normalized.to_string(),
    }
}

pub fn normalize_source_family_name(family: &str) -> String {
    let normalized = family.trim().to_ascii_lowercase().replace(['-', ' '], "_");
    normalized
        .strip_prefix("family:")
        .unwrap_or(&normalized)
        .to_string()
}

pub fn source_names_for_family(family: &str) -> Option<&'static [&'static str]> {
    match normalize_source_family_name(family).as_str() {
        "core" => Some(CORE_SOURCE_NAMES),
        "open" => Some(OPEN_SOURCE_NAMES),
        "keyed" => Some(KEYED_SOURCE_NAMES),
        "augment" => Some(AUGMENT_ONLY_SOURCE_NAMES),
        "citation_graph" => Some(&["openalex", "semantic_scholar"]),
        "resolver" => Some(&["crossref", "datacite", "unpaywall"]),
        "archive" => Some(&[
            "arxiv",
            "jstage",
            "inspirehep",
            "hal",
            "dblp",
            "internet_archive",
        ]),
        "catalog" => Some(&[
            "europepmc",
            "scielo",
            "core",
            "cinii",
            "ads",
            "lens",
            "open_library",
            "google_books",
            "hathitrust",
            "worldcat",
        ]),
        "holder_catalog" => Some(&[
            "google_scholar",
            "crossref",
            "cinii",
            "core",
            "lens",
            "dblp",
            "open_library",
            "google_books",
            "hathitrust",
            "worldcat",
        ]),
        "mirror_hunt" => Some(MIRROR_HUNT_SOURCE_NAMES),
        _ => None,
    }
}

/// Main search engine.
pub struct SearchEngine {
    client: Client,
    keys: ApiKeys,
    tier: SourceTier,
}

impl SearchEngine {
    pub fn new(keys: ApiKeys, tier: SourceTier) -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .user_agent("lit_search/0.1 (gororoba research; mailto:research@gororoba.dev)")
            .build()
            .expect("Failed to build HTTP client");

        Self { client, keys, tier }
    }

    /// Search all enabled sources and return deduplicated results.
    pub async fn search(&self, query: &str, limit: usize, year_min: u32) -> Vec<Paper> {
        self.search_with_sources(query, limit, year_min, &[])
            .await
            .papers
    }

    /// Search using explicit source selection and return results plus execution stats.
    pub async fn search_with_sources(
        &self,
        query: &str,
        limit: usize,
        year_min: u32,
        requested_sources: &[String],
    ) -> SearchExecutionOutcome {
        let plan = resolve_requested_sources(self.tier, requested_sources);
        let source_runs = join_all(
            plan.primary_sources
                .iter()
                .map(|source| self.run_named_source(source, query, limit, year_min)),
        )
        .await;

        let mut all_papers = Vec::new();
        let mut cache_hit_count = 0;
        let mut source_reports = Vec::new();

        for invalid in &plan.invalid_sources {
            source_reports.push(SourceExecutionReport {
                source: invalid.clone(),
                adapted_query: query.to_string(),
                result_count: 0,
                cache_hit: false,
                error: String::new(),
                skipped_reason: "unknown_source".to_string(),
            });
        }

        for source_run in source_runs {
            match source_run.outcome {
                Ok(outcome) => {
                    let normalized_papers = normalize_fetched_papers(outcome.papers);
                    if outcome.cache_hit {
                        cache_hit_count += 1;
                    }
                    source_reports.push(SourceExecutionReport {
                        source: source_run.source,
                        adapted_query: source_run.adapted_query,
                        result_count: normalized_papers.len(),
                        cache_hit: outcome.cache_hit,
                        error: String::new(),
                        skipped_reason: String::new(),
                    });
                    all_papers.extend(normalized_papers);
                }
                Err(err) => {
                    source_reports.push(SourceExecutionReport {
                        source: source_run.source,
                        adapted_query: source_run.adapted_query,
                        result_count: 0,
                        cache_hit: false,
                        error: err.to_string(),
                        skipped_reason: String::new(),
                    });
                }
            }
        }

        if plan.augment_unpaywall {
            if self.keys.unpaywall_email.is_empty() {
                source_reports.push(SourceExecutionReport {
                    source: "unpaywall".to_string(),
                    adapted_query: query.to_string(),
                    result_count: 0,
                    cache_hit: false,
                    error: String::new(),
                    skipped_reason: "missing_unpaywall_email".to_string(),
                });
            } else {
                let mut augmented = 0;
                for paper in &mut all_papers {
                    if paper.pdf_url.is_empty() && !paper.doi.is_empty() {
                        if let Ok(Some(pdf_url)) =
                            sources::check_unpaywall(&self.client, &paper.doi, &self.keys).await
                        {
                            paper.pdf_url = pdf_url;
                            augmented += 1;
                        }
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
                source_reports.push(SourceExecutionReport {
                    source: "unpaywall".to_string(),
                    adapted_query: query.to_string(),
                    result_count: augmented,
                    cache_hit: false,
                    error: String::new(),
                    skipped_reason: String::new(),
                });
            }
        }

        let raw_papers = all_papers;
        let raw_result_count = raw_papers.len();
        let mut results = deduplicate(raw_papers.clone());
        results.sort_by_key(|paper| std::cmp::Reverse(paper.citation_count));
        let deduplicated_result_count = results.len();

        SearchExecutionOutcome {
            papers: results,
            raw_papers,
            report: SearchExecutionReport {
                query: query.to_string(),
                limit,
                year_min,
                requested_sources: plan.requested_sources,
                raw_result_count,
                deduplicated_result_count,
                cache_hit_count,
                source_reports,
            },
        }
    }

    /// Search a topic plus domain-specific expansion queries.
    pub async fn search_topic(
        &self,
        topic: &str,
        domains: &[String],
        limit: usize,
        year_min: u32,
    ) -> Vec<Paper> {
        self.search_topic_with_sources(topic, domains, limit, year_min, &[])
            .await
            .papers
    }

    /// Search a topic plus domain-specific expansion queries with explicit sources.
    pub async fn search_topic_with_sources(
        &self,
        topic: &str,
        domains: &[String],
        limit: usize,
        year_min: u32,
        requested_sources: &[String],
    ) -> MultiQueryExecutionOutcome {
        let mut queries = vec![topic.to_string()];
        for expanded in get_domain_queries(topic, domains) {
            if !queries
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&expanded))
            {
                queries.push(expanded);
            }
        }
        self.search_queries_with_sources(&queries, limit, year_min, requested_sources)
            .await
    }

    /// Search by DOI across all sources.
    pub async fn search_by_doi(&self, doi: &str) -> Vec<Paper> {
        self.search(&format!("DOI:{doi}"), 5, 0).await
    }

    /// Search multiple explicit queries sequentially and merge the results.
    pub async fn search_queries(
        &self,
        queries: &[String],
        limit: usize,
        year_min: u32,
    ) -> Vec<Paper> {
        self.search_queries_with_sources(queries, limit, year_min, &[])
            .await
            .papers
    }

    /// Search multiple explicit queries sequentially with explicit sources.
    pub async fn search_queries_with_sources(
        &self,
        queries: &[String],
        limit: usize,
        year_min: u32,
        requested_sources: &[String],
    ) -> MultiQueryExecutionOutcome {
        let deduped_queries = deduplicate_queries(queries);
        let mut all_papers = Vec::new();
        let mut all_raw_papers = Vec::new();
        let mut query_reports = Vec::new();
        let mut raw_result_count = 0;
        let mut requested_report_sources = Vec::new();

        for query in &deduped_queries {
            tracing::info!("Sequential search query: {}", query);
            let outcome = self
                .search_with_sources(query, limit, year_min, requested_sources)
                .await;
            raw_result_count += outcome.report.raw_result_count;
            if requested_report_sources.is_empty() {
                requested_report_sources = outcome.report.requested_sources.clone();
            }
            query_reports.push(outcome.report);
            all_raw_papers.extend(outcome.raw_papers);
            all_papers.extend(outcome.papers);
        }

        let mut results = deduplicate(all_papers);
        results.sort_by_key(|paper| std::cmp::Reverse(paper.citation_count));
        let deduplicated_result_count = results.len();

        MultiQueryExecutionOutcome {
            papers: results,
            raw_papers: all_raw_papers,
            report: MultiQueryExecutionReport {
                queries: deduped_queries,
                limit,
                year_min,
                requested_sources: requested_report_sources,
                raw_result_count,
                deduplicated_result_count,
                query_reports,
            },
        }
    }

    /// Search multiple explicit queries in parallel and merge the results.
    pub async fn search_queries_parallel(
        &self,
        queries: &[String],
        limit: usize,
        year_min: u32,
    ) -> Vec<Paper> {
        self.search_queries_parallel_with_sources(queries, limit, year_min, &[])
            .await
            .papers
    }

    /// Search multiple explicit queries in parallel with explicit sources.
    pub async fn search_queries_parallel_with_sources(
        &self,
        queries: &[String],
        limit: usize,
        year_min: u32,
        requested_sources: &[String],
    ) -> MultiQueryExecutionOutcome {
        let deduped_queries = deduplicate_queries(queries);
        let tasks = deduped_queries.iter().map(|query| async move {
            tracing::info!("Parallel search query: {}", query);
            self.search_with_sources(query, limit, year_min, requested_sources)
                .await
        });
        let outcomes = join_all(tasks).await;
        let mut all_papers = Vec::new();
        let mut all_raw_papers = Vec::new();
        let mut query_reports = Vec::new();
        let mut raw_result_count = 0;
        let mut requested_report_sources = Vec::new();

        for outcome in outcomes {
            raw_result_count += outcome.report.raw_result_count;
            if requested_report_sources.is_empty() {
                requested_report_sources = outcome.report.requested_sources.clone();
            }
            query_reports.push(outcome.report);
            all_raw_papers.extend(outcome.raw_papers);
            all_papers.extend(outcome.papers);
        }

        let mut results = deduplicate(all_papers);
        results.sort_by_key(|paper| std::cmp::Reverse(paper.citation_count));
        let deduplicated_result_count = results.len();

        MultiQueryExecutionOutcome {
            papers: results,
            raw_papers: all_raw_papers,
            report: MultiQueryExecutionReport {
                queries: deduped_queries,
                limit,
                year_min,
                requested_sources: requested_report_sources,
                raw_result_count,
                deduplicated_result_count,
                query_reports,
            },
        }
    }

    /// Access the underlying HTTP client (for download module reuse).
    pub fn client(&self) -> &Client {
        &self.client
    }

    async fn run_named_source(
        &self,
        source: &str,
        query: &str,
        limit: usize,
        year_min: u32,
    ) -> SourceTaskOutcome {
        let adapted_query = if source == "google_scholar" {
            query.to_string()
        } else {
            adapt_query(query, source, year_min)
        };

        let outcome = match source {
            "openalex" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_openalex(&self.client, &adapted_query, limit, year_min),
                )
                .await
            }
            "semantic_scholar" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_semantic_scholar(
                        &self.client,
                        &adapted_query,
                        limit,
                        &self.keys,
                    ),
                )
                .await
            }
            "arxiv" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_arxiv(&self.client, &adapted_query, limit),
                )
                .await
            }
            "crossref" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_crossref(&self.client, &adapted_query, limit),
                )
                .await
            }
            "europepmc" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_europepmc(&self.client, &adapted_query, limit),
                )
                .await
            }
            "hal" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_hal(&self.client, &adapted_query, limit),
                )
                .await
            }
            "datacite" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_datacite(&self.client, &adapted_query, limit),
                )
                .await
            }
            "scielo" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_scielo(&self.client, &adapted_query, limit),
                )
                .await
            }
            "inspirehep" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_inspirehep(&self.client, &adapted_query, limit),
                )
                .await
            }
            "dblp" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_dblp(&self.client, &adapted_query, limit),
                )
                .await
            }
            "jstage" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_jstage(&self.client, &adapted_query, limit),
                )
                .await
            }
            "open_library" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_open_library(&self.client, &adapted_query, limit),
                )
                .await
            }
            "internet_archive" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_internet_archive(&self.client, &adapted_query, limit),
                )
                .await
            }
            "google_books" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_google_books(&self.client, &adapted_query, limit),
                )
                .await
            }
            "hathitrust" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_hathitrust(&self.client, &adapted_query, limit),
                )
                .await
            }
            "worldcat" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_worldcat(&self.client, &adapted_query, limit),
                )
                .await
            }
            "core" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_core(&self.client, &adapted_query, limit, &self.keys),
                )
                .await
            }
            "cinii" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_cinii(&self.client, &adapted_query, limit, &self.keys),
                )
                .await
            }
            "ads" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_ads(&self.client, &adapted_query, limit, &self.keys),
                )
                .await
            }
            "lens" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_lens(&self.client, &adapted_query, limit, &self.keys),
                )
                .await
            }
            "google_scholar" => {
                cached_source_search_outcome(
                    source,
                    &adapted_query,
                    limit,
                    sources::search_google_scholar(&self.client, &adapted_query, limit),
                )
                .await
            }
            _ => Err(SourceError::Unavailable(format!("unknown source {source}"))),
        };

        SourceTaskOutcome {
            source: source.to_string(),
            adapted_query,
            outcome,
        }
    }
}

fn deduplicate_queries(queries: &[String]) -> Vec<String> {
    let mut deduped = Vec::new();
    for query in queries {
        if !deduped
            .iter()
            .any(|existing: &String| existing.eq_ignore_ascii_case(query))
        {
            deduped.push(query.clone());
        }
    }
    deduped
}

fn normalize_fetched_papers(papers: Vec<Paper>) -> Vec<Paper> {
    papers
        .into_iter()
        .filter_map(normalize_fetched_paper)
        .collect()
}

fn normalize_fetched_paper(mut paper: Paper) -> Option<Paper> {
    if is_placeholder_paper(&paper) {
        return None;
    }

    if paper.url.trim().is_empty() {
        let canonical_doi = canonicalize_doi(&paper.doi);
        if !canonical_doi.is_empty() {
            paper.url = format!("https://doi.org/{canonical_doi}");
        }
    }

    if paper.pdf_url.trim().is_empty() && paper.url.to_ascii_lowercase().ends_with(".pdf") {
        paper.pdf_url = paper.url.clone();
    }

    Some(paper)
}

fn is_placeholder_paper(paper: &Paper) -> bool {
    let normalized_title = paper.title.trim().to_ascii_lowercase();
    if normalized_title.is_empty() {
        return true;
    }
    if normalized_title == "error" {
        return true;
    }
    let combined_url = format!("{} {}", paper.url, paper.pdf_url).to_ascii_lowercase();
    combined_url.contains("/api/errors")
}

fn resolve_requested_sources(
    tier: SourceTier,
    requested_sources: &[String],
) -> RequestedSourcePlan {
    if requested_sources.is_empty() {
        let primary_sources = source_names_for_tier(tier)
            .iter()
            .map(|source| (*source).to_string())
            .collect::<Vec<_>>();
        let mut requested = primary_sources.clone();
        let augment_unpaywall = tier == SourceTier::All;
        if augment_unpaywall {
            requested.push("unpaywall".to_string());
        }
        return RequestedSourcePlan {
            requested_sources: requested,
            primary_sources,
            augment_unpaywall,
            invalid_sources: Vec::new(),
        };
    }

    let mut requested = Vec::new();
    let mut primary_sources = Vec::new();
    let mut invalid_sources = Vec::new();
    let mut augment_unpaywall = false;

    for source in requested_sources {
        let normalized = normalize_source_name(source);
        let is_explicit_source = SEARCHABLE_SOURCE_NAMES
            .iter()
            .any(|candidate| candidate == &normalized.as_str())
            || normalized == "unpaywall";
        if is_explicit_source {
            if requested.iter().any(|existing| existing == &normalized) {
                continue;
            }
            requested.push(normalized.clone());
            if normalized == "unpaywall" {
                augment_unpaywall = true;
            } else {
                primary_sources.push(normalized);
            }
            continue;
        }
        if let Some(family_sources) = source_names_for_family(&normalized) {
            for family_source in family_sources {
                let family_source = family_source.to_string();
                if requested.iter().any(|existing| existing == &family_source) {
                    continue;
                }
                requested.push(family_source.clone());
                if family_source == "unpaywall" {
                    augment_unpaywall = true;
                } else {
                    primary_sources.push(family_source);
                }
            }
            continue;
        }
        if requested.iter().any(|existing| existing == &normalized) {
            continue;
        }
        requested.push(normalized.clone());
        if normalized == "unpaywall" {
            augment_unpaywall = true;
        } else if SEARCHABLE_SOURCE_NAMES
            .iter()
            .any(|candidate| candidate == &normalized.as_str())
        {
            primary_sources.push(normalized);
        } else {
            invalid_sources.push(normalized);
        }
    }

    RequestedSourcePlan {
        requested_sources: requested,
        primary_sources,
        augment_unpaywall,
        invalid_sources,
    }
}

async fn cached_source_search_outcome<F>(
    source: &str,
    query: &str,
    limit: usize,
    fetch: F,
) -> Result<SourceFetchOutcome, SourceError>
where
    F: std::future::Future<Output = Result<Vec<Paper>, SourceError>>,
{
    if let Some(papers) = get_cached_search(query, source, limit) {
        tracing::debug!("{source} cache hit: {} results", papers.len());
        return Ok(SourceFetchOutcome {
            papers,
            cache_hit: true,
        });
    }

    let papers = fetch.await?;
    put_cached_search(query, source, limit, &papers);
    Ok(SourceFetchOutcome {
        papers,
        cache_hit: false,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        Paper, SourceTier, deduplicate_queries, normalize_fetched_paper, normalize_source_name,
        resolve_requested_sources, source_names_for_family,
    };

    #[test]
    fn deduplicate_queries_is_case_insensitive_and_order_preserving() {
        let queries = vec![
            "Cayley Dickson".to_string(),
            "cayley dickson".to_string(),
            "zero divisors".to_string(),
        ];
        let deduped = deduplicate_queries(&queries);
        assert_eq!(deduped, vec!["Cayley Dickson", "zero divisors"]);
    }

    #[test]
    fn normalize_source_name_handles_aliases_and_spacing() {
        assert_eq!(normalize_source_name("S2"), "semantic_scholar");
        assert_eq!(normalize_source_name("Google Scholar"), "google_scholar");
        assert_eq!(normalize_source_name("crossref"), "crossref");
    }

    #[test]
    fn resolve_requested_sources_tracks_unpaywall_and_invalids() {
        let requested = vec![
            "S2".to_string(),
            "unpaywall".to_string(),
            "Google Scholar".to_string(),
            "mystery".to_string(),
        ];
        let plan = resolve_requested_sources(SourceTier::Open, &requested);
        assert_eq!(
            plan.primary_sources,
            vec!["semantic_scholar".to_string(), "google_scholar".to_string()]
        );
        assert!(plan.augment_unpaywall);
        assert_eq!(plan.invalid_sources, vec!["mystery".to_string()]);
    }

    #[test]
    fn resolve_requested_sources_expands_source_families() {
        let requested = vec!["family:resolver".to_string(), "archive".to_string()];
        let plan = resolve_requested_sources(SourceTier::Open, &requested);
        assert!(plan.primary_sources.contains(&"crossref".to_string()));
        assert!(plan.primary_sources.contains(&"datacite".to_string()));
        assert!(plan.augment_unpaywall);
        assert!(plan.primary_sources.contains(&"arxiv".to_string()));
        assert!(plan.primary_sources.contains(&"jstage".to_string()));
        assert!(source_names_for_family("archive").is_some());
    }

    #[test]
    fn resolve_requested_sources_expands_mirror_hunt_family() {
        let requested = vec!["mirror_hunt".to_string()];
        let plan = resolve_requested_sources(SourceTier::Open, &requested);
        assert!(plan.primary_sources.contains(&"openalex".to_string()));
        assert!(plan.primary_sources.contains(&"google_scholar".to_string()));
        assert!(plan.primary_sources.contains(&"open_library".to_string()));
        assert!(
            plan.primary_sources
                .contains(&"internet_archive".to_string())
        );
        assert!(plan.primary_sources.contains(&"hathitrust".to_string()));
        assert!(plan.primary_sources.contains(&"worldcat".to_string()));
        assert!(plan.primary_sources.contains(&"core".to_string()));
        assert!(plan.augment_unpaywall);
        assert!(source_names_for_family("mirror_hunt").is_some());
    }

    #[test]
    fn resolve_requested_sources_keeps_core_as_explicit_source() {
        let requested = vec!["core".to_string()];
        let plan = resolve_requested_sources(SourceTier::Open, &requested);
        assert_eq!(plan.primary_sources, vec!["core".to_string()]);
        assert_eq!(plan.requested_sources, vec!["core".to_string()]);
        assert!(!plan.augment_unpaywall);
        assert!(plan.invalid_sources.is_empty());
    }

    #[test]
    fn normalize_fetched_paper_adds_doi_url_and_filters_placeholders() {
        let paper = Paper {
            title: "Linear algebras with associativity not assumed".to_string(),
            doi: "https://doi.org/10.1215/S0012-7094-35-00112-0".to_string(),
            ..Default::default()
        };
        let normalized = normalize_fetched_paper(paper).expect("paper should be retained");
        assert_eq!(
            normalized.url,
            "https://doi.org/10.1215/s0012-7094-35-00112-0"
        );

        let placeholder = Paper {
            title: "Error".to_string(),
            url: "https://arxiv.org/api/errors".to_string(),
            ..Default::default()
        };
        assert!(normalize_fetched_paper(placeholder).is_none());
    }
}
