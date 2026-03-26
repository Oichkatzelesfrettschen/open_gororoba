//! Unified search engine combining multiple sources.

use crate::{
    cache::{get_cached_search, put_cached_search},
    dedup::deduplicate,
    domain_queries::get_domain_queries,
    models::Paper,
    query_adapter::adapt_query,
    sources::{self, ApiKeys, SourceError},
};
use reqwest::Client;
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
        let mut all_papers = Vec::new();

        let openalex_query = adapt_query(query, "openalex", year_min);
        let semantic_query = adapt_query(query, "semantic_scholar", year_min);
        let arxiv_query = adapt_query(query, "arxiv", year_min);

        // Tier 0: always-on
        let (oa_res, s2_res) = tokio::join!(
            cached_source_search(
                "openalex",
                &openalex_query,
                limit,
                sources::search_openalex(&self.client, &openalex_query, limit, year_min),
            ),
            cached_source_search(
                "semantic_scholar",
                &semantic_query,
                limit,
                sources::search_semantic_scholar(&self.client, &semantic_query, limit, &self.keys),
            ),
        );

        if let Ok(papers) = oa_res {
            tracing::info!("OpenAlex: {} results", papers.len());
            all_papers.extend(papers);
        } else {
            tracing::warn!("OpenAlex failed: {:?}", oa_res.err());
        }

        if let Ok(papers) = s2_res {
            tracing::info!("S2: {} results", papers.len());
            all_papers.extend(papers);
        } else {
            tracing::warn!("S2 failed: {:?}", s2_res.err());
        }

        // Tier 0: arXiv
        if let Ok(papers) = cached_source_search(
            "arxiv",
            &arxiv_query,
            limit,
            sources::search_arxiv(&self.client, &arxiv_query, limit),
        )
        .await
        {
            tracing::info!("arXiv: {} results", papers.len());
            all_papers.extend(papers);
        }

        // Tier 1: open sources (no key needed)
        if self.tier != SourceTier::Core {
            let crossref_query = adapt_query(query, "crossref", year_min);
            let inspirehep_query = adapt_query(query, "inspirehep", year_min);
            let dblp_query = adapt_query(query, "dblp", year_min);
            let europepmc_query = adapt_query(query, "europepmc", year_min);
            let hal_query = adapt_query(query, "hal", year_min);
            let datacite_query = adapt_query(query, "datacite", year_min);
            let scielo_query = adapt_query(query, "scielo", year_min);
            let jstage_query = adapt_query(query, "jstage", year_min);

            let (cr, ihep, dblp, epmc, hal, dc, scielo, jst) = tokio::join!(
                cached_source_search(
                    "crossref",
                    &crossref_query,
                    limit,
                    sources::search_crossref(&self.client, &crossref_query, limit),
                ),
                cached_source_search(
                    "inspirehep",
                    &inspirehep_query,
                    limit,
                    sources::search_inspirehep(&self.client, &inspirehep_query, limit),
                ),
                cached_source_search(
                    "dblp",
                    &dblp_query,
                    limit,
                    sources::search_dblp(&self.client, &dblp_query, limit),
                ),
                cached_source_search(
                    "europepmc",
                    &europepmc_query,
                    limit,
                    sources::search_europepmc(&self.client, &europepmc_query, limit),
                ),
                cached_source_search(
                    "hal",
                    &hal_query,
                    limit,
                    sources::search_hal(&self.client, &hal_query, limit),
                ),
                cached_source_search(
                    "datacite",
                    &datacite_query,
                    limit,
                    sources::search_datacite(&self.client, &datacite_query, limit),
                ),
                cached_source_search(
                    "scielo",
                    &scielo_query,
                    limit,
                    sources::search_scielo(&self.client, &scielo_query, limit),
                ),
                cached_source_search(
                    "jstage",
                    &jstage_query,
                    limit,
                    sources::search_jstage(&self.client, &jstage_query, limit),
                ),
            );

            for (name, res) in [
                ("Crossref", cr),
                ("InspireHEP", ihep),
                ("DBLP", dblp),
                ("EuropePMC", epmc),
                ("HAL", hal),
                ("DataCite", dc),
                ("SciELO", scielo),
                ("J-STAGE", jst),
            ] {
                match res {
                    Ok(papers) if !papers.is_empty() => {
                        tracing::info!("{name}: {} results", papers.len());
                        all_papers.extend(papers);
                    }
                    Err(e) => tracing::debug!("{name}: {e}"),
                    _ => {}
                }
            }
        }

        // Tier 2: keyed sources + Google Scholar
        if self.tier == SourceTier::All {
            let core_query = adapt_query(query, "core", year_min);
            let cinii_query = adapt_query(query, "cinii", year_min);
            let ads_query = adapt_query(query, "ads", year_min);
            let lens_query = adapt_query(query, "lens", year_min);
            let (core_r, cinii_r, ads_r, lens_r, gs_r) = tokio::join!(
                cached_source_search(
                    "core",
                    &core_query,
                    limit,
                    sources::search_core(&self.client, &core_query, limit, &self.keys),
                ),
                cached_source_search(
                    "cinii",
                    &cinii_query,
                    limit,
                    sources::search_cinii(&self.client, &cinii_query, limit, &self.keys),
                ),
                cached_source_search(
                    "ads",
                    &ads_query,
                    limit,
                    sources::search_ads(&self.client, &ads_query, limit, &self.keys),
                ),
                cached_source_search(
                    "lens",
                    &lens_query,
                    limit,
                    sources::search_lens(&self.client, &lens_query, limit, &self.keys),
                ),
                cached_source_search(
                    "google_scholar",
                    query,
                    limit,
                    sources::search_google_scholar(&self.client, query, limit),
                ),
            );

            for (name, res) in [
                ("CORE", core_r),
                ("CiNii", cinii_r),
                ("ADS", ads_r),
                ("Lens", lens_r),
                ("GScholar", gs_r),
            ] {
                match res {
                    Ok(papers) if !papers.is_empty() => {
                        tracing::info!("{name}: {} results", papers.len());
                        all_papers.extend(papers);
                    }
                    Err(e) => tracing::debug!("{name}: {e}"),
                    _ => {}
                }
            }
        }

        // Check Unpaywall for OA PDFs on papers with DOIs but no PDF
        if self.tier == SourceTier::All && !self.keys.unpaywall_email.is_empty() {
            for paper in &mut all_papers {
                if paper.pdf_url.is_empty() && !paper.doi.is_empty() {
                    if let Ok(Some(pdf_url)) =
                        sources::check_unpaywall(&self.client, &paper.doi, &self.keys).await
                    {
                        paper.pdf_url = pdf_url;
                    }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }

        // Deduplicate and sort
        let mut results = deduplicate(all_papers);
        results.sort_by_key(|p| std::cmp::Reverse(p.citation_count));

        results
    }

    /// Search a topic plus domain-specific expansion queries.
    pub async fn search_topic(
        &self,
        topic: &str,
        domains: &[String],
        limit: usize,
        year_min: u32,
    ) -> Vec<Paper> {
        let mut queries = vec![topic.to_string()];
        for expanded in get_domain_queries(topic, domains) {
            if !queries
                .iter()
                .any(|existing| existing.eq_ignore_ascii_case(&expanded))
            {
                queries.push(expanded);
            }
        }

        let mut all_papers = Vec::new();
        for expanded_query in queries {
            tracing::info!("Expanded search query: {}", expanded_query);
            all_papers.extend(self.search(&expanded_query, limit, year_min).await);
        }

        let mut results = deduplicate(all_papers);
        results.sort_by_key(|p| std::cmp::Reverse(p.citation_count));
        results
    }

    /// Search by DOI across all sources.
    pub async fn search_by_doi(&self, doi: &str) -> Vec<Paper> {
        self.search(&format!("DOI:{doi}"), 5, 0).await
    }

    /// Access the underlying HTTP client (for download module reuse).
    pub fn client(&self) -> &Client {
        &self.client
    }
}

async fn cached_source_search<F>(
    source: &str,
    query: &str,
    limit: usize,
    fetch: F,
) -> Result<Vec<Paper>, SourceError>
where
    F: std::future::Future<Output = Result<Vec<Paper>, SourceError>>,
{
    if let Some(papers) = get_cached_search(query, source, limit) {
        tracing::debug!("{source} cache hit: {} results", papers.len());
        return Ok(papers);
    }

    let papers = fetch.await?;
    put_cached_search(query, source, limit, &papers);
    Ok(papers)
}
