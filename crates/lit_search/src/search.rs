//! Unified search engine combining multiple sources.

use crate::dedup::deduplicate;
use crate::models::Paper;
use crate::sources::{self, ApiKeys};
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
    pub async fn search(
        &self,
        query: &str,
        limit: usize,
        year_min: u32,
    ) -> Vec<Paper> {
        let mut all_papers = Vec::new();

        // Tier 0: always-on
        let (oa_res, s2_res) = tokio::join!(
            sources::search_openalex(&self.client, query, limit, year_min),
            sources::search_semantic_scholar(&self.client, query, limit, &self.keys),
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
        if let Ok(papers) = sources::search_arxiv(&self.client, query, limit).await {
            tracing::info!("arXiv: {} results", papers.len());
            all_papers.extend(papers);
        }

        // Tier 1: open sources (no key needed)
        if self.tier != SourceTier::Core {
            let (cr, ihep, dblp, epmc, hal, dc, scielo, jst) = tokio::join!(
                sources::search_crossref(&self.client, query, limit),
                sources::search_inspirehep(&self.client, query, limit),
                sources::search_dblp(&self.client, query, limit),
                sources::search_europepmc(&self.client, query, limit),
                sources::search_hal(&self.client, query, limit),
                sources::search_datacite(&self.client, query, limit),
                sources::search_scielo(&self.client, query, limit),
                sources::search_jstage(&self.client, query, limit),
            );

            for (name, res) in [
                ("Crossref", cr), ("InspireHEP", ihep), ("DBLP", dblp),
                ("EuropePMC", epmc), ("HAL", hal), ("DataCite", dc),
                ("SciELO", scielo), ("J-STAGE", jst),
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
            let (core_r, cinii_r, ads_r, lens_r, gs_r) = tokio::join!(
                sources::search_core(&self.client, query, limit, &self.keys),
                sources::search_cinii(&self.client, query, limit, &self.keys),
                sources::search_ads(&self.client, query, limit, &self.keys),
                sources::search_lens(&self.client, query, limit, &self.keys),
                sources::search_google_scholar(&self.client, query, limit),
            );

            for (name, res) in [
                ("CORE", core_r), ("CiNii", cinii_r), ("ADS", ads_r), ("Lens", lens_r),
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
                    if let Ok(Some(pdf_url)) = sources::check_unpaywall(
                        &self.client, &paper.doi, &self.keys
                    ).await { paper.pdf_url = pdf_url; }
                    tokio::time::sleep(Duration::from_millis(100)).await;
                }
            }
        }

        // Deduplicate and sort
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
