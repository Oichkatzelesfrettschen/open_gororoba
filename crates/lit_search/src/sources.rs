//! Academic paper source clients.
//!
//! Each source implements the `PaperSource` trait: an async function that
//! takes a query string and returns a list of papers.

use crate::models::{Author, Paper};
use reqwest::Client;
use serde_json::Value;

/// Error type for source queries.
#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("Rate limited")]
    RateLimited,
    #[error("Source unavailable: {0}")]
    Unavailable(String),
}

/// Configuration for API keys and credentials.
#[derive(Debug, Clone, Default)]
pub struct ApiKeys {
    pub s2_api_key: String,
    pub core_api_key: String,
    pub cinii_appid: String,
    pub unpaywall_email: String,
    pub ads_token: String,
    pub lens_api_key: String,
}

impl ApiKeys {
    /// Load from environment variables.
    pub fn from_env() -> Self {
        Self {
            s2_api_key: std::env::var("S2_API_KEY").unwrap_or_default(),
            core_api_key: std::env::var("CORE_API_KEY").unwrap_or_default(),
            cinii_appid: std::env::var("CINII_APPID").unwrap_or_default(),
            unpaywall_email: std::env::var("UNPAYWALL_EMAIL").unwrap_or_default(),
            ads_token: std::env::var("NASA_ADS_TOKEN").unwrap_or_default(),
            lens_api_key: std::env::var("LENS_API_KEY").unwrap_or_default(),
        }
    }
}

// ---------------------------------------------------------------------------
// OpenAlex (Tier 0, no key)
// ---------------------------------------------------------------------------

pub async fn search_openalex(
    client: &Client,
    query: &str,
    limit: usize,
    year_min: u32,
) -> Result<Vec<Paper>, SourceError> {
    let mut url = format!(
        "https://api.openalex.org/works?search={}&per_page={}",
        urlencoding::encode(query),
        limit.min(50)
    );
    if year_min > 0 {
        url.push_str(&format!("&filter=publication_year:>={year_min}"));
    }

    let resp: Value = client.get(&url).send().await?.json().await?;
    let results = resp["results"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(results.iter().map(|w| {
        let authors: Vec<Author> = w["authorships"].as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|a| {
                a["author"]["display_name"].as_str().map(|n| Author {
                    name: n.to_string(),
                    affiliation: String::new(),
                })
            })
            .collect();

        Paper {
            paper_id: w["id"].as_str().unwrap_or("").to_string(),
            title: w["title"].as_str().unwrap_or("").to_string(),
            authors,
            year: w["publication_year"].as_u64().unwrap_or(0) as u32,
            r#abstract: String::new(), // OpenAlex doesn't return abstracts in search
            venue: w["primary_location"]["source"]["display_name"]
                .as_str().unwrap_or("").to_string(),
            citation_count: w["cited_by_count"].as_u64().unwrap_or(0) as u32,
            doi: w["doi"].as_str().unwrap_or("").to_string(),
            arxiv_id: String::new(),
            url: w["id"].as_str().unwrap_or("").to_string(),
            pdf_url: w["open_access"]["oa_url"].as_str().unwrap_or("").to_string(),
            source: "openalex".to_string(),
        }
    }).collect())
}

// ---------------------------------------------------------------------------
// Semantic Scholar (Tier 0, key optional)
// ---------------------------------------------------------------------------

pub async fn search_semantic_scholar(
    client: &Client,
    query: &str,
    limit: usize,
    keys: &ApiKeys,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://api.semanticscholar.org/graph/v1/paper/search?query={}&limit={}&fields=title,authors,year,abstract,venue,citationCount,externalIds,openAccessPdf,url",
        urlencoding::encode(query),
        limit.min(100)
    );

    let mut req = client.get(&url);
    if !keys.s2_api_key.is_empty() {
        req = req.header("x-api-key", &keys.s2_api_key);
    }

    let resp: Value = req.send().await?.json().await?;
    if resp["message"].as_str().is_some_and(|m| m.contains("Too Many")) {
        return Err(SourceError::RateLimited);
    }

    let data = resp["data"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(data.iter().map(|p| {
        let authors: Vec<Author> = p["authors"].as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|a| {
                a["name"].as_str().map(|n| Author {
                    name: n.to_string(),
                    affiliation: String::new(),
                })
            })
            .collect();

        let ext = &p["externalIds"];
        Paper {
            paper_id: p["paperId"].as_str().unwrap_or("").to_string(),
            title: p["title"].as_str().unwrap_or("").to_string(),
            authors,
            year: p["year"].as_u64().unwrap_or(0) as u32,
            r#abstract: p["abstract"].as_str().unwrap_or("").to_string(),
            venue: p["venue"].as_str().unwrap_or("").to_string(),
            citation_count: p["citationCount"].as_u64().unwrap_or(0) as u32,
            doi: ext["DOI"].as_str().unwrap_or("").to_string(),
            arxiv_id: ext["ArXiv"].as_str().unwrap_or("").to_string(),
            url: p["url"].as_str().unwrap_or("").to_string(),
            pdf_url: p["openAccessPdf"]["url"].as_str().unwrap_or("").to_string(),
            source: "semantic_scholar".to_string(),
        }
    }).collect())
}

// ---------------------------------------------------------------------------
// Crossref (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_crossref(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://api.crossref.org/works?query={}&rows={}",
        urlencoding::encode(query),
        limit.min(50)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let items = resp["message"]["items"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(items.iter().map(|item| {
        let authors: Vec<Author> = item["author"].as_array()
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|a| {
                let given = a["given"].as_str().unwrap_or("");
                let family = a["family"].as_str().unwrap_or("");
                if family.is_empty() { return None; }
                Some(Author {
                    name: format!("{given} {family}").trim().to_string(),
                    affiliation: String::new(),
                })
            })
            .collect();

        let title = item["title"].as_array()
            .and_then(|t| t.first())
            .and_then(|t| t.as_str())
            .unwrap_or("");

        Paper {
            paper_id: item["DOI"].as_str().unwrap_or("").to_string(),
            title: title.to_string(),
            authors,
            year: item["published"]["date-parts"][0][0].as_u64().unwrap_or(0) as u32,
            r#abstract: String::new(),
            venue: item["container-title"].as_array()
                .and_then(|t| t.first())
                .and_then(|t| t.as_str())
                .unwrap_or("").to_string(),
            citation_count: item["is-referenced-by-count"].as_u64().unwrap_or(0) as u32,
            doi: item["DOI"].as_str().unwrap_or("").to_string(),
            arxiv_id: String::new(),
            url: item["URL"].as_str().unwrap_or("").to_string(),
            pdf_url: String::new(),
            source: "crossref".to_string(),
        }
    }).collect())
}

// ---------------------------------------------------------------------------
// CORE (Tier 2, key required)
// ---------------------------------------------------------------------------

pub async fn search_core(
    client: &Client,
    query: &str,
    limit: usize,
    keys: &ApiKeys,
) -> Result<Vec<Paper>, SourceError> {
    if keys.core_api_key.is_empty() {
        return Err(SourceError::Unavailable("CORE_API_KEY not set".into()));
    }

    let url = format!(
        "https://api.core.ac.uk/v3/search/works/?q={}&limit={}&apiKey={}",
        urlencoding::encode(query),
        limit.min(100),
        keys.core_api_key
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let results = resp["results"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(results.iter().map(|w| {
        Paper {
            paper_id: w["id"].as_str().or(w["id"].as_u64().map(|_| "")).unwrap_or("").to_string(),
            title: w["title"].as_str().unwrap_or("").to_string(),
            authors: Vec::new(),
            year: w["yearPublished"].as_u64().unwrap_or(0) as u32,
            r#abstract: w["abstract"].as_str().unwrap_or("").to_string(),
            venue: String::new(),
            citation_count: 0,
            doi: w["doi"].as_str().unwrap_or("").to_string(),
            arxiv_id: String::new(),
            url: w["downloadUrl"].as_str().unwrap_or("").to_string(),
            pdf_url: w["downloadUrl"].as_str().unwrap_or("").to_string(),
            source: "core".to_string(),
        }
    }).collect())
}

// ---------------------------------------------------------------------------
// Unpaywall (Tier 2, email required)
// ---------------------------------------------------------------------------

pub async fn check_unpaywall(
    client: &Client,
    doi: &str,
    keys: &ApiKeys,
) -> Result<Option<String>, SourceError> {
    if keys.unpaywall_email.is_empty() || doi.is_empty() {
        return Ok(None);
    }

    let url = format!(
        "https://api.unpaywall.org/v2/{}?email={}",
        urlencoding::encode(doi),
        keys.unpaywall_email
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let pdf_url = resp["best_oa_location"]["url_for_pdf"]
        .as_str()
        .filter(|s| !s.is_empty())
        .map(|s| s.to_string());

    Ok(pdf_url)
}

// Encode URL component
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}
