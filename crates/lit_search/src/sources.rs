//! Academic paper source clients.
//!
//! Each source implements the `PaperSource` trait: an async function that
//! takes a query string and returns a list of papers.

use crate::models::{Author, Paper};
use reqwest::Client;
use serde_json::Value;

fn json_string(value: &Value) -> String {
    value.as_str().unwrap_or("").trim().to_string()
}

fn json_string_array(value: &Value) -> Vec<String> {
    value
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::trim))
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn json_year(value: &Value) -> u32 {
    if let Some(year) = value.as_u64() {
        return year as u32;
    }
    value
        .as_str()
        .and_then(|text| text.get(..4))
        .and_then(|year| year.parse().ok())
        .unwrap_or(0)
}

fn paper_authors(names: Vec<String>) -> Vec<Author> {
    names
        .into_iter()
        .map(|name| Author {
            name,
            affiliation: String::new(),
        })
        .collect()
}

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

    Ok(results
        .iter()
        .map(|w| {
            let authors: Vec<Author> = w["authorships"]
                .as_array()
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
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                citation_count: w["cited_by_count"].as_u64().unwrap_or(0) as u32,
                doi: w["doi"].as_str().unwrap_or("").to_string(),
                arxiv_id: String::new(),
                url: w["id"].as_str().unwrap_or("").to_string(),
                pdf_url: w["open_access"]["oa_url"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                source: "openalex".to_string(),
            }
        })
        .collect())
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
    if resp["message"]
        .as_str()
        .is_some_and(|m| m.contains("Too Many"))
    {
        return Err(SourceError::RateLimited);
    }

    let data = resp["data"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(data
        .iter()
        .map(|p| {
            let authors: Vec<Author> = p["authors"]
                .as_array()
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
        })
        .collect())
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
    let items = resp["message"]["items"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    Ok(items
        .iter()
        .map(|item| {
            let authors: Vec<Author> = item["author"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|a| {
                    let given = a["given"].as_str().unwrap_or("");
                    let family = a["family"].as_str().unwrap_or("");
                    if family.is_empty() {
                        return None;
                    }
                    Some(Author {
                        name: format!("{given} {family}").trim().to_string(),
                        affiliation: String::new(),
                    })
                })
                .collect();

            let title = item["title"]
                .as_array()
                .and_then(|t| t.first())
                .and_then(|t| t.as_str())
                .unwrap_or("");
            let primary_url = item["resource"]["primary"]["URL"]
                .as_str()
                .filter(|value| !value.trim().is_empty())
                .or_else(|| item["URL"].as_str())
                .unwrap_or("");
            let pdf_url = crossref_link_url(item).unwrap_or_default();

            Paper {
                paper_id: item["DOI"].as_str().unwrap_or("").to_string(),
                title: title.to_string(),
                authors,
                year: item["published"]["date-parts"][0][0].as_u64().unwrap_or(0) as u32,
                r#abstract: String::new(),
                venue: item["container-title"]
                    .as_array()
                    .and_then(|t| t.first())
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                citation_count: item["is-referenced-by-count"].as_u64().unwrap_or(0) as u32,
                doi: item["DOI"].as_str().unwrap_or("").to_string(),
                arxiv_id: String::new(),
                url: primary_url.to_string(),
                pdf_url,
                source: "crossref".to_string(),
            }
        })
        .collect())
}

fn crossref_link_url(item: &Value) -> Option<String> {
    item["link"].as_array().and_then(|links| {
        links
            .iter()
            .filter_map(|link| link["URL"].as_str())
            .map(str::trim)
            .find(|url| !url.is_empty())
            .map(ToOwned::to_owned)
    })
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

    Ok(results
        .iter()
        .map(|w| Paper {
            paper_id: w["id"]
                .as_str()
                .or(w["id"].as_u64().map(|_| ""))
                .unwrap_or("")
                .to_string(),
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
        })
        .collect())
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

// ---------------------------------------------------------------------------
// arXiv (Tier 0, no key) -- Atom XML API
// ---------------------------------------------------------------------------

pub async fn search_arxiv(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{}&max_results={}",
        urlencoding::encode(query),
        limit.min(50)
    );

    let text = client.get(&url).send().await?.text().await?;
    let mut papers = Vec::new();

    // Simple XML parsing without a full XML crate -- extract entries
    for entry in text.split("<entry>").skip(1) {
        let extract = |tag: &str| -> String {
            entry
                .split(&format!("<{tag}>"))
                .nth(1)
                .and_then(|s| s.split(&format!("</{tag}>")).next())
                .unwrap_or("")
                .trim()
                .replace('\n', " ")
                .to_string()
        };

        let id_full = extract("id");
        let arxiv_id = id_full.rsplit('/').next().unwrap_or("").to_string();

        papers.push(Paper {
            paper_id: arxiv_id.clone(),
            title: extract("title"),
            year: extract("published")
                .get(..4)
                .and_then(|y| y.parse().ok())
                .unwrap_or(0),
            r#abstract: extract("summary"),
            arxiv_id: arxiv_id.clone(),
            url: id_full,
            pdf_url: format!("https://arxiv.org/pdf/{arxiv_id}"),
            source: "arxiv".to_string(),
            ..Default::default()
        });
    }

    Ok(papers)
}

// ---------------------------------------------------------------------------
// InspireHEP (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_inspirehep(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://inspirehep.net/api/literature?q={}&size={}&fields=titles,authors,earliest_date,arxiv_eprints,dois,citation_count",
        urlencoding::encode(query),
        limit.min(25)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let hits = resp["hits"]["hits"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    Ok(hits
        .iter()
        .map(|h| {
            let meta = &h["metadata"];
            let title = meta["titles"]
                .as_array()
                .and_then(|t| t.first())
                .and_then(|t| t["title"].as_str())
                .unwrap_or("");
            let arxiv = meta["arxiv_eprints"]
                .as_array()
                .and_then(|a| a.first())
                .and_then(|a| a["value"].as_str())
                .unwrap_or("");
            let doi = meta["dois"]
                .as_array()
                .and_then(|d| d.first())
                .and_then(|d| d["value"].as_str())
                .unwrap_or("");

            Paper {
                paper_id: h["id"].as_str().unwrap_or("").to_string(),
                title: title.to_string(),
                year: meta["earliest_date"]
                    .as_str()
                    .and_then(|d| d.get(..4))
                    .and_then(|y| y.parse().ok())
                    .unwrap_or(0),
                citation_count: meta["citation_count"].as_u64().unwrap_or(0) as u32,
                doi: doi.to_string(),
                arxiv_id: arxiv.to_string(),
                pdf_url: if !arxiv.is_empty() {
                    format!("https://arxiv.org/pdf/{arxiv}")
                } else {
                    String::new()
                },
                source: "inspirehep".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// DBLP (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_dblp(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://dblp.org/search/publ/api?q={}&h={}&format=json",
        urlencoding::encode(query),
        limit.min(30)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let hits = resp["result"]["hits"]["hit"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    Ok(hits
        .iter()
        .map(|h| {
            let info = &h["info"];
            Paper {
                paper_id: info["key"].as_str().unwrap_or("").to_string(),
                title: info["title"].as_str().unwrap_or("").to_string(),
                year: info["year"]
                    .as_str()
                    .and_then(|y| y.parse().ok())
                    .unwrap_or(0),
                venue: info["venue"].as_str().unwrap_or("").to_string(),
                doi: info["doi"].as_str().unwrap_or("").to_string(),
                url: info["ee"].as_str().unwrap_or("").to_string(),
                source: "dblp".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// EuropePMC (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_europepmc(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={}&resultType=core&pageSize={}&format=json",
        urlencoding::encode(query),
        limit.min(25)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let results = resp["resultList"]["result"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    Ok(results
        .iter()
        .map(|r| Paper {
            paper_id: r["id"].as_str().unwrap_or("").to_string(),
            title: r["title"].as_str().unwrap_or("").to_string(),
            year: r["pubYear"]
                .as_str()
                .and_then(|y| y.parse().ok())
                .unwrap_or(0),
            r#abstract: r["abstractText"].as_str().unwrap_or("").to_string(),
            doi: r["doi"].as_str().unwrap_or("").to_string(),
            citation_count: r["citedByCount"].as_u64().unwrap_or(0) as u32,
            source: "europepmc".to_string(),
            ..Default::default()
        })
        .collect())
}

// ---------------------------------------------------------------------------
// HAL (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_hal(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://api.archives-ouvertes.fr/search/?q={}&rows={}&fl=title_s,authFullName_s,producedDateY_i,doiId_s,uri_s,fileMain_s&wt=json",
        urlencoding::encode(query),
        limit.min(25)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let docs = resp["response"]["docs"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    Ok(docs
        .iter()
        .map(|d| {
            let authors: Vec<Author> = d["authFullName_s"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|a| {
                    a.as_str().map(|n| Author {
                        name: n.to_string(),
                        affiliation: String::new(),
                    })
                })
                .collect();

            Paper {
                paper_id: d["uri_s"].as_str().unwrap_or("").to_string(),
                title: d["title_s"]
                    .as_array()
                    .and_then(|t| t.first())
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                authors,
                year: d["producedDateY_i"].as_u64().unwrap_or(0) as u32,
                doi: d["doiId_s"].as_str().unwrap_or("").to_string(),
                url: d["uri_s"].as_str().unwrap_or("").to_string(),
                pdf_url: d["fileMain_s"].as_str().unwrap_or("").to_string(),
                source: "hal".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// DataCite (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_datacite(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://api.datacite.org/dois?query={}&page[size]={}",
        urlencoding::encode(query),
        limit.min(25)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let data = resp["data"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(data
        .iter()
        .map(|d| {
            let attrs = &d["attributes"];
            Paper {
                paper_id: attrs["doi"].as_str().unwrap_or("").to_string(),
                title: attrs["titles"]
                    .as_array()
                    .and_then(|t| t.first())
                    .and_then(|t| t["title"].as_str())
                    .unwrap_or("")
                    .to_string(),
                year: attrs["publicationYear"].as_u64().unwrap_or(0) as u32,
                doi: attrs["doi"].as_str().unwrap_or("").to_string(),
                url: attrs["url"].as_str().unwrap_or("").to_string(),
                source: "datacite".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// SciELO (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_scielo(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://search.scielo.org/?q={}&count={}&output=json&lang=en",
        urlencoding::encode(query),
        limit.min(20)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let docs = resp["docs"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(docs
        .iter()
        .map(|d| Paper {
            paper_id: d["id"].as_str().unwrap_or("").to_string(),
            title: d["title"].as_str().unwrap_or("").to_string(),
            year: d["year"].as_str().and_then(|y| y.parse().ok()).unwrap_or(0),
            doi: d["doi"].as_str().unwrap_or("").to_string(),
            source: "scielo".to_string(),
            ..Default::default()
        })
        .collect())
}

// ---------------------------------------------------------------------------
// J-STAGE (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_jstage(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://api.jstage.jst.go.jp/searchapi/do?keyword={}&count={}&service=3",
        urlencoding::encode(query),
        limit.min(20)
    );

    let text = client.get(&url).send().await?.text().await?;
    let mut papers = Vec::new();

    // J-STAGE returns XML; simple extraction
    for entry in text.split("<entry>").skip(1) {
        let extract = |tag: &str| -> String {
            entry
                .split(&format!("<{tag}>"))
                .nth(1)
                .and_then(|s| s.split(&format!("</{tag}>")).next())
                .unwrap_or("")
                .trim()
                .to_string()
        };

        papers.push(Paper {
            paper_id: extract("article_link"),
            title: extract("article_title"),
            doi: extract("cdjournal"),
            url: extract("article_link"),
            source: "jstage".to_string(),
            ..Default::default()
        });
    }

    Ok(papers)
}

// ---------------------------------------------------------------------------
// Open Library (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_open_library(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://openlibrary.org/search.json?q={}&fields=key,title,author_name,first_publish_year,ia,public_scan_b,edition_count&limit={}",
        urlencoding::encode(query),
        limit.min(25)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let docs = resp["docs"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(docs
        .iter()
        .map(|doc| {
            let ol_key = json_string(&doc["key"]);
            let ia_id = json_string_array(&doc["ia"])
                .into_iter()
                .next()
                .unwrap_or_default();
            let public_scan = doc["public_scan_b"].as_bool().unwrap_or(false);
            let url = if public_scan && !ia_id.is_empty() {
                format!("https://archive.org/details/{ia_id}")
            } else if !ol_key.is_empty() {
                format!("https://openlibrary.org{ol_key}")
            } else {
                String::new()
            };

            Paper {
                paper_id: if !ol_key.is_empty() {
                    ol_key.clone()
                } else {
                    format!("openlibrary:{}", json_string(&doc["title"]))
                },
                title: json_string(&doc["title"]),
                authors: paper_authors(json_string_array(&doc["author_name"])),
                year: json_year(&doc["first_publish_year"]),
                venue: if let Some(count) = doc["edition_count"].as_u64() {
                    format!("{count} editions")
                } else {
                    String::new()
                },
                url,
                source: "open_library".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Internet Archive (Tier 1, no key)
// ---------------------------------------------------------------------------

pub async fn search_internet_archive(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://archive.org/advancedsearch.php?q={}&fl[]=identifier&fl[]=title&fl[]=creator&fl[]=year&fl[]=mediatype&fl[]=downloads&output=json&rows={}",
        urlencoding::encode(query),
        limit.min(20)
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let docs = resp["response"]["docs"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    let mut papers = Vec::new();
    for doc in docs {
        let identifier = json_string(&doc["identifier"]);
        if identifier.is_empty() {
            continue;
        }
        let detail_url = format!("https://archive.org/details/{identifier}");
        let pdf_url = lookup_internet_archive_pdf_url(client, &identifier)
            .await
            .unwrap_or_default();
        papers.push(Paper {
            paper_id: identifier.clone(),
            title: json_string(&doc["title"]),
            authors: paper_authors(json_string_array(&doc["creator"])),
            year: json_year(&doc["year"]),
            citation_count: doc["downloads"].as_u64().unwrap_or(0) as u32,
            venue: json_string(&doc["mediatype"]),
            url: detail_url,
            pdf_url,
            source: "internet_archive".to_string(),
            ..Default::default()
        });
    }

    Ok(papers)
}

async fn lookup_internet_archive_pdf_url(
    client: &Client,
    identifier: &str,
) -> Result<String, SourceError> {
    let url = format!("https://archive.org/metadata/{identifier}");
    let resp: Value = client.get(&url).send().await?.json().await?;
    let files = resp["files"].as_array().cloned().unwrap_or_default();

    for file in &files {
        let name = json_string(&file["name"]);
        if name.to_ascii_lowercase().ends_with(".pdf") {
            return Ok(format!("https://archive.org/download/{identifier}/{name}"));
        }
    }

    Ok(String::new())
}

// ---------------------------------------------------------------------------
// Google Books (Tier 1, no key, quota-limited)
// ---------------------------------------------------------------------------

pub async fn search_google_books(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    let url = format!(
        "https://www.googleapis.com/books/v1/volumes?q={}&maxResults={}&projection=lite",
        urlencoding::encode(query),
        limit.min(20)
    );

    let resp = client.get(&url).send().await?;
    if resp.status().as_u16() == 429 {
        return Err(SourceError::RateLimited);
    }
    let body: Value = resp.json().await?;
    if body["error"]["status"].as_str() == Some("RESOURCE_EXHAUSTED") {
        return Err(SourceError::RateLimited);
    }
    let items = body["items"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(items
        .iter()
        .map(|item| {
            let volume_info = &item["volumeInfo"];
            let access_info = &item["accessInfo"];
            let doi = volume_info["industryIdentifiers"]
                .as_array()
                .and_then(|ids| {
                    ids.iter().find_map(|identifier| {
                        let value = identifier["identifier"].as_str().unwrap_or("").trim();
                        if value.starts_with("10.") && value.contains('/') {
                            Some(value.to_string())
                        } else {
                            None
                        }
                    })
                })
                .unwrap_or_default();

            Paper {
                paper_id: json_string(&item["id"]),
                title: json_string(&volume_info["title"]),
                authors: paper_authors(json_string_array(&volume_info["authors"])),
                year: json_year(&volume_info["publishedDate"]),
                r#abstract: json_string(&volume_info["description"]),
                venue: json_string(&volume_info["publisher"]),
                citation_count: 0,
                doi,
                arxiv_id: String::new(),
                url: volume_info["infoLink"]
                    .as_str()
                    .or_else(|| volume_info["previewLink"].as_str())
                    .or_else(|| volume_info["canonicalVolumeLink"].as_str())
                    .unwrap_or("")
                    .to_string(),
                pdf_url: access_info["pdf"]["downloadLink"]
                    .as_str()
                    .unwrap_or("")
                    .to_string(),
                source: "google_books".to_string(),
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// CiNii (Tier 2, appid required)
// ---------------------------------------------------------------------------

pub async fn search_cinii(
    client: &Client,
    query: &str,
    limit: usize,
    keys: &ApiKeys,
) -> Result<Vec<Paper>, SourceError> {
    if keys.cinii_appid.is_empty() {
        return Err(SourceError::Unavailable("CINII_APPID not set".into()));
    }

    let url = format!(
        "https://cir.nii.ac.jp/opensearch/articles?q={}&count={}&appid={}&format=json",
        urlencoding::encode(query),
        limit.min(20),
        keys.cinii_appid
    );

    let resp: Value = client.get(&url).send().await?.json().await?;
    let items = resp["items"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(items
        .iter()
        .map(|item| Paper {
            paper_id: item["@id"].as_str().unwrap_or("").to_string(),
            title: item["title"].as_str().unwrap_or("").to_string(),
            doi: item["doi"].as_str().unwrap_or("").to_string(),
            url: item["@id"].as_str().unwrap_or("").to_string(),
            source: "cinii".to_string(),
            ..Default::default()
        })
        .collect())
}

// ---------------------------------------------------------------------------
// ADS (Tier 2, token required)
// ---------------------------------------------------------------------------

pub async fn search_ads(
    client: &Client,
    query: &str,
    limit: usize,
    keys: &ApiKeys,
) -> Result<Vec<Paper>, SourceError> {
    if keys.ads_token.is_empty() {
        return Err(SourceError::Unavailable("NASA_ADS_TOKEN not set".into()));
    }

    let url = format!(
        "https://api.adsabs.harvard.edu/v1/search/query?q={}&rows={}&fl=title,author,year,doi,bibcode,citation_count,identifier",
        urlencoding::encode(query),
        limit.min(25)
    );

    let resp: Value = client
        .get(&url)
        .header("Authorization", format!("Bearer {}", keys.ads_token))
        .send()
        .await?
        .json()
        .await?;

    let docs = resp["response"]["docs"]
        .as_array()
        .unwrap_or(&Vec::new())
        .clone();

    Ok(docs
        .iter()
        .map(|d| {
            let authors: Vec<Author> = d["author"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|a| {
                    a.as_str().map(|n| Author {
                        name: n.to_string(),
                        affiliation: String::new(),
                    })
                })
                .collect();

            let arxiv = d["identifier"]
                .as_array()
                .unwrap_or(&Vec::new())
                .iter()
                .filter_map(|id| id.as_str())
                .find(|id| id.starts_with("arXiv:"))
                .map(|id| id.trim_start_matches("arXiv:").to_string())
                .unwrap_or_default();

            Paper {
                paper_id: d["bibcode"].as_str().unwrap_or("").to_string(),
                title: d["title"]
                    .as_array()
                    .and_then(|t| t.first())
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string(),
                authors,
                year: d["year"].as_str().and_then(|y| y.parse().ok()).unwrap_or(0),
                citation_count: d["citation_count"].as_u64().unwrap_or(0) as u32,
                doi: d["doi"]
                    .as_array()
                    .and_then(|d| d.first())
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string(),
                arxiv_id: arxiv,
                source: "ads".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Lens.org (Tier 2, key required)
// ---------------------------------------------------------------------------

pub async fn search_lens(
    client: &Client,
    query: &str,
    limit: usize,
    keys: &ApiKeys,
) -> Result<Vec<Paper>, SourceError> {
    if keys.lens_api_key.is_empty() {
        return Err(SourceError::Unavailable("LENS_API_KEY not set".into()));
    }

    let body = serde_json::json!({
        "query": { "match": { "title": query } },
        "size": limit.min(50),
        "include": ["title", "year_published", "authors", "external_ids", "source_urls"]
    });

    let resp: Value = client
        .post("https://api.lens.org/scholarly/search")
        .header("Authorization", format!("Bearer {}", keys.lens_api_key))
        .json(&body)
        .send()
        .await?
        .json()
        .await?;

    let data = resp["data"].as_array().unwrap_or(&Vec::new()).clone();

    Ok(data
        .iter()
        .map(|d| {
            let empty_vec = Vec::new();
            let ext_ids = d["external_ids"].as_array().unwrap_or(&empty_vec);
            let doi = ext_ids
                .iter()
                .find(|id| id["type"].as_str() == Some("doi"))
                .and_then(|id| id["value"].as_str())
                .unwrap_or("");

            Paper {
                paper_id: d["lens_id"].as_str().unwrap_or("").to_string(),
                title: d["title"].as_str().unwrap_or("").to_string(),
                year: d["year_published"].as_u64().unwrap_or(0) as u32,
                doi: doi.to_string(),
                source: "lens".to_string(),
                ..Default::default()
            }
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Google Scholar (Tier 2, no key, web scraping -- best-effort)
// ---------------------------------------------------------------------------

pub async fn search_google_scholar(
    client: &Client,
    query: &str,
    limit: usize,
) -> Result<Vec<Paper>, SourceError> {
    use scraper::{Html, Selector};

    let num = limit.min(20);
    let url = format!(
        "https://scholar.google.com/scholar?q={}&num={}&hl=en",
        urlencoding::encode(query),
        num
    );

    let resp = client
        .get(&url)
        .header("Accept", "text/html")
        .header("Accept-Language", "en-US,en;q=0.9")
        .send()
        .await?;

    let status = resp.status();
    let text = resp.text().await?;

    // CAPTCHA or block detection
    if status.as_u16() == 429 || text.contains("unusual traffic") || text.contains("CAPTCHA") {
        tracing::warn!("Google Scholar: rate limited or CAPTCHA triggered");
        return Err(SourceError::RateLimited);
    }

    if status.as_u16() != 200 {
        return Err(SourceError::Unavailable(format!(
            "Google Scholar returned HTTP {}",
            status.as_u16()
        )));
    }

    let document = Html::parse_document(&text);

    // Each result lives in a <div class="gs_r gs_or gs_scl"> container
    let result_sel = Selector::parse("div.gs_r.gs_or.gs_scl")
        .map_err(|e| SourceError::Unavailable(format!("selector parse: {e:?}")))?;
    let title_sel = Selector::parse("h3.gs_rt a")
        .map_err(|e| SourceError::Unavailable(format!("selector parse: {e:?}")))?;
    let title_nolink_sel = Selector::parse("h3.gs_rt")
        .map_err(|e| SourceError::Unavailable(format!("selector parse: {e:?}")))?;
    let info_sel = Selector::parse("div.gs_a")
        .map_err(|e| SourceError::Unavailable(format!("selector parse: {e:?}")))?;
    let snippet_sel = Selector::parse("div.gs_rs")
        .map_err(|e| SourceError::Unavailable(format!("selector parse: {e:?}")))?;
    let pdf_sel = Selector::parse("div.gs_or_ggsm a")
        .map_err(|e| SourceError::Unavailable(format!("selector parse: {e:?}")))?;

    let mut papers = Vec::new();

    for result in document.select(&result_sel) {
        // Title + URL
        let (title, result_url) = if let Some(a) = result.select(&title_sel).next() {
            let t: String = a.text().collect::<String>().trim().to_string();
            let u = a.value().attr("href").unwrap_or("").to_string();
            (t, u)
        } else if let Some(h3) = result.select(&title_nolink_sel).next() {
            (
                h3.text().collect::<String>().trim().to_string(),
                String::new(),
            )
        } else {
            continue;
        };

        if title.is_empty() {
            continue;
        }

        // Author/venue line: "A Author, B Author - Journal, Year - publisher"
        let info_text = result
            .select(&info_sel)
            .next()
            .map(|el| el.text().collect::<String>())
            .unwrap_or_default();

        let (authors, year, venue) = parse_gs_info_line(&info_text);

        // Snippet as abstract
        let snippet = result
            .select(&snippet_sel)
            .next()
            .map(|el| el.text().collect::<String>().trim().to_string())
            .unwrap_or_default();

        // PDF link (sidebar green link)
        let pdf_url = result
            .select(&pdf_sel)
            .next()
            .and_then(|a| a.value().attr("href"))
            .unwrap_or("")
            .to_string();

        papers.push(Paper {
            paper_id: format!("gs:{}", title.len()),
            title,
            authors,
            year,
            r#abstract: snippet,
            venue,
            url: result_url,
            pdf_url,
            source: "google_scholar".to_string(),
            ..Default::default()
        });
    }

    Ok(papers)
}

/// Parse Google Scholar info line: "A Author, B Author - Journal, Year - publisher"
fn parse_gs_info_line(info: &str) -> (Vec<Author>, u32, String) {
    let parts: Vec<&str> = info.split(" - ").collect();
    let mut authors = Vec::new();
    let mut year = 0u32;
    let mut venue = String::new();

    // First segment: authors
    if let Some(author_str) = parts.first() {
        // Remove ellipsis artifacts
        let cleaned = author_str.replace("...", "").replace('\u{2026}', "");
        for name in cleaned.split(',') {
            let name = name.trim();
            if !name.is_empty() && name.len() > 1 {
                authors.push(Author {
                    name: name.to_string(),
                    affiliation: String::new(),
                });
            }
        }
    }

    // Second segment: "Journal, Year" or just "Year"
    if let Some(venue_year) = parts.get(1) {
        // Try to extract a 4-digit year
        for word in venue_year.split(|c: char| !c.is_ascii_digit()) {
            if word.len() == 4
                && let Ok(y) = word.parse::<u32>()
                && (1900..=2030).contains(&y)
            {
                year = y;
            }
        }
        // Everything before the year is venue
        if year > 0 {
            let yr_str = year.to_string();
            if let Some(pos) = venue_year.find(&yr_str) {
                venue = venue_year[..pos]
                    .trim()
                    .trim_end_matches(',')
                    .trim()
                    .to_string();
            }
        } else {
            venue = venue_year.trim().to_string();
        }
    }

    (authors, year, venue)
}

// Encode URL component
mod urlencoding {
    pub fn encode(s: &str) -> String {
        url::form_urlencoded::byte_serialize(s.as_bytes()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::crossref_link_url;
    use serde_json::json;

    #[test]
    fn crossref_link_url_prefers_first_present_link() {
        let item = json!({
            "link": [
                {"URL": "https://projecteuclid.org/journalArticle/Download?urlid=10.1215/S0012-7094-35-00112-0"},
                {"URL": "https://example.org/secondary.pdf"}
            ]
        });
        assert_eq!(
            crossref_link_url(&item).as_deref(),
            Some(
                "https://projecteuclid.org/journalArticle/Download?urlid=10.1215/S0012-7094-35-00112-0"
            )
        );
    }

    #[test]
    fn crossref_link_url_ignores_empty_entries() {
        let item = json!({
            "link": [
                {"URL": ""},
                {"URL": "   "},
                {"URL": "https://example.org/fallback.pdf"}
            ]
        });
        assert_eq!(
            crossref_link_url(&item).as_deref(),
            Some("https://example.org/fallback.pdf")
        );
    }
}
