//! Citation verification helpers for BibTeX-backed literature checks.
//!
//! Ported in spirit from AutoResearchClaw's `verify.py`.

use crate::{SearchEngine, models::Paper};
use blake3::Hasher;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum VerifyStatus {
    Verified,
    Suspicious,
    Hallucinated,
    Skipped,
}

impl VerifyStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Verified => "verified",
            Self::Suspicious => "suspicious",
            Self::Hallucinated => "hallucinated",
            Self::Skipped => "skipped",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationResult {
    pub cite_key: String,
    pub title: String,
    pub status: VerifyStatus,
    pub confidence: f32,
    pub method: String,
    pub details: String,
    pub matched_paper: Option<Paper>,
    pub relevance_score: Option<f32>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VerificationReport {
    pub total: usize,
    pub verified: usize,
    pub suspicious: usize,
    pub hallucinated: usize,
    pub skipped: usize,
    pub results: Vec<CitationResult>,
}

impl VerificationReport {
    pub fn integrity_score(&self) -> f32 {
        let verifiable = self.total.saturating_sub(self.skipped);
        if verifiable == 0 {
            1.0
        } else {
            self.verified as f32 / verifiable as f32
        }
    }
}

#[derive(Debug, Clone)]
struct BibEntry {
    key: String,
    title: String,
    arxiv_id: String,
    doi: String,
}

pub async fn verify_citations(
    bib_text: &str,
    engine: &SearchEngine,
    inter_verify_delay: Duration,
) -> VerificationReport {
    let entries = parse_bibtex_entries(bib_text);
    let mut report = VerificationReport {
        total: entries.len(),
        ..VerificationReport::default()
    };
    let start = Instant::now();
    let timeout = Duration::from_secs(300);

    for (index, entry) in entries.iter().enumerate() {
        if start.elapsed() > timeout {
            for remaining in &entries[index..] {
                push_result(
                    &mut report,
                    CitationResult {
                        cite_key: remaining.key.clone(),
                        title: remaining.title.clone(),
                        status: VerifyStatus::Skipped,
                        confidence: 0.0,
                        method: "skipped".to_string(),
                        details: "Verification timeout exceeded".to_string(),
                        matched_paper: None,
                        relevance_score: None,
                    },
                );
            }
            break;
        }

        if entry.title.trim().is_empty() {
            push_result(
                &mut report,
                CitationResult {
                    cite_key: entry.key.clone(),
                    title: entry.title.clone(),
                    status: VerifyStatus::Skipped,
                    confidence: 0.0,
                    method: "skipped".to_string(),
                    details: "No title in BibTeX entry".to_string(),
                    matched_paper: None,
                    relevance_score: None,
                },
            );
            continue;
        }

        if let Some(mut cached) = read_verify_cache(&entry.title) {
            cached.cite_key = entry.key.clone();
            push_result(&mut report, cached);
            continue;
        }

        let mut result = if !entry.doi.is_empty() {
            verify_by_doi(engine, &entry.doi, &entry.title).await
        } else {
            None
        };
        if result.is_none() {
            result = verify_by_openalex(engine, &entry.title).await;
        }
        if result.is_none() && !entry.arxiv_id.is_empty() {
            result = verify_by_arxiv_id(engine, &entry.arxiv_id, &entry.title).await;
        }
        let mut final_result = if let Some(result) = result {
            result
        } else {
            verify_by_title_search(engine, &entry.title).await
        };
        final_result.cite_key = entry.key.clone();

        if final_result.status != VerifyStatus::Skipped {
            write_verify_cache(&entry.title, &final_result);
        }
        push_result(&mut report, final_result);
        tokio::time::sleep(inter_verify_delay).await;
    }

    report
}

fn push_result(report: &mut VerificationReport, result: CitationResult) {
    match result.status {
        VerifyStatus::Verified => report.verified += 1,
        VerifyStatus::Suspicious => report.suspicious += 1,
        VerifyStatus::Hallucinated => report.hallucinated += 1,
        VerifyStatus::Skipped => report.skipped += 1,
    }
    report.results.push(result);
}

async fn verify_by_doi(
    engine: &SearchEngine,
    doi: &str,
    expected_title: &str,
) -> Option<CitationResult> {
    let encoded = percent_encode(doi);
    let crossref_url = format!("https://api.crossref.org/works/{encoded}");
    let response = engine.client().get(&crossref_url).send().await.ok()?;
    if response.status() == reqwest::StatusCode::NOT_FOUND
        && (doi.starts_with("10.48550/") || doi.starts_with("10.5281/"))
    {
        return verify_by_datacite(engine, doi, expected_title).await;
    }
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    let found_title = body["message"]["title"]
        .as_array()
        .and_then(|titles| titles.first())
        .and_then(|title| title.as_str())
        .unwrap_or("");
    Some(classify_title_match(
        expected_title,
        found_title,
        "doi",
        "CrossRef",
    ))
}

async fn verify_by_datacite(
    engine: &SearchEngine,
    doi: &str,
    expected_title: &str,
) -> Option<CitationResult> {
    let encoded = percent_encode(doi);
    let url = format!("https://api.datacite.org/dois/{encoded}");
    let response = engine.client().get(&url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    let found_title = body["data"]["attributes"]["titles"]
        .as_array()
        .and_then(|titles| titles.first())
        .and_then(|title| title["title"].as_str())
        .unwrap_or("");
    Some(classify_title_match(
        expected_title,
        found_title,
        "doi",
        "DataCite",
    ))
}

async fn verify_by_openalex(engine: &SearchEngine, title: &str) -> Option<CitationResult> {
    let url = format!(
        "https://api.openalex.org/works?filter=title.search:{}&per_page=5&mailto=researchclaw@users.noreply.github.com",
        percent_encode(title)
    );
    let response = engine.client().get(&url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let body: Value = response.json().await.ok()?;
    let mut best_title = "";
    let mut best_similarity = 0.0_f32;
    for result in body["results"].as_array().into_iter().flatten() {
        if let Some(found_title) = result["title"].as_str() {
            let similarity = title_similarity(title, found_title);
            if similarity > best_similarity {
                best_similarity = similarity;
                best_title = found_title;
            }
        }
    }
    if best_title.is_empty() {
        return Some(CitationResult {
            cite_key: String::new(),
            title: title.to_string(),
            status: VerifyStatus::Hallucinated,
            confidence: 0.7,
            method: "openalex".to_string(),
            details: "No results found via OpenAlex".to_string(),
            matched_paper: None,
            relevance_score: None,
        });
    }
    Some(classify_similarity(
        title,
        best_title,
        best_similarity,
        "openalex",
        "OpenAlex",
        None,
    ))
}

async fn verify_by_arxiv_id(
    engine: &SearchEngine,
    arxiv_id: &str,
    expected_title: &str,
) -> Option<CitationResult> {
    let url = format!(
        "https://export.arxiv.org/api/query?id_list={}",
        percent_encode(arxiv_id)
    );
    let response = engine.client().get(&url).send().await.ok()?;
    if !response.status().is_success() {
        return None;
    }
    let text = response.text().await.ok()?;
    let found_title = text
        .split("<entry>")
        .nth(1)
        .and_then(|entry| entry.split("<title>").nth(1))
        .and_then(|rest| rest.split("</title>").next())
        .map(|title| title.replace('\n', " ").trim().to_string())
        .unwrap_or_default();
    if found_title.is_empty() || found_title.eq_ignore_ascii_case("error") {
        return Some(CitationResult {
            cite_key: String::new(),
            title: expected_title.to_string(),
            status: VerifyStatus::Hallucinated,
            confidence: 0.9,
            method: "arxiv_id".to_string(),
            details: format!("arXiv ID {arxiv_id} returned error or empty response"),
            matched_paper: None,
            relevance_score: None,
        });
    }
    Some(classify_title_match(
        expected_title,
        &found_title,
        "arxiv_id",
        "arXiv",
    ))
}

async fn verify_by_title_search(engine: &SearchEngine, title: &str) -> CitationResult {
    let results = engine.search(title, 5, 0).await;
    if results.is_empty() {
        return CitationResult {
            cite_key: String::new(),
            title: title.to_string(),
            status: VerifyStatus::Hallucinated,
            confidence: 0.7,
            method: "title_search".to_string(),
            details: "No results found via title search".to_string(),
            matched_paper: None,
            relevance_score: None,
        };
    }

    let mut best_similarity = 0.0_f32;
    let mut best_paper = None;
    for paper in &results {
        let similarity = title_similarity(title, &paper.title);
        if similarity > best_similarity {
            best_similarity = similarity;
            best_paper = Some(paper.clone());
        }
    }

    let best_title = best_paper
        .as_ref()
        .map(|paper| paper.title.clone())
        .unwrap_or_default();

    classify_similarity(
        title,
        &best_title,
        best_similarity,
        "title_search",
        "title search",
        best_paper,
    )
}

fn classify_title_match(
    expected_title: &str,
    found_title: &str,
    method: &str,
    backend: &str,
) -> CitationResult {
    let similarity = title_similarity(expected_title, found_title);
    classify_similarity(
        expected_title,
        found_title,
        similarity,
        method,
        backend,
        None,
    )
}

fn classify_similarity(
    expected_title: &str,
    found_title: &str,
    similarity: f32,
    method: &str,
    backend: &str,
    matched_paper: Option<Paper>,
) -> CitationResult {
    let (status, details) = if found_title.is_empty() {
        (
            VerifyStatus::Verified,
            format!("{backend} resolved identifier (no title comparison)"),
        )
    } else if similarity >= 0.80 {
        (
            VerifyStatus::Verified,
            format!("Confirmed via {backend}: '{found_title}'"),
        )
    } else if similarity >= 0.50 {
        (
            VerifyStatus::Suspicious,
            format!("{backend} match differs (sim={similarity:.2}): '{found_title}'"),
        )
    } else {
        (
            VerifyStatus::Hallucinated,
            format!("{backend} best match too weak (sim={similarity:.2}): '{found_title}'"),
        )
    };

    CitationResult {
        cite_key: String::new(),
        title: expected_title.to_string(),
        status,
        confidence: similarity,
        method: method.to_string(),
        details,
        matched_paper,
        relevance_score: None,
    }
}

fn title_similarity(left: &str, right: &str) -> f32 {
    fn words(title: &str) -> std::collections::BTreeSet<String> {
        let re = Regex::new(r"[^a-z0-9\s]").expect("valid regex");
        re.replace_all(&title.to_ascii_lowercase(), "")
            .split_whitespace()
            .filter(|word| !word.is_empty())
            .map(ToString::to_string)
            .collect()
    }

    let left_words = words(left);
    let right_words = words(right);
    if left_words.is_empty() || right_words.is_empty() {
        return 0.0;
    }
    left_words.intersection(&right_words).count() as f32
        / left_words.len().max(right_words.len()) as f32
}

fn parse_bibtex_entries(bib_text: &str) -> Vec<BibEntry> {
    let mut entries = Vec::new();
    let entry_head_re = Regex::new(r"^@(\w+)\s*\{\s*([^,\s]+)\s*,").expect("valid regex");
    let mut current: Option<BibEntry> = None;

    for raw_line in bib_text.lines() {
        let line = raw_line.trim();
        if line.is_empty() {
            continue;
        }

        if let Some(captures) = entry_head_re.captures(line) {
            if let Some(entry) = current.take() {
                entries.push(entry);
            }
            let key = captures
                .get(2)
                .map(|m| m.as_str().trim().to_string())
                .unwrap_or_default();
            current = Some(BibEntry {
                key,
                title: String::new(),
                arxiv_id: String::new(),
                doi: String::new(),
            });
            continue;
        }

        if line == "}" {
            if let Some(entry) = current.take() {
                entries.push(entry);
            }
            continue;
        }

        let Some(entry) = current.as_mut() else {
            continue;
        };
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        let field_name = name.trim().to_ascii_lowercase();
        let field_value = normalize_bibtex_field(value);
        match field_name.as_str() {
            "title" => entry.title = field_value,
            "eprint" => entry.arxiv_id = field_value,
            "doi" => entry.doi = field_value,
            _ => {}
        }
    }

    if let Some(entry) = current.take() {
        entries.push(entry);
    }

    entries
}

fn normalize_bibtex_field(value: &str) -> String {
    value
        .trim()
        .trim_end_matches(',')
        .trim()
        .trim_start_matches('{')
        .trim_end_matches('}')
        .trim()
        .to_string()
}

fn verify_cache_dir() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from).map(|home| {
        home.join(".cache")
            .join("lit_search")
            .join("citation_verify")
    })
}

fn verify_cache_key(title: &str) -> String {
    let mut hasher = Hasher::new();
    hasher.update(title.trim().to_ascii_lowercase().as_bytes());
    hasher.finalize().to_hex()[..16].to_string()
}

fn percent_encode(value: &str) -> String {
    url::form_urlencoded::byte_serialize(value.as_bytes()).collect()
}

fn read_verify_cache(title: &str) -> Option<CitationResult> {
    let cache_dir = verify_cache_dir()?;
    let _ = fs::create_dir_all(&cache_dir);
    let path = cache_dir.join(format!("{}.json", verify_cache_key(title)));
    let text = fs::read_to_string(path).ok()?;
    serde_json::from_str(&text).ok()
}

fn write_verify_cache(title: &str, result: &CitationResult) {
    let Some(cache_dir) = verify_cache_dir() else {
        return;
    };
    let _ = fs::create_dir_all(&cache_dir);
    let path = cache_dir.join(format!("{}.json", verify_cache_key(title)));
    if let Ok(text) = serde_json::to_string_pretty(result) {
        let _ = fs::write(path, text);
    }
}

#[cfg(test)]
mod tests {
    use super::{VerifyStatus, parse_bibtex_entries, title_similarity};

    #[test]
    fn parses_basic_bibtex_fields() {
        let input = r#"
@article{jacobson1958,
  title = {Composition algebras and their automorphisms},
  doi = {10.1007/BF02854388},
  eprint = {1234.5678}
}
"#;
        let entries = parse_bibtex_entries(input);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].key, "jacobson1958");
        assert_eq!(entries[0].doi, "10.1007/BF02854388");
    }

    #[test]
    fn title_similarity_rewards_overlap() {
        let similarity = title_similarity(
            "Composition algebras and their automorphisms",
            "Composition Algebras and Their Automorphisms",
        );
        assert!(similarity > 0.9);
    }

    #[test]
    fn verify_status_string_is_stable() {
        assert_eq!(VerifyStatus::Verified.as_str(), "verified");
    }
}
