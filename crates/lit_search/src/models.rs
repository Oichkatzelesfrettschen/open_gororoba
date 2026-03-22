//! Data models for literature search results.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct Author {
    pub name: String,
    #[serde(default)]
    pub affiliation: String,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Paper {
    pub paper_id: String,
    pub title: String,
    #[serde(default)]
    pub authors: Vec<Author>,
    #[serde(default)]
    pub year: u32,
    #[serde(default)]
    pub r#abstract: String,
    #[serde(default)]
    pub venue: String,
    #[serde(default)]
    pub citation_count: u32,
    #[serde(default)]
    pub doi: String,
    #[serde(default)]
    pub arxiv_id: String,
    #[serde(default)]
    pub url: String,
    #[serde(default)]
    pub pdf_url: String,
    #[serde(default)]
    pub source: String,
}

impl Paper {
    /// Normalized dedup key: DOI if available, else arXiv ID, else title hash.
    pub fn dedup_key(&self) -> String {
        if !self.doi.is_empty() {
            return format!("doi:{}", self.doi.to_lowercase());
        }
        if !self.arxiv_id.is_empty() {
            return format!("arxiv:{}", self.arxiv_id);
        }
        // Normalize title for fuzzy matching
        let normalized: String = self.title.to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();
        format!("title:{}", normalized.trim())
    }

    /// Citation key: lastname + year + first keyword.
    pub fn cite_key(&self) -> String {
        let last = self.authors.first()
            .map(|a| {
                a.name.split_whitespace().last()
                    .unwrap_or("anon")
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_ascii_alphabetic())
                    .collect::<String>()
            })
            .unwrap_or_else(|| "anon".to_string());
        let yr = if self.year > 0 { self.year.to_string() } else { "0000".to_string() };
        let kw = self.title.split_whitespace()
            .find(|w| w.len() > 3 && w.chars().all(|c| c.is_alphabetic()))
            .map(|w| w.to_lowercase())
            .unwrap_or_default();
        format!("{last}{yr}{kw}")
    }
}
