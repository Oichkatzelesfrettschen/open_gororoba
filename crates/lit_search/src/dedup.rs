//! Deduplication of paper results across sources.
//!
//! Strategy: DOI match (canonicalized) -> arXiv ID match (exact) -> fuzzy title match.

use crate::models::Paper;
use std::collections::HashSet;
use strsim::normalized_levenshtein;

pub fn canonicalize_doi(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let lowered = trimmed.to_ascii_lowercase();
    let without_scheme = lowered
        .strip_prefix("https://")
        .or_else(|| lowered.strip_prefix("http://"))
        .unwrap_or(&lowered);
    let without_host = without_scheme
        .strip_prefix("doi.org/")
        .or_else(|| without_scheme.strip_prefix("dx.doi.org/"))
        .unwrap_or(without_scheme);
    without_host.trim_matches('/').to_string()
}

/// Deduplicate papers, preferring entries with more metadata.
pub fn deduplicate(papers: Vec<Paper>) -> Vec<Paper> {
    let mut seen_keys: HashSet<String> = HashSet::new();
    let mut result: Vec<Paper> = Vec::new();

    // Sort by citation count descending (prefer well-cited versions)
    let mut papers = papers;
    papers.sort_by_key(|p| std::cmp::Reverse(p.citation_count));

    for paper in papers {
        let key = paper.dedup_key();

        // Exact match on DOI or arXiv ID
        if seen_keys.contains(&key) {
            continue;
        }

        // Fuzzy title match against existing results
        let title_normalized: String = paper
            .title
            .to_lowercase()
            .chars()
            .filter(|c| c.is_alphanumeric() || c.is_whitespace())
            .collect();

        let is_fuzzy_dup = result.iter().any(|existing| {
            let existing_norm: String = existing
                .title
                .to_lowercase()
                .chars()
                .filter(|c| c.is_alphanumeric() || c.is_whitespace())
                .collect();
            normalized_levenshtein(&title_normalized, &existing_norm) > 0.85
        });

        if is_fuzzy_dup {
            continue;
        }

        seen_keys.insert(key);
        result.push(paper);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_exact_doi_dedup() {
        let papers = vec![
            Paper {
                paper_id: "1".into(),
                title: "Test Paper".into(),
                doi: "10.1234/test".into(),
                source: "openalex".into(),
                citation_count: 10,
                ..Default::default()
            },
            Paper {
                paper_id: "2".into(),
                title: "Test Paper".into(),
                doi: "10.1234/test".into(),
                source: "crossref".into(),
                citation_count: 5,
                ..Default::default()
            },
        ];
        let result = deduplicate(papers);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].source, "openalex"); // higher citation count
    }

    #[test]
    fn test_canonical_doi_dedup() {
        let papers = vec![
            Paper {
                paper_id: "1".into(),
                title: "Exact Same Paper".into(),
                doi: "https://doi.org/10.1007/BF02854388".into(),
                source: "openalex".into(),
                citation_count: 10,
                ..Default::default()
            },
            Paper {
                paper_id: "2".into(),
                title: "Exact Same Paper".into(),
                doi: "10.1007/bf02854388".into(),
                source: "crossref".into(),
                citation_count: 8,
                ..Default::default()
            },
        ];
        let result = deduplicate(papers);
        assert_eq!(result.len(), 1);
        assert_eq!(canonicalize_doi("https://doi.org/10.1007/BF02854388"), "10.1007/bf02854388");
    }

    #[test]
    fn test_fuzzy_title_dedup() {
        let papers = vec![
            Paper {
                paper_id: "1".into(),
                title: "The Geometry of Sedenion Zero Divisors".into(),
                source: "s2".into(),
                citation_count: 5,
                ..Default::default()
            },
            Paper {
                paper_id: "2".into(),
                title: "Geometry of sedenion zero-divisors".into(),
                source: "core".into(),
                citation_count: 3,
                ..Default::default()
            },
        ];
        let result = deduplicate(papers);
        assert_eq!(result.len(), 1);
    }
}

// Default is derived on Paper in models.rs
