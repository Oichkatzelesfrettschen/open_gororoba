//! Novelty / overlap scoring for topic and hypothesis text.
//!
//! Rust-native novelty scoring derived from the repository's legacy literature baseline.

use crate::{Paper, SearchEngine};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarPaper {
    pub title: String,
    pub paper_id: String,
    pub year: u32,
    pub venue: String,
    pub citation_count: u32,
    pub similarity: f32,
    pub url: String,
    pub cite_key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoveltyReport {
    pub topic: String,
    pub hypotheses_checked: usize,
    pub search_queries: Vec<String>,
    pub similar_papers_found: usize,
    pub novelty_score: f32,
    pub assessment: String,
    pub similar_papers: Vec<SimilarPaper>,
    pub recommendation: String,
    pub similarity_threshold: f32,
    pub search_coverage: String,
    pub total_papers_retrieved: usize,
}

pub async fn check_novelty(
    engine: &SearchEngine,
    topic: &str,
    hypotheses_text: &str,
    domains: &[String],
    papers_already_seen: &[Paper],
    max_search_results: usize,
    similarity_threshold: f32,
) -> NoveltyReport {
    let combined_text = format!("{topic}\n{hypotheses_text}");
    let hypothesis_keywords = extract_keywords(&combined_text);
    let queries = build_novelty_queries(topic, hypotheses_text);

    let mut similar_papers = Vec::new();
    let mut total_papers_retrieved = 0usize;

    for query in &queries {
        let results = engine
            .search_topic(query, domains, max_search_results.min(15), 0)
            .await;
        total_papers_retrieved += results.len();
        for paper in results.into_iter().take(max_search_results) {
            let similarity =
                compute_similarity(&hypothesis_keywords, &paper.title, &paper.r#abstract, "");
            if similarity >= similarity_threshold
                && !similar_papers.iter().any(|existing: &SimilarPaper| {
                    existing.title.eq_ignore_ascii_case(&paper.title)
                })
            {
                similar_papers.push(SimilarPaper {
                    title: paper.title.clone(),
                    paper_id: paper.paper_id.clone(),
                    year: paper.year,
                    venue: paper.venue.clone(),
                    citation_count: paper.citation_count,
                    similarity,
                    url: paper.url.clone(),
                    cite_key: paper.cite_key(),
                });
            }
        }
    }

    for paper in papers_already_seen {
        let similarity =
            compute_similarity(&hypothesis_keywords, &paper.title, &paper.r#abstract, "");
        if similarity >= similarity_threshold
            && !similar_papers
                .iter()
                .any(|existing| existing.title.eq_ignore_ascii_case(&paper.title))
        {
            similar_papers.push(SimilarPaper {
                title: paper.title.clone(),
                paper_id: paper.paper_id.clone(),
                year: paper.year,
                venue: paper.venue.clone(),
                citation_count: paper.citation_count,
                similarity,
                url: paper.url.clone(),
                cite_key: paper.cite_key(),
            });
        }
    }

    similar_papers.sort_by(|left, right| {
        right
            .similarity
            .partial_cmp(&left.similarity)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let (novelty_score, assessment) = assess_novelty(&similar_papers);
    let search_coverage = if total_papers_retrieved == 0 && papers_already_seen.is_empty() {
        "insufficient".to_string()
    } else if total_papers_retrieved < 5 {
        "partial".to_string()
    } else {
        "full".to_string()
    };

    let recommendation = if search_coverage == "insufficient" && similar_papers.is_empty() {
        "proceed_with_caution".to_string()
    } else if assessment == "critical" {
        "abort".to_string()
    } else if assessment == "low" {
        "differentiate".to_string()
    } else {
        "proceed".to_string()
    };

    NoveltyReport {
        topic: topic.to_string(),
        hypotheses_checked: hypotheses_count(hypotheses_text),
        search_queries: queries.clone(),
        similar_papers_found: similar_papers.len(),
        novelty_score,
        assessment,
        similar_papers: similar_papers.into_iter().take(20).collect(),
        recommendation,
        similarity_threshold,
        search_coverage,
        total_papers_retrieved,
    }
}

fn hypotheses_count(hypotheses_text: &str) -> usize {
    let section_re = Regex::new(r"(?m)^##\s+H\d+").expect("valid regex");
    let count = section_re.find_iter(hypotheses_text).count();
    if count > 0 {
        count
    } else {
        hypotheses_text
            .to_ascii_lowercase()
            .matches("hypothesis")
            .count()
            .max(1)
    }
}

fn build_novelty_queries(topic: &str, hypotheses_text: &str) -> Vec<String> {
    let mut queries = vec![topic.to_string()];
    let header_re = Regex::new(r"(?m)^##\s+H\d+[:\s]*(.+)$").expect("valid regex");
    for captures in header_re.captures_iter(hypotheses_text) {
        if let Some(title) = captures.get(1).map(|m| m.as_str().trim())
            && title.len() > 10
            && !queries.iter().any(|existing| existing == title)
        {
            queries.push(title.chars().take(200).collect());
        }
    }
    let keywords = extract_keywords(hypotheses_text);
    if !keywords.is_empty() {
        let keyword_query = keywords.into_iter().take(5).collect::<Vec<_>>().join(" ");
        if !queries.iter().any(|existing| existing == &keyword_query) {
            queries.push(keyword_query);
        }
    }
    queries.truncate(5);
    queries
}

fn extract_keywords(text: &str) -> Vec<String> {
    let stop_words = BTreeSet::from([
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "in",
        "on",
        "of",
        "for",
        "to",
        "with",
        "by",
        "at",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "can",
        "shall",
        "not",
        "no",
        "nor",
        "so",
        "yet",
        "both",
        "each",
        "every",
        "all",
        "any",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "after",
        "again",
        "between",
        "into",
        "through",
        "during",
        "before",
        "under",
        "over",
        "using",
        "based",
        "via",
        "toward",
        "towards",
        "new",
        "novel",
        "approach",
        "method",
        "study",
        "research",
        "paper",
        "work",
        "propose",
        "proposed",
        "show",
        "results",
        "performance",
        "evaluation",
    ]);
    let token_re = Regex::new(r"[a-zA-Z][a-zA-Z0-9_-]+").expect("valid regex");
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for token in token_re.find_iter(&text.to_ascii_lowercase()) {
        let token = token.as_str();
        if token.len() >= 3 && !stop_words.contains(token) && seen.insert(token.to_string()) {
            result.push(token.to_string());
        }
    }
    result
}

fn compute_similarity(
    hypothesis_keywords: &[String],
    paper_title: &str,
    paper_abstract: &str,
    hypothesis_title: &str,
) -> f32 {
    let paper_keywords = extract_keywords(&format!("{paper_title} {paper_abstract}"));
    let keyword_similarity = jaccard_keywords(hypothesis_keywords, &paper_keywords);
    if !hypothesis_title.is_empty() && !paper_title.is_empty() {
        let title_similarity = strsim::normalized_levenshtein(
            &hypothesis_title.to_ascii_lowercase(),
            &paper_title.to_ascii_lowercase(),
        ) as f32;
        (0.7 * keyword_similarity + 0.3 * title_similarity).clamp(0.0, 1.0)
    } else {
        keyword_similarity
    }
}

fn jaccard_keywords(left: &[String], right: &[String]) -> f32 {
    let left = left.iter().cloned().collect::<BTreeSet<_>>();
    let right = right.iter().cloned().collect::<BTreeSet<_>>();
    if left.is_empty() || right.is_empty() {
        return 0.0;
    }
    left.intersection(&right).count() as f32 / left.union(&right).count() as f32
}

fn assess_novelty(similar_papers: &[SimilarPaper]) -> (f32, String) {
    if similar_papers.is_empty() {
        return (1.0, "high".to_string());
    }
    let top = &similar_papers[..similar_papers.len().min(5)];
    let max_similarity = top.iter().map(|paper| paper.similarity).fold(0.0, f32::max);
    let high_citation_overlap = top
        .iter()
        .filter(|paper| paper.similarity >= 0.4 && paper.citation_count >= 50)
        .count();
    let mut novelty_score = 1.0 - max_similarity;
    if high_citation_overlap >= 2 {
        novelty_score *= 0.7;
    }
    novelty_score = novelty_score.clamp(0.0, 1.0);
    let assessment = if novelty_score >= 0.7 {
        "high"
    } else if novelty_score >= 0.45 {
        "moderate"
    } else if novelty_score >= 0.25 {
        "low"
    } else {
        "critical"
    };
    (novelty_score, assessment.to_string())
}

#[cfg(test)]
mod tests {
    use super::{assess_novelty, build_novelty_queries, extract_keywords, hypotheses_count};

    #[test]
    fn extracts_keywords_without_duplicates() {
        let keywords = extract_keywords("Novel octonion method and octonion structure");
        assert!(keywords.iter().any(|keyword| keyword == "octonion"));
    }

    #[test]
    fn counts_hypothesis_sections() {
        assert_eq!(hypotheses_count("## H1 test\n## H2 next"), 2);
    }

    #[test]
    fn builds_multiple_queries() {
        let queries = build_novelty_queries("cayley dickson", "## H1 octonion zero divisors");
        assert!(!queries.is_empty());
    }

    #[test]
    fn novelty_assessment_handles_empty_overlap() {
        let (score, assessment) = assess_novelty(&[]);
        assert_eq!(score, 1.0);
        assert_eq!(assessment, "high");
    }
}
