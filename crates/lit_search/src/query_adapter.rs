//! Per-source query adaptation for academic search APIs.
//!
//! Rust-native query adapter derived from the repository's legacy literature baseline.

/// Adapt a generic query for a specific source's syntax.
pub fn adapt_query(query: &str, source: &str, year_min: u32) -> String {
    let source_key = normalize_source(source);
    match source_key.as_str() {
        "openalex" => clean_query(query),
        "semantic_scholar" | "s2" => adapt_semantic_scholar(query),
        "arxiv" => adapt_arxiv(query),
        "ads" => adapt_ads(query, year_min),
        "inspirehep" => adapt_inspirehep(query, year_min),
        "crossref" => clean_query(query),
        "europepmc" => adapt_europepmc(query, year_min),
        "core" => adapt_core(query, year_min),
        "hal" => clean_query(query),
        "datacite" => adapt_datacite(query, year_min),
        "scielo" => adapt_scielo(query, year_min),
        "cinii" => adapt_cinii(query),
        "dblp" => adapt_dblp(query),
        "jstage" => clean_query(query),
        "lens" => clean_query(query),
        _ => clean_query(query),
    }
}

/// Expand a set of generic queries into per-source adapted variants.
pub fn expand_queries(
    queries: &[String],
    sources: &[&str],
    year_min: u32,
) -> std::collections::BTreeMap<String, Vec<String>> {
    let mut expanded = std::collections::BTreeMap::new();
    for source in sources {
        let source_key = normalize_source(source);
        let mut adapted = Vec::new();
        let mut seen = std::collections::BTreeSet::new();
        for query in queries {
            let candidate = adapt_query(query, &source_key, year_min);
            let normalized = candidate.trim().to_ascii_lowercase();
            if !normalized.is_empty() && seen.insert(normalized) {
                adapted.push(candidate);
            }
        }
        expanded.insert(source_key, adapted);
    }
    expanded
}

fn adapt_semantic_scholar(query: &str) -> String {
    let cleaned = clean_query(query);
    if cleaned.len() > 200 {
        cleaned.chars().take(200).collect()
    } else {
        cleaned
    }
}

fn adapt_arxiv(query: &str) -> String {
    let cleaned = clean_query(query);
    if ["ti:", "au:", "abs:", "all:"]
        .iter()
        .any(|field| cleaned.contains(field))
    {
        return cleaned;
    }

    let terms = extract_phrases_and_terms(&cleaned);
    if terms.is_empty() {
        return cleaned;
    }

    terms
        .into_iter()
        .map(|term| {
            if term.contains(' ') {
                format!("all:\"{term}\"")
            } else {
                format!("all:{term}")
            }
        })
        .collect::<Vec<_>>()
        .join(" AND ")
}

fn adapt_ads(query: &str, year_min: u32) -> String {
    let cleaned = clean_query(query);
    if ["title:", "author:", "abs:", "bibcode:", "bibstem:", "year:"]
        .iter()
        .any(|field| cleaned.contains(field))
    {
        return cleaned;
    }

    let mut parts = extract_phrases_and_terms(&cleaned)
        .into_iter()
        .map(|term| {
            if term.contains(' ') {
                format!("abs:\"{term}\"")
            } else {
                format!("abs:{term}")
            }
        })
        .collect::<Vec<_>>();
    if year_min > 0 {
        parts.push(format!("year:{year_min}-9999"));
    }
    if parts.is_empty() {
        cleaned
    } else {
        parts.join(" ")
    }
}

fn adapt_inspirehep(query: &str, year_min: u32) -> String {
    let cleaned = clean_query(query);
    if ["t:", "a:", "k:", "find ", "d:"]
        .iter()
        .any(|field| cleaned.contains(field))
    {
        return cleaned;
    }

    let mut parts = extract_phrases_and_terms(&cleaned)
        .into_iter()
        .map(|term| {
            if term.contains(' ') {
                format!("t:\"{term}\"")
            } else {
                format!("t:{term}")
            }
        })
        .collect::<Vec<_>>();
    if year_min > 0 {
        parts.push(format!("d:{year_min}->2030"));
    }
    if parts.is_empty() {
        cleaned
    } else {
        parts.join(" and ")
    }
}

fn adapt_europepmc(query: &str, year_min: u32) -> String {
    let cleaned = clean_query(query);
    let upper = cleaned.to_ascii_uppercase();
    if ["TITLE:", "AUTH:", "ABSTRACT:"]
        .iter()
        .any(|field| upper.contains(field))
    {
        return cleaned;
    }

    let mut parts = extract_phrases_and_terms(&cleaned)
        .into_iter()
        .map(|term| {
            if term.contains(' ') {
                format!("\"{term}\"")
            } else {
                term
            }
        })
        .collect::<Vec<_>>();
    if year_min > 0 {
        parts.push(format!("PUB_YEAR:[{year_min} TO 2030]"));
    }
    if parts.is_empty() {
        cleaned
    } else {
        parts.join(" AND ")
    }
}

fn adapt_core(query: &str, year_min: u32) -> String {
    let cleaned = clean_query(query);
    if ["title:", "authors:", "fullText:", "_exists_:"]
        .iter()
        .any(|field| cleaned.contains(field))
    {
        return cleaned;
    }

    let mut parts = extract_phrases_and_terms(&cleaned)
        .into_iter()
        .map(|term| {
            if term.contains(' ') {
                format!("title:\"{term}\"")
            } else {
                term
            }
        })
        .collect::<Vec<_>>();
    if year_min > 0 {
        parts.push(format!("yearPublished>={year_min}"));
    }
    if parts.is_empty() {
        cleaned
    } else {
        parts.join(" AND ")
    }
}

fn adapt_datacite(query: &str, year_min: u32) -> String {
    let mut cleaned = clean_query(query);
    if year_min > 0 && !cleaned.contains("publicationYear") {
        cleaned.push_str(&format!(" AND publicationYear:[{year_min} TO *]"));
    }
    cleaned
}

fn adapt_scielo(query: &str, year_min: u32) -> String {
    let mut cleaned = clean_query(query);
    if year_min > 0 && !cleaned.contains("publication_year:") {
        cleaned.push_str(&format!(" AND publication_year:{year_min}"));
    }
    cleaned
}

fn adapt_cinii(query: &str) -> String {
    let cleaned = clean_query(query);
    if cleaned.len() > 100 {
        cleaned.chars().take(100).collect()
    } else {
        cleaned
    }
}

fn adapt_dblp(query: &str) -> String {
    let cleaned = clean_query(query);
    let words = cleaned
        .split_whitespace()
        .filter(|word| {
            let upper = word.to_ascii_uppercase();
            upper != "AND" && upper != "OR" && upper != "NOT"
        })
        .take(6)
        .collect::<Vec<_>>();
    words.join(" ")
}

fn normalize_source(source: &str) -> String {
    source.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

fn clean_query(query: &str) -> String {
    let mut cleaned = String::with_capacity(query.len());
    let mut last_was_space = false;
    for ch in query.trim().chars() {
        if matches!(ch, '*' | '_' | '`' | '#') {
            continue;
        }
        if ch.is_whitespace() {
            if !last_was_space {
                cleaned.push(' ');
                last_was_space = true;
            }
        } else {
            cleaned.push(ch);
            last_was_space = false;
        }
    }
    cleaned.trim().to_string()
}

fn extract_phrases_and_terms(query: &str) -> Vec<String> {
    let mut phrases = extract_quoted_phrases(query);
    let scrubbed = scrub_quoted_segments(query);
    let stopwords = [
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "across",
        "multiple",
        "three",
        "result",
        "comprehensive",
        "using",
        "based",
        "between",
        "various",
        "different",
        "several",
        "about",
        "their",
        "these",
        "those",
        "which",
        "where",
        "when",
        "have",
        "been",
        "some",
        "each",
        "also",
        "much",
        "very",
        "more",
        "than",
        "does",
        "what",
        "such",
        "only",
        "other",
        "like",
    ];

    let meaningful = scrubbed
        .split_whitespace()
        .map(trim_token)
        .filter(|token| token.len() > 2)
        .filter(|token| {
            !stopwords
                .iter()
                .any(|stopword| token.eq_ignore_ascii_case(stopword))
        })
        .map(ToString::to_string)
        .collect::<Vec<_>>();

    let mut index = 0;
    while index < meaningful.len() {
        if index + 1 < meaningful.len()
            && meaningful[index].len() > 3
            && meaningful[index + 1].len() > 3
        {
            phrases.push(format!("{} {}", meaningful[index], meaningful[index + 1]));
            index += 2;
        } else {
            phrases.push(meaningful[index].clone());
            index += 1;
        }
    }

    phrases
}

fn extract_quoted_phrases(query: &str) -> Vec<String> {
    let mut phrases = Vec::new();
    let mut in_quotes = false;
    let mut current = String::new();
    for ch in query.chars() {
        if ch == '"' {
            if in_quotes {
                let phrase = current.trim();
                if !phrase.is_empty() {
                    phrases.push(phrase.to_string());
                }
                current.clear();
                in_quotes = false;
            } else {
                in_quotes = true;
            }
            continue;
        }
        if in_quotes {
            current.push(ch);
        }
    }
    phrases
}

fn scrub_quoted_segments(query: &str) -> String {
    let mut result = String::with_capacity(query.len());
    let mut in_quotes = false;
    for ch in query.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
            result.push(' ');
        } else if in_quotes {
            result.push(' ');
        } else {
            result.push(ch);
        }
    }
    result
}

fn trim_token(token: &str) -> &str {
    token.trim_matches(|ch: char| !ch.is_alphanumeric() && ch != ':' && ch != '-')
}

#[cfg(test)]
mod tests {
    use super::{adapt_query, expand_queries};

    #[test]
    fn arxiv_adds_field_prefixes_for_plain_queries() {
        let adapted = adapt_query("Composition algebras automorphisms", "arxiv", 0);
        assert!(
            adapted.contains("all:Composition") || adapted.contains("all:\"Composition algebras\"")
        );
    }

    #[test]
    fn dblp_strips_boolean_tokens() {
        let adapted = adapt_query("cayley dickson AND zero divisors OR octonion", "dblp", 0);
        assert!(!adapted.contains("AND"));
        assert!(!adapted.contains("OR"));
    }

    #[test]
    fn datacite_adds_year_filter() {
        let adapted = adapt_query("Freudenthal octaven", "datacite", 1951);
        assert!(adapted.contains("publicationYear:[1951 TO *]"));
    }

    #[test]
    fn expand_queries_deduplicates_per_source() {
        let queries = vec!["Cayley Dickson".to_string(), "Cayley Dickson".to_string()];
        let expanded = expand_queries(&queries, &["openalex"], 0);
        assert_eq!(expanded["openalex"].len(), 1);
    }
}
