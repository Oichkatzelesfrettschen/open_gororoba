//! Text analysis: section splitting, equation extraction, metadata parsing.

use regex::Regex;
use std::sync::LazyLock;

use crate::{Equation, PaperMetadata, Section};

// Regex patterns compiled once via LazyLock.

/// Matches section headers like "1. Introduction" or "2 Preliminaries"
static SECTION_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(\d+(?:\.\d+)*)[.\s]+([A-Z][^\n]{2,80})$").unwrap()
});

/// Matches display equations: $$...$$ or \begin{equation}...\end{equation}
static DISPLAY_EQ_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\$\$(.+?)\$\$|\\\[(.+?)\\\]|\\begin\{equation\}(.+?)\\end\{equation\}|\\begin\{align\}(.+?)\\end\{align\}")
        .unwrap()
});

/// Matches inline equations: $...$
static INLINE_EQ_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?s)\$([^$]+?)\$").unwrap()
});

/// Matches equation labels: \label{eq:...}
static LABEL_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\\label\{([^}]+)\}").unwrap()
});

/// Split extracted text into sections based on numbered headings.
pub fn split_sections(text: &str) -> Vec<Section> {
    let mut sections = Vec::new();
    let matches: Vec<_> = SECTION_RE.find_iter(text).collect();

    if matches.is_empty() {
        // No section structure detected -- return full text as one section
        return vec![Section {
            number: None,
            title: "Full Text".into(),
            text: text.to_string(),
        }];
    }

    // Text before first section header
    let preamble = text[..matches[0].start()].trim();
    if !preamble.is_empty() {
        sections.push(Section {
            number: Some("0".into()),
            title: "Preamble".into(),
            text: preamble.to_string(),
        });
    }

    for (i, m) in matches.iter().enumerate() {
        let caps = SECTION_RE.captures(m.as_str()).unwrap();
        let number = caps.get(1).map(|c| c.as_str().to_string());
        let title = caps.get(2).map_or("", |c| c.as_str()).trim().to_string();

        let start = m.end();
        let end = if i + 1 < matches.len() {
            matches[i + 1].start()
        } else {
            text.len()
        };

        let body = text[start..end].trim().to_string();
        sections.push(Section {
            number,
            title,
            text: body,
        });
    }

    sections
}

/// Extract LaTeX equations from text.
pub fn extract_equations(text: &str) -> Vec<Equation> {
    let mut equations = Vec::new();

    // Display equations first (higher priority)
    for caps in DISPLAY_EQ_RE.captures_iter(text) {
        let latex = caps
            .get(1)
            .or(caps.get(2))
            .or(caps.get(3))
            .or(caps.get(4))
            .map_or("", |m| m.as_str())
            .trim()
            .to_string();

        let label = LABEL_RE
            .captures(&latex)
            .and_then(|c| c.get(1))
            .map(|m| m.as_str().to_string());

        if !latex.is_empty() {
            equations.push(Equation {
                label,
                latex,
                display: true,
            });
        }
    }

    // Inline equations (skip those already captured as display)
    // Build a set of byte ranges covered by display equations to avoid double-matching
    let display_ranges: Vec<(usize, usize)> = DISPLAY_EQ_RE
        .find_iter(text)
        .map(|m| (m.start(), m.end()))
        .collect();

    for caps in INLINE_EQ_RE.captures_iter(text) {
        let m = caps.get(0).unwrap();
        // Skip if this match falls inside a display equation range
        if display_ranges
            .iter()
            .any(|&(start, end)| m.start() >= start && m.end() <= end)
        {
            continue;
        }
        let latex = caps.get(1).map_or("", |m| m.as_str()).trim().to_string();
        // Skip very short matches (likely dollar amounts or other non-math)
        if latex.len() >= 3 && !latex.chars().all(|c| c.is_ascii_digit() || c == '.') {
            equations.push(Equation {
                label: None,
                latex,
                display: false,
            });
        }
    }

    equations
}

/// Try to extract paper metadata from the first page text.
///
/// Heuristic: title is typically the first substantial line,
/// authors are lines with comma-separated names before the abstract.
pub fn extract_metadata(first_page: &str) -> PaperMetadata {
    let lines: Vec<&str> = first_page
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    let title = lines.first().map_or("Unknown".into(), |l| l.to_string());

    // Authors heuristic: lines between title and "Abstract" or first section
    let mut authors = Vec::new();
    let mut found_abstract = false;
    let mut abstract_text = String::new();

    for (i, line) in lines.iter().enumerate() {
        if i == 0 {
            continue; // skip title
        }
        let lower = line.to_lowercase();
        if lower.starts_with("abstract") {
            found_abstract = true;
            // Collect abstract text (rest of line + following lines)
            let rest = line
                .strip_prefix("Abstract")
                .or(line.strip_prefix("abstract"))
                .or(line.strip_prefix("ABSTRACT"))
                .unwrap_or(line)
                .trim_start_matches(['.', ':', '-', ' ']);
            if !rest.is_empty() {
                abstract_text.push_str(rest);
            }
            continue;
        }
        if found_abstract {
            // Collect until next section header
            if SECTION_RE.is_match(line) {
                break;
            }
            if !abstract_text.is_empty() {
                abstract_text.push(' ');
            }
            abstract_text.push_str(line);
        } else if i < 5 && !lower.starts_with("arxiv") && !lower.starts_with("http") {
            // Likely author line
            authors.push(line.to_string());
        }
    }

    PaperMetadata {
        title,
        authors,
        arxiv: None,
        year: None,
        abstract_text: if abstract_text.is_empty() {
            None
        } else {
            Some(abstract_text)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_sections_with_numbers() {
        let text = "Preamble text here.\n\n1. Introduction\nSome intro text.\n\n2. Methods\nSome methods text.";
        let sections = split_sections(text);
        assert!(sections.len() >= 2, "should detect numbered sections");
        assert!(
            sections.iter().any(|s| s.title.contains("Introduction")),
            "should find Introduction section"
        );
    }

    #[test]
    fn test_split_sections_no_structure() {
        let text = "Just some plain text without section headers.";
        let sections = split_sections(text);
        assert_eq!(sections.len(), 1);
        assert_eq!(sections[0].title, "Full Text");
    }

    #[test]
    fn test_extract_display_equations() {
        let text = r"Some text $$E = mc^2$$ more text";
        let eqs = extract_equations(text);
        assert_eq!(eqs.len(), 1);
        assert!(eqs[0].display);
        assert!(eqs[0].latex.contains("mc^2"));
    }

    #[test]
    fn test_extract_inline_equations() {
        let text = r"The variable $\alpha$ is important.";
        let eqs = extract_equations(text);
        assert_eq!(eqs.len(), 1);
        assert!(!eqs[0].display);
        assert_eq!(eqs[0].latex, r"\alpha");
    }

    #[test]
    fn test_extract_metadata() {
        let text = "The 42 Assessors and the Box-Kites they Fly\nRobert P. C. de Marrais\nAbstract: This paper studies zero-divisors.";
        let meta = extract_metadata(text);
        assert!(meta.title.contains("42 Assessors"));
        assert!(!meta.authors.is_empty());
        assert!(meta.abstract_text.is_some());
    }
}
