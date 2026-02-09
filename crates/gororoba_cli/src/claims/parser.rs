//! Markdown table parser with escaped-pipe awareness.
//!
//! Replaces ~13 independent Python implementations of `_parse_table_line_strict`.
//! Handles backtick-quoted code spans (pipes inside backticks are not cell delimiters)
//! and backslash-escaped pipes.
//!
//! # Format
//! ```text
//! | C-001 | Claim text | `path/to/file.rs` | **Verified** | 2026-01-15 | notes |
//! ```
//! The parser splits on unescaped, non-code-span `|` characters.

use regex::Regex;
use std::sync::LazyLock;

/// A parsed row from a markdown table.
#[derive(Debug, Clone)]
pub struct ParsedRow {
    /// 1-based line number in the source file.
    pub lineno: usize,
    /// The raw line text.
    pub raw: String,
    /// The extracted cell contents (trimmed).
    pub cells: Vec<String>,
}

/// A parsed claim row with typed fields extracted from the 6-column matrix.
#[derive(Debug, Clone)]
pub struct ClaimRow {
    /// Claim identifier, e.g. "C-001".
    pub claim_id: String,
    /// Numeric part of the claim ID.
    pub claim_num: u32,
    /// The claim text (column 2).
    pub claim_text: String,
    /// The "Where stated" column (column 3).
    pub where_stated: String,
    /// The raw status cell (column 4).
    pub status_cell: String,
    /// The extracted status token (between `**` markers), or empty.
    pub status_token: String,
    /// The "Last verified" cell (column 5).
    pub last_verified: String,
    /// The ISO date prefix from last_verified, if valid.
    pub last_verified_date: Option<String>,
    /// The "Evidence/notes" cell (column 6).
    pub evidence_notes: String,
    /// 1-based line number in the source file.
    pub lineno: usize,
}

static CLAIM_ID_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^C-(\d{3})$").expect("valid regex"));

static STATUS_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^\*\*([^*]+)\*\*").expect("valid regex"));

static ISO_DATE_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"^(\d{4}-\d{2}-\d{2})\b").expect("valid regex"));

static BACKTICK_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"`([^`]+)`").expect("valid regex"));

/// Split a markdown table row into cells, respecting escaped pipes and code spans.
///
/// Returns `None` if the line doesn't start and end with `|`.
pub fn parse_table_line(line: &str) -> Option<Vec<String>> {
    let trimmed = line.trim_end();
    if !trimmed.starts_with('|') || !trimmed.ends_with('|') {
        return None;
    }

    let mut cells = Vec::new();
    let mut buf = String::new();
    let mut escaped = false;
    let mut in_code = false;

    // Skip leading pipe.
    let chars: Vec<char> = trimmed.chars().collect();
    let mut i = 1;
    while i < chars.len() {
        let ch = chars[i];
        if escaped {
            buf.push(ch);
            escaped = false;
        } else if ch == '\\' {
            escaped = true;
            buf.push(ch);
        } else if ch == '`' {
            in_code = !in_code;
            buf.push(ch);
        } else if ch == '|' && !in_code {
            cells.push(buf.trim().to_string());
            buf.clear();
        } else {
            buf.push(ch);
        }
        i += 1;
    }

    // Trailing content after final pipe should be empty.
    let remainder = buf.trim();
    if !remainder.is_empty() {
        return None;
    }

    // Drop trailing empty cell from final `|`.
    if cells.last().is_some_and(|c| c.is_empty()) {
        cells.pop();
    }

    Some(cells)
}

/// Iterate over all table rows in markdown text, yielding parsed rows.
///
/// Skips separator lines (containing `---`).
pub fn iter_table_rows(text: &str) -> Vec<ParsedRow> {
    let mut rows = Vec::new();
    for (lineno_0, line) in text.lines().enumerate() {
        let lineno = lineno_0 + 1;
        if let Some(cells) = parse_table_line(line) {
            // Skip separator rows like | --- | --- | --- |.
            if cells
                .iter()
                .all(|c| c.chars().all(|ch| ch == '-' || ch == ':' || ch == ' '))
            {
                continue;
            }
            // Skip header rows (first table row with non-C- content).
            rows.push(ParsedRow {
                lineno,
                raw: line.to_string(),
                cells,
            });
        }
    }
    rows
}

/// Parse all claim rows from the 6-column CLAIMS_EVIDENCE_MATRIX.md format.
///
/// Only returns rows where column 1 matches `C-NNN`.
pub fn parse_claim_rows(text: &str) -> Vec<ClaimRow> {
    let mut claims = Vec::new();
    for row in iter_table_rows(text) {
        if row.cells.len() != 6 {
            continue;
        }
        let claim_id = &row.cells[0];
        let caps = match CLAIM_ID_RE.captures(claim_id) {
            Some(c) => c,
            None => continue,
        };
        let claim_num: u32 = caps[1].parse().unwrap_or(0);

        let status_cell = &row.cells[3];
        let status_token = STATUS_RE
            .captures(status_cell)
            .map(|c| c[1].trim().to_string())
            .unwrap_or_default();

        let last_verified = &row.cells[4];
        let last_verified_date = ISO_DATE_RE.captures(last_verified).and_then(|c| {
            let date_str = c[1].to_string();
            // Validate it's a real date.
            if is_valid_iso_date(&date_str) {
                Some(date_str)
            } else {
                None
            }
        });

        claims.push(ClaimRow {
            claim_id: claim_id.clone(),
            claim_num,
            claim_text: row.cells[1].clone(),
            where_stated: row.cells[2].clone(),
            status_cell: status_cell.clone(),
            status_token,
            last_verified: last_verified.clone(),
            last_verified_date,
            evidence_notes: row.cells[5].clone(),
            lineno: row.lineno,
        });
    }
    claims
}

/// Extract backtick-quoted paths from text.
pub fn extract_backtick_paths(text: &str) -> Vec<String> {
    BACKTICK_RE
        .captures_iter(text)
        .map(|c| c[1].to_string())
        .filter(|t| looks_like_repo_path(t))
        .map(|t| normalize_repo_path(&t))
        .collect()
}

/// Check if a token looks like a repository file path.
pub fn looks_like_repo_path(token: &str) -> bool {
    let t = token.trim();
    if t.is_empty() || t.contains(' ') {
        return false;
    }
    if t.contains('/') {
        return true;
    }
    t.ends_with(".py")
        || t.ends_with(".md")
        || t.ends_with(".csv")
        || t.ends_with(".json")
        || t.ends_with(".txt")
        || t.ends_with(".rs")
        || t.ends_with(".toml")
}

/// Normalize path-like references (strip pytest node IDs like `::TestFoo`).
pub fn normalize_repo_path(token: &str) -> String {
    let t = token.trim();
    if let Some(idx) = t.find("::") {
        let candidate = t[..idx].trim();
        if looks_like_repo_path(candidate) {
            return candidate.to_string();
        }
    }
    t.to_string()
}

/// Validate an ISO date string (YYYY-MM-DD).
fn is_valid_iso_date(s: &str) -> bool {
    if s.len() != 10 {
        return false;
    }
    let parts: Vec<&str> = s.split('-').collect();
    if parts.len() != 3 {
        return false;
    }
    let year: u32 = match parts[0].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let month: u32 = match parts[1].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    let day: u32 = match parts[2].parse() {
        Ok(v) => v,
        Err(_) => return false,
    };
    if !(2020..=2099).contains(&year) {
        return false;
    }
    if !(1..=12).contains(&month) {
        return false;
    }
    let max_day = match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if year.is_multiple_of(4) && (!year.is_multiple_of(100) || year.is_multiple_of(400)) {
                29
            } else {
                28
            }
        }
        _ => return false,
    };
    (1..=max_day).contains(&day)
}

/// Strip markdown formatting (bold, code, italic) for plain-text output.
pub fn strip_markdown(text: &str) -> String {
    text.replace('`', "").replace("**", "").replace('*', "")
}

/// Shorten text to a maximum length, appending "..." if truncated.
pub fn shorten(text: &str, limit: usize) -> String {
    let clean = strip_markdown(text).replace('\n', " ");
    let trimmed = clean.trim();
    if trimmed.len() <= limit {
        trimmed.to_string()
    } else {
        format!("{}...", trimmed[..limit.saturating_sub(3)].trim_end())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_table_line_basic() {
        let cells = parse_table_line("| A | B | C |").unwrap();
        assert_eq!(cells, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_parse_table_line_claim_row() {
        let line = "| C-001 | Some claim | `path/file.rs` | **Verified** | 2026-01-15 | notes |";
        let cells = parse_table_line(line).unwrap();
        assert_eq!(cells.len(), 6);
        assert_eq!(cells[0], "C-001");
        assert_eq!(cells[3], "**Verified**");
    }

    #[test]
    fn test_parse_table_line_escaped_pipe() {
        let line = r"| C-002 | Claim with \| pipe | `file.rs` | **Verified** | 2026-01-15 | n |";
        let cells = parse_table_line(line).unwrap();
        assert_eq!(cells.len(), 6);
        assert!(cells[1].contains(r"\|"));
    }

    #[test]
    fn test_parse_table_line_backtick_pipe() {
        let line = "| C-003 | Claim | `fn|method` | **Verified** | 2026-01-15 | n |";
        let cells = parse_table_line(line).unwrap();
        assert_eq!(cells.len(), 6);
        assert_eq!(cells[2], "`fn|method`");
    }

    #[test]
    fn test_parse_table_line_not_table() {
        assert!(parse_table_line("Not a table row").is_none());
        assert!(parse_table_line("| Missing end").is_none());
    }

    #[test]
    fn test_parse_claim_rows() {
        let text = "\
| Claim | Description | Where | Status | Verified | Notes |
| --- | --- | --- | --- | --- | --- |
| C-001 | First claim | `src/lib.rs` | **Verified** | 2026-01-15 | tested |
| C-002 | Second claim | `docs/foo.md` | **Speculative** | 2025-12-01 | pending |
| Not a claim | blah | blah | blah | blah | blah |
";
        let claims = parse_claim_rows(text);
        assert_eq!(claims.len(), 2);
        assert_eq!(claims[0].claim_id, "C-001");
        assert_eq!(claims[0].claim_num, 1);
        assert_eq!(claims[0].status_token, "Verified");
        assert_eq!(claims[0].last_verified_date.as_deref(), Some("2026-01-15"));
        assert_eq!(claims[1].claim_id, "C-002");
        assert_eq!(claims[1].status_token, "Speculative");
    }

    #[test]
    fn test_extract_backtick_paths() {
        let text = "See `src/foo.rs` and `docs/bar.md` and `cargo test` and `tests/t.py::TestFoo`";
        let paths = extract_backtick_paths(text);
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&"src/foo.rs".to_string()));
        assert!(paths.contains(&"docs/bar.md".to_string()));
        assert!(paths.contains(&"tests/t.py".to_string())); // stripped ::TestFoo
    }

    #[test]
    fn test_is_valid_iso_date() {
        assert!(is_valid_iso_date("2026-01-15"));
        assert!(is_valid_iso_date("2024-02-29")); // leap year
        assert!(!is_valid_iso_date("2023-02-29")); // not leap year
        assert!(!is_valid_iso_date("2026-13-01")); // bad month
        assert!(!is_valid_iso_date("2026-00-01")); // zero month
        assert!(!is_valid_iso_date("not-a-date"));
    }

    #[test]
    fn test_looks_like_repo_path() {
        assert!(looks_like_repo_path("src/foo.rs"));
        assert!(looks_like_repo_path("docs/bar.md"));
        assert!(looks_like_repo_path("file.py"));
        assert!(!looks_like_repo_path("cargo test"));
        assert!(!looks_like_repo_path(""));
    }

    #[test]
    fn test_shorten() {
        assert_eq!(shorten("Short text", 20), "Short text");
        assert_eq!(
            shorten("A very long text that exceeds the limit", 20),
            "A very long text..."
        );
    }
}
