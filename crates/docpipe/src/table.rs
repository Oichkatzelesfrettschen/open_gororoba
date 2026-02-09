//! Table detection heuristic for extracted PDF text.
//!
//! Strategy: detect runs of lines with consistent column separators
//! (tabs, multiple spaces, pipe characters). This is inherently
//! heuristic since PDFs have no semantic table markup.

use crate::{PageText, Table};

/// Minimum number of columns to consider something a table row.
const MIN_COLS: usize = 2;
/// Minimum number of data rows (excluding header) to form a table.
const MIN_ROWS: usize = 2;

/// Detect tables in extracted page text.
///
/// Heuristic: lines that split into a consistent number of columns
/// (separated by 2+ spaces or tabs) across multiple consecutive lines
/// are likely table rows.
pub fn detect_tables(pages: &[PageText]) -> Vec<Table> {
    let mut tables = Vec::new();

    for page in pages {
        let lines: Vec<&str> = page.text.lines().collect();
        let mut i = 0;

        while i < lines.len() {
            // Try to find a run of lines with consistent column structure
            let cols = split_columns(lines[i]);
            if cols.len() >= MIN_COLS {
                let expected_cols = cols.len();
                let mut run_start = i;
                let mut run_end = i + 1;

                // Extend the run while column count is consistent
                while run_end < lines.len() {
                    let next_cols = split_columns(lines[run_end]);
                    // Allow +/- 1 column tolerance for merged cells
                    if next_cols.len() >= MIN_COLS
                        && (next_cols.len() as i32 - expected_cols as i32).unsigned_abs() <= 1
                    {
                        run_end += 1;
                    } else {
                        break;
                    }
                }

                let run_len = run_end - run_start;
                if run_len > MIN_ROWS {
                    // First row is header, rest are data
                    let headers = split_columns(lines[run_start])
                        .into_iter()
                        .map(|s| s.to_string())
                        .collect();
                    run_start += 1;

                    // Skip separator lines (all dashes/equals)
                    if run_start < run_end && is_separator_line(lines[run_start]) {
                        run_start += 1;
                    }

                    let rows: Vec<Vec<String>> = (run_start..run_end)
                        .filter(|&j| !is_separator_line(lines[j]))
                        .map(|j| {
                            split_columns(lines[j])
                                .into_iter()
                                .map(|s| s.to_string())
                                .collect()
                        })
                        .collect();

                    if rows.len() >= MIN_ROWS {
                        tables.push(Table {
                            page_num: page.page_num,
                            headers,
                            rows,
                        });
                    }
                }

                i = run_end;
            } else {
                i += 1;
            }
        }
    }

    tables
}

/// Split a line into columns by 2+ whitespace or tab characters.
fn split_columns(line: &str) -> Vec<&str> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return Vec::new();
    }

    // Try pipe-separated first
    if trimmed.contains('|') {
        let cols: Vec<&str> = trimmed
            .split('|')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if cols.len() >= MIN_COLS {
            return cols;
        }
    }

    // Fall back to whitespace-separated (2+ spaces or tab)
    let parts: Vec<&str> = trimmed
        .split('\t')
        .flat_map(|part| {
            // Split by 2+ consecutive spaces
            part.split("  ").map(|s| s.trim()).filter(|s| !s.is_empty())
        })
        .collect();

    parts
}

/// Check if a line is a table separator (all dashes, equals, etc.)
fn is_separator_line(line: &str) -> bool {
    let trimmed = line.trim();
    !trimmed.is_empty()
        && trimmed
            .chars()
            .all(|c| c == '-' || c == '=' || c == '+' || c == '|' || c == ' ')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_pipe_table() {
        let text = "Name | Value | Unit\n---|---|---\nalpha | 1.5 | rad\nbeta | 2.0 | rad\ngamma | 3.0 | rad";
        let pages = vec![PageText {
            page_num: 1,
            text: text.to_string(),
        }];
        let tables = detect_tables(&pages);
        assert!(!tables.is_empty(), "should detect pipe-separated table");
        assert_eq!(tables[0].headers.len(), 3);
    }

    #[test]
    fn test_detect_space_table() {
        let text = "Dimension  Components  Motifs\n16         7           1\n32         15          2\n64         31          4";
        let pages = vec![PageText {
            page_num: 1,
            text: text.to_string(),
        }];
        let tables = detect_tables(&pages);
        assert!(!tables.is_empty(), "should detect space-separated table");
    }

    #[test]
    fn test_no_table_in_prose() {
        let text = "This is just a paragraph of regular text. It does not contain any tabular data whatsoever.";
        let pages = vec![PageText {
            page_num: 1,
            text: text.to_string(),
        }];
        let tables = detect_tables(&pages);
        assert!(tables.is_empty(), "should not detect tables in prose");
    }

    #[test]
    fn test_split_columns_pipe() {
        let cols = split_columns("A | B | C");
        assert_eq!(cols, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_split_columns_spaces() {
        let cols = split_columns("alpha  1.5  rad");
        assert_eq!(cols.len(), 3);
    }

    #[test]
    fn test_separator_detection() {
        assert!(is_separator_line("---+---+---"));
        assert!(is_separator_line("========"));
        assert!(!is_separator_line("hello world"));
    }
}
