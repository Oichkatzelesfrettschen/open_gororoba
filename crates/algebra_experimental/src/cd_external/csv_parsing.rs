//! Nested-tuple, lattice-point, and CSV adjacency parsers for the
//! external Cayley-Dickson data files.
//!
//! The external CSVs ship in two complementary surface formats:
//! 1. Cayley-Dickson nested-tuple strings of the form `((A, B), (C, D))`
//!    representing the doubling tree of an n-dimensional basis element.
//! 2. Lattice-point strings of the form `[i1, i2, ..., in]` (integer
//!    coordinates of a lattice vector) and CSV adjacency matrices.
//!
//! The parser is two-pass for the nested-tuple format: first a recursive
//! `CdTree` is built, then `tree_to_vec` flattens it to a power-of-two
//! length vector, padding zero subtrees implicitly.

// ---------------------------------------------------------------------------
// Nested-tuple parser
// ---------------------------------------------------------------------------

/// Parse a Cayley-Dickson nested-tuple string to a flat coefficient vector.
///
/// The nested-tuple format represents the doubling tree:
/// `((A, B), (C, D))` = quaternion with real=A, i=B, j=C, k=D
/// `(((A, B), (C, D)), ((E, F), (G, H)))` = octonion
///
/// **Important:** The `(0, 0)` shorthand at any tree level represents an
/// entire zero subtree. The parser first builds a tree, then balances it
/// so both children of every node have equal size (padding with zeros).
///
/// Returns None if parsing fails.
pub fn parse_nested_tuple(s: &str) -> Option<Vec<f64>> {
    let s = s.trim();
    let tree = parse_tree(s)?;
    Some(tree_to_vec(&tree))
}

/// Internal tree representation for nested tuples.
#[derive(Debug)]
enum CdTree {
    Leaf(f64),
    Pair(Box<CdTree>, Box<CdTree>),
}

fn tree_to_vec(tree: &CdTree) -> Vec<f64> {
    match tree {
        CdTree::Leaf(v) => vec![*v],
        CdTree::Pair(l, r) => {
            let mut lv = tree_to_vec(l);
            let mut rv = tree_to_vec(r);
            // Pad to equal power-of-2 size
            let half = lv.len().max(rv.len()).next_power_of_two();
            lv.resize(half, 0.0);
            rv.resize(half, 0.0);
            lv.extend(rv);
            lv
        }
    }
}

fn parse_tree(s: &str) -> Option<CdTree> {
    let s = s.trim();

    // Base case: a plain number
    if let Ok(v) = s.parse::<f64>() {
        return Some(CdTree::Leaf(v));
    }

    // Must be (A, B) where A and B are sub-expressions
    if !s.starts_with('(') || !s.ends_with(')') {
        return None;
    }

    // Strip outer parens
    let inner = &s[1..s.len() - 1];

    // Find the comma that splits at this level (matching parens)
    let split = find_top_level_comma(inner)?;
    let left = &inner[..split];
    let right = &inner[split + 1..];

    let l = parse_tree(left)?;
    let r = parse_tree(right)?;
    Some(CdTree::Pair(Box::new(l), Box::new(r)))
}

/// Find the position of the first top-level comma in a string,
/// respecting nested parentheses.
fn find_top_level_comma(s: &str) -> Option<usize> {
    let mut depth = 0i32;
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ',' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

/// Convert a flat coefficient vector back to a basis index.
///
/// For a standard basis element (exactly one non-zero entry = 1.0),
/// returns the index of that entry. Returns None otherwise.
pub fn vec_to_basis_index(v: &[f64]) -> Option<usize> {
    let mut found = None;
    for (i, &val) in v.iter().enumerate() {
        if val.abs() > 0.5 {
            if found.is_some() {
                return None; // multiple nonzero
            }
            found = Some(i);
        }
    }
    found
}

// ---------------------------------------------------------------------------
// Lattice mapping
// ---------------------------------------------------------------------------

/// Parse a lattice point string like "[-1, -1, -1, -1, -1, -1, -1, -1]"
/// into a vector of integers.
pub fn parse_lattice_point(s: &str) -> Option<Vec<i32>> {
    let s = s.trim();
    if !s.starts_with('[') || !s.ends_with(']') {
        return None;
    }
    let inner = &s[1..s.len() - 1];
    let vals: Result<Vec<i32>, _> = inner.split(',').map(|x| x.trim().parse::<i32>()).collect();
    vals.ok()
}

// ---------------------------------------------------------------------------
// CSV adjacency matrix parsing
// ---------------------------------------------------------------------------

/// Parse a CSV adjacency matrix (first row = column headers, subsequent rows
/// contain numeric values).
///
/// The first row is treated as a header and skipped. For data rows, all
/// fields that parse as f64 are collected. Non-numeric fields (e.g., string
/// row labels) are silently skipped.
///
/// Returns the matrix as `Vec<Vec<f64>>` (row-major).
pub fn parse_adjacency_csv(content: &str) -> Vec<Vec<f64>> {
    let mut rows = Vec::new();
    let lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return rows;
    }

    // Determine expected column count from header
    let header_fields: Vec<&str> = lines[0].split(',').collect();
    let n_cols = header_fields.len();

    // Skip header row
    for line in &lines[1..] {
        let fields: Vec<&str> = line.split(',').collect();
        if fields.is_empty() {
            continue;
        }

        let vals: Vec<f64> = fields
            .iter()
            .filter_map(|f| f.trim().parse::<f64>().ok())
            .collect();

        // If we got one extra field compared to header, assume first was a row index
        // that happened to be numeric. Take last n_cols values.
        let final_vals = if vals.len() > n_cols {
            vals[vals.len() - n_cols..].to_vec()
        } else {
            vals
        };

        if !final_vals.is_empty() {
            rows.push(final_vals);
        }
    }
    rows
}
