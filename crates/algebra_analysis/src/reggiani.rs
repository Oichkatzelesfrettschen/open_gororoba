//! Reggiani's 84 standard zero-divisors in sedenions.
//!
//! The 42 primitive assessors each generate two unit zero-divisors
//! (e_low + e_high) and (e_low - e_high), giving 84 "standard" zero-divisors.
//! Each standard ZD has a 4-dimensional annihilator subspace, and exactly
//! 4 standard ZD partners (other standard ZDs that annihilate it).
//!
//! # Literature
//! - Reggiani (2024): Geometry of sedenion zero divisors, Table 1

use crate::{
    annihilator::{
        annihilator_info, is_reggiani_zd, left_multiplication_matrix, nullspace_basis,
        right_multiplication_matrix,
    },
    boxkites::{diagonal_zero_products_exact, primitive_assessors},
};
use cd_kernel::cayley_dickson::cd_multiply;

/// A standard zero-divisor: a diagonal of a primitive assessor.
#[derive(Debug, Clone)]
pub struct StandardZeroDivisor {
    pub assessor_low: usize,
    pub assessor_high: usize,
    pub diagonal_sign: i32,
    pub vector: Vec<f64>,
}

impl StandardZeroDivisor {
    fn key(&self) -> (usize, usize, i32) {
        (self.assessor_low, self.assessor_high, self.diagonal_sign)
    }
}

/// Generate all 84 standard zero-divisors (42 assessors x 2 signs).
pub fn standard_zero_divisors() -> Vec<StandardZeroDivisor> {
    let assessors = primitive_assessors();
    let mut out = Vec::with_capacity(84);
    for a in &assessors {
        for sign in [1i32, -1] {
            let mut v = vec![0.0; 16];
            v[a.low] = 1.0;
            v[a.high] = sign as f64;
            out.push(StandardZeroDivisor {
                assessor_low: a.low,
                assessor_high: a.high,
                diagonal_sign: sign,
                vector: v,
            });
        }
    }
    out
}

/// Find the 4 standard zero-divisor partners of `zd` -- other standard ZDs
/// whose product with `zd` is zero.
///
/// Uses integer-exact diagonal zero-product detection from boxkites.
pub fn standard_zero_divisor_partners(zd: &StandardZeroDivisor) -> Vec<StandardZeroDivisor> {
    let a_pair = (zd.assessor_low, zd.assessor_high);
    let s = zd.diagonal_sign as i8;

    let all = standard_zero_divisors();
    let mut partners = Vec::new();

    for cand in &all {
        if cand.assessor_low == zd.assessor_low
            && cand.assessor_high == zd.assessor_high
            && cand.diagonal_sign == zd.diagonal_sign
        {
            continue; // skip self
        }
        let b_pair = (cand.assessor_low, cand.assessor_high);
        let t = cand.diagonal_sign as i8;

        let sols = diagonal_zero_products_exact(16, a_pair, b_pair);
        if sols.contains(&(s, t)) {
            partners.push(StandardZeroDivisor {
                assessor_low: cand.assessor_low,
                assessor_high: cand.assessor_high,
                diagonal_sign: cand.diagonal_sign,
                vector: cand.vector.clone(),
            });
        }
    }

    assert_eq!(
        partners.len(),
        4,
        "Expected 4 standard partners for ({}, {}, {}), got {}",
        zd.assessor_low,
        zd.assessor_high,
        zd.diagonal_sign,
        partners.len()
    );

    partners.sort_by_key(|p| p.key());
    partners
}

/// Verify that a standard ZD satisfies all Reggiani consistency checks:
/// - Squared norm is 2
/// - Has nontrivial left and right annihilator
/// - Nullspace basis vectors actually annihilate
/// - The 4 standard partners span the annihilator subspace
pub fn assert_standard_zero_divisor_annihilators(zd: &StandardZeroDivisor) -> Result<(), String> {
    let v = &zd.vector;
    let norm_sq: f64 = v.iter().map(|x| x * x).sum();
    if (norm_sq - 2.0).abs() > 1e-12 {
        return Err(format!("Squared norm = {norm_sq}, expected 2.0"));
    }

    if !is_reggiani_zd(v, 1e-12) {
        return Err("Not in Reggiani ZD(S)".to_string());
    }

    let info = annihilator_info(v, 16, 1e-12);
    if info.left_nullity == 0 || info.right_nullity == 0 {
        return Err(format!("Expected nontrivial annihilators, got {:?}", info));
    }

    // Verify nullspace basis vectors actually annihilate
    let la = left_multiplication_matrix(v, 16);
    let ra = right_multiplication_matrix(v, 16);
    let left_basis = nullspace_basis(&la, 1e-12);
    let right_basis = nullspace_basis(&ra, 1e-12);

    for col in 0..left_basis.ncols() {
        let b: Vec<f64> = left_basis.column(col).iter().copied().collect();
        let prod = cd_multiply(v, &b);
        if prod.iter().any(|&x| x.abs() > 1e-10) {
            return Err(format!(
                "Left annihilator basis vector {col} does not annihilate"
            ));
        }
    }

    for col in 0..right_basis.ncols() {
        let b: Vec<f64> = right_basis.column(col).iter().copied().collect();
        let prod = cd_multiply(&b, v);
        if prod.iter().any(|&x| x.abs() > 1e-10) {
            return Err(format!(
                "Right annihilator basis vector {col} does not annihilate"
            ));
        }
    }

    // Verify 4 standard partners span the left annihilator subspace
    let partners = standard_zero_divisor_partners(zd);
    let partner_cols: Vec<Vec<f64>> = partners.iter().map(|p| p.vector.clone()).collect();

    // Check: left_basis^T @ partner_matrix should have rank 4
    let k = left_basis.ncols();
    let n = 4;
    let mut coords = nalgebra::DMatrix::zeros(k, n);
    for (j, pvec) in partner_cols.iter().enumerate() {
        for i in 0..k {
            let mut dot = 0.0;
            for r in 0..16 {
                dot += left_basis[(r, i)] * pvec[r];
            }
            coords[(i, j)] = dot;
        }
    }
    let svd = nalgebra::SVD::new(coords, false, false);
    let rank = svd.singular_values.iter().filter(|&&s| s > 1e-8).count();
    if rank != 4 {
        return Err(format!(
            "Expected rank 4 for partner-annihilator projection, got {rank}"
        ));
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Partner graph adjacency matrix and orbit analysis
// ---------------------------------------------------------------------------

/// 84×84 adjacency matrix of the standard zero-divisor partner graph.
///
/// Entry `(i, j)` is `true` when ZD `j` annihilates ZD `i` (i.e. ZD `j`
/// is among the 4 standard partners of ZD `i`).
///
/// # Reggiani (2024) §3
///
/// Every standard zero-divisor has exactly 4 partners.  The resulting
/// graph is therefore 4-regular on 84 vertices.
pub fn partner_adjacency_matrix() -> Vec<Vec<bool>> {
    let zds = standard_zero_divisors();
    let n = zds.len(); // 84
    let mut adj = vec![vec![false; n]; n];

    for (i, zd) in zds.iter().enumerate() {
        let partners = standard_zero_divisor_partners(zd);
        for p in &partners {
            if let Some(j) = zds.iter().position(|z| z.key() == p.key()) {
                adj[i][j] = true;
            }
        }
    }
    adj
}

/// Gram matrix of inner products between all 84 standard ZDs.
///
/// Entry `(i, j)` = ⟨v_i , v_j⟩ (standard R^16 dot product).
/// Diagonal entries equal 2 (squared norm of each standard ZD).
///
/// The Gram matrix encodes the geometry of the zero-divisor set
/// inside R^16 and is a key invariant discussed in Reggiani (2024).
pub fn gram_matrix() -> Vec<Vec<f64>> {
    let zds = standard_zero_divisors();
    let n = zds.len();
    let mut g = vec![vec![0.0_f64; n]; n];
    for i in 0..n {
        for j in i..n {
            let dot: f64 = zds[i]
                .vector
                .iter()
                .zip(zds[j].vector.iter())
                .map(|(a, b)| a * b)
                .sum();
            g[i][j] = dot;
            g[j][i] = dot;
        }
    }
    g
}

/// Count the number of undirected connected components (orbits) in the
/// partner graph.
///
/// Each orbit is a maximal set of standard ZDs reachable from one
/// another via the partner relation (treated as undirected: an edge
/// exists between i and j when `adj[i][j] || adj[j][i]`).  For
/// sedenions (dim 16) the partner graph has a characteristic orbit
/// structure that reflects the box-kite topology of de Marrais (2000).
pub fn partner_graph_orbits() -> Vec<Vec<usize>> {
    let adj = partner_adjacency_matrix();
    let n = adj.len(); // 84
    let mut visited = vec![false; n];
    let mut orbits = Vec::new();

    for start in 0..n {
        if visited[start] {
            continue;
        }
        // BFS using undirected reachability (follow edges in both directions)
        let mut queue = std::collections::VecDeque::new();
        queue.push_back(start);
        visited[start] = true;
        let mut component = Vec::new();
        while let Some(v) = queue.pop_front() {
            component.push(v);
            for u in 0..n {
                if !visited[u] && (adj[v][u] || adj[u][v]) {
                    visited[u] = true;
                    queue.push_back(u);
                }
            }
        }
        component.sort();
        orbits.push(component);
    }
    orbits
}

/// Summary statistics for the partner graph.
#[derive(Debug, Clone)]
pub struct PartnerGraphStats {
    /// Number of vertices (84 for sedenions).
    pub n_vertices: usize,
    /// Number of directed edges (each ZD has 4 partners).
    pub n_directed_edges: usize,
    /// Whether the partner relation is symmetric (undirected).
    pub is_symmetric: bool,
    /// Number of connected components (orbits).
    pub n_orbits: usize,
    /// Sizes of each orbit.
    pub orbit_sizes: Vec<usize>,
}

/// Compute summary statistics for the 84-vertex partner graph.
pub fn partner_graph_stats() -> PartnerGraphStats {
    let adj = partner_adjacency_matrix();
    let n = adj.len();

    let mut n_edges = 0;
    let mut symmetric = true;
    for (i, row) in adj.iter().enumerate().take(n) {
        for (j, &val) in row.iter().enumerate().take(n) {
            if val {
                n_edges += 1;
            }
            if val != adj[j][i] {
                symmetric = false;
            }
        }
    }

    let orbits = partner_graph_orbits();
    let orbit_sizes: Vec<usize> = orbits.iter().map(|o| o.len()).collect();

    PartnerGraphStats {
        n_vertices: n,
        n_directed_edges: n_edges,
        is_symmetric: symmetric,
        n_orbits: orbits.len(),
        orbit_sizes,
    }
}

/// Eigenvalue spectrum of the 84x84 partner adjacency matrix.
///
/// Interprets the partner graph as a "Hamiltonian" and diagonalises it.
/// The resulting spectrum is the "band structure" of the zero-divisor
/// crystal -- analogous to tight-binding band energies in condensed matter.
///
/// # Cross-domain connection (Insight I-126)
///
/// If the spectrum contains highly degenerate eigenvalues ("flat bands"),
/// this proves that zero-divisor frustration on box-kite triangles produces
/// the same algebraic localization mechanism as kagome frustrated hopping.
pub fn partner_graph_spectrum() -> Vec<f64> {
    let adj = partner_adjacency_matrix();
    let n = adj.len();
    let mut mat = nalgebra::DMatrix::zeros(n, n);
    for i in 0..n {
        for j in 0..n {
            if adj[i][j] {
                mat[(i, j)] = 1.0;
            }
        }
    }
    let eig = mat.symmetric_eigenvalues();
    let mut spectrum: Vec<f64> = eig.iter().copied().collect();
    spectrum.sort_by(|a, b| a.partial_cmp(b).unwrap());
    spectrum
}

/// Spectrum analysis: degeneracy count at each distinct eigenvalue.
///
/// Returns sorted (eigenvalue, degeneracy) pairs. Eigenvalues within
/// `tol` of each other are considered degenerate.
pub fn partner_graph_degeneracies(tol: f64) -> Vec<(f64, usize)> {
    let spectrum = partner_graph_spectrum();
    let mut groups: Vec<(f64, usize)> = Vec::new();
    for &ev in &spectrum {
        if let Some(last) = groups.last_mut().filter(|(v, _)| (ev - v).abs() < tol) {
            last.1 += 1;
            continue;
        }
        groups.push((ev, 1));
    }
    groups
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standard_zero_divisors_count_is_84() {
        let zds = standard_zero_divisors();
        assert_eq!(zds.len(), 84);

        for zd in &zds {
            let norm_sq: f64 = zd.vector.iter().map(|x| x * x).sum();
            assert!(
                (norm_sq - 2.0).abs() < 1e-12,
                "ZD ({},{},{}) has norm_sq={norm_sq}",
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
            );
            assert!(
                is_reggiani_zd(&zd.vector, 1e-12),
                "ZD ({},{},{}) not in Reggiani ZD(S)",
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
            );
        }
    }

    #[test]
    fn test_all_standard_zds_have_nullity_4_4() {
        for zd in &standard_zero_divisors() {
            let info = annihilator_info(&zd.vector, 16, 1e-12);
            assert_eq!(
                (info.left_nullity, info.right_nullity),
                (4, 4),
                "ZD ({},{},{}) has nullity ({},{})",
                zd.assessor_low,
                zd.assessor_high,
                zd.diagonal_sign,
                info.left_nullity,
                info.right_nullity,
            );
        }
    }

    #[test]
    fn test_standard_zd_annihilator_consistency() {
        // Full Reggiani consistency check on all 84 ZDs
        for zd in &standard_zero_divisors() {
            assert_standard_zero_divisor_annihilators(zd).unwrap_or_else(|e| {
                panic!(
                    "ZD ({},{},{}) failed: {e}",
                    zd.assessor_low, zd.assessor_high, zd.diagonal_sign
                );
            });
        }
    }

    #[test]
    fn test_partner_adjacency_is_4_regular() {
        let adj = partner_adjacency_matrix();
        assert_eq!(adj.len(), 84);
        for (i, row) in adj.iter().enumerate() {
            let degree: usize = row.iter().filter(|&&x| x).count();
            assert_eq!(
                degree, 4,
                "ZD {i} has partner degree {degree}, expected 4"
            );
        }
    }

    #[test]
    fn test_gram_matrix_diagonal_is_2() {
        let g = gram_matrix();
        assert_eq!(g.len(), 84);
        for i in 0..84 {
            assert!(
                (g[i][i] - 2.0).abs() < 1e-12,
                "Gram diagonal [{i}][{i}] = {}, expected 2.0",
                g[i][i]
            );
        }
    }

    #[test]
    fn test_gram_matrix_is_symmetric() {
        let g = gram_matrix();
        for i in 0..84 {
            for j in (i + 1)..84 {
                assert!(
                    (g[i][j] - g[j][i]).abs() < 1e-14,
                    "Gram[{i}][{j}] = {} != Gram[{j}][{i}] = {}",
                    g[i][j],
                    g[j][i]
                );
            }
        }
    }

    #[test]
    fn test_partner_graph_orbit_structure() {
        let stats = partner_graph_stats();
        assert_eq!(stats.n_vertices, 84);
        // Each ZD has exactly 4 partners (directed)
        assert_eq!(stats.n_directed_edges, 84 * 4);
        // Orbit sizes must sum to 84
        let total: usize = stats.orbit_sizes.iter().sum();
        assert_eq!(total, 84, "orbit sizes must sum to 84");
        // At least 1 orbit
        assert!(stats.n_orbits >= 1);
    }

    #[test]
    fn test_partner_graph_spectrum_has_84_eigenvalues() {
        let spectrum = partner_graph_spectrum();
        assert_eq!(spectrum.len(), 84);
        // 4-regular graph: largest eigenvalue = 4 (all-ones eigenvector for
        // each connected component, scaled by vertex degree)
        let max_ev = spectrum.last().unwrap();
        assert!(
            (*max_ev - 4.0).abs() < 0.01,
            "Max eigenvalue {max_ev}, expected 4.0 for 4-regular graph"
        );
    }

    #[test]
    fn test_partner_graph_has_degenerate_eigenvalues() {
        let degs = partner_graph_degeneracies(0.01);
        // Print spectrum for analysis
        for (ev, deg) in &degs {
            if *deg > 1 {
                eprintln!("  eigenvalue {ev:.4}: degeneracy {deg}");
            }
        }
        // A 4-regular graph on 84 vertices with nontrivial symmetry must have
        // at least some degenerate eigenvalues (orbit structure forces this).
        let max_deg = degs.iter().map(|(_, d)| *d).max().unwrap();
        assert!(
            max_deg > 1,
            "Expected degenerate eigenvalues in partner graph spectrum"
        );
    }
}
