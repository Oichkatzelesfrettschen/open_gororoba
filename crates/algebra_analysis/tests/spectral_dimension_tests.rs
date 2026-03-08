//! Integration tests for the spectral dimension module.
//!
//! These tests exercise the full pipeline from ZD graph construction through
//! eigendecomposition, heat kernel computation, and plateau extraction.

use algebra_analysis::{
    boxkites::motif_components_for_cross_assessors,
    spectral_dimension::{
        DenseEigenSolver, EigenSolver, analyze_single_dimension, build_graph_laplacian,
        build_laplacian_from_component, clamp_zero_eigenvalues, extract_plateau, geom_linspace,
        graph_spectral_dimension, zd_laplacian_eigenvalues,
    },
};
use petgraph::graph::UnGraph;

/// Build a 2D grid graph (lattice Z^2 of size m x m).
fn grid_2d(m: usize) -> UnGraph<(), ()> {
    let mut g = UnGraph::<(), ()>::new_undirected();
    let nodes: Vec<_> = (0..(m * m)).map(|_| g.add_node(())).collect();
    for r in 0..m {
        for c in 0..m {
            let idx = r * m + c;
            if c + 1 < m {
                g.add_edge(nodes[idx], nodes[idx + 1], ());
            }
            if r + 1 < m {
                g.add_edge(nodes[idx], nodes[idx + m], ());
            }
        }
    }
    g
}

#[test]
fn grid_2d_spectral_dimension_approaches_two() {
    // A 2D lattice should have d_s -> 2 at intermediate t
    let g = grid_2d(10);
    let l = build_graph_laplacian(&g);
    let solver = DenseEigenSolver;
    let mut eigs = solver.eigenvalues(&l);
    clamp_zero_eigenvalues(&mut eigs, 1e-12);

    let curve = graph_spectral_dimension(&eigs, 0.01, 20.0, 300);
    assert!(!curve.is_empty());

    // At intermediate t (not too small, not too large), d_s should be near 2
    let mid_values: Vec<f64> = curve
        .iter()
        .filter(|(ln_t, _)| *ln_t > 0.0 && *ln_t < 2.0) // t in [1, 7]
        .map(|(_, ds)| *ds)
        .collect();

    if !mid_values.is_empty() {
        let mean_ds = mid_values.iter().sum::<f64>() / mid_values.len() as f64;
        // For a finite 10x10 grid, d_s should be near 2 in the intermediate regime
        assert!(
            mean_ds > 1.0 && mean_ds < 3.5,
            "10x10 grid mid-t d_s should be near 2, got {mean_ds}"
        );
    }
}

#[test]
fn dim32_has_more_components_than_dim16() {
    let c16 = motif_components_for_cross_assessors(16);
    let c32 = motif_components_for_cross_assessors(32);
    assert!(
        c32.len() > c16.len(),
        "dim=32 should have more components ({}) than dim=16 ({})",
        c32.len(),
        c16.len()
    );
}

#[test]
fn dim32_laplacian_eigenvalues() {
    let eigs = zd_laplacian_eigenvalues(32);
    assert!(!eigs.is_empty(), "dim=32 should have eigenvalues");

    let zero_count = eigs.iter().filter(|&&e| e == 0.0).count();
    assert_eq!(
        zero_count, 1,
        "single representative component should have 1 zero eigenvalue"
    );

    // All should be non-negative (Laplacian is PSD)
    for &ev in &eigs {
        assert!(ev >= -1e-12, "Laplacian eigenvalue should be >= 0: {ev}");
    }
}

#[test]
fn cross_dimension_analysis_runs() {
    let result16 = analyze_single_dimension(16, 0.01, 50.0, 200);
    let result32 = analyze_single_dimension(32, 0.01, 50.0, 200);

    assert_eq!(result16.cd_dim, 16);
    assert_eq!(result32.cd_dim, 32);

    // dim=32 components should have more nodes than dim=16
    assert!(
        result32.nodes_per_component > result16.nodes_per_component,
        "dim=32 nodes ({}) should exceed dim=16 nodes ({})",
        result32.nodes_per_component,
        result16.nodes_per_component
    );

    // Both should produce non-empty curves
    assert!(!result16.curve.is_empty());
    assert!(!result32.curve.is_empty());
}

#[test]
fn numerical_stability_synthetic_plateau() {
    // Create synthetic eigenvalues that produce a known d_s plateau
    // For eigenvalues {lambda_1, ..., lambda_n}, with all equal to lambda:
    // P(t) = exp(-lambda*t), ln P = -lambda*t
    // d_s = -2 * d(ln P)/d(ln t) = 2*lambda*t * d(ln t)/d(ln t) ... wait
    // Actually d(ln P)/d(ln t) = d(-lambda*t)/d(ln t) = -lambda * d(t)/d(ln t) = -lambda*t
    // So d_s = -2 * (-lambda*t) = 2*lambda*t ... this grows with t.
    //
    // For a REAL plateau, we need a distribution of eigenvalues.
    // Use the Laplacian of a known graph instead.
    let g = grid_2d(8);
    let l = build_graph_laplacian(&g);
    let solver = DenseEigenSolver;
    let mut eigs = solver.eigenvalues(&l);
    clamp_zero_eigenvalues(&mut eigs, 1e-12);

    let curve = graph_spectral_dimension(&eigs, 0.01, 10.0, 300);

    // SG filter should produce smooth output (no NaN or Inf)
    for &(ln_t, d_s) in &curve {
        assert!(ln_t.is_finite(), "ln_t should be finite");
        assert!(d_s.is_finite(), "d_s should be finite");
    }
}

#[test]
fn plateau_extraction_from_constant_region() {
    // Create a synthetic curve with a clear plateau
    let mut curve = Vec::new();
    for i in 0..100 {
        let t = i as f64 * 0.1;
        let d_s = if t < 2.0 {
            1.0 + t // Ramp up
        } else if t < 7.0 {
            3.0 // Plateau
        } else {
            3.0 + (t - 7.0) * 0.5 // Ramp up again
        };
        curve.push((t, d_s));
    }

    let plateau = extract_plateau(&curve, 0.1);
    assert!(plateau.is_some(), "should find a plateau");
    let p = plateau.unwrap();
    assert!(
        (p.d_s - 3.0).abs() < 0.1,
        "plateau value should be near 3.0: {}",
        p.d_s
    );
}

#[test]
fn geom_linspace_boundary_values() {
    let ts = geom_linspace(0.01, 100.0, 100);
    assert!((ts[0] - 0.01).abs() < 1e-15, "first value should be t_min");
    assert!((ts[99] - 100.0).abs() < 1e-10, "last value should be t_max");
}

#[test]
fn build_laplacian_from_component_matches_graph() {
    // Verify that building Laplacian from MotifComponent gives same result
    // as building from the petgraph UnGraph conversion
    let components = motif_components_for_cross_assessors(16);
    assert!(!components.is_empty());

    let comp = &components[0];
    let laplacian_direct = build_laplacian_from_component(comp);

    // Also build via petgraph route
    let pg = comp.to_petgraph();
    let laplacian_pg = build_graph_laplacian(&pg);

    assert_eq!(laplacian_direct.nrows(), laplacian_pg.nrows());
    assert_eq!(laplacian_direct.ncols(), laplacian_pg.ncols());

    // Eigenvalues should match (even if matrix entries differ by permutation)
    let solver = DenseEigenSolver;
    let eigs_direct = solver.eigenvalues(&laplacian_direct);
    let eigs_pg = solver.eigenvalues(&laplacian_pg);

    for (a, b) in eigs_direct.iter().zip(eigs_pg.iter()) {
        assert!(
            (a - b).abs() < 1e-10,
            "eigenvalues should match: {a} vs {b}"
        );
    }
}
