//! Absorber Topology Benchmark for C-010.
//!
//! Scores local vs non-local coupling topologies against mode-suppression
//! targets using the multi-resonator TCMT framework.
//!
//! The sedenion ZD incidence graph has 7 disconnected K6 cliques, enforcing
//! perfect inter-clique isolation. Local topologies (ring, lattice) introduce
//! leakage paths. This module quantifies that gap.
//!
//! # Physics
//!
//! Each topology is an N-node coupling graph where nodes are Kerr cavities
//! and edges carry coupling strengths. We drive a single "target clique"
//! and measure energy leakage into non-target nodes after transient decay.
//!
//! Metrics:
//! - **Isolation ratio**: E_target / E_total (1.0 = perfect isolation)
//! - **Crosstalk dB**: 10*log10(E_leak / E_target) (more negative = better)
//! - **Spectral gap**: eigenvalue gap between intra-clique and inter-clique modes
//!
//! # References
//! - C-010: Metamaterial ZD topology (CLOSED_NEGATIVE_RESULT)
//! - de Marrais (2000): Box-kite structure
//! - Suh et al., IEEE JQE 40, 1511 (2004): Multi-port TCMT

use crate::tcmt::{InputField, KerrCavity};
use num_complex::Complex64;

/// Coupling topology for an absorber network.
#[derive(Debug, Clone)]
pub struct CouplingTopology {
    /// Descriptive name.
    pub name: String,
    /// Number of nodes (cavities).
    pub n_nodes: usize,
    /// Symmetric adjacency matrix: coupling[i][j] = coupling strength.
    pub adjacency: Vec<Vec<f64>>,
}

/// Result of a single topology benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    /// Topology name.
    pub topology_name: String,
    /// Number of nodes.
    pub n_nodes: usize,
    /// Isolation ratio: E_target / E_total (1.0 = perfect).
    pub isolation_ratio: f64,
    /// Crosstalk in dB: 10*log10(E_leak / E_target).
    /// More negative = better isolation. -inf for perfect.
    pub crosstalk_db: f64,
    /// Spectral gap of coupling matrix (0 = fully connected spectrum).
    pub spectral_gap: f64,
    /// Number of connected components in the topology.
    pub n_components: usize,
    /// Per-node final energies.
    pub node_energies: Vec<f64>,
}

/// Comparative benchmark across multiple topologies.
#[derive(Debug, Clone)]
pub struct ComparativeBenchmark {
    /// Results for each topology tested.
    pub results: Vec<BenchmarkResult>,
    /// Indices of target nodes that were driven.
    pub driven_nodes: Vec<usize>,
    /// Number of simulation steps used.
    pub n_steps: usize,
}

// ---------------------------------------------------------------------------
// Topology generators
// ---------------------------------------------------------------------------

impl CouplingTopology {
    /// Create the ZD clique topology: `n_cliques` disconnected complete
    /// subgraphs of size `clique_size`.
    ///
    /// For sedenions: `disconnected_cliques(7, 6)` -> 42 nodes, 7 K6 cliques.
    pub fn disconnected_cliques(n_cliques: usize, clique_size: usize, kappa: f64) -> Self {
        let n = n_cliques * clique_size;
        let adjacency = (0..n)
            .map(|node| {
                let clique_base = (node / clique_size) * clique_size;
                (0..n)
                    .map(|other| {
                        if other >= clique_base
                            && other < clique_base + clique_size
                            && other != node
                        {
                            kappa
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        Self {
            name: format!("{} x K{}", n_cliques, clique_size),
            n_nodes: n,
            adjacency,
        }
    }

    /// Ring topology: each node coupled to its two nearest neighbors.
    pub fn ring(n: usize, kappa: f64) -> Self {
        let adjacency = (0..n)
            .map(|i| {
                let prev = if i == 0 { n - 1 } else { i - 1 };
                let next = (i + 1) % n;
                let mut row = vec![0.0; n];
                row[prev] = kappa;
                row[next] = kappa;
                row
            })
            .collect();

        Self {
            name: format!("Ring({})", n),
            n_nodes: n,
            adjacency,
        }
    }

    /// Complete graph: all nodes coupled to all others.
    pub fn complete(n: usize, kappa: f64) -> Self {
        let adjacency = (0..n)
            .map(|i| (0..n).map(|j| if i == j { 0.0 } else { kappa }).collect())
            .collect();

        Self {
            name: format!("K{}", n),
            n_nodes: n,
            adjacency,
        }
    }

    /// Nearest-neighbor lattice: node i coupled to i-1 and i+1 (open chain).
    pub fn chain(n: usize, kappa: f64) -> Self {
        let adjacency = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                if i > 0 {
                    row[i - 1] = kappa;
                }
                if i + 1 < n {
                    row[i + 1] = kappa;
                }
                row
            })
            .collect();

        Self {
            name: format!("Chain({})", n),
            n_nodes: n,
            adjacency,
        }
    }

    /// Count connected components via BFS on nonzero edges.
    pub fn connected_components(&self) -> Vec<Vec<usize>> {
        let n = self.n_nodes;
        let mut visited = vec![false; n];
        let mut components = Vec::new();

        for start in 0..n {
            if visited[start] {
                continue;
            }
            let mut comp = Vec::new();
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                if visited[node] {
                    continue;
                }
                visited[node] = true;
                comp.push(node);
                for (neighbor, &weight) in self.adjacency[node].iter().enumerate() {
                    if weight.abs() > 1e-15 && !visited[neighbor] {
                        stack.push(neighbor);
                    }
                }
            }
            comp.sort_unstable();
            components.push(comp);
        }
        components
    }

    /// Compute eigenvalues of the coupling matrix (real, symmetric).
    ///
    /// Uses the Jacobi eigenvalue algorithm for small matrices.
    /// Returns sorted eigenvalues in ascending order.
    pub fn eigenvalues(&self) -> Vec<f64> {
        jacobi_eigenvalues(&self.adjacency)
    }

    /// Spectral gap: smallest nonzero eigenvalue magnitude.
    /// For disconnected graphs, this captures the intra-component gap.
    pub fn spectral_gap(&self) -> f64 {
        let eigs = self.eigenvalues();
        let mut nonzero: Vec<f64> = eigs.iter().map(|e| e.abs()).filter(|e| *e > 1e-12).collect();
        nonzero.sort_by(|a, b| a.partial_cmp(b).unwrap());
        nonzero.first().copied().unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Benchmark engine
// ---------------------------------------------------------------------------

/// Run the absorber topology benchmark.
///
/// Drives `driven_nodes` and measures energy leakage into non-driven nodes
/// after `n_steps` of RK4 integration.
///
/// # Arguments
/// * `topologies` - Set of topologies to compare.
/// * `base_cavity` - Template Kerr cavity for all nodes.
/// * `frequency_spacing` - Frequency offset between nodes (rad/s).
/// * `driven_nodes` - Indices of nodes to drive with input field.
/// * `drive_amplitude` - Input field amplitude.
/// * `dt` - Integration timestep (s).
/// * `n_steps` - Number of RK4 steps.
#[allow(clippy::too_many_arguments)]
pub fn run_benchmark(
    topologies: &[CouplingTopology],
    base_cavity: &KerrCavity,
    frequency_spacing: f64,
    driven_nodes: &[usize],
    drive_amplitude: f64,
    dt: f64,
    n_steps: usize,
) -> ComparativeBenchmark {
    let results: Vec<BenchmarkResult> = topologies
        .iter()
        .map(|topo| benchmark_single(topo, base_cavity, frequency_spacing, driven_nodes, drive_amplitude, dt, n_steps))
        .collect();

    ComparativeBenchmark {
        results,
        driven_nodes: driven_nodes.to_vec(),
        n_steps,
    }
}

/// Benchmark a single topology.
fn benchmark_single(
    topo: &CouplingTopology,
    base_cavity: &KerrCavity,
    frequency_spacing: f64,
    driven_nodes: &[usize],
    drive_amplitude: f64,
    dt: f64,
    n_steps: usize,
) -> BenchmarkResult {
    let n = topo.n_nodes;

    // Build per-node cavities with frequency offsets
    let cavities: Vec<KerrCavity> = (0..n)
        .map(|i| {
            KerrCavity::new(
                base_cavity.omega_0 + (i as f64) * frequency_spacing,
                base_cavity.q_intrinsic,
                base_cavity.q_external,
                base_cavity.n_linear,
                base_cavity.n2,
                base_cavity.v_eff,
            )
        })
        .collect();

    // Build input fields: nonzero only for driven nodes
    let inputs: Vec<InputField> = (0..n)
        .map(|i| {
            let amp = if driven_nodes.contains(&i) {
                drive_amplitude
            } else {
                0.0
            };
            InputField {
                amplitude: Complex64::new(amp, 0.0),
                omega: cavities[i].omega_0,
            }
        })
        .collect();

    // Initial state: all at rest
    let mut amplitudes: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); n];

    // RK4 integration with coupling
    for _step in 0..n_steps {
        let new_amplitudes = rk4_coupled_step(&cavities, &topo.adjacency, &amplitudes, &inputs, dt);
        amplitudes = new_amplitudes;
    }

    // Compute per-node energies
    let node_energies: Vec<f64> = amplitudes.iter().map(|a| a.norm_sqr()).collect();
    let e_total: f64 = node_energies.iter().sum();
    let e_target: f64 = driven_nodes
        .iter()
        .map(|&i| node_energies.get(i).copied().unwrap_or(0.0))
        .sum();
    let e_leak = e_total - e_target;

    let isolation_ratio = if e_total > 0.0 {
        e_target / e_total
    } else {
        1.0
    };

    let crosstalk_db = if e_target > 1e-30 && e_leak > 1e-30 {
        10.0 * (e_leak / e_target).log10()
    } else if e_leak <= 1e-30 {
        f64::NEG_INFINITY
    } else {
        0.0
    };

    let components = topo.connected_components();
    let spectral_gap = topo.spectral_gap();

    BenchmarkResult {
        topology_name: topo.name.clone(),
        n_nodes: n,
        isolation_ratio,
        crosstalk_db,
        spectral_gap,
        n_components: components.len(),
        node_energies,
    }
}

/// Single RK4 step for coupled cavity network.
///
/// da_i/dt = f_i(a_i, s_in_i) + sum_j kappa_ij * a_j
///
/// where f_i is the single-cavity TCMT derivative.
fn rk4_coupled_step(
    cavities: &[KerrCavity],
    coupling: &[Vec<f64>],
    amplitudes: &[Complex64],
    inputs: &[InputField],
    dt: f64,
) -> Vec<Complex64> {
    let n = cavities.len();

    let deriv = |amps: &[Complex64], _t: f64| -> Vec<Complex64> {
        (0..n)
            .map(|i| {
                // TCMT derivative (Liu et al. 2013):
                //   da/dt = (-gamma_tot/2 + j*delta - j*gamma_kerr*|a|^2) * a
                //         + sqrt(2*gamma_e) * s_in
                let a = amps[i];
                let delta = inputs[i].omega - cavities[i].omega_0;
                let gamma_tot = cavities[i].gamma_total();
                let a_norm_sq = a.norm_sqr();
                let kerr_shift = -cavities[i].gamma_kerr() * a_norm_sq;

                let linear = Complex64::new(-gamma_tot / 2.0, delta + kerr_shift);
                let coupling_in = cavities[i].coupling_coefficient() * inputs[i].amplitude;

                let base = linear * a + coupling_in;

                // Inter-cavity coupling: j * kappa_ij * a_j
                let coupling_term: Complex64 = (0..n)
                    .map(|j| {
                        if i != j {
                            Complex64::new(0.0, coupling[i][j]) * amps[j]
                        } else {
                            Complex64::new(0.0, 0.0)
                        }
                    })
                    .sum();

                base + coupling_term
            })
            .collect()
    };

    let k1 = deriv(amplitudes, 0.0);

    let s2: Vec<Complex64> = (0..n)
        .map(|i| amplitudes[i] + 0.5 * dt * k1[i])
        .collect();
    let k2 = deriv(&s2, 0.5 * dt);

    let s3: Vec<Complex64> = (0..n)
        .map(|i| amplitudes[i] + 0.5 * dt * k2[i])
        .collect();
    let k3 = deriv(&s3, 0.5 * dt);

    let s4: Vec<Complex64> = (0..n)
        .map(|i| amplitudes[i] + dt * k3[i])
        .collect();
    let k4 = deriv(&s4, dt);

    (0..n)
        .map(|i| amplitudes[i] + (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
        .collect()
}

// ---------------------------------------------------------------------------
// Jacobi eigenvalue algorithm (for small symmetric matrices)
// ---------------------------------------------------------------------------

/// Jacobi eigenvalue algorithm for real symmetric matrices.
/// Returns eigenvalues sorted in ascending order.
#[allow(clippy::needless_range_loop)] // Matrix algorithm: index-based access is clearer
fn jacobi_eigenvalues(matrix: &[Vec<f64>]) -> Vec<f64> {
    let n = matrix.len();
    if n == 0 {
        return vec![];
    }

    // Flatten to working copy
    let mut a: Vec<Vec<f64>> = matrix.to_vec();

    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i][j].abs() > max_val {
                    max_val = a[i][j].abs();
                    p = i;
                    q = j;
                }
            }
        }

        // Converged?
        if max_val < 1e-14 {
            break;
        }

        // Compute rotation angle
        let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
        };

        let (sin_t, cos_t) = theta.sin_cos();

        // Apply Jacobi rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                new_a[p][i] = new_a[i][p];
                new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                new_a[q][i] = new_a[i][q];
            }
        }
        new_a[p][p] = cos_t * cos_t * a[p][p]
            + 2.0 * sin_t * cos_t * a[p][q]
            + sin_t * sin_t * a[q][q];
        new_a[q][q] = sin_t * sin_t * a[p][p]
            - 2.0 * sin_t * cos_t * a[p][q]
            + cos_t * cos_t * a[q][q];
        new_a[p][q] = 0.0;
        new_a[q][p] = 0.0;

        a = new_a;
    }

    let mut eigs: Vec<f64> = (0..n).map(|i| a[i][i]).collect();
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    eigs
}

// ---------------------------------------------------------------------------
// Standard topology suite for C-010
// ---------------------------------------------------------------------------

/// Build the standard C-010 topology comparison suite.
///
/// Returns topologies matching the sedenion configuration:
/// - ZD cliques: 7 x K6 (42 nodes, disconnected)
/// - Ring: 42 nodes in a ring
/// - Chain: 42 nodes in an open chain
/// - Complete: 42 nodes, all-to-all
///
/// All use the same coupling strength for fair comparison.
pub fn standard_c010_suite(kappa: f64) -> Vec<CouplingTopology> {
    vec![
        CouplingTopology::disconnected_cliques(7, 6, kappa),
        CouplingTopology::ring(42, kappa),
        CouplingTopology::chain(42, kappa),
        CouplingTopology::complete(42, kappa),
    ]
}

/// Build a reduced topology suite for fast tests.
///
/// Uses 2 x K3 (6 nodes) instead of 7 x K6 (42 nodes).
pub fn reduced_c010_suite(kappa: f64) -> Vec<CouplingTopology> {
    vec![
        CouplingTopology::disconnected_cliques(2, 3, kappa),
        CouplingTopology::ring(6, kappa),
        CouplingTopology::chain(6, kappa),
        CouplingTopology::complete(6, kappa),
    ]
}

impl ComparativeBenchmark {
    /// Print a human-readable summary table.
    pub fn summary_table(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!(
            "{:<20} {:>6} {:>8} {:>12} {:>10} {:>6}",
            "Topology", "Nodes", "Comps", "Isolation", "Xtalk(dB)", "Gap"
        ));
        lines.push("-".repeat(66));
        for r in &self.results {
            let xtalk_str = if r.crosstalk_db.is_infinite() {
                "-inf".to_string()
            } else {
                format!("{:.1}", r.crosstalk_db)
            };
            lines.push(format!(
                "{:<20} {:>6} {:>8} {:>12.6} {:>10} {:>6.4}",
                r.topology_name, r.n_nodes, r.n_components, r.isolation_ratio, xtalk_str, r.spectral_gap,
            ));
        }
        lines.join("\n")
    }

    /// Check if the ZD clique topology (first entry) achieves strictly
    /// better isolation than all local topologies.
    pub fn zd_dominates(&self) -> bool {
        if self.results.is_empty() {
            return false;
        }
        let zd_isolation = self.results[0].isolation_ratio;
        self.results[1..].iter().all(|r| r.isolation_ratio < zd_isolation)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn test_cavity() -> KerrCavity {
        KerrCavity::new(
            1.0e15,  // omega_0 ~ 1 PHz
            1.0e6,   // Q_intrinsic
            5.0e5,   // Q_external
            1.5,     // n_linear
            2.0e-18, // n2
            1.0e-18, // v_eff
        )
    }

    // -- Topology structure tests --

    #[test]
    fn test_disconnected_cliques_structure() {
        let topo = CouplingTopology::disconnected_cliques(3, 4, 1.0);
        assert_eq!(topo.n_nodes, 12);
        let comps = topo.connected_components();
        assert_eq!(comps.len(), 3, "3 cliques -> 3 components");
        for comp in &comps {
            assert_eq!(comp.len(), 4, "each clique has 4 nodes");
        }
    }

    #[test]
    fn test_ring_single_component() {
        let topo = CouplingTopology::ring(10, 1.0);
        let comps = topo.connected_components();
        assert_eq!(comps.len(), 1, "ring is connected");
        assert_eq!(comps[0].len(), 10);
    }

    #[test]
    fn test_chain_single_component() {
        let topo = CouplingTopology::chain(8, 0.5);
        let comps = topo.connected_components();
        assert_eq!(comps.len(), 1, "chain is connected");
    }

    #[test]
    fn test_complete_single_component() {
        let topo = CouplingTopology::complete(7, 0.3);
        let comps = topo.connected_components();
        assert_eq!(comps.len(), 1, "complete graph is connected");
        // K7 has 7*6/2 = 21 edges
        let n_edges: usize = (0..7)
            .map(|i| (i + 1..7).filter(|&j| topo.adjacency[i][j].abs() > 1e-15).count())
            .sum();
        assert_eq!(n_edges, 21);
    }

    #[test]
    fn test_clique_edge_count() {
        // 7 x K6: each K6 has 6*5/2 = 15 edges, total 105
        let topo = CouplingTopology::disconnected_cliques(7, 6, 1.0);
        let n_edges: usize = (0..42)
            .map(|i| ((i + 1)..42).filter(|&j| topo.adjacency[i][j].abs() > 1e-15).count())
            .sum();
        assert_eq!(n_edges, 7 * 15, "7 x K6 has 105 edges");
    }

    // -- Eigenvalue tests --

    #[test]
    fn test_eigenvalues_2x2() {
        // [[0, 1], [1, 0]] has eigenvalues -1, +1
        let topo = CouplingTopology {
            name: "test".to_string(),
            n_nodes: 2,
            adjacency: vec![vec![0.0, 1.0], vec![1.0, 0.0]],
        };
        let eigs = topo.eigenvalues();
        assert!((eigs[0] - (-1.0)).abs() < 1e-10);
        assert!((eigs[1] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_disconnected_cliques_zero_eigenvalues() {
        // 2 x K2: 4 nodes, 2 components
        // Adjacency has block structure => 2 zero eigenvalues from disconnection?
        // Actually K2 adj = [[0,k],[k,0]], eigs = -k, +k per block
        // So 2xK2: eigs = {-k, +k, -k, +k} => no zeros unless k=0
        let topo = CouplingTopology::disconnected_cliques(2, 2, 1.0);
        let eigs = topo.eigenvalues();
        assert_eq!(eigs.len(), 4);
        // Should have 2 eigenvalues at -1 and 2 at +1
        let neg_count = eigs.iter().filter(|&&e| (e - (-1.0)).abs() < 1e-10).count();
        let pos_count = eigs.iter().filter(|&&e| (e - 1.0).abs() < 1e-10).count();
        assert_eq!(neg_count, 2);
        assert_eq!(pos_count, 2);
    }

    // -- Benchmark integration tests --

    #[test]
    fn test_reduced_benchmark_zd_dominates() {
        let cavity = test_cavity();
        let suite = reduced_c010_suite(1.0e10);
        // Drive nodes 0,1,2 (first clique in the 2xK3 topology)
        let driven = vec![0, 1, 2];
        let result = run_benchmark(&suite, &cavity, 1.0e12, &driven, 1.0e-3, 1.0e-12, 2000);

        // ZD cliques should have perfect isolation (driven nodes are one full clique)
        let zd_result = &result.results[0];
        assert!(
            zd_result.isolation_ratio > 0.99,
            "ZD cliques: isolation = {}, expected > 0.99",
            zd_result.isolation_ratio
        );

        // Connected topologies should show crosstalk
        for r in &result.results[1..] {
            assert!(
                r.isolation_ratio < zd_result.isolation_ratio,
                "{}: isolation {} should be < ZD {}",
                r.topology_name,
                r.isolation_ratio,
                zd_result.isolation_ratio,
            );
        }

        assert!(result.zd_dominates(), "ZD topology should dominate");
    }

    #[test]
    fn test_no_drive_zero_energy() {
        let cavity = test_cavity();
        let topo = CouplingTopology::ring(6, 1.0e10);
        let result = benchmark_single(&topo, &cavity, 1.0e12, &[], 0.0, 1.0e-12, 100);
        let total_energy: f64 = result.node_energies.iter().sum();
        assert!(
            total_energy < 1e-30,
            "no drive => zero energy, got {}",
            total_energy
        );
    }

    #[test]
    fn test_summary_table_format() {
        let cavity = test_cavity();
        let suite = reduced_c010_suite(1.0e10);
        let result = run_benchmark(&suite, &cavity, 1.0e12, &[0, 1, 2], 1.0e-3, 1.0e-12, 500);
        let table = result.summary_table();
        assert!(table.contains("Topology"), "table should have header");
        assert!(table.contains("2 x K3"), "table should have ZD entry");
        assert!(table.contains("Ring"), "table should have Ring entry");
    }

    #[test]
    fn test_spectral_gap_disconnected_vs_connected() {
        let kappa = 1.0;
        let disc = CouplingTopology::disconnected_cliques(2, 3, kappa);
        let ring = CouplingTopology::ring(6, kappa);

        // Both should have nonzero spectral gaps (intra-component modes)
        assert!(disc.spectral_gap() > 0.0);
        assert!(ring.spectral_gap() > 0.0);
    }

    #[test]
    fn test_jacobi_identity_matrix() {
        // 3x3 identity should have all eigenvalues = 1
        let mat = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        let eigs = jacobi_eigenvalues(&mat);
        for e in &eigs {
            assert!((e - 1.0).abs() < 1e-12, "identity eigenvalue should be 1, got {}", e);
        }
    }

    #[test]
    fn test_jacobi_known_spectrum() {
        // [[2, 1], [1, 2]] has eigenvalues 1, 3
        let mat = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
        let eigs = jacobi_eigenvalues(&mat);
        assert!((eigs[0] - 1.0).abs() < 1e-12);
        assert!((eigs[1] - 3.0).abs() < 1e-12);
    }
}
